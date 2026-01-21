from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_HORIZON,
    DEFAULT_SEQ_LENGTH,
    DEFAULT_TEST_GWS,
    DEFAULT_VAL_GWS,
    TARGET_COLUMN,
)
from .data_loading import load_premier_league_gameweek_stats
from .modeling import build_lstm_model
from .preprocessing import PreprocessArtifacts, fit_preprocessor_on_timesteps, select_and_coerce_numeric, transform_sequences
from .sequences import SequenceDataset, build_sequences, split_by_end_gw
from .evaluation import evaluate_fpl_model
from .role_modeling import (
    build_feature_weight_vector,
    infer_role_from_window,
    list_roles,
    position_to_role,
    role_loss_weight,
)


def _xgi_cap_from_raw_sequences(
    X_raw: np.ndarray,
    roles: np.ndarray,
    *,
    feature_columns: list[str],
) -> np.ndarray:
    """Compute a simple xGI-based cap used for bias-penalty training.

    The goal is not to "tell the model what's right"; it's to punish the model
    when it predicts points far above what recent xGI plausibly supports.
    """
    idx = {c: i for i, c in enumerate(feature_columns)}

    minutes_i = idx.get("minutes")
    xgi90_i = idx.get("expected_goal_involvements_per_90")
    xg90_i = idx.get("expected_goals_per_90")
    xa90_i = idx.get("expected_assists_per_90")

    # Per-xGI expected points rate (rough heuristic).
    # Assume xGI split ~60% goals / 40% assists.
    goal_pts = {
        "FWD": 4.0,
        "MID": 5.0,
        "MID_AM": 5.0,
        "MID_DM": 5.0,
        "DEF": 6.0,
        "GK": 6.0,
    }

    caps: list[float] = []
    for i in range(X_raw.shape[0]):
        last = X_raw[i, -1, :]
        minutes = float(last[minutes_i]) if minutes_i is not None else 90.0
        minutes = float(np.nan_to_num(minutes, nan=0.0))
        p60 = float(np.clip(minutes / 60.0, 0.0, 1.0))
        m90 = float(np.clip(minutes / 90.0, 0.0, 1.0))

        if xgi90_i is not None:
            xgi90 = float(last[xgi90_i])
        else:
            xg90 = float(last[xg90_i]) if xg90_i is not None else 0.0
            xa90 = float(last[xa90_i]) if xa90_i is not None else 0.0
            xgi90 = xg90 + xa90
        xgi90 = float(np.nan_to_num(xgi90, nan=0.0))
        xgi = max(xgi90, 0.0) * m90

        role = str(roles[i] if i < len(roles) else "").strip()
        gpts = float(goal_pts.get(role, 5.0))
        rate = 0.6 * gpts + 0.4 * 3.0

        # Base appearance points + xGI-driven expectation.
        cap = 2.0 * p60 + rate * xgi

        # Give DEF/GK a small allowance for CS points without overtrusting them.
        if role in {"DEF", "GK"}:
            cap += 0.8 * p60

        caps.append(float(max(cap, 0.0)))

    return np.asarray(caps, dtype=float)


def _set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


def _get_available_features(df: pd.DataFrame, requested_features: list[str]) -> list[str]:
    """Return only the features that actually exist in the DataFrame.
    
    This allows the model to gracefully handle cases where some engineered
    features might not be available in the data.
    """
    available = [f for f in requested_features if f in df.columns]
    missing = set(requested_features) - set(available)
    if missing:
        print(f"Warning: {len(missing)} requested features not found in data: {missing}")
    return available


def _permutation_importance_mae(
    *,
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    feature_columns: list[str],
    baseline_mae: float,
    max_features: int = 25,
    seed: int = 42,
) -> pd.DataFrame:
    """Permutation importance by shuffling each feature across samples (per timestep)."""
    rng = np.random.default_rng(seed)
    n_features = X.shape[-1]
    n_eval = min(max_features, n_features)

    # Evaluate only features with non-trivial variance.
    variances = np.var(X.reshape(-1, n_features), axis=0)
    order = np.argsort(-variances)
    order = order[:n_eval]

    rows: list[dict] = []
    for j in order:
        Xp = X.copy()
        for t in range(Xp.shape[1]):
            idx = rng.permutation(Xp.shape[0])
            Xp[:, t, j] = Xp[idx, t, j]
        pred = model.predict(Xp, verbose=0)
        mae = float(np.mean(np.abs(pred - y)))
        rows.append(
            {
                "feature": feature_columns[int(j)] if int(j) < len(feature_columns) else str(int(j)),
                "baseline_mae": float(baseline_mae),
                "shuffled_mae": mae,
                "delta_mae": float(mae - baseline_mae),
                "variance": float(variances[int(j)]),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("delta_mae", ascending=False).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM model for FPL point projections")
    parser.add_argument("--season", default="2025-2026")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--seq-length", type=int, default=DEFAULT_SEQ_LENGTH)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--val-gws", type=int, default=DEFAULT_VAL_GWS)
    parser.add_argument("--test-gws", type=int, default=DEFAULT_TEST_GWS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target",
        default=TARGET_COLUMN,
        help="Training target column (e.g., total_points or xp_proxy).",
    )

    parser.add_argument(
        "--no-role-loss-weighting",
        action="store_true",
        help="Disable per-role loss weighting (sample_weight) during training.",
    )

    parser.add_argument(
        "--bias-penalty-alpha",
        type=float,
        default=0.0,
        help="Add alpha * ReLU(pred - xGI_cap) penalty to discourage overpredicting low-xGI players (0 disables).",
    )

    parser.add_argument(
        "--monitor",
        choices=["val_loss", "val_macro_mse"],
        default="val_macro_mse",
        help="Early stopping / LR scheduling monitor. val_macro_mse is per-role macro average on validation.",
    )
    parser.add_argument(
        "--split-by-role",
        action="store_true",
        help="Train separate models per role (GK/DEF/MID/FWD). Saves under artifacts/models/<role>/.",
    )
    parser.add_argument(
        "--mid-split",
        action="store_true",
        help="When splitting by role, split MID into MID_DM and MID_AM using a heuristic.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Write per-role calibration bins and permutation importances under artifacts/diagnostics/.",
    )
    parser.add_argument(
        "--permute-max-features",
        type=int,
        default=25,
        help="Max features to evaluate for permutation importance (0 disables).",
    )
    args = parser.parse_args()

    _set_seed(args.seed)

    repo_root = Path(args.repo_root)
    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load data with feature engineering applied
    print("Loading and engineering features...")
    raw = load_premier_league_gameweek_stats(repo_root=repo_root, season=args.season, apply_feature_engineering=True)

    # Map player_id -> base position (stable across season)
    pid_to_pos: dict[int, object] = {}
    if "player_id" in raw.columns and "position" in raw.columns:
        meta = (
            raw.sort_values(["player_id", "gw"]).groupby("player_id", sort=False, as_index=False).tail(1)[
                ["player_id", "position"]
            ]
        )
        pid_to_pos = {int(r["player_id"]): r["position"] for _, r in meta.iterrows()}
    
    # Use only available features
    available_features = _get_available_features(raw, DEFAULT_FEATURE_COLUMNS)
    print(f"Using {len(available_features)} features for training")

    target_col = str(args.target)
    # Select and coerce numeric data
    df = select_and_coerce_numeric(raw, available_features, target_col)
    
    print(f"Building sequences with seq_length={args.seq_length}, horizon={args.horizon}")
    dataset = build_sequences(
        df=df,
        feature_columns=available_features,
        target_column=target_col,
        seq_length=args.seq_length,
        horizon=args.horizon,
    )

    # Determine role for each sequence (based on player position + last-timestep stats)
    role_labels: list[str] = []
    for i in range(dataset.X.shape[0]):
        pid = int(dataset.player_id[i])
        pos = pid_to_pos.get(pid)
        role = infer_role_from_window(pos, pd.DataFrame(dataset.X[i], columns=available_features), mid_split=args.mid_split)
        if role is None:
            role = position_to_role(pos)
        role_labels.append(role)
    roles_for_seq = np.asarray(role_labels, dtype=object)

    max_end_gw = int(dataset.end_gw.max())
    # Ensure some room: train | val | test in time order
    val_max_end_gw = max_end_gw - args.test_gws
    train_max_end_gw = val_max_end_gw - args.val_gws
    if train_max_end_gw < 1:
        raise ValueError(
            f"Not enough gameweeks to split: max_end_gw={max_end_gw}, "
            f"val_gws={args.val_gws}, test_gws={args.test_gws}."
        )

    print(f"Splitting data: train_max_gw={train_max_end_gw}, val_max_gw={val_max_end_gw}, test_max_gw={max_end_gw}")
    train_ds, val_ds, test_ds = split_by_end_gw(
        dataset,
        train_max_end_gw=train_max_end_gw,
        val_max_end_gw=val_max_end_gw,
    )
    # Split roles alongside the dataset splits so we can apply per-sample weights.
    train_roles, val_roles, test_roles = split_by_end_gw(
        type("_R", (), {"X": roles_for_seq, "y": roles_for_seq, "player_id": roles_for_seq, "end_gw": dataset.end_gw})(),
        train_max_end_gw=train_max_end_gw,
        val_max_end_gw=val_max_end_gw,
    )
    train_roles = train_roles.X
    val_roles = val_roles.X
    test_roles = test_roles.X
    
    print(f"Dataset sizes - Train: {len(train_ds.X)}, Val: {len(val_ds.X)}, Test: {len(test_ds.X)}")

    def _train_one(
        *,
        role: str,
        train_subset: SequenceDataset,
        val_subset: SequenceDataset,
        test_subset: SequenceDataset,
        feature_cols: list[str],
        out_dir: Path,
    ) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)

        n_features = train_subset.X.shape[-1]
        train_timesteps_2d = train_subset.X.reshape(train_subset.X.shape[0] * args.seq_length, n_features)
        print(f"Fitting preprocessing for {role} (median imputation + standardization)...")
        pipeline = fit_preprocessor_on_timesteps(train_timesteps_2d)

        print(f"Transforming sequences for {role}...")
        X_train = transform_sequences(pipeline, train_subset.X)
        X_val = transform_sequences(pipeline, val_subset.X)
        X_test = transform_sequences(pipeline, test_subset.X)

        # Apply role-aware weights after preprocessing.
        w = build_feature_weight_vector(feature_cols, role)
        X_train = X_train * w
        X_val = X_val * w
        X_test = X_test * w

        print(f"Building LSTM model for {role} (features={n_features}, horizon={args.horizon})")
        model = build_lstm_model(seq_length=args.seq_length, num_features=n_features, horizon=args.horizon)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        ]

        print(f"Training {role} for up to {args.epochs} epochs...")
        history = model.fit(
            X_train,
            train_subset.y,
            validation_data=(X_val, val_subset.y),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        print(f"Evaluating {role} on test set...")
        test_loss, test_mae = model.evaluate(X_test, test_subset.y, verbose=0)
        print(f"{role} Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

        # Ranking-centric diagnostics on this role's slice.
        pred = model.predict(X_test, verbose=0)
        metrics, calib_bins, per_role = evaluate_fpl_model(
            y_true=test_subset.y,
            y_pred=pred,
            player_id=test_subset.player_id,
            roles=np.asarray([role] * len(test_subset.player_id), dtype=object),
            top_n=50,
            n_calibration_bins=10,
        )
        print(
            f"{role} rank_corr={metrics['rank_correlation']:.3f} "
            f"top_50_recall={metrics.get('top_50_recall', float('nan')):.3f} "
            f"calib_err_rel={metrics['calibration_error_rel']:.3f}"
        )

        model_path = out_dir / "model.keras"
        preprocess_path = out_dir / "preprocess.joblib"
        metadata_path = out_dir / "meta.json"

        model.save(str(model_path))
        preprocess_artifacts = PreprocessArtifacts(feature_columns=feature_cols, pipeline=pipeline)
        preprocess_artifacts.save(str(preprocess_path))

        metadata = {
            "role": role,
            "season": args.season,
            "seq_length": args.seq_length,
            "horizon": args.horizon,
            "target_column": target_col,
            "num_features": int(n_features),
            "feature_columns": feature_cols,
            "test_loss": float(test_loss),
            "test_mae": float(test_mae),
            "rank_correlation": float(metrics["rank_correlation"]),
            "top_50_recall": float(metrics.get("top_50_recall", float("nan"))),
            "calibration_error": float(metrics["calibration_error"]),
            "calibration_error_rel": float(metrics["calibration_error_rel"]),
            "epochs_trained": int(len(history.history["loss"])),
            "final_train_loss": float(history.history["loss"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]),
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        if args.diagnostics:
            diag_dir = artifacts_dir / "diagnostics"
            diag_dir.mkdir(parents=True, exist_ok=True)
            baseline_mae = float(np.mean(np.abs(pred - test_subset.y)))
            calib_bins.to_csv(diag_dir / f"calibration_{role}.csv", index=False)
            per_role.to_csv(diag_dir / f"per_role_{role}.csv", index=False)
            with open(diag_dir / f"metrics_{role}.json", "w") as f:
                json.dump(metrics, f, indent=2)

            if args.permute_max_features and args.permute_max_features > 0:
                pimps = _permutation_importance_mae(
                    model=model,
                    X=X_test,
                    y=test_subset.y,
                    feature_columns=feature_cols,
                    baseline_mae=baseline_mae,
                    max_features=int(args.permute_max_features),
                    seed=int(args.seed),
                )
                pimps.to_csv(diag_dir / f"permutation_importance_{role}.csv", index=False)

        return metadata

    if args.split_by_role:
        model_root = artifacts_dir / "models"
        model_root.mkdir(parents=True, exist_ok=True)

        wanted_roles = list_roles(mid_split=args.mid_split)
        meta_models: dict[str, dict] = {"split_by_role": True, "mid_split": bool(args.mid_split), "roles": {}}

        for role in wanted_roles:
            role_train_mask = train_roles == role
            role_val_mask = val_roles == role
            role_test_mask = test_roles == role

            if not np.any(role_train_mask):
                print(f"Skipping {role}: no training samples")
                continue

            def _sub(ds: SequenceDataset, m: np.ndarray) -> SequenceDataset:
                return type(ds)(X=ds.X[m], y=ds.y[m], player_id=ds.player_id[m], end_gw=ds.end_gw[m])

            role_train = _sub(train_ds, role_train_mask)
            role_val = _sub(val_ds, role_val_mask) if np.any(role_val_mask) else _sub(train_ds, role_train_mask)
            role_test = _sub(test_ds, role_test_mask) if np.any(role_test_mask) else _sub(val_ds, role_val_mask)

            out_dir = model_root / role
            md = _train_one(
                role=role,
                train_subset=role_train,
                val_subset=role_val,
                test_subset=role_test,
                feature_cols=available_features,
                out_dir=out_dir,
            )
            meta_models["roles"][role] = md

        with open(model_root / "meta_models.json", "w") as f:
            json.dump(meta_models, f, indent=2)

        print("\nRole-split training complete!")
        return

    # ---- Single model (backwards-compatible) ----
    n_features = train_ds.X.shape[-1]
    train_timesteps_2d = train_ds.X.reshape(train_ds.X.shape[0] * args.seq_length, n_features)
    print("Fitting preprocessing pipeline (median imputation + standardization)...")
    pipeline = fit_preprocessor_on_timesteps(train_timesteps_2d)

    print("Transforming sequences...")
    X_train = transform_sequences(pipeline, train_ds.X)
    X_val = transform_sequences(pipeline, val_ds.X)
    X_test = transform_sequences(pipeline, test_ds.X)

    # Apply per-sample role-aware weights (strong guardrail against DM/consistency bias).
    unique_roles = sorted(set(str(r) for r in np.unique(np.concatenate([train_roles, val_roles, test_roles]))))
    role_to_w = {r: build_feature_weight_vector(available_features, r) for r in unique_roles}

    def _apply_role_weights(X: np.ndarray, roles: np.ndarray) -> np.ndarray:
        W = np.stack([role_to_w.get(str(r), np.ones(X.shape[-1])) for r in roles], axis=0)
        return X * W[:, None, :]

    X_train = _apply_role_weights(X_train, train_roles)
    X_val = _apply_role_weights(X_val, val_roles)
    X_test = _apply_role_weights(X_test, test_roles)

    # Role-weighted loss (sample weights): attacker errors matter more; MID-DM stability matters less.
    if not bool(args.no_role_loss_weighting):
        sw_train = np.asarray([role_loss_weight(r) for r in train_roles], dtype=float)
        sw_val = np.asarray([role_loss_weight(r) for r in val_roles], dtype=float)
    else:
        sw_train = None
        sw_val = None

    # Optional xGI-based bias penalty: augment y with a per-sample cap so loss can access it.
    use_bias_penalty = float(args.bias_penalty_alpha) > 0.0
    if use_bias_penalty:
        cap_train = _xgi_cap_from_raw_sequences(train_ds.X, train_roles, feature_columns=available_features)
        cap_val = _xgi_cap_from_raw_sequences(val_ds.X, val_roles, feature_columns=available_features)
        cap_test = _xgi_cap_from_raw_sequences(test_ds.X, test_roles, feature_columns=available_features)

        cap_train = np.repeat(cap_train.reshape(-1, 1), args.horizon, axis=1)
        cap_val = np.repeat(cap_val.reshape(-1, 1), args.horizon, axis=1)
        cap_test = np.repeat(cap_test.reshape(-1, 1), args.horizon, axis=1)
        y_train = np.stack([train_ds.y, cap_train], axis=-1)
        y_val = np.stack([val_ds.y, cap_val], axis=-1)
        y_test = np.stack([test_ds.y, cap_test], axis=-1)
    else:
        y_train = train_ds.y
        y_val = val_ds.y
        y_test = test_ds.y

    print(f"Building LSTM model (seq_length={args.seq_length}, features={n_features}, horizon={args.horizon})")
    model = build_lstm_model(seq_length=args.seq_length, num_features=n_features, horizon=args.horizon)

    # If using bias penalty, override the compile loss/metrics so they can slice y_true.
    if use_bias_penalty:
        alpha = float(args.bias_penalty_alpha)

        def _loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            # y_true: (batch, horizon, 2) -> [:,:,0]=target, [:,:,1]=cap
            y = y_true[..., 0]
            cap = y_true[..., 1]
            mse = tf.reduce_mean(tf.square(y_pred - y), axis=-1)
            penalty = tf.reduce_mean(tf.nn.relu(y_pred - cap), axis=-1)
            return mse + alpha * penalty

        def _mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            y = y_true[..., 0]
            return tf.reduce_mean(tf.abs(y_pred - y), axis=-1)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=_loss,
            metrics=[tf.keras.metrics.MeanMetricWrapper(_mae, name="mae")],
        )

    class _PerRoleValMetrics(tf.keras.callbacks.Callback):
        def __init__(self, Xv: np.ndarray, yv: np.ndarray, roles_v: np.ndarray):
            super().__init__()
            self.Xv = Xv
            self.yv = yv
            self.roles_v = np.asarray(roles_v, dtype=object)

        def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
            logs = logs or {}
            pred = self.model.predict(self.Xv, verbose=0)

            if self.yv.ndim == 3 and self.yv.shape[-1] >= 1:
                y_true = self.yv[..., 0]
            else:
                y_true = self.yv

            per_role: dict[str, float] = {}
            for r in sorted(set(str(v) for v in np.unique(self.roles_v))):
                m = self.roles_v == r
                if not np.any(m):
                    continue
                err = pred[m] - y_true[m]
                per_role[r] = float(np.mean(np.square(err)))

            if per_role:
                macro = float(np.mean(list(per_role.values())))
                logs["val_macro_mse"] = macro
                for r, v in per_role.items():
                    logs[f"val_mse_{r}"] = float(v)

    monitor = str(args.monitor)
    val_metrics_cb = _PerRoleValMetrics(X_val, y_val, val_roles)

    callbacks = [
        val_metrics_cb,
        tf.keras.callbacks.EarlyStopping(monitor=monitor, mode="min", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode="min", factor=0.5, patience=4, min_lr=1e-6),
    ]

    print(f"Training for up to {args.epochs} epochs with batch_size={args.batch_size}...")
    history = model.fit(
        X_train,
        y_train,
        sample_weight=sw_train,
        validation_data=(X_val, y_val, sw_val) if sw_val is not None else (X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nEvaluating on test set...")
    if sw_train is not None:
        sw_test = np.asarray([role_loss_weight(r) for r in test_roles], dtype=float)
    else:
        sw_test = None
    test_loss, test_mae = model.evaluate(X_test, y_test, sample_weight=sw_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

    # Ranking-centric diagnostics: compute from raw y and predictions.
    pred = model.predict(X_test, verbose=0)
    y_true_for_metrics = y_test[..., 0] if (y_test.ndim == 3 and y_test.shape[-1] >= 1) else y_test
    metrics, calib_bins, per_role = evaluate_fpl_model(
        y_true=y_true_for_metrics,
        y_pred=pred,
        player_id=test_ds.player_id,
        roles=test_roles,
        top_n=50,
        n_calibration_bins=10,
    )
    print(
        "Test ranking diagnostics: "
        f"rank_corr={metrics['rank_correlation']:.3f} "
        f"top_50_recall={metrics.get('top_50_recall', float('nan')):.3f} "
        f"calib_err_rel={metrics['calibration_error_rel']:.3f}"
    )

    if args.diagnostics:
        diag_dir = artifacts_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        calib_bins.to_csv(diag_dir / "calibration_ALL.csv", index=False)
        per_role.to_csv(diag_dir / "per_role_ALL.csv", index=False)
        with open(diag_dir / "metrics_ALL.json", "w") as f:
            json.dump(metrics, f, indent=2)

    model_path = artifacts_dir / "model.keras"
    preprocess_path = artifacts_dir / "preprocess.joblib"
    metadata_path = artifacts_dir / "meta.json"

    print(f"\nSaving artifacts to {artifacts_dir}...")
    model.save(str(model_path))
    preprocess_artifacts = PreprocessArtifacts(feature_columns=available_features, pipeline=pipeline)
    preprocess_artifacts.save(str(preprocess_path))

    metadata = {
        "season": args.season,
        "seq_length": args.seq_length,
        "horizon": args.horizon,
        "target_column": target_col,
        "num_features": n_features,
        "feature_columns": available_features,
        "test_loss": float(test_loss),
        "test_mae": float(test_mae),
        "rank_correlation": float(metrics["rank_correlation"]),
        "top_50_recall": float(metrics.get("top_50_recall", float("nan"))),
        "calibration_error": float(metrics["calibration_error"]),
        "calibration_error_rel": float(metrics["calibration_error_rel"]),
        "epochs_trained": len(history.history["loss"]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Preprocessor saved to {preprocess_path}")
    print(f"Metadata saved to {metadata_path}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

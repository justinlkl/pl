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
from .sequences import build_sequences, split_by_end_gw
from .role_modeling import (
    build_feature_weight_vector,
    infer_role_from_window,
    list_roles,
    position_to_role,
)


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


def _calibration_bins(pred: np.ndarray, actual: np.ndarray, *, n_bins: int = 10) -> pd.DataFrame:
    """Return equal-frequency calibration bins for regression."""
    pred = pred.reshape(-1)
    actual = actual.reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(actual)
    pred = pred[mask]
    actual = actual[mask]
    if pred.size == 0:
        return pd.DataFrame(columns=["bin", "count", "pred_mean", "actual_mean", "pred_min", "pred_max"])

    q = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(pred, q)
    # Ensure monotonically increasing edges.
    edges = np.unique(edges)
    if edges.size < 3:
        return pd.DataFrame(
            {
                "bin": [0],
                "count": [int(pred.size)],
                "pred_mean": [float(np.mean(pred))],
                "actual_mean": [float(np.mean(actual))],
                "pred_min": [float(np.min(pred))],
                "pred_max": [float(np.max(pred))],
            }
        )

    bin_ids = np.digitize(pred, edges[1:-1], right=False)
    rows: list[dict] = []
    for b in range(int(np.max(bin_ids)) + 1):
        m = bin_ids == b
        if not np.any(m):
            continue
        rows.append(
            {
                "bin": int(b),
                "count": int(np.sum(m)),
                "pred_mean": float(np.mean(pred[m])),
                "actual_mean": float(np.mean(actual[m])),
                "pred_min": float(np.min(pred[m])),
                "pred_max": float(np.max(pred[m])),
            }
        )
    return pd.DataFrame(rows)


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
        train_subset: "SequenceDataset",
        val_subset: "SequenceDataset",
        test_subset: "SequenceDataset",
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
            "epochs_trained": int(len(history.history["loss"])),
            "final_train_loss": float(history.history["loss"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]),
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        if args.diagnostics:
            diag_dir = artifacts_dir / "diagnostics"
            diag_dir.mkdir(parents=True, exist_ok=True)
            pred = model.predict(X_test, verbose=0)
            baseline_mae = float(np.mean(np.abs(pred - test_subset.y)))
            calib = _calibration_bins(pred, test_subset.y, n_bins=10)
            calib.to_csv(diag_dir / f"calibration_{role}.csv", index=False)

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

            def _sub(ds: "SequenceDataset", m: np.ndarray) -> "SequenceDataset":
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

    print(f"Building LSTM model (seq_length={args.seq_length}, features={n_features}, horizon={args.horizon})")
    model = build_lstm_model(seq_length=args.seq_length, num_features=n_features, horizon=args.horizon)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]

    print(f"Training for up to {args.epochs} epochs with batch_size={args.batch_size}...")
    history = model.fit(
        X_train,
        train_ds.y,
        validation_data=(X_val, val_ds.y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nEvaluating on test set...")
    test_loss, test_mae = model.evaluate(X_test, test_ds.y, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

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

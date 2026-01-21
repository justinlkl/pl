from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import DEFAULT_FEATURE_COLUMNS, DEFAULT_HORIZON, DEFAULT_SEQ_LENGTH, TARGET_COLUMN
from .data_loading import load_premier_league_gameweek_stats
from .modeling import build_lstm_model
from .preprocessing import PreprocessArtifacts, fit_preprocessor_on_timesteps, select_and_coerce_numeric, transform_sequences
from .sequences import build_sequences, split_by_end_gw
from .evaluation import evaluate_fpl_model
from .role_modeling import build_feature_weight_vector, infer_role_from_window, position_to_role
from .role_modeling import role_loss_weight
from .role_modeling import fit_role_projection_multipliers, save_role_scaling


def _xgi_cap_from_raw_sequences(
    X_raw: np.ndarray,
    roles: np.ndarray,
    *,
    feature_columns: list[str],
) -> np.ndarray:
    idx = {c: i for i, c in enumerate(feature_columns)}
    minutes_i = idx.get("minutes")
    xgi90_i = idx.get("expected_goal_involvements_per_90")
    xg90_i = idx.get("expected_goals_per_90")
    xa90_i = idx.get("expected_assists_per_90")

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

        cap = 2.0 * p60 + rate * xgi
        if role in {"DEF", "GK"}:
            cap += 0.8 * p60
        caps.append(float(max(cap, 0.0)))

    return np.asarray(caps, dtype=float)


def _require_pycaret() -> None:
    try:
        import pycaret  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "PyCaret is not installed in this environment.\n"
            "Install the ensemble dependencies, then re-run:\n\n"
            "  pip install pycaret lightgbm catboost\n\n"
            f"Original import error: {exc}"
        )


def _set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


def _get_available_features(df: pd.DataFrame, requested_features: list[str]) -> list[str]:
    available = [f for f in requested_features if f in df.columns]
    missing = sorted(set(requested_features) - set(available))
    if missing:
        print(f"Warning: {len(missing)} configured features not in data (skipping): {missing}")
    return available


def _build_tabular_base(df: pd.DataFrame, *, feature_columns: list[str], target_column: str) -> pd.DataFrame:
    """Create a per-(player_id, gw) tabular frame with engineered features."""
    keep = ["player_id", "gw", "web_name", *feature_columns, target_column]
    out = df[keep].copy()
    out = out.sort_values(["gw", "player_id"]).reset_index(drop=True)

    # Next-GW target(s) are constructed later from sequence labels.
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LSTM (Phase A) and PyCaret tree+meta ensemble (Phase B / Layer 2) without shuffling"
    )
    parser.add_argument("--season", default="2025-2026")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--seq-length", type=int, default=DEFAULT_SEQ_LENGTH)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--val-gws", type=int, default=3)
    parser.add_argument("--test-gws", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=8,
        help="Early stopping patience (epochs) for LSTM training.",
    )
    parser.add_argument(
        "--lr-plateau-patience",
        type=int,
        default=4,
        help="ReduceLROnPlateau patience (epochs) for LSTM training.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--target",
        default=TARGET_COLUMN,
        help="Training target column (e.g., total_points or xp_proxy).",
    )

    parser.add_argument(
        "--no-role-loss-weighting",
        action="store_true",
        help="Disable per-role loss weighting (sample_weight) during LSTM training.",
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
        help="Early stopping / LR scheduling monitor for LSTM training.",
    )

    parser.add_argument(
        "--mid-split",
        action="store_true",
        help="Use a heuristic to split MID into MID_DM vs MID_AM for role weights.",
    )

    parser.add_argument(
        "--lstm-pred-scale",
        type=float,
        default=1.5,
        help="Scale lstm_pred_h* features before training the stacker (boost LSTM influence).",
    )

    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--folds", type=int, default=5)

    parser.add_argument(
        "--lstm-only",
        action="store_true",
        help="Train and save only the LSTM + preprocessing artifacts (skip PyCaret stacking).",
    )

    parser.add_argument(
        "--backend",
        choices=["sklearn", "pycaret"],
        default="sklearn",
        help=(
            "Stacking backend. Default is sklearn (works on Python 3.13). "
            "pycaret requires Python < 3.12 and extra dependencies."
        ),
    )

    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--out-dir", default=None)

    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Write LSTM and ensemble evaluation files under <out_dir>/diagnostics/.",
    )
    args = parser.parse_args()

    use_pycaret = (not bool(args.lstm_only)) and (str(args.backend).lower() == "pycaret")
    if use_pycaret:
        _require_pycaret()
        from pycaret.regression import (  # type: ignore
            create_model,
            finalize_model,
            predict_model,
            save_model,
            setup,
            stack_models,
        )
    else:
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, StackingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold
        import joblib

    _set_seed(args.seed)

    repo_root = Path(args.repo_root)
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else (repo_root / "artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path(args.out_dir) if args.out_dir else (artifacts_dir / "ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and engineering features...")
    raw = load_premier_league_gameweek_stats(repo_root=repo_root, season=args.season, apply_feature_engineering=True)

    target_col = str(args.target)

    feature_columns = _get_available_features(raw, DEFAULT_FEATURE_COLUMNS)
    df_num = select_and_coerce_numeric(raw, feature_columns, target_col)

    print(f"Building sequences (no shuffle): seq_length={args.seq_length}, horizon={args.horizon}")
    dataset = build_sequences(
        df=df_num,
        feature_columns=feature_columns,
        target_column=target_col,
        seq_length=args.seq_length,
        horizon=args.horizon,
    )

    # Role labels per sequence so we can apply per-sample feature weights.
    pid_to_pos: dict[int, object] = {}
    if "player_id" in raw.columns and "position" in raw.columns:
        meta = (
            raw.sort_values(["player_id", "gw"]).groupby("player_id", sort=False, as_index=False).tail(1)[
                ["player_id", "position"]
            ]
        )
        pid_to_pos = {int(r["player_id"]): r["position"] for _, r in meta.iterrows()}

    role_labels: list[str] = []
    for i in range(dataset.X.shape[0]):
        pid = int(dataset.player_id[i])
        pos = pid_to_pos.get(pid)
        role = infer_role_from_window(pos, pd.DataFrame(dataset.X[i], columns=feature_columns), mid_split=bool(args.mid_split))
        if role is None:
            role = position_to_role(pos)
        role_labels.append(role)
    roles_for_seq = np.asarray(role_labels, dtype=object)

    max_end_gw = int(dataset.end_gw.max())
    val_max_end_gw = max_end_gw - args.test_gws
    train_max_end_gw = val_max_end_gw - args.val_gws
    if train_max_end_gw < 1:
        raise ValueError(
            f"Not enough gameweeks to split: max_end_gw={max_end_gw}, val_gws={args.val_gws}, test_gws={args.test_gws}."
        )

    train_ds, val_ds, test_ds = split_by_end_gw(dataset, train_max_end_gw=train_max_end_gw, val_max_end_gw=val_max_end_gw)

    print("Fitting preprocessing on training timesteps only...")
    n_features = train_ds.X.shape[-1]
    train_timesteps_2d = train_ds.X.reshape(train_ds.X.shape[0] * args.seq_length, n_features)
    pipeline = fit_preprocessor_on_timesteps(train_timesteps_2d)

    X_train = transform_sequences(pipeline, train_ds.X)
    X_val = transform_sequences(pipeline, val_ds.X)
    X_test = transform_sequences(pipeline, test_ds.X)

    # Apply per-sample role weights (guardrail against MID-DM stability bias).
    def _apply_role_weights(X: np.ndarray, roles: np.ndarray) -> np.ndarray:
        uniq = sorted(set(str(r) for r in np.unique(roles)))
        role_to_w = {r: build_feature_weight_vector(feature_columns, r) for r in uniq}
        W = np.stack([role_to_w.get(str(r), np.ones(X.shape[-1])) for r in roles], axis=0)
        return X * W[:, None, :]

    # Split roles alongside the end_gw split
    train_roles, val_roles, test_roles = split_by_end_gw(
        type("_R", (), {"X": roles_for_seq, "y": roles_for_seq, "player_id": roles_for_seq, "end_gw": dataset.end_gw})(),
        train_max_end_gw=train_max_end_gw,
        val_max_end_gw=val_max_end_gw,
    )
    train_roles = train_roles.X
    val_roles = val_roles.X
    test_roles = test_roles.X

    X_train = _apply_role_weights(X_train, train_roles)
    X_val = _apply_role_weights(X_val, val_roles)
    X_test = _apply_role_weights(X_test, test_roles)

    # Role-weighted loss: attacker errors matter more; MID-DM stability matters less.
    if not bool(args.no_role_loss_weighting):
        sw_train = np.asarray([role_loss_weight(r) for r in train_roles], dtype=float)
        sw_val = np.asarray([role_loss_weight(r) for r in val_roles], dtype=float)
    else:
        sw_train = None
        sw_val = None

    use_bias_penalty = float(args.bias_penalty_alpha) > 0.0
    if use_bias_penalty:
        cap_train = _xgi_cap_from_raw_sequences(train_ds.X, train_roles, feature_columns=feature_columns)
        cap_val = _xgi_cap_from_raw_sequences(val_ds.X, val_roles, feature_columns=feature_columns)
        cap_train = np.repeat(cap_train.reshape(-1, 1), args.horizon, axis=1)
        cap_val = np.repeat(cap_val.reshape(-1, 1), args.horizon, axis=1)
        y_train = np.stack([train_ds.y, cap_train], axis=-1)
        y_val = np.stack([val_ds.y, cap_val], axis=-1)
    else:
        y_train = train_ds.y
        y_val = val_ds.y

    print("Training LSTM (Phase A)...")
    lstm = build_lstm_model(seq_length=args.seq_length, num_features=n_features, horizon=args.horizon)

    if use_bias_penalty:
        alpha = float(args.bias_penalty_alpha)

        def _loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            y = y_true[..., 0]
            cap = y_true[..., 1]
            mse = tf.reduce_mean(tf.square(y_pred - y), axis=-1)
            penalty = tf.reduce_mean(tf.nn.relu(y_pred - cap), axis=-1)
            return mse + alpha * penalty

        def _mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            y = y_true[..., 0]
            return tf.reduce_mean(tf.abs(y_pred - y), axis=-1)

        lstm.compile(
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
                logs["val_macro_mse"] = float(np.mean(list(per_role.values())))
                for r, v in per_role.items():
                    logs[f"val_mse_{r}"] = float(v)

    monitor = str(args.monitor)
    callbacks = [
        _PerRoleValMetrics(X_val, y_val, val_roles),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="min",
            patience=int(args.early_stopping_patience),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode="min",
            factor=0.5,
            patience=int(args.lr_plateau_patience),
            min_lr=1e-6,
        ),
    ]
    lstm.fit(
        X_train,
        y_train,
        sample_weight=sw_train,
        validation_data=(X_val, y_val, sw_val) if sw_val is not None else (X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate LSTM on the held-out test window.
    X_test_pred = X_test
    y_test_true = test_ds.y
    lstm_pred_test = lstm.predict(X_test_pred, verbose=0)
    metrics_lstm, calib_lstm, per_role_lstm = evaluate_fpl_model(
        y_true=y_test_true,
        y_pred=lstm_pred_test,
        player_id=test_ds.player_id,
        roles=test_roles,
        top_n=50,
        n_calibration_bins=10,
    )
    print(
        "LSTM test diagnostics: "
        f"rank_corr={metrics_lstm['rank_correlation']:.3f} "
        f"top_50_recall={metrics_lstm.get('top_50_recall', float('nan')):.3f} "
        f"calib_err_rel={metrics_lstm['calibration_error_rel']:.3f} "
        f"mae={metrics_lstm['mae']:.3f}"
    )

    print("Saving LSTM artifacts...")
    (out_dir / "lstm_model.keras").parent.mkdir(parents=True, exist_ok=True)
    lstm.save(str(out_dir / "lstm_model.keras"))
    PreprocessArtifacts(feature_columns=feature_columns, pipeline=pipeline).save(str(out_dir / "preprocess.joblib"))

    if args.lstm_only:
        print(f"\nSaved LSTM-only artifacts to: {out_dir}")
        print("Skipping PyCaret stacking (Phase B) because --lstm-only was set.")

        if args.diagnostics:
            diag_dir = out_dir / "diagnostics"
            diag_dir.mkdir(parents=True, exist_ok=True)
            (diag_dir / "metrics_LSTM.json").write_text(pd.Series(metrics_lstm).to_json(indent=2))
            calib_lstm.to_csv(diag_dir / "calibration_LSTM.csv", index=False)
            per_role_lstm.to_csv(diag_dir / "per_role_LSTM.csv", index=False)
        return

    # Build LSTM predictions for ALL sequence samples (train+val+test) to feed Phase B.
    print("Generating LSTM predictions for stacking features...")
    X_all = transform_sequences(pipeline, dataset.X)
    X_all = _apply_role_weights(X_all, roles_for_seq)
    lstm_preds = lstm.predict(X_all, verbose=0)

    seq_df = pd.DataFrame(
        {
            "player_id": dataset.player_id,
            "end_gw": dataset.end_gw,
            "role": roles_for_seq,
        }
    )
    for k in range(args.horizon):
        seq_df[f"target_h{k+1}"] = dataset.y[:, k]
        seq_df[f"lstm_pred_h{k+1}"] = lstm_preds[:, k]

    # Join to the last-timestep (player_id, gw=end_gw) tabular features.
    base = _build_tabular_base(df_num, feature_columns=feature_columns, target_column=target_col)
    base = base.rename(columns={"gw": "end_gw"})

    # Apply role weights to the last-timestep tabular features as well.
    if "player_id" in base.columns and "end_gw" in base.columns:
        base = base.merge(raw[[c for c in ["player_id", "gw", "position"] if c in raw.columns]].rename(columns={"gw": "end_gw"}),
                          on=["player_id", "end_gw"], how="left")
        roles_row: list[str] = []
        for _, r in base.iterrows():
            pos = r.get("position")
            window = pd.DataFrame([r])
            role = infer_role_from_window(pos, window, mid_split=bool(args.mid_split))
            if role is None:
                role = position_to_role(pos)
            roles_row.append(role)
        roles_row_arr = np.asarray(roles_row, dtype=object)
        uniq = sorted(set(str(v) for v in np.unique(roles_row_arr)))
        role_to_w = {r: build_feature_weight_vector(feature_columns, r) for r in uniq}
        W = np.stack([role_to_w.get(str(r), np.ones(len(feature_columns))) for r in roles_row_arr], axis=0)
        base.loc[:, feature_columns] = base[feature_columns].to_numpy(dtype=float) * W

    tab = seq_df.merge(base, on=["player_id", "end_gw"], how="left")
    tab = tab.sort_values(["end_gw", "player_id"]).reset_index(drop=True)

    # Drop any rows where the join failed.
    tab = tab.dropna(subset=[*feature_columns]).copy()

    # Split stacking table by time: train+val only, evaluate on held-out test.
    train_tab = tab[tab["end_gw"] <= val_max_end_gw].copy()
    test_tab = tab[tab["end_gw"] > val_max_end_gw].copy()
    if train_tab.empty or test_tab.empty:
        raise ValueError(
            f"Ensemble split failed: train_tab={len(train_tab)} rows, test_tab={len(test_tab)} rows. "
            f"val_max_end_gw={val_max_end_gw}, max_end_gw={max_end_gw}."
        )

    # Phase B: Train per-horizon stacked model on train+val only, evaluate on held-out test.
    print(f"Training stacking models (Phase B) backend={args.backend}...")

    # Collect test predictions per horizon so we can evaluate the ensemble end-to-end.
    ensemble_pred_test = np.zeros((len(test_tab), args.horizon), dtype=float)
    ensemble_true_test = np.zeros((len(test_tab), args.horizon), dtype=float)

    for h in range(1, args.horizon + 1):
        target_col = f"target_h{h}"
        print(f"\n=== Horizon {h}: training target={target_col} ===")

        data_h_all = train_tab.drop(columns=[c for c in train_tab.columns if c.startswith("target_h") and c != target_col]).copy()
        data_h_test = test_tab.drop(columns=[c for c in test_tab.columns if c.startswith("target_h") and c != target_col]).copy()

        # Keep only one LSTM prediction column for this horizon, plus all engineered tabular features.
        keep_cols = [
            "player_id",
            "web_name",
            "end_gw",
            *feature_columns,
            f"lstm_pred_h{h}",
            target_col,
        ]
        data_h = data_h_all[keep_cols].copy()
        data_h_test = data_h_test[keep_cols].copy()

        # Boost LSTM influence in the stacker by scaling the LSTM prediction feature.
        data_h[f"lstm_pred_h{h}"] = pd.to_numeric(data_h[f"lstm_pred_h{h}"], errors="coerce") * float(args.lstm_pred_scale)
        data_h_test[f"lstm_pred_h{h}"] = pd.to_numeric(data_h_test[f"lstm_pred_h{h}"], errors="coerce") * float(
            args.lstm_pred_scale
        )

        if use_pycaret:
            ignore_cols = ["player_id", "web_name", "end_gw", target_col]

            setup(
                data=data_h,
                target=target_col,
                session_id=args.seed,
                fold_strategy="timeseries",
                fold=args.folds,
                data_split_shuffle=False,
                fold_shuffle=False,
                ignore_features=ignore_cols,
                verbose=False,
            )

            lgbm = create_model("lightgbm", early_stopping_rounds=args.early_stopping_rounds)
            cat = create_model("catboost", early_stopping_rounds=args.early_stopping_rounds)

            ridge = create_model("ridge")
            stacked = stack_models(estimator_list=[cat, lgbm], meta_model=ridge)
            final = finalize_model(stacked)

            save_model(final, str(out_dir / f"stack_h{h}"))

            preds_h = predict_model(final, data=data_h)
            preds_h = preds_h.sort_values(["end_gw", "player_id"]).reset_index(drop=True)
            preds_h.to_csv(out_dir / f"train_preds_h{h}.csv", index=False)

            preds_test_h = predict_model(final, data=data_h_test)
            if "prediction_label" not in preds_test_h.columns:
                raise ValueError("PyCaret prediction output missing 'prediction_label' column")
            ensemble_pred_test[:, h - 1] = (
                pd.to_numeric(preds_test_h["prediction_label"], errors="coerce").to_numpy(dtype=float)
            )
            ensemble_true_test[:, h - 1] = pd.to_numeric(data_h_test[target_col], errors="coerce").to_numpy(dtype=float)

        else:
            # sklearn backend: simple stacking on numeric features.
            feature_cols = [c for c in keep_cols if c not in {"player_id", "web_name", "end_gw", target_col}]
            Xtr = data_h[feature_cols].to_numpy(dtype=float)
            ytr = pd.to_numeric(data_h[target_col], errors="coerce").to_numpy(dtype=float)
            Xte = data_h_test[feature_cols].to_numpy(dtype=float)
            yte = pd.to_numeric(data_h_test[target_col], errors="coerce").to_numpy(dtype=float)

            # Note: StackingRegressor requires a partitioning CV (each sample appears in exactly
            # one test fold). TimeSeriesSplit leaves early samples out of all test folds, which
            # causes a runtime error. KFold(with shuffle=False) gives a full partition.
            cv = KFold(n_splits=int(args.folds), shuffle=False)
            base = [
                ("hgb", HistGradientBoostingRegressor(random_state=args.seed)),
                ("etr", ExtraTreesRegressor(n_estimators=300, random_state=args.seed, n_jobs=-1)),
            ]
            model = StackingRegressor(
                estimators=base,
                final_estimator=Ridge(alpha=1.0, random_state=args.seed),
                cv=cv,
                passthrough=False,
                n_jobs=-1,
            )
            model.fit(Xtr, ytr)
            joblib.dump(model, out_dir / f"stack_h{h}.joblib")

            pred_te = model.predict(Xte)
            ensemble_pred_test[:, h - 1] = np.asarray(pred_te, dtype=float)
            ensemble_true_test[:, h - 1] = np.asarray(yte, dtype=float)

    # Evaluate the ensemble on the held-out test window at player-level.
    metrics_ens, calib_ens, per_role_ens = evaluate_fpl_model(
        y_true=ensemble_true_test,
        y_pred=ensemble_pred_test,
        player_id=test_tab["player_id"].to_numpy(dtype=int),
        roles=test_tab["role"].to_numpy(dtype=object),
        top_n=50,
        n_calibration_bins=10,
    )
    print(
        "Ensemble test diagnostics: "
        f"rank_corr={metrics_ens['rank_correlation']:.3f} "
        f"top_50_recall={metrics_ens.get('top_50_recall', float('nan')):.3f} "
        f"calib_err_rel={metrics_ens['calibration_error_rel']:.3f} "
        f"mae={metrics_ens['mae']:.3f}"
    )

    # ---- Per-role calibration optimization (best-effort) ----
    try:
        overrides, report = fit_role_projection_multipliers(
            y_true=ensemble_true_test,
            y_pred=ensemble_pred_test,
            player_id=test_tab["player_id"].to_numpy(dtype=int),
            roles=test_tab["role"].to_numpy(dtype=object),
        )
        if overrides:
            save_role_scaling(
                out_dir / "role_scaling.json",
                overrides=overrides,
                report=report,
                meta={"fitted_on": "test_window"},
            )
            print(f"Saved role scaling overrides to {out_dir / 'role_scaling.json'}")
    except Exception as exc:
        print(f"Warning: failed to fit/save role scaling overrides: {exc}")

    if args.diagnostics:
        diag_dir = out_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        (diag_dir / "metrics_LSTM.json").write_text(pd.Series(metrics_lstm).to_json(indent=2))
        calib_lstm.to_csv(diag_dir / "calibration_LSTM.csv", index=False)
        per_role_lstm.to_csv(diag_dir / "per_role_LSTM.csv", index=False)

        (diag_dir / "metrics_ENSEMBLE.json").write_text(pd.Series(metrics_ens).to_json(indent=2))
        calib_ens.to_csv(diag_dir / "calibration_ENSEMBLE.csv", index=False)
        per_role_ens.to_csv(diag_dir / "per_role_ENSEMBLE.csv", index=False)

    print(f"\nSaved stacked models to: {out_dir}")


if __name__ == "__main__":
    main()

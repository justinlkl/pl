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
from .role_modeling import build_feature_weight_vector, infer_role_from_window, position_to_role


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
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--target",
        default=TARGET_COLUMN,
        help="Training target column (e.g., total_points or xp_proxy).",
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

    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if not args.lstm_only:
        _require_pycaret()
        from pycaret.regression import (  # type: ignore
            create_model,
            finalize_model,
            predict_model,
            save_model,
            setup,
            stack_models,
        )

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

    print("Training LSTM (Phase A)...")
    lstm = build_lstm_model(seq_length=args.seq_length, num_features=n_features, horizon=args.horizon)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]
    lstm.fit(
        X_train,
        train_ds.y,
        validation_data=(X_val, val_ds.y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print("Saving LSTM artifacts...")
    (out_dir / "lstm_model.keras").parent.mkdir(parents=True, exist_ok=True)
    lstm.save(str(out_dir / "lstm_model.keras"))
    PreprocessArtifacts(feature_columns=feature_columns, pipeline=pipeline).save(str(out_dir / "preprocess.joblib"))

    if args.lstm_only:
        print(f"\nSaved LSTM-only artifacts to: {out_dir}")
        print("Skipping PyCaret stacking (Phase B) because --lstm-only was set.")
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

    # Phase B: Train per-horizon stacked model using TimeSeriesSplit.
    print("Training PyCaret models (Phase B) with fold_strategy='timeseries'...")
    ignore_cols = ["player_id", "web_name", "end_gw", target_col]

    for h in range(1, args.horizon + 1):
        target_col = f"target_h{h}"
        print(f"\n=== Horizon {h}: training target={target_col} ===")

        data_h = tab.drop(columns=[c for c in tab.columns if c.startswith("target_h") and c != target_col]).copy()

        # Keep only one LSTM prediction column for this horizon, plus all engineered tabular features.
        keep_cols = [
            "player_id",
            "web_name",
            "end_gw",
            *feature_columns,
            f"lstm_pred_h{h}",
            target_col,
        ]
        data_h = data_h[keep_cols].copy()

        # Boost LSTM influence in the stacker by scaling the LSTM prediction feature.
        data_h[f"lstm_pred_h{h}"] = pd.to_numeric(data_h[f"lstm_pred_h{h}"], errors="coerce") * float(args.lstm_pred_scale)

        # Setup: no shuffle, time-series CV based on row order (sorted by end_gw).
        exp = setup(
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

        # Train base models with early stopping where supported.
        lgbm = create_model("lightgbm", early_stopping_rounds=args.early_stopping_rounds)
        cat = create_model("catboost", early_stopping_rounds=args.early_stopping_rounds)

        # Stack: meta learner (ridge is a strong default).
        ridge = create_model("ridge")
        stacked = stack_models(estimator_list=[cat, lgbm], meta_model=ridge)
        final = finalize_model(stacked)

        save_model(final, str(out_dir / f"stack_h{h}"))

        # Write quick in-sample preds for inspection (sorted by time).
        preds_h = predict_model(final, data=data_h)
        preds_h = preds_h.sort_values(["end_gw", "player_id"]).reset_index(drop=True)
        preds_h.to_csv(out_dir / f"train_preds_h{h}.csv", index=False)

    print(f"\nSaved stacked models to: {out_dir}")


if __name__ == "__main__":
    main()

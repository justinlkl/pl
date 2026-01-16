from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import DEFAULT_FEATURE_COLUMNS, DEFAULT_HORIZON, DEFAULT_SEQ_LENGTH, TARGET_COLUMN
from .data_loading import load_premier_league_gameweek_stats
from .preprocessing import PreprocessArtifacts, select_and_coerce_numeric, transform_sequences


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


def _get_available_features(df: pd.DataFrame, requested_features: list[str]) -> list[str]:
    available = [f for f in requested_features if f in df.columns]
    return available


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict using stacked PyCaret models + LSTM features")
    parser.add_argument("--season", default="2025-2026")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--seq-length", type=int, default=DEFAULT_SEQ_LENGTH)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)

    parser.add_argument("--ensemble-dir", default=None, help="Directory containing lstm_model.keras, preprocess.joblib, stack_h*.pkl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    _require_pycaret()
    from pycaret.regression import load_model, predict_model  # type: ignore

    repo_root = Path(args.repo_root)
    ensemble_dir = Path(args.ensemble_dir) if args.ensemble_dir else (repo_root / "artifacts" / "ensemble")

    outputs_dir = repo_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else (outputs_dir / "ensemble_projections.csv")

    # Load LSTM + preprocessing
    lstm = tf.keras.models.load_model(str(ensemble_dir / "lstm_model.keras"))
    prep = PreprocessArtifacts.load(str(ensemble_dir / "preprocess.joblib"))

    model_horizon = int(lstm.output_shape[-1])
    if args.horizon != model_horizon:
        print(f"Warning: --horizon={args.horizon} but LSTM outputs horizon={model_horizon}. Using horizon={model_horizon}.")
        args.horizon = model_horizon

    raw = load_premier_league_gameweek_stats(repo_root=repo_root, season=args.season, apply_feature_engineering=True)
    feature_columns = _get_available_features(raw, DEFAULT_FEATURE_COLUMNS)

    df = select_and_coerce_numeric(raw, prep.feature_columns, TARGET_COLUMN)

    last_gw = int(df["gw"].max())
    next_gws = list(range(last_gw + 1, last_gw + 1 + args.horizon))

    # Build last seq_length window per player for LSTM.
    rows: list[dict] = []
    X_list: list[np.ndarray] = []
    last_feature_rows: list[np.ndarray] = []

    for player_id, g in df.sort_values(["player_id", "gw"]).groupby("player_id", sort=False):
        g = g.sort_values("gw")
        if len(g) < args.seq_length:
            continue

        window = g.iloc[-args.seq_length:]
        X_window = window[prep.feature_columns].to_numpy(dtype=float)
        if X_window.shape != (args.seq_length, len(prep.feature_columns)):
            continue

        X_list.append(X_window)
        last_feature_rows.append(window.iloc[-1][prep.feature_columns].to_numpy(dtype=float))

        web_name = raw.loc[raw["player_id"] == player_id, "web_name"].dropna()
        name = str(web_name.iloc[-1]) if len(web_name) else ""
        rows.append({"player_id": int(player_id), "web_name": name})

    if not X_list:
        raise ValueError("No players had enough history to build sequences.")

    X = np.stack(X_list, axis=0)
    X = transform_sequences(prep.pipeline, X)

    lstm_preds = lstm.predict(X, verbose=0)

    # Build per-horizon stacked predictions.
    out = pd.DataFrame(rows)

    for i, gw in enumerate(next_gws, start=1):
        stack = load_model(str(ensemble_dir / f"stack_h{i}"))

        # Build a minimal tabular row for predict_model: last-timestep engineered features + lstm_pred_hi
        tab = pd.DataFrame(last_feature_rows, columns=prep.feature_columns)
        tab[f"lstm_pred_h{i}"] = lstm_preds[:, i - 1]

        pred_df = predict_model(stack, data=tab)
        out[f"GW{gw}_proj_points"] = pred_df["prediction_label"].to_numpy(dtype=float)

    out = out.sort_values(f"GW{next_gws[0]}_proj_points", ascending=False).reset_index(drop=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote ensemble projections: {output_path}")


if __name__ == "__main__":
    main()

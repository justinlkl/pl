from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import DEFAULT_FEATURE_COLUMNS, DEFAULT_HORIZON, DEFAULT_SEQ_LENGTH, TARGET_COLUMN
from .data_loading import load_premier_league_gameweek_stats
from .preprocessing import PreprocessArtifacts, select_and_coerce_numeric, transform_sequences


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FPL projection points table")
    parser.add_argument("--season", default="2025-2026")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--seq-length", type=int, default=DEFAULT_SEQ_LENGTH)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else (repo_root / "artifacts")
    outputs_dir = repo_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output) if args.output else (outputs_dir / "projections.csv")

    model = tf.keras.models.load_model(artifacts_dir / "model.keras")
    prep = PreprocessArtifacts.load(str(artifacts_dir / "preprocess.joblib"))

    raw = load_premier_league_gameweek_stats(repo_root=repo_root, season=args.season)

    # Keep names for table, but select numeric columns for modeling.
    df = select_and_coerce_numeric(raw, prep.feature_columns, TARGET_COLUMN)

    last_gw = int(df["gw"].max())
    next_gws = list(range(last_gw + 1, last_gw + 1 + args.horizon))

    # Build the last seq_length timesteps per player.
    rows: list[dict] = []
    X_list: list[np.ndarray] = []

    for player_id, g in df.sort_values(["player_id", "gw"]).groupby("player_id", sort=False):
        g = g.sort_values("gw")
        if len(g) < args.seq_length:
            continue

        window = g.iloc[-args.seq_length:]
        X_window = window[prep.feature_columns].to_numpy(dtype=float)
        if X_window.shape != (args.seq_length, len(prep.feature_columns)):
            continue

        X_list.append(X_window)

        # Try to get a stable display name from the raw dataframe.
        web_name = raw.loc[raw["player_id"] == player_id, "web_name"].dropna()
        name = str(web_name.iloc[-1]) if len(web_name) else ""

        rows.append({"player_id": int(player_id), "web_name": name})

    if not X_list:
        raise ValueError("No players had enough history to build sequences.")

    X = np.stack(X_list, axis=0)
    X = transform_sequences(prep.pipeline, X)

    preds = model.predict(X, verbose=0)
    # preds shape: (n_players, horizon)

    out = pd.DataFrame(rows)
    for i, gw in enumerate(next_gws):
        out[f"GW{gw}_proj_points"] = preds[:, i]

    out = out.sort_values(f"GW{next_gws[0]}_proj_points", ascending=False).reset_index(drop=True)

    out.to_csv(output_path, index=False)
    print(f"Wrote projections: {output_path}")


if __name__ == "__main__":
    main()

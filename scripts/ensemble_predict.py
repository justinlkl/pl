#!/usr/bin/env python3
"""Generate multi-week projections using the leakage-free weighted ensemble.

This uses the artifacts produced by scripts/train_ensemble_stacker.py in
artifacts/ensemble_stacker_trained/:
- lstm_model.keras
- base_xgb.joblib
- base_ridge.joblib
- ensemble_config.joblib (weights)
- feature_names.json

It performs iterative (recursive) forecasting for `--horizon` weeks by appending
predicted points back into the history and recomputing lag/rolling features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


def load_training_data(repo_root: Path, season: str) -> pd.DataFrame:
    """Load per-gameweek player stats and attach gw/player_id columns."""
    base = repo_root / "FPL-Core-Insights" / "data" / season / "By Gameweek"
    if not base.exists():
        raise FileNotFoundError(f"Gameweek folder not found: {base}")

    frames: list[pd.DataFrame] = []
    for gw_dir in sorted(base.iterdir()):
        if not gw_dir.is_dir() or not gw_dir.name.startswith("GW"):
            continue
        gw_num = int(gw_dir.name.replace("GW", ""))
        f = gw_dir / "player_gameweek_stats.csv"
        if not f.exists():
            continue
        gdf = pd.read_csv(f)
        gdf["gw"] = gw_num
        gdf = gdf.rename(columns={"id": "player_id", "event_points": "points"})
        frames.append(gdf)

    if not frames:
        raise FileNotFoundError("No player_gameweek_stats.csv files found")

    return pd.concat(frames, ignore_index=True)


def prepare_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag/rolling features using only past data (no leakage)."""
    df = df.sort_values(["player_id", "gw"]).reset_index(drop=True)

    forbidden_cols = ["total_points", "event_points", "form", "value_form", "proj_points"]
    df = df.drop(columns=[c for c in forbidden_cols if c in df.columns], errors="ignore")

    if "points" not in df.columns and "event_points" in df.columns:
        df = df.rename(columns={"event_points": "points"})

    # Ensure numeric
    df["points"] = pd.to_numeric(df.get("points"), errors="coerce").fillna(0.0)

    for lag in [1, 2, 3]:
        df[f"points_lag{lag}"] = df.groupby("player_id")["points"].shift(lag).fillna(0.0)

    for window in [3, 5, 10]:
        df[f"points_roll{window}"] = (
            df.groupby("player_id")["points"]
            .rolling(window, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(0, drop=True)
            .fillna(0.0)
        )

    if "expected_goals" in df.columns:
        df["xG_roll5"] = (
            pd.to_numeric(df["expected_goals"], errors="coerce")
            .groupby(df["player_id"])
            .rolling(5, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(0, drop=True)
            .fillna(0.0)
        )
    else:
        df["xG_roll5"] = 0.0

    if "expected_assists" in df.columns:
        df["xA_roll5"] = (
            pd.to_numeric(df["expected_assists"], errors="coerce")
            .groupby(df["player_id"])
            .rolling(5, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(0, drop=True)
            .fillna(0.0)
        )
    else:
        df["xA_roll5"] = 0.0

    if "minutes" in df.columns:
        df["minutes_roll3"] = (
            pd.to_numeric(df["minutes"], errors="coerce")
            .groupby(df["player_id"])
            .rolling(3, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(0, drop=True)
            .fillna(0.0)
        )
    else:
        df["minutes_roll3"] = 0.0

    return df


def predict_with_weighted_ensemble(
    lstm_model: tf.keras.Model,
    xgb_model,
    ridge_model,
    X_new: np.ndarray,
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> np.ndarray:
    """Production prediction using weighted ensemble."""
    w_lstm, w_xgb, w_ridge = weights

    lstm_pred = lstm_model.predict(X_new, verbose=0)
    lstm_pred = lstm_pred[:, 0] if lstm_pred.ndim > 1 else lstm_pred

    X_last = X_new[:, -1, :]
    xgb_pred = xgb_model.predict(X_last)
    ridge_pred = ridge_model.predict(X_last)

    return (w_lstm * lstm_pred) + (w_xgb * xgb_pred) + (w_ridge * ridge_pred)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--season", type=str, default="2025-2026")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--seq-length", type=int, default=5)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory containing ensemble_stacker_trained artifacts (defaults to repo_root/artifacts/ensemble_stacker_trained)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (defaults to outputs/projections_ensemble.csv)",
    )
    args = parser.parse_args()

    repo_root = args.repo_root
    model_dir = args.model_dir or (repo_root / "artifacts" / "ensemble_stacker_trained")
    output_path = args.output or (repo_root / "outputs" / "projections_ensemble.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lstm_path = model_dir / "lstm_model.keras"
    if not lstm_path.exists():
        raise FileNotFoundError(
            f"Missing {lstm_path}. Re-run scripts/train_ensemble_stacker.py after the latest patch so it saves the LSTM."
        )

    lstm = tf.keras.models.load_model(lstm_path, compile=False)
    xgb_model = joblib.load(model_dir / "base_xgb.joblib")
    ridge_model = joblib.load(model_dir / "base_ridge.joblib")

    weights = (0.5, 0.3, 0.2)
    cfg_path = model_dir / "ensemble_config.joblib"
    if cfg_path.exists():
        cfg = joblib.load(cfg_path)
        if isinstance(cfg, dict) and cfg.get("type") == "weighted":
            w = cfg.get("weights")
            if isinstance(w, (list, tuple)) and len(w) == 3:
                weights = (float(w[0]), float(w[1]), float(w[2]))

    feature_names_path = model_dir / "feature_names.json"
    if not feature_names_path.exists():
        raise FileNotFoundError(
            f"Missing {feature_names_path}. Re-run scripts/train_ensemble_stacker.py after the latest patch so it saves feature names."
        )
    feature_cols = json.loads(feature_names_path.read_text(encoding="utf-8"))

    raw = load_training_data(repo_root, args.season)
    raw["player_id"] = pd.to_numeric(raw["player_id"], errors="coerce")
    raw = raw.dropna(subset=["player_id"]).copy()
    raw["player_id"] = raw["player_id"].astype(int)

    # Keep names for output.
    name_map = {}
    if "web_name" in raw.columns:
        latest_names = raw.sort_values(["player_id", "gw"]).groupby("player_id", as_index=False).tail(1)[
            ["player_id", "web_name"]
        ]
        name_map = {int(r["player_id"]): str(r.get("web_name") or "") for _, r in latest_names.iterrows()}

    last_gw = int(pd.to_numeric(raw["gw"], errors="coerce").dropna().max())

    # Cache latest non-target columns used to fill future rows.
    fill_cols = [c for c in ["minutes", "expected_goals", "expected_assists"] if c in raw.columns]
    latest_fill = (
        raw.sort_values(["player_id", "gw"]).groupby("player_id", as_index=False).tail(1)[["player_id"] + fill_cols]
        if fill_cols
        else raw.sort_values(["player_id", "gw"]).groupby("player_id", as_index=False).tail(1)[["player_id"]]
    )
    latest_fill = latest_fill.set_index("player_id")

    preds_by_h: dict[int, dict[int, float]] = {h: {} for h in range(1, int(args.horizon) + 1)}

    df_aug = raw.copy()

    for h in range(1, int(args.horizon) + 1):
        df_feat = prepare_historical_features(df_aug)

        # Build windows ending at end_gw = last_gw + (h-1)
        end_gw = last_gw + (h - 1)

        X_list: list[np.ndarray] = []
        pids: list[int] = []

        for pid, g in df_feat[df_feat["gw"] <= end_gw].groupby("player_id", sort=False):
            g = g.sort_values("gw")
            if len(g) < int(args.seq_length):
                continue
            win = g.iloc[-int(args.seq_length) :]
            win_df = win.reindex(columns=feature_cols).fillna(0.0)
            X_list.append(win_df.to_numpy(dtype=float))
            pids.append(int(pid))

        if not X_list:
            break

        X_new = np.stack(X_list, axis=0)

        y_pred = predict_with_weighted_ensemble(lstm, xgb_model, ridge_model, X_new, weights=weights)
        y_pred = np.asarray(y_pred, dtype=float)

        for pid, pred in zip(pids, y_pred.tolist(), strict=False):
            preds_by_h[h][pid] = float(pred)

        # Append predicted next-gw row so next horizon can use it.
        next_gw = end_gw + 1
        add_rows = []
        for pid, pred in zip(pids, y_pred.tolist(), strict=False):
            row = {"player_id": int(pid), "gw": int(next_gw), "points": float(pred)}
            if fill_cols and int(pid) in latest_fill.index:
                for c in fill_cols:
                    row[c] = latest_fill.loc[int(pid)].get(c)
            add_rows.append(row)

        if add_rows:
            df_aug = pd.concat([df_aug, pd.DataFrame(add_rows)], ignore_index=True)

    # Build output.
    out_rows = []
    for pid in sorted(set().union(*[set(d.keys()) for d in preds_by_h.values()])):
        row = {"player_id": int(pid), "web_name": name_map.get(int(pid), ""), "end_gw": int(last_gw)}
        for h in range(1, int(args.horizon) + 1):
            row[f"ensemble_pred_h{h}"] = float(preds_by_h.get(h, {}).get(pid, 0.0))
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out):,} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

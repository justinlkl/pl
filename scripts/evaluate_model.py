#!/usr/bin/env python3
"""Evaluate projections against actual FPL gameweek results.

Usage: python scripts/evaluate_model.py --gw 23

Saves per-run metrics to `reports/evaluation_history.csv` and a summary
to `reports/evaluation_summary.csv`.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from src.fpl_projection.data_loading import load_premier_league_gameweek_stats


def detect_insights_season(root: Path) -> str:
    insights = root / "FPL-Core-Insights" / "data"
    if not insights.exists():
        raise FileNotFoundError("FPL-Core-Insights/data not found in workspace root")
    seasons = [p.name for p in insights.iterdir() if p.is_dir()]
    if not seasons:
        raise FileNotFoundError("No season folders found under FPL-Core-Insights/data")
    # Prefer the lexicographically last (newest) folder
    return sorted(seasons)[-1]


def safe_read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return pd.read_csv(p)


def compute_metrics(df: pd.DataFrame, pred_col: str, true_col: str = "event_points") -> dict:
    y_true = pd.to_numeric(df[true_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y_pred = pd.to_numeric(df[pred_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"mae": mae, "rmse": rmse, "n": int(len(y_true))}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate projections vs actuals")
    p.add_argument("--gw", type=int, required=True, help="Gameweek to evaluate (e.g. 23)")
    p.add_argument("--projections", type=Path, default=Path("outputs/projections_internal.csv"))
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--season", type=str, default=None, help="Insights season folder (auto-detected if omitted)")
    p.add_argument("--out-history", type=Path, default=Path("reports/evaluation_history.csv"))
    p.add_argument("--out-summary", type=Path, default=Path("reports/evaluation_summary.csv"))
    args = p.parse_args(argv)

    repo_root = args.repo_root.resolve()
    season = args.season or detect_insights_season(repo_root)

    proj_path: Path = args.projections
    print(f"Loading projections: {proj_path}")
    proj = safe_read_csv(proj_path)

    # Determine projection column for the requested GW
    gw_col = f"GW{args.gw}_proj_points"
    if gw_col not in proj.columns:
        # Fallback: if single `proj_points` is present, use that (assumed to be next-GW)
        if "proj_points" in proj.columns:
            print(f"Warning: {gw_col} not found, falling back to `proj_points`")
            gw_col = "proj_points"
        else:
            raise ValueError(f"Could not find projection column for GW {args.gw} in {proj_path}")

    # Load actuals from Insights gameweek files
    print(f"Loading actuals from FPL-Core-Insights season: {season}")
    actuals = load_premier_league_gameweek_stats(repo_root=repo_root, season=season, apply_feature_engineering=False)

    # The loader returns per-(player_id, gw) rows. We join on player_id and gw.
    if "player_id" not in proj.columns:
        # Try to align by web_name+club
        join_cols = [c for c in ["web_name", "club", "team", "team_code"] if c in proj.columns and c in actuals.columns]
        if not join_cols:
            raise ValueError("Projections file lacks `player_id` and no common name/team keys found for join")
        print(f"Joining projections to actuals on: {join_cols} + gw")
        left = proj.copy()
        left["gw"] = args.gw
        merged = left.merge(actuals, on=join_cols + ["gw"], how="inner", suffixes=("_proj", "_act"))
    else:
        left = proj.copy()
        left["gw"] = args.gw
        merged = left.merge(actuals, on=["player_id", "gw"], how="inner", suffixes=("_proj", "_act"))

    if merged.empty:
        raise ValueError("No matching rows after merging projections and actuals. Check player ids/names and season files.")

    # Determine true column name (actuals may have been suffixed during merge)
    true_col = "event_points"
    if true_col not in merged.columns:
        if f"{true_col}_act" in merged.columns:
            true_col = f"{true_col}_act"
        elif "total_points" in merged.columns:
            true_col = "total_points"
        elif "total_points_act" in merged.columns:
            true_col = "total_points_act"
        else:
            # pick any column that looks like points
            cand = [c for c in merged.columns if "point" in c]
            if cand:
                true_col = cand[0]
            else:
                raise ValueError("Could not locate a suitable ground-truth points column in merged actuals")

    # Compute overall metrics
    overall = compute_metrics(merged, pred_col=gw_col, true_col=true_col)

    # Metrics per position
    pos_metrics: dict[str, dict] = {}
    if "position" in merged.columns:
        for pos, g in merged.groupby("position"):
            pos_metrics[pos] = compute_metrics(g, pred_col=gw_col, true_col="event_points")

    # Save history row(s)
    out_history = args.out_history
    out_history.parent.mkdir(parents=True, exist_ok=True)

    run_time = datetime.utcnow().isoformat() + "Z"
    # Write a row per position plus overall
    rows = []
    rows.append(
        {
            "run_time": run_time,
            "season": season,
            "gw": int(args.gw),
            "scope": "overall",
            "mae": overall["mae"],
            "rmse": overall["rmse"],
            "n": overall["n"],
        }
    )
    for pos, m in pos_metrics.items():
        rows.append(
            {
                "run_time": run_time,
                "season": season,
                "gw": int(args.gw),
                "scope": pos,
                "mae": m["mae"],
                "rmse": m["rmse"],
                "n": m["n"],
            }
        )

    hist_df = pd.DataFrame(rows)
    if out_history.exists():
        prev = pd.read_csv(out_history)
        hist_df = pd.concat([prev, hist_df], ignore_index=True)
    hist_df.to_csv(out_history, index=False)
    print(f"Wrote evaluation history to: {out_history}")

    # Save summary file for quick consumption
    out_summary = args.out_summary
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary = {"run_time": run_time, "season": season, "gw": int(args.gw), "overall_mae": overall["mae"], "overall_rmse": overall["rmse"], "n": overall["n"], "per_position": pos_metrics}
    with open(out_summary, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote evaluation summary to: {out_summary}")

    # Print quick text summary
    print("Summary:")
    print(f" GW {args.gw} - MAE: {overall['mae']:.3f}, RMSE: {overall['rmse']:.3f}, n={overall['n']}")
    for pos, m in pos_metrics.items():
        print(f"  {pos}: MAE={m['mae']:.3f}, RMSE={m['rmse']:.3f}, n={m['n']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

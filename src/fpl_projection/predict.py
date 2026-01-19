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
    parser.add_argument(
        "--internal-output",
        default=None,
        help=(
            "Optional: write a second CSV that retains player_id for downstream joins (site/streamlit). "
            "If omitted, writes to outputs/projections_internal.csv."
        ),
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else (repo_root / "artifacts")
    outputs_dir = repo_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output) if args.output else (outputs_dir / "projections.csv")
    internal_output_path = (
        Path(args.internal_output) if args.internal_output else (outputs_dir / "projections_internal.csv")
    )

    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(artifacts_dir / "model.keras")
    prep = PreprocessArtifacts.load(str(artifacts_dir / "preprocess.joblib"))

    # Make horizon consistent with the loaded model.
    # This avoids runtime errors if the artifacts were trained with a different horizon.
    model_horizon = int(model.output_shape[-1])
    if args.horizon != model_horizon:
        print(
            f"Warning: --horizon={args.horizon} but loaded model outputs horizon={model_horizon}. "
            f"Using model horizon={model_horizon}."
        )
        args.horizon = model_horizon

    raw = load_premier_league_gameweek_stats(repo_root=repo_root, season=args.season)

    insights_root = repo_root / "FPL-Core-Insights" / "data" / args.season
    teams_path = insights_root / "teams.csv"
    playerstats_path = insights_root / "playerstats.csv"

    def _pos_to_abbrev(v: object) -> str:
        s = str(v or "").strip().lower()
        if s in {"goalkeeper", "gk"}:
            return "GK"
        if s in {"defender", "def"}:
            return "DEF"
        if s in {"midfielder", "mid"}:
            return "MID"
        if s in {"forward", "fwd", "striker"}:
            return "FWD"
        return str(v or "").strip()

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

        rows.append({"player_id": int(player_id)})

    if not X_list:
        raise ValueError("No players had enough history to build sequences.")

    X = np.stack(X_list, axis=0)
    X = transform_sequences(prep.pipeline, X)

    preds = model.predict(X, verbose=0)
    # preds shape: (n_players, horizon)

    out = pd.DataFrame(rows)

    # Attach player metadata (name/team/position) from the latest available row.
    meta_cols = [c for c in ["player_id", "web_name", "team_code", "position"] if c in raw.columns]
    meta = (
        raw.sort_values(["player_id", "gw"])
        .groupby("player_id", sort=False, as_index=False)
        .tail(1)[meta_cols]
        .copy()
    )
    if "position" in meta.columns:
        meta["position"] = meta["position"].apply(_pos_to_abbrev)
    out = out.merge(meta, on="player_id", how="left")

    # Attach club short/full names from teams.csv (team_code -> short_name/name)
    if teams_path.exists() and "team_code" in out.columns:
        try:
            teams = pd.read_csv(teams_path)
            teams = teams[[c for c in ["code", "name", "short_name"] if c in teams.columns]].copy()
            if "code" in teams.columns:
                teams["code"] = pd.to_numeric(teams["code"], errors="coerce").astype("Int64")
                out["team_code"] = pd.to_numeric(out["team_code"], errors="coerce").astype("Int64")
                out = out.merge(teams, left_on="team_code", right_on="code", how="left")
                out = out.drop(columns=[c for c in ["code"] if c in out.columns])
                if "short_name" in out.columns:
                    out = out.rename(columns={"short_name": "club"})
                if "name" in out.columns:
                    out = out.rename(columns={"name": "club_name"})
        except Exception:
            pass

    # Attach season-to-date stats from playerstats.csv (latest gw per player)
    if playerstats_path.exists():
        try:
            stats = pd.read_csv(playerstats_path)
            if "id" in stats.columns and "player_id" not in stats.columns:
                stats = stats.rename(columns={"id": "player_id"})
            if "player_id" in stats.columns and "gw" in stats.columns:
                stats["player_id"] = pd.to_numeric(stats["player_id"], errors="coerce").astype("Int64")
                stats["gw"] = pd.to_numeric(stats["gw"], errors="coerce").astype("Int64")
                stats = stats.dropna(subset=["player_id", "gw"]).copy()
                stats["player_id"] = stats["player_id"].astype(int)
                stats["gw"] = stats["gw"].astype(int)

                keep = [
                    "player_id",
                    "gw",
                    "now_cost",
                    "total_points",
                    "points_per_game",
                    "minutes",
                    "starts",
                    "goals_scored",
                    "assists",
                    "clean_sheets",
                    "expected_goals",
                    "expected_assists",
                    "expected_goal_involvements",
                    "bonus",
                    "bps",
                    "ict_index",
                ]
                keep = [c for c in keep if c in stats.columns]
                latest = (
                    stats.sort_values(["player_id", "gw"]).groupby("player_id", sort=False, as_index=False).tail(1)[
                        keep
                    ]
                )
                ren = {
                    "gw": "season_stats_gw",
                    "now_cost": "price",
                    "total_points": "season_total_points",
                    "points_per_game": "season_points_per_game",
                    "minutes": "season_minutes",
                    "starts": "season_starts",
                    "goals_scored": "season_goals_scored",
                    "assists": "season_assists",
                    "clean_sheets": "season_clean_sheets",
                    "expected_goals": "season_xg",
                    "expected_assists": "season_xa",
                    "expected_goal_involvements": "season_xgi",
                    "bonus": "season_bonus",
                    "bps": "season_bps",
                    "ict_index": "season_ict_index",
                }
                latest = latest.rename(columns={k: v for k, v in ren.items() if k in latest.columns})
                out = out.merge(latest, on="player_id", how="left")

                # Convert price to £m (FPL now_cost is tenths)
                if "price" in out.columns:
                    out["price"] = pd.to_numeric(out["price"], errors="coerce") / 10.0
        except Exception:
            pass
    for i, gw in enumerate(next_gws):
        out[f"GW{gw}_proj_points"] = preds[:, i]

    # Convenience: next-GW projected points
    if next_gws:
        out["proj_points"] = out[f"GW{next_gws[0]}_proj_points"]

    out = out.sort_values(f"GW{next_gws[0]}_proj_points", ascending=False).reset_index(drop=True)

    # Internal CSV keeps player_id for joins (site/streamlit)
    internal_output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(internal_output_path, index=False)

    # Public CSV: remove player_id and team_code by default, keep club + position + season stats
    public = out.copy()
    drop_cols = [c for c in ["player_id", "team_code"] if c in public.columns]
    if drop_cols:
        public = public.drop(columns=drop_cols)

    preferred_order = [
        "web_name",
        "club",
        "club_name",
        "position",
        "price",
        "season_stats_gw",
        "season_total_points",
        "season_points_per_game",
        "season_minutes",
        "season_starts",
        "season_goals_scored",
        "season_assists",
        "season_clean_sheets",
        "season_xg",
        "season_xa",
        "season_xgi",
        "season_bps",
        "season_bonus",
        "season_ict_index",
        "proj_points",
    ]
    proj_cols = [c for c in public.columns if c.startswith("GW") and c.endswith("_proj_points")]
    ordered = [c for c in preferred_order if c in public.columns] + [c for c in proj_cols if c in public.columns]
    ordered += [c for c in public.columns if c not in set(ordered)]
    public = public[ordered]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    public.to_csv(output_path, index=False)
    print(f"Wrote projections (public): {output_path}")
    print(f"Wrote projections (internal): {internal_output_path}")


if __name__ == "__main__":
    main()

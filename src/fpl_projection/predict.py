from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import DEFAULT_FEATURE_COLUMNS, DEFAULT_HORIZON, DEFAULT_SEQ_LENGTH, TARGET_COLUMN
from .data_loading import load_premier_league_gameweek_stats
from .preprocessing import PreprocessArtifacts, select_and_coerce_numeric, transform_sequences
from .role_modeling import build_feature_weight_vector, infer_role_from_window, list_roles


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FPL projection points table")
    parser.add_argument("--season", default="2025-2026")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--seq-length", type=int, default=DEFAULT_SEQ_LENGTH)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--use-role-models",
        action="store_true",
        help="Use per-role models under artifacts/models if present (otherwise use artifacts/model.keras).",
    )
    parser.add_argument(
        "--mid-split",
        action="store_true",
        help="If role models exist, split MID into MID_DM/MID_AM using a heuristic.",
    )
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

    role_models_dir = artifacts_dir / "models"
    use_role_models = bool(args.use_role_models) and role_models_dir.exists() and role_models_dir.is_dir()

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

    # Determine horizon from artifacts (role models or single model)
    if use_role_models:
        # Prefer the first role's horizon as canonical.
        roles = list_roles(mid_split=args.mid_split)
        model_horizon = None
        for r in roles:
            candidate = role_models_dir / r / "model.keras"
            if candidate.exists():
                m = tf.keras.models.load_model(candidate)
                model_horizon = int(m.output_shape[-1])
                break
        if model_horizon is None:
            raise FileNotFoundError(f"No role models found under: {role_models_dir}")
        if args.horizon != model_horizon:
            print(
                f"Warning: --horizon={args.horizon} but role models output horizon={model_horizon}. "
                f"Using model horizon={model_horizon}."
            )
            args.horizon = model_horizon
    else:
        model = tf.keras.models.load_model(artifacts_dir / "model.keras")
        prep = PreprocessArtifacts.load(str(artifacts_dir / "preprocess.joblib"))
        model_horizon = int(model.output_shape[-1])
        if args.horizon != model_horizon:
            print(
                f"Warning: --horizon={args.horizon} but loaded model outputs horizon={model_horizon}. "
                f"Using model horizon={model_horizon}."
            )
            args.horizon = model_horizon

    # Determine last available gameweek from raw (works for both role-model and single-model paths).
    if "gw" not in raw.columns:
        raise ValueError("Input data is missing required column: gw")
    last_gw = int(pd.to_numeric(raw["gw"], errors="coerce").dropna().max())

    # Keep names for table, but select numeric columns for modeling.
    # For role models, we'll select per-role based on each role's saved feature_columns.
    if not use_role_models:
        df = select_and_coerce_numeric(raw, prep.feature_columns, TARGET_COLUMN)
    next_gws = list(range(last_gw + 1, last_gw + 1 + args.horizon))

    if use_role_models:
        # Build player-level metadata early (for role assignment)
        meta_cols = [c for c in ["player_id", "web_name", "team_code", "position"] if c in raw.columns]
        meta = (
            raw.sort_values(["player_id", "gw"])
            .groupby("player_id", sort=False, as_index=False)
            .tail(1)[meta_cols]
            .copy()
        )
        if "position" in meta.columns:
            meta["position"] = meta["position"].apply(_pos_to_abbrev)

        # We'll fill predictions into this dict keyed by player_id.
        preds_by_pid: dict[int, np.ndarray] = {}

        roles = list_roles(mid_split=args.mid_split)
        for role in roles:
            model_path = role_models_dir / role / "model.keras"
            prep_path = role_models_dir / role / "preprocess.joblib"
            if not model_path.exists() or not prep_path.exists():
                continue

            model_r = tf.keras.models.load_model(model_path)
            prep_r = PreprocessArtifacts.load(str(prep_path))

            # Select modeling columns needed for this role model.
            df_r = select_and_coerce_numeric(raw, prep_r.feature_columns, TARGET_COLUMN)

            X_list: list[np.ndarray] = []
            pid_list: list[int] = []

            for player_id, g in df_r.sort_values(["player_id", "gw"]).groupby("player_id", sort=False):
                g = g.sort_values("gw")
                if len(g) < args.seq_length:
                    continue
                window = g.iloc[-args.seq_length:]

                # Assign role based on position + last-timestep engineered metrics.
                pos = None
                if "position" in meta.columns:
                    match = meta.loc[meta["player_id"] == int(player_id)]
                    if not match.empty:
                        pos = match.iloc[0].get("position")

                inferred = infer_role_from_window(pos, window, mid_split=args.mid_split)
                if inferred != role:
                    continue

                X_window = window[prep_r.feature_columns].to_numpy(dtype=float)
                if X_window.shape != (args.seq_length, len(prep_r.feature_columns)):
                    continue
                X_list.append(X_window)
                pid_list.append(int(player_id))

            if not X_list:
                continue

            X = np.stack(X_list, axis=0)
            X = transform_sequences(prep_r.pipeline, X)
            w = build_feature_weight_vector(prep_r.feature_columns, role)
            X = X * w
            preds = model_r.predict(X, verbose=0)

            for pid, p in zip(pid_list, preds, strict=False):
                preds_by_pid[int(pid)] = p

        if not preds_by_pid:
            raise ValueError("No players had enough history to build sequences for any role model.")

        out = pd.DataFrame({"player_id": list(preds_by_pid.keys())})
        preds_matrix = np.stack([preds_by_pid[int(pid)] for pid in out["player_id"].to_list()], axis=0)

        # Attach player metadata
        out = out.merge(meta, on="player_id", how="left")
        preds = preds_matrix
    else:
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
        X = X * build_feature_weight_vector(prep.feature_columns, "ALL")

        preds = model.predict(X, verbose=0)
        out = pd.DataFrame(rows)

    if not use_role_models:
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

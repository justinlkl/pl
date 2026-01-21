from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import DEFAULT_FEATURE_COLUMNS, DEFAULT_HORIZON, DEFAULT_SEQ_LENGTH, TARGET_COLUMN
from .data_loading import load_premier_league_gameweek_stats
from .preprocessing import PreprocessArtifacts, select_and_coerce_numeric, transform_sequences
from .role_modeling import (
    build_feature_weight_vector,
    infer_mid_subrole_from_window,
    infer_role_from_window,
    position_to_role,
    scale_projection_matrix,
)


def _build_opponent_lookup(
    *,
    repo_root: Path,
    season: str,
    gws: list[int],
) -> dict[int, dict[int, str]]:
    """Return {gw: {team_code: 'OPP(H);OPP2(A)'}} mapping from fixtures.csv."""

    insights_root = repo_root / "FPL-Core-Insights" / "data" / season
    teams_path = insights_root / "teams.csv"
    if not teams_path.exists():
        return {}

    try:
        teams = pd.read_csv(teams_path)
        teams = teams[[c for c in ["code", "short_name", "name"] if c in teams.columns]].copy()
        teams["code"] = pd.to_numeric(teams.get("code"), errors="coerce").astype("Int64")
        teams = teams.dropna(subset=["code"]).copy()
        teams["code"] = teams["code"].astype(int)
        if "short_name" in teams.columns:
            code_to_short = {int(r["code"]): str(r["short_name"]) for _, r in teams.iterrows()}
        elif "name" in teams.columns:
            code_to_short = {int(r["code"]): str(r["name"]) for _, r in teams.iterrows()}
        else:
            code_to_short = {int(r["code"]): str(int(r["code"])) for _, r in teams.iterrows()}
    except Exception:
        return {}

    out: dict[int, dict[int, str]] = {}
    pl_base = insights_root / "By Tournament" / "Premier League"

    for gw in gws:
        fx_path = pl_base / f"GW{int(gw)}" / "fixtures.csv"
        if not fx_path.exists():
            continue
        try:
            fx = pd.read_csv(fx_path)
        except Exception:
            continue
        if fx.empty or ("home_team" not in fx.columns) or ("away_team" not in fx.columns):
            continue

        home_team = pd.to_numeric(fx["home_team"], errors="coerce")
        away_team = pd.to_numeric(fx["away_team"], errors="coerce")
        valid = home_team.notna() & away_team.notna()
        if not bool(valid.any()):
            continue

        home_team = home_team[valid].astype(int)
        away_team = away_team[valid].astype(int)

        by_team: dict[int, list[str]] = {}
        for h, a in zip(home_team.tolist(), away_team.tolist(), strict=False):
            opp_a = f"{code_to_short.get(a, str(a))}(H)"
            opp_h = f"{code_to_short.get(h, str(h))}(A)"
            by_team.setdefault(int(h), []).append(opp_a)
            by_team.setdefault(int(a), []).append(opp_h)

        out[int(gw)] = {k: ";".join(v) for k, v in by_team.items()}

    return out


def _latest_per_player(df: pd.DataFrame, *, cols: list[str]) -> pd.DataFrame:
    keep = [c for c in cols if c in df.columns]
    if not keep:
        return pd.DataFrame(columns=cols)
    out = (
        df.sort_values(["player_id", "gw"])
        .groupby("player_id", sort=False, as_index=False)
        .tail(1)[keep]
        .copy()
    )
    return out


def _coalesce_suffix(out: pd.DataFrame, base_cols: list[str], suffix: str) -> pd.DataFrame:
    for c in base_cols:
        sc = f"{c}{suffix}"
        if sc in out.columns:
            if c in out.columns:
                out[c] = out[c].combine_first(out[sc])
            else:
                out = out.rename(columns={sc: c})
                continue
            out = out.drop(columns=[sc])
    return out


def _infer_is_mid_dm(latest_features: pd.DataFrame) -> dict[int, bool]:
    if latest_features.empty or "player_id" not in latest_features.columns:
        return {}

    out: dict[int, bool] = {}
    for _, row in latest_features.iterrows():
        pid = int(row["player_id"])
        window = pd.DataFrame([row])
        out[pid] = infer_mid_subrole_from_window(window) == "MID_DM"
    return out


def _require_pycaret() -> None:
    try:
        import importlib

        importlib.import_module("pycaret")
    except Exception as exc:  # pragma: no cover
        if sys.version_info >= (3, 12):
            raise SystemExit(
                "PyCaret does not support Python 3.12+ in this setup.\n"
                "Use a Python 3.11 environment for ensemble predictions.\n\n"
                "Conda example:\n"
                "  conda create -n fpl311 python=3.11 -y\n"
                "  conda run -n fpl311 python -m pip install -r requirements-ensemble.txt\n"
                "  conda run -n fpl311 python -m pip install -e .\n\n"
                f"Original import error: {exc}"
            )
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
    parser.add_argument(
        "--start-gw",
        type=int,
        default=None,
        help="Override the first projected gameweek label (e.g., 23).",
    )
    parser.add_argument(
        "--mid-split",
        action="store_true",
        help="Use a heuristic to split MID into MID_DM vs MID_AM for role weights/masking.",
    )
    parser.add_argument(
        "--no-role-scaling",
        action="store_true",
        help="Disable post-model role-based projection scaling multipliers.",
    )

    parser.add_argument(
        "--lstm-only",
        action="store_true",
        help="Skip PyCaret stacked models and output raw LSTM horizon predictions.",
    )

    parser.add_argument(
        "--backend",
        choices=["auto", "sklearn", "pycaret"],
        default="auto",
        help=(
            "Stacking backend. auto prefers sklearn if stack_h*.joblib exists, otherwise pycaret. "
            "pycaret requires Python < 3.12 and extra dependencies."
        ),
    )

    parser.add_argument(
        "--lstm-pred-scale",
        type=float,
        default=1.5,
        help="Scale lstm_pred_h* features before feeding the stacker (must match training).",
    )

    parser.add_argument("--ensemble-dir", default=None, help="Directory containing lstm_model.keras, preprocess.joblib, stack_h*.pkl")
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
    ensemble_dir = Path(args.ensemble_dir) if args.ensemble_dir else (repo_root / "artifacts" / "ensemble")

    backend = str(args.backend).lower()
    if backend == "auto":
        backend = "sklearn" if (ensemble_dir / "stack_h1.joblib").exists() else "pycaret"

    if not bool(args.lstm_only) and backend == "pycaret":
        _require_pycaret()
        from pycaret.regression import load_model, predict_model  # type: ignore
    elif not bool(args.lstm_only) and backend == "sklearn":
        import joblib

    outputs_dir = repo_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output) if args.output else (outputs_dir / "projections.csv")
    internal_output_path = (
        Path(args.internal_output) if args.internal_output else (outputs_dir / "projections_internal.csv")
    )

    # If the user passes a custom output path (often relative like "outputs/foo.csv"),
    # ensure its parent directory exists.
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load LSTM + preprocessing
    lstm = tf.keras.models.load_model(str(ensemble_dir / "lstm_model.keras"), compile=False)
    prep = PreprocessArtifacts.load(str(ensemble_dir / "preprocess.joblib"))

    model_horizon = int(lstm.output_shape[-1])
    if args.horizon != model_horizon:
        print(f"Warning: --horizon={args.horizon} but LSTM outputs horizon={model_horizon}. Using horizon={model_horizon}.")
        args.horizon = model_horizon

    raw = load_premier_league_gameweek_stats(repo_root=repo_root, season=args.season, apply_feature_engineering=True)
    feature_columns = _get_available_features(raw, DEFAULT_FEATURE_COLUMNS)

    df = select_and_coerce_numeric(raw, prep.feature_columns, TARGET_COLUMN)

    last_gw = int(df["gw"].max())
    if args.start_gw is not None:
        start_gw = int(args.start_gw)
    else:
        start_gw = int(last_gw + 1)
    next_gws = list(range(start_gw, start_gw + int(args.horizon)))

    # Build last seq_length window per player for LSTM.
    rows: list[dict] = []
    X_list: list[np.ndarray] = []
    last_feature_rows: list[np.ndarray] = []
    roles: list[str] = []

    pid_to_pos: dict[int, object] = {}
    if "player_id" in raw.columns and "position" in raw.columns:
        meta_pos = (
            raw.sort_values(["player_id", "gw"]).groupby("player_id", sort=False, as_index=False).tail(1)[
                ["player_id", "position"]
            ]
        )
        pid_to_pos = {int(r["player_id"]): r.get("position") for _, r in meta_pos.iterrows()}

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

        pos = pid_to_pos.get(int(player_id))
        role = infer_role_from_window(pos, window[prep.feature_columns], mid_split=bool(args.mid_split))
        if role is None:
            role = position_to_role(pos)
        roles.append(str(role))

    if not X_list:
        raise ValueError("No players had enough history to build sequences.")

    X = np.stack(X_list, axis=0)
    X = transform_sequences(prep.pipeline, X)

    # Apply per-sample role weights
    uniq = sorted(set(roles))
    role_to_w = {r: build_feature_weight_vector(prep.feature_columns, r) for r in uniq}
    W = np.stack([role_to_w.get(r, np.ones(len(prep.feature_columns))) for r in roles], axis=0)
    X = X * W[:, None, :]

    lstm_preds = lstm.predict(X, verbose=0)

    out = pd.DataFrame(rows)

    def _have_all_sklearn() -> bool:
        return all((ensemble_dir / f"stack_h{i}.joblib").exists() for i in range(1, args.horizon + 1))

    def _have_all_pycaret() -> bool:
        return all(
            (ensemble_dir / f"stack_h{i}.pkl").exists() or (ensemble_dir / f"stack_h{i}").exists()
            for i in range(1, args.horizon + 1)
        )

    have_all_stacks = _have_all_sklearn() if backend == "sklearn" else _have_all_pycaret()
    if args.lstm_only or (not have_all_stacks):
        if (not args.lstm_only) and (not have_all_stacks):
            print("Warning: stacked models not found; falling back to LSTM-only projections.")
        for i, gw in enumerate(next_gws, start=1):
            out[f"GW{gw}_proj_points"] = lstm_preds[:, i - 1]
    else:
        # Build per-horizon stacked predictions.
        for i, gw in enumerate(next_gws, start=1):
            # Build a minimal tabular row: last-timestep engineered features + lstm_pred_hi
            tab = pd.DataFrame(last_feature_rows, columns=prep.feature_columns)
            tab[f"lstm_pred_h{i}"] = np.asarray(lstm_preds[:, i - 1], dtype=float) * float(args.lstm_pred_scale)

            if backend == "pycaret":
                stack = load_model(str(ensemble_dir / f"stack_h{i}"))
                pred_df = predict_model(stack, data=tab)
                out[f"GW{gw}_proj_points"] = pred_df["prediction_label"].to_numpy(dtype=float)
            else:
                model = joblib.load(ensemble_dir / f"stack_h{i}.joblib")
                feature_cols = [*prep.feature_columns, f"lstm_pred_h{i}"]
                Xte = tab[feature_cols].to_numpy(dtype=float)
                out[f"GW{gw}_proj_points"] = np.asarray(model.predict(Xte), dtype=float)

    # Enrich output with teams + playerstats snapshot fields to match projections.csv.
    insights_root = repo_root / "FPL-Core-Insights" / "data" / args.season
    teams_path = insights_root / "teams.csv"
    playerstats_path = insights_root / "playerstats.csv"

    # Attach latest position/team_code from raw for role masking + team join.
    meta_cols = [c for c in ["player_id", "team_code", "position"] if c in raw.columns]
    if meta_cols:
        meta = (
            raw.sort_values(["player_id", "gw"])  # type: ignore[arg-type]
            .groupby("player_id", sort=False, as_index=False)
            .tail(1)[meta_cols]
            .copy()
        )
        out = out.merge(meta, on="player_id", how="left")

    # ---- Role-adjusted global scaling (post-model calibration) ----
    latest_features = _latest_per_player(raw, cols=[
        "player_id",
        "tackles_per_90",
        "cbi_per_90",
        "defcon_actions_per_90",
        "expected_goal_involvements_per_90",
        "threat",
        "creativity",
    ])
    is_mid_dm = _infer_is_mid_dm(latest_features)

    if "position" in out.columns and "player_id" in out.columns:
        pos = out["position"].astype(str)
        role_scale = pos.copy()
        mid_mask = pos.eq("MID")
        role_scale.loc[mid_mask] = np.where(
            out.loc[mid_mask, "player_id"].map(is_mid_dm).fillna(False).to_numpy(),
            "MID_DM",
            "MID_AM",
        )
        out["role"] = role_scale
    else:
        out["role"] = ""

    if not args.no_role_scaling:
        gw_cols = [c for c in out.columns if c.startswith("GW") and c.endswith("_proj_points")]
        if gw_cols:
            scaled = scale_projection_matrix(out[gw_cols].to_numpy(dtype=float), out["role"].to_numpy(dtype=object))
            out.loc[:, gw_cols] = scaled

    # team_code -> club short_name
    if teams_path.exists() and "team_code" in out.columns:
        try:
            teams = pd.read_csv(teams_path)
            teams = teams[[c for c in ["code", "short_name", "name"] if c in teams.columns]].copy()
            teams["code"] = pd.to_numeric(teams.get("code"), errors="coerce").astype("Int64")
            out["team_code"] = pd.to_numeric(out.get("team_code"), errors="coerce").astype("Int64")
            out = out.merge(teams, left_on="team_code", right_on="code", how="left")
            out = out.drop(columns=[c for c in ["code"] if c in out.columns])
            if "short_name" in out.columns:
                out = out.rename(columns={"short_name": "team"})
        except Exception:
            pass

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

                requested = [
                    "player_id",
                    "gw",
                    "first_name",
                    "second_name",
                    "web_name",
                    "chance_of_playing_this_round",
                    "chance_of_playing_next_round",
                    "news",
                    "now_cost",
                    "selected_by_percent",
                    "value_form",
                    "value_season",
                    "total_points",
                    "event_points",
                    "points_per_game",
                    "form",
                    "expected_goals_per_90",
                    "expected_assists_per_90",
                    "expected_goal_involvements_per_90",
                    "expected_goals_conceded_per_90",
                    "influence",
                    "creativity",
                    "threat",
                    "ict_index",
                    "minutes",
                    "goals_scored",
                    "assists",
                    "clean_sheets",
                    "goals_conceded",
                    "starts",
                    "defensive_contribution_per_90",
                    "tackles",
                    "clearances_blocks_interceptions",
                    "recoveries",
                ]
                latest_stats = _latest_per_player(stats, cols=requested)
                out = out.merge(latest_stats, on="player_id", how="left", suffixes=("", "_ps"))
                out = _coalesce_suffix(out, requested, "_ps")
                if "gw" in out.columns and "season_stats_gw" not in out.columns:
                    out = out.rename(columns={"gw": "season_stats_gw"})
        except Exception:
            pass

    # Role-aware masking
    if "position" in out.columns:
        pos = out["position"].astype(str)
        is_def_gk = pos.isin(["DEF", "GK"])
        mid_dm_mask = pos.eq("MID") & out["player_id"].map(is_mid_dm).fillna(False)

        if "expected_goals_conceded_per_90" in out.columns:
            out.loc[~is_def_gk, "expected_goals_conceded_per_90"] = np.nan
        for c in ["defensive_contribution_per_90", "tackles", "clearances_blocks_interceptions", "recoveries"]:
            if c in out.columns:
                out.loc[~(is_def_gk | mid_dm_mask), c] = np.nan

    # Add opponent labels for each projected GW (best-effort).
    if "team_code" in out.columns and next_gws:
        try:
            opp_lookup = _build_opponent_lookup(repo_root=repo_root, season=args.season, gws=next_gws)
            for gw in next_gws:
                m = opp_lookup.get(int(gw), {})
                out[f"GW{gw}_opp"] = out["team_code"].map(lambda x: m.get(int(x)) if pd.notna(x) else None)
        except Exception:
            pass

    # Sum of projected points across the upcoming 5 fixtures (from start_gw).
    gw_cols = [c for c in out.columns if c.startswith("GW") and c.endswith("_proj_points")]
    if gw_cols:
        # Ensure deterministic ordering by GW number.
        def _gw_num(c: str) -> int:
            try:
                return int(str(c).split("GW", 1)[1].split("_", 1)[0])
            except Exception:
                return 10**9

        gw_cols = sorted(gw_cols, key=_gw_num)
        first_five = gw_cols[:5]
        first_six = gw_cols[:6]
        out["proj_points_next_5"] = (
            out[first_five]
            .apply(lambda s: pd.to_numeric(s, errors="coerce"))
            .fillna(0.0)
            .sum(axis=1)
        )
        out["proj_points_next_6"] = (
            out[first_six]
            .apply(lambda s: pd.to_numeric(s, errors="coerce"))
            .fillna(0.0)
            .sum(axis=1)
        )
    else:
        out["proj_points_next_5"] = 0.0
        out["proj_points_next_6"] = 0.0

    # Sort after scaling so ranking reflects calibrated outputs.
    # Default: rank by the multi-GW total (next_6), not just the next GW.
    if "proj_points_next_6" in out.columns:
        out = out.sort_values("proj_points_next_6", ascending=False).reset_index(drop=True)
    elif "proj_points_next_5" in out.columns:
        out = out.sort_values("proj_points_next_5", ascending=False).reset_index(drop=True)
    elif next_gws:
        out = out.sort_values(f"GW{next_gws[0]}_proj_points", ascending=False).reset_index(drop=True)

    # Put requested fields first (then projections)
    preferred_order = [
        "first_name",
        "second_name",
        "web_name",
        "position",
        "team",
        "proj_points_next_5",
        "proj_points_next_6",
        "chance_of_playing_this_round",
        "chance_of_playing_next_round",
        "news",
        "now_cost",
        "selected_by_percent",
        "value_form",
        "value_season",
        "total_points",
        "event_points",
        "points_per_game",
        "form",
        "expected_goals_per_90",
        "expected_assists_per_90",
        "expected_goal_involvements_per_90",
        "expected_goals_conceded_per_90",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "starts",
        "defensive_contribution_per_90",
        "tackles",
        "clearances_blocks_interceptions",
        "recoveries",
    ]
    proj_cols = [c for c in out.columns if c.startswith("GW") and c.endswith("_proj_points")]
    opp_cols = [c for c in out.columns if c.startswith("GW") and c.endswith("_opp")]
    ordered = [c for c in preferred_order if c in out.columns] + [c for c in proj_cols if c in out.columns]
    ordered += [c for c in opp_cols if c in out.columns and c not in set(ordered)]
    ordered += [c for c in out.columns if c not in set(ordered)]
    out = out[ordered]

    # Internal CSV keeps player_id for joins (site/streamlit)
    internal_output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(internal_output_path, index=False)

    # Public CSV: remove player_id and team_code by default.
    public = out.copy()
    drop_cols = [c for c in ["player_id", "team_code", "role"] if c in public.columns]
    if drop_cols:
        public = public.drop(columns=drop_cols)

    public.to_csv(output_path, index=False)
    print(f"Wrote projections (public): {output_path}")
    print(f"Wrote projections (internal): {internal_output_path}")


if __name__ == "__main__":
    main()

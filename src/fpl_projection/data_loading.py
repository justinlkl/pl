from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .feature_engineering import engineer_all_features
from .insights_schema import (
    DROP_COLUMNS,
    ESSENTIAL_MATCH_STATS,
    ESSENTIAL_PLAYER_STATS,
    ESSENTIAL_TEAM_STATS,
    OPTIONAL_STATS,
    select_insights_columns,
)


def _build_fixture_difficulty_table(*, base: Path, teams_path: Path) -> pd.DataFrame:
    """Build a per-(gw, team_code) fixture difficulty table.

    We derive opponent + home/away from each GW's fixtures.csv and map opponent
    strength from teams.csv. This is best-effort and returns an empty dataframe
    if fixture inputs are missing.
    """

    fixture_files = sorted(
        base.glob("GW*/fixtures.csv"), key=lambda p: _extract_gw_from_path(p) or 0
    )
    if not fixture_files or not teams_path.exists():
        return pd.DataFrame()

    teams = pd.read_csv(teams_path)
    if "code" not in teams.columns:
        return pd.DataFrame()
    teams = teams.copy()
    teams["code"] = pd.to_numeric(teams["code"], errors="coerce")
    teams = teams.dropna(subset=["code"]).copy()
    teams["code"] = teams["code"].astype(int)

    opp_strength = None
    if "strength" in teams.columns:
        opp_strength = teams[["code", "strength"]].rename(columns={"strength": "opp_strength"})

    opp_elo = None
    if "elo" in teams.columns:
        opp_elo = teams[["code", "elo"]].rename(columns={"elo": "opp_elo"})

    frames: list[pd.DataFrame] = []
    for file_path in fixture_files:
        fx = pd.read_csv(file_path)
        if fx.empty:
            continue

        # Prefer explicit gameweek column, otherwise infer from folder.
        if "gameweek" in fx.columns:
            fx["gw"] = pd.to_numeric(fx["gameweek"], errors="coerce")
        elif "gw" in fx.columns:
            fx["gw"] = pd.to_numeric(fx["gw"], errors="coerce")
        else:
            gw = _extract_gw_from_path(file_path)
            fx["gw"] = gw

        if "home_team" not in fx.columns or "away_team" not in fx.columns:
            continue

        keep = fx[["gw", "home_team", "away_team"]].copy()
        keep["gw"] = pd.to_numeric(keep["gw"], errors="coerce")
        keep["home_team"] = pd.to_numeric(keep["home_team"], errors="coerce")
        keep["away_team"] = pd.to_numeric(keep["away_team"], errors="coerce")
        keep = keep.dropna(subset=["gw", "home_team", "away_team"]).copy()
        if keep.empty:
            continue
        keep["gw"] = keep["gw"].astype(int)
        keep["home_team"] = keep["home_team"].astype(int)
        keep["away_team"] = keep["away_team"].astype(int)

        home = keep.rename(columns={"home_team": "team_code", "away_team": "opp_team_code"})
        home["fixture_is_home"] = 1

        away = keep.rename(columns={"away_team": "team_code", "home_team": "opp_team_code"})
        away["fixture_is_home"] = 0

        frames.extend([home, away])

    if not frames:
        return pd.DataFrame()

    sched = pd.concat(frames, ignore_index=True)

    # Map opponent strength signals.
    if opp_strength is not None:
        sched = sched.merge(opp_strength, left_on="opp_team_code", right_on="code", how="left")
        sched = sched.drop(columns=["code"])
    else:
        sched["opp_strength"] = pd.NA

    if opp_elo is not None:
        sched = sched.merge(opp_elo, left_on="opp_team_code", right_on="code", how="left")
        sched = sched.drop(columns=["code"])
    else:
        sched["opp_elo"] = pd.NA

    # Simple difficulty score: stronger opponent + small away penalty.
    sched["fixture_difficulty"] = pd.to_numeric(sched["opp_strength"], errors="coerce").astype(float)
    sched.loc[sched["fixture_is_home"] == 0, "fixture_difficulty"] = (
        sched.loc[sched["fixture_is_home"] == 0, "fixture_difficulty"] + 0.25
    )

    # If a team has multiple fixtures in a GW (double GW), average the difficulty.
    agg = sched.groupby(["gw", "team_code"], as_index=False).agg(
        {
            "fixture_is_home": "mean",
            "opp_strength": "mean",
            "opp_elo": "mean",
            "fixture_difficulty": "mean",
        }
    )
    return agg


@dataclass(frozen=True)
class DataPaths:
    repo_root: Path

    @property
    def insights_data_root(self) -> Path:
        return self.repo_root / "FPL-Core-Insights" / "data"


def load_insights_playerstats(
    *, repo_root: Path, season: str, include_optional: bool = False
) -> pd.DataFrame:
    """Load and filter Insights `playerstats.csv`.

    Keeps the columns from `ESSENTIAL_PLAYER_STATS` (+ `OPTIONAL_STATS` if requested)
    while excluding `DROP_COLUMNS`.

    Returns a normalized dataframe with `player_id` and `gw` as ints when available.
    """

    paths = DataPaths(repo_root=repo_root)
    path = paths.insights_data_root / season / "playerstats.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Insights playerstats.csv at: {path}")

    df = pd.read_csv(path)
    if df.empty:
        return df

    keep = list(ESSENTIAL_PLAYER_STATS)
    if include_optional:
        keep += list(OPTIONAL_STATS)

    df = select_insights_columns(
        df,
        keep=keep,
        always_keep=("id", "gw"),
        drop=DROP_COLUMNS,
        context=f"playerstats.csv ({season})",
    )

    # Normalize IDs
    if "id" in df.columns and "player_id" not in df.columns:
        df = df.rename(columns={"id": "player_id"})
    if "player_id" in df.columns:
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    if "gw" in df.columns:
        df["gw"] = pd.to_numeric(df["gw"], errors="coerce").astype("Int64")

    if "player_id" in df.columns and "gw" in df.columns:
        df = df.dropna(subset=["player_id", "gw"]).copy()
        df["player_id"] = df["player_id"].astype(int)
        df["gw"] = df["gw"].astype(int)

    return df


def load_insights_teams(*, repo_root: Path, season: str) -> pd.DataFrame:
    """Load and filter Insights `teams.csv` using `ESSENTIAL_TEAM_STATS`."""

    paths = DataPaths(repo_root=repo_root)
    path = paths.insights_data_root / season / "teams.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Insights teams.csv at: {path}")

    df = pd.read_csv(path)
    if df.empty:
        return df

    df = select_insights_columns(
        df,
        keep=ESSENTIAL_TEAM_STATS,
        always_keep=("id", "code"),
        drop=DROP_COLUMNS,
        context=f"teams.csv ({season})",
    )
    return df


def load_insights_player_match_stats(
    *, repo_root: Path, season: str, include_optional: bool = False
) -> pd.DataFrame:
    """Load and filter Insights match-level player stats, if present.

    The exact file path varies by season/export. This loader searches a few common
    locations under `FPL-Core-Insights/data/<season>/`.
    """

    paths = DataPaths(repo_root=repo_root)
    base = paths.insights_data_root / season

    candidates = [
        base / "player_match_stats.csv",
        base / "player_match_stats" / "player_match_stats.csv",
        base / "playermatchstats" / "player_match_stats.csv",
        base / "playermatchstats" / "player_match_stats" / "player_match_stats.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "Could not find player_match_stats.csv under Insights season folder. "
            f"Looked in: {', '.join(str(p) for p in candidates)}"
        )

    df = pd.read_csv(path)
    if df.empty:
        return df

    keep = list(ESSENTIAL_MATCH_STATS)
    if include_optional:
        keep += list(OPTIONAL_STATS)

    df = select_insights_columns(
        df,
        keep=keep,
        always_keep=("player_id", "match_id"),
        drop=DROP_COLUMNS,
        context=f"player_match_stats.csv ({season})",
    )
    return df


def _extract_gw_from_path(path: Path) -> int | None:
    # Matches .../GW12/... or ...\\GW12\\...
    match = re.search(r"(?:^|[\\/])GW(\d+)(?:[\\/]|$)", str(path))
    if not match:
        return None
    return int(match.group(1))


def load_premier_league_gameweek_stats(
    *, 
    repo_root: Path, 
    season: str, 
    apply_feature_engineering: bool = True,
    previous_season: str | None = None,
    handle_new_players: bool = True,
) -> pd.DataFrame:
    """Load Premier League player_gameweek_stats.csv across all GW folders.

    Expected layout (from FPL-Core-Insights):
    data/<season>/By Tournament/Premier League/GW*/player_gameweek_stats.csv
    
    Args:
        repo_root: Root directory of the repository
        season: Season identifier (e.g., "2025-2026")
        apply_feature_engineering: If True, apply all feature engineering transformations
        previous_season: Previous season identifier (e.g., "2024-2025") for handling new players
        handle_new_players: If True and previous_season provided, fill missing features for new players
        
    Returns:
        DataFrame with loaded data and optionally engineered features
    """

    paths = DataPaths(repo_root=repo_root)
    base = paths.insights_data_root / season / "By Tournament" / "Premier League"

    files = sorted(base.glob("GW*/player_gameweek_stats.csv"), key=lambda p: _extract_gw_from_path(p) or 0)
    if not files:
        raise FileNotFoundError(
            f"No gameweek files found under: {base}. "
            "Verify the season folder and that the repo was cloned." 
        )

    frames: list[pd.DataFrame] = []
    for file_path in files:
        df = pd.read_csv(file_path)
        if df.empty:
            continue
        if "gw" not in df.columns:
            gw = _extract_gw_from_path(file_path)
            if gw is None:
                raise ValueError(f"Could not infer gw from path: {file_path}")
            df["gw"] = gw
        frames.append(df)

    if not frames:
        raise ValueError(f"All discovered gameweek files were empty under: {base}")

    all_df = pd.concat(frames, ignore_index=True)

    # Normalize key columns.
    if "id" not in all_df.columns:
        raise ValueError("Expected an 'id' column for player id.")

    all_df = all_df.rename(columns={"id": "player_id"})
    all_df["player_id"] = pd.to_numeric(all_df["player_id"], errors="coerce").astype("Int64")
    all_df["gw"] = pd.to_numeric(all_df["gw"], errors="coerce").astype("Int64")

    # Keep only valid rows.
    all_df = all_df.dropna(subset=["player_id", "gw"]).copy()
    all_df["player_id"] = all_df["player_id"].astype(int)
    all_df["gw"] = all_df["gw"].astype(int)

    # Attach season-level player metadata (position, team_code, etc.)
    # This lives at: data/<season>/players.csv
    try:
        players_path = paths.insights_data_root / season / "players.csv"
        if players_path.exists():
            players = pd.read_csv(players_path)
            if "player_id" in players.columns:
                players["player_id"] = pd.to_numeric(players["player_id"], errors="coerce").astype("Int64")
                players = players.dropna(subset=["player_id"]).copy()
                players["player_id"] = players["player_id"].astype(int)
                # Avoid bringing duplicated name columns over gameweek stats.
                keep_cols = [c for c in players.columns if c not in {"first_name", "second_name", "web_name"}]
                players = players[keep_cols].drop_duplicates(subset=["player_id"])
                all_df = all_df.merge(players, on="player_id", how="left")
    except Exception:
        # Metadata enrich is best-effort; core stats can still load without it.
        pass

    # Attach team strength features from teams.csv (crucial for modeling)
    try:
        if "team_code" in all_df.columns or "team" in all_df.columns:
            teams_path = paths.insights_data_root / season / "teams.csv"
            if teams_path.exists():
                teams = pd.read_csv(teams_path)
                team_key = "code" if "code" in teams.columns else "team_code"
                
                if team_key in teams.columns:
                    teams[team_key] = pd.to_numeric(teams[team_key], errors="coerce")
                    teams = teams.dropna(subset=[team_key]).copy()
                    teams[team_key] = teams[team_key].astype(int)
                    
                    # Select team strength columns
                    strength_cols = [team_key]
                    for col in [
                        "strength_overall_home", "strength_overall_away",
                        "strength_attack_home", "strength_attack_away",
                        "strength_defence_home", "strength_defence_away",
                        "elo",
                    ]:
                        if col in teams.columns:
                            strength_cols.append(col)
                    
                    teams_subset = teams[strength_cols].drop_duplicates(subset=[team_key])
                    
                    # Merge on team_code (player's team)
                    merge_key = "team_code" if "team_code" in all_df.columns else "team"
                    all_df = all_df.merge(
                        teams_subset,
                        left_on=merge_key,
                        right_on=team_key,
                        how="left",
                        suffixes=("", "_team"),
                    )
                    
                    # Clean up duplicate key column if created
                    if team_key != merge_key and team_key in all_df.columns:
                        all_df = all_df.drop(columns=[team_key])
    except Exception as exc:
        print(f"Warning: failed to merge team strength features: {exc}")

    # Attach per-GW player availability/market signals from playerstats.csv (best-effort).
    # We intentionally avoid leaky columns (e.g., ep_next/ep_this) and keep this merge
    # small to reduce mixed-type merge issues.
    try:
        stats_path = paths.insights_data_root / season / "playerstats.csv"
        if stats_path.exists():
            stats = pd.read_csv(stats_path)
            if "id" in stats.columns and "player_id" not in stats.columns:
                stats = stats.rename(columns={"id": "player_id"})

            if "player_id" in stats.columns and "gw" in stats.columns:
                stats["player_id"] = pd.to_numeric(stats["player_id"], errors="coerce").astype("Int64")
                stats["gw"] = pd.to_numeric(stats["gw"], errors="coerce").astype("Int64")
                stats = stats.dropna(subset=["player_id", "gw"]).copy()
                stats["player_id"] = stats["player_id"].astype(int)
                stats["gw"] = stats["gw"] .astype(int)

                keep_cols = [
                    # Identity
                    "player_id",
                    "gw",

                    # Availability/market
                    "chance_of_playing_next_round",
                    "chance_of_playing_this_round",
                    "selected_by_percent",

                    # Value + form
                    "now_cost",
                    "value_season",
                    "form",

                    # Set pieces (high-signal)
                    "penalties_order",

                    # Defensive (new rules)
                    "defensive_contribution",
                    "defensive_contribution_per_90",
                    "tackles",
                    "clearances_blocks_interceptions",
                    "recoveries",
                ]

                # Allow the UI/other consumers to opt into optional fields later; for training
                # keep this lean and drop known bad/leaky columns.
                stats = select_insights_columns(
                    stats,
                    keep=keep_cols,
                    always_keep=("player_id", "gw"),
                    drop=DROP_COLUMNS,
                    context=f"playerstats.csv ({season})",
                )
                stats = stats.drop_duplicates(subset=["player_id", "gw"], keep="last")

                all_df = all_df.merge(stats, on=["player_id", "gw"], how="left")

                # If the base dataset already had these columns, pandas will suffix them.
                # Coalesce back into canonical names so modeling code can depend on them.
                def _coalesce(base: str) -> None:
                    x = f"{base}_x"
                    y = f"{base}_y"
                    if base in all_df.columns:
                        return
                    if x in all_df.columns and y in all_df.columns:
                        all_df[base] = all_df[x].combine_first(all_df[y])
                        all_df.drop(columns=[x, y], inplace=True)
                        return
                    if x in all_df.columns:
                        all_df.rename(columns={x: base}, inplace=True)
                        return
                    if y in all_df.columns:
                        all_df.rename(columns={y: base}, inplace=True)

                for c in (
                    "chance_of_playing_next_round",
                    "chance_of_playing_this_round",
                    "selected_by_percent",
                    "now_cost",
                    "value_season",
                    "form",
                    "penalties_order",
                    "defensive_contribution",
                    "defensive_contribution_per_90",
                    "tackles",
                    "clearances_blocks_interceptions",
                    "recoveries",
                ):
                    _coalesce(c)
    except Exception as exc:
        # Best-effort; don't block core loading, but make failure visible.
        print(f"Warning: failed to merge playerstats.csv availability signals: {exc}")

    # Prefer web_name for display.
    if "web_name" not in all_df.columns:
        all_df["web_name"] = ""

    # Attach simple fixture difficulty features from GW-level fixtures.csv.
    # Best-effort: if fixtures/teams aren't available, we still create the columns
    # so downstream preprocessing can depend on a stable schema.
    fixture_cols = ["fixture_is_home", "opp_strength", "opp_elo", "fixture_difficulty"]
    try:
        if "team_code" in all_df.columns:
            teams_path = paths.insights_data_root / season / "teams.csv"
            fixture_table = _build_fixture_difficulty_table(base=base, teams_path=teams_path)
            if not fixture_table.empty:
                all_df = all_df.merge(
                    fixture_table,
                    on=["gw", "team_code"],
                    how="left",
                )
    except Exception as exc:
        print(f"Warning: failed to merge fixture difficulty features: {exc}")

    for c in fixture_cols:
        if c not in all_df.columns:
            all_df[c] = pd.NA

    # Safety: ensure leaky official prediction columns never flow into modeling.
    for c in ("ep_next", "ep_this"):
        if c in all_df.columns:
            all_df = all_df.drop(columns=[c])
    
    # Apply feature engineering if requested
    if apply_feature_engineering:
        # Optionally load previous season for new player handling
        previous_season_df = None
        if handle_new_players and previous_season:
            try:
                print(f"Loading previous season ({previous_season}) for new player handling...")
                previous_season_df = load_premier_league_gameweek_stats(
                    repo_root=repo_root,
                    season=previous_season,
                    apply_feature_engineering=False,  # Don't recurse
                    handle_new_players=False,
                )
                print(f"Loaded {len(previous_season_df)} records from {previous_season}")
            except Exception as e:
                print(f"Warning: Could not load previous season data: {e}")
                print("New players will not have position-based priors applied")
        
        all_df = engineer_all_features(
            all_df,
            handle_new_players=handle_new_players and previous_season_df is not None,
            previous_season_df=previous_season_df,
        )

    return all_df


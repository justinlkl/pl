from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .feature_engineering import engineer_all_features


@dataclass(frozen=True)
class DataPaths:
    repo_root: Path

    @property
    def insights_data_root(self) -> Path:
        return self.repo_root / "FPL-Core-Insights" / "data"


def _extract_gw_from_path(path: Path) -> int | None:
    # Matches .../GW12/... or ...\\GW12\\...
    match = re.search(r"(?:^|[\\/])GW(\d+)(?:[\\/]|$)", str(path))
    if not match:
        return None
    return int(match.group(1))


def load_premier_league_gameweek_stats(
    *, repo_root: Path, season: str, apply_feature_engineering: bool = True
) -> pd.DataFrame:
    """Load Premier League player_gameweek_stats.csv across all GW folders.

    Expected layout (from FPL-Core-Insights):
    data/<season>/By Tournament/Premier League/GW*/player_gameweek_stats.csv
    
    Args:
        repo_root: Root directory of the repository
        season: Season identifier (e.g., "2025-2026")
        apply_feature_engineering: If True, apply all feature engineering transformations
        
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

    # Attach per-GW player availability/market signals from playerstats.csv (best-effort).
    # This file includes chance_of_playing_* and selected_by_percent, which help avoid
    # over-ranking likely-bench players.
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

                # Only bring the availability/market signals needed for modeling.
                # Keeping this small avoids mixed-type merge issues and keeps loading fast.
                keep_cols = [
                    "player_id",
                    "gw",
                    "chance_of_playing_next_round",
                    "chance_of_playing_this_round",
                    "selected_by_percent",
                    "ep_next",
                    "ep_this",
                ]
                keep_cols = [c for c in keep_cols if c in stats.columns]
                stats = stats[keep_cols].drop_duplicates(subset=["player_id", "gw"], keep="last")

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
                    "ep_next",
                    "ep_this",
                ):
                    _coalesce(c)
    except Exception as exc:
        # Best-effort; don't block core loading, but make failure visible.
        print(f"Warning: failed to merge playerstats.csv availability signals: {exc}")

    # Prefer web_name for display.
    if "web_name" not in all_df.columns:
        all_df["web_name"] = ""
    
    # Apply feature engineering if requested
    if apply_feature_engineering:
        all_df = engineer_all_features(all_df)

    return all_df


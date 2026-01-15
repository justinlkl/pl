from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


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
    *, repo_root: Path, season: str
) -> pd.DataFrame:
    """Load Premier League player_gameweek_stats.csv across all GW folders.

    Expected layout (from FPL-Core-Insights):
    data/<season>/By Tournament/Premier League/GW*/player_gameweek_stats.csv
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

    # Prefer web_name for display.
    if "web_name" not in all_df.columns:
        all_df["web_name"] = ""

    return all_df

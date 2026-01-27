"""Fixture difficulty and opponent strength features.

Computes rolling opponent defensive/attacking metrics to enhance prediction signals
beyond historical player patterns alone.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


def build_opponent_strength_table(
    *,
    df: pd.DataFrame,
    repo_root: Path,
    season: str,
) -> pd.DataFrame:
    """Build rolling opponent defensive strength metrics per GW/team.
    
    Aggregates opponent performance (goals conceded, xGC, clean sheets) to
    estimate defensive difficulty for upcoming fixtures.
    
    Args:
        df: Player gameweek stats (must contain gw, team_code, goals_conceded, etc.)
        repo_root: Repository root for accessing fixtures data
        season: Season identifier (e.g., "2025-2026")
        
    Returns:
        DataFrame with opponent strength metrics indexed by (gw, team_code)
    """
    
    # Group by opponent (team_code) and gw to get defensive metrics
    opponent_agg = (
        df.groupby(["gw", "team_code"], as_index=False)
        .agg({
            "goals_conceded": "mean",
            "expected_goals_conceded": "mean",
            "clean_sheets": "mean",
            "defensive_contribution": "sum",
        })
        .rename(columns={
            "goals_conceded": "opp_goals_conceded",
            "expected_goals_conceded": "opp_xgc",
            "clean_sheets": "opp_clean_sheets",
            "defensive_contribution": "opp_defcon_sum",
        })
    )
    
    # Rolling average (last 6 GWs) to smooth recent defensive form
    opponent_agg["rolling_opp_gc"] = (
        opponent_agg.sort_values(["team_code", "gw"])
        .groupby("team_code")["opp_goals_conceded"]
        .transform(lambda x: x.rolling(window=6, min_periods=1).mean())
    )
    opponent_agg["rolling_opp_xgc"] = (
        opponent_agg.sort_values(["team_code", "gw"])
        .groupby("team_code")["opp_xgc"]
        .transform(lambda x: x.rolling(window=6, min_periods=1).mean())
    )
    opponent_agg["rolling_opp_cs_rate"] = (
        opponent_agg.sort_values(["team_code", "gw"])
        .groupby("team_code")["opp_clean_sheets"]
        .transform(lambda x: x.rolling(window=6, min_periods=1).mean())
    )
    
    # Invert: lower defensive strength (higher opponent GC) = easier for attackers
    opponent_agg["opponent_def_strength"] = (
        1.0 / (opponent_agg["rolling_opp_gc"] + 0.1)  # +0.1 to avoid div by 0
    )
    
    return opponent_agg


def add_fixture_features(
    df: pd.DataFrame,
    opponent_strength: pd.DataFrame,
) -> pd.DataFrame:
    """Merge opponent strength metrics onto player data.
    
    For each player's upcoming gameweek, attach rolling opponent defensive metrics.
    Also apply home/away multiplier to reflect home advantage.
    
    Args:
        df: Player gameweek stats
        opponent_strength: Pre-computed opponent strength table
        
    Returns:
        DataFrame with added fixture features
    """
    
    df = df.copy()
    
    # Assume next GW fixture is stored or can be inferred from sequence end_gw
    # For simplicity: merge opponent strength as a GW+1 lookup or current GW lookup
    
    # Merge opponent strength (defensive quality of upcoming opponent)
    df = df.merge(
        opponent_strength[["gw", "team_code", "rolling_opp_gc", "rolling_opp_xgc", 
                          "rolling_opp_cs_rate", "opponent_def_strength"]],
        left_on=["gw", "opponent_team_code"],
        right_on=["gw", "team_code"],
        how="left",
        suffixes=("", "_opp"),
    )
    
    # Home/away adjustment: attackers gain ~2% boost at home
    is_home = (df.get("fixture_is_home") == 1).fillna(False)
    df["opponent_strength_adj"] = np.where(
        is_home,
        df["opponent_def_strength"] * 1.02,  # Home advantage for attacking
        df["opponent_def_strength"] * 0.98,   # Away penalty
    )
    
    # Fill missing values with league average
    league_avg_strength = df["opponent_def_strength"].mean()
    df["opponent_def_strength"] = df["opponent_def_strength"].fillna(league_avg_strength)
    df["opponent_strength_adj"] = df["opponent_strength_adj"].fillna(league_avg_strength)
    
    return df

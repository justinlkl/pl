from __future__ import annotations

"""Handle new players and teams across seasons.

This module provides utilities for:
1. Identifying new/removed players between seasons
2. Handling promoted/relegated teams
3. Filling missing historical features for new players with position-based priors
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


# Relegated teams from 2024-2025 season (not in 2025-2026)
RELEGATED_TEAMS_2024_25 = {
    "Leicester",
    "Leicester City",
    "Southampton", 
    "Ipswich",
    "Ipswich Town",
}

# Promoted teams for 2025-2026 season (flexible matching)
PROMOTED_TEAMS_2025_26 = {
    "Leeds",
    "Burnley",
    "Sunderland",
}


def identify_new_players(
    df_current: pd.DataFrame,
    df_previous: pd.DataFrame | None = None,
    *,
    player_id_col: str = "player_id",
) -> set[int]:
    """Identify players who are new in the current season.
    
    Args:
        df_current: Current season dataframe with player data
        df_previous: Previous season dataframe (if None, all players treated as new)
        player_id_col: Name of the player ID column
        
    Returns:
        Set of player IDs that are new in the current season
    """
    if df_previous is None or df_previous.empty:
        return set(df_current[player_id_col].unique())
    
    players_previous = set(df_previous[player_id_col].unique())
    players_current = set(df_current[player_id_col].unique())
    
    new_players = players_current - players_previous
    
    print(f"Found {len(new_players)} new players in current season")
    print(f"Previous season had {len(players_previous)} players")
    print(f"Current season has {len(players_current)} players")
    
    return new_players


def identify_removed_players(
    df_current: pd.DataFrame,
    df_previous: pd.DataFrame,
    *,
    player_id_col: str = "player_id",
) -> set[int]:
    """Identify players who were in previous season but not in current season.
    
    Args:
        df_current: Current season dataframe
        df_previous: Previous season dataframe
        player_id_col: Name of the player ID column
        
    Returns:
        Set of player IDs removed from previous season
    """
    players_previous = set(df_previous[player_id_col].unique())
    players_current = set(df_current[player_id_col].unique())
    
    removed_players = players_previous - players_current
    
    print(f"Found {len(removed_players)} players removed from previous season")
    
    return removed_players


def filter_relegated_teams(
    df: pd.DataFrame,
    *,
    team_col: str = "team",
    relegated_teams: set[str] = RELEGATED_TEAMS_2024_25,
) -> pd.DataFrame:
    """Remove players from relegated teams.
    
    Args:
        df: DataFrame with player data
        team_col: Name of the team column
        relegated_teams: Set of relegated team names
        
    Returns:
        DataFrame with relegated team players removed
    """
    if team_col not in df.columns:
        print(f"Warning: Column '{team_col}' not found in dataframe")
        return df
    
    before_count = len(df)
    
    # Case-insensitive matching
    team_lower = df[team_col].astype(str).str.lower()
    relegated_lower = {t.lower() for t in relegated_teams}
    
    mask = ~team_lower.isin(relegated_lower)
    df_filtered = df[mask].copy()
    
    removed_count = before_count - len(df_filtered)
    
    if removed_count > 0:
        print(f"Removed {removed_count} records from relegated teams: {relegated_teams}")
    
    return df_filtered


def calculate_position_priors(
    df: pd.DataFrame,
    *,
    position_col: str = "position",
    feature_columns: list[str] | None = None,
) -> dict[str, pd.Series]:
    """Calculate average feature values per position (for filling new players).
    
    Args:
        df: DataFrame with player data
        position_col: Name of the position column
        feature_columns: List of features to calculate priors for (if None, uses numeric columns)
        
    Returns:
        Dictionary mapping position -> Series of average feature values
    """
    if position_col not in df.columns:
        raise ValueError(f"Position column '{position_col}' not found")
    
    # Use only numeric columns if feature_columns not specified
    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude ID and gameweek columns
        exclude = {"player_id", "gw", "id", "team_id", "team_code"}
        feature_columns = [c for c in feature_columns if c not in exclude]
    
    # Filter to columns that actually exist
    feature_columns = [c for c in feature_columns if c in df.columns]
    
    position_priors = {}
    
    for pos in ["GK", "DEF", "MID", "FWD"]:
        # Case-insensitive position matching
        pos_lower = pos.lower()
        pos_mask = df[position_col].astype(str).str.lower().str.contains(pos_lower, na=False)
        
        if pos_mask.sum() == 0:
            print(f"Warning: No players found for position {pos}")
            continue
        
        # Calculate mean for each feature
        pos_data = df.loc[pos_mask, feature_columns]
        position_priors[pos] = pos_data.mean()
        
        print(f"Position {pos}: calculated priors from {pos_mask.sum()} players")
    
    return position_priors


def fill_new_player_features(
    df: pd.DataFrame,
    new_player_ids: set[int],
    position_priors: dict[str, pd.Series],
    *,
    player_id_col: str = "player_id",
    position_col: str = "position",
    rolling_features: list[str] | None = None,
    cumulative_features: list[str] | None = None,
) -> pd.DataFrame:
    """Fill missing rolling/cumulative features for new players using position priors.
    
    Args:
        df: DataFrame with player data
        new_player_ids: Set of new player IDs
        position_priors: Position-based feature priors from calculate_position_priors
        player_id_col: Name of the player ID column
        position_col: Name of the position column
        rolling_features: List of rolling feature names to fill
        cumulative_features: List of cumulative feature names to fill
        
    Returns:
        DataFrame with filled features for new players
    """
    if rolling_features is None:
        rolling_features = []
    if cumulative_features is None:
        cumulative_features = []
    
    df = df.copy()
    
    filled_count = 0
    
    for player_id in new_player_ids:
        player_mask = df[player_id_col] == player_id
        
        if player_mask.sum() == 0:
            continue
        
        # Get player's position (use first non-null value)
        player_positions = df.loc[player_mask, position_col].dropna()
        if len(player_positions) == 0:
            continue
        
        pos = str(player_positions.iloc[0]).upper()
        
        # Normalize position to GK/DEF/MID/FWD
        if "GK" in pos or "GOAL" in pos:
            pos_key = "GK"
        elif "DEF" in pos or "DF" in pos:
            pos_key = "DEF"
        elif "MID" in pos or "MF" in pos:
            pos_key = "MID"
        elif "FWD" in pos or "FW" in pos or "FORWARD" in pos or "STRIKER" in pos:
            pos_key = "FWD"
        else:
            pos_key = "MID"  # Default fallback
        
        if pos_key not in position_priors:
            continue
        
        priors = position_priors[pos_key]
        
        # Fill rolling features with position average
        for feat in rolling_features:
            if feat in df.columns and feat in priors.index:
                # Only fill if missing (NaN or 0 for new players)
                missing_mask = player_mask & ((df[feat].isna()) | (df[feat] == 0))
                if missing_mask.sum() > 0:
                    df.loc[missing_mask, feat] = priors[feat]
                    filled_count += missing_mask.sum()
        
        # Fill cumulative features with 0 (no history yet)
        for feat in cumulative_features:
            if feat in df.columns:
                missing_mask = player_mask & df[feat].isna()
                if missing_mask.sum() > 0:
                    df.loc[missing_mask, feat] = 0.0
                    filled_count += missing_mask.sum()
    
    if filled_count > 0:
        print(f"Filled {filled_count} feature values for {len(new_player_ids)} new players using position priors")
    
    return df


def handle_new_players_full_pipeline(
    df_current: pd.DataFrame,
    df_previous: pd.DataFrame | None = None,
    *,
    player_id_col: str = "player_id",
    position_col: str = "position",
    rolling_features: list[str] | None = None,
    cumulative_features: list[str] | None = None,
) -> pd.DataFrame:
    """Full pipeline to handle new players in current season.
    
    This function:
    1. Identifies new players
    2. Calculates position-based priors from current season data
    3. Fills missing rolling/cumulative features for new players
    
    Args:
        df_current: Current season dataframe
        df_previous: Previous season dataframe (optional)
        player_id_col: Name of player ID column
        position_col: Name of position column
        rolling_features: List of rolling features to fill
        cumulative_features: List of cumulative features to fill
        
    Returns:
        DataFrame with new player features filled
    """
    # Default feature lists if not provided
    if rolling_features is None:
        rolling_features = [
            "rolling_3_xg",
            "rolling_3_xa",
            "rolling_3_xgi",
            "rolling_5_xg",
            "rolling_5_xa",
            "rolling_5_xgi",
            "rolling_5_points",
            "rolling_5_minutes",
            "rolling_5_appearances",
            "rolling_5_defensive",
            "rolling_5_defensive_def",
            "rolling_5_defensive_gk",
            "rolling_5_defensive_mid",
            "rolling_5_defensive_fwd",
        ]
    
    if cumulative_features is None:
        cumulative_features = [
            "cumulative_xg",
            "cumulative_xa",
            "cumulative_xgi",
        ]
    
    # Identify new players
    new_players = identify_new_players(df_current, df_previous, player_id_col=player_id_col)
    
    if len(new_players) == 0:
        print("No new players to handle")
        return df_current
    
    # Calculate position priors from current season
    all_features = rolling_features + cumulative_features
    position_priors = calculate_position_priors(
        df_current,
        position_col=position_col,
        feature_columns=all_features,
    )
    
    # Fill features for new players
    df_filled = fill_new_player_features(
        df_current,
        new_players,
        position_priors,
        player_id_col=player_id_col,
        position_col=position_col,
        rolling_features=rolling_features,
        cumulative_features=cumulative_features,
    )
    
    return df_filled


def get_team_mapping(
    repo_root: Path,
    season: str,
) -> dict[str, dict[str, any]]:
    """Load team metadata for a season.
    
    Args:
        repo_root: Repository root path
        season: Season identifier (e.g., "2025-2026")
        
    Returns:
        Dictionary mapping team_id -> {name, code, ...}
    """
    from .data_loading import load_insights_teams
    
    teams_df = load_insights_teams(repo_root=repo_root, season=season)
    
    team_mapping = {}
    for _, row in teams_df.iterrows():
        team_id = row.get("id")
        if pd.notna(team_id):
            team_mapping[int(team_id)] = row.to_dict()
    
    return team_mapping


def validate_teams(
    df: pd.DataFrame,
    expected_promoted: set[str] = PROMOTED_TEAMS_2025_26,
    expected_relegated: set[str] = RELEGATED_TEAMS_2024_25,
    *,
    team_col: str = "team",
) -> dict[str, list[str]]:
    """Validate that promoted teams are present and relegated teams are absent.
    
    Args:
        df: DataFrame with team data
        expected_promoted: Set of expected promoted team names
        expected_relegated: Set of expected relegated team names
        team_col: Name of the team column
        
    Returns:
        Dictionary with validation results
    """
    if team_col not in df.columns:
        return {"error": [f"Team column '{team_col}' not found"]}
    
    teams_in_df = set(df[team_col].dropna().astype(str).unique())
    teams_lower = {t.lower() for t in teams_in_df}
    
    # Check promoted teams
    promoted_lower = {t.lower() for t in expected_promoted}
    promoted_found = [t for t in expected_promoted if t.lower() in teams_lower]
    promoted_missing = [t for t in expected_promoted if t.lower() not in teams_lower]
    
    # Check relegated teams
    relegated_lower = {t.lower() for t in expected_relegated}
    relegated_found = [t for t in expected_relegated if t.lower() in teams_lower]
    relegated_missing = [t for t in expected_relegated if t.lower() not in teams_lower]
    
    results = {
        "promoted_found": promoted_found,
        "promoted_missing": promoted_missing,
        "relegated_still_present": relegated_found,
        "relegated_removed": relegated_missing,
        "all_teams": sorted(teams_in_df),
    }
    
    # Print summary
    print("\n=== Team Validation ===")
    print(f"Promoted teams found: {len(promoted_found)}/{len(expected_promoted)}")
    if promoted_missing:
        print(f"  Missing: {promoted_missing}")
    
    print(f"Relegated teams removed: {len(relegated_missing)}/{len(expected_relegated)}")
    if relegated_found:
        print(f"  WARNING: Still present: {relegated_found}")
    
    print(f"\nTotal teams in dataset: {len(teams_in_df)}")
    
    return results

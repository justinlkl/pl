from __future__ import annotations

"""Comprehensive data processing for FPL multi-season training.

Handles:
1. Column alignment between 24-25 and 25-26 seasons
2. Defensive contribution recalculation for 24-25
3. New player handling
4. New team (promoted) default ratings
5. Player transfer tracking
6. Feature engineering for training
"""

from pathlib import Path

import numpy as np
import pandas as pd


# Default strength ratings for promoted teams
DEFAULT_PROMOTED_TEAM_STRENGTH = {
    "strength": 2,  # Weak rating
    "strength_overall_home": 750,
    "strength_overall_away": 700,
    "strength_attack_home": 700,
    "strength_attack_away": 650,
    "strength_defence_home": 700,
    "strength_defence_away": 650,
    "elo": 1450,
}


def calculate_defensive_contribution_legacy(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate defensive contribution for 24-25 season (not in original data).
    
    Uses match-level stats when available, or aggregates from season stats.
    
    Args:
        df: DataFrame with defensive stats
        
    Returns:
        DataFrame with defensive_contribution column added
    """
    df = df.copy()
    
    # Components: tackles + clearances + blocks + interceptions
    tackles = pd.to_numeric(df.get("tackles", 0), errors="coerce").fillna(0.0)
    
    # Try to get CBI as combined column or individual
    if "clearances_blocks_interceptions" in df.columns:
        cbi = pd.to_numeric(df["clearances_blocks_interceptions"], errors="coerce").fillna(0.0)
    else:
        clearances = pd.to_numeric(df.get("clearances", 0), errors="coerce").fillna(0.0)
        blocks = pd.to_numeric(df.get("blocks", 0), errors="coerce").fillna(0.0)
        interceptions = pd.to_numeric(df.get("interceptions", 0), errors="coerce").fillna(0.0)
        cbi = clearances + blocks + interceptions
    
    df["defensive_contribution"] = tackles + cbi
    
    return df


def calculate_defensive_contribution_points(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate FPL defensive contribution points based on 25-26 rules.
    
    DEF/GK: 2 pts if CBIT (tackles + clearances + blocks + interceptions) >= 10
    MID/FWD: 2 pts if CBIRT (CBIT + recoveries) >= 12
    
    Args:
        df: DataFrame with position and defensive stats
        
    Returns:
        DataFrame with defcon_points column added
    """
    df = df.copy()
    
    # Ensure defensive_contribution exists
    if "defensive_contribution" not in df.columns:
        df = calculate_defensive_contribution_legacy(df)
    
    # Get position
    pos_norm = df.get("position", pd.Series(["" for _ in range(len(df))])).astype(str).str.strip().str.lower()
    is_def_gk = pos_norm.isin(["defender", "def", "df", "goalkeeper", "gk"])
    is_mid_fwd = pos_norm.isin(["midfielder", "mid", "mf", "forward", "fwd", "fw", "striker"])
    
    # CBIT = tackles + clearances + blocks + interceptions
    defcon = pd.to_numeric(df["defensive_contribution"], errors="coerce").fillna(0.0)
    
    # For MID/FWD, add recoveries
    recoveries = pd.to_numeric(df.get("recoveries", 0), errors="coerce").fillna(0.0)
    
    # Calculate points
    defcon_points = np.zeros(len(df), dtype=float)
    
    # DEF/GK: CBIT >= 10 → +2
    defcon_points[is_def_gk & (defcon >= 10)] = 2.0
    
    # MID/FWD: CBIRT >= 12 → +2
    cbirt = defcon + recoveries
    defcon_points[is_mid_fwd & (cbirt >= 12)] = 2.0
    
    df["defcon_points"] = defcon_points
    df["cbit"] = defcon.astype(float)
    df["cbirt"] = cbirt.astype(float)
    
    return df


def calculate_adjusted_points(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate adjusted_points = total_points + defcon_points (for 24-25).
    
    For 25-26, defensive contribution is already in total_points.
    
    Args:
        df: DataFrame with total_points and defcon_points
        
    Returns:
        DataFrame with adjusted_points column
    """
    df = df.copy()
    
    # Calculate defcon points if not present
    if "defcon_points" not in df.columns:
        df = calculate_defensive_contribution_points(df)
    
    total_points = pd.to_numeric(df.get("total_points", 0), errors="coerce").fillna(0.0)
    defcon_points = pd.to_numeric(df.get("defcon_points", 0), errors="coerce").fillna(0.0)
    
    # For 24-25: add defcon points
    # For 25-26: already included in total_points, so just copy
    # We'll mark which season it is with a flag
    df["adjusted_points"] = total_points + defcon_points
    
    return df


def align_column_names(df: pd.DataFrame, *, season: str) -> pd.DataFrame:
    """Align column names to consistent naming across seasons.
    
    Args:
        df: DataFrame to align
        season: "2024-2025" or "2025-2026"
        
    Returns:
        DataFrame with aligned column names
    """
    df = df.copy()
    
    # Standard renames for 25-26
    renames_2526 = {
        "goals_scored": "goals",
        "assists": "assists",  # Should be same
    }
    
    # Apply renames if this is 25-26
    if "2025" in season and "2026" in season:
        for old_col, new_col in renames_2526.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})
    
    # Ensure standard ID column
    if "id" in df.columns and "player_id" not in df.columns:
        df = df.rename(columns={"id": "player_id"})
    
    return df


def add_new_team_defaults(teams_df: pd.DataFrame, *, promoted_teams: set[str]) -> pd.DataFrame:
    """Add default strength ratings for promoted teams if missing.
    
    Args:
        teams_df: Teams DataFrame
        promoted_teams: Set of promoted team names
        
    Returns:
        Teams DataFrame with defaults filled
    """
    teams_df = teams_df.copy()
    
    # Normalize team names for matching
    team_names_lower = teams_df.get("name", pd.Series()).astype(str).str.lower()
    promoted_lower = {t.lower() for t in promoted_teams}
    
    for idx, team_name in enumerate(team_names_lower):
        if team_name in promoted_lower:
            # Fill missing strength columns with defaults
            for col, default_val in DEFAULT_PROMOTED_TEAM_STRENGTH.items():
                if col in teams_df.columns:
                    if pd.isna(teams_df.at[idx, col]) or teams_df.at[idx, col] == 0:
                        teams_df.at[idx, col] = default_val
    
    return teams_df


def track_player_transfers(df: pd.DataFrame) -> pd.DataFrame:
    """Track when players change teams and flag transfer events.
    
    Args:
        df: DataFrame with player_id, gw, team_code (or team_id)
        
    Returns:
        DataFrame with team_changed flag added
    """
    df = df.copy()
    
    team_col = None
    for col in ["team_code", "team_id", "team"]:
        if col in df.columns:
            team_col = col
            break
    
    if team_col is None:
        df["team_changed"] = False
        return df
    
    # Sort by player and gameweek
    df = df.sort_values(["player_id", "gw"]).reset_index(drop=True)
    
    # Track team changes
    df["prev_team"] = df.groupby("player_id")[team_col].shift(1)
    df["team_changed"] = (df[team_col] != df["prev_team"]) & df["prev_team"].notna()
    
    # Drop temporary column
    df = df.drop(columns=["prev_team"])
    
    return df


def reset_rolling_on_transfer(df: pd.DataFrame, *, rolling_cols: list[str]) -> pd.DataFrame:
    """Reset team-based rolling features when player transfers.
    
    Args:
        df: DataFrame with team_changed flag and rolling columns
        rolling_cols: List of rolling column names to reset
        
    Returns:
        DataFrame with rolling features reset on transfer
    """
    df = df.copy()
    
    if "team_changed" not in df.columns:
        return df
    
    # Team-based rolling features (reset on transfer)
    team_based_features = [
        col for col in rolling_cols
        if any(x in col.lower() for x in ["team", "defence", "attack", "opponent"])
    ]
    
    # Reset to NaN when transfer occurs (will be filled by imputer)
    for col in team_based_features:
        if col in df.columns:
            df.loc[df["team_changed"], col] = np.nan
    
    return df


def mark_new_players(df: pd.DataFrame, *, previous_season_ids: set[int] | None) -> pd.DataFrame:
    """Add is_new_player flag for players not in previous season.
    
    Args:
        df: Current season DataFrame
        previous_season_ids: Set of player IDs from previous season
        
    Returns:
        DataFrame with is_new_player column
    """
    df = df.copy()
    
    if previous_season_ids is None or len(previous_season_ids) == 0:
        df["is_new_player"] = True
        return df
    
    df["is_new_player"] = ~df["player_id"].isin(previous_season_ids)
    
    return df


def calculate_set_piece_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create boolean flags for set piece takers.
    
    Args:
        df: DataFrame with set piece order columns
        
    Returns:
        DataFrame with is_penalty_taker, is_freekick_taker flags
    """
    df = df.copy()
    
    # Penalty taker
    if "penalties_order" in df.columns:
        pen_order = pd.to_numeric(df["penalties_order"], errors="coerce")
        df["is_penalty_taker"] = (pen_order == 1).astype(float)
    else:
        df["is_penalty_taker"] = 0.0
    
    # Free kick taker
    if "direct_freekicks_order" in df.columns:
        fk_order = pd.to_numeric(df["direct_freekicks_order"], errors="coerce")
        df["is_freekick_taker"] = (fk_order == 1).astype(float)
    else:
        df["is_freekick_taker"] = 0.0
    
    # Corners
    if "corners_and_indirect_freekicks_order" in df.columns:
        corner_order = pd.to_numeric(df["corners_and_indirect_freekicks_order"], errors="coerce")
        df["is_corner_taker"] = (corner_order == 1).astype(float)
    else:
        df["is_corner_taker"] = 0.0
    
    return df


def process_season_data(
    df: pd.DataFrame,
    *,
    season: str,
    previous_season_ids: set[int] | None = None,
    recalculate_defcon: bool = False,
) -> pd.DataFrame:
    """Process a single season's data with all transformations.
    
    Args:
        df: Raw season DataFrame
        season: Season identifier
        previous_season_ids: Player IDs from previous season (for new player flagging)
        recalculate_defcon: If True, recalculate defensive contribution (for 24-25)
        
    Returns:
        Processed DataFrame ready for feature engineering
    """
    df = df.copy()
    
    print(f"\nProcessing {season} season data...")
    print(f"  Initial shape: {df.shape}")
    
    # 1. Align column names
    df = align_column_names(df, season=season)
    
    # 2. Calculate defensive contributions if needed
    if recalculate_defcon or "defensive_contribution" not in df.columns:
        print(f"  Calculating defensive contributions...")
        df = calculate_defensive_contribution_legacy(df)
        df = calculate_defensive_contribution_points(df)
        df = calculate_adjusted_points(df)
    
    # 3. Mark new players
    df = mark_new_players(df, previous_season_ids=previous_season_ids)
    new_player_count = df["is_new_player"].sum() if "is_new_player" in df.columns else 0
    print(f"  New players: {new_player_count}")
    
    # 4. Track transfers
    df = track_player_transfers(df)
    transfer_count = df["team_changed"].sum() if "team_changed" in df.columns else 0
    print(f"  Player transfers: {transfer_count}")
    
    # 5. Set piece flags
    df = calculate_set_piece_flags(df)
    
    print(f"  Final shape: {df.shape}")
    
    return df


def combine_seasons(
    df_2425: pd.DataFrame,
    df_2526: pd.DataFrame,
    *,
    recalculate_2425_defcon: bool = True,
) -> pd.DataFrame:
    """Combine 24-25 and 25-26 seasons into single training dataset.
    
    Args:
        df_2425: 2024-2025 season data
        df_2526: 2025-2026 season data
        recalculate_2425_defcon: Recalculate defensive contributions for 24-25
        
    Returns:
        Combined DataFrame with season identifier
    """
    print("\n" + "="*60)
    print("COMBINING SEASONS FOR TRAINING")
    print("="*60)
    
    # Get player IDs from 24-25 for new player detection
    previous_season_ids = set(df_2425["player_id"].unique()) if "player_id" in df_2425.columns else set()
    
    # Process each season
    df_2425_processed = process_season_data(
        df_2425,
        season="2024-2025",
        previous_season_ids=None,  # All players are "existing" in first season
        recalculate_defcon=recalculate_2425_defcon,
    )
    
    df_2526_processed = process_season_data(
        df_2526,
        season="2025-2026",
        previous_season_ids=previous_season_ids,
        recalculate_defcon=False,  # 25-26 already has it
    )
    
    # Find common columns
    common_cols = list(set(df_2425_processed.columns) & set(df_2526_processed.columns))
    print(f"\nCommon columns: {len(common_cols)}")
    
    # Add season identifier
    df_2425_processed["season"] = "2024-2025"
    df_2526_processed["season"] = "2025-2026"
    
    if "season" not in common_cols:
        common_cols.append("season")
    
    # Combine
    df_combined = pd.concat(
        [df_2425_processed[common_cols], df_2526_processed[common_cols]],
        ignore_index=True,
    )
    
    print(f"\nCombined dataset:")
    print(f"  Total records: {len(df_combined)}")
    print(f"  Unique players: {df_combined['player_id'].nunique()}")
    print(f"  24-25 records: {(df_combined['season'] == '2024-2025').sum()}")
    print(f"  25-26 records: {(df_combined['season'] == '2025-2026').sum()}")
    
    return df_combined


# Final feature list for training
CORE_IDENTITY_COLS = [
    "player_id",
    "gw",
    "position",
    "team_code",
    "web_name",
]

CORE_TRAINING_FEATURES = [
    # Playing time
    "minutes",
    "starts",
    
    # Expected stats (total)
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    
    # Expected stats (per-90)
    "expected_goals_per_90",
    "expected_assists_per_90",
    "expected_goal_involvements_per_90",
    "expected_goals_conceded_per_90",
    
    # Defensive (NEW RULES)
    "defensive_contribution",
    "defensive_contribution_per_90",
    "tackles",
    "clearances_blocks_interceptions",
    "recoveries",
    "cbit",
    "cbirt",
    
    # GK-specific
    "saves",
    "saves_per_90",
    "clean_sheets",
    "clean_sheets_per_90",
    "goals_conceded_per_90",
    
    # ICT
    "influence",
    "creativity",
    "threat",
    "ict_index",
    
    # Form
    "form",
    "points_per_game",
    
    # Price/value
    "now_cost",
    "value_form",
    "value_season",
    "selected_by_percent",
    
    # Set pieces
    "is_penalty_taker",
    "is_freekick_taker",
    "is_corner_taker",
    "penalties_order",
    
    # Context
    "chance_of_playing_this_round",
    "chance_of_playing_next_round",
    
    # Flags
    "is_new_player",
    "team_changed",
]

TARGET_COLS = [
    "total_points",
    "adjusted_points",
    "defcon_points",
]


def get_training_feature_list(*, include_engineered: bool = True) -> list[str]:
    """Get complete list of training features.
    
    Args:
        include_engineered: Include engineered features (rolling, cumulative, etc.)
        
    Returns:
        List of feature column names
    """
    features = CORE_TRAINING_FEATURES.copy()
    
    if include_engineered:
        # Add rolling features
        rolling_features = [
            "rolling_3_xg",
            "rolling_3_xa",
            "rolling_3_xgi",
            "rolling_5_xg",
            "rolling_5_xa",
            "rolling_5_xgi",
            "rolling_5_points",
            "rolling_5_minutes",
            "rolling_5_defensive",
        ]
        
        # Add cumulative features
        cumulative_features = [
            "cumulative_xg",
            "cumulative_xa",
            "cumulative_xgi",
        ]
        
        features.extend(rolling_features)
        features.extend(cumulative_features)
    
    return features

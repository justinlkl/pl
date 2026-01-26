"""Position-specific feature sets for FPL modeling.

This module defines which features to use for training position-specific models.
Features are organized into universal (all positions) and position-specific sets.
"""

from typing import Dict, List

# === UNIVERSAL FEATURES (All positions) ===
UNIVERSAL_FEATURES = [
    # Form metrics
    "adjusted_points_rolling_3",
    "adjusted_points_rolling_5",
    "points_std_5",
    
    # Minutes played
    "minutes_played_rolling_3",
    "minutes_played_rolling_5",
    "minutes_trend",
    "minutes_per_90",
    
    # Context
    "is_home",
    "fixture_difficulty",
    
    # Price/value
    "now_cost",
    "value_form",
    "value_season",
    
    # Team strength (from teams.csv)
    "strength_attack_home",
    "strength_attack_away",
    "strength_defence_home",
    "strength_defence_away",
    "strength_overall_home",
    "strength_overall_away",
    "elo",
    
    # Opponent strength
    "opp_strength",
    "opp_elo",
]

# === GK-SPECIFIC FEATURES ===
GK_FEATURES = [
    # Saves
    "saves_per_90",
    "saves_rolling_3",
    "saves_rolling_5",
    "save_pct",
    
    # Goals conceded
    "goals_conceded",
    "goals_conceded_per_90",
    "expected_goals_conceded",
    "expected_goals_conceded_per_90",
    
    # Clean sheets
    "clean_sheets",
    "clean_sheets_rolling_3",
    
    # Defensive contribution
    "defcon_points",
    "defcon_actions",
]

# === DEF-SPECIFIC FEATURES ===
DEF_FEATURES = [
    # Defensive actions
    "cbi_per_90",
    "cbi_rolling_3",
    "cbi_rolling_5",
    "clearances_per_90",
    "tackles_per_90",
    "interceptions_per_90",
    "blocks_per_90",
    
    # Defensive metrics
    "duel_success_rate",
    "tackles_won",
    "tackles_won_per_90",
    
    # Goals conceded (team defense)
    "expected_goals_conceded",
    "expected_goals_conceded_per_90",
    "clean_sheets",
    "clean_sheets_rolling_3",
    
    # Attacking threat (for attacking defenders)
    "xg_per_90",
    "xa_per_90",
    "xgi_per_90",
    "xg_rolling_3",
    
    # Defensive contribution points
    "defcon_points",
    "defcon_actions",
]

# === MID-SPECIFIC FEATURES ===
MID_FEATURES = [
    # Expected stats (attacking)
    "xgi_per_90",
    "xg_per_90",
    "xa_per_90",
    "xg_rolling_3",
    "xa_rolling_5",
    "xgi_rolling_3",
    
    # Defensive contribution (box-to-box midfielders)
    "cbirt_per_90",
    "cbirt_rolling_5",
    "defcon_points",
    "defcon_actions",
    
    # Creativity
    "chances_created",
    "chances_created_per_90",
    "key_passes",
    "key_passes_per_90",
    
    # Shooting
    "shots_on_target_per_90",
    "shot_accuracy",
    "total_shots_per_90",
    
    # Set pieces
    "is_penalty_taker",
    "is_freekick_taker",
    "is_corner_taker",
    "penalties_order",
]

# === FWD-SPECIFIC FEATURES ===
FWD_FEATURES = [
    # Goals
    "goals_per_90",
    "goals_rolling_3",
    "goals_rolling_5",
    
    # Expected stats
    "xg_per_90",
    "xa_per_90",
    "xgi_per_90",
    "xg_rolling_3",
    "xg_rolling_5",
    "xgi_rolling_3",
    
    # Shooting
    "shots_on_target_per_90",
    "total_shots_per_90",
    "shot_accuracy",
    "big_chances_missed",
    "big_chances_missed_per_90",
    
    # Set pieces
    "is_penalty_taker",
    "penalties_order",
    
    # Link-up play
    "chances_created",
    "chances_created_per_90",
]


def get_position_features(position: str, include_universal: bool = True) -> List[str]:
    """Get features for a specific position.
    
    Args:
        position: Position code ('GK', 'DEF', 'MID', or 'FWD')
        include_universal: Whether to include universal features
        
    Returns:
        List of feature names for the position
    """
    position_map = {
        "GK": GK_FEATURES,
        "DEF": DEF_FEATURES,
        "MID": MID_FEATURES,
        "FWD": FWD_FEATURES,
    }
    
    position = position.upper()
    if position not in position_map:
        raise ValueError(f"Unknown position: {position}. Must be one of {list(position_map.keys())}")
    
    features = []
    if include_universal:
        features.extend(UNIVERSAL_FEATURES)
    features.extend(position_map[position])
    
    return features


def get_all_position_features() -> Dict[str, List[str]]:
    """Get complete feature lists for all positions.
    
    Returns:
        Dictionary mapping position codes to feature lists (including universal features)
    """
    return {
        "GK": get_position_features("GK"),
        "DEF": get_position_features("DEF"),
        "MID": get_position_features("MID"),
        "FWD": get_position_features("FWD"),
    }


def filter_available_features(
    features: List[str], available_columns: List[str], warn_missing: bool = True
) -> List[str]:
    """Filter feature list to only include columns available in dataframe.
    
    Args:
        features: List of desired feature names
        available_columns: Columns actually present in the dataframe
        warn_missing: Whether to print warnings for missing features
        
    Returns:
        List of features that are actually available
    """
    available_set = set(available_columns)
    filtered = [f for f in features if f in available_set]
    
    if warn_missing:
        missing = set(features) - available_set
        if missing:
            print(f"Warning: {len(missing)} features not available: {sorted(missing)[:5]}...")
    
    return filtered


def save_feature_config(output_path: str = "models/feature_config.json") -> None:
    """Save feature configuration to JSON file for reproducibility.
    
    Args:
        output_path: Path to save JSON config file
    """
    import json
    from pathlib import Path
    
    config = get_all_position_features()
    
    # Create directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Feature configuration saved to {output_path}")
    for pos, features in config.items():
        print(f"  {pos}: {len(features)} features")


if __name__ == "__main__":
    # Save config when run as script
    save_feature_config()

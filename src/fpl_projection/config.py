from __future__ import annotations

"""Project configuration.

This repository uses the FPL-Core-Insights `player_gameweek_stats.csv` schema.
The recommended feature set below is mapped to the column names that actually
exist in that dataset.
"""

# ============================================================================
# RECOMMENDED CORE TRAINING SET FOR FPL ML
# ============================================================================

# Core Predictors: essential player & context metrics.
CORE_PREDICTORS: list[str] = [
    # Availability/selection signals (from playerstats.csv if present)
    "chance_of_playing_next_round",
    "chance_of_playing_this_round",
    "selected_by_percent",
    "ep_next",
    "ep_this",

    # Price/value context
    "now_cost",
    # FPL dataset already provides value metrics; treat value_season as the primary "value".
    "value_season",

    # Availability
    "minutes",
    "starts",  # maps to "games_started"
    # Engineered in feature_engineering.py
    "minutes_per_start",

    # Expected rates already available in the dataset
    "expected_goals_per_90",
    "expected_assists_per_90",
    "expected_goal_involvements_per_90",
    "expected_goals_conceded_per_90",

    # Expected totals
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",

    # ICT + form
    "ict_index",
    "influence",
    "creativity",
    "threat",
    "form",

    # Engineered: lightweight expected bonus proxy from ICT components
    "bps_bonus_proxy",

    # Position flags (engineered from season players.csv metadata)
    "pos_gk",
    "pos_def",
    "pos_mid",
    "pos_fwd",
]

# Performance Enhancers: Advanced defensive & efficiency metrics
PERFORMANCE_ENHANCERS: list[str] = [
    # Align with FPL 2025/26 scoring logic for defensive contributions
    # Role-weighted defensive channels (feature_engineering.calculate_role_weighted_features)
    "defcon_points_def",
    "defcon_points_gk",
    "defcon_actions_per_90_def",
    "defcon_actions_per_90_gk",
    "cbi_per_90_def",
    "cbi_per_90_gk",
    "tackles_per_90_def",
    "tackles_per_90_gk",
    "gi_minus_xgi",
    "xgi_overperformance",
    "xgi_underperformance",
    "goals_minus_xg",
    "assists_minus_xa",
    "goal_involvements",
]

# Rolling & Cumulative Features (calculated during preprocessing, train-only)
ROLLING_FEATURES: list[str] = [
    "rolling_3_xg",       # 3-game rolling xG
    "rolling_3_xa",       # 3-game rolling xA
    "rolling_3_xgi",      # 3-game rolling xGI
    "rolling_5_points",   # 5-game rolling points
    "rolling_5_minutes",  # 5-game rolling minutes
    # Role-weighted rolling defensive channels
    "rolling_5_defensive_def",
    "rolling_5_defensive_gk",
]

# Cumulative Features (calculated during preprocessing, train-only)
CUMULATIVE_FEATURES: list[str] = [
    "cumulative_xg",      # Season cumulative xG
    "cumulative_xa",      # Season cumulative xA
    "cumulative_xgi",     # Season cumulative xGI
]

# Complete training feature set
DEFAULT_FEATURE_COLUMNS: list[str] = CORE_PREDICTORS + PERFORMANCE_ENHANCERS + ROLLING_FEATURES + CUMULATIVE_FEATURES

# Target variable: Points scored in the gameweek
TARGET_COLUMN = "total_points"

# Sequence/Temporal Parameters
DEFAULT_SEQ_LENGTH = 5  # Look back 5 gameweeks
DEFAULT_HORIZON = 6  # Predict next 6 gameweeks

# Train/Val/Test Split Parameters (by gameweek)
DEFAULT_VAL_GWS = 3    # Last 3 gameweeks for validation
DEFAULT_TEST_GWS = 3   # Last 3 gameweeks for testing

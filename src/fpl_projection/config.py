from __future__ import annotations

# Curated feature set: performance + enhanced defensive/context metrics.
# You can tune this list, but keep it stable between training and prediction.
DEFAULT_FEATURE_COLUMNS: list[str] = [
    # Availability / price / selection context
    "now_cost",
    "selected_by_percent",
    "transfers_in_event",
    "transfers_out_event",
    "chance_of_playing_next_round",
    "form",
    "points_per_game",
    "ep_next",

    # Core match returns
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "yellow_cards",
    "red_cards",
    "bonus",
    "bps",

    # Advanced / expected
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",

    # ICT
    "influence",
    "creativity",
    "threat",
    "ict_index",

    # Enhanced defensive & contextual metrics
    "tackles",
    "clearances_blocks_interceptions",
    "recoveries",
    "defensive_contribution",
    "defensive_contribution_per_90",
    "starts_per_90",
]

TARGET_COLUMN = "total_points"

DEFAULT_SEQ_LENGTH = 5
DEFAULT_HORIZON = 5

# Split by sequence end gameweek (time-based split).
DEFAULT_VAL_GWS = 3
DEFAULT_TEST_GWS = 3

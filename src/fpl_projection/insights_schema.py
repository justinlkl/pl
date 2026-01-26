from __future__ import annotations

"""FPL-Core-Insights column schema helpers.

This module centralizes which columns we keep from FPL-Core-Insights exports for
ML training and/or merging.

Notes:
- "Drop" columns are excluded from training features, but some (e.g. IDs) may be
  kept as join keys when explicitly requested.
- The upstream dataset can vary by season; all selectors are best-effort and
  never error if a column is missing.
"""

from collections.abc import Iterable

import pandas as pd


# 7e2 KEEP - Essential Training Features
# Player Match Stats (player_match_stats.csv)
ESSENTIAL_MATCH_STATS: list[str] = [
    # Identity
    "player_id",
    "match_id",

    # Playing time
    "minutes_played",
    "start_min",
    "finish_min",

    # Attacking stats
    "goals",
    "assists",
    "xg",
    "xa",
    "xgot",
    "total_shots",
    "shots_on_target",
    "big_chances_missed",
    "successful_dribbles",
    "touches_opposition_box",
    "touches",
    "chances_created",
    "penalties_scored",
    "penalties_missed",

    # Defensive stats (NEW RULES!)
    "tackles",
    "interceptions",
    "clearances",
    "blocks",
    "recoveries",
    "tackles_won",
    "headed_clearances",

    # Duel stats
    "duels_won",
    "duels_lost",
    "ground_duels_won",
    "aerial_duels_won",

    # GK-specific
    "saves",
    "goals_conceded",
    "xgot_faced",
    "goals_prevented",
    "high_claim",
    "sweeper_actions",
    "gk_accurate_passes",
    "gk_accurate_long_balls",

    # Team context
    "team_goals_conceded",

    # Passing
    "accurate_passes",
    "accurate_passes_percent",
    "final_third_passes",
    "accurate_crosses",
    "accurate_long_balls",
]


# Player Stats (playerstats.csv)
ESSENTIAL_PLAYER_STATS: list[str] = [
    # Identity
    "id",  # player_id
    "gw",

    # Price & ownership
    "now_cost",
    "selected_by_percent",

    # Actual FPL points
    "total_points",
    "event_points",
    "points_per_game",
    "bonus",
    "bps",

    # Form indicators
    "form",

    # Underlying stats (cumulative)
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",

    # Per-90 stats
    "expected_goals_per_90",
    "expected_assists_per_90",
    "expected_goal_involvements_per_90",
    "expected_goals_conceded_per_90",

    # ICT Index
    "influence",
    "creativity",
    "threat",
    "ict_index",

    # Set pieces
    "corners_and_indirect_freekicks_order",
    "direct_freekicks_order",
    "penalties_order",
    "set_piece_threat",

    # NEW 25-26 columns
    "minutes",
    "goals_scored",
    "clean_sheets",
    "saves",
    "defensive_contribution",
    "tackles",
    "clearances_blocks_interceptions",
    "saves_per_90",
    "clean_sheets_per_90",
    "goals_conceded_per_90",
    "defensive_contribution_per_90",
]


# Team Stats (teams.csv)
ESSENTIAL_TEAM_STATS: list[str] = [
    "id",
    "name",
    "short_name",

    # NOTE: `code` is required for joining to fixtures.csv via team_code.
    "code",

    # Strength ratings
    "strength",
    "strength_overall_home",
    "strength_overall_away",
    "strength_attack_home",
    "strength_attack_away",
    "strength_defence_home",
    "strength_defence_away",

    # Team quality rating
    "elo",
]


# 7e1 OPTIONAL - Useful but Not Critical
OPTIONAL_STATS: list[str] = [
    "first_name",
    "second_name",
    "web_name",

    # Status indicators
    "status",
    "chance_of_playing_next_round",
    "chance_of_playing_this_round",

    # Transfer activity
    "transfers_in",
    "transfers_in_event",
    "transfers_out",
    "transfers_out_event",

    # Value metrics
    "value_form",
    "value_season",

    # Rankings (derived)
    "points_per_game_rank",
    "influence_rank",
    "creativity_rank",
    "threat_rank",
    "ict_index_rank",

    # Match-level context
    "was_fouled",
    "fouls_committed",
    "offsides",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "penalties_saved",
]


# 534 DROP - Not Useful for Training
DROP_COLUMNS: set[str] = {
    # Pure identifiers (not features)
    "match_id",  # keep for merging when needed
    "player_code",
    "team_code",
    "pulse_id",
    "fotmob_name",

    # Leaky features
    "ep_next",
    "ep_this",

    # Derived rankings
    "now_cost_rank",
    "now_cost_rank_type",
    "selected_rank",
    "selected_rank_type",
    "form_rank",
    "form_rank_type",
    "influence_rank_type",
    "creativity_rank_type",
    "threat_rank_type",
    "ict_index_rank_type",
    "points_per_game_rank_type",

    # Event-specific metadata
    "cost_change_event",
    "cost_change_event_fall",
    "cost_change_start",
    "cost_change_start_fall",

    # Percentage versions
    "accurate_passes_percent",
    "accurate_crosses_percent",
    "accurate_long_balls_percent",
    "ground_duels_won_percent",
    "aerial_duels_won_percent",
    "successful_dribbles_percent",
    "tackles_won_percent",

    # Text descriptions
    "news",
    "news_added",
    "corners_and_indirect_freekicks_text",
    "direct_freekicks_text",
    "penalties_text",

    # Gameweek metadata
    "deadline_time",
    "deadline_time_epoch",
    "deadline_time_game_offset",
    "average_entry_score",
    "finished",
    "data_checked",
    "kickoff_time",

    # Match URLs and metadata
    "match_url",
    "snapshot_time",
    "release_time",

    # Dream team
    "dreamteam_count",

    # Cup-related
    "cup_leagues_created",
    "h2h_ko_matches_created",

    # Flags
    "is_previous",
    "is_current",
    "is_next",
    "can_enter",
    "can_manage",
    "released",
    "overrides",
}


def select_insights_columns(
    df: pd.DataFrame,
    *,
    keep: Iterable[str],
    always_keep: Iterable[str] = (),
    drop: set[str] = DROP_COLUMNS,
    context: str = "",
) -> pd.DataFrame:
    """Best-effort column selector for Insights CSVs.

    - Keeps any columns present in `keep` excluding those in `drop`.
    - Forces columns in `always_keep` to be included if present, even if in `drop`.
    - Never raises on missing columns.
    """

    keep_list = list(dict.fromkeys(keep))
    always_keep_list = list(dict.fromkeys(always_keep))

    out_cols: list[str] = []

    # 1) Always-keep join keys first
    for c in always_keep_list:
        if c in df.columns and c not in out_cols:
            out_cols.append(c)

    # 2) Then keep-list, excluding dropped unless explicitly always_kept
    always_keep_set = set(always_keep_list)
    for c in keep_list:
        if c in df.columns and c not in out_cols:
            if c in drop and c not in always_keep_set:
                continue
            out_cols.append(c)

    if not out_cols:
        # Return empty-but-valid frame (preserve index)
        return df.iloc[:, 0:0].copy()

    missing = [c for c in keep_list if c not in df.columns]
    if context and missing:
        # Keep this as a print so CLI runs show it; Streamlit callers can ignore.
        print(f"Warning: missing expected columns in {context}: {missing[:12]}" + (" ..." if len(missing) > 12 else ""))

    return df[out_cols].copy()

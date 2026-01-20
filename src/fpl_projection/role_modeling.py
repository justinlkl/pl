from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


ROLE_GK = "GK"
ROLE_DEF = "DEF"
ROLE_MID = "MID"
ROLE_FWD = "FWD"
ROLE_MID_DM = "MID_DM"
ROLE_MID_AM = "MID_AM"


def _norm_pos(v: Any) -> str:
    return str(v or "").strip().lower()


def position_to_role(position: Any) -> str:
    s = _norm_pos(position)
    if s in {"goalkeeper", "gk"}:
        return ROLE_GK
    if s in {"defender", "def", "df"}:
        return ROLE_DEF
    if s in {"midfielder", "mid", "mf"}:
        return ROLE_MID
    if s in {"forward", "fwd", "fw", "striker"}:
        return ROLE_FWD
    # Fall back to MID behavior for unknown outfield roles.
    return ROLE_MID


def infer_mid_subrole_from_window(window: pd.DataFrame) -> str:
    """Heuristic MID split: DM/CM vs AM/Wing.

    Uses last-timestep engineered metrics (per-90 defensive vs attacking involvement).
    """
    if window.empty:
        return ROLE_MID_AM

    last = window.iloc[-1]

    def _num(key: str) -> float:
        return float(pd.to_numeric(last.get(key, np.nan), errors="coerce"))

    tackles = np.nan_to_num(_num("tackles_per_90"), nan=0.0)
    cbi = np.nan_to_num(_num("cbi_per_90"), nan=0.0)
    defcon = np.nan_to_num(_num("defcon_actions_per_90"), nan=0.0)
    defense = tackles + cbi + 0.5 * defcon

    xgi90 = np.nan_to_num(_num("expected_goal_involvements_per_90"), nan=0.0)
    threat = np.nan_to_num(_num("threat"), nan=0.0)
    creativity = np.nan_to_num(_num("creativity"), nan=0.0)
    attack = xgi90 + 0.005 * (threat + creativity)

    # If defense dominates attacking involvement, treat as DM/CM.
    if defense >= max(attack * 1.25, 0.6):
        return ROLE_MID_DM
    return ROLE_MID_AM


def infer_role_from_window(position: Any, window: pd.DataFrame, *, mid_split: bool) -> str:
    base = position_to_role(position)
    if base != ROLE_MID or not mid_split:
        return base
    return infer_mid_subrole_from_window(window)


# Role-aware weights applied AFTER preprocessing (standardization).
# Where a feature name is absent, multiplier defaults to 1.0.
ROLE_WEIGHTS: dict[str, dict[str, float]] = {
    ROLE_GK: {
        # Defensive/context
        "expected_goals_conceded_per_90": 1.0,
        "expected_goals_conceded": 1.0,
        # Attacking is mostly irrelevant
        "expected_goal_involvements_per_90": 0.2,
        "expected_goals_per_90": 0.2,
        "expected_assists_per_90": 0.2,
        "threat": 0.2,
        "creativity": 0.2,
        "influence": 0.3,
        "ict_index": 0.3,
        # Market/availability
        "now_cost_log1p": 0.25,
        "selected_by_percent_log1p": 0.4,
        "chance_of_playing_next_round": 1.0,
        "chance_of_playing_this_round": 1.0,
        "ep_next": 0.8,
        "ep_this": 0.8,
        # Defensive channels if present
        "defcon_points_gk": 1.0,
        "defcon_actions_per_90_gk": 1.0,
        "cbi_per_90_gk": 1.0,
        "tackles_per_90_gk": 1.0,
        "rolling_5_defensive_gk": 1.0,
    },
    ROLE_DEF: {
        "defcon_points_def": 1.2,
        "defcon_actions_per_90_def": 1.0,
        "cbi_per_90_def": 1.2,
        "tackles_per_90_def": 1.0,
        "rolling_5_defensive_def": 1.1,
        "expected_goals_conceded_per_90": 1.0,
        "expected_goals_conceded": 1.0,
        # Attack secondary
        "expected_goal_involvements_per_90": 0.5,
        "expected_goals_per_90": 0.5,
        "expected_assists_per_90": 0.4,
        "threat": 0.4,
        "creativity": 0.4,
        "influence": 0.7,
        "ict_index": 0.6,
        # Market/availability
        "now_cost_log1p": 0.25,
        "selected_by_percent_log1p": 0.4,
        "chance_of_playing_next_round": 1.0,
        "chance_of_playing_this_round": 1.0,
        "ep_next": 0.8,
        "ep_this": 0.8,
    },
    ROLE_MID_DM: {
        "defcon_points_mid": 0.9,
        "defcon_actions_per_90_mid": 0.9,
        "cbi_per_90_mid": 0.8,
        "tackles_per_90_mid": 0.9,
        "rolling_5_defensive_mid": 0.9,
        "expected_goal_involvements_per_90": 0.65,
        "expected_goals_per_90": 0.6,
        "expected_assists_per_90": 0.7,
        "threat": 0.6,
        "creativity": 0.7,
        "influence": 0.9,
        "ict_index": 0.8,
        "now_cost_log1p": 0.25,
        "selected_by_percent_log1p": 0.4,
        "chance_of_playing_next_round": 1.0,
        "chance_of_playing_this_round": 1.0,
        "ep_next": 0.8,
        "ep_this": 0.8,
    },
    ROLE_MID_AM: {
        # Strongly down-weight defensive for AM/Wing
        "defcon_points_mid": 0.15,
        "defcon_actions_per_90_mid": 0.2,
        "cbi_per_90_mid": 0.15,
        "tackles_per_90_mid": 0.2,
        "rolling_5_defensive_mid": 0.15,
        # Attack prioritized
        "expected_goal_involvements_per_90": 1.2,
        "expected_goals_per_90": 1.2,
        "expected_assists_per_90": 1.0,
        "threat": 1.0,
        "creativity": 1.0,
        "influence": 0.9,
        "ict_index": 0.9,
        # Regression signals
        "gi_minus_xgi": 0.6,
        "xgi_underperformance": 0.5,
        "xgi_overperformance": -0.3,
        "goals_minus_xg": 0.4,
        "assists_minus_xa": 0.4,
        # Market/availability
        "now_cost_log1p": 0.25,
        "selected_by_percent_log1p": 0.4,
        "chance_of_playing_next_round": 1.0,
        "chance_of_playing_this_round": 1.0,
        "ep_next": 0.9,
        "ep_this": 0.9,
    },
    ROLE_MID: {
        # If not splitting mids, use a conservative midpoint.
        "defcon_points_mid": 0.4,
        "defcon_actions_per_90_mid": 0.4,
        "cbi_per_90_mid": 0.35,
        "tackles_per_90_mid": 0.4,
        "rolling_5_defensive_mid": 0.35,
        "expected_goal_involvements_per_90": 1.0,
        "expected_goals_per_90": 1.0,
        "expected_assists_per_90": 0.9,
        "threat": 0.9,
        "creativity": 0.9,
        "ict_index": 0.85,
        "now_cost_log1p": 0.25,
        "selected_by_percent_log1p": 0.4,
        "chance_of_playing_next_round": 1.0,
        "chance_of_playing_this_round": 1.0,
        "ep_next": 0.85,
        "ep_this": 0.85,
    },
    ROLE_FWD: {
        # Down-weight defensive
        "defcon_points_fwd": 0.1,
        "defcon_actions_per_90_fwd": 0.15,
        "cbi_per_90_fwd": 0.1,
        "tackles_per_90_fwd": 0.15,
        "rolling_5_defensive_fwd": 0.1,
        # Attack prioritized
        "expected_goal_involvements_per_90": 1.25,
        "expected_goals_per_90": 1.25,
        "expected_assists_per_90": 0.7,
        "threat": 1.0,
        "creativity": 0.5,
        "influence": 0.7,
        "ict_index": 0.9,
        # Regression
        "gi_minus_xgi": 0.6,
        "xgi_underperformance": 0.5,
        "xgi_overperformance": -0.35,
        "goals_minus_xg": 0.4,
        "assists_minus_xa": 0.3,
        # Market/availability
        "now_cost_log1p": 0.25,
        "selected_by_percent_log1p": 0.4,
        "chance_of_playing_next_round": 1.0,
        "chance_of_playing_this_round": 1.0,
        "ep_next": 0.95,
        "ep_this": 0.95,
    },
}


@dataclass(frozen=True)
class RoleSpec:
    role: str
    feature_weights: np.ndarray


def build_feature_weight_vector(feature_columns: list[str], role: str) -> np.ndarray:
    weights = np.ones(len(feature_columns), dtype=float)
    role_map = ROLE_WEIGHTS.get(role, {})
    for i, col in enumerate(feature_columns):
        if col in role_map:
            weights[i] = float(role_map[col])
    return weights


def list_roles(*, mid_split: bool) -> list[str]:
    if mid_split:
        return [ROLE_GK, ROLE_DEF, ROLE_MID_DM, ROLE_MID_AM, ROLE_FWD]
    return [ROLE_GK, ROLE_DEF, ROLE_MID, ROLE_FWD]

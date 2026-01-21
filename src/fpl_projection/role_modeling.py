from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROLE_GK = "GK"
ROLE_DEF = "DEF"
ROLE_MID = "MID"
ROLE_FWD = "FWD"
ROLE_MID_DM = "MID_DM"
ROLE_MID_AM = "MID_AM"


# Post-model calibration multipliers (applied to projected points AFTER prediction/stacking).
# These help align the absolute scale and positional distributions with commercial tools.
ROLE_PROJECTION_MULTIPLIER: dict[str, float] = {
    ROLE_FWD: 1.25,
    ROLE_MID_AM: 1.15,
    ROLE_MID: 1.10,
    ROLE_MID_DM: 0.70,
    ROLE_DEF: 1.05,
    ROLE_GK: 1.00,
}


ROLE_PROJECTION_MULTIPLIER_BOUNDS: dict[str, tuple[float, float]] = {
    ROLE_FWD: (0.7, 1.8),
    ROLE_MID_AM: (0.6, 1.6),
    ROLE_MID: (0.6, 1.6),
    ROLE_MID_DM: (0.3, 1.2),
    ROLE_DEF: (0.6, 1.5),
    ROLE_GK: (0.6, 1.4),
}


# Role-aware loss weights (applied during training via per-sample `sample_weight`).
# Principle: punish attacker mispredictions more; suppress MID-DM stability from dominating.
ROLE_LOSS_WEIGHTS: dict[str, float] = {
    ROLE_FWD: 1.35,
    ROLE_MID_AM: 1.25,
    ROLE_MID: 1.05,
    ROLE_MID_DM: 0.55,
    ROLE_DEF: 0.95,
    ROLE_GK: 0.85,
}


def role_loss_weight(role: Any, *, overrides: dict[str, float] | None = None) -> float:
    r = str(role or "").strip()
    if overrides and r in overrides:
        try:
            return float(overrides[r])
        except Exception:
            return 1.0
    return float(ROLE_LOSS_WEIGHTS.get(r, 1.0))


def role_projection_multiplier(role: Any, *, overrides: dict[str, float] | None = None) -> float:
    r = str(role or "").strip()
    if overrides and r in overrides:
        try:
            return float(overrides[r])
        except Exception:
            return 1.0
    return float(ROLE_PROJECTION_MULTIPLIER.get(r, 1.0))


def scale_projection_matrix(
    preds: np.ndarray,
    roles: list[str] | np.ndarray,
    *,
    overrides: dict[str, float] | None = None,
) -> np.ndarray:
    """Scale an (n_players, horizon) prediction matrix by per-player role multipliers."""
    roles_arr = np.asarray(roles, dtype=object)
    mult = np.asarray([role_projection_multiplier(r, overrides=overrides) for r in roles_arr], dtype=float)
    return preds * mult.reshape(-1, 1)


def _to_totals(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape(-1)
    return x.reshape(x.shape[0], -1).sum(axis=1)


def fit_role_projection_multipliers(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    player_id: np.ndarray,
    roles: np.ndarray,
    bounds: dict[str, tuple[float, float]] | None = None,
    grid_step: float = 0.01,
    min_count: int = 25,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Fit per-role post-model scaling multipliers.

    We fit a scalar multiplier per role that minimizes MAE on player-level totals.
    This is a pragmatic calibration step to align positional distributions to
    recent historical accuracy.

    Returns:
      overrides: {role: multiplier}
      report: DataFrame with counts + mae before/after per role
    """

    b = bounds or ROLE_PROJECTION_MULTIPLIER_BOUNDS

    true_total = _to_totals(np.asarray(y_true, dtype=float))
    pred_total = _to_totals(np.asarray(y_pred, dtype=float))

    df = pd.DataFrame(
        {
            "player_id": np.asarray(player_id).astype(int),
            "role": np.asarray(roles, dtype=object).astype(str),
            "true_total": true_total,
            "pred_total": pred_total,
        }
    )

    def _mode(series: pd.Series) -> str:
        vc = series.value_counts(dropna=False)
        return str(vc.index[0]) if not vc.empty else ""

    by_player = (
        df.groupby("player_id", as_index=False)
        .agg(true_total=("true_total", "mean"), pred_total=("pred_total", "mean"), role=("role", _mode))
        .reset_index(drop=True)
    )

    overrides: dict[str, float] = {}
    rows: list[dict[str, object]] = []

    for role, g in by_player.groupby("role"):
        role = str(role)
        if int(len(g)) < int(min_count):
            continue

        lo, hi = b.get(role, (0.5, 1.5))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = (0.5, 1.5)

        yt = g["true_total"].to_numpy(dtype=float)
        yp = g["pred_total"].to_numpy(dtype=float)
        mask = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[mask]
        yp = yp[mask]
        if yt.size < int(min_count):
            continue

        base_mae = float(np.mean(np.abs(yp - yt)))
        if float(np.mean(np.abs(yp))) < 1e-9:
            continue

        n = int(max(2, np.floor((hi - lo) / float(grid_step)) + 1))
        grid = np.linspace(float(lo), float(hi), n, dtype=float)

        best_s = 1.0
        best_mae = float("inf")
        for s in grid:
            mae = float(np.mean(np.abs(s * yp - yt)))
            if mae < best_mae:
                best_mae = mae
                best_s = float(s)

        overrides[role] = float(best_s)
        rows.append(
            {
                "role": role,
                "count": int(yt.size),
                "mae_before": float(base_mae),
                "mae_after": float(best_mae),
                "multiplier": float(best_s),
            }
        )

    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values(["role"]).reset_index(drop=True)
    return overrides, report


def save_role_scaling(
    path: Path,
    *,
    overrides: dict[str, float],
    report: pd.DataFrame | None = None,
    meta: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "overrides": {k: float(v) for k, v in overrides.items()},
    }
    if report is not None and not report.empty:
        payload["report"] = report.to_dict(orient="records")
    if meta:
        payload["meta"] = meta
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_role_scaling(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("overrides"), dict):
        out: dict[str, float] = {}
        for k, v in payload["overrides"].items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out
    return None


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

        # Bonus proxy
        "bps_bonus_proxy": 0.6,
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

        # Bonus proxy
        "bps_bonus_proxy": 0.65,
        # Market/availability
        "now_cost_log1p": 0.25,
        "selected_by_percent_log1p": 0.4,
        "chance_of_playing_next_round": 1.0,
        "chance_of_playing_this_round": 1.0,
        "ep_next": 0.8,
        "ep_this": 0.8,
    },
    ROLE_MID_DM: {
        # Defensive stability should not dominate MID scoring.
        "defcon_points_mid": 0.2,
        "defcon_actions_per_90_mid": 0.2,
        "cbi_per_90_mid": 0.15,
        "tackles_per_90_mid": 0.2,
        "rolling_5_defensive_mid": 0.2,

        # Attacking still matters, but DM should be moderated.
        "expected_goal_involvements_per_90": 0.8,
        "expected_goals_per_90": 0.8,
        "expected_assists_per_90": 0.9,
        "expected_goals_conceded_per_90": 0.1,
        "threat": 0.75,
        "creativity": 0.85,
        "influence": 0.85,
        "ict_index": 0.8,

        # Bonus proxy
        "bps_bonus_proxy": 0.45,

        # Reduce "minutes consistency" dominance
        "minutes": 0.35,
        "starts": 0.40,
        "rolling_5_minutes": 0.35,
        "rolling_5_points": 0.6,
        "form": 1.3,
        "value_season": 0.7,
        "now_cost_log1p": 0.25,
        "selected_by_percent_log1p": 0.4,
        "chance_of_playing_next_round": 1.0,
        "chance_of_playing_this_round": 1.0,
        "ep_next": 0.8,
        "ep_this": 0.8,
    },
    ROLE_MID_AM: {
        # Strongly down-weight defensive for AM/Wing
        "defcon_points_mid": 0.05,
        "defcon_actions_per_90_mid": 0.08,
        "cbi_per_90_mid": 0.06,
        "tackles_per_90_mid": 0.08,
        "rolling_5_defensive_mid": 0.06,
        # Attack prioritized
        "expected_goal_involvements_per_90": 1.3,
        "expected_goals_per_90": 1.4,
        "expected_assists_per_90": 1.2,
        "expected_goals_conceded_per_90": 0.05,
        "threat": 1.1,
        "creativity": 1.1,
        "influence": 0.9,
        "ict_index": 0.9,

        # Bonus proxy
        "bps_bonus_proxy": 0.6,
        "minutes": 0.5,
        "starts": 0.6,
        "rolling_5_minutes": 0.45,
        "rolling_5_points": 0.8,
        "form": 1.3,
        "value_season": 0.75,
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
        "defcon_points_mid": 0.1,
        "defcon_actions_per_90_mid": 0.12,
        "cbi_per_90_mid": 0.1,
        "tackles_per_90_mid": 0.12,
        "rolling_5_defensive_mid": 0.1,
        "expected_goal_involvements_per_90": 1.15,
        "expected_goals_per_90": 1.2,
        "expected_assists_per_90": 1.05,
        "expected_goals_conceded_per_90": 0.08,
        "threat": 1.0,
        "creativity": 1.0,
        "ict_index": 0.85,

        # Bonus proxy
        "bps_bonus_proxy": 0.55,
        "minutes": 0.45,
        "starts": 0.55,
        "rolling_5_minutes": 0.45,
        "rolling_5_points": 0.75,
        "form": 1.2,
        "value_season": 0.75,
        "now_cost_log1p": 0.25,
        "selected_by_percent_log1p": 0.4,
        "chance_of_playing_next_round": 1.0,
        "chance_of_playing_this_round": 1.0,
        "ep_next": 0.85,
        "ep_this": 0.85,
    },
    ROLE_FWD: {
        # Down-weight defensive
        "defcon_points_fwd": 0.05,
        "defcon_actions_per_90_fwd": 0.06,
        "cbi_per_90_fwd": 0.05,
        "tackles_per_90_fwd": 0.06,
        "rolling_5_defensive_fwd": 0.05,
        # Attack prioritized
        "expected_goal_involvements_per_90": 1.4,
        "expected_goals_per_90": 1.2,
        "expected_assists_per_90": 0.9,
        "expected_goals_conceded_per_90": 0.05,
        "threat": 1.0,
        "creativity": 0.5,
        "influence": 0.7,
        "ict_index": 0.9,

        # Bonus proxy
        "bps_bonus_proxy": 0.55,
        "minutes": 0.7,
        "starts": 0.8,
        "rolling_5_minutes": 0.65,
        "rolling_5_points": 0.8,
        "form": 1.2,
        "value_season": 0.75,
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

"""Feature engineering for FPL ML model.

Handles calculation of derived features including:
- Per-90 normalization
- Rolling aggregates
- Cumulative statistics
- Performance vs expectation metrics
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def calculate_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple one-hot position flags when `position` is available.

    FPL-Core-Insights provides `position` in the season-level players.csv, which we
    merge into the per-GW stats in data_loading.py.
    """
    df = df.copy()
    pos = df.get("position")
    if pos is None:
        df["pos_gk"] = 0.0
        df["pos_def"] = 0.0
        df["pos_mid"] = 0.0
        df["pos_fwd"] = 0.0
        return df

    pos_norm = pos.astype(str).str.strip().str.lower()
    df["pos_gk"] = pos_norm.isin(["goalkeeper", "gk"]).astype(float)
    df["pos_def"] = pos_norm.isin(["defender", "def", "df"]).astype(float)
    df["pos_mid"] = pos_norm.isin(["midfielder", "mid", "mf"]).astype(float)
    df["pos_fwd"] = pos_norm.isin(["forward", "fwd", "fw", "striker"]).astype(float)
    return df


def calculate_defensive_contribution_points(df: pd.DataFrame) -> pd.DataFrame:
    """Compute FPL 2025/26 Defensive Contributions *points* (0 or 2) with cap.

    Official rules (2025/26):
    - DEF: 2 pts for 10+ CBIT
    - MID/FWD: 2 pts for 12+ CBIRT
    - Cap: 2 pts per match

    The dataset often includes a `defensive_contribution` count already (from the
    official API). We convert actions -> points to prevent the model from over-weighting
    raw action volume beyond the scoring cap.
    """
    df = df.copy()

    # Best-effort actions count
    if "defensive_contribution" in df.columns:
        actions = pd.to_numeric(df["defensive_contribution"], errors="coerce").fillna(0.0)
    else:
        tackles = pd.to_numeric(df.get("tackles", 0), errors="coerce").fillna(0.0)
        cbi = pd.to_numeric(df.get("clearances_blocks_interceptions", 0), errors="coerce").fillna(0.0)
        recoveries = pd.to_numeric(df.get("recoveries", 0), errors="coerce").fillna(0.0)
        actions = tackles + cbi + recoveries

    # Position-aware threshold
    pos_norm = df.get("position", pd.Series(["" for _ in range(len(df))])).astype(str).str.strip().str.lower()
    is_gk = pos_norm.isin(["goalkeeper", "gk"])
    is_def = pos_norm.isin(["defender", "def", "df"])
    is_mid_fwd = ~(is_gk | is_def)

    # GK are not listed as eligible in the provided rules.
    threshold = np.where(is_def.to_numpy(), 10.0, np.where(is_mid_fwd.to_numpy(), 12.0, np.inf))

    df["defcon_actions"] = actions
    df["defcon_points"] = ((actions.to_numpy() >= threshold) & np.isfinite(threshold)).astype(float) * 2.0
    return df


def calculate_per_90_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate/normalize per-90-minute metrics used by the model.

    The FPL-Core-Insights dataset already includes some per-90 columns
    (e.g., expected_goals_per_90). We preserve those, and only compute
    additional per-90 defensive rates that aren't present.
    """
    df = df.copy()
    
    # Avoid division by zero
    minutes_played = df["minutes"].fillna(0)
    minutes_per_90 = np.maximum(minutes_played / 90, 0.01)  # Min 0.01 to avoid div by zero
    
    # Per-90 defensive metrics (not provided directly in the dataset)
    if "tackles" in df.columns:
        df["tackles_per_90"] = df["tackles"].fillna(0) / minutes_per_90
    else:
        df["tackles_per_90"] = 0.0

    if "clearances_blocks_interceptions" in df.columns:
        df["cbi_per_90"] = df["clearances_blocks_interceptions"].fillna(0) / minutes_per_90
    else:
        df["cbi_per_90"] = 0.0
    
    # Defensive actions per 90 (kept for context, but prefer capped defcon_points in modeling)
    if "defcon_actions" in df.columns:
        df["defcon_actions_per_90"] = pd.to_numeric(df["defcon_actions"], errors="coerce").fillna(0.0) / minutes_per_90
    elif "tackles" in df.columns and "clearances_blocks_interceptions" in df.columns:
        def_total = df["tackles"].fillna(0) + df["clearances_blocks_interceptions"].fillna(0)
        df["defcon_actions_per_90"] = def_total / minutes_per_90
    else:
        df["defcon_actions_per_90"] = 0.0
    
    # Combined tackles + cbi per 90
    if "tackles" in df.columns and "clearances_blocks_interceptions" in df.columns:
        cbit_total = df["tackles"].fillna(0) + df["clearances_blocks_interceptions"].fillna(0)
        df["cbit_per_90"] = cbit_total / minutes_per_90
    else:
        df["cbit_per_90"] = 0.0
    
    return df


def calculate_availability_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute availability features like minutes_per_start."""
    df = df.copy()

    if "minutes" in df.columns and "starts" in df.columns:
        starts = df["starts"].fillna(0)
        df["minutes_per_start"] = df["minutes"].fillna(0) / np.maximum(starts, 1)
    else:
        df["minutes_per_start"] = 0.0

    return df


def calculate_performance_vs_expectation(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate over/underperformance metrics vs expected stats.
    
    Args:
        df: DataFrame with actual and expected statistics
        
    Returns:
        DataFrame with performance difference columns
    """
    df = df.copy()
    
    # Goal involvements (goals + assists)
    goals = df.get("goals_scored", 0).fillna(0)
    assists = df.get("assists", 0).fillna(0)
    df["goal_involvements"] = goals + assists
    
    # Expected goal involvements
    xg = df.get("expected_goals", 0).fillna(0)
    xa = df.get("expected_assists", 0).fillna(0)
    xgi = xg + xa
    
    # Actual vs Expected
    df["goals_minus_xg"] = goals - xg
    df["assists_minus_xa"] = assists - xa
    df["gi_minus_xgi"] = df["goal_involvements"] - xgi
    
    # Classification of over/underperformance
    df["xgi_overperformance"] = np.maximum(df["gi_minus_xgi"], 0)
    df["xgi_underperformance"] = np.minimum(df["gi_minus_xgi"], 0)
    
    # Expected goal/assist combinations
    df["xg_plus_xa"] = xg + xa
    
    return df


def calculate_rolling_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Calculate rolling aggregate features per player.
    
    Args:
        df: DataFrame with player gameweek stats (must be sorted by player_id, gw)
        windows: List of window sizes for rolling calculations. Default: [3, 5]
        
    Returns:
        DataFrame with rolling feature columns added
    """
    if windows is None:
        windows = [3, 5]
    
    df = df.copy()
    
    # Ensure sorted by player and gameweek for proper rolling
    df = df.sort_values(["player_id", "gw"]).reset_index(drop=True)
    
    for window in windows:
        prefix = f"rolling_{window}"
        
        # Group by player and calculate rolling stats
        for player_id, group in df.groupby("player_id", sort=False):
            player_mask = df["player_id"] == player_id
            
            # xG rolling
            if "expected_goals" in df.columns:
                df.loc[player_mask, f"{prefix}_xg"] = (
                    group["expected_goals"].fillna(0).rolling(window=window, min_periods=1).sum().values
                )
            
            # xA rolling
            if "expected_assists" in df.columns:
                df.loc[player_mask, f"{prefix}_xa"] = (
                    group["expected_assists"].fillna(0).rolling(window=window, min_periods=1).sum().values
                )
            
            # xGI rolling
            if "expected_goal_involvements" in df.columns:
                df.loc[player_mask, f"{prefix}_xgi"] = (
                    group["expected_goal_involvements"].fillna(0).rolling(window=window, min_periods=1).sum().values
                )
            
            # Points rolling (only for window=5, not 3)
            if window == 5 and "total_points" in df.columns:
                df.loc[player_mask, f"{prefix}_points"] = (
                    group["total_points"].fillna(0).rolling(window=window, min_periods=1).sum().values
                )
            
            # Minutes rolling (only for window=5, not 3)
            if window == 5 and "minutes" in df.columns:
                df.loc[player_mask, f"{prefix}_minutes"] = (
                    group["minutes"].fillna(0).rolling(window=window, min_periods=1).sum().values
                )
            
            # Defensive rolling (only for window=5)
            if window == 5 and "clearances_blocks_interceptions" in df.columns:
                df.loc[player_mask, f"{prefix}_defensive"] = (
                    group["clearances_blocks_interceptions"].fillna(0).rolling(window=window, min_periods=1).sum().values
                )
    
    # Fill any NaNs from rolling calculations with 0
    rolling_cols = [c for c in df.columns if c.startswith("rolling_")]
    df[rolling_cols] = df[rolling_cols].fillna(0)
    
    return df


def calculate_cumulative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate season cumulative statistics per player.
    
    Args:
        df: DataFrame with player gameweek stats (must be sorted by player_id, gw)
        
    Returns:
        DataFrame with cumulative feature columns added
    """
    df = df.copy()
    
    # Ensure sorted by player and gameweek
    df = df.sort_values(["player_id", "gw"]).reset_index(drop=True)
    
    # Group by player and calculate cumulative stats
    for player_id, group in df.groupby("player_id", sort=False):
        player_mask = df["player_id"] == player_id
        
        # Cumulative xG
        if "expected_goals" in df.columns:
            df.loc[player_mask, "cumulative_xg"] = (
                group["expected_goals"].fillna(0).cumsum().values
            )
        
        # Cumulative xA
        if "expected_assists" in df.columns:
            df.loc[player_mask, "cumulative_xa"] = (
                group["expected_assists"].fillna(0).cumsum().values
            )
        
        # Cumulative xGI
        if "expected_goal_involvements" in df.columns:
            df.loc[player_mask, "cumulative_xgi"] = (
                group["expected_goal_involvements"].fillna(0).cumsum().values
            )
    
    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering transformations.
    
    This is the complete pipeline:
    1. Per-90 minute normalization
    2. Performance vs expectation metrics
    3. Rolling aggregates (3 and 5 gameweek windows)
    4. Cumulative season statistics
    
    Args:
        df: Raw player gameweek statistics DataFrame
        
    Returns:
        DataFrame with all engineered features
    """
    df = df.copy()
    
    # Sort for groupby operations
    df = df.sort_values(["player_id", "gw"]).reset_index(drop=True)
    
    # Apply transformations in sequence
    df = calculate_position_features(df)
    df = calculate_availability_features(df)
    df = calculate_defensive_contribution_points(df)
    df = calculate_per_90_metrics(df)
    df = calculate_performance_vs_expectation(df)
    df = calculate_rolling_features(df, windows=[3, 5])
    df = calculate_cumulative_features(df)
    
    return df


def get_inference_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for inference (excludes cumulative features for current gameweek).
    
    During inference, we only have the current gameweek and a few recent ones,
    so cumulative features may not be as reliable. This returns the core + performance
    + rolling features suitable for prediction.
    
    Args:
        df: DataFrame with engineered features
        
    Returns:
        DataFrame with inference-safe features
    """
    from .config import CORE_PREDICTORS, PERFORMANCE_ENHANCERS, ROLLING_FEATURES
    
    df = df.copy()
    
    # Use only the most recent features that will be available at prediction time
    inference_cols = CORE_PREDICTORS + PERFORMANCE_ENHANCERS + ROLLING_FEATURES
    
    # Keep only columns that exist in the data
    available_cols = [c for c in inference_cols if c in df.columns]
    
    return df[available_cols]

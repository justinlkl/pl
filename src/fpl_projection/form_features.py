"""Player form trend detection and momentum features.

Analyzes recent performance patterns to capture momentum, consistency, and
form changes that inform short-term predictions.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def calculate_rolling_form(
    df: pd.DataFrame,
    window: int = 5,
    metric: str = 'total_points',
) -> pd.DataFrame:
    """Calculate rolling average form over recent gameweeks.
    
    Args:
        df: Player gameweek stats with player_id, gw, total_points (or other metric)
        window: Number of past gameweeks to consider
        metric: Column name to compute rolling average for
        
    Returns:
        DataFrame with rolling_form_{window}gw column
    """
    
    df = df.copy()
    
    # Sort by player and GW to ensure proper ordering
    df = df.sort_values(['player_id', 'gw'])
    
    # Compute rolling average within each player
    col_name = f'rolling_form_{window}gw'
    df[col_name] = (
        df.groupby('player_id')[metric]
        .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    )
    
    return df


def calculate_form_momentum(
    df: pd.DataFrame,
    window: int = 5,
    metric: str = 'total_points',
) -> pd.DataFrame:
    """Calculate momentum (slope) of form trend.
    
    Positive momentum = improving form, negative = declining form.
    
    Args:
        df: Player gameweek stats
        window: Number of past gameweeks to regress
        metric: Column to track momentum on
        
    Returns:
        DataFrame with momentum_slope and momentum_direction columns
    """
    
    df = df.copy()
    df = df.sort_values(['player_id', 'gw'])
    
    def compute_momentum(group):
        """Fit linear trend and extract slope."""
        if len(group) < 2:
            return pd.Series({
                'momentum_slope': 0.0,
                'momentum_direction': 'neutral',
                'momentum_strength': 0.0,
            })
        
        # Get last `window` gameweeks
        recent = group.tail(window)
        x = np.arange(len(recent))
        y = recent[metric].values
        
        # Linear regression: y = slope * x + intercept
        if len(recent) > 1 and np.std(x) > 0:
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = 0.0
        
        # Classify direction
        if slope > 0.1:
            direction = 'improving'
        elif slope < -0.1:
            direction = 'declining'
        else:
            direction = 'stable'
        
        # Strength = absolute slope magnitude
        strength = abs(slope)
        
        return pd.Series({
            'momentum_slope': slope,
            'momentum_direction': direction,
            'momentum_strength': strength,
        })
    
    momentum = df.groupby('player_id', group_keys=False).apply(compute_momentum)
    df = pd.concat([df, momentum], axis=1)
    
    return df


def calculate_form_consistency(
    df: pd.DataFrame,
    window: int = 5,
    metric: str = 'total_points',
) -> pd.DataFrame:
    """Calculate consistency of recent performance (inverse of volatility).
    
    High consistency = low std dev = reliable performer.
    Low consistency = high std dev = inconsistent/boom-bust.
    
    Args:
        df: Player gameweek stats
        window: Number of past gameweeks
        metric: Column to measure consistency on
        
    Returns:
        DataFrame with form_consistency and form_volatility columns
    """
    
    df = df.copy()
    df = df.sort_values(['player_id', 'gw'])
    
    # Compute rolling std (volatility)
    col_name_vol = f'form_volatility_{window}gw'
    col_name_cons = f'form_consistency_{window}gw'
    
    df[col_name_vol] = (
        df.groupby('player_id')[metric]
        .transform(lambda x: x.rolling(window=window, min_periods=1).std())
    )
    
    # Consistency = 1 / (1 + volatility) -> [0, 1] range
    # Higher = more consistent
    df[col_name_cons] = 1.0 / (1.0 + df[col_name_vol].fillna(0))
    
    return df


def identify_form_peaks_valleys(
    df: pd.DataFrame,
    window: int = 5,
    metric: str = 'total_points',
) -> pd.DataFrame:
    """Identify recent peaks and valleys in player performance.
    
    Useful for detecting injury recovery, form breaks, etc.
    
    Args:
        df: Player gameweek stats
        window: Number of past gameweeks
        metric: Column to analyze
        
    Returns:
        DataFrame with peak_gw, valley_gw, peak_to_current, valley_to_current columns
    """
    
    df = df.copy()
    df = df.sort_values(['player_id', 'gw'])
    
    def find_extremes(group):
        """Find most recent peak and valley."""
        if len(group) < 2:
            return pd.Series({
                'peak_gw': None,
                'valley_gw': None,
                'peak_to_current': 0.0,
                'valley_to_current': 0.0,
                'recent_max': 0.0,
                'recent_min': 0.0,
            })
        
        recent = group.tail(window)
        current_gw = recent.iloc[-1]['gw']
        current_val = recent.iloc[-1][metric]
        
        # Find peak (max value) and valley (min value)
        peak_idx = recent[metric].idxmax()
        valley_idx = recent[metric].idxmin()
        
        peak_gw = recent.loc[peak_idx, 'gw']
        valley_gw = recent.loc[valley_idx, 'gw']
        peak_val = recent.loc[peak_idx, metric]
        valley_val = recent.loc[valley_idx, metric]
        
        # How far from current?
        peak_to_current = current_gw - peak_gw
        valley_to_current = current_gw - valley_gw
        
        return pd.Series({
            'peak_gw': peak_gw,
            'valley_gw': valley_gw,
            'peak_to_current': peak_to_current,
            'valley_to_current': valley_to_current,
            'recent_max': peak_val,
            'recent_min': valley_val,
        })
    
    extremes = df.groupby('player_id', group_keys=False).apply(find_extremes)
    df = pd.concat([df, extremes], axis=1)
    
    return df


def integrate_form_features(
    df: pd.DataFrame,
    rolling_windows: list[int] = [3, 5],
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Full integration: add all form-related features to player data.
    
    Combines:
    - Rolling form averages
    - Momentum (trend direction and strength)
    - Consistency (volatility inverse)
    - Peaks/valleys detection
    
    Args:
        df: Player gameweek stats
        rolling_windows: List of window sizes to compute rolling metrics for
        metrics: List of columns to track form on (default: ['total_points'])
        
    Returns:
        DataFrame with all form features merged
    """
    
    if metrics is None:
        metrics = ['total_points']
    
    try:
        df_with_form = df.copy()
        
        # For each metric and window combination
        for metric in metrics:
            for window in rolling_windows:
                # Rolling form
                df_with_form = calculate_rolling_form(
                    df_with_form,
                    window=window,
                    metric=metric,
                )
                
                # Form consistency
                df_with_form = calculate_form_consistency(
                    df_with_form,
                    window=window,
                    metric=metric,
                )
        
        # Momentum (use primary metric, single window)
        primary_metric = metrics[0] if metrics else 'total_points'
        primary_window = rolling_windows[0] if rolling_windows else 5
        df_with_form = calculate_form_momentum(
            df_with_form,
            window=primary_window,
            metric=primary_metric,
        )
        
        # Peaks/valleys (use primary metric)
        df_with_form = identify_form_peaks_valleys(
            df_with_form,
            window=primary_window,
            metric=primary_metric,
        )
        
        logger.info(f"✅ Form features integrated: rolling windows {rolling_windows}, metrics {metrics}")
        return df_with_form
        
    except Exception as e:
        logger.warning(f"⚠️  Form feature integration failed ({e}); continuing without")
        return df


def get_form_summary(df: pd.DataFrame) -> dict:
    """Generate summary statistics of form features in dataset.
    
    Useful for diagnostics and understanding feature distributions.
    
    Args:
        df: DataFrame with form features
        
    Returns:
        Dictionary with form feature statistics
    """
    
    form_cols = [col for col in df.columns if 'form' in col.lower() or 'momentum' in col.lower()]
    
    summary = {}
    for col in form_cols:
        if df[col].dtype in [np.float32, np.float64]:
            summary[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
            }
    
    return summary

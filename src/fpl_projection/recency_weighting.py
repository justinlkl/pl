"""
Recency Weighting Module for FPL ML Models

Implements exponential decay weighting where recent gameweeks get higher weights
than older gameweeks. This is crucial for:
- Recent form being more predictive than distant history
- Adapting to mid-season rule changes
- Prioritizing fresh trends over stale patterns
"""

import numpy as np
import pandas as pd
from typing import Tuple


def compute_recency_weights(
    gws: np.ndarray,
    current_gw: int = None,
    half_life: int = 10,
) -> np.ndarray:
    """Compute exponential decay weights based on gameweek recency.
    
    Older gameweeks get exponentially lower weight.
    Weight formula: 2^(-(current_gw - gw) / half_life)
    
    Args:
        gws: Array of gameweek numbers
        current_gw: Current gameweek (defaults to max in gws)
        half_life: Gameweeks for weight to decay to 0.5
                   Default 10 means GW 10 back has 0.5 weight
                   
    Returns:
        Normalized weights summing to 1.0 per sample
    """
    if current_gw is None:
        current_gw = np.max(gws)
    
    # Exponential decay: weight = 2^(-(age / half_life))
    age = np.maximum(current_gw - gws, 0)
    weights = 2.0 ** (-age / half_life)
    
    return weights


def apply_recency_weights_to_sequences(
    X: np.ndarray,
    gws: np.ndarray,
    current_gw: int = None,
    half_life: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply recency weights to sequence dataset.
    
    For sequences, we weight each sequence by the GW of its final timestep
    (most recent game in that player's sequence).
    
    Args:
        X: Sequence array shape (n_samples, seq_length, n_features)
        gws: Gameweek array shape (n_samples,) - GW of last timestep in sequence
        current_gw: Current gameweek (auto-detect if None)
        half_life: Decay half-life in gameweeks
        
    Returns:
        Tuple of (X, sample_weights) where sample_weights shape (n_samples,)
    """
    weights = compute_recency_weights(gws, current_gw, half_life)
    
    # Normalize to have mean 1.0 (so loss scale doesn't change much)
    weights = weights / np.mean(weights)
    
    return X, weights


def create_gw_based_sample_weights(
    df: pd.DataFrame,
    current_gw: int = None,
    half_life: int = 10,
    gw_col: str = "gw",
) -> np.ndarray:
    """Create sample weights from dataframe with GW column.
    
    Args:
        df: DataFrame with gameweek column
        current_gw: Current gameweek
        half_life: Exponential decay parameter
        gw_col: Name of gameweek column
        
    Returns:
        Sample weights array
    """
    gws = df[gw_col].values
    weights = compute_recency_weights(gws, current_gw, half_life)
    weights = weights / np.mean(weights)
    return weights


def analyze_weight_distribution(
    gws: np.ndarray,
    current_gw: int = None,
    half_life: int = 10,
) -> pd.DataFrame:
    """Analyze how weights are distributed across gameweeks.
    
    Useful for understanding the impact of recency weighting.
    """
    if current_gw is None:
        current_gw = np.max(gws)
    
    weights = compute_recency_weights(gws, current_gw, half_life)
    
    # Group by GW
    unique_gws = sorted(np.unique(gws))
    rows = []
    
    for gw in unique_gws:
        mask = gws == gw
        gw_weights = weights[mask]
        rows.append({
            "gw": int(gw),
            "age": current_gw - gw,
            "count": np.sum(mask),
            "mean_weight": float(np.mean(gw_weights)),
            "min_weight": float(np.min(gw_weights)),
            "max_weight": float(np.max(gw_weights)),
            "total_weight": float(np.sum(gw_weights)),
        })
    
    return pd.DataFrame(rows)


# Recommended parameters for different scenarios
RECENCY_PROFILES = {
    "aggressive": {
        "half_life": 5,
        "description": "Favor last 5 GWs heavily, discount old data",
    },
    "balanced": {
        "half_life": 10,
        "description": "Default: weight halves every 10 GWs",
    },
    "conservative": {
        "half_life": 15,
        "description": "More gradual decay, value historical patterns",
    },
    "training": {
        "half_life": 8,
        "description": "For initial training: balance history with recent form",
    },
    "finetuning": {
        "half_life": 5,
        "description": "For weekly fine-tuning: focus on latest trends",
    },
}


if __name__ == "__main__":
    # Demo: show weight distributions
    import matplotlib.pyplot as plt
    
    # Example: 22 gameweeks of data
    gws = np.arange(1, 23)
    
    print("Recency Weight Profiles")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, (name, params) in enumerate(RECENCY_PROFILES.items()):
        ax = axes[idx]
        half_life = params["half_life"]
        weights = compute_recency_weights(gws, current_gw=22, half_life=half_life)
        
        ax.bar(gws, weights, alpha=0.7, color=f"C{idx}")
        ax.set_title(f"{name.upper()}\n(half_life={half_life} GWs)")
        ax.set_xlabel("Gameweek")
        ax.set_ylabel("Weight")
        ax.grid(True, alpha=0.3)
        
        print(f"\n{name.upper()} (half_life={half_life}):")
        print(f"  {params['description']}")
        
        # Show a few key weights
        print(f"  GW 1 (oldest):  weight = {weights[0]:.3f}")
        print(f"  GW 11 (middle): weight = {weights[10]:.3f}")
        print(f"  GW 22 (latest): weight = {weights[-1]:.3f}")
    
    plt.tight_layout()
    plt.savefig("recency_weights_demo.png", dpi=100)
    print("\n✅ Saved visualization to recency_weights_demo.png")

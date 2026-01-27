"""Uncertainty estimation for point projections.

Provides confidence intervals via Monte Carlo dropout and bootstrap methods
to help users understand prediction reliability.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def predict_with_uncertainty(
    model: tf.keras.Model,
    X: np.ndarray,
    n_simulations: int = 100,
    keep_dropout_active: bool = True,
) -> dict[str, np.ndarray]:
    """Generate predictions with uncertainty bounds via Monte Carlo dropout.
    
    Args:
        model: Trained Keras model (should have Dropout layers)
        X: Input features
        n_simulations: Number of forward passes with dropout enabled
        keep_dropout_active: If True, enables dropout at inference time
        
    Returns:
        Dictionary with mean, std, and confidence interval bounds
        Keys: 'mean', 'std', 'lower_95', 'upper_95', 'lower_68', 'upper_68'
    """
    
    predictions = []
    
    for _ in range(n_simulations):
        # Forward pass with training=True keeps dropout active
        if keep_dropout_active:
            pred = model(X, training=True)
        else:
            pred = model(X, training=False)
        
        if isinstance(pred, tf.Tensor):
            pred = pred.numpy()
        
        predictions.append(pred)
    
    predictions = np.array(predictions)  # Shape: (n_simulations, n_samples, horizon)
    
    return {
        "mean": np.mean(predictions, axis=0),
        "std": np.std(predictions, axis=0),
        "lower_95": np.percentile(predictions, 2.5, axis=0),
        "upper_95": np.percentile(predictions, 97.5, axis=0),
        "lower_68": np.percentile(predictions, 16, axis=0),   # ±1 σ
        "upper_68": np.percentile(predictions, 84, axis=0),
    }


def bootstrap_uncertainty(
    predictions: np.ndarray,
    y_true: np.ndarray,
    n_bootstrap: int = 1000,
) -> dict[str, np.ndarray]:
    """Estimate uncertainty via residual bootstrap.
    
    Args:
        predictions: Model predictions (n_samples, horizon)
        y_true: Actual values (n_samples, horizon)
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Dictionary with bootstrapped confidence intervals
    """
    
    residuals = y_true - predictions
    n_samples = residuals.shape[0]
    
    # Bootstrap resample residuals and add to predictions
    bootstrap_preds = []
    for _ in range(n_bootstrap):
        boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_residuals = residuals[boot_indices]
        boot_pred = predictions + boot_residuals
        bootstrap_preds.append(boot_pred)
    
    bootstrap_preds = np.array(bootstrap_preds)
    
    return {
        "mean": np.mean(bootstrap_preds, axis=0),
        "std": np.std(bootstrap_preds, axis=0),
        "lower_95": np.percentile(bootstrap_preds, 2.5, axis=0),
        "upper_95": np.percentile(bootstrap_preds, 97.5, axis=0),
    }


def convert_uncertainty_to_csv(
    predictions_df,
    uncertainty_dict,
    gws: list[int],
) -> None:
    """Export uncertainty-aware projections to CSV with confidence bounds.
    
    Args:
        predictions_df: Base projections DataFrame
        uncertainty_dict: Output from predict_with_uncertainty
        gws: List of gameweeks (GW24, GW25, etc.)
    """
    
    df = predictions_df.copy()
    
    mean_pred = uncertainty_dict["mean"]
    std_pred = uncertainty_dict["std"]
    lower_95 = uncertainty_dict["lower_95"]
    upper_95 = uncertainty_dict["upper_95"]
    
    for i, gw in enumerate(gws):
        col_base = f"GW{gw}"
        df[f"{col_base}_proj"] = mean_pred[:, i] if mean_pred.ndim > 1 else mean_pred
        df[f"{col_base}_std"] = std_pred[:, i] if std_pred.ndim > 1 else std_pred
        df[f"{col_base}_lower_95"] = lower_95[:, i] if lower_95.ndim > 1 else lower_95
        df[f"{col_base}_upper_95"] = upper_95[:, i] if upper_95.ndim > 1 else upper_95
    
    return df

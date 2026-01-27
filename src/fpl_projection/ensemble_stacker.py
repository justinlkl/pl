"""Ensemble stacking meta-learner combining LSTM + XGBoost + Linear models.

Stacks multiple base models for robust predictions with improved variance
handling compared to individual models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

import tensorflow as tf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib


class EnsembleStacker:
    """Meta-learner that combines LSTM, XGBoost, and Ridge predictions."""
    
    def __init__(
        self,
        xgb_params: dict[str, Any] | None = None,
        ridge_alpha: float = 1.0,
        meta_params: dict[str, Any] | None = None,
    ):
        """Initialize ensemble components.
        
        Args:
            xgb_params: Hyperparameters for GradientBoostingRegressor
            ridge_alpha: Regularization for Ridge regression
            meta_params: Hyperparameters for meta-model (also GBR)
        """
        self.xgb_model = GradientBoostingRegressor(
            **(xgb_params or {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 5,
                "random_state": 42,
            })
        )
        self.linear_model = Ridge(alpha=ridge_alpha)
        self.meta_model = GradientBoostingRegressor(
            **(meta_params or {
                "n_estimators": 50,
                "learning_rate": 0.05,
                "max_depth": 3,
                "random_state": 42,
            })
        )
        
        self.meta_scaler = StandardScaler()
        self.xgb_is_fitted = False
        self.linear_is_fitted = False
        self.meta_is_fitted = False
        
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        lstm_model: tf.keras.Model,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Train ensemble components.
        
        Args:
            X_train: Training features (2D or 3D for LSTM)
            y_train: Training targets
            lstm_model: Pre-trained LSTM model
            X_val: Optional validation features for early stopping
            y_val: Optional validation targets
            
        Returns:
            Dictionary of training metrics
        """
        
        # Get LSTM predictions on training data
        if len(X_train.shape) == 3:
            # LSTM format (sequences)
            lstm_pred_train = lstm_model.predict(X_train, verbose=0)
        else:
            lstm_pred_train = lstm_model.predict(X_train, verbose=0)
        
        # Train XGBoost (expects 2D input)
        X_train_2d = X_train.reshape(X_train.shape[0], -1) if len(X_train.shape) > 2 else X_train
        self.xgb_model.fit(X_train_2d, y_train)
        self.xgb_is_fitted = True
        xgb_pred_train = self.xgb_model.predict(X_train_2d)
        
        # Train Ridge
        self.linear_model.fit(X_train_2d, y_train)
        self.linear_is_fitted = True
        linear_pred_train = self.linear_model.predict(X_train_2d)
        
        # Stack base predictions as meta-features
        meta_features_train = np.column_stack([
            lstm_pred_train.flatten() if lstm_pred_train.ndim > 1 else lstm_pred_train,
            xgb_pred_train.flatten() if xgb_pred_train.ndim > 1 else xgb_pred_train,
            linear_pred_train.flatten() if linear_pred_train.ndim > 1 else linear_pred_train,
        ])
        meta_features_train = self.meta_scaler.fit_transform(meta_features_train)
        
        # Train meta-model
        self.meta_model.fit(meta_features_train, y_train)
        self.meta_is_fitted = True
        
        # Compute training metrics
        meta_pred = self.meta_model.predict(meta_features_train)
        train_mae = float(np.mean(np.abs(meta_pred - y_train)))
        train_rmse = float(np.sqrt(np.mean((meta_pred - y_train) ** 2)))
        
        metrics = {
            "train_mae": train_mae,
            "train_rmse": train_rmse,
        }
        
        # Validation metrics (optional)
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val, lstm_model)
            val_mae = float(np.mean(np.abs(val_pred - y_val)))
            val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
            metrics["val_mae"] = val_mae
            metrics["val_rmse"] = val_rmse
        
        return metrics
    
    def predict(
        self,
        X: np.ndarray,
        lstm_model: tf.keras.Model,
    ) -> np.ndarray:
        """Generate ensemble predictions.
        
        Args:
            X: Input features
            lstm_model: Pre-trained LSTM model
            
        Returns:
            Array of predictions
        """
        
        if not (self.xgb_is_fitted and self.linear_is_fitted and self.meta_is_fitted):
            raise ValueError("Ensemble not fitted yet; call fit() first")
        
        # LSTM predictions
        lstm_pred = lstm_model.predict(X, verbose=0)
        
        # XGBoost predictions
        X_2d = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
        xgb_pred = self.xgb_model.predict(X_2d)
        
        # Ridge predictions
        linear_pred = self.linear_model.predict(X_2d)
        
        # Stack and scale
        meta_features = np.column_stack([
            lstm_pred.flatten() if lstm_pred.ndim > 1 else lstm_pred,
            xgb_pred.flatten() if xgb_pred.ndim > 1 else xgb_pred,
            linear_pred.flatten() if linear_pred.ndim > 1 else linear_pred,
        ])
        meta_features = self.meta_scaler.transform(meta_features)
        
        # Meta-model prediction
        return self.meta_model.predict(meta_features)
    
    def save(self, dirpath: Path) -> None:
        """Save ensemble to disk.
        
        Args:
            dirpath: Directory to save models
        """
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.xgb_model, dirpath / "xgb_model.joblib")
        joblib.dump(self.linear_model, dirpath / "linear_model.joblib")
        joblib.dump(self.meta_model, dirpath / "meta_model.joblib")
        joblib.dump(self.meta_scaler, dirpath / "meta_scaler.joblib")
    
    @classmethod
    def load(cls, dirpath: Path) -> EnsembleStacker:
        """Load ensemble from disk.
        
        Args:
            dirpath: Directory containing saved models
            
        Returns:
            Fitted EnsembleStacker instance
        """
        dirpath = Path(dirpath)
        stacker = cls()
        
        stacker.xgb_model = joblib.load(dirpath / "xgb_model.joblib")
        stacker.linear_model = joblib.load(dirpath / "linear_model.joblib")
        stacker.meta_model = joblib.load(dirpath / "meta_model.joblib")
        stacker.meta_scaler = joblib.load(dirpath / "meta_scaler.joblib")
        
        stacker.xgb_is_fitted = True
        stacker.linear_is_fitted = True
        stacker.meta_is_fitted = True
        
        return stacker

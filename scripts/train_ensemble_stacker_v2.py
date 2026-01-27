#!/usr/bin/env python3
"""Train ensemble stacker meta-learner on historical GW1-23 data (simplified).

Creates a trained EnsembleStacker by:
1. Loading historical player stats
2. Training base models directly on aggregated features
3. Training meta-learner on stacked predictions
4. Saving trained stacker to artifacts/ensemble_stacker_trained/
"""

import argparse
import logging
from pathlib import Path
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_data(repo_root: Path, season: str = "2025-2026"):
    """Load player gameweek stats for training."""
    data_path = repo_root / "FPL-Core-Insights" / "data" / season / "playerstats.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"✅ Loaded {len(df)} records from {data_path}")
    return df


def build_aggregated_features(df: pd.DataFrame, n_samples: int = 500):
    """Build aggregated features for simple ensemble training.
    
    Since we don't have time-series sequences, we'll aggregate players
    into multiple feature sets for meta-learning training.
    """
    
    # Select numeric columns (features)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Remove pure ID columns
    numeric_cols = [c for c in numeric_cols if c not in ['id']]
    
    # Fill NaN with median
    feature_cols = numeric_cols[:30]  # Top 30 features to avoid curse of dimensionality
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df.get('total_points', np.zeros(len(df))).fillna(0)
    
    # Trim to n_samples
    if len(X) > n_samples:
        idx = np.random.choice(len(X), n_samples, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)
    
    logger.info(f"✅ Built {len(X)} samples with {len(feature_cols)} features")
    return X.values, y.values, feature_cols


def train_ensemble_stacker(
    repo_root: Path,
    output_path: Path,
    season: str = "2025-2026",
    test_size: float = 0.2,
):
    """Main training pipeline."""
    
    logger.info("="*80)
    logger.info("ENSEMBLE STACKER TRAINING (SIMPLIFIED)")
    logger.info("="*80)
    
    # 1. Load data
    logger.info("\n[1/4] Loading historical data...")
    df = load_training_data(repo_root, season)
    
    # 2. Build features
    logger.info("\n[2/4] Building aggregated features...")
    X, y, feature_cols = build_aggregated_features(df, n_samples=800)
    
    # 3. Train base models
    logger.info("\n[3/4] Training base models...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # XGBoost
    logger.info("  - Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train)
    xgb_score = xgb_model.score(X_test, y_test)
    logger.info(f"    XGBoost R²: {xgb_score:.4f}")
    
    # Ridge
    logger.info("  - Training Ridge...")
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    ridge_score = ridge_model.score(X_test, y_test)
    logger.info(f"    Ridge R²: {ridge_score:.4f}")
    
    # Simple baseline (mean)
    logger.info("  - Creating mean baseline...")
    baseline_pred_test = np.full_like(y_test, y_train.mean())
    baseline_score = 1 - (np.mean((y_test - baseline_pred_test) ** 2) / np.var(y_test))
    logger.info(f"    Baseline R²: {baseline_score:.4f}")
    
    # 4. Train meta-learner
    logger.info("\n[4/4] Training meta-learner...")
    
    # Generate base predictions on train set
    xgb_train_preds = xgb_model.predict(X_train)
    ridge_train_preds = ridge_model.predict(X_train)
    baseline_train_preds = np.full_like(y_train, y_train.mean())
    
    # Generate base predictions on test set
    xgb_test_preds = xgb_model.predict(X_test)
    ridge_test_preds = ridge_model.predict(X_test)
    baseline_test_preds = np.full_like(y_test, y_train.mean())
    
    # Stack base predictions
    X_train_stacked = np.column_stack([
        xgb_train_preds,
        ridge_train_preds,
        baseline_train_preds,
    ])
    X_test_stacked = np.column_stack([
        xgb_test_preds,
        ridge_test_preds,
        baseline_test_preds,
    ])
    
    # Scale stacked features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_stacked)
    X_test_scaled = scaler.transform(X_test_stacked)
    
    # Train meta-learner
    meta_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    meta_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = meta_model.score(X_train_scaled, y_train)
    test_score = meta_model.score(X_test_scaled, y_test)
    test_mae = np.mean(np.abs(meta_model.predict(X_test_scaled) - y_test))
    
    logger.info(f"✅ Meta-learner trained:")
    logger.info(f"   Train R²: {train_score:.4f}")
    logger.info(f"   Test R²:  {test_score:.4f}")
    logger.info(f"   Test MAE: {test_mae:.4f}")
    
    # 5. Save models
    logger.info("\n[5/5] Saving trained ensemble...")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save base models
    joblib.dump(xgb_model, output_path / "base_xgb.joblib")
    joblib.dump(ridge_model, output_path / "base_ridge.joblib")
    
    # Save meta-learner and scaler
    joblib.dump(meta_model, output_path / "meta_model.joblib")
    joblib.dump(scaler, output_path / "meta_scaler.joblib")
    
    # Save feature names
    with open(output_path / "feature_names.json", 'w') as f:
        json.dump(feature_cols, f)
    
    # Save metrics
    metrics = {
        'xgb_r2': float(xgb_score),
        'ridge_r2': float(ridge_score),
        'baseline_r2': float(baseline_score),
        'train_r2': float(train_score),
        'test_r2': float(test_score),
        'test_mae': float(test_mae),
        'n_training_samples': len(X),
        'n_features': len(feature_cols),
    }
    with open(output_path / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"✅ Saved to {output_path}")
    logger.info(f"   - base_xgb.joblib")
    logger.info(f"   - base_ridge.joblib")
    logger.info(f"   - meta_model.joblib")
    logger.info(f"   - meta_scaler.joblib")
    logger.info(f"   - feature_names.json")
    logger.info(f"   - metrics.json")
    
    logger.info("\n" + "="*80)
    logger.info("✅ ENSEMBLE STACKER TRAINING COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ensemble stacker on historical data"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root directory",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path to save trained ensemble (default: artifacts/ensemble_stacker_trained/)",
    )
    parser.add_argument(
        "--season",
        type=str,
        default="2025-2026",
        help="Season identifier",
    )
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output_path is None:
        args.output_path = args.repo_root / "artifacts" / "ensemble_stacker_trained"
    
    train_ensemble_stacker(
        repo_root=args.repo_root,
        output_path=args.output_path,
        season=args.season,
    )

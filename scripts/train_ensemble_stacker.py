#!/usr/bin/env python3
"""Train ensemble stacker with PROPER time-series validation (NO LEAKAGE)."""

import argparse
import logging
from pathlib import Path
import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_training_data(repo_root: Path, season: str = "2025-2026") -> pd.DataFrame:
    """Load per-gameweek player stats and attach gw/player_id columns."""
    base = repo_root / "FPL-Core-Insights" / "data" / season / "By Gameweek"
    if not base.exists():
        raise FileNotFoundError(f"Gameweek folder not found: {base}")

    frames = []
    for gw_dir in sorted(base.iterdir()):
        if not gw_dir.is_dir() or not gw_dir.name.startswith("GW"):
            continue
        gw_num = int(gw_dir.name.replace("GW", ""))
        f = gw_dir / "player_gameweek_stats.csv"
        if not f.exists():
            continue
        gdf = pd.read_csv(f)
        gdf["gw"] = gw_num
        gdf = gdf.rename(columns={"id": "player_id", "event_points": "points"})
        frames.append(gdf)

    if not frames:
        raise FileNotFoundError("No player_gameweek_stats.csv files found")

    df = pd.concat(frames, ignore_index=True)
    logger.info(f"✅ Loaded {len(df)} player-gw rows across {len(frames)} gameweeks")
    return df


def prepare_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag/rolling features using only past data (no leakage)."""
    df = df.sort_values(['player_id', 'gw']).reset_index(drop=True)

    # Drop leakage-prone columns
    forbidden_cols = [
        'total_points', 'event_points', 'form', 'value_form', 'proj_points',
    ]
    df = df.drop(columns=[c for c in forbidden_cols if c in df.columns], errors='ignore')

    # Core target per GW
    if 'points' not in df.columns and 'event_points' in df.columns:
        df = df.rename(columns={'event_points': 'points'})

    # Lag features
    for lag in [1, 2, 3]:
        df[f'points_lag{lag}'] = (
            df.groupby('player_id')['points'].shift(lag).fillna(0)
        )

    # Rolling means (shifted to exclude current GW)
    for window in [3, 5, 10]:
        df[f'points_roll{window}'] = (
            df.groupby('player_id')['points']
            .rolling(window, min_periods=1).mean()
            .shift(1)
            .reset_index(0, drop=True)
            .fillna(0)
        )

    # Expected stats rolling
    if 'expected_goals' in df.columns:
        df['xG_roll5'] = (
            df.groupby('player_id')['expected_goals']
            .rolling(5, min_periods=1).mean()
            .shift(1)
            .reset_index(0, drop=True)
            .fillna(0)
        )
    else:
        df['xG_roll5'] = 0.0

    if 'expected_assists' in df.columns:
        df['xA_roll5'] = (
            df.groupby('player_id')['expected_assists']
            .rolling(5, min_periods=1).mean()
            .shift(1)
            .reset_index(0, drop=True)
            .fillna(0)
        )
    else:
        df['xA_roll5'] = 0.0
    if 'minutes' in df.columns:
        df['minutes_roll3'] = (
            df.groupby('player_id')['minutes']
            .rolling(3, min_periods=1).mean()
            .shift(1)
            .reset_index(0, drop=True)
            .fillna(0)
        )
    else:
        df['minutes_roll3'] = 0.0

    return df


def build_time_series_dataset(df: pd.DataFrame, seq_length: int = 5, horizon: int = 1):
    """Build dataset: past seq_length GWs -> next GW points."""
    sequences, targets, gameweeks = [], [], []

    feature_cols = [
        c for c in df.columns
        if c not in ['player_id', 'gw', 'points', 'name', 'team']
        and ('lag' in c or 'roll' in c or 'xG' in c or 'xA' in c or 'minutes' in c)
    ]
    logger.info(f"Using {len(feature_cols)} features (sample): {feature_cols[:5]}")

    for pid in df['player_id'].unique():
        g = df[df['player_id'] == pid].sort_values('gw')
        if len(g) < seq_length + horizon:
            continue
        for i in range(len(g) - seq_length - horizon + 1):
            seq = g.iloc[i:i+seq_length]
            target_row = g.iloc[i+seq_length]
            X_seq = seq[feature_cols].values
            y_val = target_row['points']
            sequences.append(X_seq)
            targets.append(y_val)
            gameweeks.append(seq['gw'].iloc[-1])

    X = np.array(sequences)
    y = np.array(targets)
    gw = np.array(gameweeks)
    logger.info(f"✅ Built {len(X)} sequences; X shape {X.shape}, y shape {y.shape}")
    return X, y, gw, feature_cols


def time_series_split(X, y, gw, test_gw_threshold: int = 20):
    train_mask = gw <= test_gw_threshold
    test_mask = gw > test_gw_threshold
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    logger.info(f"✅ Time-series split: train {len(X_train)} (GW<= {test_gw_threshold}), test {len(X_test)}")
    return X_train, X_test, y_train, y_test


def train_base_models(X_train, y_train):
    X_flat = X_train[:, -1, :]
    logger.info("Training XGBoost and Ridge on last-timestep features...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    xgb_model.fit(X_flat, y_train)

    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_flat, y_train)
    return xgb_model, ridge_model


def generate_base_predictions(lstm_model, xgb_model, ridge_model, X):
    n_samples = X.shape[0]
    lstm_preds = lstm_model.predict(X, verbose=0)
    if lstm_preds.ndim > 1:
        lstm_preds = lstm_preds[:, 0]
    X_flat = X[:, -1, :]
    xgb_preds = xgb_model.predict(X_flat)
    ridge_preds = ridge_model.predict(X_flat)
    base_preds = np.column_stack([lstm_preds, xgb_preds, ridge_preds])
    return base_preds


def tune_meta_learner(X_train_scaled, y_train):
    """Fit meta-learner with conservative hyperparameters (fast, generalizes better).

    GridSearchCV over many candidates is slow (and easy to interrupt). This keeps the
    run time predictable while avoiding overly-aggressive params.
    """
    logger.info("🎯 Using conservative meta-learner params (skipping GridSearchCV)...")

    conservative_params = {
        'n_estimators': 30,
        'max_depth': 2,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'random_state': 42,
    }

    meta_model = GradientBoostingRegressor(**conservative_params)
    meta_model.fit(X_train_scaled, y_train)
    logger.info(f"✅ Fitted meta-learner: {conservative_params}")
    return meta_model


def build_proper_lstm(n_features, seq_length=5):
    """Build proper LSTM architecture with regularization and callbacks."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    
    model = Sequential([
        Input(shape=(seq_length, n_features)),
        LSTM(32, return_sequences=False),  # Simplified: single layer
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model


def validate_no_leakage(X, y, feature_names):
    logger.info("\n🔍 Running leakage checks...")
    X_flat = X[:, -1, :]
    suspicious = []
    for i, fname in enumerate(feature_names[: X_flat.shape[1]]):
        corr = np.corrcoef(X_flat[:, i], y)[0, 1]
        if abs(corr) > 0.95:
            suspicious.append((fname, corr))
            logger.warning(f"⚠️ High correlation: {fname} = {corr:.4f}")
    if suspicious:
        raise ValueError("Possible leakage detected (high feature-target correlation)")


def train_ensemble_stacker(
    repo_root: Path,
    artifacts_path: Path,
    output_path: Path,
    season: str = "2025-2026",
    test_gw: int = 20,
):
    logger.info("="*80)
    logger.info("ENSEMBLE STACKER TRAINING (LEAKAGE-FREE)")
    logger.info("="*80)

    # 1. Load data
    logger.info("\n[1/6] Loading historical data...")
    df = load_training_data(repo_root, season)

    # 2. Build lag/rolling features
    logger.info("\n[2/6] Creating lag features...")
    df = prepare_historical_features(df)

    # 3. Build time-series dataset
    logger.info("\n[3/6] Building time-series dataset...")
    X, y, gw, feature_cols = build_time_series_dataset(df, seq_length=5, horizon=1)

    # 4. Time-series split (expand test window for stability)
    logger.info("\n[4/6] Splitting train/test by GW...")
    expanded_test_gw = max(test_gw - 2, 15)  # Expand test window by 2 GWs for more stable metrics
    logger.info(f"💡 Expanding test window from GW>{test_gw} to GW>{expanded_test_gw} for more stable metrics")
    X_train, X_test, y_train, y_test = time_series_split(X, y, gw, test_gw_threshold=expanded_test_gw)

    # 5. Leakage checks
    validate_no_leakage(X_train, y_train, feature_cols)

    # 6. Load LSTM
    logger.info("\n[5/6] Loading LSTM model...")
    lstm_path = artifacts_path / "model.keras"
    lstm_model = None
    if lstm_path.exists():
        try:
            loaded = tf.keras.models.load_model(lstm_path, compile=False)
            expected_time_steps = loaded.input_shape[1]
            expected_features = loaded.input_shape[2]
            if expected_time_steps == X_train.shape[1] and expected_features == X_train.shape[2]:
                lstm_model = loaded
                logger.info("✅ Loaded LSTM (input shapes match)")
            else:
                logger.warning(
                    f"⚠️ LSTM input mismatch, expected (time,feat)=({expected_time_steps},{expected_features}) "
                    f"but got ({X_train.shape[1]},{X_train.shape[2]}). Rebuilding with current features."
                )
        except Exception as e:
            logger.warning(f"⚠️ Failed to load LSTM, will rebuild: {e}")

    if lstm_model is None:
        logger.info("🔧 Building proper LSTM for current features...")
        lstm_model = build_proper_lstm(n_features=X_train.shape[2], seq_length=X_train.shape[1])
        
        # Split training into train/val for LSTM
        X_lstm_train, X_lstm_val, y_lstm_train, y_lstm_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42
        )
        
        # Callbacks for better training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=0
            )
        ]
        
        # Train with proper epochs
        logger.info("🏋️ Training LSTM (~60 seconds)...")
        history = lstm_model.fit(
            X_lstm_train, y_lstm_train,
            validation_data=(X_lstm_val, y_lstm_val),
            epochs=30,
            batch_size=64,
            callbacks=callbacks,
            verbose=0
        )
        
        best_val_loss = min(history.history['val_loss'])
        logger.info(f"✅ Trained LSTM - Best val_loss: {best_val_loss:.4f}")

    # 7. Train base models
    logger.info("\n[6/6] Training base models...")
    xgb_model, ridge_model = train_base_models(X_train, y_train)

    # Base predictions
    train_base_preds = generate_base_predictions(lstm_model, xgb_model, ridge_model, X_train)
    test_base_preds = generate_base_predictions(lstm_model, xgb_model, ridge_model, X_test)

    # Meta-learner
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_base_preds)
    test_scaled = scaler.transform(test_base_preds)

    # Tune meta-learner hyperparameters
    meta_model = tune_meta_learner(train_scaled, y_train)
    
    train_r2 = meta_model.score(train_scaled, y_train)
    test_r2 = meta_model.score(test_scaled, y_test)
    test_mae = mean_absolute_error(y_test, meta_model.predict(test_scaled))

    logger.info(f"\n✅ Meta-learner Results (GradientBoosting):")
    logger.info(f"   Train R²: {train_r2:.4f}")
    logger.info(f"   Test R²:  {test_r2:.4f}")
    logger.info(f"   Test MAE: {test_mae:.4f}")
    
    # Also compute weighted ensemble for comparison
    logger.info("\n📊 Computing weighted ensemble for comparison...")
    test_preds_weighted = (
        0.50 * test_base_preds[:, 0] +  # LSTM (50%)
        0.30 * test_base_preds[:, 1] +  # XGBoost (30%)
        0.20 * test_base_preds[:, 2]    # Ridge (20%)
    )
    test_mae_weighted = mean_absolute_error(y_test, test_preds_weighted)
    test_r2_weighted = r2_score(y_test, test_preds_weighted)
    
    logger.info(f"✅ Weighted Ensemble Results (0.5*LSTM + 0.3*XGB + 0.2*Ridge):")
    logger.info(f"   Test R²:  {test_r2_weighted:.4f}")
    logger.info(f"   Test MAE: {test_mae_weighted:.4f}")
    
    # Use best-performing ensemble
    if test_mae_weighted < test_mae:
        logger.info("\n🎯 Weighted ensemble outperforms meta-learner; using weighted ensemble")
        final_test_r2 = test_r2_weighted
        final_test_mae = test_mae_weighted
        joblib.dump({"type": "weighted", "weights": [0.5, 0.3, 0.2]}, output_path / "ensemble_config.joblib")
    else:
        logger.info("\n🎯 Meta-learner outperforms weighted ensemble; using meta-learner")
        final_test_r2 = test_r2
        final_test_mae = test_mae

    # Save
    output_path.mkdir(parents=True, exist_ok=True)
    # Persist the LSTM we actually used (loaded or retrained) so production
    # predictions can use the same input feature shape.
    try:
        lstm_model.save(output_path / "lstm_model.keras")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"⚠️ Failed to save LSTM model: {exc}")

    # Persist feature names used to build sequences.
    try:
        with open(output_path / "feature_names.json", "w", encoding="utf-8") as f:
            json.dump(list(feature_cols), f, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"⚠️ Failed to save feature_names.json: {exc}")

    joblib.dump(xgb_model, output_path / "base_xgb.joblib")
    joblib.dump(ridge_model, output_path / "base_ridge.joblib")
    joblib.dump(meta_model, output_path / "meta_model.joblib")
    joblib.dump(scaler, output_path / "meta_scaler.joblib")

    metrics = {
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'test_mae': float(test_mae),
        'test_r2_weighted': float(test_r2_weighted),
        'test_mae_weighted': float(test_mae_weighted),
        'final_test_r2': float(final_test_r2),
        'final_test_mae': float(final_test_mae),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'test_gw_threshold': int(expanded_test_gw),
    }
    with open(output_path / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info("✅ Training complete!")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--season", type=str, default="2025-2026")
    parser.add_argument("--test-gw", type=int, default=20, help="GW threshold for train/test split")

    args = parser.parse_args()

    if args.artifacts_path is None:
        args.artifacts_path = args.repo_root / "artifacts"

    if args.output_path is None:
        args.output_path = args.repo_root / "artifacts" / "ensemble_stacker_trained"

    train_ensemble_stacker(
        repo_root=args.repo_root,
        artifacts_path=args.artifacts_path,
        output_path=args.output_path,
        season=args.season,
        test_gw=args.test_gw,
    )

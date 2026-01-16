from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_HORIZON,
    DEFAULT_SEQ_LENGTH,
    DEFAULT_TEST_GWS,
    DEFAULT_VAL_GWS,
    TARGET_COLUMN,
)
from .data_loading import load_premier_league_gameweek_stats
from .modeling import build_lstm_model
from .preprocessing import PreprocessArtifacts, fit_preprocessor_on_timesteps, select_and_coerce_numeric, transform_sequences
from .sequences import build_sequences, split_by_end_gw


def _set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


def _get_available_features(df: pd.DataFrame, requested_features: list[str]) -> list[str]:
    """Return only the features that actually exist in the DataFrame.
    
    This allows the model to gracefully handle cases where some engineered
    features might not be available in the data.
    """
    available = [f for f in requested_features if f in df.columns]
    missing = set(requested_features) - set(available)
    if missing:
        print(f"Warning: {len(missing)} requested features not found in data: {missing}")
    return available


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM model for FPL point projections")
    parser.add_argument("--season", default="2025-2026")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--seq-length", type=int, default=DEFAULT_SEQ_LENGTH)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--val-gws", type=int, default=DEFAULT_VAL_GWS)
    parser.add_argument("--test-gws", type=int, default=DEFAULT_TEST_GWS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _set_seed(args.seed)

    repo_root = Path(args.repo_root)
    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load data with feature engineering applied
    print("Loading and engineering features...")
    raw = load_premier_league_gameweek_stats(repo_root=repo_root, season=args.season, apply_feature_engineering=True)
    
    # Use only available features
    available_features = _get_available_features(raw, DEFAULT_FEATURE_COLUMNS)
    print(f"Using {len(available_features)} features for training")
    
    # Select and coerce numeric data
    df = select_and_coerce_numeric(raw, available_features, TARGET_COLUMN)
    
    print(f"Building sequences with seq_length={args.seq_length}, horizon={args.horizon}")
    dataset = build_sequences(
        df=df,
        feature_columns=available_features,
        target_column=TARGET_COLUMN,
        seq_length=args.seq_length,
        horizon=args.horizon,
    )

    max_end_gw = int(dataset.end_gw.max())
    # Ensure some room: train | val | test in time order
    val_max_end_gw = max_end_gw - args.test_gws
    train_max_end_gw = val_max_end_gw - args.val_gws
    if train_max_end_gw < 1:
        raise ValueError(
            f"Not enough gameweeks to split: max_end_gw={max_end_gw}, "
            f"val_gws={args.val_gws}, test_gws={args.test_gws}."
        )

    print(f"Splitting data: train_max_gw={train_max_end_gw}, val_max_gw={val_max_end_gw}, test_max_gw={max_end_gw}")
    train_ds, val_ds, test_ds = split_by_end_gw(
        dataset,
        train_max_end_gw=train_max_end_gw,
        val_max_end_gw=val_max_end_gw,
    )
    
    print(f"Dataset sizes - Train: {len(train_ds.X)}, Val: {len(val_ds.X)}, Test: {len(test_ds.X)}")

    # Fit preprocessing on training timesteps only.
    n_features = train_ds.X.shape[-1]
    train_timesteps_2d = train_ds.X.reshape(train_ds.X.shape[0] * args.seq_length, n_features)
    print("Fitting preprocessing pipeline (median imputation + standardization)...")
    pipeline = fit_preprocessor_on_timesteps(train_timesteps_2d)

    print("Transforming sequences...")
    X_train = transform_sequences(pipeline, train_ds.X)
    X_val = transform_sequences(pipeline, val_ds.X)
    X_test = transform_sequences(pipeline, test_ds.X)

    print(f"Building LSTM model (seq_length={args.seq_length}, features={n_features}, horizon={args.horizon})")
    model = build_lstm_model(seq_length=args.seq_length, num_features=n_features, horizon=args.horizon)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]

    print(f"Training for up to {args.epochs} epochs with batch_size={args.batch_size}...")
    history = model.fit(
        X_train,
        train_ds.y,
        validation_data=(X_val, val_ds.y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_mae = model.evaluate(X_test, test_ds.y, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

    # Save model and preprocessing artifacts
    model_path = artifacts_dir / "model.keras"
    preprocess_path = artifacts_dir / "preprocess.joblib"
    metadata_path = artifacts_dir / "meta.json"
    
    print(f"\nSaving artifacts to {artifacts_dir}...")
    model.save(str(model_path))
    preprocess_artifacts = PreprocessArtifacts(feature_columns=available_features, pipeline=pipeline)
    preprocess_artifacts.save(str(preprocess_path))
    
    # Save metadata
    metadata = {
        "season": args.season,
        "seq_length": args.seq_length,
        "horizon": args.horizon,
        "num_features": n_features,
        "feature_columns": available_features,
        "test_loss": float(test_loss),
        "test_mae": float(test_mae),
        "epochs_trained": len(history.history["loss"]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
    }
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {model_path}")
    print(f"Preprocessor saved to {preprocess_path}")
    print(f"Metadata saved to {metadata_path}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

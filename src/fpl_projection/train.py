from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM model for FPL point projections")
    parser.add_argument("--season", default="2025-2026")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--seq-length", type=int, default=DEFAULT_SEQ_LENGTH)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--val-gws", type=int, default=DEFAULT_VAL_GWS)
    parser.add_argument("--test-gws", type=int, default=DEFAULT_TEST_GWS)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _set_seed(args.seed)

    repo_root = Path(args.repo_root)
    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    raw = load_premier_league_gameweek_stats(repo_root=repo_root, season=args.season)
    df = select_and_coerce_numeric(raw, DEFAULT_FEATURE_COLUMNS, TARGET_COLUMN)

    dataset = build_sequences(
        df=df,
        feature_columns=DEFAULT_FEATURE_COLUMNS,
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

    train_ds, val_ds, test_ds = split_by_end_gw(
        dataset,
        train_max_end_gw=train_max_end_gw,
        val_max_end_gw=val_max_end_gw,
    )

    # Fit preprocessing on training timesteps only.
    n_features = train_ds.X.shape[-1]
    train_timesteps_2d = train_ds.X.reshape(train_ds.X.shape[0] * args.seq_length, n_features)
    pipeline = fit_preprocessor_on_timesteps(train_timesteps_2d)

    X_train = transform_sequences(pipeline, train_ds.X)
    X_val = transform_sequences(pipeline, val_ds.X)
    X_test = transform_sequences(pipeline, test_ds.X)

    model = build_lstm_model(seq_length=args.seq_length, num_features=n_features, horizon=args.horizon)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]

    history = model.fit(
        X_train,
        train_ds.y,
        validation_data=(X_val, val_ds.y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    test_metrics = model.evaluate(X_test, test_ds.y, verbose=0)

    model_path = artifacts_dir / "model.keras"
    model.save(model_path)

    preprocess_path = artifacts_dir / "preprocess.joblib"
    PreprocessArtifacts(feature_columns=DEFAULT_FEATURE_COLUMNS, pipeline=pipeline).save(str(preprocess_path))

    meta = {
        "season": args.season,
        "seq_length": args.seq_length,
        "horizon": args.horizon,
        "feature_columns": DEFAULT_FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "split": {
            "train_max_end_gw": train_max_end_gw,
            "val_max_end_gw": val_max_end_gw,
            "max_end_gw": max_end_gw,
        },
        "history_keys": list(history.history.keys()),
        "test_metrics": {"loss": float(test_metrics[0]), "mae": float(test_metrics[1])},
    }

    (artifacts_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved preprocessing: {preprocess_path}")
    print(f"Saved metadata: {artifacts_dir / 'meta.json'}")


if __name__ == "__main__":
    main()

"""
Script: Weekly Fine-Tuning (Every Week)
Fast adaptation to latest gameweek trends
Expects base model to exist at artifacts/model_base.keras
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fpl_projection.data_loading import load_premier_league_gameweek_stats
from src.fpl_projection.feature_engineering import engineer_all_features
from src.fpl_projection.recency_weighting import create_gw_based_sample_weights
from src.fpl_projection.sequences import build_sequences
from src.fpl_projection.preprocessing import fit_preprocessor_on_timesteps, transform_sequences


def finetune_weekly(
    repo_root: Path = None,
    artifacts_dir: Path = None,
    season: str = "2025-2026",
    latest_gw: int = None,
    seq_length: int = 10,
    horizon: int = 6,
    epochs: int = 3,
    learning_rate: float = 0.00001,
    freeze_layers: int = 2,
):
    """Fine-tune model on latest gameweek data.
    
    Args:
        repo_root: Repository root
        artifacts_dir: Artifacts directory
        season: Season to fine-tune on
        latest_gw: Latest gameweek (auto-detect if None)
        seq_length: Sequence length
        horizon: Prediction horizon
        epochs: Fine-tuning epochs (typically 3-5)
        learning_rate: Fine-tuning learning rate (lower than initial training)
        freeze_layers: Number of initial layers to freeze
    """
    
    if repo_root is None:
        repo_root = Path(__file__).parent.parent.parent
    if artifacts_dir is None:
        artifacts_dir = repo_root / "artifacts"
    
    start_time = time.time()
    
    print("=" * 80)
    print("WEEKLY FINE-TUNING")
    print("Fast Adaptation to Latest Gameweek")
    print("=" * 80)
    
    # Load base model
    base_model_path = artifacts_dir / "model_base.keras"
    if not base_model_path.exists():
        print(f"❌ Base model not found at {base_model_path}")
        print("   Run: python scripts/10_train_base_model.py")
        return
    
    print(f"\n[1/5] Loading base model from {base_model_path.name}...")
    model = tf.keras.models.load_model(str(base_model_path))
    print(f"  ✅ Model loaded")
    print(f"     Parameters: {model.count_params():,}")
    
    # Load latest data
    print(f"\n[2/5] Loading latest {season} data...")
    df = load_premier_league_gameweek_stats(
        repo_root=repo_root,
        season=season,
        apply_feature_engineering=False,
    )
    
    if latest_gw is None:
        latest_gw = df["gw"].max()
    
    # Get only data up to latest_gw
    df = df[df["gw"] <= latest_gw].copy()
    print(f"  ✅ Data loaded: {df.shape[0]:,} records through GW {latest_gw}")
    
    # Feature engineering
    print(f"\n[3/5] Feature engineering...")
    df = engineer_all_features(df)
    print(f"  ✅ Features ready")
    
    # Build sequences with recent emphasis
    print(f"\n[4/5] Building training sequences...")
    from src.fpl_projection.config import DEFAULT_FEATURE_COLUMNS
    
    available_features = [f for f in DEFAULT_FEATURE_COLUMNS if f in df.columns]
    
    X, y, rows = build_sequences(
        df,
        feature_columns=available_features,
        seq_length=seq_length,
        horizon=horizon,
        include_sequence_metadata=True,
    )
    
    print(f"  ✅ Sequences built: {X.shape[0]:,} sequences")
    
    # Apply recency weights (aggressive for fine-tuning)
    sequence_gws = np.array([r["end_gw"] for r in rows])
    sample_weights = create_gw_based_sample_weights(
        pd.DataFrame({"gw": sequence_gws}),
        current_gw=latest_gw,
        half_life=5,  # Aggressive: focus on last 5 GWs
    )
    
    print(f"  Recency weights: min={np.min(sample_weights):.3f}, " 
          f"max={np.max(sample_weights):.3f}, mean={np.mean(sample_weights):.3f}")
    
    # Freeze early layers (optional)
    if freeze_layers > 0:
        print(f"\n[5/5] Preparing for fine-tuning (freeze_layers={freeze_layers})...")
        for layer in model.layers[:freeze_layers]:
            layer.trainable = False
        print(f"  ✅ Froze {freeze_layers} layers, {sum(l.trainable for l in model.layers)} trainable")
    else:
        print(f"\n[5/5] Preparing for fine-tuning...")
    
    # Recompile with lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mae", metrics=["mae"])
    print(f"  ✅ Compiled with LR={learning_rate}")
    
    # Fine-tune
    print(f"\n  Fine-tuning on {X.shape[0]:,} sequences...")
    history = model.fit(
        X,
        y,
        sample_weight=sample_weights,
        epochs=epochs,
        batch_size=32,  # Smaller batch for fast updates
        verbose=1,
    )
    
    # Save fine-tuned model
    model_path = artifacts_dir / "model.keras"
    model.save(str(model_path))
    print(f"\n✅ Fine-tuned model saved to {model_path.name}")
    
    # Save fine-tuning metadata
    elapsed = time.time() - start_time
    metadata = {
        "fine_tune_type": "weekly",
        "base_model": str(base_model_path.name),
        "season": season,
        "latest_gw": int(latest_gw),
        "sequences_used": int(X.shape[0]),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "freeze_layers": freeze_layers,
        "final_loss": float(history.history["loss"][-1]),
        "elapsed_time_seconds": float(elapsed),
    }
    
    metadata_path = artifacts_dir / "finetuning_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n" + "=" * 80)
    print("FINE-TUNING COMPLETE")
    print("=" * 80)
    print(f"Time elapsed: {elapsed:.1f} seconds")
    print(f"Final MAE: {history.history['mae'][-1]:.4f}")
    print(f"Model saved: {model_path.relative_to(repo_root)}")
    print(f"\nNext: python -m src.fpl_projection.ensemble_predict --season {season}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekly fine-tuning on latest gameweek")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--season", default="2025-2026")
    parser.add_argument("--latest-gw", type=int, default=None, help="Latest GW (auto-detect if None)")
    parser.add_argument("--epochs", type=int, default=3, help="Fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=0.00001, help="Fine-tuning learning rate")
    parser.add_argument("--freeze-layers", type=int, default=2, help="Freeze first N layers")
    
    args = parser.parse_args()
    finetune_weekly(
        repo_root=args.repo_root,
        artifacts_dir=args.artifacts_dir,
        season=args.season,
        latest_gw=args.latest_gw,
        epochs=args.epochs,
        learning_rate=args.lr,
        freeze_layers=args.freeze_layers,
    )

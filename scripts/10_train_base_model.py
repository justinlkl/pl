"""
Script: Initial Training (Every 6 Weeks)
Combines 24-25 + 25-26 seasons with recency weighting
Trains robust base model for 6-week period
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fpl_projection.data_processor import combine_seasons
from src.fpl_projection.data_loading import load_premier_league_gameweek_stats
from src.fpl_projection.feature_engineering import engineer_all_features
from src.fpl_projection.recency_weighting import (
    create_gw_based_sample_weights,
    RECENCY_PROFILES,
    analyze_weight_distribution,
)
from src.fpl_projection.sequences import build_sequences, split_by_end_gw
from src.fpl_projection.preprocessing import fit_preprocessor_on_timesteps, transform_sequences


def train_base_model(
    repo_root: Path = None,
    output_dir: Path = None,
    epochs: int = 50,
    batch_size: int = 256,
    half_life: int = 8,
    val_gws: int = 2,
    test_gws: int = 2,
):
    """Train base model on combined 24-25 + 25-26 data with recency weighting.
    
    Args:
        repo_root: Repository root directory
        output_dir: Output directory for model and artifacts
        epochs: Training epochs
        batch_size: Batch size
        half_life: Recency weighting half-life (gameweeks)
        val_gws: Validation gameweeks
        test_gws: Test gameweeks
    """
    
    if repo_root is None:
        repo_root = Path(__file__).parent.parent.parent
    if output_dir is None:
        output_dir = repo_root / "artifacts"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("INITIAL BASE MODEL TRAINING")
    print("Combine 24-25 + 25-26 with Recency Weighting")
    print("=" * 80)
    
    # Load both seasons
    print("\n[1/4] Loading data...")
    try:
        df_2425 = load_premier_league_gameweek_stats(
            repo_root=repo_root,
            season="2024-2025",
            apply_feature_engineering=False,
        )
        print(f"  ✅ 2024-2025: {df_2425.shape[0]:,} records")
    except Exception as e:
        print(f"  ⚠️  2024-2025 not available: {e}")
        df_2425 = None
    
    df_2526 = load_premier_league_gameweek_stats(
        repo_root=repo_root,
        season="2025-2026",
        apply_feature_engineering=False,
    )
    print(f"  ✅ 2025-2026: {df_2526.shape[0]:,} records")
    
    # Combine seasons
    print("\n[2/4] Combining seasons with defensive contribution recalculation...")
    if df_2425 is not None:
        df_combined = combine_seasons(
            df_2425,
            df_2526,
            recalculate_2425_defcon=True,
        )
        print(f"  ✅ Combined: {df_combined.shape[0]:,} records from {df_combined['gw'].max()} gameweeks")
        print(f"     Span: 24-25 (GW 1-38) + 25-26 (GW 1-{df_2526['gw'].max()})")
    else:
        df_combined = df_2526
        print(f"  ℹ️  Using 25-26 only: {df_combined.shape[0]:,} records")
    
    # Feature engineering
    print("\n[3/4] Feature engineering...")
    df_combined = engineer_all_features(df_combined)
    print(f"  ✅ Features created: {df_combined.shape[1]} columns")
    
    # Recency weighting analysis
    print("\n[4/4] Analyzing recency weights (half_life={})...".format(half_life))
    weight_df = analyze_weight_distribution(
        df_combined["gw"].values,
        current_gw=df_combined["gw"].max(),
        half_life=half_life,
    )
    print("\n  Weight Distribution by Gameweek:")
    print(weight_df.to_string(index=False))
    
    sample_weights = create_gw_based_sample_weights(
        df_combined,
        current_gw=df_combined["gw"].max(),
        half_life=half_life,
    )
    
    print(f"\n  Weight Statistics:")
    print(f"    Min: {np.min(sample_weights):.3f}")
    print(f"    Max: {np.max(sample_weights):.3f}")
    print(f"    Mean: {np.mean(sample_weights):.3f}")
    print(f"    Median: {np.median(sample_weights):.3f}")
    
    # Save training config
    config = {
        "training_type": "base_model",
        "combined_seasons": df_2425 is not None,
        "total_records": int(df_combined.shape[0]),
        "total_gameweeks": int(df_combined["gw"].max()),
        "recency_half_life": half_life,
        "epochs": epochs,
        "batch_size": batch_size,
        "weight_stats": {
            "min": float(np.min(sample_weights)),
            "max": float(np.max(sample_weights)),
            "mean": float(np.mean(sample_weights)),
            "median": float(np.median(sample_weights)),
        },
    }
    
    config_path = output_dir / "training_config_base.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n✅ Training config saved to {config_path.relative_to(repo_root)}")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Build sequences and train model (integration with train.py)")
    print("2. Save model to: artifacts/model_base.keras")
    print("3. After 6 weeks, run weekly fine-tuning:")
    print("   python scripts/10_weekly_finetune.py")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train base model with combined seasons")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--half-life", type=int, default=8, help="Recency weighting half-life")
    
    args = parser.parse_args()
    train_base_model(
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        half_life=args.half_life,
    )

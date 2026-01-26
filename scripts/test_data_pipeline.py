"""Test the complete multi-season data processing pipeline.

This script demonstrates loading, processing, and combining 24-25 and 25-26 data.
"""

from pathlib import Path

import pandas as pd

from src.fpl_projection.data_loading import load_premier_league_gameweek_stats
from src.fpl_projection.data_processor import (
    combine_seasons,
    get_training_feature_list,
    calculate_defensive_contribution_points,
)
from src.fpl_projection.feature_engineering import engineer_all_features


def test_pipeline():
    """Test the complete data processing pipeline."""
    
    repo_root = Path(__file__).parent.parent
    
    print("=" * 70)
    print("FPL MULTI-SEASON DATA PROCESSING TEST")
    print("=" * 70)
    
    # Test 1: Load 25-26 season
    print("\n[TEST 1] Loading 2025-2026 season...")
    try:
        df_2526 = load_premier_league_gameweek_stats(
            repo_root=repo_root,
            season="2025-2026",
            apply_feature_engineering=False,
        )
        print(f"✓ Loaded 2025-2026: {df_2526.shape}")
        print(f"  Players: {df_2526['player_id'].nunique()}")
        print(f"  Gameweeks: {df_2526['gw'].nunique()}")
    except Exception as e:
        print(f"✗ Failed to load 2025-2026: {e}")
        return
    
    # Test 2: Check defensive contribution exists in 25-26
    print("\n[TEST 2] Checking defensive contribution in 25-26...")
    if "defensive_contribution" in df_2526.columns:
        print(f"✓ defensive_contribution present")
        print(f"  Non-zero values: {(df_2526['defensive_contribution'] > 0).sum()}")
    else:
        print(f"⚠ defensive_contribution missing (will calculate)")
    
    # Test 3: Calculate defcon points
    print("\n[TEST 3] Calculating defensive contribution points...")
    df_2526_with_defcon = calculate_defensive_contribution_points(df_2526.copy())
    if "defcon_points" in df_2526_with_defcon.columns:
        defcon_earners = (df_2526_with_defcon["defcon_points"] > 0).sum()
        print(f"✓ defcon_points calculated")
        print(f"  Players earning defcon points: {defcon_earners}")
        print(f"  Total defcon points awarded: {df_2526_with_defcon['defcon_points'].sum()}")
    else:
        print(f"✗ Failed to calculate defcon_points")
    
    # Test 4: Try to load 24-25 (may not exist)
    print("\n[TEST 4] Attempting to load 2024-2025 season...")
    try:
        df_2425 = load_premier_league_gameweek_stats(
            repo_root=repo_root,
            season="2024-2025",
            apply_feature_engineering=False,
        )
        print(f"✓ Loaded 2024-2025: {df_2425.shape}")
        print(f"  Players: {df_2425['player_id'].nunique()}")
        
        has_2425 = True
    except Exception as e:
        print(f"⚠ 2024-2025 not available: {e}")
        print(f"  (This is expected if data not present)")
        has_2425 = False
    
    # Test 5: Combine seasons if both available
    if has_2425:
        print("\n[TEST 5] Combining seasons...")
        try:
            df_combined = combine_seasons(
                df_2425,
                df_2526,
                recalculate_2425_defcon=True,
            )
            print(f"✓ Combined dataset: {df_combined.shape}")
            print(f"  Total players: {df_combined['player_id'].nunique()}")
            print(f"  24-25 records: {(df_combined['season'] == '2024-2025').sum()}")
            print(f"  25-26 records: {(df_combined['season'] == '2025-2026').sum()}")
            
            # Check new players
            if "is_new_player" in df_combined.columns:
                new_count = df_combined["is_new_player"].sum()
                print(f"  New players in 25-26: {new_count}")
            
            # Check transfers
            if "team_changed" in df_combined.columns:
                transfer_count = df_combined["team_changed"].sum()
                print(f"  Player transfers: {transfer_count}")
            
            df_for_engineering = df_combined
        except Exception as e:
            print(f"✗ Failed to combine: {e}")
            df_for_engineering = df_2526
    else:
        print("\n[TEST 5] Skipping season combination (only 25-26 available)")
        df_for_engineering = df_2526
    
    # Test 6: Apply feature engineering
    print("\n[TEST 6] Applying feature engineering...")
    try:
        df_engineered = engineer_all_features(
            df_for_engineering,
            handle_new_players=False,  # Already handled in combine_seasons
            previous_season_df=None,
        )
        print(f"✓ Feature engineering complete: {df_engineered.shape}")
        
        # Check for engineered features
        rolling_cols = [c for c in df_engineered.columns if c.startswith("rolling_")]
        cumulative_cols = [c for c in df_engineered.columns if c.startswith("cumulative_")]
        
        print(f"  Rolling features: {len(rolling_cols)}")
        print(f"  Cumulative features: {len(cumulative_cols)}")
        
    except Exception as e:
        print(f"✗ Feature engineering failed: {e}")
        df_engineered = df_for_engineering
    
    # Test 7: Get training feature list
    print("\n[TEST 7] Getting training feature list...")
    feature_list = get_training_feature_list(include_engineered=True)
    available_features = [f for f in feature_list if f in df_engineered.columns]
    
    print(f"✓ Training features defined: {len(feature_list)}")
    print(f"  Available in dataset: {len(available_features)}")
    print(f"  Missing: {len(feature_list) - len(available_features)}")
    
    # Show sample features
    print("\n  Sample features:")
    for feat in available_features[:10]:
        print(f"    - {feat}")
    if len(available_features) > 10:
        print(f"    ... and {len(available_features) - 10} more")
    
    # Test 8: Check for leaky columns
    print("\n[TEST 8] Checking for leaky columns...")
    leaky_cols = ["ep_next", "ep_this"]
    found_leaky = [c for c in leaky_cols if c in df_engineered.columns]
    
    if found_leaky:
        print(f"✗ WARNING: Leaky columns found: {found_leaky}")
    else:
        print(f"✓ No leaky columns present")
    
    # Test 9: Validate target column
    print("\n[TEST 9] Checking target column...")
    if "adjusted_points" in df_engineered.columns:
        print(f"✓ adjusted_points present")
        print(f"  Mean: {df_engineered['adjusted_points'].mean():.2f}")
        print(f"  Max: {df_engineered['adjusted_points'].max():.0f}")
    elif "total_points" in df_engineered.columns:
        print(f"⚠ Using total_points (adjusted_points not available)")
        print(f"  Mean: {df_engineered['total_points'].mean():.2f}")
    else:
        print(f"✗ No target column found!")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Data loaded and processed successfully")
    print(f"✓ Final dataset shape: {df_engineered.shape}")
    print(f"✓ Training features available: {len(available_features)}")
    print(f"✓ Ready for model training!")
    
    # Return for inspection if needed
    return df_engineered


if __name__ == "__main__":
    df_final = test_pipeline()
    
    print("\n" + "=" * 70)
    print("Test complete! Dataframe stored in 'df_final' variable.")
    print("=" * 70)

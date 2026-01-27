#!/usr/bin/env python3
"""Display and summarize enhanced projection results."""

import pandas as pd
import sys

def main():
    df = pd.read_csv('outputs/projections_enhanced.csv')
    
    print("\n" + "="*100)
    print("ENHANCED PROJECTION RESULTS FOR GW24-29")
    print("="*100)
    print(f"\nTotal Players: {len(df)}")
    print(f"Gameweeks: 24-29 (6 weeks ahead)")
    print(f"Columns: {len(df.columns)}")
    
    # Show top 15 by GW24 projection
    print("\n" + "="*100)
    print("TOP 15 PROJECTIONS FOR GW24 (with 95% confidence intervals)")
    print("="*100)
    top_gw24 = df.nlargest(15, 'GW24_proj')[['player_id', 'web_name', 'position', 'role', 'GW24_proj', 'GW24_std', 'GW24_lower_95', 'GW24_upper_95']]
    for idx, (_, row) in enumerate(top_gw24.iterrows(), 1):
        print(f"{idx:2d}. {row['web_name']:15s} ({row['position']:10s} {row['role']:7s}) | "
              f"Proj: {row['GW24_proj']:6.2f} | Std: {row['GW24_std']:5.2f} | "
              f"95% CI: [{row['GW24_lower_95']:6.2f}, {row['GW24_upper_95']:6.2f}]")
    
    # Show projection spread across gameweeks
    print("\n" + "="*100)
    print("SAMPLE: WATKINS (TOP PLAYER) - PROJECTIONS ACROSS ALL 6 GAMEWEEKS")
    print("="*100)
    watkins = df[df['web_name'] == 'Watkins'].iloc[0]
    for gw in range(24, 30):
        proj = watkins[f'GW{gw}_proj']
        std = watkins[f'GW{gw}_std']
        lower = watkins[f'GW{gw}_lower_95']
        upper = watkins[f'GW{gw}_upper_95']
        print(f"GW{gw}: {proj:.2f} ± {std:.2f} [95% CI: {lower:.2f} - {upper:.2f}]")
    
    # Show some negatively projected players (bench/injured likely)
    print("\n" + "="*100)
    print("BOTTOM 10 PROJECTIONS FOR GW24 (likely injured/benched players)")
    print("="*100)
    bottom_gw24 = df.nsmallest(10, 'GW24_proj')[['player_id', 'web_name', 'position', 'role', 'GW24_proj', 'GW24_std']]
    for idx, (_, row) in enumerate(bottom_gw24.iterrows(), 1):
        print(f"{idx:2d}. {row['web_name']:15s} ({row['position']:10s} {row['role']:7s}) | "
              f"Proj: {row['GW24_proj']:7.3f} | Std: {row['GW24_std']:5.3f}")
    
    print("\n" + "="*100)
    print("STATISTICAL SUMMARY")
    print("="*100)
    print(f"\nGW24 Projection Statistics:")
    print(f"  Mean projection:   {df['GW24_proj'].mean():.2f} points")
    print(f"  Median projection: {df['GW24_proj'].median():.2f} points")
    print(f"  Std of projections: {df['GW24_proj'].std():.2f}")
    print(f"  Min/Max:           {df['GW24_proj'].min():.2f} / {df['GW24_proj'].max():.2f}")
    print(f"\nUncertainty (std dev) Statistics:")
    print(f"  Mean std dev:      {df['GW24_std'].mean():.2f}")
    print(f"  Min/Max:           {df['GW24_std'].min():.3f} / {df['GW24_std'].max():.3f}")
    print(f"\nConfidence Interval Width (95%):")
    print(f"  Mean CI width:     {(df['GW24_upper_95'] - df['GW24_lower_95']).mean():.2f}")
    print(f"  Min/Max:           {(df['GW24_upper_95'] - df['GW24_lower_95']).min():.2f} / {(df['GW24_upper_95'] - df['GW24_lower_95']).max():.2f}")
    
    print("\n" + "="*100)
    print("✅ All projections successfully generated with uncertainty bounds!")
    print("="*100 + "\n")

if __name__ == '__main__':
    main()

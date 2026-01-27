#!/usr/bin/env python3
"""Generate projections.json for site consumption.

Converts projections_enhanced.csv to optimized JSON format with:
- Top players by projection
- Statistical summaries
- Gameweek schedules
- Model metadata
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_projections(csv_path: Path) -> pd.DataFrame:
    """Load enhanced projections CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Projections not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"✅ Loaded {len(df)} player projections from {csv_path}")
    return df


def get_gameweeks(df: pd.DataFrame) -> list[int]:
    """Extract gameweek numbers from projection columns."""
    gw_cols = [col for col in df.columns if col.startswith('GW') and col.endswith('_proj')]
    gameweeks = sorted([int(col.split('_')[0][2:]) for col in gw_cols])
    return gameweeks


def build_player_projections(df: pd.DataFrame, gameweeks: list[int]) -> list[dict]:
    """Convert each player row to projection object."""
    players = []
    
    for _, row in df.iterrows():
        player_obj = {
            'player_id': int(row['player_id']),
            'name': str(row['web_name']),
            'team_code': int(row['team_code']) if pd.notna(row['team_code']) else None,
            'position': str(row['position']),
            'role': str(row['role']),
            'projections': {},
        }
        
        # Add gameweek projections
        for gw in gameweeks:
            proj_col = f'GW{gw}_proj'
            std_col = f'GW{gw}_std'
            lower_col = f'GW{gw}_lower_95'
            upper_col = f'GW{gw}_upper_95'
            
            if all(col in df.columns for col in [proj_col, std_col, lower_col, upper_col]):
                player_obj['projections'][f'GW{gw}'] = {
                    'projection': float(row[proj_col]) if pd.notna(row[proj_col]) else 0.0,
                    'uncertainty': float(row[std_col]) if pd.notna(row[std_col]) else 0.0,
                    'ci_lower': float(row[lower_col]) if pd.notna(row[lower_col]) else 0.0,
                    'ci_upper': float(row[upper_col]) if pd.notna(row[upper_col]) else 0.0,
                }
        
        players.append(player_obj)
    
    return players


def get_top_players(df: pd.DataFrame, gameweek: int, n: int = 20) -> list[dict]:
    """Get top N players for a specific gameweek."""
    proj_col = f'GW{gameweek}_proj'
    
    if proj_col not in df.columns:
        return []
    
    top = df.nlargest(n, proj_col)[
        ['player_id', 'web_name', 'position', 'role', proj_col]
    ].copy()
    
    top.rename(columns={proj_col: 'projection'}, inplace=True)
    top['player_id'] = top['player_id'].astype(int)
    
    return top.to_dict('records')


def build_statistics(df: pd.DataFrame, gameweeks: list[int]) -> dict:
    """Build summary statistics for projections."""
    stats = {}
    
    for gw in gameweeks:
        proj_col = f'GW{gw}_proj'
        std_col = f'GW{gw}_std'
        
        if proj_col in df.columns:
            stats[f'GW{gw}'] = {
                'mean_projection': float(df[proj_col].mean()),
                'median_projection': float(df[proj_col].median()),
                'std_projection': float(df[proj_col].std()),
                'min_projection': float(df[proj_col].min()),
                'max_projection': float(df[proj_col].max()),
            }
            
            if std_col in df.columns:
                stats[f'GW{gw}']['mean_uncertainty'] = float(df[std_col].mean())
    
    return stats


def generate_projections_json(
    csv_path: Path,
    output_path: Path,
    include_all_players: bool = False,
    top_n: int = 20,
) -> dict:
    """Generate complete projections JSON structure.
    
    Args:
        csv_path: Path to projections_enhanced.csv
        output_path: Path to save projections.json
        include_all_players: If True, include all players; if False, only top N per GW
        top_n: Number of top players to include per gameweek (if not including all)
        
    Returns:
        Complete projections dictionary
    """
    
    logger.info("="*80)
    logger.info("GENERATING PROJECTIONS.JSON FOR SITE")
    logger.info("="*80)
    
    # Load data
    logger.info("\n[1/4] Loading projections...")
    df = load_projections(csv_path)
    gameweeks = get_gameweeks(df)
    logger.info(f"✅ Found gameweeks: {gameweeks}")
    
    # Build structure
    logger.info("\n[2/4] Building player projections...")
    players = build_player_projections(df, gameweeks) if include_all_players else []
    logger.info(f"✅ Built {len(players)} player records")
    
    # Build statistics
    logger.info("\n[3/4] Computing statistics...")
    stats = build_statistics(df, gameweeks)
    logger.info(f"✅ Computed stats for {len(stats)} gameweeks")
    
    # Build top players
    logger.info("\n[4/4] Identifying top players per gameweek...")
    top_players = {}
    for gw in gameweeks:
        top_players[f'GW{gw}'] = get_top_players(df, gw, n=top_n)
    logger.info(f"✅ Identified top {top_n} players for each gameweek")
    
    # Assemble final structure
    projections_json = {
        'meta': {
            'generated_at': datetime.now().isoformat(),
            'version': '1.0',
            'gameweeks': gameweeks,
            'total_players': len(df),
            'model': {
                'type': 'LSTM + Ensemble',
                'features': ['form', 'fixture_difficulty', 'opponent_strength', 'uncertainty'],
            },
        },
        'top_players': top_players,
        'statistics': stats,
    }
    
    # Add all players if requested
    if include_all_players:
        projections_json['players'] = players
    
    # Save to file
    logger.info(f"\n[5/5] Saving JSON...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(projections_json, f, indent=2)
    
    logger.info(f"✅ Saved to {output_path}")
    logger.info(f"   Size: {len(json.dumps(projections_json))} bytes")
    
    logger.info("\n" + "="*80)
    logger.info("✅ PROJECTIONS.JSON GENERATION COMPLETE")
    logger.info("="*80)
    
    return projections_json


def generate_summary_json(
    csv_path: Path,
    output_path: Path,
) -> dict:
    """Generate minimal summary JSON for quick loading.
    
    Contains only top 5 players per gameweek + statistics.
    """
    
    logger.info("Generating summary JSON (lightweight)...")
    
    df = load_projections(csv_path)
    gameweeks = get_gameweeks(df)
    
    summary = {
        'meta': {
            'generated_at': datetime.now().isoformat(),
            'gameweeks': gameweeks,
        },
        'top_5': {},
        'stats': build_statistics(df, gameweeks),
    }
    
    for gw in gameweeks:
        summary['top_5'][f'GW{gw}'] = get_top_players(df, gw, n=5)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✅ Saved summary to {output_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate projections.json for site consumption"
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("outputs/projections_enhanced.csv"),
        help="Path to projections_enhanced.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("site"),
        help="Output directory for projections.json",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include all players in JSON (larger file)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top players per gameweek to include",
    )
    
    args = parser.parse_args()
    
    # Generate main JSON
    generate_projections_json(
        csv_path=args.csv_path,
        output_path=args.output_dir / "projections.json",
        include_all_players=args.include_all,
        top_n=args.top_n,
    )
    
    # Also generate lightweight summary
    generate_summary_json(
        csv_path=args.csv_path,
        output_path=args.output_dir / "projections_summary.json",
    )

"""Fixture difficulty and opponent strength features with FPL API integration.

Builds opponent strength metrics from actual fixtures to enhance prediction signals
with fixture-driven difficulty modulation.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def load_all_fixtures(repo_root: Path, season: str) -> pd.DataFrame:
    """Load and combine fixtures from all gameweeks.
    
    Args:
        repo_root: Repository root containing FPL-Core-Insights data
        season: Season identifier (e.g., "2025-2026")
        
    Returns:
        Combined fixtures DataFrame with gameweek and opponent mappings
    """
    fixtures_path = repo_root / "FPL-Core-Insights" / "data" / season / "By Gameweek"
    
    all_fixtures = []
    gw_dirs = sorted([d for d in fixtures_path.iterdir() if d.is_dir() and d.name.startswith("GW")])
    
    for gw_dir in gw_dirs:
        fixture_file = gw_dir / "fixtures.csv"
        if fixture_file.exists():
            gw_fixtures = pd.read_csv(fixture_file)
            all_fixtures.append(gw_fixtures)
    
    if not all_fixtures:
        logger.warning(f"No fixtures found in {fixtures_path}")
        return pd.DataFrame()
    
    fixtures = pd.concat(all_fixtures, ignore_index=True)
    return fixtures


def load_team_lookup(repo_root: Path, season: str) -> pd.DataFrame:
    """Load team code mapping from teams.csv.
    
    Args:
        repo_root: Repository root
        season: Season identifier
        
    Returns:
        DataFrame with columns: code (team_code), name (team_name), id, etc.
    """
    teams_file = repo_root / "FPL-Core-Insights" / "data" / season / "teams.csv"
    
    if teams_file.exists():
        teams = pd.read_csv(teams_file)
        return teams
    else:
        logger.warning(f"Teams file not found: {teams_file}")
        return pd.DataFrame()


def build_opponent_lookup(
    *,
    fixtures: pd.DataFrame,
    teams: pd.DataFrame,
) -> pd.DataFrame:
    """Build home_team -> away_team_code and away_team -> home_team_code mappings.
    
    Creates a lookup table showing which team each team faces in each GW and their
    opponent's team code (for merging opponent strength metrics).
    
    Args:
        fixtures: DataFrame with gameweek, home_team, away_team
        teams: DataFrame with id -> code mapping
        
    Returns:
        DataFrame with columns: gw, team_code, opponent_team_code, is_home
    """
    
    if fixtures.empty or teams.empty:
        return pd.DataFrame()
    
    # Create team_id -> code mapping
    team_id_to_code = dict(zip(teams['id'], teams['code']))
    
    # Extract home fixtures
    home_fixtures = fixtures[['gameweek', 'home_team', 'away_team']].copy()
    home_fixtures.rename(columns={'gameweek': 'gw', 'home_team': 'team_id', 'away_team': 'opp_team_id'}, inplace=True)
    home_fixtures['is_home'] = 1
    
    # Extract away fixtures
    away_fixtures = fixtures[['gameweek', 'home_team', 'away_team']].copy()
    away_fixtures.rename(columns={'gameweek': 'gw', 'away_team': 'team_id', 'home_team': 'opp_team_id'}, inplace=True)
    away_fixtures['is_home'] = 0
    
    # Combine
    all_fixtures = pd.concat([home_fixtures, away_fixtures], ignore_index=True)
    
    # Map team IDs to codes
    all_fixtures['team_code'] = all_fixtures['team_id'].map(team_id_to_code)
    all_fixtures['opponent_team_code'] = all_fixtures['opp_team_id'].map(team_id_to_code)
    
    # Filter valid rows
    all_fixtures = all_fixtures.dropna(subset=['team_code', 'opponent_team_code'])
    
    return all_fixtures[['gw', 'team_code', 'opponent_team_code', 'is_home']].astype({
        'gw': 'int32',
        'team_code': 'int32',
        'opponent_team_code': 'int32',
        'is_home': 'int32',
    })


def build_opponent_strength_table(
    *,
    df: pd.DataFrame,
    rolling_window: int = 6,
) -> pd.DataFrame:
    """Build rolling opponent defensive strength metrics per GW/team.
    
    Aggregates opponent performance (goals conceded, xGC, clean sheets) across
    all players to estimate defensive difficulty of facing each opponent.
    
    Args:
        df: Player gameweek stats with gw, team_code, goals_conceded, expected_goals_conceded, clean_sheets
        rolling_window: Window size for rolling average
        
    Returns:
        DataFrame with opponent strength metrics indexed by (gw, team_code)
    """
    
    # Group by gw and team_code (opponent) to get aggregate defensive stats
    opponent_agg = df.groupby(['gw', 'team_code'], as_index=False).agg({
        'goals_conceded': 'mean',
        'expected_goals_conceded': 'mean',
        'clean_sheets': 'mean',
    }).rename(columns={
        'goals_conceded': 'opp_gc_avg',
        'expected_goals_conceded': 'opp_xgc_avg',
        'clean_sheets': 'opp_cs_avg',
    })
    
    # Sort for rolling window calculation
    opponent_agg = opponent_agg.sort_values(['team_code', 'gw'])
    
    # Compute rolling averages within each team (looking back at their past defensive performance)
    opponent_agg['rolling_opp_gc'] = (
        opponent_agg.groupby('team_code')['opp_gc_avg']
        .transform(lambda x: x.rolling(window=rolling_window, min_periods=1).mean())
    )
    opponent_agg['rolling_opp_xgc'] = (
        opponent_agg.groupby('team_code')['opp_xgc_avg']
        .transform(lambda x: x.rolling(window=rolling_window, min_periods=1).mean())
    )
    opponent_agg['rolling_opp_cs_rate'] = (
        opponent_agg.groupby('team_code')['opp_cs_avg']
        .transform(lambda x: x.rolling(window=rolling_window, min_periods=1).mean())
    )
    
    # Compute opponent strength index (defensive quality)
    # Higher GC conceded = weaker defense = easier to score against
    opponent_agg['opponent_def_strength'] = (
        1.0 / (opponent_agg['rolling_opp_gc'] + 0.1)  # +0.1 to avoid division by zero
    )
    
    return opponent_agg[[
        'gw', 'team_code', 
        'rolling_opp_gc', 'rolling_opp_xgc', 'rolling_opp_cs_rate',
        'opponent_def_strength'
    ]]


def add_fixture_features(
    df: pd.DataFrame,
    opponent_strength: pd.DataFrame,
    fixture_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Merge fixture information and opponent strength metrics onto player data.
    
    For each player's current gameweek, attach:
    1. Opponent team code (from fixture_lookup)
    2. Rolling opponent defensive metrics (from opponent_strength)
    3. Home/away adjustment multiplier
    
    Args:
        df: Player gameweek stats with gw, team_code
        opponent_strength: Pre-computed opponent strength table (gw, team_code -> strength metrics)
        fixture_lookup: Fixture lookup with (gw, team_code) -> opponent_team_code, is_home
        
    Returns:
        DataFrame with added fixture features
    """
    
    df = df.copy()
    
    # Merge fixture information (who did this team face this GW?)
    df = df.merge(
        fixture_lookup[['gw', 'team_code', 'opponent_team_code', 'is_home']],
        on=['gw', 'team_code'],
        how='left',
    )
    
    # Merge opponent strength (how good is the opponent's defense?)
    df = df.merge(
        opponent_strength[
            ['gw', 'team_code', 'rolling_opp_gc', 'rolling_opp_xgc', 
             'rolling_opp_cs_rate', 'opponent_def_strength']
        ],
        left_on=['gw', 'opponent_team_code'],
        right_on=['gw', 'team_code'],
        how='left',
        suffixes=('', '_opp'),
    )
    
    # Apply home/away adjustment
    # Attackers get ~2% boost at home, defenders get ~1.5% boost at home
    df['fixture_difficulty_adj'] = np.where(
        df['is_home'] == 1,
        df['opponent_def_strength'] * 1.02,  # Home advantage
        df['opponent_def_strength'] * 0.98,   # Away penalty
    )
    
    # Fill missing values with league average
    league_avg_strength = df['opponent_def_strength'].mean()
    league_avg_gc = df['rolling_opp_gc'].mean()
    
    df['opponent_def_strength'] = df['opponent_def_strength'].fillna(league_avg_strength)
    df['fixture_difficulty_adj'] = df['fixture_difficulty_adj'].fillna(league_avg_strength)
    df['rolling_opp_gc'] = df['rolling_opp_gc'].fillna(league_avg_gc)
    
    return df


def integrate_fixture_features(
    *,
    df: pd.DataFrame,
    repo_root: Path,
    season: str,
    graceful_fallback: bool = True,
) -> pd.DataFrame:
    """Full integration: load fixtures, build strength metrics, merge onto player data.
    
    Args:
        df: Player gameweek stats
        repo_root: Repository root
        season: Season identifier
        graceful_fallback: If True, return original df on any error; if False, raise
        
    Returns:
        DataFrame with fixture features merged
    """
    
    try:
        # Load fixtures and team mappings
        fixtures = load_all_fixtures(repo_root, season)
        teams = load_team_lookup(repo_root, season)
        
        if fixtures.empty or teams.empty:
            logger.warning("Fixtures or teams data empty")
            if graceful_fallback:
                return df
            raise ValueError("Missing fixture or team data")
        
        # Build lookup tables
        fixture_lookup = build_opponent_lookup(fixtures=fixtures, teams=teams)
        opponent_strength = build_opponent_strength_table(df=df, rolling_window=6)
        
        if fixture_lookup.empty or opponent_strength.empty:
            logger.warning("Failed to build fixture/opponent tables")
            if graceful_fallback:
                return df
            raise ValueError("Failed to build lookup tables")
        
        # Merge features
        df_with_fixtures = add_fixture_features(df, opponent_strength, fixture_lookup)
        
        logger.info("✅ Fixture features integrated successfully")
        return df_with_fixtures
        
    except Exception as e:
        logger.warning(f"⚠️  Fixture features failed ({e}); continuing without")
        if graceful_fallback:
            return df
        raise

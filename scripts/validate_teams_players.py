"""Validate teams and players for 2025-2026 season.

This script checks:
1. Promoted teams (Leeds, Burnley, Sunderland) are present
2. Relegated teams (Leicester, Southampton, Ipswich) are absent
3. New players identified and counted
"""

from pathlib import Path

import pandas as pd

from src.fpl_projection.data_loading import (
    load_insights_playerstats,
    load_insights_teams,
    load_premier_league_gameweek_stats,
)
from src.fpl_projection.new_entities import (
    PROMOTED_TEAMS_2025_26,
    RELEGATED_TEAMS_2024_25,
    identify_new_players,
    identify_removed_players,
    validate_teams,
)


def main():
    repo_root = Path(__file__).parent.parent
    
    print("=" * 60)
    print("VALIDATING 2025-2026 SEASON DATA")
    print("=" * 60)
    
    # Load teams
    print("\n1. Loading teams data...")
    teams_2526 = load_insights_teams(repo_root=repo_root, season="2025-2026")
    
    print(f"\nTeams in 2025-2026 season: {len(teams_2526)}")
    print("\nTeam names:")
    for name in sorted(teams_2526["name"].tolist()):
        print(f"  - {name}")
    
    # Validate promoted/relegated
    print("\n" + "=" * 60)
    print("2. Validating team changes...")
    print("=" * 60)
    
    validation = validate_teams(
        teams_2526,
        expected_promoted=PROMOTED_TEAMS_2025_26,
        expected_relegated=RELEGATED_TEAMS_2024_25,
        team_col="name",
    )
    
    # Load player data
    print("\n" + "=" * 60)
    print("3. Loading player data...")
    print("=" * 60)
    
    print("\nLoading 2025-2026 player stats...")
    players_2526 = load_insights_playerstats(
        repo_root=repo_root,
        season="2025-2026",
        include_optional=False
    )
    
    print(f"Loaded {len(players_2526)} player-gameweek records")
    print(f"Unique players: {players_2526['player_id'].nunique()}")
    
    # Try to load previous season
    print("\nAttempting to load 2024-2025 player stats...")
    try:
        players_2425_files = list((repo_root / "FPL-Core-Insights" / "data" / "2024-2025" / "playerstats").glob("*.csv"))
        
        if players_2425_files:
            print(f"Found {len(players_2425_files)} player stats files for 2024-2025")
            # Load first file as sample
            players_2425_sample = pd.read_csv(players_2425_files[0])
            if "id" in players_2425_sample.columns:
                players_2425_sample = players_2425_sample.rename(columns={"id": "player_id"})
            
            print(f"Sample file has {players_2425_sample['player_id'].nunique()} unique players")
            
            # Concatenate all files for full comparison
            print("Loading all 2024-2025 files...")
            all_2425 = []
            for f in players_2425_files:
                df = pd.read_csv(f)
                if "id" in df.columns:
                    df = df.rename(columns={"id": "player_id"})
                all_2425.append(df)
            players_2425 = pd.concat(all_2425, ignore_index=True)
            
            print(f"Total 2024-2025 records: {len(players_2425)}")
            print(f"Unique players in 2024-2025: {players_2425['player_id'].nunique()}")
            
            # Identify new/removed players
            print("\n" + "=" * 60)
            print("4. Identifying player changes...")
            print("=" * 60)
            
            new_players = identify_new_players(players_2526, players_2425)
            removed_players = identify_removed_players(players_2526, players_2425)
            
            print(f"\nNew players in 2025-2026: {len(new_players)}")
            print(f"Removed players from 2024-2025: {len(removed_players)}")
            
            # Show sample new players
            if len(new_players) > 0:
                print("\nSample new players (first 10):")
                new_player_sample = players_2526[players_2526["player_id"].isin(new_players)]
                
                if "web_name" in new_player_sample.columns:
                    sample_names = (
                        new_player_sample[["player_id", "web_name"]]
                        .drop_duplicates()
                        .head(10)
                    )
                    for _, row in sample_names.iterrows():
                        print(f"  ID {row['player_id']}: {row['web_name']}")
                else:
                    print(f"  Player IDs: {sorted(list(new_players))[:10]}")
        
        else:
            print("No 2024-2025 playerstats files found - cannot compare seasons")
            print("(This is expected if 2024-2025 data is not in the dataset)")
    
    except Exception as e:
        print(f"Could not load 2024-2025 data: {e}")
        print("(This is expected if that season's data is not available)")
    
    # Load gameweek stats to check features
    print("\n" + "=" * 60)
    print("5. Checking feature engineering...")
    print("=" * 60)
    
    print("\nLoading gameweek stats with feature engineering...")
    gw_stats = load_premier_league_gameweek_stats(
        repo_root=repo_root,
        season="2025-2026",
        apply_feature_engineering=True
    )
    
    print(f"Loaded {len(gw_stats)} gameweek records")
    print(f"Total columns: {len(gw_stats.columns)}")
    
    # Check for rolling/cumulative features
    rolling_cols = [c for c in gw_stats.columns if c.startswith("rolling_")]
    cumulative_cols = [c for c in gw_stats.columns if c.startswith("cumulative_")]
    
    print(f"\nRolling features: {len(rolling_cols)}")
    for col in sorted(rolling_cols)[:10]:
        print(f"  - {col}")
    if len(rolling_cols) > 10:
        print(f"  ... and {len(rolling_cols) - 10} more")
    
    print(f"\nCumulative features: {len(cumulative_cols)}")
    for col in sorted(cumulative_cols):
        print(f"  - {col}")
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    print(f"\n✓ Teams loaded: {len(teams_2526)}")
    print(f"✓ Expected promoted teams found: {len(validation['promoted_found'])}/{len(PROMOTED_TEAMS_2025_26)}")
    
    if validation['promoted_missing']:
        print(f"  ⚠ Missing: {validation['promoted_missing']}")
    
    print(f"✓ Expected relegated teams removed: {len(validation['relegated_removed'])}/{len(RELEGATED_TEAMS_2024_25)}")
    
    if validation['relegated_still_present']:
        print(f"  ⚠ WARNING: Still present: {validation['relegated_still_present']}")
    
    print(f"\n✓ Players in 2025-2026: {players_2526['player_id'].nunique()}")
    print(f"✓ Gameweek records: {len(gw_stats)}")
    print(f"✓ Rolling features: {len(rolling_cols)}")
    print(f"✓ Cumulative features: {len(cumulative_cols)}")
    
    print("\n" + "=" * 60)
    print("Validation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

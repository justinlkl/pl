# New Players & Teams Handling

## Overview

This document explains how the FPL projection system handles new players and team changes (promotions/relegations) between seasons.

## Team Changes (2025-2026 Season)

### Relegated Teams (Not in 2025-2026)
The following teams were relegated from the 2024-2025 season:
- **Leicester** / Leicester City
- **Southampton**
- **Ipswich** / Ipswich Town

### Promoted Teams (Added in 2025-2026)
The following teams were promoted for the 2025-2026 season:
- **Leeds**
- **Burnley**
- **Sunderland**

### Validation

The `validate_teams()` function in `src/fpl_projection/new_entities.py` checks that:
1. Promoted teams are present in the current season
2. Relegated teams are absent from the current season

## New Player Handling

### Problem
Players who joined in the 2025-2026 season have no historical data from previous seasons. This means they lack:
- Rolling aggregates (rolling_3_xg, rolling_5_minutes, etc.)
- Cumulative statistics (cumulative_xg, cumulative_xa, etc.)

Without these features, the ML model cannot make predictions for new players.

### Solution: Position-Based Priors

The system fills missing features for new players using **position-average priors** from the current season:

```python
from src.fpl_projection.new_entities import handle_new_players_full_pipeline

# This is automatically called during feature engineering
df_filled = handle_new_players_full_pipeline(
    df_current=current_season_data,
    df_previous=previous_season_data,  # Optional
    player_id_col="player_id",
    position_col="position",
)
```

### How It Works

1. **Identify New Players**: Compare player IDs between current and previous season
2. **Calculate Position Priors**: For each position (GK, DEF, MID, FWD), calculate average values for:
   - Rolling features (rolling_3_xg, rolling_5_minutes, etc.)
   - Cumulative features (cumulative_xg, cumulative_xa, etc.)
3. **Fill Missing Features**: For new players with missing/zero values:
   - **Rolling features**: Filled with position average
   - **Cumulative features**: Filled with 0 (no history yet)

### Example

For a new MID player:
- `rolling_5_xg` → Average rolling_5_xg for all MIDs
- `rolling_5_minutes` → Average rolling_5_minutes for all MIDs
- `cumulative_xg` → 0 (new player, no cumulative history)

### Integration

The new player handling is automatically integrated into the feature engineering pipeline:

```python
# In src/fpl_projection/data_loading.py
df = load_premier_league_gameweek_stats(
    repo_root=repo_root,
    season="2025-2026",
    previous_season="2024-2025",  # Enable new player handling
    apply_feature_engineering=True,
    handle_new_players=True,
)
```

## Functions

### Core Functions

#### `identify_new_players(df_current, df_previous)`
Identifies player IDs that are in current season but not in previous season.

**Returns**: Set of new player IDs

#### `identify_removed_players(df_current, df_previous)`
Identifies player IDs that were in previous season but not in current season.

**Returns**: Set of removed player IDs

#### `calculate_position_priors(df, feature_columns)`
Calculates average feature values for each position (GK, DEF, MID, FWD).

**Returns**: Dictionary mapping position → Series of average feature values

#### `fill_new_player_features(df, new_player_ids, position_priors)`
Fills missing rolling/cumulative features for new players using position priors.

**Returns**: DataFrame with filled features

#### `handle_new_players_full_pipeline(df_current, df_previous)`
Complete pipeline that:
1. Identifies new players
2. Calculates position priors
3. Fills missing features

**Returns**: DataFrame with new player features filled

### Team Functions

#### `filter_relegated_teams(df, team_col, relegated_teams)`
Removes records for players from relegated teams.

**Returns**: Filtered DataFrame

#### `validate_teams(df, expected_promoted, expected_relegated)`
Validates that promoted teams are present and relegated teams are absent.

**Returns**: Dictionary with validation results

## Usage Examples

### Validate Teams

```python
from pathlib import Path
from src.fpl_projection.data_loading import load_insights_teams
from src.fpl_projection.new_entities import (
    validate_teams,
    PROMOTED_TEAMS_2025_26,
    RELEGATED_TEAMS_2024_25,
)

teams = load_insights_teams(repo_root=Path("."), season="2025-2026")

validation = validate_teams(
    teams,
    expected_promoted=PROMOTED_TEAMS_2025_26,
    expected_relegated=RELEGATED_TEAMS_2024_25,
    team_col="name",
)

print(validation["promoted_found"])  # ['Leeds', 'Burnley', 'Sunderland']
print(validation["relegated_removed"])  # ['Leicester', 'Southampton', 'Ipswich']
```

### Handle New Players Manually

```python
from pathlib import Path
from src.fpl_projection.data_loading import load_premier_league_gameweek_stats
from src.fpl_projection.new_entities import handle_new_players_full_pipeline

# Load current season
df_2526 = load_premier_league_gameweek_stats(
    repo_root=Path("."),
    season="2025-2026",
    apply_feature_engineering=False,  # Don't apply yet
)

# Load previous season (optional)
try:
    df_2425 = load_premier_league_gameweek_stats(
        repo_root=Path("."),
        season="2024-2025",
        apply_feature_engineering=False,
    )
except:
    df_2425 = None

# Apply feature engineering (will auto-handle new players)
from src.fpl_projection.feature_engineering import engineer_all_features

df_final = engineer_all_features(
    df_2526,
    handle_new_players=True,
    previous_season_df=df_2425,
)
```

### Run Validation Script

```bash
python scripts/validate_teams_players.py
```

This script will:
1. Load and display all teams in 2025-2026
2. Validate promoted/relegated teams
3. Identify new and removed players
4. Check that rolling/cumulative features are present

## Implementation Details

### Position Normalization

The system normalizes position strings to standard values:
- GK/Goalkeeper → `GK`
- DEF/Defender/DF → `DEF`
- MID/Midfielder/MF → `MID`
- FWD/Forward/FW/Striker → `FWD`

### Feature Lists

**Rolling Features** (filled with position average):
- `rolling_3_xg`, `rolling_3_xa`, `rolling_3_xgi`
- `rolling_5_xg`, `rolling_5_xa`, `rolling_5_xgi`
- `rolling_5_points`, `rolling_5_minutes`, `rolling_5_appearances`
- `rolling_5_defensive`, `rolling_5_defensive_def`, `rolling_5_defensive_gk`, etc.

**Cumulative Features** (filled with 0):
- `cumulative_xg`, `cumulative_xa`, `cumulative_xgi`

### Automatic vs Manual

**Automatic** (Recommended):
```python
# Automatically handles new players
df = load_premier_league_gameweek_stats(
    repo_root=Path("."),
    season="2025-2026",
    previous_season="2024-2025",
    apply_feature_engineering=True,
    handle_new_players=True,
)
```

**Manual** (For custom workflows):
```python
# Load raw data
df = load_premier_league_gameweek_stats(
    repo_root=Path("."),
    season="2025-2026",
    apply_feature_engineering=False,
)

# Apply feature engineering with custom settings
from src.fpl_projection.feature_engineering import engineer_all_features

df = engineer_all_features(
    df,
    handle_new_players=False,  # Disable new player handling
    previous_season_df=None,
)
```

## Files

- **`src/fpl_projection/new_entities.py`**: Core new player/team handling functions
- **`src/fpl_projection/feature_engineering.py`**: Updated to call new player handling
- **`src/fpl_projection/data_loading.py`**: Updated to optionally load previous season
- **`scripts/validate_teams_players.py`**: Validation script for teams and players

## Notes

- New player handling is **optional** and only runs when:
  1. `apply_feature_engineering=True`
  2. `handle_new_players=True`
  3. `previous_season` is provided (or `previous_season_df` is passed to `engineer_all_features`)

- If previous season data is unavailable, new players will have **missing/zero** values for rolling/cumulative features, which will be handled by the median imputer during preprocessing

- The position priors are calculated from the **current season** data, not historical seasons, to ensure they reflect current gameplay patterns

- Relegated team filtering can be applied manually using `filter_relegated_teams()` if needed

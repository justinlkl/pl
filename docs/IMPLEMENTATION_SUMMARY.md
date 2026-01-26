# Implementation Summary: New Players & Teams Handling

## What Was Implemented

### ✅ 1. Team Validation (Promoted/Relegated)
**Files**: 
- `src/fpl_projection/new_entities.py`

**Features**:
- Identified relegated teams: **Leicester**, **Southampton**, **Ipswich**
- Identified promoted teams: **Leeds**, **Burnley**, **Sunderland**
- `validate_teams()` function checks dataset contains correct teams
- `filter_relegated_teams()` function removes relegated team players

**Validation Result**:
```
✅ Promoted teams found: 3/3 (Leeds, Burnley, Sunderland)
✅ Relegated teams removed: 5/5 (all variants removed)
✅ Total teams in 2025-2026: 20 teams
```

### ✅ 2. New Player Identification
**Files**: 
- `src/fpl_projection/new_entities.py`

**Features**:
- `identify_new_players()`: Finds players in current season not in previous
- `identify_removed_players()`: Finds players removed from previous season

**Example Output**:
```python
new_players = identify_new_players(df_2526, df_2425)
# Returns: {player_id_1, player_id_2, ...}
```

### ✅ 3. Position-Based Priors for New Players
**Files**: 
- `src/fpl_projection/new_entities.py`

**Features**:
- `calculate_position_priors()`: Calculates average features per position (GK/DEF/MID/FWD)
- `fill_new_player_features()`: Fills missing rolling/cumulative features for new players

**Strategy**:
For each new player:
1. Determine their position (GK/DEF/MID/FWD)
2. **Rolling features** → Fill with position average from current season
3. **Cumulative features** → Fill with 0 (no history yet)

**Rolling Features Filled**:
- `rolling_3_xg`, `rolling_3_xa`, `rolling_3_xgi`
- `rolling_5_xg`, `rolling_5_xa`, `rolling_5_xgi`
- `rolling_5_points`, `rolling_5_minutes`, `rolling_5_appearances`
- `rolling_5_defensive`, `rolling_5_defensive_def`, `rolling_5_defensive_gk`, etc.

**Cumulative Features Filled**:
- `cumulative_xg`, `cumulative_xa`, `cumulative_xgi`

### ✅ 4. Automatic Integration
**Files**: 
- `src/fpl_projection/feature_engineering.py` (updated)
- `src/fpl_projection/data_loading.py` (updated)

**Changes**:
```python
# Updated function signature
def load_premier_league_gameweek_stats(
    *, 
    repo_root: Path, 
    season: str, 
    apply_feature_engineering: bool = True,
    previous_season: str | None = None,  # NEW
    handle_new_players: bool = True,     # NEW
) -> pd.DataFrame:
    ...
```

**Usage**:
```python
# Automatically handles new players
df = load_premier_league_gameweek_stats(
    repo_root=Path("."),
    season="2025-2026",
    previous_season="2024-2025",  # Enable new player handling
    apply_feature_engineering=True,
    handle_new_players=True,
)
```

### ✅ 5. Validation Script
**Files**: 
- `scripts/validate_teams_players.py`

**What It Does**:
1. Loads 2025-2026 teams and validates promoted/relegated
2. Loads player data and identifies new/removed players
3. Checks that rolling/cumulative features are present
4. Provides comprehensive validation report

**Run**:
```bash
python scripts/validate_teams_players.py
```

### ✅ 6. Documentation
**Files**: 
- `docs/NEW_PLAYERS_TEAMS.md`

**Contains**:
- Complete explanation of team changes
- New player handling strategy
- Function reference
- Usage examples
- Implementation details

## How It Works

### Flow Diagram

```
1. Load current season (2025-2026)
   ↓
2. [Optional] Load previous season (2024-2025)
   ↓
3. Apply feature engineering:
   - Calculate per-90 metrics
   - Calculate rolling features (3-game, 5-game)
   - Calculate cumulative features
   ↓
4. [If previous_season provided] Handle new players:
   - Identify new player IDs
   - Calculate position priors from current season
   - Fill missing rolling features with position average
   - Fill missing cumulative features with 0
   ↓
5. Return complete dataframe ready for training/prediction
```

### Example: New Midfielder

```python
# New MID player joins in 2025-2026
# Before filling:
{
    "player_id": 12345,
    "position": "MID",
    "rolling_5_xg": NaN,          # Missing
    "rolling_5_minutes": NaN,     # Missing
    "cumulative_xg": NaN,         # Missing
}

# After position-based filling:
{
    "player_id": 12345,
    "position": "MID",
    "rolling_5_xg": 0.45,         # MID average
    "rolling_5_minutes": 280.5,   # MID average
    "cumulative_xg": 0.0,         # No history
}
```

## Testing

### Quick Test
```python
from pathlib import Path
from src.fpl_projection.data_loading import load_insights_teams
from src.fpl_projection.new_entities import validate_teams

teams = load_insights_teams(repo_root=Path("."), season="2025-2026")
validate_teams(teams, team_col="name")
```

**Expected Output**:
```
=== Team Validation ===
Promoted teams found: 3/3
Relegated teams removed: 5/5
Total teams in dataset: 20
```

### Full Validation
```bash
python scripts/validate_teams_players.py
```

## Benefits

1. **No Manual Data Fixing**: Automatically handles new players
2. **Position-Aware**: Uses appropriate priors for each position
3. **Flexible**: Works even if previous season data is unavailable
4. **Backward Compatible**: Can be disabled with `handle_new_players=False`
5. **Validated**: Checks promoted/relegated teams are correct

## Future Enhancements

Potential improvements (not yet implemented):

1. **Team-Level Priors**: Use team strength instead of just position
2. **Transfer Market Data**: Incorporate transfer fee/value for better priors
3. **Similar Player Matching**: Find similar players from previous season
4. **Confidence Scores**: Flag predictions for new players as lower confidence

## Files Changed

| File | Changes |
|------|---------|
| `src/fpl_projection/new_entities.py` | **NEW** - Core new player/team handling |
| `src/fpl_projection/feature_engineering.py` | Updated `engineer_all_features()` signature |
| `src/fpl_projection/data_loading.py` | Updated `load_premier_league_gameweek_stats()` signature |
| `scripts/validate_teams_players.py` | **NEW** - Validation script |
| `docs/NEW_PLAYERS_TEAMS.md` | **NEW** - Documentation |

## Constants

```python
# Relegated 2024-2025
RELEGATED_TEAMS_2024_25 = {
    "Leicester", "Leicester City",
    "Southampton", 
    "Ipswich", "Ipswich Town",
}

# Promoted 2025-2026
PROMOTED_TEAMS_2025_26 = {
    "Leeds", "Burnley", "Sunderland"
}
```

---

**Status**: ✅ **Complete and Ready to Use**

The system now automatically handles:
- ✅ Promoted teams (Leeds, Burnley, Sunderland)
- ✅ Relegated teams removed (Leicester, Southampton, Ipswich)
- ✅ New players filled with position-based priors
- ✅ Removed players identified and handled

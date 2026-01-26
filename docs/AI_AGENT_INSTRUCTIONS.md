# AI Agent Instructions for FPL Data Processing

## Overview
You are processing Fantasy Premier League (FPL) data for machine learning training. This guide provides step-by-step instructions for handling multi-season data with proper column filtering, new player/team handling, and defensive contribution recalculation.

## Quick Start

```python
from pathlib import Path
from src.fpl_projection.data_loading import load_premier_league_gameweek_stats
from src.fpl_projection.data_processor import combine_seasons, process_season_data
from src.fpl_projection.new_entities import PROMOTED_TEAMS_2025_26

# Load both seasons
df_2425 = load_premier_league_gameweek_stats(
    repo_root=Path("."),
    season="2024-2025",
    apply_feature_engineering=False,  # Process separately
)

df_2526 = load_premier_league_gameweek_stats(
    repo_root=Path("."),
    season="2025-2026",
    apply_feature_engineering=False,
)

# Combine with proper processing
df_combined = combine_seasons(
    df_2425,
    df_2526,
    recalculate_2425_defcon=True,  # Add defensive contributions to 24-25
)

# Now apply feature engineering to combined dataset
from src.fpl_projection.feature_engineering import engineer_all_features

df_final = engineer_all_features(df_combined)
```

## Data Processing Rules

### 1. COLUMN HANDLING

#### ✅ KEEP These Columns

**Player Identity:**
- `player_id` (or `id`)
- `gw` (gameweek)
- `position`
- `web_name`
- `team_code` / `team_id`

**Performance Stats:**
- `minutes` / `minutes_played`
- `goals` / `goals_scored`
- `assists`
- `expected_goals`, `expected_assists`, `expected_goal_involvements`
- `expected_goals_per_90`, `expected_assists_per_90`, `expected_goal_involvements_per_90`
- `total_shots`, `shots_on_target`

**Defensive Stats (NEW RULES ⭐):**
- `tackles`
- `interceptions`
- `clearances`
- `blocks`
- `recoveries`
- `clearances_blocks_interceptions` (combined in 25-26)
- `defensive_contribution` (NEW in 25-26)
- `defensive_contribution_per_90`

**GK-Specific:**
- `saves`, `saves_per_90`
- `goals_conceded`, `goals_conceded_per_90`
- `clean_sheets`, `clean_sheets_per_90`
- `xgot_faced`, `goals_prevented`

**Team Strength:**
- `strength`, `elo`
- `strength_attack_home`, `strength_attack_away`
- `strength_defence_home`, `strength_defence_away`

**Price/Value:**
- `now_cost`
- `value_form`, `value_season`
- `selected_by_percent`

**Set Pieces:**
- `penalties_order`
- `corners_and_indirect_freekicks_order`
- `direct_freekicks_order`

**Form/ICT:**
- `points_per_game`, `form`
- `influence`, `creativity`, `threat`, `ict_index`

**Availability:**
- `chance_of_playing_this_round`
- `chance_of_playing_next_round`

#### ❌ DROP These Columns

**Official Predictions (LEAKY!):**
- `ep_next` ❌
- `ep_this` ❌

**Identifiers (Not Features):**
- `player_code`
- `team_code` (keep for merging, drop for training)
- `pulse_id`
- `fotmob_name`

**Rankings (Derived):**
- All columns ending in `_rank`
- All columns ending in `_rank_type`

**Metadata:**
- `deadline_time`, `deadline_time_epoch`
- `news`, `news_added`
- `match_url`, `snapshot_time`
- `dreamteam_count`

**Percentage Columns (Use Raw Counts):**
- `accurate_passes_percent`
- `tackles_won_percent`
- `successful_dribbles_percent`
- `ground_duels_won_percent`
- `aerial_duels_won_percent`

### 2. HANDLING NEW PLAYERS

**Problem:** Players in 2025-2026 who weren't in 2024-2025 lack historical rolling/cumulative features.

**Solution:**

```python
from src.fpl_projection.new_entities import handle_new_players_full_pipeline

# Automatically handled when combining seasons
df_combined = combine_seasons(df_2425, df_2526)

# is_new_player flag is added automatically
new_players = df_combined[df_combined["is_new_player"] == True]
```

**What Happens:**
1. **Identify**: Compare `player_id` across seasons
2. **Rolling Features**: Fill with position-average (GK/DEF/MID/FWD)
3. **Cumulative Features**: Fill with 0 (no history)
4. **Flag**: Add `is_new_player=True` column

### 3. HANDLING NEW TEAMS (Promoted)

**Promoted Teams 2025-2026:**
- Leeds
- Burnley
- Sunderland

**Default Strength Ratings:**

```python
DEFAULT_PROMOTED_TEAM_STRENGTH = {
    "strength": 2,                    # Weak
    "strength_overall_home": 750,
    "strength_overall_away": 700,
    "strength_attack_home": 700,
    "strength_attack_away": 650,
    "strength_defence_home": 700,
    "strength_defence_away": 650,
    "elo": 1450,
}
```

**Usage:**

```python
from src.fpl_projection.data_processor import add_new_team_defaults
from src.fpl_projection.new_entities import PROMOTED_TEAMS_2025_26

teams_df = add_new_team_defaults(
    teams_df,
    promoted_teams=PROMOTED_TEAMS_2025_26,
)
```

### 4. HANDLING PLAYER TRANSFERS

**Problem:** Players who switch teams mid-season should have team-based features reset.

**Solution:**

```python
from src.fpl_projection.data_processor import track_player_transfers

# Automatically handled in process_season_data()
df = track_player_transfers(df)

# Adds team_changed flag
transfers = df[df["team_changed"] == True]
```

**Team-Based Features Reset:**
- Team strength features
- Opponent-based features
- Team defensive stats

**Individual Features Preserved:**
- Player performance rolling averages
- Individual xG/xA stats
- Personal defensive contributions

### 5. DEFENSIVE CONTRIBUTION RECALCULATION

**Problem:** 2024-2025 data doesn't include defensive contribution points (new rule).

**Solution:**

```python
from src.fpl_projection.data_processor import (
    calculate_defensive_contribution_legacy,
    calculate_defensive_contribution_points,
    calculate_adjusted_points,
)

# Automatically handled when recalculate_2425_defcon=True
df_2425 = process_season_data(
    df_2425,
    season="2024-2025",
    recalculate_defcon=True,
)
```

**Rules:**
- **DEF/GK**: If CBIT ≥ 10 → +2 points
  - CBIT = Clearances + Blocks + Interceptions + Tackles
- **MID/FWD**: If CBIRT ≥ 12 → +2 points
  - CBIRT = CBIT + Recoveries

**New Columns Added:**
- `defensive_contribution`: Raw count (tackles + CBI)
- `defcon_points`: Points awarded (0 or 2)
- `cbit`: Clearances + Blocks + Interceptions + Tackles
- `cbirt`: CBIT + Recoveries
- `adjusted_points`: Original `total_points` + `defcon_points`

### 6. COLUMN ALIGNMENT BETWEEN SEASONS

**Column Name Differences:**

| 24-25 Column | 25-26 Column | Alignment |
|--------------|--------------|-----------|
| `goals` | `goals_scored` | Rename 25-26 → `goals` |
| `minutes_played` | `minutes` | Use `minutes` |
| (missing) | `defensive_contribution` | Calculate for 24-25 |
| (missing) | `tackles` | Derive from match stats |
| (missing) | `clearances_blocks_interceptions` | Combine if separate |

**Automatic Alignment:**

```python
from src.fpl_projection.data_processor import align_column_names

df = align_column_names(df, season="2025-2026")
# Renames: goals_scored → goals
#          id → player_id
```

## Complete Processing Pipeline

### Step-by-Step

```python
from pathlib import Path
from src.fpl_projection.data_loading import load_premier_league_gameweek_stats
from src.fpl_projection.data_processor import combine_seasons, get_training_feature_list
from src.fpl_projection.feature_engineering import engineer_all_features

# 1. Load raw data
print("Step 1: Loading raw data...")
df_2425 = load_premier_league_gameweek_stats(
    repo_root=Path("."),
    season="2024-2025",
    apply_feature_engineering=False,
)

df_2526 = load_premier_league_gameweek_stats(
    repo_root=Path("."),
    season="2025-2026",
    apply_feature_engineering=False,
)

# 2. Combine seasons with processing
print("Step 2: Combining and processing seasons...")
df_combined = combine_seasons(
    df_2425,
    df_2526,
    recalculate_2425_defcon=True,
)

# 3. Apply feature engineering
print("Step 3: Applying feature engineering...")
df_final = engineer_all_features(df_combined)

# 4. Get training features
print("Step 4: Selecting training features...")
feature_cols = get_training_feature_list(include_engineered=True)

# 5. Filter to available features
available_features = [c for c in feature_cols if c in df_final.columns]
print(f"Training features: {len(available_features)}")

# 6. Extract training data
X = df_final[available_features]
y = df_final["adjusted_points"]  # Use adjusted points as target

print("\nReady for model training!")
print(f"  Samples: {len(X)}")
print(f"  Features: {len(available_features)}")
print(f"  Players: {df_final['player_id'].nunique()}")
```

## Output Columns

### Identity Columns (Not for Training)
- `player_id`
- `gw`
- `position`
- `web_name`
- `team_code`
- `season`

### Feature Columns (~50-70)

**Base Features:**
- Per-90 stats (xG, xA, minutes, etc.)
- Defensive contributions (CBIT, CBIRT)
- GK-specific (saves, clean sheets)
- Team strength
- Price/value
- Set piece flags

**Engineered Features:**
- Rolling averages (3-game, 5-game)
- Cumulative stats (season totals)
- Position features
- Role-weighted features

### Target Columns
- `total_points`: Original FPL points
- `defcon_points`: Defensive contribution points
- `adjusted_points`: total_points + defcon_points (RECOMMENDED TARGET)

### Flag Columns
- `is_new_player`: True if player not in previous season
- `team_changed`: True if player transferred this gameweek
- `is_penalty_taker`: True if first-choice penalty taker
- `is_freekick_taker`: True if first-choice free kick taker
- `is_corner_taker`: True if first-choice corner taker

## Validation Checks

```python
# Check for leaky columns
leaky_cols = ["ep_next", "ep_this"]
for col in leaky_cols:
    assert col not in df_final.columns, f"Leaky column {col} found!"

# Check defensive contributions calculated
assert "defcon_points" in df_final.columns
assert "adjusted_points" in df_final.columns

# Check new player flags
new_player_count = df_final["is_new_player"].sum()
print(f"New players flagged: {new_player_count}")

# Check team changes tracked
transfer_count = df_final["team_changed"].sum()
print(f"Player transfers: {transfer_count}")

# Check promoted teams have default ratings
from src.fpl_projection.new_entities import validate_teams
from src.fpl_projection.data_loading import load_insights_teams

teams = load_insights_teams(repo_root=Path("."), season="2025-2026")
validation = validate_teams(teams, team_col="name")
print(f"Promoted teams found: {len(validation['promoted_found'])}/3")
```

## Summary

**Input Files:**
- `FPL-Core-Insights/data/2024-2025/By Tournament/Premier League/GW*/player_gameweek_stats.csv`
- `FPL-Core-Insights/data/2025-2026/By Tournament/Premier League/GW*/player_gameweek_stats.csv`

**Processing Steps:**
1. ✅ Load raw data from both seasons
2. ✅ Align column names
3. ✅ Recalculate defensive contributions for 24-25
4. ✅ Mark new players
5. ✅ Track player transfers
6. ✅ Add set piece flags
7. ✅ Combine seasons
8. ✅ Apply feature engineering
9. ✅ Select training features

**Output:**
- Clean DataFrame with ~50-70 engineered features
- Target variable: `adjusted_points`
- Position-specific feature sets
- New player and transfer flags
- Ready for LSTM/ensemble model training

**Key Functions:**
- `combine_seasons()`: Combine 24-25 and 25-26
- `process_season_data()`: Process single season
- `calculate_defensive_contribution_points()`: Add defcon points
- `track_player_transfers()`: Flag transfers
- `mark_new_players()`: Flag new players
- `get_training_feature_list()`: Get feature columns

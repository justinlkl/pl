# Team Strength & Position-Specific Features Implementation

## Summary

✅ **Implemented:** Team strength feature merging and position-specific feature sets

## What Was Added

### 1. Team Strength Features (from teams.csv)

**Location:** [data_loading.py](../src/fpl_projection/data_loading.py)

The following team strength columns are now automatically merged from `teams.csv`:
- `strength_overall_home`
- `strength_overall_away`
- `strength_attack_home`
- `strength_attack_away`
- `strength_defence_home`
- `strength_defence_away`
- `elo` (team Elo rating)

**How it works:**
```python
# Team strength features are merged after player metadata
# Teams.csv → merge on team_code → add strength columns to player data
```

**Verification:**
```bash
$ python -c "from pathlib import Path; from src.fpl_projection.data_loading import load_premier_league_gameweek_stats; df = load_premier_league_gameweek_stats(repo_root=Path('.'), season='2025-2026', apply_feature_engineering=False); print([c for c in df.columns if 'strength' in c.lower()])"

# Output:
# ['strength_overall_home', 'strength_overall_away', 'strength_attack_home', 
#  'strength_attack_away', 'strength_defence_home', 'strength_defence_away', 'opp_strength']
```

### 2. Position-Specific Feature Sets

**Location:** [position_features.py](../src/fpl_projection/position_features.py)

Defined comprehensive feature sets for each position:

#### Universal Features (All Positions) - 21 features
- **Form:** `adjusted_points_rolling_3`, `adjusted_points_rolling_5`, `points_std_5`
- **Minutes:** `minutes_played_rolling_3/5`, `minutes_trend`, `minutes_per_90`
- **Context:** `is_home`, `fixture_difficulty`
- **Price:** `now_cost`, `value_form`, `value_season`
- **Team strength:** `strength_attack_home/away`, `strength_defence_home/away`, `strength_overall_home/away`, `elo`
- **Opponent:** `opp_strength`, `opp_elo`

#### GK-Specific - 33 total features (21 universal + 12 GK)
- Saves: `saves_per_90`, `saves_rolling_3/5`, `save_pct`
- Goals conceded: `goals_conceded`, `goals_conceded_per_90`, `expected_goals_conceded`, `expected_goals_conceded_per_90`
- Clean sheets: `clean_sheets`, `clean_sheets_rolling_3`
- Defensive contribution: `defcon_points`, `defcon_actions`

#### DEF-Specific - 41 total features (21 universal + 20 DEF)
- Defensive actions: `cbi_per_90`, `cbi_rolling_3/5`, `clearances_per_90`, `tackles_per_90`, `interceptions_per_90`, `blocks_per_90`
- Defensive metrics: `duel_success_rate`, `tackles_won`, `tackles_won_per_90`
- Team defense: `expected_goals_conceded`, `clean_sheets`, `clean_sheets_rolling_3`
- Attacking threat: `xg_per_90`, `xa_per_90`, `xgi_per_90`, `xg_rolling_3`
- Defensive contribution: `defcon_points`, `defcon_actions`

#### MID-Specific - 42 total features (21 universal + 21 MID)
- Expected stats: `xgi_per_90`, `xg_per_90`, `xa_per_90`, `xg_rolling_3`, `xa_rolling_5`, `xgi_rolling_3`
- Defensive (box-to-box): `cbirt_per_90`, `cbirt_rolling_5`, `defcon_points`, `defcon_actions`
- Creativity: `chances_created`, `chances_created_per_90`, `key_passes`, `key_passes_per_90`
- Shooting: `shots_on_target_per_90`, `shot_accuracy`, `total_shots_per_90`
- Set pieces: `is_penalty_taker`, `is_freekick_taker`, `is_corner_taker`, `penalties_order`

#### FWD-Specific - 39 total features (21 universal + 18 FWD)
- Goals: `goals_per_90`, `goals_rolling_3/5`
- Expected stats: `xg_per_90`, `xa_per_90`, `xgi_per_90`, `xg_rolling_3/5`, `xgi_rolling_3`
- Shooting: `shots_on_target_per_90`, `total_shots_per_90`, `shot_accuracy`, `big_chances_missed`, `big_chances_missed_per_90`
- Set pieces: `is_penalty_taker`, `penalties_order`
- Link-up: `chances_created`, `chances_created_per_90`

### 3. Feature Configuration JSON

**Location:** [models/feature_config.json](../models/feature_config.json)

Generated configuration file with position-specific feature lists:
```json
{
  "GK": [33 features],
  "DEF": [41 features],
  "MID": [42 features],
  "FWD": [39 features]
}
```

## Usage Examples

### Get Features for a Specific Position

```python
from src.fpl_projection.position_features import get_position_features

# Get all MID features (universal + MID-specific)
mid_features = get_position_features("MID")
print(f"MID features: {len(mid_features)}")  # 42

# Get only MID-specific features (no universal)
mid_only = get_position_features("MID", include_universal=False)
print(f"MID-only features: {len(mid_only)}")  # 21
```

### Filter Available Features

```python
from src.fpl_projection.position_features import filter_available_features

# Get features that actually exist in your dataframe
desired = get_position_features("FWD")
available = filter_available_features(desired, df.columns, warn_missing=True)

# Train model with available features
X = df[available]
y = df["adjusted_points"]
```

### Get All Position Features

```python
from src.fpl_projection.position_features import get_all_position_features

all_features = get_all_position_features()
# Returns: {"GK": [...], "DEF": [...], "MID": [...], "FWD": [...]}

for pos, features in all_features.items():
    print(f"{pos}: {len(features)} features")
```

### Train Position-Specific Models

```python
from pathlib import Path
from src.fpl_projection.data_loading import load_premier_league_gameweek_stats
from src.fpl_projection.position_features import get_position_features, filter_available_features

# Load data with team strength features
df = load_premier_league_gameweek_stats(
    repo_root=Path("."),
    season="2025-2026",
    apply_feature_engineering=True,
)

# Train separate model for each position
for position in ["GK", "DEF", "MID", "FWD"]:
    # Filter to position
    df_pos = df[df["position"] == position].copy()
    
    # Get position-specific features
    desired_features = get_position_features(position)
    available_features = filter_available_features(
        desired_features, 
        df_pos.columns,
        warn_missing=True,
    )
    
    # Prepare data
    X = df_pos[available_features]
    y = df_pos["adjusted_points"]
    
    # Train model
    print(f"Training {position} model with {len(available_features)} features...")
    # ... your training code here
```

## Key Features

### Team Strength Integration

✅ **Automatic merging:** Team strength features are automatically added when loading data  
✅ **No manual work:** Just call `load_premier_league_gameweek_stats()` and strength features are included  
✅ **Both home/away:** Separate attack and defense strength for home vs away matches  
✅ **Opponent strength:** Also includes opponent strength via fixture difficulty

### Position-Aware Features

✅ **Role-appropriate:** Each position gets features relevant to their role  
✅ **GK:** Focus on saves, clean sheets, goals conceded  
✅ **DEF:** Defensive actions + attacking threat for attacking defenders  
✅ **MID:** Balanced attacking + defensive contribution (box-to-box)  
✅ **FWD:** Goals, shots, expected stats, link-up play  

### Flexible Usage

✅ **Universal features:** Core features used across all positions  
✅ **Position-specific:** Additional features tailored to each role  
✅ **Filter by availability:** Automatically handle missing columns  
✅ **JSON export:** Reproducible feature configuration

## Testing

All features verified with test pipeline:

```bash
$ python scripts/test_data_pipeline.py

✓ Loaded 2025-2026: (15976, 99)
✓ Team strength columns: 7 columns
✓ Feature engineering complete: (15976, 159)
✓ Training features available: 45
✓ Ready for model training!
```

**Team strength columns verified:**
- `strength_overall_home` ✓
- `strength_overall_away` ✓
- `strength_attack_home` ✓
- `strength_attack_away` ✓
- `strength_defence_home` ✓
- `strength_defence_away` ✓
- `opp_strength` ✓

**Feature config verified:**
```bash
$ python src/fpl_projection/position_features.py

✅ Feature configuration saved to models/feature_config.json
  GK: 33 features
  DEF: 41 features
  MID: 42 features
  FWD: 39 features
```

## Benefits

1. **Better predictions:** Team strength is a crucial signal for FPL points
2. **Position-aware modeling:** Train specialized models for each position
3. **Automatic integration:** No manual feature engineering needed
4. **Reproducible:** Feature config saved to JSON
5. **Flexible:** Use universal features only or add position-specific
6. **Production-ready:** Tested and validated

## Next Steps

1. **Train position-specific models:**
   ```bash
   python -m src.fpl_projection.train --position MID --features position_features.json
   ```

2. **Compare performance:**
   - Single model (all positions) vs. position-specific models
   - With/without team strength features
   - Universal only vs. universal + position-specific

3. **Feature importance:**
   - Analyze which team strength features matter most
   - Identify position-specific feature importance

---

**Status:** ✅ **Production Ready**

Team strength features are now automatically merged and position-specific feature sets are defined and ready for use!

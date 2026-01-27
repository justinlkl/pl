from pathlib import Path
import pandas as pd
from src.fpl_projection.data_loading import load_premier_league_gameweek_stats

repo_root = Path('.')
season = '2025-2026'
proj = pd.read_csv('outputs/projections_internal.csv')
print('Projections: rows=', len(proj), 'player_id dtype=', proj['player_id'].dtype if 'player_id' in proj.columns else None)
print('Projection sample ids:', proj['player_id'].head(10).tolist() if 'player_id' in proj.columns else None)

actuals = load_premier_league_gameweek_stats(repo_root=repo_root, season=season, apply_feature_engineering=False)
print('\nActuals: rows=', len(actuals), 'columns=', actuals.columns.tolist())
print('Actuals sample ids (gw=23):')
print(actuals[actuals['gw']==23].head(10))

# Quick merge check
left = proj.copy()
left['gw'] = 23
merged = left.merge(actuals, on=['player_id','gw'], how='inner')
print('\nMerge on player_id+gw: merged_rows=', len(merged))
if len(merged)==0:
    # try name+team
    join_cols = [c for c in ['web_name','club','team','team_code'] if c in proj.columns and c in actuals.columns]
    print('Try joining on:', join_cols)
    merged2 = left.merge(actuals, on=join_cols+['gw'], how='inner')
    print('Merge on name/team+gw: merged_rows=', len(merged2))

print('\nDone')

# v1.1 Fixture Difficulty

This folder captures the Phase-1 A/B results after adding a simple fixture difficulty feature.

## Change
- Adds `fixture_difficulty` + `fixture_is_home` to the default feature set.
- `fixture_difficulty` is derived per (gw, team_code) from `GW*/fixtures.csv` and `teams.csv`:
  - base = opponent `strength` (1–5)
  - away penalty = `+0.25` when `fixture_is_home == 0`
  - double gameweeks: difficulty is averaged across fixtures

## Data / setup
- Season: `2025-2026`
- Target: `xp_blend`
- Split: time-aware end-gw split (train/val/test in chronological order)
- MID split: enabled (`--mid-split`)
- Bias penalty: enabled (`--bias-penalty-alpha 0.3`)
- Longer training attempt: `--epochs 120 --early-stopping-patience 15 --lr-plateau-patience 7` (early stopping may stop before max epochs)

## Command used
Ensemble (sklearn backend; diagnostics written to `artifacts/ensemble/diagnostics/`):
- `python -m src.fpl_projection.ensemble_train --target xp_blend --mid-split --bias-penalty-alpha 0.3 --epochs 120 --early-stopping-patience 15 --lr-plateau-patience 7 --diagnostics --backend sklearn`

## Summary (held-out test)

| Model | MAE | Rank Corr | Top-50 Recall | Calib Err Rel |
|---|---:|---:|---:|---:|
| LSTM | 3.349 | 0.812 | 0.36 | 0.146 |
| Ensemble | 3.050 | 0.874 | 0.54 | 0.100 |

Notes:
- Metrics are computed at player-level by aggregating test sequences per player.
- `calibration_error_rel` is an ECE-style weighted absolute bin gap divided by mean actual total.

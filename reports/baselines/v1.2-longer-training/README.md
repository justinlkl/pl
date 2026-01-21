# v1.2 Longer Training

This folder captures the Phase-1 A/B results after training longer (higher max epochs + patience) on top of the fixture difficulty feature set.

## Data / setup
- Season: `2025-2026`
- Target: `xp_blend`
- Split: time-aware end-gw split (train/val/test in chronological order)
- MID split: enabled (`--mid-split`)
- Bias penalty: enabled (`--bias-penalty-alpha 0.3`)

## Command used
- `python -m src.fpl_projection.ensemble_train --target xp_blend --mid-split --bias-penalty-alpha 0.3 --epochs 250 --early-stopping-patience 25 --lr-plateau-patience 10 --diagnostics --backend sklearn`

## Summary (held-out test)

| Model | MAE | Rank Corr | Top-50 Recall | Calib Err Rel |
|---|---:|---:|---:|---:|
| LSTM | 3.349 | 0.812 | 0.36 | 0.146 |
| Ensemble | 3.050 | 0.874 | 0.54 | 0.100 |

Notes:
- Early stopping halted training before max epochs; results match v1.1 under these settings.

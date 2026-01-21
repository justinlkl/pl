# v1.0 Ensemble Baseline

This folder captures the canonical Phase-1 A/B results used to lock the first ensemble baseline.

## Data / setup
- Season: `2025-2026`
- Target: `xp_blend`
- Split: time-aware end-gw split (train/val/test in chronological order)
- MID split: enabled (`--mid-split`)
- Bias penalty: enabled (`--bias-penalty-alpha 0.3`)

## Commands used
LSTM-only (diagnostics written to `artifacts/diagnostics/`):
- `python -m src.fpl_projection.train --target xp_blend --mid-split --bias-penalty-alpha 0.3 --epochs 50 --diagnostics --permute-max-features 0`

Ensemble (sklearn backend; diagnostics written to `artifacts/ensemble/diagnostics/`):
- `python -m src.fpl_projection.ensemble_train --target xp_blend --mid-split --bias-penalty-alpha 0.3 --epochs 50 --diagnostics --backend sklearn`

## Summary (held-out test)

| Model | MAE | Rank Corr | Top-50 Recall | Calib Err Rel |
|---|---:|---:|---:|---:|
| LSTM | 3.370 | 0.809 | 0.40 | 0.160 |
| Ensemble | 3.049 | 0.875 | 0.58 | 0.103 |

Notes:
- Metrics are computed at player-level by aggregating test sequences per player.
- `calibration_error_rel` is an ECE-style weighted absolute bin gap divided by mean actual total.

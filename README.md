# FPL Projection Model (LSTM)

This project trains an LSTM-based time-series model to predict Fantasy Premier League (FPL) player **projection points** for the next gameweeks.

It uses the dataset from `FPL-Core-Insights/` (cloned from https://github.com/olbauday/FPL-Core-Insights).

Note: `FPL-Core-Insights/`, `artifacts/`, and `outputs/` are treated as local data/build outputs and are git-ignored.

## Pipeline

- **Data processing**: loads Premier League `player_gameweek_stats.csv` per gameweek
- **Feature engineering**: uses a curated set of performance + enhanced defensive/context features (tackles, recoveries, defensive_contribution, xG/xA, ICT, etc.)
- **Missing data**: median imputation per feature
- **Normalization**: standard scaling per feature
- **Sequences**: 5-gameweek windows per player
- **Model**: 2-layer LSTM → Dense → multi-horizon regression head
- **Targets**:
	- `xp_proxy`: smoother expected-points proxy from xG/xA + simple CS approximation
	- `xp_blend`: `0.7*xp_proxy + 0.3*total_points` (recommended default)
- **Anti-DM bias guardrails**:
	- Role-aware feature weighting (downweights MID_DM defensive stability signals)
	- Role-weighted loss during training (attacker errors cost more)
	- Optional xGI overprediction penalty (discourages projecting low-xGI players too high)
	- Post-model role scaling at inference time (calibration vs. typical positional distributions)

## Quickstart

1) Install deps

```powershell
C:/Users/justinlam/Desktop/pl/.venv/Scripts/python.exe -m pip install -r requirements.txt
C:/Users/justinlam/Desktop/pl/.venv/Scripts/python.exe -m pip install -e .
```

Optional (ensemble models):

```powershell
# Recommended in a Python 3.11/3.12 environment
C:/Users/justinlam/Desktop/pl/.venv/Scripts/python.exe -m pip install -r requirements-ensemble.txt
```

2) Train

```powershell
C:/Users/justinlam/Desktop/pl/.venv/Scripts/python.exe -m src.fpl_projection.train --season 2025-2026 --seq-length 5 --horizon 6 --target xp_blend --mid-split --bias-penalty-alpha 0.3
```

Useful flags:
- `--target xp_blend` (default recommendation) or `--target xp_proxy`
- `--bias-penalty-alpha 0.3` to add `alpha * ReLU(pred - xGI_cap)` penalty (0 disables)
- `--no-role-loss-weighting` to disable per-role sample weights
- `--monitor val_macro_mse` (default) to early-stop on per-role macro validation MSE

3) Predict (writes a projection table)

```powershell
C:/Users/justinlam/Desktop/pl/.venv/Scripts/python.exe -m src.fpl_projection.predict --season 2025-2026 --horizon 6 --mid-split
```

By default, predictions apply post-model role scaling for calibration. Disable with `--no-role-scaling`.

Outputs:
- `artifacts/model.keras`
- `artifacts/preprocess.joblib`
- `outputs/projections.csv`

## Static UI (site)

This repo now also includes a no-build-step static UI (React via CDN + Bootstrap), similar to your previous `fpl/site` setup.

1) Generate site data from the latest projection output:

```powershell
C:/Users/justinlam/Desktop/pl/.venv/Scripts/python.exe scripts/generate_site.py
```

2) Serve the site locally:

```powershell
Set-Location c:\Users\justinlam\Desktop\pl\site
C:/Users/justinlam/Desktop/pl/.venv/Scripts/python.exe -m http.server 8000 --bind 127.0.0.1
```

Open: http://127.0.0.1:8000

## Optional: team storage API (site_api)

If you want to persist saved teams (instead of keeping them in browser memory), run the FastAPI backend:

```powershell
C:/Users/justinlam/Desktop/pl/.venv/Scripts/python.exe -m pip install -r site_api/requirements.txt
C:/Users/justinlam/Desktop/pl/.venv/Scripts/python.exe -m uvicorn site_api.app:app --reload --port 8080
```

Endpoints:
- `GET /team/{user_id}`
- `POST /team` with JSON `{ "user_id": "...", "team": [...] }`

## Notes

- The model predicts points purely from historical player/gameweek stats (plus context columns present in the dataset). It does not simulate future fixtures unless those fixture-context columns are present in the data.

## Data setup

If `FPL-Core-Insights/` is missing, clone it into the repo root:

```powershell
Set-Location c:\Users\justinlam\Desktop\pl
git clone https://github.com/olbauday/FPL-Core-Insights.git
```

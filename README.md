# FPL Projection Model (LSTM)

This project trains an LSTM-based time-series model to predict Fantasy Premier League (FPL) player **projection points** for the next gameweeks.

It uses the dataset from the included subfolder `FPL-Core-Insights/` (cloned from https://github.com/olbauday/FPL-Core-Insights).

## Pipeline

- **Data processing**: loads Premier League `player_gameweek_stats.csv` per gameweek
- **Feature engineering**: uses a curated set of performance + enhanced defensive/context features (tackles, recoveries, defensive_contribution, xG/xA, ICT, etc.)
- **Missing data**: median imputation per feature
- **Normalization**: standard scaling per feature
- **Sequences**: 5-gameweek windows per player
- **Model**: LSTM(64, tanh) → Dropout(0.2) → Dense(32, relu) → Dense(horizon)

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
C:/Users/justinlam/Desktop/pl/.venv/Scripts/python.exe -m fpl_projection.train --season 2025-2026 --seq-length 5 --horizon 6
```

3) Predict (writes a projection table)

```powershell
C:/Users/justinlam/Desktop/pl/.venv/Scripts/python.exe -m fpl_projection.predict --season 2025-2026 --horizon 6
```

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

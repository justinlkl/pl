# FPL Projection Model (LSTM)

This project trains an LSTM-based time-series model to predict Fantasy Premier League (FPL) player **projection points** for the next gameweeks.

It uses the dataset from `FPL-Core-Insights/` (cloned from https://github.com/olbauday/FPL-Core-Insights).

Note: `FPL-Core-Insights/`, `artifacts/`, and `outputs/` are treated as local data/build outputs and are git-ignored.

## Pipeline

- **Data ingest**: load per-GW `player_gameweek_stats.csv` from `FPL-Core-Insights/data/<season>/By Tournament/Premier League/GW*/` via [src/fpl_projection/data_loading.py](src/fpl_projection/data_loading.py).
- **Feature engineering**: defensive/context features, rolling/cumulative forms, position priors (new-player handling) in [src/fpl_projection/feature_engineering.py](src/fpl_projection/feature_engineering.py).
- **Prep**: median imputation + standard scaling on all timesteps per role-aware feature weighting.
- **Sequences**: 5-GW windows per player (configurable) built in [src/fpl_projection/sequences.py](src/fpl_projection/sequences.py).
- **Model**: 2-layer LSTM → Dense → multi-horizon regression head defined in [src/fpl_projection/modeling.py](src/fpl_projection/modeling.py).
- **Targets**:
	- `xp_proxy`: xG/xA-based proxy with simple CS allowance.
	- `xp_blend`: `0.7*xp_proxy + 0.3*total_points` (recommended default).
- **Bias guardrails**: role-aware sample weights, optional xGI penalty, post-model role calibration in [src/fpl_projection/role_modeling.py](src/fpl_projection/role_modeling.py).

## Full run (example: post-GW23, season 2025-2026)

1) Update data

```powershell
Set-Location c:\Users\justinlam\Desktop\pl\FPL-Core-Insights
git pull
```

2) Train LSTM on latest data (GW1-23)

```powershell
Set-Location c:\Users\justinlam\Desktop\pl
C:/Users/justinlam/Desktop/pl/.venv/Scripts/python.exe -m src.fpl_projection.train --season 2025-2026 --seq-length 5 --horizon 6 --target xp_blend --mid-split --bias-penalty-alpha 0.3 --epochs 30
```

Artifacts land in `artifacts/model.keras`, `artifacts/preprocess.joblib`, `artifacts/meta.json`, `artifacts/role_scaling.json`.

3) Predict horizons (next 6 GWs)

```powershell
C:/Users/justinlam/Desktop/pl/.venv/Scripts/python.exe -m src.fpl_projection.predict --season 2025-2026 --horizon 6 --mid-split
```

Outputs: `outputs/projections.csv` (public), `outputs/projections_internal.csv` (with `player_id`).

4) Optional: evaluate vs actuals for a GW

```powershell
C:/Users/justinlam/Desktop/pl/.venv/Scripts/python.exe -m scripts.evaluate_model --gw 23 --projections outputs/projections_internal.csv --season 2025-2026
```

Writes `reports/evaluation_history.csv` and `reports/evaluation_summary.csv`.

5) Stage static site data

```powershell
C:/Users/justinlam/Desktop/pl/.venv/Scripts/python.exe scripts/generate_site.py
```

Produces `site/data/projections.json` (plus fixtures/stats/teams). Serve locally:

```powershell
Set-Location c:\Users\justinlam\Desktop\pl\site
C:/Users/justinlam/Desktop/pl/.venv/Scripts/python.exe -m http.server 8000 --bind 127.0.0.1
```

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

## Latest status (post-GW23 run)

- Data: pulled `FPL-Core-Insights` on 2026-01-27 (covers GW1-23 for 2025-2026).
- Model: retrained with `xp_blend`, seq_length=5, horizon=6, mid-split, bias penalty alpha=0.3, 30 epochs.
- Projections: refreshed in `outputs/projections*.csv` and `site/data/projections.json`.
- Eval: GW23 vs actuals → MAE 0.988, RMSE 1.921 (see `reports/evaluation_summary.csv`).

## Legacy / optional pieces

- [main_fpl_model.R](main_fpl_model.R): legacy R draft, not used in the Python pipeline.
- [streamlit_app.py](streamlit_app.py): deprecated in favor of the static site under `site/`.
- [Dockerfile](Dockerfile) / [docker-compose.yml](docker-compose.yml): previously used for Streamlit; not maintained for the current static site flow.
- [scripts/blend_weights.py](scripts/blend_weights.py): ad-hoc ensemble explorer; not invoked by the standard train/predict pipeline.
- [scripts/fpl_gameweek_gate.py](scripts/fpl_gameweek_gate.py): CI gate helper for GitHub Actions; optional for local runs.

## Data setup

If `FPL-Core-Insights/` is missing, clone it into the repo root:

```powershell
Set-Location c:\Users\justinlam\Desktop\pl
git clone https://github.com/olbauday/FPL-Core-Insights.git
```

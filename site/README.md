# Site

This folder contains a static front-end (React via CDN + Bootstrap) to explore projections and build a draft squad.

## Quick start

1) Stage site data from your pipeline outputs:

```powershell
C:/Users/justinlam/Desktop/pl/.venv/Scripts/python.exe scripts/generate_site.py
```

2) Serve the site:

```powershell
Set-Location c:\Users\justinlam\Desktop\pl\site
C:/Users/justinlam/Desktop/pl/.venv/Scripts/python.exe -m http.server 8000 --bind 127.0.0.1
```

Open: http://127.0.0.1:8000

## Data contract

The UI loads:
- `site/data/projections.json` (required)
- `site/data/fixtures.json` (optional)
- `site/data/stats.csv` (optional)
- `site/data/teams.json` (optional)
- `site/data/my_team.json` (optional, local fallback for "My Team")

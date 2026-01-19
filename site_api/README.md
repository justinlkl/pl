# site_api

Optional FastAPI backend to persist saved teams.

## Run

```powershell
C:/Users/justinlam/Desktop/pl/.venv/Scripts/python.exe -m pip install -r site_api/requirements.txt
C:/Users/justinlam/Desktop/pl/.venv/Scripts/python.exe -m uvicorn site_api.app:app --reload --port 8080
```

Endpoints:
- `GET /team/{user_id}`
- `POST /team` with JSON `{ "user_id": "abc", "team": [...] }`

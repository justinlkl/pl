from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
STORE = ROOT / "site_data"
STORE.mkdir(parents=True, exist_ok=True)

TEAMS_FILE = STORE / "teams.json"

app = FastAPI(title="FPL Team Storage")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TeamPayload(BaseModel):
    user_id: str
    team: list


def _load() -> dict:
    if not TEAMS_FILE.exists():
        return {}
    try:
        return json.loads(TEAMS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(data: dict) -> None:
    TEAMS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


@app.get("/team/{user_id}")
def get_team(user_id: str):
    return _load().get(user_id, [])


@app.post("/team")
def post_team(payload: TeamPayload):
    data = _load()
    data[payload.user_id] = payload.team
    _save(data)
    return {"status": "ok"}

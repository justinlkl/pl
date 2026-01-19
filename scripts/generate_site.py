#!/usr/bin/env python3
"""Generate static-site data files from projection outputs.

Usage:
    python scripts/generate_site.py

Writes:
    - site/data/projections.json (from outputs/projections_internal.csv preferred, else outputs/projections.csv, fallback data/projections.csv)
    - site/data/fixtures.json (copied from data/fixtures.json if present)
    - site/data/stats.csv (copied from data/stats.csv or data/opta_stats*.csv if present)
    - site/data/teams.json (best-effort from public FPL bootstrap-static)
    - site/data/my_team.json (local fallback so the UI doesn't 404)

Notes:
    This is intentionally lightweight (no build step, no extra deps).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
SITE_DATA = SITE / "data"

OUTPUTS_PROJ = ROOT / "outputs" / "projections.csv"
OUTPUTS_PROJ_INTERNAL = ROOT / "outputs" / "projections_internal.csv"
DATA_PROJ = ROOT / "data" / "projections.csv"

OUT_PROJ = SITE_DATA / "projections.json"
LEGACY_OUT_PROJ = SITE / "projections.json"

FIXTURES_SRC = ROOT / "data" / "fixtures.json"
FIXTURES_OUT = SITE_DATA / "fixtures.json"

STATS_SRC = ROOT / "data" / "stats.csv"
OPTA_STATS_SRC = ROOT / "data" / "opta_stats.csv"
OPTA_EXAMPLE_SRC = ROOT / "data" / "opta_stats.example.csv"
STATS_OUT = SITE_DATA / "stats.csv"

TEAMS_OUT = SITE_DATA / "teams.json"
MY_TEAM_OUT = SITE_DATA / "my_team.json"


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _to_int(v: Any) -> int | None:
    f = _to_float(v)
    if f is None:
        return None
    try:
        return int(f)
    except Exception:
        return None


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append({k: v for k, v in r.items()})
    return rows


def _normalize_projection_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize common fields to what the static UI expects."""
    out: list[dict[str, Any]] = []

    for r in rows:
        rr = dict(r)

        # IDs
        for k in ("player_id", "element", "id"):
            if rr.get(k) not in (None, ""):
                rr["player_id"] = _to_int(rr.get(k))
                break

        # Names
        for k in ("web_name", "name", "player"):
            if rr.get(k) not in (None, ""):
                rr["web_name"] = str(rr.get(k) or "").strip()
                break

        # Team
        if rr.get("team") in (None, "") and rr.get("team_name") not in (None, ""):
            rr["team"] = rr.get("team_name")

        # Position
        if rr.get("position") in (None, "") and rr.get("pos") not in (None, ""):
            rr["position"] = rr.get("pos")
        pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        try:
            p = _to_int(rr.get("position"))
            if p in pos_map:
                rr["position"] = pos_map[p]
        except Exception:
            pass

        # Price (handle FPL now_cost in tenths)
        price = None
        for k in ("price", "now_cost", "value", "cost"):
            if rr.get(k) not in (None, ""):
                price = _to_float(rr.get(k))
                break
        if price is None:
            price = 0.0
        if price > 1000:
            price = price / 10.0
        rr["price"] = float(price)

        # Projected points
        proj = None
        for k in ("proj_points", "projected_points", "xPts", "xpts"):
            if rr.get(k) not in (None, ""):
                proj = _to_float(rr.get(k))
                break
        rr["proj_points"] = float(proj or 0.0)

        # Optional quantiles
        for q in ("p10", "p50", "p90"):
            if rr.get(q) not in (None, ""):
                rr[q] = float(_to_float(rr.get(q)) or 0.0)

        # Optional minutes probability
        for k in ("minutes_prob", "minutes_probability", "min_prob"):
            if rr.get(k) not in (None, ""):
                rr["minutes_prob"] = float(_to_float(rr.get(k)) or 0.0)
                break

        # Optional fixture difficulty
        for k in ("fixture_difficulty", "fdr", "difficulty"):
            if rr.get(k) not in (None, ""):
                rr["fixture_difficulty"] = int(_to_int(rr.get(k)) or 3)
                break

        # Optional team id
        for k in ("team_id", "teamid"):
            if rr.get(k) not in (None, ""):
                rr["team_id"] = _to_int(rr.get(k))
                break

        out.append(rr)

    return out


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _stage_teams_json() -> None:
    teams_payload = None
    try:
        r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=20)
        r.raise_for_status()
        teams_payload = r.json()
    except Exception:
        teams_payload = None

    teams: list[dict[str, Any]] = []
    if isinstance(teams_payload, dict) and isinstance(teams_payload.get("teams"), list):
        for t in teams_payload["teams"]:
            try:
                tid = int(t.get("id"))
            except Exception:
                continue
            teams.append(
                {
                    "id": tid,
                    "short_name": str(t.get("short_name") or tid),
                    "team_name": str(t.get("name") or t.get("short_name") or tid),
                }
            )

    if teams:
        _write_json(TEAMS_OUT, {"teams": teams})


def main() -> None:
    SITE_DATA.mkdir(parents=True, exist_ok=True)

    # Projections
    if OUTPUTS_PROJ_INTERNAL.exists():
        src = OUTPUTS_PROJ_INTERNAL
    else:
        src = OUTPUTS_PROJ if OUTPUTS_PROJ.exists() else DATA_PROJ
    if not src.exists():
        raise SystemExit(f"Missing projections CSV: {OUTPUTS_PROJ} (or {DATA_PROJ})")

    rows = _read_csv_rows(src)
    normalized = _normalize_projection_rows(rows)
    _write_json(OUT_PROJ, normalized)
    try:
        LEGACY_OUT_PROJ.write_text(OUT_PROJ.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    # Fixtures
    if FIXTURES_SRC.exists():
        try:
            FIXTURES_OUT.write_text(FIXTURES_SRC.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass

    # Stats
    stats_src = None
    if STATS_SRC.exists():
        stats_src = STATS_SRC
    elif OPTA_STATS_SRC.exists():
        stats_src = OPTA_STATS_SRC
    elif OPTA_EXAMPLE_SRC.exists():
        stats_src = OPTA_EXAMPLE_SRC
    if stats_src is not None:
        try:
            STATS_OUT.write_text(stats_src.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass

    # Teams lookup (for badges)
    try:
        _stage_teams_json()
    except Exception:
        pass

    # My Team fallback
    try:
        if not MY_TEAM_OUT.exists():
            _write_json(MY_TEAM_OUT, {"picks": []})
    except Exception:
        pass

    print(f"Staged site data -> {SITE_DATA.resolve()}")
    print(f"- projections: {OUT_PROJ} ({len(normalized)} rows)")
    if FIXTURES_OUT.exists():
        print(f"- fixtures: {FIXTURES_OUT}")
    if STATS_OUT.exists():
        print(f"- stats: {STATS_OUT}")
    if TEAMS_OUT.exists():
        print(f"- teams: {TEAMS_OUT}")
    if MY_TEAM_OUT.exists():
        print(f"- my team: {MY_TEAM_OUT}")


if __name__ == "__main__":
    main()

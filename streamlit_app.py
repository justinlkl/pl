"""Streamlit app for FPL Team Builder and Projections.

This replaces the previous app with a simpler, tabbed interface that:
- loads projections from `outputs/projections.csv` (or `data/projections.csv` fallback)
- loads fixtures from `data/fixtures.json` (if present)
- pulls picks from the public FPL entry API for a given `ENTRY_ID`

Features: My Team (pulls picks), Projections, Fixtures Ticker, Key Stats table,
Player Profile, Transfer Planner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import base64
import json
import re
from collections import Counter

import pandas as pd
import requests
import streamlit as st


def get_player_photo_url(player_id: int) -> str:
    """Official Premier League CDN player photo (best-effort)."""
    return f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{int(player_id)}.png"


def get_team_badge_url(team_code: int) -> str:
    """Official Premier League CDN team badge (best-effort)."""
    return f"https://resources.premierleague.com/premierleague/badges/t{int(team_code)}.png"


def get_shirt_url(team_code: int) -> str:
    """FPL kit image (best-effort)."""
    return f"https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_{int(team_code)}-110.png"


STAT_GLOSSARY: dict[str, str] = {
    "£M": "Price in £m",
    "App": "Appearances / starts proxy (starts)",
    "Mins": "Minutes played",
    "S": "Starts",
    "OT": "Shots on Target",
    "In": "Influence",
    "BC": "Big Chances",
    "xG": "Expected Goals",
    "G": "Goals Scored",
    "%xGI": "% share of team expected goal involvements (if available)",
    "%GI": "% share of team goal involvements (if available)",
    "xGI": "Expected Goal Involvements (xG + xA)",
    "GI": "Goal Involvements (Goals + Assists)",
    "KP": "Key Passes / Chances Created",
    "BCC": "Big Chances Created",
    "xA": "Expected Assists",
    "A": "Assists",
    "DC": "Defensive Contributions (actions counted by FPL)",
    "xPts": "Projected points (sum across projected GWs)",
    "P10": "10th percentile projected points (if available)",
    "P50": "Median projected points (if available)",
    "P90": "90th percentile projected points (if available)",
    "BPS": "Bonus Points System score",
    "B": "Bonus points",
    "Pts": "Actual FPL points (historical)",
    "Min%": "Chance of playing (minutes probability)",
}


STAT_FULL_NAMES: dict[str, str] = {
    "Name": "Name",
    "Club": "Club",
    "£M": "Price (£m)",
    "App": "Appearances",
    "Mins": "Minutes",
    "S": "Starts",
    "OT": "Shots on Target",
    "In": "Influence",
    "BC": "Big Chances",
    "xG": "Expected Goals",
    "G": "Goals",
    "%xGI": "% Team xGI",
    "%GI": "% Team GI",
    "xGI": "Expected Goal Involvements",
    "GI": "Goal Involvements",
    "KP": "Key Passes",
    "BCC": "Big Chances Created",
    "xA": "Expected Assists",
    "A": "Assists",
    "DC": "Defensive Contributions",
    "xPts": "Projected Points",
    "P10": "P10 Projected Points",
    "P50": "P50 Projected Points",
    "P90": "P90 Projected Points",
    "BPS": "BPS",
    "B": "Bonus",
    "Pts": "Points",
    "Min%": "Minutes %",
}


DATA = Path("data")
OUTPUTS = Path("outputs")

INSIGHTS_ROOT = Path("FPL-Core-Insights") / "data"
DEFAULT_INSIGHTS_SEASON = "2025-2026"

PROJ_JSON = DATA / "projections.json"
PROJ_CSV_FALLBACK = DATA / "projections.csv"
PROJ_CSV = OUTPUTS / "projections.csv"
PROJ_CSV_INTERNAL = OUTPUTS / "projections_internal.csv"

FIXTURES_JSON = DATA / "fixtures.json"

# Optional BYO stats hooks
STATS_CSV = DATA / "stats.csv"
OPTA_CSV = DATA / "opta_stats.csv"
OPTA_EXAMPLE_CSV = DATA / "opta_stats.example.csv"

# Config
ENTRY_ID = 1093603
GW = 20
BUDGET = 100.0
MAX_PER_CLUB = 3
POS_CAP = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

# Allowed formations for Starting XI (GK is always 1)
ALLOWED_FORMATIONS: dict[str, tuple[int, int, int, int]] = {
    "3-4-3": (1, 3, 4, 3),
    "3-5-2": (1, 3, 5, 2),
    "4-5-1": (1, 4, 5, 1),
    "4-4-2": (1, 4, 4, 2),
    "4-3-3": (1, 4, 3, 3),
    "5-3-2": (1, 5, 3, 2),
    "5-4-1": (1, 5, 4, 1),
}


@st.cache_data
def load_json(path_or_url: str | Path) -> pd.DataFrame:
    p = str(path_or_url)
    if p.startswith("http"):
        r = requests.get(p, timeout=20)
        r.raise_for_status()
        payload = r.json()
        if isinstance(payload, dict) and "fixtures" in payload and isinstance(payload["fixtures"], list):
            return pd.DataFrame(payload["fixtures"])
        if isinstance(payload, dict) and "teams" in payload and isinstance(payload["teams"], list):
            return pd.DataFrame(payload["teams"])
        return pd.DataFrame(payload)

    try:
        with open(p, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict) and "fixtures" in payload and isinstance(payload["fixtures"], list):
            return pd.DataFrame(payload["fixtures"])
        if isinstance(payload, dict) and "teams" in payload and isinstance(payload["teams"], list):
            return pd.DataFrame(payload["teams"])
        return pd.DataFrame(payload)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [
        str(c).strip().lower().replace(" ", "_").replace("%", "pct").replace("-", "_")
        for c in out.columns
    ]
    return out


@st.cache_data
def load_opta_stats() -> pd.DataFrame:
    """Load Opta-style advanced stats from data/opta_stats.csv.

    This is a BYO dataset hook (no external Opta API/scraping). Expected identifiers:
    - player_id (FPL element id) preferred, OR
    - (web_name + team) as a fallback.
    """
    if not OPTA_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(OPTA_CSV)
        df = _normalize_cols(df)

        # Harmonize common id/name columns
        if "element" in df.columns and "player_id" not in df.columns:
            df = df.rename(columns={"element": "player_id"})
        if "id" in df.columns and "player_id" not in df.columns:
            df = df.rename(columns={"id": "player_id"})
        if "name" in df.columns and "web_name" not in df.columns:
            df = df.rename(columns={"name": "web_name"})
        if "club" in df.columns and "team" not in df.columns:
            df = df.rename(columns={"club": "team"})

        if "player_id" in df.columns:
            df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
        if "web_name" in df.columns:
            df["web_name"] = df["web_name"].astype(str)
        if "team" in df.columns:
            df["team"] = df["team"].astype(str)
        return df
    except Exception:
        return pd.DataFrame()


def enrich_with_opta_stats(projections: pd.DataFrame) -> pd.DataFrame:
    """Merge Opta stats into projections if data/opta_stats.csv exists."""
    if projections is None or projections.empty:
        return projections
    opta = load_opta_stats()
    if opta.empty:
        return projections

    # Prefer exact join on FPL player_id
    if "player_id" in projections.columns and "player_id" in opta.columns and opta["player_id"].notna().any():
        return projections.merge(opta, on="player_id", how="left", suffixes=("", "_opta"))

    # Best-effort join on (web_name, team)
    if (
        "web_name" in projections.columns
        and "team" in projections.columns
        and "web_name" in opta.columns
        and "team" in opta.columns
    ):
        return projections.merge(opta, on=["web_name", "team"], how="left", suffixes=("", "_opta"))

    return projections


@st.cache_data
def get_my_team(entry_id: int, gw: int) -> pd.DataFrame:
    url = f"https://fantasy.premierleague.com/api/entry/{entry_id}/event/{gw}/picks/"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        picks = r.json().get("picks", [])
        return pd.DataFrame(picks)
    except Exception:
        # No local fallback message in UI (as requested). Still return empty on failure.
        return pd.DataFrame(columns=["element", "position", "is_captain"])


def badge_path(name: str) -> str:
    candidates = [
        Path("site") / "assets" / "badges" / f"{name}.svg",
        Path("assets") / "badges" / f"{name}.svg",
        Path("site") / "assets" / "badges" / f"{name}.png",
        Path("assets") / "badges" / f"{name}.png",
    ]
    for p in candidates:
        try:
            if p.exists():
                return str(p)
        except Exception:
            continue
    return str(candidates[0])


@st.cache_data
def load_team_lookup() -> dict[int, dict[str, str]]:
    """Load team lookup from FPL bootstrap-static, with data/teams.(csv|json) fallback.

    Returns: {team_id: {"short_name": str, "team_name": str}}
    """
    out: dict[int, dict[str, str]] = {}
    p_json = DATA / "teams.json"
    p_csv = DATA / "teams.csv"

    try:
        r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=20)
        r.raise_for_status()
        payload = r.json()
        teams = payload.get("teams", []) if isinstance(payload, dict) else []
        for t in teams:
            try:
                tid = int(t.get("id"))
                out[tid] = {
                    "short_name": str(t.get("short_name") or tid),
                    "team_name": str(t.get("name") or t.get("short_name") or tid),
                }
            except Exception:
                continue
    except Exception:
        pass

    try:
        if p_json.exists():
            payload = json.loads(p_json.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for k, v in payload.items():
                    try:
                        tid = int(k)
                        if tid in out:
                            continue
                        if isinstance(v, dict):
                            out[tid] = {
                                "short_name": str(v.get("short_name") or v.get("short") or v.get("name") or tid),
                                "team_name": str(v.get("team_name") or v.get("name") or tid),
                            }
                        else:
                            out[tid] = {"short_name": str(v), "team_name": str(v)}
                    except Exception:
                        continue
            elif isinstance(payload, list):
                for t in payload:
                    try:
                        tid = int(t.get("id"))
                        if tid in out:
                            continue
                        out[tid] = {
                            "short_name": str(t.get("short_name") or t.get("short") or t.get("name") or tid),
                            "team_name": str(t.get("team_name") or t.get("name") or t.get("short_name") or tid),
                        }
                    except Exception:
                        continue

        elif p_csv.exists():
            df = pd.read_csv(p_csv)
            for _, r in df.iterrows():
                try:
                    tid = int(r.get("id") if "id" in df.columns else r.get("team_id"))
                    if tid in out:
                        continue
                    short_name = r.get("short_name") if "short_name" in df.columns else None
                    team_name = r.get("team_name") if "team_name" in df.columns else None
                    out[tid] = {
                        "short_name": str(short_name or tid),
                        "team_name": str(team_name or short_name or tid),
                    }
                except Exception:
                    continue
    except Exception:
        pass

    return out


def team_short_name(team_id: Any) -> str:
    try:
        tid = int(team_id)
    except Exception:
        return str(team_id) if team_id is not None else ""
    t = load_team_lookup().get(tid)
    return (t or {}).get("short_name") or str(tid)


def team_full_name(team_id: Any) -> str:
    try:
        tid = int(team_id)
    except Exception:
        return str(team_id) if team_id is not None else ""
    t = load_team_lookup().get(tid)
    return (t or {}).get("team_name") or (t or {}).get("short_name") or str(tid)


def price_options_from_proj(
    df: pd.DataFrame, step: float = 0.5, default_min: float = 3.5, default_max: float = 15.0
):
    if df is None or df.empty or "price" not in df.columns:
        lo, hi = default_min, default_max
    else:
        lo = float(df["price"].min())
        hi = float(df["price"].max())
        lo = max(default_min, (int(lo * 2) / 2.0))
        hi = max(lo + step, (int(hi * 2 + 1) / 2.0))
        hi = min(hi, float(default_max))
    opts = []
    v = lo
    while v <= hi + 1e-9:
        opts.append(round(v, 1))
        v += step
    return opts


def normalize_projections(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column names for projections so the app is robust to schema differences."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()

    # player id
    id_col = next((c for c in df.columns if c.lower() in ("player_id", "element", "id")), None)
    if id_col:
        df["player_id"] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")

    # name
    name_col = next((c for c in df.columns if c.lower() in ("web_name", "name", "player", "full_name")), None)
    if name_col:
        df["web_name"] = df[name_col]

    # team (preserve numeric ids where possible)
    team_id_col = next((c for c in df.columns if c.lower() in ("team_id", "teamid")), None)
    if team_id_col:
        df["team_id"] = pd.to_numeric(df[team_id_col], errors="coerce").astype("Int64")

    team_col = next((c for c in df.columns if c.lower() in ("team", "team_name", "club")), None)
    if team_col:
        df["team"] = df[team_col]
        if "team_id" not in df.columns:
            parsed = pd.to_numeric(df["team"], errors="coerce")
            if parsed.notna().any():
                df["team_id"] = parsed.astype("Int64")

    # position
    pos_col = next((c for c in df.columns if c.lower() in ("position", "pos", "position_name")), None)
    if pos_col:
        df["position"] = df[pos_col]

    POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    if "position" not in df.columns and "element_type" in df.columns:
        # FPL bootstrap-static uses element_type: 1 GK, 2 DEF, 3 MID, 4 FWD
        df["position"] = pd.to_numeric(df["element_type"], errors="coerce").map(POS_MAP)

    if "position" in df.columns:
        try:
            df["position"] = df["position"].apply(lambda v: POS_MAP.get(int(v), v) if pd.notna(v) else v)
        except Exception:
            pass

    # price
    # Prefer FPL bootstrap `now_cost` (tenths) over a placeholder `price` column.
    price_col = None
    lower_to_col = {str(c).lower(): c for c in df.columns}
    if "now_cost" in lower_to_col:
        price_col = lower_to_col["now_cost"]
    else:
        price_col = next((c for c in df.columns if str(c).lower() in ("price", "value", "£m")), None)
    if price_col:
        price = pd.to_numeric(df[price_col], errors="coerce")
        # FPL now_cost is in tenths (e.g., 75 => £7.5m)
        if str(price_col).lower() == "now_cost":
            price = price / 10.0
        elif price.max() > 1000:
            price = price / 10.0
        df["price"] = price.fillna(0.0)
    else:
        df["price"] = 0.0

    # projected points
    proj_col = next(
        (
            c
            for c in df.columns
            if c.lower() in ("proj_points", "projected_points", "proj", "predicted_points", "projected")
        ),
        None,
    )
    if proj_col:
        df["proj_points"] = pd.to_numeric(df[proj_col], errors="coerce").fillna(0.0)
    else:
        df["proj_points"] = 0.0

    # If the projections file is the multi-GW output (GWxx_proj_points columns), compute a total.
    gw_cols = [c for c in df.columns if re.match(r"^GW\d+_proj_points$", str(c))]
    if gw_cols and ("proj_points" not in df.columns or pd.to_numeric(df["proj_points"], errors="coerce").fillna(0).sum() == 0):
        df["proj_points"] = (
            df[gw_cols]
            .apply(lambda s: pd.to_numeric(s, errors="coerce"))
            .fillna(0.0)
            .sum(axis=1)
        )

    for q in ("p10", "p50", "p90"):
        q_col = next((c for c in df.columns if c.lower() == q), None)
        df[q] = pd.to_numeric(df[q_col], errors="coerce").fillna(0.0) if q_col else 0.0

    # minutes probability (optional)
    mp_col = next((c for c in df.columns if c.lower() in ("minutes_prob", "minutes_probability", "min_prob")), None)
    if mp_col:
        df["minutes_prob"] = pd.to_numeric(df[mp_col], errors="coerce").fillna(0.0)
        df["_has_minutes_prob"] = True
    else:
        df["minutes_prob"] = 1.0
        df["_has_minutes_prob"] = False

    # If we only have FPL bootstrap team id, treat it as team_id for labeling.
    if "team_id" not in df.columns and "fpl_team_id" in df.columns:
        df["team_id"] = pd.to_numeric(df["fpl_team_id"], errors="coerce").astype("Int64")

    # team labels + badges
    if "team" in df.columns or "team_id" in df.columns:
        team_lookup = load_team_lookup()

        def _team_short(team_id: int | None, team_raw: Any) -> str:
            if team_id is not None and team_id in team_lookup:
                return team_lookup[team_id].get("short_name") or str(team_id)
            if pd.notna(team_raw):
                return str(team_raw)
            return ""

        if "team_id" in df.columns:
            df["team"] = df.apply(
                lambda r: _team_short((int(r["team_id"]) if pd.notna(r["team_id"]) else None), r.get("team")),
                axis=1,
            )
        else:
            df["team"] = df["team"].apply(lambda t: _team_short(None, t))

        def _badge_for_row(r) -> str:
            tid = None
            try:
                tid = int(r.get("team_id")) if pd.notna(r.get("team_id")) else None
            except Exception:
                tid = None
            short = r.get("team")
            p = badge_path(short) if short else ""
            if p and Path(p).exists():
                return p
            return badge_path(str(tid)) if tid is not None else p

        df["badge"] = df.apply(_badge_for_row, axis=1)

    return df


@st.cache_data
def load_bootstrap_elements() -> pd.DataFrame:
    """Load FPL bootstrap-static elements for richer table columns."""
    try:
        r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=20)
        r.raise_for_status()
        payload = r.json()
        els = payload.get("elements", []) if isinstance(payload, dict) else []
        df = pd.DataFrame(els)
        if not df.empty and "id" in df.columns:
            df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
        return df
    except Exception:
        return pd.DataFrame()


def enrich_with_fpl_stats(projections: pd.DataFrame) -> pd.DataFrame:
    if projections is None or projections.empty:
        return projections
    els = load_bootstrap_elements()
    if els.empty or "id" not in els.columns or "player_id" not in projections.columns:
        return projections

    keep = [
        # IDs / identity
        "id",
        "first_name",
        "second_name",
        "web_name",
        "element_type",

        # Availability / market signals
        "chance_of_playing_this_round",
        "chance_of_playing_next_round",
        "news",
        "now_cost",
        "selected_by_percent",
        "value_form",
        "value_season",
        "ep_next",
        "ep_this",

        # Points
        "total_points",
        "event_points",
        "points_per_game",
        "form",

        # Expected / ICT
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
        "expected_goals_per_90",
        "expected_assists_per_90",
        "expected_goal_involvements_per_90",
        "expected_goals_conceded_per_90",
        "influence",
        "creativity",
        "threat",
        "ict_index",

        # Underlying basic stats
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "starts",

        # Bonus
        "bonus",
        "bps",

        # Team id
        "team",
    ]
    keep = [c for c in keep if c in els.columns]
    stats = els[keep].copy().rename(
        columns={
            "id": "player_id",
            "team": "fpl_team_id",
            "web_name": "web_name_fpl",
        }
    )
    for c in stats.columns:
        if c in ("player_id", "web_name_fpl", "first_name", "second_name", "news"):
            continue
        stats[c] = pd.to_numeric(stats[c], errors="coerce")

    out = projections.merge(stats, on="player_id", how="left", suffixes=("", "_fpl"))

    # Fill missing display name from bootstrap if needed
    if "web_name" not in out.columns and "web_name_fpl" in out.columns:
        out["web_name"] = out["web_name_fpl"]
    elif "web_name" in out.columns and "web_name_fpl" in out.columns:
        out["web_name"] = out["web_name"].fillna(out["web_name_fpl"])

    # Promote team id to team_id if projections don't have it
    if "team_id" not in out.columns and "fpl_team_id" in out.columns:
        out["team_id"] = pd.to_numeric(out["fpl_team_id"], errors="coerce").astype("Int64")

    out["xGI"] = (out.get("expected_goals", 0).fillna(0) + out.get("expected_assists", 0).fillna(0))
    out["GI"] = (out.get("goals_scored", 0).fillna(0) + out.get("assists", 0).fillna(0))

    team_key = "team_id" if "team_id" in out.columns else "fpl_team_id"
    if team_key in out.columns:
        totals = out.groupby(team_key, dropna=False).agg(team_xGI=("xGI", "sum"), team_GI=("GI", "sum")).reset_index()
        out = out.merge(totals, on=team_key, how="left")
        out["pct_xGI"] = out.apply(
            lambda r: (100.0 * float(r.get("xGI", 0) or 0) / float(r.get("team_xGI", 0) or 1))
            if float(r.get("team_xGI", 0) or 0) > 0
            else 0.0,
            axis=1,
        )
        out["pct_GI"] = out.apply(
            lambda r: (100.0 * float(r.get("GI", 0) or 0) / float(r.get("team_GI", 0) or 1))
            if float(r.get("team_GI", 0) or 0) > 0
            else 0.0,
            axis=1,
        )
    else:
        out["pct_xGI"] = 0.0
        out["pct_GI"] = 0.0

    out["gametime"] = (pd.to_numeric(out.get("minutes_prob", 0), errors="coerce").fillna(0.0) * 100.0)
    return out


@st.cache_data
def load_insights_playerstats(*, season: str = DEFAULT_INSIGHTS_SEASON) -> pd.DataFrame:
    """Load FPL-Core-Insights season-level playerstats.csv (latest GW per player)."""
    path = INSIGHTS_ROOT / season / "playerstats.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    if "id" in df.columns and "player_id" not in df.columns:
        df = df.rename(columns={"id": "player_id"})

    if "player_id" not in df.columns:
        return pd.DataFrame()

    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    if "gw" in df.columns:
        df["gw"] = pd.to_numeric(df["gw"], errors="coerce").astype("Int64")

    # Keep the latest gameweek snapshot per player.
    if "gw" in df.columns:
        df = df.dropna(subset=["player_id", "gw"]).copy()
        df["player_id"] = df["player_id"].astype(int)
        df["gw"] = df["gw"].astype(int)
        df = df.sort_values(["player_id", "gw"]).groupby("player_id", sort=False, as_index=False).tail(1)
    else:
        df = df.dropna(subset=["player_id"]).copy()
        df["player_id"] = df["player_id"].astype(int)

    return df


def enrich_with_insights_playerstats(projections: pd.DataFrame, *, season: str = DEFAULT_INSIGHTS_SEASON) -> pd.DataFrame:
    """Merge Insights playerstats into projections (fills tackle/CBI/recoveries/defcon etc.)."""
    if projections is None or projections.empty or "player_id" not in projections.columns:
        return projections

    stats = load_insights_playerstats(season=season)
    if stats.empty or "player_id" not in stats.columns:
        return projections

    # Only bring across the fields we want to display.
    desired = [
        "player_id",
        "chance_of_playing_this_round",
        "chance_of_playing_next_round",
        "now_cost",
        "selected_by_percent",
        "value_form",
        "value_season",
        "total_points",
        "event_points",
        "points_per_game",
        "form",
        "expected_goals_per_90",
        "expected_assists_per_90",
        "expected_goal_involvements_per_90",
        "expected_goals_conceded_per_90",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "starts",
        "defensive_contribution",
        "defensive_contribution_per_90",
        "tackles",
        "clearances_blocks_interceptions",
        "recoveries",
        "clean_sheets_per_90",
        "goals_conceded_per_90",
        "starts_per_90",
        "news",
    ]
    keep = [c for c in desired if c in stats.columns]
    s2 = stats[keep].copy()

    # Keep string columns as strings; coerce numeric where appropriate.
    for c in s2.columns:
        if c in ("player_id", "news"):
            continue
        s2[c] = pd.to_numeric(s2[c], errors="coerce")

    out = projections.merge(s2, on="player_id", how="left", suffixes=("", "_insights"))

    # Prefer live bootstrap for some fields, but fill any missing from insights.
    def _fill(col: str) -> None:
        alt = f"{col}_insights"
        if col in out.columns and alt in out.columns:
            out[col] = out[col].fillna(out[alt])

    for col in (
        "chance_of_playing_this_round",
        "chance_of_playing_next_round",
        "now_cost",
        "selected_by_percent",
        "value_form",
        "value_season",
        "total_points",
        "event_points",
        "points_per_game",
        "form",
        "expected_goals_per_90",
        "expected_assists_per_90",
        "expected_goal_involvements_per_90",
        "expected_goals_conceded_per_90",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "starts",
        "defensive_contribution",
        "defensive_contribution_per_90",
        "tackles",
        "clearances_blocks_interceptions",
        "recoveries",
        "clean_sheets_per_90",
        "goals_conceded_per_90",
        "starts_per_90",
        "news",
    ):
        _fill(col)

    # Drop the suffixed columns to avoid clutter.
    out = out.drop(columns=[c for c in out.columns if str(c).endswith("_insights")])
    return out


def build_clean_playerstats_view(
    df: pd.DataFrame,
    *,
    per_90: bool,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    def _series(name: str, default=None) -> pd.Series:
        if name in df.columns:
            return df[name]
        return pd.Series([default] * len(df))

    minutes = pd.to_numeric(_series("minutes"), errors="coerce")

    # MID-DM heuristic for showing defensive columns.
    pos = _series("position", "").astype(str)
    xgi90 = pd.to_numeric(_series("expected_goal_involvements_per_90"), errors="coerce").fillna(0.0)
    defcon90 = pd.to_numeric(_series("defensive_contribution_per_90"), errors="coerce").fillna(0.0)
    threat = pd.to_numeric(_series("threat"), errors="coerce").fillna(0.0)
    creativity = pd.to_numeric(_series("creativity"), errors="coerce").fillna(0.0)

    defense_proxy = defcon90
    attack_proxy = xgi90 + 0.005 * (threat + creativity)
    is_mid_dm = (pos == "MID") & (defense_proxy >= (attack_proxy * 1.25).clip(lower=0.6))
    show_def_cols = pos.isin(["DEF", "GK"]) | is_mid_dm

    selected_by_percent = pd.to_numeric(_series("selected_by_percent"), errors="coerce")

    view = pd.DataFrame(
        {
            "web_name": _series("web_name", ""),
            "position": pos,
            "team": _series("team", ""),

            "chance_of_playing_this_round": _series("chance_of_playing_this_round"),
            "chance_of_playing_next_round": _series("chance_of_playing_next_round"),
            "news": _series("news", ""),

            "now_cost": _series("now_cost"),
            "selected_by_percent": selected_by_percent.map(lambda x: f"{x:.1f}%" if pd.notna(x) else pd.NA),
            "value_form": _series("value_form"),
            "value_season": _series("value_season"),

            "total_points": _series("total_points"),
            "event_points": _series("event_points"),
            "points_per_game": _series("points_per_game"),
            "form": _series("form"),

            "expected_goals_per_90": _series("expected_goals_per_90"),
            "expected_assists_per_90": _series("expected_assists_per_90"),
            "expected_goal_involvements_per_90": _series("expected_goal_involvements_per_90"),
            "expected_goals_conceded_per_90": _series("expected_goals_conceded_per_90"),

            "influence": _series("influence"),
            "creativity": _series("creativity"),
            "threat": _series("threat"),
            "ict_index": _series("ict_index"),

            "minutes": minutes,
            "goals_scored": _series("goals_scored"),
            "assists": _series("assists"),
            "clean_sheets": _series("clean_sheets"),
            "goals_conceded": _series("goals_conceded"),
            "starts": _series("starts"),

            "defensive_contribution_per_90": _series("defensive_contribution_per_90"),
            "tackles": _series("tackles"),
            "clearances_blocks_interceptions": _series("clearances_blocks_interceptions"),
            "recoveries": _series("recoveries"),
        }
    )

    # Chance-of-playing sometimes comes as 0..1; normalize to 0..100 if needed.
    for c in ("chance_of_playing_this_round", "chance_of_playing_next_round"):
        s = pd.to_numeric(view[c], errors="coerce")
        if s.notna().any() and float(s.dropna().max()) <= 1.0:
            view[c] = (s * 100.0).round(0)
        else:
            view[c] = s

    # Defensive-only: xGC/90
    view.loc[~pos.isin(["DEF", "GK"]), "expected_goals_conceded_per_90"] = pd.NA

    # Role-limited defensive columns
    for c in ("defensive_contribution_per_90", "tackles", "clearances_blocks_interceptions", "recoveries"):
        view.loc[~show_def_cols, c] = pd.NA

    # Optional: convert the raw box-score stats to per-90 for display.
    if per_90:
        mins = pd.to_numeric(view["minutes"], errors="coerce")
        denom = (mins / 90.0).where(mins.notna() & (mins > 0), other=pd.NA)
        per90_cols = ["goals_scored", "assists", "clean_sheets", "goals_conceded", "starts", "tackles", "clearances_blocks_interceptions", "recoveries"]
        for c in per90_cols:
            view[c] = pd.to_numeric(view[c], errors="coerce") / denom

    # Order by projected points if present, else by total_points.
    if "proj_points" in df.columns:
        view["proj_points"] = pd.to_numeric(df.get("proj_points"), errors="coerce")
        view = view.sort_values("proj_points", ascending=False)
    else:
        view = view.sort_values("total_points", ascending=False)

    return view


def _first_col(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series([None] * len(df))


def _compute_def_contrib(df: pd.DataFrame) -> pd.Series:
    candidates = ["defensive_contributions", "def_contrib", "def_contribution"]
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")

    parts = []
    for c in ["tackles_won", "tackles", "interceptions", "recoveries", "blocks", "clearances", "aerials_won"]:
        if c in df.columns:
            parts.append(pd.to_numeric(df[c], errors="coerce").fillna(0.0))
    if not parts:
        return pd.Series([None] * len(df))
    s = parts[0]
    for p in parts[1:]:
        s = s + p
    return s


def _drop_empty_columns(view: pd.DataFrame, *, keep: list[str] | None = None) -> pd.DataFrame:
    if view is None or view.empty:
        return pd.DataFrame()
    if keep is None:
        keep = []

    out = view.copy()
    for c in list(out.columns):
        if c in keep:
            continue
        s = out[c]
        if s.isna().all():
            out = out.drop(columns=[c])
            continue

        # Numeric-empty: all zeros or NaNs
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().any() and float(num.fillna(0.0).abs().sum()) == 0.0:
            out = out.drop(columns=[c])
            continue

        # Object-empty: blank strings
        if num.isna().all() and s.dtype == object:
            ss = s.astype(str).str.strip()
            if (ss == "") .all() or (ss.str.lower() == "nan").all():
                out = out.drop(columns=[c])

    return out


def _to_per_90(series: pd.Series, minutes: pd.Series) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce")
    mins = pd.to_numeric(minutes, errors="coerce")
    denom = (mins / 90.0).where(mins.notna() & (mins > 0), other=pd.NA)
    return num / denom


def build_key_stats_view(
    df: pd.DataFrame,
    *,
    per_90: bool = False,
    use_full_names: bool = False,
    include_uncertainty: bool = False,
) -> pd.DataFrame:
    """Key Stats table built from projections (requested: show projections file)."""
    if df is None or df.empty:
        return pd.DataFrame()

    # Requested column set (filled best-effort from live FPL bootstrap stats + our projections)
    ot = _first_col(df, ["shots_on_target", "sot", "shots_on_tgt", "shot_on_target"])
    bc = _first_col(df, ["big_chances", "big_chance", "big_chances_total"])
    kp = _first_col(df, ["key_passes", "chances_created", "kp"])
    bcc = _first_col(df, ["big_chances_created", "big_chance_created", "bcc"])
    dc = _compute_def_contrib(df)

    minutes = df.get("minutes", pd.Series(dtype=float))

    view = pd.DataFrame(
        {
            "Name": df.get("web_name", pd.Series(dtype=str)),
            "Club": df.get("team", pd.Series(dtype=str)),
            "£M": df.get("price", 0),
            "App": df.get("starts", pd.Series(dtype=float)),
            "Mins": minutes,
            "S": df.get("starts", pd.Series(dtype=float)),
            "OT": ot,
            "In": df.get("influence", pd.Series(dtype=float)),
            "BC": bc,
            "xG": df.get("expected_goals", pd.Series(dtype=float)),
            "G": df.get("goals_scored", pd.Series(dtype=float)),
            "%xGI": df.get("pct_xGI", pd.Series(dtype=float)),
            "%GI": df.get("pct_GI", pd.Series(dtype=float)),
            "xGI": df.get("xGI", pd.Series(dtype=float)),
            "GI": df.get("GI", pd.Series(dtype=float)),
            "KP": kp,
            "BCC": bcc,
            "xA": df.get("expected_assists", pd.Series(dtype=float)),
            "A": df.get("assists", pd.Series(dtype=float)),
            "DC": dc,
            "xPts": df.get("proj_points", 0),
            "P10": df.get("p10", pd.Series(dtype=float)) if include_uncertainty else pd.Series([pd.NA] * len(df)),
            "P50": df.get("p50", pd.Series(dtype=float)) if include_uncertainty else pd.Series([pd.NA] * len(df)),
            "P90": df.get("p90", pd.Series(dtype=float)) if include_uncertainty else pd.Series([pd.NA] * len(df)),
            "BPS": df.get("bps", pd.Series(dtype=float)),
            "B": df.get("bonus", pd.Series(dtype=float)),
            "Pts": df.get("total_points", pd.Series(dtype=float)),
        }
    )

    if bool(df.get("_has_minutes_prob", False).any() if "_has_minutes_prob" in df.columns else False):
        view["Min%"] = (pd.to_numeric(df.get("minutes_prob", 0), errors="coerce").fillna(0.0) * 100.0).round(0)

    # Sort default
    if "xPts" in view.columns:
        view = view.sort_values("xPts", ascending=False)

    # Optional per-90 view (skip identifiers / price)
    if per_90:
        # Only convert underlying action metrics; keep point-like columns as-is.
        do_not_convert = {"Name", "Club", "£M", "Mins", "xPts", "P10", "P50", "P90", "BPS", "B", "Pts", "Min%"}
        for c in list(view.columns):
            if str(c).startswith("%"):
                do_not_convert.add(c)
            if c in do_not_convert:
                continue
            view[c] = _to_per_90(view[c], view["Mins"])

        # Make it obvious it is per-90
        rename_map = {c: (f"{c}/90" if c not in do_not_convert else c) for c in view.columns}
        view = view.rename(columns=rename_map)

    # Drop columns that are fully empty (this gets rid of OT/BC/Pxx when projections file doesn't include them)
    view = _drop_empty_columns(view, keep=["Name", "Club", "£M"])

    if use_full_names:
        view = view.rename(columns={k: v for k, v in STAT_FULL_NAMES.items() if k in view.columns})

    return view


def _fdr_colors(fdr: int) -> tuple[str, str]:
    palette = {
        1: ("#BFEAD2", "#0B3D2E"),
        2: ("#D9F2E4", "#0B3D2E"),
        3: ("#F6E7B2", "#5A4300"),
        4: ("#F8D1B0", "#5A2A00"),
        5: ("#F6B8B8", "#5A0000"),
    }
    try:
        f = int(fdr)
    except Exception:
        f = 3
    return palette.get(f, palette[3])


def _read_b64(path: Path) -> str:
    try:
        raw = path.read_bytes()
        return base64.b64encode(raw).decode("utf-8")
    except Exception:
        return ""


def _build_fixture_index(fixtures_df: pd.DataFrame) -> dict[int, dict[int, list[dict[str, Any]]]]:
    out: dict[int, dict[int, list[dict[str, Any]]]] = {}
    if fixtures_df is None or fixtures_df.empty:
        return out
    if "event" not in fixtures_df.columns:
        return out

    for _, r in fixtures_df.iterrows():
        try:
            gw = int(r.get("event"))
        except Exception:
            continue
        th = r.get("team_h")
        ta = r.get("team_a")
        if pd.isna(th) or pd.isna(ta):
            continue
        try:
            th = int(th)
            ta = int(ta)
        except Exception:
            continue

        def _push(team_id: int, opp_id: int, ha: str, fdr_val: Any):
            try:
                fdr_int = int(fdr_val) if pd.notna(fdr_val) else 3
            except Exception:
                fdr_int = 3
            team_map = out.setdefault(team_id, {})
            team_map.setdefault(gw, []).append({"opp_id": opp_id, "ha": ha, "fdr": fdr_int})

        _push(th, ta, "H", r.get("team_h_difficulty"))
        _push(ta, th, "A", r.get("team_a_difficulty"))

    return out


def _render_ticker_html(
    fixtures_index: dict[int, dict[int, list[dict[str, Any]]]],
    gw_start: int,
    gw_end: int,
    only_team_ids: set[int] | None = None,
) -> str:
    teams = load_team_lookup()
    team_ids = sorted(teams.keys())
    if only_team_ids:
        team_ids = [t for t in team_ids if t in only_team_ids]

    def pill(text: str, fdr: int) -> str:
        bg, fg = _fdr_colors(fdr)
        return (
            "<span style='display:inline-block;padding:3px 8px;border-radius:8px;"
            f"background:{bg};color:{fg};font-size:12px;line-height:16px;margin:2px;white-space:nowrap'>"
            f"{text}</span>"
        )

    cols = list(range(gw_start, gw_end + 1))
    html = [
        "<div style='overflow:auto'>",
        "<table style='border-collapse:separate;border-spacing:0 6px;width:100%'>",
        "<thead><tr>",
        "<th style='text-align:left;padding:6px 10px;font-weight:600'>Team</th>",
    ]
    for gw in cols:
        html.append(f"<th style='text-align:center;padding:6px 10px;font-weight:600'>GW{gw}</th>")
    html.append("</tr></thead><tbody>")

    for tid in team_ids:
        recall = fixtures_index.get(tid, {})
        t_short = teams.get(tid, {}).get("short_name") or str(tid)
        badge = badge_path(t_short)
        badge_html = ""
        try:
            if badge and Path(badge).exists():
                badge_html = (
                    f"<img src='{badge}' style='width:18px;height:18px;vertical-align:-3px;margin-right:6px'/>"
                )
        except Exception:
            badge_html = ""

        html.append("<tr>")
        html.append(
            "<td style='padding:6px 10px;background:#fff;border:1px solid #eee;border-radius:10px;white-space:nowrap'>"
            f"{badge_html}{t_short}</td>"
        )
        for gw in cols:
            fixtures_here = recall.get(gw, [])
            if not fixtures_here:
                cell = ""
            else:
                parts = []
                for fx in fixtures_here:
                    opp_short = teams.get(int(fx["opp_id"]), {}).get("short_name") or str(fx["opp_id"])
                    parts.append(pill(f"{opp_short} ({fx['ha']})", int(fx["fdr"])))
                cell = "".join(parts)

            html.append(
                "<td style='padding:6px 10px;background:#fff;border:1px solid #eee;border-radius:10px;text-align:center'>"
                + (cell or "<span style='color:#bbb'>—</span>")
                + "</td>"
            )
        html.append("</tr>")
    html.append("</tbody></table></div>")
    return "".join(html)


def _team_summary(team_df: pd.DataFrame) -> dict[str, Any]:
    if team_df is None or team_df.empty:
        return {"cost": 0.0, "counts": Counter(), "clubs": Counter()}
    cost = float(team_df.get("price", 0).fillna(0).sum()) if "price" in team_df.columns else 0.0
    counts = Counter(team_df.get("position", []).tolist())
    clubs = Counter(team_df.get("team", []).tolist())
    return {"cost": cost, "counts": counts, "clubs": clubs}


def _derive_formation(starters_df: pd.DataFrame) -> str | None:
    if starters_df is None or starters_df.empty or "position" not in starters_df.columns:
        return None
    counts = Counter(starters_df["position"].tolist())
    gk = int(counts.get("GK", 0))
    d = int(counts.get("DEF", 0))
    m = int(counts.get("MID", 0))
    f = int(counts.get("FWD", 0))
    if gk != 1 or (gk + d + m + f) != 11:
        return None
    for formation, (gk_n, def_n, mid_n, fwd_n) in ALLOWED_FORMATIONS.items():
        if (gk, d, m, f) == (gk_n, def_n, mid_n, fwd_n):
            return formation
    return None


def _validate_starting_xi(team_df: pd.DataFrame, starter_ids: list[int]) -> tuple[bool, str, str | None]:
    if team_df is None or team_df.empty:
        return False, "No squad loaded", None
    if len(starter_ids) != 11:
        return False, "Starting XI must have 11 players", None
    starters = team_df[team_df["player_id"].astype(int).isin([int(x) for x in starter_ids])].copy()
    if starters.empty or len(starters) != 11:
        return False, "Starting XI selection contains unknown players", None
    # If we don't have position data, allow selection but skip formation validation.
    if "position" not in starters.columns:
        return True, "", None

    formation = _derive_formation(starters)
    if not formation:
        return False, "Invalid formation. Use one of: 343, 352, 451, 442, 433, 532, 541", None
    return True, "", formation


def _best_xi_ids(team_df: pd.DataFrame) -> list[int]:
    if team_df is None or team_df.empty:
        return []
    t = team_df.copy()
    t["_score"] = pd.to_numeric(t.get("proj_points", 0), errors="coerce").fillna(0.0)
    if "position" not in t.columns:
        return t.sort_values("_score", ascending=False)["player_id"].astype(int).head(11).tolist()
    best: tuple[float, list[int]] = (-1e9, [])
    for _, (gk_n, def_n, mid_n, fwd_n) in ALLOWED_FORMATIONS.items():
        ids: list[int] = []
        for pos, n in [("GK", gk_n), ("DEF", def_n), ("MID", mid_n), ("FWD", fwd_n)]:
            pool = t[t["position"] == pos].sort_values("_score", ascending=False).head(n)
            if len(pool) < n:
                ids = []
                break
            ids.extend(pool["player_id"].astype(int).tolist())
        if not ids:
            continue
        score = float(t[t["player_id"].astype(int).isin(ids)]["_score"].sum())
        if score > best[0]:
            best = (score, ids)
    return best[1]


def _render_pitch_html(starters_df: pd.DataFrame, bench_df: pd.DataFrame) -> str:
    pitch_path = Path("site") / "assets" / "pitch_vertical.svg"
    if not pitch_path.exists():
        pitch_path = Path("site") / "assets" / "pitch.svg"
    b64 = _read_b64(pitch_path)
    bg = f"data:image/svg+xml;base64,{b64}" if b64 else ""

    formation = _derive_formation(starters_df)
    gk_n, def_n, mid_n, fwd_n = ALLOWED_FORMATIONS.get(formation or "3-5-2", ALLOWED_FORMATIONS["3-5-2"])

    starters = starters_df.copy() if starters_df is not None else pd.DataFrame()
    bench = bench_df.copy() if bench_df is not None else pd.DataFrame()
    if not starters.empty and "proj_points" in starters.columns:
        starters = starters.sort_values("proj_points", ascending=False)
    if not bench.empty and "proj_points" in bench.columns:
        bench = bench.sort_values("proj_points", ascending=False)

    def badge_or_fallback(short: str) -> str:
        p = badge_path(short)
        try:
            if p and Path(p).exists():
                return f"<img src='{p}' style='width:22px;height:22px;border-radius:50%;background:#fff;padding:2px'/>"
        except Exception:
            pass
        return f"<span style='display:inline-block;width:22px;height:22px;border-radius:50%;background:#fff;line-height:22px;font-size:10px;font-weight:700;color:#333'>{short[:3]}</span>"

    def card(row: pd.Series) -> str:
        name = str(row.get("web_name", ""))
        team = str(row.get("team", ""))
        price = float(row.get("price", 0) or 0)
        return (
            "<div style='min-width:110px;max-width:140px;background:rgba(255,255,255,0.92);"
            "border-radius:10px;padding:6px 8px;text-align:center;border:1px solid rgba(0,0,0,0.06)'>"
            f"<div style='margin-bottom:4px'>{badge_or_fallback(team)}</div>"
            f"<div style='font-weight:700;font-size:12px;line-height:14px'>{name}</div>"
            f"<div style='font-size:11px;color:#555'>£{price:.1f}m</div>"
            "</div>"
        )

    def row_html(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "<div style='display:flex;justify-content:center;gap:14px;margin:12px 0'></div>"
        items = "".join([card(df.iloc[i]) for i in range(len(df))])
        return f"<div style='display:flex;justify-content:center;gap:14px;flex-wrap:wrap;margin:12px 0'>{items}</div>"

    def _take(pos: str, n: int) -> pd.DataFrame:
        if starters.empty:
            return pd.DataFrame()
        return starters[starters["position"] == pos].head(n)

    gk_row = row_html(_take("GK", gk_n))
    def_row = row_html(_take("DEF", def_n))
    mid_row = row_html(_take("MID", mid_n))
    fwd_row = row_html(_take("FWD", fwd_n))

    bench_items = "".join([card(bench.iloc[i]) for i in range(min(len(bench), 4))]) if not bench.empty else ""
    bench_html = (
        "<div style='display:flex;justify-content:center;gap:10px;flex-wrap:wrap;margin-top:10px'>"
        + (bench_items or "<span style='color:#eee'>Add players to see bench</span>")
        + "</div>"
    )

    formation_html = ""
    if formation:
        formation_html = (
            "<div style='text-align:center;color:#e8f5e9;font-weight:800;margin-bottom:6px'>"
            + formation
            + "</div>"
        )

    return (
        "<div style='width:100%;border-radius:14px;overflow:hidden;border:1px solid #e6e6e6'>"
        + f"<div style='background-image:url({bg});background-size:cover;background-position:center;"
        + "padding:14px 10px 10px 10px;min-height:760px'>"
        + formation_html
        + f"{gk_row}{def_row}{mid_row}{fwd_row}"
        + "</div>"
        + "<div style='background:#0f2b1a;padding:10px'>"
        + "<div style='color:#fff;font-weight:700;font-size:12px;margin-bottom:6px'>Bench</div>"
        + f"{bench_html}"
        + "</div></div>"
    )


def _render_interactive_cards(
    *,
    title: str,
    team_df: pd.DataFrame,
    starters_ids: list[int],
    bench_ids: list[int],
    state_prefix: str,
    allow_remove: bool,
    on_remove_player: Callable[[int], None] | None = None,
) -> tuple[list[int], list[int]]:
    if team_df is None or team_df.empty:
        st.info("No squad loaded yet.")
        return starters_ids, bench_ids

    starters_ids = [int(x) for x in starters_ids]
    bench_ids = [int(x) for x in bench_ids]

    starters_set = set(starters_ids)
    bench_ids = [x for x in bench_ids if x not in starters_set]

    sel_key = f"{state_prefix}_swap_sel"
    role_key = f"{state_prefix}_swap_role"
    st.session_state.setdefault(sel_key, None)
    st.session_state.setdefault(role_key, None)

    st.markdown(f"**{title}**")
    st.caption("Swap: click 🔴↓ on a starter, then 🟢↑ on a bench player (or vice versa).")

    starters_df = team_df[team_df["player_id"].astype(int).isin(starters_ids)].copy()
    bench_df = team_df[team_df["player_id"].astype(int).isin(bench_ids)].copy()

    ok, _, formation = _validate_starting_xi(team_df, starters_ids)
    if ok and formation:
        st.caption(f"Formation: {formation}")

    def _attempt_swap(id_a: int, role_a: str, id_b: int, role_b: str):
        nonlocal starters_ids, bench_ids
        if role_a == role_b:
            st.session_state[sel_key] = id_b
            st.session_state[role_key] = role_b
            return

        if role_a == "starter" and role_b == "bench":
            new_starters = [x for x in starters_ids if x != id_a] + [id_b]
            new_bench = [x for x in bench_ids if x != id_b] + [id_a]
        elif role_a == "bench" and role_b == "starter":
            new_starters = [x for x in starters_ids if x != id_b] + [id_a]
            new_bench = [x for x in bench_ids if x != id_a] + [id_b]
        else:
            return

        ok2, msg2, _ = _validate_starting_xi(team_df, new_starters)
        if ok2:
            starters_ids = new_starters
            bench_ids = new_bench
            st.session_state[sel_key] = None
            st.session_state[role_key] = None
            st.rerun()
        else:
            st.error(msg2)

    def _render_player_card(r: pd.Series, role: str):
        pid = int(r.get("player_id"))
        picked = (st.session_state.get(sel_key) == pid) and (st.session_state.get(role_key) == role)

        a, b, c = st.columns([1, 10, 1])
        with a:
            if role == "starter":
                if st.button("🔴↓", key=f"{state_prefix}_down_{pid}", help="Select starter to bench"):
                    sel = st.session_state.get(sel_key)
                    sel_role = st.session_state.get(role_key)
                    if sel is None:
                        st.session_state[sel_key] = pid
                        st.session_state[role_key] = "starter"
                        st.rerun()
                    else:
                        _attempt_swap(int(sel), str(sel_role), pid, "starter")
            else:
                if st.button("🟢↑", key=f"{state_prefix}_up_{pid}", help="Select bench to start"):
                    sel = st.session_state.get(sel_key)
                    sel_role = st.session_state.get(role_key)
                    if sel is None:
                        st.session_state[sel_key] = pid
                        st.session_state[role_key] = "bench"
                        st.rerun()
                    else:
                        _attempt_swap(int(sel), str(sel_role), pid, "bench")

        with b:
            team_code = r.get("team_code")
            team_code_num = None
            try:
                if team_code is not None and str(team_code) != "":
                    team_code_num = int(float(team_code))
            except Exception:
                team_code_num = None

            photo_url = get_player_photo_url(pid) if pid else "https://via.placeholder.com/80x80?text=N%2FA"
            badge_url = get_team_badge_url(team_code_num) if team_code_num else ""

            badge_fallback = r.get("badge")
            badge_html = ""
            if badge_url:
                fallback_src = str(badge_fallback or "")
                badge_html = (
                    f"<img src='{badge_url}' style='width:22px;height:22px;vertical-align:-5px;margin-right:8px' "
                    f"onerror=\"this.src='{fallback_src}'\"/>"
                )
            else:
                try:
                    if badge_fallback and Path(str(badge_fallback)).exists():
                        badge_html = f"<img src='{badge_fallback}' style='width:22px;height:22px;vertical-align:-5px;margin-right:8px'/>"
                except Exception:
                    badge_html = ""

            border = "2px solid #16a34a" if picked else "1px solid rgba(0,0,0,0.08)"
            st.markdown(
                "<div style='background:#fff;border-radius:14px;padding:12px 12px;margin:10px 0;"
                "box-shadow:0 6px 14px rgba(0,0,0,0.10);"
                f"border:{border}'>"
                "<div style='display:flex;align-items:center;gap:12px'>"
                f"<img src='{photo_url}' style='width:54px;height:54px;border-radius:50%;object-fit:cover;"
                "border:3px solid #00ff87;background:#f3f4f6' "
                "onerror=\"this.src='https://via.placeholder.com/80x80?text=N%2FA'\"/>"
                "<div style='flex:1'>"
                f"<div style='font-weight:900;font-size:14px;color:#111'>{r.get('web_name')}</div>"
                f"<div style='color:#6b7280;font-size:12px'>{badge_html}{r.get('team')} · {r.get('position')}</div>"
                "<div style='margin-top:6px;display:flex;justify-content:space-between;align-items:center'>"
                f"<span style='font-weight:800;color:#111'>£{float(r.get('price',0) or 0):.1f}m</span>"
                f"<span style='font-weight:800;color:#065f46;background:rgba(0,255,135,0.20);padding:2px 8px;border-radius:999px;font-size:12px'>"
                f"Pred {float(r.get('proj_points',0) or 0):.1f}"
                "</span>"
                "</div>"
                "</div>"
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        with c:
            if allow_remove:
                if st.button("❌", key=f"{state_prefix}_rm_{pid}", help="Remove from squad"):
                    if on_remove_player:
                        on_remove_player(pid)
                    st.rerun()
            else:
                st.button("✖", key=f"{state_prefix}_rm_disabled_{pid}", disabled=True)

    st.markdown("### Starters")
    starter_sort_cols = [c for c in ["position", "proj_points"] if c in starters_df.columns]
    starters_view = starters_df.sort_values(starter_sort_cols, ascending=[True, False][: len(starter_sort_cols)]) if starter_sort_cols else starters_df
    for _, r in starters_view.iterrows():
        _render_player_card(r, "starter")

    st.markdown("### Bench")
    bench_sort_cols = [c for c in ["position", "proj_points"] if c in bench_df.columns]
    bench_view = bench_df.sort_values(bench_sort_cols, ascending=[True, False][: len(bench_sort_cols)]) if bench_sort_cols else bench_df
    for _, r in bench_view.iterrows():
        _render_player_card(r, "bench")

    if st.button("Clear swap selection", key=f"{state_prefix}_clear_sel"):
        st.session_state[sel_key] = None
        st.session_state[role_key] = None
        st.rerun()

    return starters_ids, bench_ids


def _can_add_player(team_df: pd.DataFrame, p: pd.Series) -> tuple[bool, str]:
    if p is None or p.empty:
        return False, "Invalid player"
    if "player_id" in p and team_df is not None and not team_df.empty and "player_id" in team_df.columns:
        if int(p["player_id"]) in set(team_df["player_id"].astype(int).tolist()):
            return False, "Already added"

    summary = _team_summary(team_df)
    cost = summary["cost"] + float(p.get("price", 0) or 0)
    if cost > BUDGET + 1e-9:
        return False, "Over budget"
    if len(team_df) >= 15:
        return False, "Squad full"

    pos = str(p.get("position", ""))
    if pos in POS_CAP and summary["counts"].get(pos, 0) >= POS_CAP[pos]:
        return False, f"Max {pos} reached"

    club = str(p.get("team", ""))
    if club and summary["clubs"].get(club, 0) >= MAX_PER_CLUB:
        return False, "Max 3 per club"

    return True, ""


def load_projections() -> pd.DataFrame:
    # Prefer outputs/projections.csv (your model output)
    if PROJ_CSV_INTERNAL.exists():
        return load_csv(PROJ_CSV_INTERNAL)
    if PROJ_CSV.exists():
        return load_csv(PROJ_CSV)
    if PROJ_JSON.exists():
        return load_json(PROJ_JSON)
    return load_csv(PROJ_CSV_FALLBACK)


def main():
    st.set_page_config(
        page_title="FPL Analytics Hub",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
<style>
    :root {
        --fpl-purple: #37003c;
        --fpl-green: #00ff87;
        --fpl-pink: #ff2882;
    }
    .block-container { padding-top: 1.2rem; }
    .header-container {
        background: linear-gradient(135deg, var(--fpl-purple) 0%, var(--fpl-green) 100%);
        padding: 18px 16px;
        border-radius: 14px;
        margin: 0 0 18px 0;
        text-align: center;
        color: white;
    }
    .header-container h1 { margin: 0; font-size: 2.0rem; font-weight: 800; }
    .header-container p { margin: 6px 0 0 0; opacity: 0.9; }
</style>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="header-container">
    <h1>⚽ FPL Analytics Hub</h1>
    <p>Professional Fantasy Premier League Planning & Predictions</p>
</div>
""",
        unsafe_allow_html=True,
    )

    # Load data
    projections = load_projections()
    fixtures = load_json(FIXTURES_JSON) if FIXTURES_JSON.exists() else pd.DataFrame()

    # Normalize base identifiers (player_id/web_name) first
    projections = normalize_projections(projections)

    # Enrich with live FPL stats (adds element_type, now_cost, team id, etc.)
    projections = enrich_with_fpl_stats(projections)

    # Enrich with FPL-Core-Insights playerstats (tackles/CBI/recoveries/defcon + per-90 columns)
    projections = enrich_with_insights_playerstats(projections, season=DEFAULT_INSIGHTS_SEASON)

    # Normalize again so derived fields (team/position/price/proj_points) are populated
    projections = normalize_projections(projections)

    # Enrich with Opta-style stats (optional BYO)
    projections = enrich_with_opta_stats(projections)

    tabs = st.tabs(["My Team", "Projections", "Fixture Ticker", "Key Stats", "Player Profile", "Transfer Planner"])

    # Tab: My Team
    with tabs[0]:
        st.subheader("My Team")
        c_id, c_gw = st.columns([1, 1])
        with c_id:
            entry_id = int(
                st.number_input(
                    "Entry ID",
                    min_value=1,
                    value=int(st.session_state.get("entry_id", ENTRY_ID)),
                    step=1,
                )
            )
        with c_gw:
            gw = int(
                st.number_input(
                    "Gameweek",
                    min_value=1,
                    max_value=38,
                    value=int(st.session_state.get("gw", GW)),
                    step=1,
                )
            )

        st.session_state.entry_id = entry_id
        st.session_state.gw = gw

        picks = get_my_team(entry_id, gw)
        if not picks.empty and "element" in picks.columns and not projections.empty:
            el_ids = pd.to_numeric(picks["element"], errors="coerce").dropna().astype(int).tolist()

            team_df = projections[projections["player_id"].astype(int).isin(el_ids)].copy()
            team_df["__order"] = team_df["player_id"].apply(
                lambda x: el_ids.index(int(x)) if int(x) in el_ids else 999
            )
            team_df = team_df.sort_values("__order").drop(columns=["__order"])

            if not team_df.empty:
                my_key = f"{entry_id}:{gw}"
                if st.session_state.get("my_team_key") != my_key:
                    st.session_state.my_team_key = my_key
                    starters_ids = pd.to_numeric(picks[picks["position"] <= 11]["element"], errors="coerce").dropna().astype(int).tolist()
                    bench_ids = pd.to_numeric(picks[picks["position"] > 11]["element"], errors="coerce").dropna().astype(int).tolist()
                    st.session_state.my_starters_ids = starters_ids
                    st.session_state.my_bench_ids = bench_ids

                if st.button("Reset to FPL lineup", key="my_reset"):
                    st.session_state.my_starters_ids = pd.to_numeric(
                        picks[picks["position"] <= 11]["element"], errors="coerce"
                    ).dropna().astype(int).tolist()
                    st.session_state.my_bench_ids = pd.to_numeric(
                        picks[picks["position"] > 11]["element"], errors="coerce"
                    ).dropna().astype(int).tolist()
                    st.rerun()

                starters_ids = [int(x) for x in st.session_state.get("my_starters_ids", el_ids[:11])]
                bench_ids = [int(x) for x in st.session_state.get("my_bench_ids", el_ids[11:])]

                starters_ids, bench_ids = _render_interactive_cards(
                    title="My Team swaps",
                    team_df=team_df,
                    starters_ids=starters_ids,
                    bench_ids=bench_ids,
                    state_prefix="my",
                    allow_remove=False,
                )
                st.session_state.my_starters_ids = starters_ids
                st.session_state.my_bench_ids = bench_ids

                starters_df = team_df[team_df["player_id"].astype(int).isin(starters_ids)].copy()
                bench_df = team_df[team_df["player_id"].astype(int).isin(bench_ids)].copy()
                st.markdown(_render_pitch_html(starters_df, bench_df), unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Total Cost", f"£{float(team_df.get('price', 0).fillna(0).sum()):.1f}m")
                with c2:
                    st.metric("Projected Points", f"{float(team_df.get('proj_points', 0).fillna(0).sum()):.1f}")
            else:
                st.info("No matching projection rows for your picks.")
        else:
            st.info("Could not fetch picks from FPL API. Ensure entry ID and GW are correct.")

    # Tab: Projections
    with tabs[1]:
        st.subheader("Player Projections")
        if projections.empty:
            st.warning("No projections available.")
        else:
            clubs = sorted(projections.get("team", pd.Series(dtype=str)).dropna().unique().tolist())
            positions = sorted(projections.get("position", pd.Series(dtype=str)).dropna().unique().tolist())

            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                pos = st.selectbox("Position", ["All"] + positions)
            with c2:
                club = st.selectbox("Club", ["All"] + clubs)  # requested club filter
            with c3:
                price_opts = price_options_from_proj(projections)
                price_min = st.selectbox("Price min", options=price_opts, index=0)
                price_max = st.selectbox("Price max", options=price_opts, index=len(price_opts) - 1)

            filt = projections.copy()
            if pos != "All" and "position" in filt.columns:
                filt = filt[filt["position"] == pos]
            if club != "All" and "team" in filt.columns:
                filt = filt[filt["team"] == club]
            if "price" in filt.columns:
                filt = filt[(filt["price"] >= float(price_min)) & (filt["price"] <= float(price_max))]

            view = build_key_stats_view(filt)
            st.dataframe(view.head(300), width="stretch")

    # Tab: Fixture ticker
    with tabs[2]:
        st.subheader("Fixture Difficulty Ticker")
        if fixtures.empty:
            st.warning("No fixtures data available (data/fixtures.json)")
        else:
            if "event" not in fixtures.columns:
                st.warning("Fixtures file isn't in event format (missing 'event').")
            else:
                gw_min = int(pd.to_numeric(fixtures["event"], errors="coerce").dropna().min())
                gw_max = int(pd.to_numeric(fixtures["event"], errors="coerce").dropna().max())

                gw_min = max(gw_min, 21)
                gw_max = min(gw_max, 38)

                if gw_min > gw_max:
                    st.warning("No fixtures available for GW 21–38.")
                    st.stop()

                if gw_min == gw_max:
                    gw_start, gw_end = gw_min, gw_max
                else:
                    gw_start, gw_end = st.slider(
                        "Gameweek Range",
                        min_value=gw_min,
                        max_value=gw_max,
                        value=(gw_min, gw_max),
                    )

                teams = load_team_lookup()
                team_opts = [teams[k].get("short_name") for k in sorted(teams.keys())]
                selected = st.multiselect("Filter by team(s)", options=team_opts, default=[])
                selected_ids = None
                if selected:
                    inv = {v.get("short_name"): k for k, v in teams.items()}
                    selected_ids = {int(inv[s]) for s in selected if s in inv}

                fixtures_idx = _build_fixture_index(fixtures)
                html = _render_ticker_html(fixtures_idx, int(gw_start), int(gw_end), selected_ids)
                st.markdown(html, unsafe_allow_html=True)

    # Tab: Key Stats (requested to show projections file)
    with tabs[3]:
        st.subheader("Key Stats")
        if projections.empty:
            st.info("No projections available.")
        else:
            c1, c2, c3 = st.columns(3)
            per_90 = c1.toggle("Show box stats per 90", value=False)
            show_uncertainty = c2.toggle("Show P10/P50/P90", value=False)
            full_names = c3.toggle("Use full column names", value=False)

            with st.expander("What do these columns mean?"):
                glossary = pd.DataFrame(
                    {"Column": list(STAT_GLOSSARY.keys()), "Description": list(STAT_GLOSSARY.values())}
                )
                st.dataframe(glossary, width="stretch", hide_index=True)

            q = st.text_input("Search player or club")
            df_stats = projections.copy()
            if q and "web_name" in df_stats.columns:
                name_match = df_stats["web_name"].astype(str).str.contains(q, case=False, na=False)
                team_match = df_stats.get("team", pd.Series(dtype=str)).astype(str).str.contains(q, case=False, na=False)
                df_stats = df_stats[name_match | team_match]

            if show_uncertainty or full_names:
                st.caption("Note: the clean stats view ignores P10/P50/P90 and full-name mapping.")

            view = build_clean_playerstats_view(df_stats, per_90=per_90)
            st.dataframe(view, width="stretch", hide_index=True)

    # Tab: Player profile
    with tabs[4]:
        st.subheader("Player Profile")
        st.markdown("### Player directory & filters")

        teams = sorted(projections["team"].unique()) if (not projections.empty and "team" in projections.columns) else []
        positions = sorted(projections["position"].unique()) if (not projections.empty and "position" in projections.columns) else []
        price_opts = price_options_from_proj(projections)

        pf_team = st.selectbox("Team", options=["ALL"] + teams)
        pf_pos = st.selectbox("Position", options=["ALL"] + positions)
        pf_min = st.selectbox("Min price", options=price_opts, index=0)
        pf_max = st.selectbox("Max price", options=price_opts, index=len(price_opts) - 1)

        pf_df = projections.copy()
        if pf_team != "ALL":
            pf_df = pf_df[pf_df["team"] == pf_team]
        if pf_pos != "ALL":
            pf_df = pf_df[pf_df["position"] == pf_pos]
        if "price" in pf_df.columns:
            pf_df = pf_df[(pf_df["price"] >= float(pf_min)) & (pf_df["price"] <= float(pf_max))]

        display_cols = [c for c in ["web_name", "team", "position", "price", "proj_points"] if c in pf_df.columns]
        st.dataframe(pf_df[display_cols].head(200), width="stretch")

        if not pf_df.empty and "web_name" in pf_df.columns:
            sel = st.selectbox("Choose player for details", options=pf_df["web_name"].unique())
            if sel:
                p = pf_df[pf_df["web_name"] == sel].iloc[0]
                st.markdown(
                    f"### {p.get('web_name')} — {p.get('team','')} · {p.get('position','')} · £{p.get('price',0):.1f}m"
                )

                cols = st.columns([1, 1, 1])
                with cols[1]:
                    st.metric("Median Pred", f"{p.get('p50', p.get('proj_points',0)):.1f}")
                    st.metric("Minutes Prob", f"{int(round(p.get('minutes_prob',0)*100))}%")
                with cols[2]:
                    st.metric("Upside (P90)", f"{p.get('p90',0):.1f}")
                    st.metric("Downside (P10)", f"{p.get('p10',0):.1f}")

    # Tab: Transfer Planner (kept close to your original)
    with tabs[5]:
        st.subheader("Transfer Planner")
        st.caption("Squad rules enforced: £100.0m budget · Max 3 per club · 2 GK / 5 DEF / 5 MID / 3 FWD (15 total)")

        try:

            if "team_ids" not in st.session_state:
                st.session_state.team_ids = []
            if "tp_starters_ids" not in st.session_state:
                st.session_state.tp_starters_ids = []
            if "tp_bench_ids" not in st.session_state:
                st.session_state.tp_bench_ids = []

            if st.session_state.team_ids and not projections.empty:
                team_df = projections[projections["player_id"].astype(int).isin([int(x) for x in st.session_state.team_ids])].copy()
            else:
                team_df = projections.head(0).copy() if not projections.empty else pd.DataFrame()

            if not team_df.empty and len(team_df) >= 11 and (not st.session_state.tp_starters_ids):
                starters = _best_xi_ids(team_df)
                st.session_state.tp_starters_ids = starters
                st.session_state.tp_bench_ids = [
                    int(x)
                    for x in team_df["player_id"].astype(int).tolist()
                    if int(x) not in set(int(s) for s in starters)
                ]

            def _remove_from_planner(pid: int):
                st.session_state.team_ids = [x for x in st.session_state.team_ids if int(x) != int(pid)]
                st.session_state.tp_starters_ids = [x for x in st.session_state.tp_starters_ids if int(x) != int(pid)]
                st.session_state.tp_bench_ids = [x for x in st.session_state.tp_bench_ids if int(x) != int(pid)]

            left, right = st.columns([1.3, 1.0], gap="large")
            with left:
                starters_ids = [int(x) for x in st.session_state.get("tp_starters_ids", [])]
                bench_ids = [int(x) for x in st.session_state.get("tp_bench_ids", [])]

                if not team_df.empty and len(team_df) >= 11 and len(starters_ids) != 11:
                    starters_ids = _best_xi_ids(team_df)
                    bench_ids = [
                        int(x)
                        for x in team_df["player_id"].astype(int).tolist()
                        if int(x) not in set(int(s) for s in starters_ids)
                    ]

                starters_ids, bench_ids = _render_interactive_cards(
                    title="Transfer Planner lineup",
                    team_df=team_df,
                    starters_ids=starters_ids,
                    bench_ids=bench_ids,
                    state_prefix="tp",
                    allow_remove=True,
                    on_remove_player=_remove_from_planner,
                )
                st.session_state.tp_starters_ids = starters_ids
                st.session_state.tp_bench_ids = bench_ids

                starters_df = (
                    team_df[team_df["player_id"].astype(int).isin(starters_ids)].copy() if not team_df.empty else pd.DataFrame()
                )
                bench_df = (
                    team_df[team_df["player_id"].astype(int).isin(bench_ids)].copy() if not team_df.empty else pd.DataFrame()
                )
                st.markdown(_render_pitch_html(starters_df, bench_df), unsafe_allow_html=True)

                summary = _team_summary(team_df)
                st.metric("Team Cost", f"£{summary['cost']:.1f}m")
                st.write(
                    f"Positions: GK {summary['counts'].get('GK',0)}/2 · DEF {summary['counts'].get('DEF',0)}/5 · "
                    f"MID {summary['counts'].get('MID',0)}/5 · FWD {summary['counts'].get('FWD',0)}/3"
                )

            with right:
                st.markdown("**Player filters**")
                q = st.text_input("Search", value="")
                fpos = st.radio("Position", options=["ALL", "GK", "DEF", "MID", "FWD"], horizontal=True)
                price_opts = price_options_from_proj(projections)
                pmin, pmax = st.select_slider("Price", options=price_opts, value=(price_opts[0], price_opts[-1]))
                teams = (
                    sorted(projections["team"].dropna().unique().tolist())
                    if (not projections.empty and "team" in projections.columns)
                    else []
                )
                fteams = st.multiselect("Filter by club(s)", options=teams, default=[])

                cand = projections.copy()
                if q:
                    cand = cand[
                        cand["web_name"].astype(str).str.contains(q, case=False, na=False)
                        | cand.get("team", "").astype(str).str.contains(q, case=False, na=False)
                    ]
                if fpos != "ALL":
                    cand = cand[cand["position"] == fpos]
                cand = cand[(cand["price"] >= float(pmin)) & (cand["price"] <= float(pmax))]
                if fteams:
                    cand = cand[cand["team"].isin(fteams)]
                cand = cand.sort_values("proj_points", ascending=False)

                st.markdown("**Top players**")
                for _, r in cand.head(25).iterrows():
                    ok, reason = _can_add_player(team_df, r)
                    pid = int(r.get("player_id"))
                    name = r.get("web_name")
                    team = r.get("team")
                    pos = r.get("position")
                    price = float(r.get("price", 0) or 0)
                    proj = float(r.get("proj_points", 0) or 0)

                    c1, c2 = st.columns([4, 1])
                    with c1:
                        badge = r.get("badge")
                        badge_html = ""
                        try:
                            if badge and Path(str(badge)).exists():
                                badge_html = f"<img src='{badge}' style='width:26px;height:26px;vertical-align:-6px;margin-right:8px'/>"
                        except Exception:
                            badge_html = ""

                        st.markdown(
                            "<div style='background:#fff;border:1px solid #eee;border-radius:12px;padding:10px 12px;margin-bottom:10px'>"
                            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                            f"<div style='font-weight:800;font-size:15px'>{badge_html}{name} <span style='color:#999;font-weight:600;font-size:12px'>({team} · {pos})</span></div>"
                            f"<div style='color:#111;font-weight:800'>Pred {proj:.1f}</div>"
                            "</div>"
                            f"<div style='margin-top:6px;color:#444;font-size:12px'>£{price:.1f}m</div>"
                            "</div>",
                            unsafe_allow_html=True,
                        )

                    with c2:
                        if st.button("Add", key=f"add_{pid}", disabled=not ok):
                            st.session_state.team_ids = [*st.session_state.team_ids, pid]
                            st.rerun()
                        if not ok and reason:
                            st.caption(reason)

        except Exception:
            st.info("🔜 Advanced transfer planning tool coming soon!")


if __name__ == "__main__":
    main()

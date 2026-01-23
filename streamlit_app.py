"""
Streamlit app for FPL Team Builder and Projections.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable
from io import StringIO
import os
import base64
import json
import math
import re
import time
from collections import Counter

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


def get_with_retry(url: str, *, tries: int = 3, backoff: float = 1.5, timeout: int = 20) -> requests.Response:
    """requests.get with simple retry/backoff for FPL endpoints."""
    last_err: Exception | None = None
    for i in range(int(tries)):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            if i >= int(tries) - 1:
                raise
            time.sleep(backoff ** i)
    raise last_err  # pragma: no cover

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_player_photo_url(player_id: int) -> str:
    """Official Premier League CDN player photo (best-effort)."""
    return f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{int(player_id)}.png"

def get_team_badge_url(team_code: int) -> str:
    """Official Premier League CDN team badge (best-effort)."""
    return f"https://resources.premierleague.com/premierleague/badges/t{int(team_code)}.png"

def get_shirt_url(team_code: int) -> str:
    """FPL kit image (best-effort)."""
    return f"https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_{int(team_code)}-110.png"

# ============================================================================
# CONFIG
# ============================================================================

DATA = Path("data")
OUTPUTS = Path("outputs")
INSIGHTS_ROOT = Path("FPL-Core-Insights/data")
DEFAULT_INSIGHTS_SEASON = "2025-2026"
INSIGHTS_PLAYERSTATS = INSIGHTS_ROOT / DEFAULT_INSIGHTS_SEASON / "playerstats.csv"

PROJ_JSON = DATA / "projections.json"
PROJ_CSV_FALLBACK = DATA / "projections.csv"
PROJ_CSV = OUTPUTS / "projections.csv"
PROJ_CSV_INTERNAL = OUTPUTS / "projections_internal.csv"
FIXTURES_JSON = DATA / "fixtures.json"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
LIVEFPL_PRICES_URL = "https://www.livefpl.net/prices"

ENTRY_ID = 1093603
GW = 20
BUDGET = 100.0
MAX_PER_CLUB = 3
POS_CAP = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

ALLOWED_FORMATIONS = {
    "3-4-3": (1, 3, 4, 3),
    "3-5-2": (1, 3, 5, 2),
    "4-5-1": (1, 4, 5, 1),
    "4-4-2": (1, 4, 4, 2),
    "4-3-3": (1, 4, 3, 3),
    "5-3-2": (1, 5, 3, 2),
    "5-4-1": (1, 5, 4, 1),
}

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_json(path_or_url: str | Path) -> pd.DataFrame:
    """Load JSON from file or URL."""
    p = str(path_or_url)
    if p.startswith("http"):
        r = get_with_retry(p, timeout=20)
        payload = r.json()
        if isinstance(payload, dict) and "fixtures" in payload:
            return pd.DataFrame(payload["fixtures"])
        if isinstance(payload, dict) and "teams" in payload:
            return pd.DataFrame(payload["teams"])
        return pd.DataFrame(payload)
    
    try:
        with open(p, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict) and "fixtures" in payload:
            return pd.DataFrame(payload["fixtures"])
        if isinstance(payload, dict) and "teams" in payload:
            return pd.DataFrame(payload["teams"])
        return pd.DataFrame(payload)
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV."""
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_team_lookup() -> dict[int, dict[str, str]]:
    """Load team lookup from FPL bootstrap-static."""
    out: dict[int, dict[str, str]] = {}
    try:
        payload = fetch_bootstrap_json()
        teams = payload.get("teams", [])
        for t in teams:
            try:
                tid = int(t.get("id"))
                out[tid] = {
                    "short_name": str(t.get("short_name") or tid),
                    "team_name": str(t.get("name") or t.get("short_name") or tid),
                    "team_code": int(t.get("code", 0))
                }
            except Exception:
                continue
    except Exception:
        pass
    return out

def team_short_name(team_id: Any) -> str:
    """Get team short name."""
    try:
        tid = int(team_id)
    except Exception:
        return str(team_id) if team_id is not None else ""
    
    t = load_team_lookup().get(tid)
    return (t or {}).get("short_name") or str(tid)

def team_full_name(team_id: Any) -> str:
    """Get team full name."""
    try:
        tid = int(team_id)
    except Exception:
        return str(team_id) if team_id is not None else ""
    
    t = load_team_lookup().get(tid)
    return (t or {}).get("team_name") or (t or {}).get("short_name") or str(tid)

def normalize_projections(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column names for projections."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    
    # player_id
    id_col = next((c for c in df.columns if str(c).lower() in ["player_id", "element", "id"]), None)
    if id_col:
        df["player_id"] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    else:
        # Some projection exports don't include the FPL element id.
        # Best-effort: infer from FPL bootstrap-static by matching (name + team short).
        df["player_id"] = pd.Series([pd.NA] * len(df), dtype="Int64")

        try:
            els = load_bootstrap_elements()
            if not els.empty and "id" in els.columns and "team" in els.columns:
                els2 = els.copy()
                els2["team_short"] = els2["team"].map(team_short_name).astype(str).str.upper().str.strip()
                els2["_fn"] = els2.get("first_name", "").astype(str).str.lower().str.strip()
                els2["_sn"] = els2.get("second_name", "").astype(str).str.lower().str.strip()
                els2["_wn"] = els2.get("web_name", "").astype(str).str.lower().str.strip()

                proj = df.copy()
                proj["_team"] = proj.get("team", "").astype(str).str.upper().str.strip()
                proj["_fn"] = proj.get("first_name", "").astype(str).str.lower().str.strip()
                proj["_sn"] = proj.get("second_name", "").astype(str).str.lower().str.strip()
                proj["_wn"] = proj.get("web_name", "").astype(str).str.lower().str.strip()

                # Pass 1: (first_name, second_name, team_short)
                key_map_full = {
                    (r["_fn"], r["_sn"], r["team_short"]): int(r["id"])
                    for _, r in els2.iterrows()
                    if pd.notna(r.get("id"))
                }
                inferred = proj.apply(
                    lambda r: key_map_full.get((r.get("_fn"), r.get("_sn"), r.get("_team"))),
                    axis=1,
                )

                # Pass 2: (web_name, team_short) for any still-missing
                key_map_wn = {
                    (r["_wn"], r["team_short"]): int(r["id"])
                    for _, r in els2.iterrows()
                    if pd.notna(r.get("id"))
                }
                inferred2 = proj.apply(
                    lambda r: key_map_wn.get((r.get("_wn"), r.get("_team"))),
                    axis=1,
                )

                # Combine
                df["player_id"] = pd.to_numeric(inferred.fillna(inferred2), errors="coerce").astype("Int64")
        except Exception:
            # Keep player_id as NA if inference fails.
            pass
    
    # web_name
    name_col = next((c for c in df.columns if c.lower() in ["web_name", "name", "player"]), None)
    if name_col:
        df["web_name"] = df[name_col]
    
    # team_id
    team_id_col = next((c for c in df.columns if c.lower() in ["team_id", "teamid"]), None)
    if team_id_col:
        df["team_id"] = pd.to_numeric(df[team_id_col], errors="coerce").astype("Int64")
    
    # team
    team_col = next((c for c in df.columns if c.lower() in ["team", "team_name", "club"]), None)
    if team_col:
        df["team"] = df[team_col]
    
    # position
    pos_col = next((c for c in df.columns if c.lower() in ["position", "pos"]), None)
    if pos_col:
        df["position"] = df[pos_col]
    
    POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    if "position" not in df.columns and "element_type" in df.columns:
        df["position"] = pd.to_numeric(df["element_type"], errors="coerce").map(POS_MAP)

    # Normalize common string position names
    if "position" in df.columns:
        pos_s = df["position"].astype(str).str.strip().str.lower()
        df.loc[pos_s.str.contains("keeper"), "position"] = "GK"
        df.loc[pos_s.str.contains("goal"), "position"] = "GK"
        df.loc[pos_s.str.contains("def"), "position"] = "DEF"
        df.loc[pos_s.str.contains("mid"), "position"] = "MID"
        df.loc[pos_s.str.contains("for"), "position"] = "FWD"
    
    # price
    price_col = None
    if "now_cost" in df.columns:
        price_col = "now_cost"
    else:
        price_col = next((c for c in df.columns if str(c).lower() in ["price", "value"]), None)
    
    if price_col:
        price = pd.to_numeric(df[price_col], errors="coerce")
        if str(price_col).lower() == "now_cost":
            # Some sources provide now_cost in tenths (e.g. 72 -> 7.2), others already in £m (e.g. 7.2).
            # Heuristic: if values look like tenths (mostly integers and max > 25), divide by 10.
            p = price.dropna()
            if not p.empty:
                mostly_int = float(((p % 1) == 0).mean()) if len(p) else 0.0
                if (p.max() > 25 and mostly_int >= 0.8) or p.max() > 250:
                    price = price / 10.0
        elif price.max() > 1000:
            price = price / 10.0
        df["price"] = price.fillna(0.0)
    else:
        df["price"] = 0.0
    
    # proj_points
    proj_col = next((c for c in df.columns if c.lower() in ["proj_points", "projected_points", "proj"]), None)
    if proj_col:
        df["proj_points"] = pd.to_numeric(df[proj_col], errors="coerce").fillna(0.0)
    elif "proj_points_next_6" in df.columns:
        df["proj_points"] = pd.to_numeric(df["proj_points_next_6"], errors="coerce").fillna(0.0)
    elif "proj_points_next_5" in df.columns:
        df["proj_points"] = pd.to_numeric(df["proj_points_next_5"], errors="coerce").fillna(0.0)
    else:
        gw_cols = [c for c in df.columns if re.match(r"^GW\d+_proj_points$", str(c))]
        if gw_cols:
            df["proj_points"] = (
                pd.concat([pd.to_numeric(df[c], errors="coerce").fillna(0.0) for c in gw_cols], axis=1)
                .sum(axis=1)
                .astype(float)
            )
        else:
            df["proj_points"] = 0.0
    
    # team_code
    if "team_code" not in df.columns and "code" in df.columns:
        df["team_code"] = pd.to_numeric(df["code"], errors="coerce").astype("Int64")
    
    # Enrich team names if needed
    if "team" in df.columns and "team_id" in df.columns:
        team_lookup = load_team_lookup()
        def team_short(team_id, team_raw):
            if team_id is not None and team_id in team_lookup:
                return team_lookup[team_id].get("short_name") or str(team_id)
            if pd.notna(team_raw):
                return str(team_raw)
            return ""
        
        df["team"] = df.apply(lambda r: team_short(int(r["team_id"]) if pd.notna(r["team_id"]) else None, r.get("team")), axis=1)
    
    return df

def price_options_from_proj(df: pd.DataFrame, step: float = 0.5, default_min: float = 3.5, default_max: float = 15.0):
    """Generate price filter options."""
    if df is None or df.empty or "price" not in df.columns:
        lo, hi = default_min, default_max
    else:
        lo = float(df["price"].min())
        hi = float(df["price"].max())
        lo = max(default_min, int(lo * 2) / 2.0)
        hi = max(lo + step, int(hi * 2 + 1) / 2.0)
    
    opts = []
    v = lo
    while v <= hi + 1e-9:
        opts.append(round(v, 1))
        v += step
    return opts

@st.cache_data
def load_bootstrap_elements() -> pd.DataFrame:
    """Load FPL bootstrap-static elements."""
    try:
        payload = fetch_bootstrap_json()
        els = payload.get("elements", [])
        df = pd.DataFrame(els)
        if not df.empty and "id" in df.columns:
            df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60 * 30)
def fetch_bootstrap_json() -> dict[str, Any]:
    """Fetch FPL bootstrap-static (cached with TTL)."""
    return get_with_retry("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=20).json()


@st.cache_data(ttl=60 * 30)
def fetch_fixtures_json() -> list[dict[str, Any]]:
    """Fetch FPL fixtures (cached with TTL)."""
    payload = get_with_retry(FIXTURES_URL, timeout=20).json()
    return payload if isinstance(payload, list) else []


def _normalize_key(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).lower().strip()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _normalize_team_key(v: Any) -> str:
    """Normalize team names to a canonical key (handles common aliases)."""
    k = _normalize_key(v)
    if not k:
        return ""

    alias = {
        # Manchester clubs
        "manutd": "manchesterunited",
        "manunited": "manchesterunited",
        "mufc": "manchesterunited",
        "manutd": "manchesterunited",
        "mancity": "manchestercity",
        "mcfc": "manchestercity",
        # Spurs / Wolves / Forest
        "spurs": "tottenhamhotspur",
        "tottenham": "tottenhamhotspur",
        "wolves": "wolverhamptonwanderers",
        "wwfc": "wolverhamptonwanderers",
        "forest": "nottinghamforest",
        "nffc": "nottinghamforest",
        # Other common
        "westham": "westhamunited",
        "newcastle": "newcastleunited",
        "brighton": "brightonhovealbion",
        "leicester": "leicestercity",
        "ipswich": "ipswichtown",
        "sheffutd": "sheffieldunited",
        "sheffieldutd": "sheffieldunited",
        # strip suffixes
        "afcbournemouth": "bournemouth",
    }
    if k in alias:
        return alias[k]

    # remove common suffixes
    for suf in ("fc", "afc"):
        if k.endswith(suf) and len(k) > len(suf) + 2:
            k = k[: -len(suf)]
    return alias.get(k, k)


@st.cache_data(ttl=60 * 60)
def team_key_map() -> dict[str, str]:
    """Map various team tokens (short/full/aliases) to a canonical key."""
    m: dict[str, str] = {}
    lookup = load_team_lookup()
    for _, t in lookup.items():
        short = str(t.get("short_name") or "")
        full = str(t.get("team_name") or "")
        canonical = _normalize_team_key(full or short)
        if short:
            m[_normalize_team_key(short)] = canonical
            m[_normalize_key(short)] = canonical
        if full:
            m[_normalize_team_key(full)] = canonical
            m[_normalize_key(full)] = canonical
    return m


@st.cache_data(ttl=60 * 30)
def fetch_livefpl_prices() -> pd.DataFrame:
    """Fetch and parse LiveFPL price predictor table.

    Returns a dataframe with (web_name, team_livefpl, prediction_pct, prediction_eta).
    Best-effort: if parsing fails, returns empty dataframe.
    """
    try:
        html = requests.get(
            LIVEFPL_PRICES_URL,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        ).text
    except Exception:
        return pd.DataFrame()

    try:
        tables = pd.read_html(StringIO(html))
    except Exception:
        return pd.DataFrame()

    if not tables:
        return pd.DataFrame()

    main: pd.DataFrame | None = None
    for t in tables:
        try:
            cols = [str(c) for c in t.columns]
        except Exception:
            continue
        if "Progress Now" in cols and any("Prediction" in c for c in cols):
            main = t
            break

    if main is None or main.empty or "Player" not in main.columns:
        return pd.DataFrame()

    df = main.copy()

    # Team appears in an unnamed column on LiveFPL (currently Unnamed: 2)
    team_col = None
    for c in df.columns:
        if str(c).lower().strip() in {"team", "unnamed: 2"}:
            team_col = c
            break
    if team_col is None:
        # fallback: pick any unnamed column
        unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed")]
        team_col = unnamed[-1] if unnamed else None

    pred_col = next((c for c in df.columns if "Prediction" in str(c)), None)
    if pred_col is None:
        return pd.DataFrame()

    player_raw = df["Player"].astype(str).str.replace("Â£", "£", regex=False).str.strip()

    # Example: "Gakpo  MID £7.3"
    m = player_raw.str.extract(r"^(?P<name>.*?)\s{2,}(?P<pos>GK|DEF|MID|FW)\s*£(?P<price>\d+(?:\.\d+)?)")
    web_name = m["name"].astype(str).str.strip()
    pos = m["pos"].astype(str).str.strip().replace({"FW": "FWD"})
    price = pd.to_numeric(m["price"], errors="coerce")

    team_livefpl = df[team_col].astype(str).str.strip() if team_col in df.columns else ""
    pred_str = df[pred_col].astype(str).str.strip()

    pred_pct = pd.to_numeric(pred_str.str.extract(r"([-+]?\d+(?:\.\d+)?)")[0], errors="coerce")
    pred_eta = pred_str.str.replace(r"^[-+]?\d+(?:\.\d+)?%\s*", "", regex=True).str.strip()

    out = pd.DataFrame(
        {
            "web_name": web_name,
            "team_livefpl": team_livefpl,
            "pos_livefpl": pos,
            "price_livefpl": price,
            "prediction_pct": pred_pct,
            "prediction_eta": pred_eta,
        }
    )
    out = out.dropna(subset=["web_name"]).copy()
    out["name_key"] = out["web_name"].map(_normalize_key)
    out["team_key"] = out["team_livefpl"].map(_normalize_team_key)
    return out


def _bootstrap_identity(player_id: int) -> dict[str, Any]:
    """Return identifying info for a player_id from FPL bootstrap-static."""
    try:
        pid = int(player_id)
    except Exception:
        return {}

    els = load_bootstrap_elements()
    if els.empty or "id" not in els.columns:
        return {}

    try:
        row = els[els["id"].astype("Int64") == pid].iloc[0]
    except Exception:
        return {}

    name = str(row.get("web_name", "") or "").strip()
    first = str(row.get("first_name", "") or "").strip()
    second = str(row.get("second_name", "") or "").strip()
    try:
        team_id = int(row.get("team"))
    except Exception:
        team_id = None

    team_lookup = load_team_lookup()
    team_short = team_short_name(team_id) if team_id is not None else ""
    team_full = (team_lookup.get(int(team_id), {}) if team_id is not None else {}).get("team_name", "") if team_id is not None else ""

    element_type = row.get("element_type")
    try:
        element_type = int(element_type)
    except Exception:
        element_type = None
    pos = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}.get(element_type, "")

    try:
        price = float(row.get("now_cost", 0)) / 10.0
    except Exception:
        price = None

    tk_map = team_key_map()
    team_key = tk_map.get(_normalize_team_key(team_full), _normalize_team_key(team_full or team_short))

    return {
        "player_id": pid,
        "web_name": name,
        "first_name": first,
        "second_name": second,
        "team_short": team_short,
        "team_full": team_full,
        "team_key": team_key,
        "position": pos,
        "price": price,
    }


def _best_livefpl_row_for_player(player_id: int) -> dict[str, Any] | None:
    """Choose best LiveFPL row for an FPL player_id (handles ambiguous names)."""
    ident = _bootstrap_identity(player_id)
    if not ident:
        return None

    live = fetch_livefpl_prices()
    if live is None or live.empty:
        return None

    # Candidate name keys (LiveFPL uses web_name-like values most of the time)
    name_keys = [
        _normalize_key(ident.get("web_name")),
        _normalize_key(ident.get("second_name")),
        _normalize_key(f"{ident.get('first_name','')} {ident.get('second_name','')}")
    ]
    name_keys = [k for k in name_keys if k]

    cand = live[live["name_key"].isin(name_keys)].copy()
    if cand.empty and name_keys:
        # Fuzzy fallback: compute similarity against full table (table is ~600 rows)
        target = name_keys[0]
        # lightweight similarity: common prefix + length proximity
        nk = live["name_key"].astype(str)
        scores = nk.apply(lambda s: (len(os.path.commonprefix([s, target])) / max(len(target), 1)) if s else 0.0)
        best_idx = scores.idxmax() if not scores.empty else None
        if best_idx is not None and float(scores.loc[best_idx]) >= 0.8:
            cand = live.loc[[best_idx]].copy()

    if cand.empty:
        return None

    team_key = str(ident.get("team_key") or "")
    pos = str(ident.get("position") or "")
    price = ident.get("price")

    def score_row(r: pd.Series) -> float:
        s = 0.0
        if str(r.get("name_key")) in name_keys:
            s += 10.0
        if team_key and str(r.get("team_key")) == team_key:
            s += 6.0
        if pos and str(r.get("pos_livefpl")) == pos:
            s += 3.0
        try:
            lp = float(r.get("price_livefpl"))
            if price is not None:
                d = abs(float(price) - lp)
                if d <= 0.05:
                    s += 2.0
                elif d <= 0.2:
                    s += 1.0
        except Exception:
            pass
        return s

    cand["_score"] = cand.apply(score_row, axis=1)
    best = cand.sort_values(["_score"], ascending=False).iloc[0].to_dict()
    return best


@st.cache_data(ttl=60 * 30)
def compute_livefpl_trend_map(player_ids: tuple[int, ...]) -> dict[int, str]:
    """Compute a player_id -> trend label map (cached) for a set of player ids."""
    out: dict[int, str] = {}
    for pid in player_ids:
        try:
            pid_i = int(pid)
        except Exception:
            continue

        r = _best_livefpl_row_for_player(pid_i)
        if not r:
            out[pid_i] = "n/a"
            continue

        try:
            pct = float(r.get("prediction_pct"))
        except Exception:
            out[pid_i] = "n/a"
            continue

        eta = str(r.get("prediction_eta", "") or "").strip()
        if pct >= 100:
            arrow = "↑"
        elif pct <= -100:
            arrow = "↓"
        else:
            arrow = "→"
        out[pid_i] = f"{arrow} {pct:.0f}% {eta}".strip()

    return out


def estimate_price_change(player_id: int, current_price: float, proj_points: float, selected_by_pct: float) -> str:
    """Best-effort price trend label using LiveFPL predictor (% to target).

    Returns a short label: "↑ 103%", "↓ -101%", or "→ 42%" (plus an ETA when available).
    """
    _ = (current_price, proj_points, selected_by_pct)  # signature parity; values optional

    try:
        pid = int(player_id)
    except Exception:
        return "n/a"

    trend_map = compute_livefpl_trend_map((pid,))
    return trend_map.get(pid, "n/a")


@st.cache_data(ttl=3600)
def fetch_player_history(player_id: int) -> pd.DataFrame:
    """Fetch player's GW-by-GW history from FPL API."""
    url = f"https://fantasy.premierleague.com/api/element-summary/{int(player_id)}/"
    try:
        r = get_with_retry(url, timeout=15)
        history = r.json().get("history", [])
        df = pd.DataFrame(history)
        df["player_id"] = int(player_id)
        return df
    except Exception:
        return pd.DataFrame()


def create_points_chart(df: pd.DataFrame, name: str) -> go.Figure:
    """Create interactive points history chart."""
    if df is None or df.empty or "round" not in df.columns:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["round"],
            y=df.get("total_points", 0),
            name="Points",
            marker_color="#00ff87",
        )
    )

    pts = pd.to_numeric(df.get("total_points", 0), errors="coerce")
    if len(df) >= 5 and pts.notna().any():
        rolling = pts.rolling(5, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=df["round"],
                y=rolling,
                name="5-GW avg",
                line=dict(color="#ff6b6b", width=3),
            )
        )

    fig.update_layout(
        title=f"{name} - Points History",
        template="plotly_dark",
        plot_bgcolor="#37003c",
        paper_bgcolor="#37003c",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


@st.cache_data
def load_role_mae() -> dict[str, float]:
    """Load per-role MAE from artifacts diagnostics (used for uncertainty bands)."""
    p = Path("artifacts/diagnostics/per_role_ALL.csv")
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
        if df.empty or "role" not in df.columns or "mae" not in df.columns:
            return {}
        out: dict[str, float] = {}
        for _, r in df.iterrows():
            role = str(r.get("role", "")).strip()
            try:
                mae = float(r.get("mae"))
            except Exception:
                continue
            if role:
                out[role] = mae
        return out
    except Exception:
        return {}


def attach_uncertainty(df: pd.DataFrame) -> pd.DataFrame:
    """Attach sigma and (p25,p75) band using per-role MAE (Normal approx)."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    role_mae = load_role_mae()

    def _role_key(r: pd.Series) -> str:
        role = str(r.get("role", "") or "").strip()
        if role:
            return role
        pos = str(r.get("position", "") or "").strip()
        if pos == "MID":
            return "MID_AM"
        return pos

    # MAE -> sigma: MAE = sigma * sqrt(2/pi)
    k = math.sqrt(math.pi / 2.0)
    maes = out.apply(lambda r: role_mae.get(_role_key(r), role_mae.get("MID_AM" if str(r.get("position")) == "MID" else str(r.get("position")), role_mae.get("DEF", 3.37))), axis=1)
    out["sigma"] = pd.to_numeric(maes, errors="coerce").fillna(3.37) * k

    mu = pd.to_numeric(out.get("proj_points", 0), errors="coerce").fillna(0.0)
    # Normal quantile for 25/75 is +/- 0.674*sigma
    half_iqr = 0.67448975 * pd.to_numeric(out["sigma"], errors="coerce").fillna(0.0)
    out["proj_p25"] = (mu - half_iqr).clip(lower=0.0)
    out["proj_p75"] = (mu + half_iqr)
    return out


def attach_reason_codes(df: pd.DataFrame, *, fixture_index: dict[int, dict[int, list[dict[str, Any]]]], gw: int) -> pd.DataFrame:
    """Add short, human-readable reason codes + next fixtures string."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()

    def _fixtures_str(team_id: Any) -> str:
        try:
            tid = int(team_id)
        except Exception:
            return ""
        parts: list[str] = []
        for g in range(int(gw), int(gw) + 3):
            fx_list = (fixture_index.get(tid, {}) or {}).get(int(g), [])
            if not fx_list:
                continue
            fx = fx_list[0]
            opp = team_short_name(fx.get("opp_id"))
            ha = fx.get("ha")
            parts.append(f"{opp}({ha})")
        return " | ".join(parts)

    reasons: list[str] = []
    fixtures_next: list[str] = []
    for _, r in out.iterrows():
        t: list[str] = []
        xgi90 = pd.to_numeric(r.get("expected_goal_involvements_per_90"), errors="coerce")
        thr = pd.to_numeric(r.get("threat"), errors="coerce")
        cre = pd.to_numeric(r.get("creativity"), errors="coerce")
        defcon = pd.to_numeric(r.get("defcon_points"), errors="coerce")
        xgc90 = pd.to_numeric(r.get("expected_goals_conceded_per_90"), errors="coerce")

        if pd.notna(xgi90) and float(xgi90) >= 0.45:
            t.append("High xGI/90")
        if pd.notna(thr) and float(thr) >= 300:
            t.append("High Threat")
        if pd.notna(cre) and float(cre) >= 250:
            t.append("Chance creation")
        if pd.notna(defcon) and float(defcon) >= 10:
            t.append("Strong DEFCON")
        if pd.notna(xgc90) and float(xgc90) <= 1.1 and str(r.get("position")) == "DEF":
            t.append("Low xGC/90")

        role = str(r.get("role", "") or "")
        if role in ["MID_DM"] and (pd.isna(xgi90) or float(xgi90 or 0) < 0.25):
            t.append("DM role—low xGI")

        reasons.append(", ".join(t[:3]) if t else "Balanced profile")
        fixtures_next.append(_fixtures_str(r.get("team_id")))

    out["why"] = reasons
    out["fixtures_next3"] = fixtures_next
    return out

def enrich_with_fpl_stats(projections: pd.DataFrame) -> pd.DataFrame:
    """Merge FPL bootstrap stats into projections."""
    if projections is None or projections.empty:
        return projections
    
    els = load_bootstrap_elements()
    if els.empty or "id" not in els.columns or "player_id" not in projections.columns:
        return projections
    
    keep = [
        "id", "web_name", "element_type", "team",
        "now_cost", "selected_by_percent",
        "total_points", "form",
        "expected_goals_per_90", "expected_assists_per_90", "expected_goal_involvements_per_90",
        "influence", "creativity", "threat", "ict_index",
        "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded"
    ]
    keep = [c for c in keep if c in els.columns]
    
    stats = els[keep].copy().rename(columns={"id": "player_id", "team": "fpl_team_id"})
    
    for c in stats.columns:
        if c in ["player_id"]:
            continue
        stats[c] = pd.to_numeric(stats[c], errors="coerce")
    
    out = projections.merge(stats, on="player_id", how="left", suffixes=("", "_fpl"))
    
    # Fill missing data
    if "web_name" not in out.columns and "web_name_fpl" in out.columns:
        out["web_name"] = out["web_name_fpl"]
    
    if "team_id" not in out.columns and "fpl_team_id" in out.columns:
        out["team_id"] = pd.to_numeric(out["fpl_team_id"], errors="coerce").astype("Int64")
    
    # Team codes
    team_lookup = load_team_lookup()
    if "team_code" not in out.columns and "team_id" in out.columns:
        out["team_code"] = out["team_id"].map(lambda tid: team_lookup.get(int(tid), {}).get("team_code", 0) if pd.notna(tid) else 0)
    
    return out


def enrich_with_insights_playerstats(projections: pd.DataFrame) -> pd.DataFrame:
    """Merge advanced stats from FPL-Core-Insights playerstats.csv (latest GW per player)."""
    if projections is None or projections.empty or "player_id" not in projections.columns:
        return projections

    insights_path = INSIGHTS_PLAYERSTATS
    if not insights_path.exists():
        return projections

    try:
        ins = pd.read_csv(insights_path)
    except Exception:
        return projections

    if ins.empty or "id" not in ins.columns:
        return projections

    ins = ins.copy()
    ins["id"] = pd.to_numeric(ins["id"], errors="coerce").astype("Int64")
    if "gw" in ins.columns:
        ins["gw"] = pd.to_numeric(ins["gw"], errors="coerce")
        ins = ins.sort_values(["id", "gw"], ascending=[True, True])
        ins_latest = ins.dropna(subset=["id"]).groupby("id", as_index=False).tail(1)
    else:
        ins_latest = ins.dropna(subset=["id"]).drop_duplicates(subset=["id"], keep="last")

    adv_cols = [
        "id",
        "expected_goals_conceded_per_90",
        "expected_goals_conceded",
        "defensive_contribution",
        "defensive_contribution_per_90",
        "tackles",
        "clearances_blocks_interceptions",
        "recoveries",
    ]
    adv_cols = [c for c in adv_cols if c in ins_latest.columns]
    if "id" not in adv_cols or len(adv_cols) == 1:
        return projections

    adv = ins_latest[adv_cols].copy().rename(columns={"id": "player_id"})
    adv["player_id"] = pd.to_numeric(adv["player_id"], errors="coerce").astype("Int64")

    out = projections.copy()
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
    out = out.merge(adv, on="player_id", how="left", suffixes=("", "_insights"))

    # Fill missing values from insights (do not overwrite existing non-null values)
    # Note: some projection exports include placeholder blanks/zeros; treat those as missing
    _treat_zero_as_missing = {
        "expected_goals_conceded_per_90",
        "expected_goals_conceded",
    }
    for col in adv_cols:
        if col == "id":
            continue
        insights_col = f"{col}_insights"
        if insights_col not in out.columns:
            continue
        if col not in out.columns:
            out[col] = out[insights_col]
        else:
            base = out[col]
            insv = out[insights_col]

            # For numeric columns: coerce so that blanks become NaN
            base_num = pd.to_numeric(base, errors="coerce")
            ins_num = pd.to_numeric(insv, errors="coerce")

            if col in _treat_zero_as_missing:
                mask = base_num.isna() | (base_num == 0)
                out[col] = base_num.where(~mask, ins_num)
            else:
                mask = base_num.isna()
                out[col] = base_num.where(~mask, ins_num)

    drop_cols = [c for c in out.columns if c.endswith("_insights")]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    # Convenience aliases used by some tables/UI
    if "defcon_points" not in out.columns and "defensive_contribution" in out.columns:
        out["defcon_points"] = out["defensive_contribution"]
    if "cbit" not in out.columns and "clearances_blocks_interceptions" in out.columns:
        out["cbit"] = out["clearances_blocks_interceptions"]
    return out

def load_projections() -> pd.DataFrame:
    """Load projections from CSV or JSON."""
    # Prefer the richer internal CSV when present (includes player_id/team_code)
    if PROJ_CSV_INTERNAL.exists():
        df = load_csv(PROJ_CSV_INTERNAL)
    elif PROJ_CSV.exists():
        df = load_csv(PROJ_CSV)
    elif PROJ_JSON.exists():
        df = load_json(PROJ_JSON)
    elif PROJ_CSV_FALLBACK.exists():
        df = load_csv(PROJ_CSV_FALLBACK)
    else:
        return pd.DataFrame()
    
    df = normalize_projections(df)
    df = enrich_with_fpl_stats(df)
    df = enrich_with_insights_playerstats(df)
    return df

@st.cache_data(ttl=120)
def get_my_team(entry_id: int, gw: int) -> pd.DataFrame:
    """Fetch my team from FPL API."""
    url = f"https://fantasy.premierleague.com/api/entry/{entry_id}/event/{gw}/picks/"
    try:
        r = get_with_retry(url, timeout=15)
        picks = r.json().get("picks", [])
        return pd.DataFrame(picks)
    except Exception:
        return pd.DataFrame(columns=["element", "position", "is_captain"])

def build_fixture_index(fixtures_df: pd.DataFrame) -> dict[int, dict[int, list[dict[str, Any]]]]:
    """Build fixture index by team and GW."""
    out: dict[int, dict[int, list[dict[str, Any]]]] = {}
    
    if fixtures_df is None or fixtures_df.empty or "event" not in fixtures_df.columns:
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
        
        def push(team_id: int, opp_id: int, ha: str, fdr_val: Any):
            try:
                fdr_int = int(fdr_val) if pd.notna(fdr_val) else 3
            except Exception:
                fdr_int = 3
            
            team_map = out.setdefault(team_id, {})
            team_map.setdefault(gw, []).append({"opp_id": opp_id, "ha": ha, "fdr": fdr_int})
        
        push(th, ta, "H", r.get("team_h_difficulty"))
        push(ta, th, "A", r.get("team_a_difficulty"))
    
    return out

def team_summary(team_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize team cost and composition."""
    if team_df is None or team_df.empty:
        return {"cost": 0.0, "counts": Counter(), "clubs": Counter()}
    
    cost = float(team_df.get("price", 0).fillna(0).sum()) if "price" in team_df.columns else 0.0
    counts = Counter(team_df.get("position", []).tolist())
    clubs = Counter(team_df.get("team", []).tolist())
    
    return {"cost": cost, "counts": counts, "clubs": clubs}

def derive_formation(starters_df: pd.DataFrame) -> str | None:
    """Derive formation from starting XI."""
    if starters_df is None or starters_df.empty or "position" not in starters_df.columns:
        return None
    
    counts = Counter(starters_df["position"].tolist())
    gk = int(counts.get("GK", 0))
    d = int(counts.get("DEF", 0))
    m = int(counts.get("MID", 0))
    f = int(counts.get("FWD", 0))
    
    if gk != 1 or (gk + d + m + f) != 11:
        return None
    
    for formation, (gkn, defn, midn, fwdn) in ALLOWED_FORMATIONS.items():
        if (gk, d, m, f) == (gkn, defn, midn, fwdn):
            return formation
    
    return None

def validate_starting_xi(team_df: pd.DataFrame, starter_ids: list[int]) -> tuple[bool, str, str | None]:
    """Validate starting XI formation."""
    if team_df is None or team_df.empty:
        return False, "No squad loaded", None
    
    if len(starter_ids) != 11:
        return False, "Starting XI must have 11 players", None
    
    starters = team_df[team_df["player_id"].astype(int).isin([int(x) for x in starter_ids])].copy()
    if starters.empty or len(starters) != 11:
        return False, "Starting XI selection contains unknown players", None
    
    if "position" not in starters.columns:
        return True, "", None
    
    formation = derive_formation(starters)
    if not formation:
        return False, f"Invalid formation. Use one of: {', '.join(ALLOWED_FORMATIONS.keys())}", None
    
    return True, "", formation


def can_remove_from_squad(team_df: pd.DataFrame, player_id: int) -> tuple[bool, str]:
    """True if removing this player still allows forming a valid XI from the remaining squad."""
    if team_df is None or team_df.empty or "player_id" not in team_df.columns:
        return False, "No squad"

    try:
        pid = int(player_id)
    except Exception:
        return False, "Invalid player"

    remaining = team_df[team_df["player_id"].astype(int) != pid].copy()
    if len(remaining) < 11:
        return False, "Need at least 11 players"

    try:
        starters = best_xi_ids(remaining)
    except Exception:
        starters = []

    if not starters or len(starters) != 11:
        return False, "Would break valid formation"

    ok, msg, _ = validate_starting_xi(remaining, starters)
    if not ok:
        return False, msg or "Would break valid formation"

    return True, ""


def best_valid_swap_in(team_df: pd.DataFrame, *, starter_out: int, starters_ids: list[int], bench_ids: list[int]) -> int | None:
    """Pick a bench player to swap in for starter_out that keeps formation valid.

    Returns the best bench player id by projected points, or None if no valid swap exists.
    """
    if team_df is None or team_df.empty:
        return None
    starters_set = [int(x) for x in starters_ids]
    bench_set = [int(x) for x in bench_ids]

    if int(starter_out) not in starters_set:
        return None

    best_id: int | None = None
    best_score = -1e18

    for b in bench_set:
        new_starters = [x for x in starters_set if int(x) != int(starter_out)] + [int(b)]
        ok, _, _ = validate_starting_xi(team_df, new_starters)
        if not ok:
            continue

        try:
            score = float(team_df[team_df["player_id"].astype(int).isin(new_starters)]["proj_points"].sum())
        except Exception:
            score = 0.0

        if score > best_score:
            best_score = score
            best_id = int(b)

    return best_id

def can_add_player(team_df: pd.DataFrame, p: pd.Series) -> tuple[bool, str]:
    """Check if player can be added to squad."""
    if p is None or p.empty:
        return False, "Invalid player"
    
    if "player_id" in p and team_df is not None and not team_df.empty and "player_id" in team_df.columns:
        if int(p["player_id"]) in set(team_df["player_id"].astype(int).tolist()):
            return False, "Already added"
    
    summary = team_summary(team_df)
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

def best_xi_ids(team_df: pd.DataFrame) -> list[int]:
    """Auto-select best starting XI."""
    if team_df is None or team_df.empty:
        return []
    
    t = team_df.copy()
    t["score"] = pd.to_numeric(t.get("proj_points", 0), errors="coerce").fillna(0.0)
    
    if "position" not in t.columns:
        return t.sort_values("score", ascending=False)["player_id"].astype(int).head(11).tolist()
    
    best: tuple[float, list[int]] = (-1e9, [])
    
    for _, (gkn, defn, midn, fwdn) in ALLOWED_FORMATIONS.items():
        ids: list[int] = []
        
        for pos, n in [("GK", gkn), ("DEF", defn), ("MID", midn), ("FWD", fwdn)]:
            pool = t[t["position"] == pos].sort_values("score", ascending=False).head(n)
            if len(pool) < n:
                ids = []
                break
            ids.extend(pool["player_id"].astype(int).tolist())
        
        if not ids:
            continue
        
        score = float(t[t["player_id"].astype(int).isin(ids)]["score"].sum())
        if score > best[0]:
            best = (score, ids)
    
    return best[1]


def _with_last_gw_points(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "last_gw_points" not in out.columns and "event_points" in out.columns:
        out["last_gw_points"] = pd.to_numeric(out["event_points"], errors="coerce")
    return out


def apply_player_filters(
    df: pd.DataFrame,
    *,
    q: str,
    position: str,
    teams: list[str],
    price_min: float,
    price_max: float,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()

    if q:
        name_s = out.get("web_name", pd.Series(["" for _ in range(len(out))]))
        team_s = out.get("team", pd.Series(["" for _ in range(len(out))]))
        out = out[
            name_s.astype(str).str.contains(q, case=False, na=False)
            | team_s.astype(str).str.contains(q, case=False, na=False)
        ]

    if position and position != "ALL" and "position" in out.columns:
        out = out[out["position"].astype(str) == position]

    if teams and "team" in out.columns:
        out = out[out["team"].isin(teams)]

    if "price" in out.columns:
        price = pd.to_numeric(out["price"], errors="coerce").fillna(0.0)
        out = out[(price >= float(price_min)) & (price <= float(price_max))]

    return out


def _fdr_css_from_value(fdr: object) -> str:
    """Color fixture difficulty cells based on numeric FDR (1-5)."""
    try:
        f = int(fdr)
    except Exception:
        return ""

    if f == 1:
        return "background-color:#006400;color:white;font-weight:700;"  # dark green
    if f == 2:
        return "background-color:#00a65a;color:white;font-weight:700;"  # green
    if f == 3:
        return "background-color:#f3f4f6;color:#111827;font-weight:700;"  # pale
    if f == 4:
        return "background-color:#ff6b6b;color:#111827;font-weight:700;"  # red
    if f == 5:
        return "background-color:#8b0000;color:white;font-weight:700;"  # dark red
    return ""

# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="FPL Analytics Hub",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #37003c; color: white; }
    .stTabs [data-baseweb="tab-list"] { background-color: #37003c; }
    .stTabs [data-baseweb="tab"] { color: white; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #00ff87; color: #37003c; }
</style>
""", unsafe_allow_html=True)

st.title("⚽ FPL Analytics Hub")
st.caption("Professional Fantasy Premier League Planning & Predictions")

# Load data
projections = load_projections()

if projections.empty:
    st.error("No projections data found. Please add `data/projections.csv` or `outputs/projections.csv`.")
    st.stop()

# Load fixtures (prefer local, else cached network)
if FIXTURES_JSON.exists():
    try:
        fixtures = load_json(FIXTURES_JSON)
    except Exception:
        fixtures = pd.DataFrame()
else:
    try:
        st.info("📥 Loading fixtures from FPL API...")
        payload = fetch_fixtures_json()
        # FPL fixtures endpoint returns a list
        fixtures = pd.DataFrame(payload) if isinstance(payload, list) else pd.DataFrame(payload.get("fixtures", []))
        if fixtures.empty:
            st.warning("⚠️ Fixtures loaded but empty.")
        else:
            st.success("✅ Fixtures loaded")
    except Exception as e:
        st.warning(f"⚠️ Could not load fixtures: {e}")
        fixtures = pd.DataFrame()

if not INSIGHTS_PLAYERSTATS.exists():
    st.warning(
        f"⚠️ Insights playerstats not found at {INSIGHTS_PLAYERSTATS}. "
        "Advanced stats like xGC/90 will be blank until this file exists."
    )

fixture_index = build_fixture_index(fixtures)

# Raw projections table (requested: link directly to outputs file)
projections_out = load_csv(PROJ_CSV) if PROJ_CSV.exists() else pd.DataFrame()

# Normalize raw outputs for consistent display + filtering
projections_out_norm = normalize_projections(projections_out) if not projections_out.empty else pd.DataFrame()
projections_out_norm = _with_last_gw_points(projections_out_norm)

projections = _with_last_gw_points(projections)

# Sidebar
st.sidebar.title("🎯 Filters")

# Cache control (helps when code/data changed but Streamlit cache is stale)
if st.sidebar.button("Clear cached data", help="Clears Streamlit caches and reruns the app"):
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass
    st.sidebar.success("Cache cleared")
    st.rerun()

with st.sidebar.expander("Data health", expanded=False):
    try:
        _rows = int(len(projections))
        _xgc = int(pd.to_numeric(projections.get("expected_goals_conceded_per_90"), errors="coerce").notna().sum())
        st.caption(f"Rows: {_rows}")
        st.caption(f"xGC/90 coverage: {_xgc}/{_rows}")
    except Exception:
        st.caption("(health metrics unavailable)")

q_sidebar = st.sidebar.text_input("Search", value="")

pos_sidebar = st.sidebar.selectbox(
    "Position",
    options=["ALL", "GK", "DEF", "MID", "FWD"],
    index=0,
)

team_options = sorted([t for t in projections.get("team", pd.Series(dtype=str)).dropna().unique().tolist()])
teams_sidebar = st.sidebar.multiselect("Teams", options=team_options, default=[])

# Force full price range (prevents accidental caps like max=4.0)
price_options = [round(x * 0.5, 1) for x in range(7, 31)]  # 3.5 to 15.0
price_min_sidebar, price_max_sidebar = st.sidebar.select_slider(
    "Price",
    options=price_options,
    value=(3.5, 15.0),
)

# Column picker for the projections/stat tables
_base_cols = ["web_name", "team", "position", "price", "proj_points"]
_exclude_cols = set(_base_cols + ["player_id", "last_gw_points"])
extra_col_options = sorted([c for c in projections.columns if c not in _exclude_cols])
extra_columns = st.sidebar.multiselect(
    "Extra columns",
    options=extra_col_options,
    default=[],
    help="Select extra features to add to tables (xG, xA, defcon, etc).",
)

# Apply sidebar filters
projections_filtered = apply_player_filters(
    projections,
    q=q_sidebar,
    position=pos_sidebar,
    teams=teams_sidebar,
    price_min=float(price_min_sidebar),
    price_max=float(price_max_sidebar),
)
projections_out_filtered = apply_player_filters(
    projections_out_norm,
    q=q_sidebar,
    position=pos_sidebar,
    teams=teams_sidebar,
    price_min=float(price_min_sidebar),
    price_max=float(price_max_sidebar),
)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Projected Points",
    "🏆 My Team",
    "📅 Fixture Difficulty",
    "📈 Key Stats",
    "🔄 Transfer Planner",
    "🎯 Differentials",
    "📉 Player Analysis",
])

with tab1:
    st.header("📊 Player Projections")

    if projections_out.empty:
        st.error("No projections data found in outputs/projections.csv")
        st.stop()

    st.caption("Showing outputs/projections.csv (normalized) + uncertainty/reasons (from diagnostics + FPL stats)")

    # Use outputs file as base (it contains per-GW columns), then enrich from full projections.
    df1 = projections_out_filtered.copy()
    if "player_id" in df1.columns and "player_id" in projections_filtered.columns:
        enrich_cols = [
            "player_id",
            "team_id",
            "expected_goal_involvements_per_90",
            "expected_goals_per_90",
            "expected_assists_per_90",
            "expected_goals_conceded_per_90",
            "threat",
            "creativity",
            "defcon_points",
            "role",
            "selected_by_percent",
        ]
        enrich_cols = [c for c in enrich_cols if c in projections_filtered.columns]
        if enrich_cols:
            df1 = df1.merge(
                projections_filtered[enrich_cols].drop_duplicates(subset=["player_id"]).copy(),
                on="player_id",
                how="left",
                suffixes=("", "_enriched"),
            )

    # Attach uncertainty + reasons
    df1 = attach_uncertainty(df1)
    df1 = attach_reason_codes(df1, fixture_index=fixture_index, gw=23)

    # Delta vs replacement (4.5m baseline by position)
    baseline: dict[str, float] = {}
    if "price" in df1.columns and "position" in df1.columns and "proj_points" in df1.columns:
        for pos in ["GK", "DEF", "MID", "FWD"]:
            pool = df1[(df1["position"].astype(str) == pos) & (pd.to_numeric(df1["price"], errors="coerce").fillna(0.0) <= 4.5)]
            if pool.empty:
                continue
            baseline[pos] = float(pd.to_numeric(pool["proj_points"], errors="coerce").fillna(0.0).quantile(0.75))
        df1["delta_vs_4.5"] = df1.apply(lambda r: float(pd.to_numeric(r.get("proj_points"), errors="coerce") or 0.0) - float(baseline.get(str(r.get("position")), 0.0)), axis=1)

    # Price trend (LiveFPL predictor)
    if "player_id" in df1.columns:
        ids = (
            pd.to_numeric(df1["player_id"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        ids_key = tuple(sorted([int(x) for x in ids]))
        trend_map = compute_livefpl_trend_map(ids_key) if ids_key else {}
        df1["price_trend"] = pd.to_numeric(df1["player_id"], errors="coerce").map(
            lambda pid: trend_map.get(int(pid), "n/a") if pd.notna(pid) else "n/a"
        )

    # Per-GW columns if present
    gw_cols = [c for c in df1.columns if str(c).startswith("GW") and str(c).endswith("_proj_points")]
    gw_cols = sorted(gw_cols, key=lambda c: int(re.findall(r"\d+", c)[0]) if re.findall(r"\d+", c) else 999)
    gw_cols_show = st.multiselect(
        "Per-GW columns",
        options=gw_cols,
        default=gw_cols,
        help="These come from outputs/projections.csv",
    )

    preferred = [
        "web_name",
        "team",
        "position",
        "price",
        "price_trend",
        "proj_points",
        "proj_p25",
        "proj_p75",
        "delta_vs_4.5",
        "fixtures_next3",
        "why",
        *gw_cols_show,
        *[c for c in extra_columns if c in df1.columns],
    ]
    cols = [c for c in preferred if c in df1.columns]
    df1 = df1[cols]

    df1 = df1.rename(
        columns={
            "web_name": "player",
            "proj_points": "proj",
            "proj_p25": "p25",
            "proj_p75": "p75",
        }
    )

    col_config = {
        "price": st.column_config.NumberColumn("£", format="%.1f"),
        "price_trend": st.column_config.TextColumn("Trend", width="small"),
        "proj": st.column_config.NumberColumn("Proj", format="%.2f"),
        "p25": st.column_config.NumberColumn("P25", format="%.2f"),
        "p75": st.column_config.NumberColumn("P75", format="%.2f"),
        "delta_vs_4.5": st.column_config.NumberColumn("Δ vs 4.5", format="%.2f"),
        "fixtures_next3": st.column_config.TextColumn("Next (3)", width="medium"),
        "why": st.column_config.TextColumn("Why", width="large"),
    }
    for c in gw_cols_show:
        col_config[c] = st.column_config.NumberColumn(c.replace("_proj_points", ""), format="%.2f")

    st.dataframe(
        df1.sort_values("proj", ascending=False) if "proj" in df1.columns else df1,
        width="stretch",
        height=600,
        column_config=col_config,
        hide_index=True,
    )

    st.download_button(
        "Download projections.csv",
        data=PROJ_CSV.read_bytes() if PROJ_CSV.exists() else b"",
        file_name="projections.csv",
        mime="text/csv",
        disabled=not PROJ_CSV.exists(),
    )

with tab2:
    st.header("🏆 My Team Builder")
    
    # Initialize session state
    if "my_team_df" not in st.session_state:
        st.session_state.my_team_df = pd.DataFrame()
    
    if "starters_ids" not in st.session_state:
        st.session_state.starters_ids = []
    
    if "bench_ids" not in st.session_state:
        st.session_state.bench_ids = []
    
    # Load team from FPL
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        entry_id_input = st.text_input("FPL Entry ID", value=str(ENTRY_ID))
    
    with col2:
        gw_input = st.text_input("Gameweek", value=str(GW))
    
    with col3:
        if st.button("🔄 Load Team"):
            picks_df = get_my_team(int(entry_id_input), int(gw_input))
            
            if not picks_df.empty:
                # Merge with projections
                team_ids = picks_df["element"].tolist()
                team_df = projections[projections["player_id"].isin(team_ids)].copy()
                
                st.session_state.my_team_df = team_df
                
                # Auto-select starting XI
                starters = best_xi_ids(team_df)
                st.session_state.starters_ids = starters
                st.session_state.bench_ids = [int(x) for x in team_df["player_id"] if int(x) not in starters]
                
                st.success(f"Loaded {len(team_df)} players!")
                st.rerun()
    
    # Display team
    team_df = st.session_state.my_team_df
    
    if not team_df.empty:
        summary = team_summary(team_df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Players", len(team_df))
        with col2:
            st.metric("Total Cost", f"${summary['cost']:.1f}")
        with col3:
            st.metric("Projected Points", f"{team_df['proj_points'].sum():.1f}")
        
        # Formation validation (silent enforcement; only warn if XI is incomplete)
        starters_count = len(st.session_state.starters_ids)
        ok, msg, formation = validate_starting_xi(team_df, st.session_state.starters_ids)
        if ok and formation:
            st.success(f"Formation: {formation}")
        elif msg and starters_count != 11:
            st.warning(msg)
        
        starters_ids = [int(x) for x in st.session_state.starters_ids]
        bench_ids = [int(x) for x in st.session_state.bench_ids]

        starters_df = team_df[team_df["player_id"].astype(int).isin(starters_ids)].copy()
        bench_df = team_df[team_df["player_id"].astype(int).isin(bench_ids)].copy()

        st.subheader("Starting XI")
        # Render kit images + bench buttons (disabled if no valid swap exists)
        for pos in ["GK", "DEF", "MID", "FWD"]:
            pos_players = starters_df[starters_df.get("position", "") == pos].copy()
            if pos_players.empty:
                continue
            st.markdown(f"**{pos}**")
            cols = st.columns(len(pos_players))
            for idx, (_, player) in enumerate(pos_players.iterrows()):
                with cols[idx]:
                    pid = int(player.get("player_id"))
                    team_code = player.get("team_code", 0)
                    col_a, col_b = st.columns([1, 2])
                    with col_a:
                        if pd.notna(team_code) and int(team_code) > 0:
                            st.image(get_shirt_url(int(team_code)), width=60)
                    with col_b:
                        if pd.notna(pid):
                            st.image(get_player_photo_url(int(pid)), width=50)
                    st.caption(str(player.get("web_name", "")))
                    st.caption(f"${float(player.get('price', 0) or 0):.1f}")

                    swap_in = best_valid_swap_in(team_df, starter_out=pid, starters_ids=starters_ids, bench_ids=bench_ids)
                    disabled = swap_in is None
                    help_txt = "Cannot bench (would break formation)" if disabled else ""
                    if st.button("Bench", key=f"bench_{pid}", disabled=disabled, help=help_txt):
                        # swap out this starter with the best valid bench player
                        st.session_state.starters_ids = [x for x in starters_ids if int(x) != pid] + [int(swap_in)]
                        st.session_state.bench_ids = [x for x in bench_ids if int(x) != int(swap_in)] + [pid]
                        st.rerun()

        st.subheader("Bench")
        if bench_df.empty:
            st.info("Bench is empty")
        else:
            bench_view = bench_df[[c for c in ["web_name", "team", "position", "price", "proj_points"] if c in bench_df.columns]].copy()
            if "price" in bench_view.columns:
                bench_view["price"] = bench_view["price"].apply(lambda v: f"${float(v):.1f}" if pd.notna(v) else "")
            st.dataframe(bench_view, width="stretch")

        st.markdown("### Quick Swap")
        if not starters_df.empty and not bench_df.empty:
            c1, c2, c3 = st.columns([2, 2, 1])
            starter_options = [(int(p["player_id"]), str(p.get("web_name", ""))) for _, p in starters_df.iterrows() if pd.notna(p.get("player_id"))]
            bench_options = [(int(p["player_id"]), str(p.get("web_name", ""))) for _, p in bench_df.iterrows() if pd.notna(p.get("player_id"))]

            with c1:
                starter_to_swap = st.selectbox(
                    "Starter",
                    options=starter_options,
                    format_func=lambda x: x[1],
                )
            with c2:
                bench_to_swap = st.selectbox(
                    "Bench",
                    options=bench_options,
                    format_func=lambda x: x[1],
                )
            new_starters = [x for x in starters_ids if int(x) != int(starter_to_swap[0])] + [int(bench_to_swap[0])]
            can_swap, swap_msg, _ = validate_starting_xi(team_df, new_starters)
            with c3:
                if st.button("⇄ Swap", disabled=not can_swap, help=(swap_msg if not can_swap else "")):
                    st.session_state.starters_ids = new_starters
                    st.session_state.bench_ids = [x for x in bench_ids if int(x) != int(bench_to_swap[0])] + [int(starter_to_swap[0])]
                    st.rerun()

        st.markdown("### 👑 Captaincy Ranking")
        if not team_df.empty and starters_ids:
            starters_df_xi = team_df[team_df["player_id"].astype(int).isin(starters_ids)].copy()
            if not starters_df_xi.empty:
                starters_df_xi["captain_score"] = (
                    pd.to_numeric(starters_df_xi.get("proj_points", 0), errors="coerce").fillna(0) * 1.0
                    + pd.to_numeric(starters_df_xi.get("expected_goals_per_90", 0), errors="coerce").fillna(0) * 10
                    + pd.to_numeric(starters_df_xi.get("expected_assists_per_90", 0), errors="coerce").fillna(0) * 5
                )
                for idx, (_, p) in enumerate(starters_df_xi.nlargest(5, "captain_score").iterrows(), 1):
                    st.write(
                        f"{idx}. **{p.get('web_name','')}** ({p.get('team','')}) "
                        f"→ {float(p.get('captain_score',0) or 0):.1f}"
                    )
    else:
        st.info("No squad loaded yet. Enter your FPL Entry ID and click 'Load Team'.")

with tab3:
    st.header("📅 Fixture Difficulty Ticker")

    if fixtures.empty or not fixture_index:
        st.error("Fixtures could not be loaded from file or FPL API.")
    else:
        teams_lookup = load_team_lookup()
        team_ids = sorted(teams_lookup.keys()) if teams_lookup else sorted(fixture_index.keys())

        # GW range
        try:
            gw_series = pd.to_numeric(fixtures.get("event"), errors="coerce") if "event" in fixtures.columns else pd.Series(dtype=float)
            gw_min = int(gw_series.dropna().min()) if not gw_series.dropna().empty else 1
            gw_max = int(gw_series.dropna().max()) if not gw_series.dropna().empty else 38
        except Exception:
            gw_min, gw_max = 1, 38

        # Start from GW23 (previous gameweeks finished)
        gw_min_ui = max(int(gw_min), 23)
        if gw_min_ui > gw_max:
            gw_min_ui = int(gw_min)

        gw_start_default = gw_min_ui
        gw_end_default = min(gw_min_ui + 5, int(gw_max))
        gw_start, gw_end = st.slider(
            "Gameweek Range",
            min_value=int(gw_min_ui),
            max_value=int(gw_max),
            value=(int(gw_start_default), int(gw_end_default)),
        )

        score_mode = st.selectbox(
            "Team difficulty score",
            options=["Total FDR"],
            index=0,
            help="Lower is easier. Computed from fixture difficulty (1-5) across the selected gameweeks.",
        )



        rows = []
        fdr_rows = []
        for tid in team_ids:
            label = team_short_name(tid)
            row = {"Team": label}
            fdr_row = {"Team": label}
            for gw in range(int(gw_start), int(gw_end) + 1):
                fx_list = (fixture_index.get(int(tid), {}) or {}).get(int(gw), [])
                if not fx_list:
                    row[f"GW{gw}"] = "—"
                    fdr_row[f"GW{gw}"] = pd.NA
                    continue

                cells = []
                fdr_vals: list[int] = []
                for fx in fx_list:
                    opp = team_short_name(fx.get("opp_id"))
                    ha = fx.get("ha")
                    fdr = fx.get("fdr")
                    # Display format: OPP (H/A). Don't show the difficulty number.
                    cells.append(f"{opp} ({ha})")
                    try:
                        if fdr is not None and pd.notna(fdr):
                            fdr_vals.append(int(fdr))
                    except Exception:
                        pass

                row[f"GW{gw}"] = ", ".join(cells)
                fdr_row[f"GW{gw}"] = max(fdr_vals) if fdr_vals else pd.NA
            rows.append(row)
            fdr_rows.append(fdr_row)

        ticker_df = pd.DataFrame(rows)
        ticker_fdr_df = pd.DataFrame(fdr_rows)
        gw_cols = [c for c in ticker_df.columns if str(c).startswith("GW")]

        # Compute per-team difficulty score so users can quickly see easiest runs.
        if gw_cols:
            fdr_numeric = ticker_fdr_df[gw_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
        else:
            fdr_numeric = pd.DataFrame(index=ticker_df.index)

        if score_mode == "Total FDR":
            team_score = fdr_numeric.sum(axis=1, skipna=True)
        else:
            team_score = fdr_numeric.sum(axis=1, skipna=True)

        score_col = score_mode.replace(" ", "_").lower()
        ticker_df.insert(1, score_col, team_score.round(2))

        # Sort and filter to show easiest teams first.
        ticker_df = ticker_df.sort_values(score_col, ascending=True, na_position="last")
        ticker_fdr_df = ticker_fdr_df.loc[ticker_df.index]
        fdr_numeric = fdr_numeric.loc[ticker_df.index] if not fdr_numeric.empty else fdr_numeric

        ticker_fdr_df = ticker_fdr_df.loc[ticker_df.index]
        fdr_numeric = fdr_numeric.loc[ticker_df.index] if not fdr_numeric.empty else fdr_numeric

        def _style_all(_: pd.DataFrame) -> pd.DataFrame:
            styles = pd.DataFrame("", index=ticker_df.index, columns=ticker_df.columns)
            for c in gw_cols:
                if not fdr_numeric.empty and c in fdr_numeric.columns:
                    styles[c] = fdr_numeric[c].apply(_fdr_css_from_value)
            return styles

        st.dataframe(
            ticker_df.style.apply(_style_all, axis=None),
            width="stretch",
            height=600,
        )

with tab4:
    st.header("📈 Key Player Statistics (FPL Official Stats)")

    fpl_stats = load_bootstrap_elements()
    if fpl_stats.empty:
        st.error("Could not load FPL stats")
    else:
        stats_cols = [
            "web_name",
            "team",
            "element_type",
            "now_cost",
            "selected_by_percent",
            "total_points",
            "minutes",
            "goals_scored",
            "assists",
            "clean_sheets",
            "expected_goals_per_90",
            "expected_assists_per_90",
            "expected_goal_involvements_per_90",
            "influence",
            "creativity",
            "threat",
            "ict_index",
            "form",
        ]
        stats_cols = [c for c in stats_cols if c in fpl_stats.columns]
        df_stats = fpl_stats[stats_cols].copy()

        # Map team id to short name
        if "team" in df_stats.columns:
            df_stats["team"] = df_stats["team"].map(team_short_name)

        # Map element_type to position
        POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        if "element_type" in df_stats.columns:
            df_stats["position"] = df_stats["element_type"].map(POS_MAP)
            df_stats = df_stats.drop(columns=["element_type"])

        # Convert now_cost (tenths) to price
        if "now_cost" in df_stats.columns:
            df_stats["price"] = pd.to_numeric(df_stats["now_cost"], errors="coerce").fillna(0.0) / 10.0
            df_stats = df_stats.drop(columns=["now_cost"])

        if "selected_by_percent" in df_stats.columns:
            df_stats = df_stats.rename(columns={"selected_by_percent": "selected_by_%"})

        # Apply sidebar filters
        df_stats = apply_player_filters(
            df_stats,
            q=q_sidebar,
            position=pos_sidebar,
            teams=teams_sidebar,
            price_min=float(price_min_sidebar),
            price_max=float(price_max_sidebar),
        )

        # Formatting
        fmt = {
            "price": lambda v: f"${float(v):.1f}" if pd.notna(v) else "",
            "selected_by_%": lambda v: f"{float(v):.1f}%" if pd.notna(v) else "",
        }
        st.dataframe(df_stats.style.format(fmt), width="stretch", height=600)

with tab5:
    st.header("🔄 Transfer Planner")

    st.caption("Squad rules enforced: $100.0 budget · Max 3 per club · 2 GK / 5 DEF / 5 MID / 3 FWD (15 total)")

    if projections.empty:
        st.error("No projections loaded.")
        st.stop()

    if "team_ids" not in st.session_state:
        st.session_state.team_ids = []
    if "tp_starters_ids" not in st.session_state:
        st.session_state.tp_starters_ids = []
    if "tp_bench_ids" not in st.session_state:
        st.session_state.tp_bench_ids = []

    # Seed from My Team if available
    if (not st.session_state.team_ids) and ("my_team_df" in st.session_state) and (not st.session_state.my_team_df.empty):
        try:
            st.session_state.team_ids = [int(x) for x in st.session_state.my_team_df["player_id"].dropna().astype(int).tolist()]
        except Exception:
            st.session_state.team_ids = []

    # Record baseline squad once (used for transfer/hits calculator)
    if "tp_base_team_ids" not in st.session_state and len(st.session_state.team_ids) == 15:
        st.session_state.tp_base_team_ids = [int(x) for x in st.session_state.team_ids]

    if st.session_state.team_ids and "player_id" in projections.columns:
        team_df = projections[projections["player_id"].astype("Int64").isin([int(x) for x in st.session_state.team_ids])].copy()
    else:
        team_df = projections.head(0).copy() if not projections.empty else pd.DataFrame()

    # Auto-pick starters/bench
    if (not team_df.empty) and (len(team_df) >= 11) and (not st.session_state.tp_starters_ids):
        starters = best_xi_ids(team_df)
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
        # First-load helper (so users don't have to visit My Team first)
        with st.expander("📥 Load squad from FPL entry", expanded=("my_team_df" not in st.session_state) or st.session_state.my_team_df.empty):
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                tp_entry_id = st.text_input("FPL Entry ID", value=str(ENTRY_ID), key="tp_entry_id")
            with c2:
                tp_gw = st.text_input("Gameweek", value=str(GW), key="tp_gw")
            with c3:
                if st.button("🔄 Load", key="tp_load_team"):
                    picks_df = get_my_team(int(tp_entry_id), int(tp_gw))
                    if not picks_df.empty:
                        team_ids = picks_df["element"].tolist()
                        team_loaded = projections[projections["player_id"].isin(team_ids)].copy()
                        st.session_state.my_team_df = team_loaded
                        st.session_state.team_ids = [int(x) for x in team_loaded["player_id"].dropna().astype(int).tolist()]
                        st.session_state.tp_starters_ids = []
                        st.session_state.tp_bench_ids = []
                        st.rerun()

        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Your squad")
        with col2:
            if ("my_team_df" in st.session_state) and (not st.session_state.my_team_df.empty):
                if st.button("📥 Import from My Team"):
                    try:
                        st.session_state.team_ids = [
                            int(x) for x in st.session_state.my_team_df["player_id"].tolist()
                        ]
                        st.session_state.tp_starters_ids = []
                        st.session_state.tp_bench_ids = []
                        st.rerun()
                    except Exception:
                        pass

        if team_df.empty:
            st.info("Add players on the right to build a squad.")
        else:
            summary = team_summary(team_df)
            st.metric("Team Cost", f"${summary['cost']:.1f}")
            st.metric("Budget Remaining", f"${max(0.0, BUDGET - summary['cost']):.1f}")
            st.write(
                f"Positions: GK {summary['counts'].get('GK',0)}/2 · DEF {summary['counts'].get('DEF',0)}/5 · "
                f"MID {summary['counts'].get('MID',0)}/5 · FWD {summary['counts'].get('FWD',0)}/3"
            )

            # Transfer/hits calculator (vs imported baseline)
            base_ids = set(int(x) for x in st.session_state.get("tp_base_team_ids", []))
            cur_ids = set(int(x) for x in team_df.get("player_id", pd.Series(dtype=int)).dropna().astype(int).tolist())
            transfers_made = max(0, len(base_ids - cur_ids)) if base_ids and cur_ids else 0
            free_transfers = st.selectbox("Free transfers available", options=[1, 2], index=0, key="tp_free_transfers")
            hits = max(0, transfers_made - int(free_transfers))
            c_a, c_b, c_c = st.columns(3)
            with c_a:
                st.metric("Transfers Made", f"{transfers_made}")
            with c_b:
                st.metric("Free Transfers", f"{int(free_transfers)}")
            with c_c:
                st.metric("Points Deduction", f"-{hits * 4}")

            starters_ids = [int(x) for x in st.session_state.get("tp_starters_ids", [])]
            bench_ids = [int(x) for x in st.session_state.get("tp_bench_ids", [])]

            if len(starters_ids) != 11 and len(team_df) >= 11:
                starters_ids = best_xi_ids(team_df)
                bench_ids = [
                    int(x)
                    for x in team_df["player_id"].astype(int).tolist()
                    if int(x) not in set(int(s) for s in starters_ids)
                ]

            st.session_state.tp_starters_ids = starters_ids
            st.session_state.tp_bench_ids = bench_ids

            ok, msg, formation = validate_starting_xi(team_df, starters_ids) if len(starters_ids) == 11 else (False, "Pick 11 starters", None)
            if ok and formation:
                st.success(f"Formation: {formation}")
            elif msg and len(starters_ids) != 11:
                st.warning(msg)

            st.markdown("**Starting XI**")
            starters_df = team_df[team_df["player_id"].astype(int).isin(starters_ids)].copy() if starters_ids else team_df.head(0)
            st.dataframe(starters_df[[c for c in ["web_name", "team", "position", "price", "proj_points"] if c in starters_df.columns]], width="stretch")

            st.markdown("**Bench**")
            bench_df = team_df[team_df["player_id"].astype(int).isin(bench_ids)].copy() if bench_ids else team_df.head(0)
            st.dataframe(bench_df[[c for c in ["web_name", "team", "position", "price", "proj_points"] if c in bench_df.columns]], width="stretch")

            st.markdown("**Remove players**")
            for _, r in team_df.sort_values("proj_points", ascending=False).iterrows():
                pid = int(r.get("player_id")) if pd.notna(r.get("player_id")) else None
                if pid is None:
                    continue

                can_rm, reason = can_remove_from_squad(team_df, pid)

                cols = st.columns([0.8, 0.8, 2, 0.8, 0.8, 0.8, 0.6])
                with cols[0]:
                    team_code = r.get("team_code", 0)
                    if pd.notna(team_code) and int(team_code) > 0:
                        st.image(get_shirt_url(int(team_code)), width=40)
                with cols[1]:
                    st.image(get_player_photo_url(int(pid)), width=40)
                with cols[2]:
                    st.write(r.get("web_name", ""))
                with cols[3]:
                    st.caption(r.get("team", ""))
                with cols[4]:
                    st.caption(r.get("position", ""))
                with cols[5]:
                    st.caption(f"£{float(r.get('price',0) or 0):.1f}")
                with cols[6]:
                    if st.button("✕", key=f"tp_remove_{pid}", disabled=not can_rm, help=(reason if not can_rm else "Remove")):
                        _remove_from_planner(pid)
                        st.rerun()

    with right:
        st.subheader("Add players")
        q = st.text_input("Search", value="")
        fpos = st.radio("Position", options=["ALL", "GK", "DEF", "MID", "FWD"], horizontal=True)
        price_opts = price_options_from_proj(projections)
        pmin, pmax = st.select_slider("Price", options=price_opts, value=(price_opts[0], price_opts[-1]))

        team_opts = sorted([t for t in projections.get("team", pd.Series(dtype=str)).dropna().unique().tolist()])
        fteams = st.multiselect("Filter by team(s)", options=team_opts, default=[])

        cand = projections.copy()
        if q:
            cand = cand[
                cand.get("web_name", "").astype(str).str.contains(q, case=False, na=False)
                | cand.get("team", "").astype(str).str.contains(q, case=False, na=False)
            ]
        if fpos != "ALL" and "position" in cand.columns:
            cand = cand[cand["position"].astype(str) == fpos]
        if "price" in cand.columns:
            cand = cand[(cand["price"] >= float(pmin)) & (cand["price"] <= float(pmax))]
        if fteams and "team" in cand.columns:
            cand = cand[cand["team"].isin(fteams)]

        cand = cand.sort_values("proj_points", ascending=False)

        st.markdown("**Top players**")
        for _, r in cand.head(25).iterrows():
            ok, reason = can_add_player(team_df, r)
            pid = r.get("player_id")
            try:
                pid_int = int(pid) if pd.notna(pid) else None
            except Exception:
                pid_int = None

            name = r.get("web_name")
            team = r.get("team")
            pos = r.get("position")
            price = float(r.get("price", 0) or 0)
            proj = float(r.get("proj_points", 0) or 0)

            c1, c2 = st.columns([4, 1])
            with c1:
                st.write(f"{name} — {team} · {pos} · ${price:.1f} · Pred {proj:.1f}")
                if reason and not ok:
                    st.caption(reason)
            with c2:
                if st.button("Add", key=f"tp_add_{pid_int or name}", disabled=(not ok) or (pid_int is None)):
                    st.session_state.team_ids = [*st.session_state.team_ids, int(pid_int)]
                    st.rerun()


with tab6:
    st.header("🎯 Differential Finder")
    st.caption("High projection + low ownership = competitive edge")

    if projections.empty:
        st.info("No projections loaded")
    elif "proj_points" not in projections.columns or "price" not in projections.columns:
        st.info("Projections are missing required columns")
    elif "selected_by_percent" not in projections.columns:
        st.warning("Ownership (selected_by_percent) not available in projections")
    else:
        min_proj = st.slider("Min projection", 2.0, 20.0, 5.0)
        max_own = st.slider("Max ownership %", 0.0, 50.0, 5.0)

        proj_pts = pd.to_numeric(projections["proj_points"], errors="coerce").fillna(0.0)
        own = pd.to_numeric(projections["selected_by_percent"], errors="coerce").fillna(0.0)
        price = pd.to_numeric(projections["price"], errors="coerce").replace(0, pd.NA)

        diffs = projections[(proj_pts > float(min_proj)) & (own < float(max_own))].copy()
        if not diffs.empty:
            diffs["value"] = pd.to_numeric(diffs["proj_points"], errors="coerce").fillna(0.0) / pd.to_numeric(diffs["price"], errors="coerce").replace(0, pd.NA)
            diffs = diffs.sort_values("value", ascending=False).head(30)

            disp_cols = [
                c
                for c in [
                    "web_name",
                    "team",
                    "position",
                    "price",
                    "proj_points",
                    "selected_by_percent",
                    "value",
                ]
                if c in diffs.columns
            ]
            st.dataframe(diffs[disp_cols], width="stretch")
        else:
            st.info("No differentials found with current filters")


with tab7:
    st.header("📉 Player Performance Analysis")
    st.caption("GW-by-GW history from the official FPL element-summary endpoint")

    if projections.empty or "player_id" not in projections.columns:
        st.info("No players available")
    else:
        player_opts = projections[[c for c in ["player_id", "web_name", "team", "position"] if c in projections.columns]].dropna(subset=["player_id"]).copy()
        if "web_name" not in player_opts.columns:
            st.info("Missing player names in projections")
        else:
            player_opts["player_id"] = pd.to_numeric(player_opts["player_id"], errors="coerce").astype("Int64")
            player_opts = player_opts.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id"]).copy()
            player_opts["display"] = player_opts["web_name"].astype(str) + " (" + player_opts.get("team", "").astype(str) + ")"

            selected = st.selectbox("Select Player", options=player_opts["display"].tolist())

            if selected:
                pid = int(player_opts[player_opts["display"] == selected]["player_id"].iloc[0])
                pname = selected.split(" (")[0]

                history = fetch_player_history(pid)
                if not history.empty:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Season Pts", int(pd.to_numeric(history.get("total_points", 0), errors="coerce").fillna(0).sum()))
                    with c2:
                        st.metric("Avg/GW", f"{pd.to_numeric(history.get('total_points', 0), errors='coerce').fillna(0).mean():.1f}")
                    with c3:
                        st.metric("Minutes", int(pd.to_numeric(history.get("minutes", 0), errors="coerce").fillna(0).sum()))

                    st.plotly_chart(create_points_chart(history, pname), width="stretch")
                else:
                    st.warning(f"No history for {pname}")

st.markdown("---")
st.caption("📊 Data from FPL API | ⚡ Powered by Machine Learning")
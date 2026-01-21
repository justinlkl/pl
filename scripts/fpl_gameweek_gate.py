from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def _load_marker(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_marker(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _find_event(events: list[dict[str, Any]], *, key: str) -> dict[str, Any] | None:
    for ev in events:
        if bool(ev.get(key)):
            return ev
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate CI work until an FPL gameweek is finalized")
    parser.add_argument(
        "--marker-path",
        default=str(Path("artifacts") / "last_postgw_event.json"),
        help="Marker file to prevent re-running the post-GW pipeline for the same event.",
    )
    parser.add_argument(
        "--require-data-checked",
        action="store_true",
        default=True,
        help="Require event.data_checked=true (points finalized) to proceed.",
    )
    parser.add_argument(
        "--no-require-data-checked",
        dest="require_data_checked",
        action="store_false",
        help="Allow proceeding when event.finished=true even if data_checked is false.",
    )
    parser.add_argument(
        "--write-github-output",
        action="store_true",
        help="Write outputs to the GitHub Actions output file ($GITHUB_OUTPUT).",
    )
    parser.add_argument(
        "--mark-processed",
        action="store_true",
        help="Write marker file for current event (use after successful pipeline run).",
    )
    args = parser.parse_args()

    marker_path = Path(args.marker_path)

    resp = requests.get(BOOTSTRAP_URL, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    events = payload.get("events")
    if not isinstance(events, list):
        raise SystemExit("FPL API response missing 'events' list")

    current = _find_event(events, key="is_current")
    nxt = _find_event(events, key="is_next")

    current_id = int(current.get("id")) if isinstance(current, dict) and current.get("id") is not None else None
    next_id = int(nxt.get("id")) if isinstance(nxt, dict) and nxt.get("id") is not None else None

    finished = bool(current.get("finished")) if isinstance(current, dict) else False
    data_checked = bool(current.get("data_checked")) if isinstance(current, dict) else False

    should_run = bool(finished and (data_checked or (not bool(args.require_data_checked))))

    # Prevent double-runs for the same GW.
    marker = _load_marker(marker_path)
    last_processed = None
    if isinstance(marker, dict) and marker.get("event_id") is not None:
        try:
            last_processed = int(marker.get("event_id"))
        except Exception:
            last_processed = None

    already_processed = (last_processed is not None) and (current_id is not None) and (last_processed == current_id)
    if already_processed:
        should_run = False

    outputs: dict[str, str] = {
        "should_run": "true" if should_run else "false",
        "current_gw": "" if current_id is None else str(current_id),
        "next_gw": "" if next_id is None else str(next_id),
        "finished": "true" if finished else "false",
        "data_checked": "true" if data_checked else "false",
        "already_processed": "true" if already_processed else "false",
        "marker_path": str(marker_path).replace("\\", "/"),
    }

    if args.mark_processed:
        if current_id is None:
            raise SystemExit("Cannot mark processed: current_gw is unknown")
        _write_marker(
            marker_path,
            {
                "event_id": current_id,
                "next_event_id": next_id,
                "finished": finished,
                "data_checked": data_checked,
            },
        )

    if args.write_github_output:
        out_file = os.environ.get("GITHUB_OUTPUT")
        if not out_file:
            raise SystemExit("GITHUB_OUTPUT not set")
        with open(out_file, "a", encoding="utf-8") as f:
            for k, v in outputs.items():
                f.write(f"{k}={v}\n")
    else:
        print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

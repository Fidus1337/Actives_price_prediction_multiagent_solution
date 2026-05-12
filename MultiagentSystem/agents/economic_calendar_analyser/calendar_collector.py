"""
Incremental CoinGlass economic calendar collector.

Fetches macro-economic events from CoinGlass API (~28 day rolling window)
and merges them into a local SQLite archive, deduplicating by
(calendar_name, publish_timestamp).

No classification at this stage — just raw data archiving.
Classification and LLM aggregation will be added in later steps.

Usage:
    python -m MultiagentSystem.agents.economic_calendar_analyser.calendar_collector
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from .economic_calendar_database_manipulator import (
    init_db, insert_events, load_events_in_range, get_db_stats
)

load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / "dev.env")

BASE_URL = "https://open-api-v4.coinglass.com/api"
API_KEY = os.getenv("COINGLASS_API_KEY")


def _ts_to_date_str(ts_ms) -> str | None:
    """Unix timestamp (ms) -> 'YYYY-MM-DD' string."""
    if ts_ms is None:
        return None
    try:
        dt = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, OSError):
        return None


def _fetch_from_api() -> list[dict]:
    """Fetch all economic calendar events from CoinGlass (single call, no pagination)."""
    if not API_KEY:
        print("[calendar_collector] ERROR: COINGLASS_API_KEY not found in dev.env")
        return []

    url = f"{BASE_URL}/calendar/economic-data"
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}

    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json().get("data", [])

    print(f"[calendar_collector] Fetched {len(data)} events from API")
    return data


def collect_calendar_events() -> dict:
    """
    Main collection function.
    Fetches all available events from API, deduplicates against
    existing archive, and saves.

    Returns stats: {before, fetched, new, after, date_range}.
    """
    init_db()
    stats_before = get_db_stats()

    fresh = _fetch_from_api()
    # Add human-readable date field before inserting
    for e in fresh:
        if "date" not in e:
            e["date"] = _ts_to_date_str(e.get("publish_timestamp"))

    new_count = insert_events(fresh)
    stats_after = get_db_stats()

    stats = {
        "before": stats_before["count"],
        "fetched": len(fresh),
        "new": new_count,
        "after": stats_after["count"],
        "date_range": stats_after["date_range"],
    }

    print(f"[calendar_collector] Archive: {stats['before']} -> {stats['after']} events (+{new_count} new)")
    print(f"[calendar_collector] Date range: {stats['date_range']}")

    return stats


def get_events_in_range(
    dt_from: datetime,
    dt_to: datetime,
) -> list[dict]:
    """
    Read events from the local archive within [dt_from, dt_to].
    """
    return load_events_in_range(dt_from, dt_to)


if __name__ == "__main__":
    stats = collect_calendar_events()
    print(f"\nDone. Stats: {json.dumps(stats, indent=2)}")

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


UNBIAS_ANALYSTS_API = "https://unbias.fyi/api/analysts"
ACCURACY_SORTS = {
    "Overall": "accuracy_score",
}


def _fetch_analysts(
    sort_key: str,
    top_n: int,
    source: str = "all",
    order: str = "desc",
    timeout_seconds: float = 20.0,
) -> list[dict[str, Any]]:
    """Fetch analysts list from unbias API for a specific sort key."""
    params = {
        "source": source,
        "sort": sort_key,
        "order": order,
    }
    response = requests.get(UNBIAS_ANALYSTS_API, params=params, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    analysts = payload.get("analysts", [])
    if not isinstance(analysts, list):
        raise ValueError("Unexpected API schema: 'analysts' must be a list")
    return analysts[:top_n]


def _normalize_entry(item: dict[str, Any], rank: int) -> dict[str, Any]:
    """Pick a stable subset of fields for downstream usage."""
    return {
        "rank": rank,
        "handle": item.get("handle"),
        "name": item.get("name"),
        "source": item.get("source"),
        "accuracy_score": item.get("accuracy_score"),
        "bull_score": item.get("bull_score"),
        "bear_score": item.get("bear_score"),
        "balanced_score": item.get("balanced_score"),
        "call_count": item.get("call_count"),
        "follower_count": item.get("follower_count"),
    }


def fetch_top_sources_by_accuracy(
    top_n: int = 5,
    source: str = "all",
    order: str = "desc",
    timeout_seconds: float = 20.0,
) -> dict[str, list[dict[str, Any]]]:
    """
    Return top analysts by overall accuracy.

    Output keys:
      - Overall
    """
    if top_n <= 0:
        raise ValueError("top_n must be > 0")

    result: dict[str, list[dict[str, Any]]] = {}
    for bucket, sort_key in ACCURACY_SORTS.items():
        rows = _fetch_analysts(
            sort_key=sort_key,
            top_n=top_n,
            source=source,
            order=order,
            timeout_seconds=timeout_seconds,
        )
        result[bucket] = [_normalize_entry(row, rank=i + 1) for i, row in enumerate(rows)]
    return result


def build_snapshot(
    top_n: int = 5,
    source: str = "all",
    order: str = "desc",
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    """Build a self-contained snapshot payload with metadata + grouped results."""
    return {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_url": UNBIAS_ANALYSTS_API,
        "params": {
            "source": source,
            "order": order,
            "top_n": top_n,
            "sorts": ACCURACY_SORTS,
        },
        "top_by_accuracy": fetch_top_sources_by_accuracy(
            top_n=top_n,
            source=source,
            order=order,
            timeout_seconds=timeout_seconds,
        ),
    }


def save_snapshot_json(
    output_path: str | Path,
    top_n: int = 5,
    source: str = "all",
    order: str = "desc",
    timeout_seconds: float = 20.0,
) -> Path:
    """Fetch top analysts and persist a JSON snapshot."""
    payload = build_snapshot(
        top_n=top_n,
        source=source,
        order=order,
        timeout_seconds=timeout_seconds,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return output_path


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "unbias_top_sources.json"
    saved = save_snapshot_json(output_path=out, top_n=15)
    print(f"Saved: {saved}")

"""
Incremental CoinGlass news collector.

Fetches the latest articles from CoinGlass API (~24 day rolling window)
and merges them into a local SQLite archive, deduplicating by
(article_title, article_release_time).

New articles are classified (bull/bear/not_correlated + strength) at
collection time via news_classifier.py, so downstream agents can skip
LLM calls and aggregate pre-classified data directly.

Usage:
    python -m MultiagentSystem.agents.news_analyser.news_collector
    python -m MultiagentSystem.agents.news_analyser.news_collector --backfill
    python -m MultiagentSystem.agents.news_analyser.news_collector --reclassify
"""

import json
import sys
from datetime import datetime, timezone

from .helpers import (
    fetch_articles_in_window,
    coinglass_get_raw,
    parse_release_time,
    strip_html,
)
from .news_archive_database_manipulator import (
    init_db,
    insert_articles,
    load_articles_in_range,
    load_all_articles,
    get_unclassified_articles,
    reset_all_classifications,
    update_classifications,
    get_db_stats,
)
from .news_classifier import classify_articles


def _prepare_for_archive(article: dict) -> dict:
    """Strip HTML and add human-readable date before storing."""
    cleaned = dict(article)
    raw_content = cleaned.get("article_content", "")
    if raw_content:
        cleaned["article_content"] = strip_html(raw_content)
    dt = parse_release_time(cleaned.get("article_release_time"))
    if dt:
        cleaned["date"] = dt.strftime("%Y-%m-%d")
    return cleaned


def fetch_all_available(max_pages: int = 50) -> list[dict]:
    """Fetch all articles currently available from the API (no date filter)."""
    results: list[dict] = []

    for page in range(1, max_pages + 1):
        resp = coinglass_get_raw("/article/list", {"page": page})
        data = resp.get("data")
        if not data or not isinstance(data, list):
            break
        results.extend(data)

        # If page returned fewer than ~20 items, we've hit the end
        if len(data) < 15:
            break

    print(f"[news_collector] Fetched {len(results)} articles from API")
    return results


def collect_news() -> dict:
    """
    Main collection function.
    Fetches all available articles from API, strips HTML, classifies,
    and saves to SQLite archive (duplicates ignored).

    Returns stats: {before, fetched, new, after, date_range}.
    """
    init_db()
    stats_before = get_db_stats()

    fresh = fetch_all_available()
    prepared = [_prepare_for_archive(a) for a in fresh]

    # Insert first — INSERT OR IGNORE deduplicates; new articles land with category=NULL
    new_count = insert_articles(prepared)

    # Classify only the newly inserted articles (unclassified ones in DB)
    if new_count > 0:
        unclassified = get_unclassified_articles()
        if unclassified:
            print(f"[news_collector] Classifying {len(unclassified)} new articles...")
            classify_articles(unclassified)
            update_classifications(unclassified)

    stats_after = get_db_stats()

    stats = {
        "before": stats_before["count"],
        "fetched": len(fresh),
        "new": new_count,
        "after": stats_after["count"],
        "date_range": stats_after["date_range"],
    }

    print(f"[news_collector] Archive: {stats_before['count']} → {stats_after['count']} articles (+{new_count} new)")
    print(f"[news_collector] Date range: {stats_after['date_range']}")

    return stats


def get_articles_in_range(
    dt_from: datetime,
    dt_to: datetime,
    fallback_to_api: bool = True,
) -> list[dict]:
    """
    Read articles from the local archive within [dt_from, dt_to].

    If the archive has no articles in this range and fallback_to_api=True,
    fetches directly from the CoinGlass API (limited to ~24 days of history).
    """
    init_db()
    articles = load_articles_in_range(dt_from, dt_to)

    if not articles and fallback_to_api:
        print(f"[news_collector] Archive empty for {dt_from.date()} → {dt_to.date()}, falling back to API")
        articles = fetch_articles_in_window(dt_from, dt_to)

    return articles


def backfill_classifications() -> dict:
    """Classify all articles in archive that lack category/strength fields.

    Idempotent: safe to run multiple times. Re-attempts 'unclassified' articles too.
    """
    init_db()
    unclassified = get_unclassified_articles()
    if not unclassified:
        print("[news_collector] All articles already classified — nothing to backfill")
        return {"backfilled": 0}

    print(f"[news_collector] Backfilling {len(unclassified)} articles...")
    classify_articles(unclassified)
    update_classifications(unclassified)

    print(f"[news_collector] Backfill complete: {len(unclassified)} articles classified")
    return {"backfilled": len(unclassified)}


def reclassify_all() -> dict:
    """Re-classify ALL articles in the archive from scratch.

    Resets existing category/strength and runs classification again
    with current logic (date-grouped batches).
    """
    init_db()
    stats = get_db_stats()
    if stats["count"] == 0:
        print("[news_collector] Archive is empty — nothing to reclassify")
        return {"reclassified": 0}

    reset_all_classifications()
    all_articles = load_all_articles()

    print(f"[news_collector] Reclassifying all {len(all_articles)} articles...")
    classify_articles(all_articles)
    update_classifications(all_articles)

    print(f"[news_collector] Reclassification complete: {len(all_articles)} articles")
    return {"reclassified": len(all_articles)}


if __name__ == "__main__":
    if "--reclassify" in sys.argv:
        result = reclassify_all()
        print(f"\nDone. Result: {json.dumps(result, indent=2)}")
    elif "--backfill" in sys.argv:
        result = backfill_classifications()
        print(f"\nDone. Result: {json.dumps(result, indent=2)}")
    else:
        stats = collect_news()
        print(f"\nDone. Stats: {json.dumps(stats, indent=2)}")

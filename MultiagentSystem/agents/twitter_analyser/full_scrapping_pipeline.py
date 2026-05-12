"""
Full Twitter scrapping pipeline: fetch and store only (no classification).

Usage:
    python -m MultiagentSystem.agents.twitter_analyser.full_scrapping_pipeline
"""

import json
import random
import time
from datetime import datetime
from pathlib import Path

from MultiagentSystem.agents.twitter_analyser.twitter_scrapper.chrome_login_before_scrapping import (
    create_driver,
)
from MultiagentSystem.agents.twitter_analyser.twitter_scrapper.twscraper import (
    fetch_tweets_sync,
)
from MultiagentSystem.agents.twitter_analyser.twitter_scrapper.twitter_db import (
    count_tweets,
    get_date_range,
    get_tweet_ids_by_author_in_range,
    get_tweets_in_range,
    init_db,
    insert_tweets,
    update_tweet_signals,
)
from MultiagentSystem.agents.twitter_analyser.twitter_news_classifier.classifier import (
    classify_tweets,
)

ACCOUNTS_CONFIG_PATH = Path(__file__).parent / "twitter_collector_settings.json"
LOG_TAG = "[twitter_collector]"


def _load_accounts_config() -> dict:
    """Load accounts + scraper settings from JSON config."""
    return json.loads(ACCOUNTS_CONFIG_PATH.read_text(encoding="utf-8"))


def _get_enabled_accounts(config: dict) -> list[str]:
    """Return enabled usernames from config."""
    accounts = config.get("accounts", [])
    enabled = []
    for item in accounts:
        username = (item.get("username") or "").strip().lstrip("@")
        if username and item.get("enabled", False):
            enabled.append(username)
    return enabled


def _normalize_limit(value: int | None) -> int | None:
    """
    Convert config max_tweets_per_author:
    -1 or 0 -> None (unlimited), positive -> same value
    """
    if value is None:
        return None
    if value <= 0:
        return None
    return value


def run_fetch_only(
    since_date: str | None = None,
    until_date: str | None = None,
    authors: list[str] | None = None,
    stop_on_existing_duplicates: bool = False,
    duplicates_threshold: int = 5,
) -> dict:
    """
    Parse tweets from configured accounts and save to SQLite archive.
    No LLM / no classification at this stage.

    Any of since_date / until_date / authors, if provided, overrides the value
    from twitter_collector_settings.json for this call only.

    stop_on_existing_duplicates:
        When True, for each source fetch the set of tweet_ids already stored
        in the DB within [since_date, until_date] and pass it to the scraper.
        As soon as the scraper encounters `duplicates_threshold` tweets that
        are already archived, it stops scrolling that source and moves on.
    duplicates_threshold:
        Number of already-archived tweets after which fetching for the current
        source is aborted. Only used when stop_on_existing_duplicates is True.
    """
    init_db()
    config = _load_accounts_config()

    scraper_settings = config.get("scraper_settings", {})

    if authors is not None:
        accounts = [a.strip().lstrip("@") for a in authors if a and a.strip()]
    else:
        accounts = _get_enabled_accounts(config)

    if since_date is None:
        since_date = scraper_settings.get("since_date")
    if until_date is None:
        until_date = scraper_settings.get("until_date")
    max_tweets = _normalize_limit(scraper_settings.get("max_tweets_per_author", 20))
    max_scrolls = int(scraper_settings.get("max_scrolls", 100))
    pause_min = float(scraper_settings.get("pause_min_sec", 3))
    pause_max = float(scraper_settings.get("pause_max_sec", 7))

    print(f"{LOG_TAG} Enabled accounts: {len(accounts)}")
    print(
        f"{LOG_TAG} Settings: since={since_date}, until={until_date}, "
        f"max_tweets_per_author={max_tweets}, max_scrolls={max_scrolls}"
    )

    if not accounts:
        print(f"{LOG_TAG} No enabled accounts in config. Nothing to do.")
        return {
            "accounts_total": 0,
            "tweets_fetched": 0,
            "tweets_inserted": 0,
            "db_total": count_tweets(),
            "db_range": get_date_range(),
        }

    total_fetched = 0
    total_inserted = 0

    driver = None
    try:
        driver = create_driver(headless=True)

        for idx, username in enumerate(accounts, start=1):
            print(f"{LOG_TAG} [{idx}/{len(accounts)}] Fetching @{username} ...")

            existing_ids: set[str] | None = None
            dup_threshold: int | None = None
            if stop_on_existing_duplicates and duplicates_threshold > 0:
                existing_ids = get_tweet_ids_by_author_in_range(
                    author_username=username,
                    since_date=since_date,
                    until_date=until_date,
                )
                dup_threshold = duplicates_threshold
                print(
                    f"{LOG_TAG} @{username}: {len(existing_ids)} tweets already in archive "
                    f"for [{since_date} -> {until_date}], "
                    f"will stop after {dup_threshold} duplicates"
                )

            try:
                tweets = fetch_tweets_sync(
                    username=username,
                    max_tweets=max_tweets,
                    since_date=since_date,
                    until_date=until_date,
                    driver=driver,
                    max_scrolls=max_scrolls,
                    existing_tweet_ids=existing_ids,
                    duplicates_stop_threshold=dup_threshold,
                )
            except Exception as exc:
                print(f"{LOG_TAG} ERROR while fetching @{username}: {exc}")
                continue

            fetched = len(tweets)
            inserted = insert_tweets(tweets)

            total_fetched += fetched
            total_inserted += inserted

            print(
                f"{LOG_TAG} @{username}: fetched={fetched}, inserted={inserted}, "
                f"duplicates={max(fetched - inserted, 0)}"
            )

            if idx < len(accounts):
                sleep_s = random.uniform(pause_min, pause_max)
                print(f"{LOG_TAG} Pause {sleep_s:.1f}s before next account...")
                time.sleep(sleep_s)

    finally:
        if driver is not None:
            driver.quit()

    result = {
        "accounts_total": len(accounts),
        "tweets_fetched": total_fetched,
        "tweets_inserted": total_inserted,
        "db_total": count_tweets(),
        "db_range": get_date_range(),
    }

    print(f"{LOG_TAG} Done: {result}")
    return result


def run_classify_unclassified(
    since_date: str,
    until_date: str,
    authors: list[str] | None = None,
) -> dict:
    """Classify tweets in the DB that have no signal yet, for a given date range.

    Reads the author list from twitter_collector_settings.json (classifier_settings.authors).
    Processes each author separately so LLM context stays focused on one voice at a time.
    Only tweets with signal_type=NULL are sent to the LLM — already classified rows are skipped.
    Results are written back to the DB via update_tweet_signals().

    Args:
        since_date: inclusive lower bound, "YYYY-MM-DD"
        until_date: inclusive upper bound, "YYYY-MM-DD"

    Returns:
        Summary dict: authors processed, tweets found, tweets classified, tweets updated in DB.
    """
    # --- Load target authors from collector settings ---
    if authors is None:
        collector_cfg = _load_accounts_config()
        authors = collector_cfg.get("classifier_settings", {}).get("authors", [])
    else:
        authors = [a.strip().lstrip("@") for a in authors if a and a.strip()]

    if not authors:
        print(f"{LOG_TAG} No authors found in classifier_settings.authors. Nothing to classify.")
        return {"authors_total": 0, "tweets_found": 0, "tweets_classified": 0, "tweets_updated": 0}

    print(f"{LOG_TAG} classify_unclassified: {len(authors)} authors, range [{since_date} → {until_date}]")

    # --- Fetch all tweets in the date range from DB ---
    # get_tweets_in_range expects datetime objects
    dt_from = datetime.strptime(since_date, "%Y-%m-%d")
    dt_to = datetime.strptime(until_date, "%Y-%m-%d")
    all_tweets_in_range = get_tweets_in_range(dt_from, dt_to)

    # Build a lookup: author_username (lowercased) → list of tweet dicts
    # Lowercase both sides so config casing doesn't have to match DB exactly
    authors_lower = {a.lower(): a for a in authors}
    tweets_by_author: dict[str, list[dict]] = {a: [] for a in authors}

    for tweet in all_tweets_in_range:
        username_lower = (tweet.get("author_username") or "").lower()
        original_author = authors_lower.get(username_lower)
        if original_author is None:
            # Tweet belongs to an author not in our config — skip
            continue
        if tweet.get("signal_type") is not None:
            # Already classified — skip
            continue
        tweets_by_author[original_author].append(tweet)

    total_found = sum(len(v) for v in tweets_by_author.values())
    total_classified = 0
    total_updated = 0

    print(f"{LOG_TAG} Unclassified tweets to process: {total_found}")

    # --- Classify author by author ---
    for author, tweets in tweets_by_author.items():
        if not tweets:
            print(f"{LOG_TAG} @{author}: no unclassified tweets in range, skipping")
            continue

        print(f"{LOG_TAG} @{author}: classifying {len(tweets)} tweet(s)...")

        # classify_tweets mutates each dict in-place,
        # adding signal_type and signal_confidence keys
        classify_tweets(tweets)

        # Count how many were actually labeled by LLM (not fallback empty-text)
        classified = [
            t for t in tweets
            if t.get("signal_type") in {"BULL", "BEAR", "NO_CORRELATION_TO_BTC"}
        ]
        total_classified += len(classified)

        # Write the new signal fields back to the DB rows (matched by tweet_id)
        updated = update_tweet_signals(classified)
        total_updated += updated

        print(f"{LOG_TAG} @{author}: classified={len(classified)}, db_updated={updated}")

    result = {
        "authors_total": len(authors),
        "tweets_found": total_found,
        "tweets_classified": total_classified,
        "tweets_updated": total_updated,
    }
    print(f"{LOG_TAG} classify_unclassified done: {result}")
    return result


if __name__ == "__main__":
    # Step 1: Fetch tweets for enabled accounts
    # run_fetch_only()

    # Step 2: Classify unclassified tweets in range
    run_classify_unclassified("2026-01-01", "2026-04-24", [                "CarpeNoctom",
                "JSeyff",
                "AltcoinPsycho",
                "DavidDuong",
                "TraderMercury",
                "_Checkmatey_",
                "CryptoHayes",
                "rektcapital"])

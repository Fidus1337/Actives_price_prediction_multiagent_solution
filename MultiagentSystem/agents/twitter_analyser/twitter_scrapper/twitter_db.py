"""
SQLite storage for twitter archive.

Single-file database, ready for Docker volume mounting.
DB path configurable via TWITTER_DB_PATH env variable.
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path

# Access to Database
_DEFAULT_PATH = Path(__file__).parent.parent / "twitter_archive.db"
DB_PATH = Path(os.getenv("TWITTER_DB_PATH", str(_DEFAULT_PATH)))

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS tweets (
    tweet_id            TEXT PRIMARY KEY,
    author_username     TEXT NOT NULL,
    author_display_name TEXT,
    text                TEXT,
    created_at          TEXT,
    date                TEXT,
    likes               INTEGER DEFAULT 0,
    retweets            INTEGER DEFAULT 0,
    replies             INTEGER DEFAULT 0,
    views               INTEGER DEFAULT 0,
    is_retweet          INTEGER DEFAULT 0,
    is_reply            INTEGER DEFAULT 0,
    lang                TEXT,
    url                 TEXT,
    signal_type         TEXT,
    signal_confidence   TEXT
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_tweets_date ON tweets (date);
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    # For avoiding conflicts between reading and writing processes 
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables and indexes if they don't exist."""
    with _get_conn() as conn:
        conn.executescript(_CREATE_TABLE + _CREATE_INDEX)
        # Backward-compatible migration for existing DB files.
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(tweets)").fetchall()}
        
        # Adding signal_type and signal_confidence columns
        if "signal_type" not in cols:
            conn.execute("ALTER TABLE tweets ADD COLUMN signal_type TEXT")
        if "signal_confidence" not in cols:
            conn.execute("ALTER TABLE tweets ADD COLUMN signal_confidence TEXT")


def insert_tweets(tweets: list[dict]) -> int:
    """Insert tweets, skip duplicates via INSERT OR IGNORE.

    Returns number of new rows inserted.
    """
    if not tweets:
        return 0
    normalized = []
    
    # The tweet must be with the signal and confidance keys
    for t in tweets:
        item = dict(t)
        item.setdefault("signal_type", None)
        item.setdefault("signal_confidence", None)
        normalized.append(item)
    
    # For every tweet add columns
    with _get_conn() as conn:
        before = conn.execute("SELECT COUNT(*) FROM tweets").fetchone()[0]
        conn.executemany(
            """
            INSERT OR IGNORE INTO tweets
                (tweet_id, author_username, author_display_name, text,
                 created_at, date, likes, retweets, replies, views,
                 is_retweet, is_reply, lang, url, signal_type, signal_confidence)
            VALUES
                (:tweet_id, :author_username, :author_display_name, :text,
                 :created_at, :date, :likes, :retweets, :replies, :views,
                 :is_retweet, :is_reply, :lang, :url, :signal_type, :signal_confidence)
            """,
            normalized,
        )
        after = conn.execute("SELECT COUNT(*) FROM tweets").fetchone()[0]
        return after - before


def count_tweets() -> int:
    with _get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM tweets").fetchone()[0]


def get_all_tweets() -> list[dict]:
    """Return all tweets from archive, newest first."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM tweets ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def update_tweet_signals(tweets: list[dict]) -> int:
    """Update signal fields for existing tweets by tweet_id."""
    if not tweets:
        return 0
    payload = []
    for t in tweets:
        payload.append({
            "tweet_id": t.get("tweet_id"),
            "signal_type": t.get("signal_type"),
            "signal_confidence": t.get("signal_confidence"),
        })

    with _get_conn() as conn:
        conn.executemany(
            """
            UPDATE tweets
            SET signal_type = :signal_type,
                signal_confidence = :signal_confidence
            WHERE tweet_id = :tweet_id
            """,
            payload,
        )
        # For sqlite3, total_changes is per-connection cumulative count.
        return conn.total_changes


def delete_tweets_by_ids(tweet_ids: list[str], chunk_size: int = 900) -> int:
    """Delete tweets by tweet_id list. Returns removed row count."""
    if not tweet_ids:
        return 0

    removed = 0
    with _get_conn() as conn:
        for i in range(0, len(tweet_ids), chunk_size):
            batch = tweet_ids[i:i + chunk_size]
            placeholders = ",".join("?" for _ in batch)
            cur = conn.execute(
                f"DELETE FROM tweets WHERE tweet_id IN ({placeholders})",
                batch,
            )
            if cur.rowcount and cur.rowcount > 0:
                removed += cur.rowcount
    return removed


def delete_tweets_in_range(since_date: str, until_date: str) -> int:
    """Delete tweets where date is within [since_date, until_date] inclusive."""
    with _get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM tweets WHERE date >= ? AND date <= ?",
            (since_date, until_date),
        )
        return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0


def get_date_range() -> str:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT MIN(date), MAX(date) FROM tweets WHERE date IS NOT NULL"
        ).fetchone()
        if row[0]:
            return f"{row[0]} -> {row[1]}"
        return "empty"


def clear_signals_in_range(
    since_date: str,
    until_date: str,
    authors: list[str],
) -> int:
    """Set signal_type and signal_confidence to NULL for tweets in [since_date, until_date]
    whose author_username is in the given authors list.

    Returns number of rows updated.
    """
    if not authors:
        return 0

    placeholders = ",".join("?" for _ in authors)
    with _get_conn() as conn:
        cur = conn.execute(
            f"""
            UPDATE tweets
            SET signal_type = NULL,
                signal_confidence = NULL
            WHERE date >= ?
              AND date <= ?
              AND author_username IN ({placeholders})
            """,
            (since_date, until_date, *authors),
        )
        return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0


def get_latest_date() -> str | None:
    """Returns MAX(date) from archive, or None if empty."""
    init_db()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT MAX(date) FROM tweets WHERE date IS NOT NULL"
        ).fetchone()
    return row[0] if row and row[0] else None


def get_tweet_ids_by_author_in_range(
    author_username: str,
    since_date: str | None = None,
    until_date: str | None = None,
) -> set[str]:
    """Return tweet_id set for the given author within an optional [since, until] range.

    Case-insensitive match on author_username. Bounds are inclusive. If both dates
    are None, returns all tweet_ids for the author.
    """
    author = (author_username or "").strip().lstrip("@").lower()
    if not author:
        return set()

    query = "SELECT tweet_id FROM tweets WHERE LOWER(author_username) = ?"
    params: list = [author]
    if since_date:
        query += " AND date >= ?"
        params.append(since_date)
    if until_date:
        query += " AND date <= ?"
        params.append(until_date)

    with _get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return {row["tweet_id"] for row in rows if row["tweet_id"]}


def get_tweets_in_range(dt_from: datetime, dt_to: datetime) -> list[dict]:
    """Return tweets within [dt_from, dt_to], newest first.

    Both bounds are inclusive (>= dt_from AND <= dt_to).
    """
    from_str = dt_from.strftime("%Y-%m-%d")
    to_str = dt_to.strftime("%Y-%m-%d")
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM tweets WHERE date >= ? AND date <= ? ORDER BY created_at DESC",
            (from_str, to_str),
        ).fetchall()
        return [dict(r) for r in rows]


# Auto-init on import
init_db()

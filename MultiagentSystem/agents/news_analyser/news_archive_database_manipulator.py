"""
SQLite persistence layer for the news archive.

Database: news_archive.db (same directory as this file)
Table:    news_articles

One-time migration from JSON:
    python -m MultiagentSystem.agents.news_analyser.news_archive_database_manipulator
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "news_archive.db"
JSON_PATH = Path(__file__).parent / "news_archive.json"


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS news_articles (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                article_title        TEXT    NOT NULL,
                article_release_time INTEGER NOT NULL,
                article_content      TEXT,
                article_picture      TEXT,
                article_description  TEXT,
                source_name          TEXT,
                source_website_logo  TEXT,
                category             TEXT,
                strength             TEXT,
                date                 TEXT,
                UNIQUE(article_title, article_release_time)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_date ON news_articles (date)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ts ON news_articles (article_release_time)"
        )
        conn.commit()


def insert_articles(articles: list[dict]) -> int:
    """INSERT OR IGNORE all articles. Returns count of newly inserted rows."""
    inserted = 0
    with sqlite3.connect(DB_PATH) as conn:
        for a in articles:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO news_articles (
                    article_title, article_release_time, article_content,
                    article_picture, article_description, source_name,
                    source_website_logo, category, strength, date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    a.get("article_title"),
                    a.get("article_release_time"),
                    a.get("article_content"),
                    a.get("article_picture"),
                    a.get("article_description"),
                    a.get("source_name"),
                    a.get("source_website_logo"),
                    a.get("category"),
                    a.get("strength"),
                    a.get("date"),
                ),
            )
            inserted += cursor.rowcount
        conn.commit()
    return inserted


def update_classifications(articles: list[dict]) -> int:
    """UPDATE category and strength for each article identified by (title, timestamp).

    Returns count of rows actually changed.
    """
    updated = 0
    with sqlite3.connect(DB_PATH) as conn:
        for a in articles:
            cursor = conn.execute(
                """
                UPDATE news_articles
                SET category = ?, strength = ?
                WHERE article_title = ? AND article_release_time = ?
                """,
                (
                    a.get("category"),
                    a.get("strength"),
                    a.get("article_title"),
                    a.get("article_release_time"),
                ),
            )
            updated += cursor.rowcount
        conn.commit()
    return updated


def load_articles_in_range(dt_from: datetime, dt_to: datetime) -> list[dict]:
    """SELECT articles where article_release_time in [dt_from, dt_to].

    Returns list[dict] ordered by article_release_time DESC.
    """
    ts_from = int(dt_from.timestamp() * 1000)
    ts_to = int(dt_to.timestamp() * 1000)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT * FROM news_articles
            WHERE article_release_time >= ? AND article_release_time <= ?
            ORDER BY article_release_time DESC
            """,
            (ts_from, ts_to),
        ).fetchall()
    return [dict(row) for row in rows]


def load_all_articles() -> list[dict]:
    """Return all articles ordered by article_release_time DESC."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM news_articles ORDER BY article_release_time DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_unclassified_articles() -> list[dict]:
    """Return articles where category IS NULL or category = 'unclassified'."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT * FROM news_articles
            WHERE category IS NULL OR category = 'unclassified'
            ORDER BY article_release_time DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def reset_all_classifications() -> None:
    """Set category and strength to NULL for every row."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE news_articles SET category = NULL, strength = NULL")
        conn.commit()


def get_db_stats() -> dict:
    """Returns {count, date_range} for logging."""
    with sqlite3.connect(DB_PATH) as conn:
        count = conn.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]
        row = conn.execute(
            "SELECT MIN(date), MAX(date) FROM news_articles"
        ).fetchone()
    date_range = f"{row[0]} -> {row[1]}" if row[0] else "empty"
    return {"count": count, "date_range": date_range}


def get_latest_date() -> str | None:
    """Returns MAX(date) from archive, or None if empty."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT MAX(date) FROM news_articles").fetchone()
    return row[0] if row and row[0] else None


def migrate_json_to_db() -> None:
    """One-time migration from news_archive.json to SQLite."""
    if not JSON_PATH.exists():
        print(f"[migrate] {JSON_PATH.name} not found — nothing to migrate")
        return

    articles = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    print(f"[migrate] Loaded {len(articles)} articles from {JSON_PATH.name}")

    inserted = insert_articles(articles)
    skipped = len(articles) - inserted
    print(f"[migrate] Inserted {inserted} new articles (skipped {skipped} duplicates)")
    print(f"[migrate] Done → {DB_PATH.name}")


if __name__ == "__main__":
    init_db()
    migrate_json_to_db()

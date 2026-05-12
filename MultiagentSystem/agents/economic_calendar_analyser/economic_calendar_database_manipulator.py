import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "calendar_archive.db"
JSON_PATH = Path(__file__).parent / "calendar_archive.json"


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL")  # safe mode for parallel access
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calendar_events(
                id                     INTEGER PRIMARY KEY AUTOINCREMENT,
                calendar_name          TEXT    NOT NULL,
                publish_timestamp      INTEGER NOT NULL,
                country_code           TEXT,
                country_name           TEXT,
                importance_level       INTEGER DEFAULT 0,
                data_effect            TEXT,
                published_value        TEXT,
                forecast_value         TEXT,
                previous_value         TEXT,
                revised_previous_value TEXT,
                has_exact_publish_time INTEGER DEFAULT 0,
                date                   TEXT,
                UNIQUE(calendar_name, publish_timestamp)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON calendar_events (date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON calendar_events (publish_timestamp)")
        conn.commit()


def insert_events(events: list[dict]) -> int:
    """INSERT OR IGNORE all events. Returns count of newly inserted."""
    inserted = 0
    with sqlite3.connect(DB_PATH) as conn:
        for e in events:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO calendar_events (
                    calendar_name, publish_timestamp, country_code, country_name,
                    importance_level, data_effect, published_value, forecast_value,
                    previous_value, revised_previous_value, has_exact_publish_time, date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    e.get("calendar_name"),
                    e.get("publish_timestamp"),
                    e.get("country_code"),
                    e.get("country_name"),
                    e.get("importance_level", 0),
                    e.get("data_effect"),
                    e.get("published_value"),
                    e.get("forecast_value"),
                    e.get("previous_value"),
                    e.get("revised_previous_value"),
                    e.get("has_exact_publish_time", 0),
                    e.get("date"),
                ),
            )
            inserted += cursor.rowcount
        conn.commit()
    return inserted


def load_events_in_range(dt_from: datetime, dt_to: datetime) -> list[dict]:
    """SELECT events where publish_timestamp in [dt_from, dt_to]. Returns list[dict]."""
    ts_from = int(dt_from.timestamp() * 1000)
    ts_to = int(dt_to.timestamp() * 1000)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT * FROM calendar_events
            WHERE publish_timestamp >= ? AND publish_timestamp <= ?
            ORDER BY publish_timestamp DESC
            """,
            (ts_from, ts_to),
        ).fetchall()
    return [dict(row) for row in rows]


def get_db_stats() -> dict:
    """Returns {count, date_range} for logging in collector."""
    with sqlite3.connect(DB_PATH) as conn:
        count = conn.execute("SELECT COUNT(*) FROM calendar_events").fetchone()[0]
        row = conn.execute("SELECT MIN(date), MAX(date) FROM calendar_events").fetchone()
    date_range = f"{row[0]} -> {row[1]}" if row[0] else "empty"
    return {"count": count, "date_range": date_range}


def get_latest_date() -> str | None:
    """Returns MAX(date) from archive, or None if empty."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT MAX(date) FROM calendar_events").fetchone()
    return row[0] if row and row[0] else None


def migrate_json_to_db():
    """One-time migration from calendar_archive.json to SQLite."""
    if not JSON_PATH.exists():
        print(f"[migrate] {JSON_PATH.name} not found — nothing to migrate")
        return

    events = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    print(f"[migrate] Loaded {len(events)} events from {JSON_PATH.name}")

    inserted = insert_events(events)
    print(f"[migrate] Inserted {inserted} new events (skipped {len(events) - inserted} duplicates)")
    print(f"[migrate] Done → {DB_PATH.name}")


if __name__ == "__main__":
    init_db()
    migrate_json_to_db()

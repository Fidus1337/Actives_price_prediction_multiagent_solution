"""SQLite-кэш предсказаний мультиагента, ключ — (config_hash, forecast_start_date).

База: cached_configs_predictions.db (эта же директория).

Раскладка (по таблице на конфиг):
- configs_registry : реестр конфигов (hash -> config_json + метаданные)
- cache_meta       : служебные key/value (резерв; настройка N живёт в configs/cache_settings.json)
- cfg_<hash>       : по таблице на конфиг, одна строка на forecast_start_date

Кэшируется ТОЛЬКО прогноз (y_predict / confidence / выходы агентов / summary / reasoning /
risks) — без y_true. y_true пересчитывается заново через add_y_true на объединённом наборе,
иначе для свежих дат (горизонт ещё не прошёл) в кэш навсегда попал бы устаревший y_true=None.

Стиль слоя — как в
MultiagentSystem/agents/news_analyser/news_archive_database_manipulator.py (sqlite3, WAL).
"""

import hashlib
import json
import re
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "cached_configs_predictions.db"

# config_hash = первые 16 hex-символов sha256 (64 бита — достаточно для уникальности кэша).
_HASH_LEN = 16
_HASH_RE = re.compile(r"^[0-9a-f]{%d}$" % _HASH_LEN)

# Зарезервированный блок agent_settings, который всегда участвует в хэше, даже если
# не перечислен в agent_envolved_in_prediction.
_VALIDATOR_KEY = "verdicts_validator"


def _json_default(obj):
    """Делает numpy-скаляры/массивы JSON-сериализуемыми без импорта numpy."""
    if hasattr(obj, "item"):      # numpy scalar (np.float64, np.bool_, ...)
        return obj.item()
    if hasattr(obj, "tolist"):    # numpy array
        return obj.tolist()
    return str(obj)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = Path(db_path)

    # --- connection -------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
            (table,),
        ).fetchone()
        return row is not None

    # --- schema -----------------------------------------------------------
    def create_database(self) -> None:
        self.init_db()

    def init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS configs_registry (
                    config_hash TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    table_name  TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def database_already_exists(self) -> bool:
        return self.db_path.exists()

    # --- hashing ----------------------------------------------------------
    def convert_config_json_into_hash(self, config: dict) -> str:
        """Хэш по полям, ОПРЕДЕЛЯЮЩИМ прогноз.

        Включаются только: horizon, neutral_threshold, отсортированный
        agent_envolved_in_prediction и блоки agent_settings для задействованных
        агентов + verdicts_validator. Всё остальное (forecast_start_date,
        n_last_dates, force_recompute, debug_save_prompts) на идентичность не влияет.
        """
        involved = sorted(config.get("agent_envolved_in_prediction", []) or [])
        all_settings = config.get("agent_settings", {}) or {}
        keep = set(involved) | {_VALIDATOR_KEY}
        settings = {k: v for k, v in all_settings.items() if k in keep}
        subset = {
            "horizon": config.get("horizon"),
            "neutral_threshold": config.get("neutral_threshold"),
            "agent_envolved_in_prediction": involved,
            "agent_settings": settings,
        }
        canonical = json.dumps(
            subset,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=_json_default,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:_HASH_LEN]

    # --- table naming -----------------------------------------------------
    @staticmethod
    def _validate_hash(config_hash: str) -> str:
        if not _HASH_RE.match(config_hash or ""):
            raise ValueError(f"Invalid config_hash: {config_hash!r}")
        return config_hash

    def _table_name(self, config_hash: str) -> str:
        # config_hash валидируется как [0-9a-f]{16}, поэтому интерполяция в SQL безопасна.
        return "cfg_" + self._validate_hash(config_hash)

    # --- registry / per-config table --------------------------------------
    def _ensure_config_table(self, conn: sqlite3.Connection, config_hash: str, config_json) -> str:
        table = self._table_name(config_hash)
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS "{table}" (
                forecast_start_date TEXT PRIMARY KEY,
                prediction_json     TEXT NOT NULL,
                created_at          TEXT NOT NULL
            )
            """
        )
        cfg_str = (
            config_json
            if isinstance(config_json, str)
            else json.dumps(config_json, ensure_ascii=False, default=_json_default)
        )
        now = _utcnow_iso()
        conn.execute(
            """
            INSERT INTO configs_registry (config_hash, config_json, table_name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(config_hash) DO UPDATE SET
                config_json = excluded.config_json,
                updated_at  = excluded.updated_at
            """,
            (config_hash, cfg_str, table, now, now),
        )
        return table

    def is_config_already_exists(self, config_hash) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM configs_registry WHERE config_hash = ?",
                (config_hash,),
            ).fetchone()
        finally:
            conn.close()
        return row is not None

    # --- read / write predictions -----------------------------------------
    def get_cached_prediction(self, config_hash, date_str) -> dict | None:
        """Прогноз по (config_hash, date) или None, если нет таблицы/строки."""
        try:
            table = self._table_name(config_hash)
        except ValueError:
            return None
        conn = self._connect()
        try:
            if not self._table_exists(conn, table):
                return None
            row = conn.execute(
                f'SELECT prediction_json FROM "{table}" WHERE forecast_start_date = ?',
                (date_str,),
            ).fetchone()
        finally:
            conn.close()
        return json.loads(row[0]) if row else None

    def upsert_prediction(self, config_hash, config_json, date_str, row_dict) -> None:
        """Сохранить строку-прогноз (без y_true) для (config_hash, date)."""
        payload = json.dumps(row_dict, ensure_ascii=False, default=_json_default)
        conn = self._connect()
        try:
            table = self._ensure_config_table(conn, config_hash, config_json)
            conn.execute(
                f"""
                INSERT INTO "{table}" (forecast_start_date, prediction_json, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(forecast_start_date) DO UPDATE SET
                    prediction_json = excluded.prediction_json,
                    created_at      = excluded.created_at
                """,
                (date_str, payload, _utcnow_iso()),
            )
            conn.commit()
        finally:
            conn.close()

    # --- reset ------------------------------------------------------------
    def reset_by_config(self, config_hash) -> bool:
        """Удалить таблицу одного конфига и его строку в реестре. True, если что-то удалили."""
        table = self._table_name(config_hash)
        conn = self._connect()
        try:
            existed = self._table_exists(conn, table) or bool(
                conn.execute(
                    "SELECT 1 FROM configs_registry WHERE config_hash = ?", (config_hash,)
                ).fetchone()
            )
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')
            conn.execute("DELETE FROM configs_registry WHERE config_hash = ?", (config_hash,))
            conn.commit()
        finally:
            conn.close()
        return existed

    def full_reset_database(self) -> int:
        """Удалить все таблицы cfg_* и очистить реестр. Возвращает число удалённых конфигов."""
        conn = self._connect()
        try:
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'cfg\\_%' ESCAPE '\\'"
                ).fetchall()
            ]
            for t in tables:
                conn.execute(f'DROP TABLE IF EXISTS "{t}"')
            conn.execute("DELETE FROM configs_registry")
            conn.commit()
        finally:
            conn.close()
        return len(tables)

    # --- retention --------------------------------------------------------
    def clean_old_records(self, save_n_last_days) -> int:
        """Скользящее окно: по всем cfg_* удалить строки старше today(UTC) - N дней.

        forecast_start_date хранится как 'YYYY-MM-DD', поэтому лексикографическое
        сравнение совпадает с хронологическим. date('now', ...) в SQLite — UTC.
        Возвращает суммарное число удалённых строк.
        """
        cutoff_modifier = f"-{int(save_n_last_days)} days"
        deleted = 0
        conn = self._connect()
        try:
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'cfg\\_%' ESCAPE '\\'"
                ).fetchall()
            ]
            for t in tables:
                cur = conn.execute(
                    f"DELETE FROM \"{t}\" WHERE forecast_start_date < date('now', ?)",
                    (cutoff_modifier,),
                )
                deleted += cur.rowcount
            conn.commit()
        finally:
            conn.close()
        return deleted

    # --- introspection (пункт 7) ------------------------------------------
    def get_config_date_count(self, config_hash) -> int:
        try:
            table = self._table_name(config_hash)
        except ValueError:
            return 0
        conn = self._connect()
        try:
            if not self._table_exists(conn, table):
                return 0
            return conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        finally:
            conn.close()

    def get_config_info(self, config_hash) -> dict | None:
        """Сводка по одному конфигу или None, если он неизвестен реестру."""
        conn = self._connect()
        try:
            reg = conn.execute(
                "SELECT config_hash, table_name, created_at FROM configs_registry WHERE config_hash = ?",
                (config_hash,),
            ).fetchone()
            if not reg:
                return None
            return self._summarize(conn, *reg)
        finally:
            conn.close()

    def list_configs(self) -> list[dict]:
        conn = self._connect()
        try:
            regs = conn.execute(
                "SELECT config_hash, table_name, created_at FROM configs_registry ORDER BY created_at DESC"
            ).fetchall()
            return [self._summarize(conn, *r) for r in regs]
        finally:
            conn.close()

    def _summarize(self, conn: sqlite3.Connection, config_hash: str, table: str, created_at: str) -> dict:
        if self._table_exists(conn, table):
            count, dmin, dmax = conn.execute(
                f'SELECT COUNT(*), MIN(forecast_start_date), MAX(forecast_start_date) FROM "{table}"'
            ).fetchone()
        else:
            count, dmin, dmax = 0, None, None
        return {
            "config_hash": config_hash,
            "cached_dates_count": count,
            "oldest_date": dmin,
            "newest_date": dmax,
            "created_at": created_at,
        }

    # --- misc -------------------------------------------------------------
    def todays_date(self) -> str:
        return datetime.now(timezone.utc).date().isoformat()


if __name__ == "__main__":
    db = Database()
    db.init_db()
    print(f"[cache] Initialized {db.db_path}")
    print(f"[cache] Configs: {db.list_configs()}")

"""Persistence for the predictions-cache settings (configs/cache_settings.json).

Same pattern as api/scheduler.py:get_settings/save_settings. The default for
save_n_last_days is seeded from the CACHED_PREDICTIONS_SAVE_N_LAST_DAYS env var
(dev.env) when the JSON file is absent; the PUT /api/cache/settings endpoint then
persists overrides to the JSON file (built into the image), so runtime changes
survive without a restart.

Kept in its own module (not the router) so both the cache router and the
predictions endpoint can import it without a circular import.
"""

import json
import os
from pathlib import Path

from api.schemas import CachePredictionCacheSettings

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "cache_settings.json"


def _default_settings() -> CachePredictionCacheSettings:
    env_n = os.getenv("CACHED_PREDICTIONS_SAVE_N_LAST_DAYS")
    if env_n:
        try:
            return CachePredictionCacheSettings(save_n_last_days=int(env_n))
        except (ValueError, TypeError):
            pass
    return CachePredictionCacheSettings()


def get_cache_settings() -> CachePredictionCacheSettings:
    """Load cache settings from JSON, falling back to env/default if absent."""
    if CONFIG_PATH.exists():
        return CachePredictionCacheSettings(**json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    return _default_settings()


def save_cache_settings(settings: CachePredictionCacheSettings) -> None:
    """Persist cache settings to JSON."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(settings.model_dump_json(indent=2), encoding="utf-8")

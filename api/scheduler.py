"""Daily auto-collection scheduler.

Fires once per day at 00:00 UTC by default — the moment a new Bybit daily candle
opens — and triggers incremental data collection for the configured agents.

Settings live in ``configs/collect_scheduler_settings.json`` (same pattern as
``configs/multiagent_config.json``) and can be changed at runtime via the
``/api/system/change_collect_scheduler_settings`` endpoint, which rewrites that
file and reschedules the job without a restart.

The scheduler is an ``AsyncIOScheduler`` running on uvicorn's event loop, so the
job shares the per-agent ``asyncio.Lock``s used by the manual collection endpoint.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from api.routers.multiagent_predictions import collect_agent_data_core
from api.schemas import CollectAgentDataRequest, CollectSchedulerSettings

logger = logging.getLogger("daily_collection")

JOB_ID = "daily_collection"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "collect_scheduler_settings.json"


def get_settings() -> CollectSchedulerSettings:
    """Load scheduler settings from JSON, falling back to defaults if absent."""
    if CONFIG_PATH.exists():
        return CollectSchedulerSettings(**json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    return CollectSchedulerSettings()


def save_settings(settings: CollectSchedulerSettings) -> None:
    """Persist scheduler settings to JSON."""
    CONFIG_PATH.write_text(settings.model_dump_json(indent=2), encoding="utf-8")


def build_trigger(settings: CollectSchedulerSettings) -> CronTrigger:
    """Cron trigger forced to UTC, independent of container/host timezone."""
    return CronTrigger(hour=settings.hour, minute=settings.minute, timezone="UTC")


def apply_settings(scheduler: AsyncIOScheduler, settings: CollectSchedulerSettings) -> str | None:
    """Reschedule the job to the new time and pause/resume it per ``enabled``.

    Returns the next run time (ISO 8601, UTC) or ``None`` when the job is paused.
    """
    scheduler.reschedule_job(JOB_ID, trigger=build_trigger(settings))
    if settings.enabled:
        scheduler.resume_job(JOB_ID)
    else:
        scheduler.pause_job(JOB_ID)
    job = scheduler.get_job(JOB_ID)
    return job.next_run_time.isoformat() if job and job.next_run_time else None


async def collect_daily_job() -> None:
    """Build the collection request from current settings and run it.

    Settings are read fresh on each fire, so author/agent/since_days changes take
    effect on the next run even without rescheduling. Dates are computed in UTC:
    ``twitter_since_date = today - since_days``, ``twitter_until_date = today``.
    """
    settings = get_settings()
    today = datetime.now(timezone.utc).date()
    request = CollectAgentDataRequest(
        agents=settings.agents,
        twitter_authors=settings.twitter_authors,
        twitter_since_date=(today - timedelta(days=settings.since_days)).isoformat(),
        twitter_until_date=today.isoformat(),
    )
    logger.info("Daily collection starting: %s", request.model_dump())
    try:
        response = await collect_agent_data_core(request)
        logger.info("Daily collection done: %s", response.model_dump())
    except Exception:
        # Never let a failed run kill the scheduler — it retries on the next cycle.
        logger.exception("Daily collection failed")


def create_scheduler() -> AsyncIOScheduler:
    """Build the scheduler with the daily collection job registered (not yet started)."""
    settings = get_settings()
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        collect_daily_job,
        build_trigger(settings),
        id=JOB_ID,
        misfire_grace_time=3600,  # still run if the container started shortly after 00:00
        coalesce=True,
        max_instances=1,
    )
    return scheduler

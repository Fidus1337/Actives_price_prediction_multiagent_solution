"""Endpoints to read and change the daily auto-collection scheduler settings."""

from fastapi import APIRouter, HTTPException, Request

from api.scheduler import JOB_ID, apply_settings, get_settings, save_settings
from api.schemas import (
    ChangeCollectSchedulerSettingsRequest,
    CollectSchedulerSettingsResponse,
)

router = APIRouter(prefix="/api/system", tags=["collect_scheduler"])

_VALID_AGENTS = {"news_analyser", "economic_calendar_analyser", "twitter_analyser"}


def _next_run(scheduler) -> str | None:
    job = scheduler.get_job(JOB_ID) if scheduler else None
    return job.next_run_time.isoformat() if job and job.next_run_time else None


@router.get(
    "/collect_scheduler_settings",
    response_model=CollectSchedulerSettingsResponse,
    summary="Read the daily collection scheduler settings",
    description="Returns current settings from configs/collect_scheduler_settings.json and the next scheduled run (UTC).",
)
async def collect_scheduler_settings(request: Request) -> CollectSchedulerSettingsResponse:
    return CollectSchedulerSettingsResponse(
        settings=get_settings(),
        next_run_time=_next_run(request.app.state.scheduler),
    )


@router.post(
    "/change_collect_scheduler_settings",
    response_model=CollectSchedulerSettingsResponse,
    summary="Change the daily collection scheduler settings",
    description=(
        "Partial update — send only the fields you want to change. Persists to "
        "configs/collect_scheduler_settings.json and reschedules the job live "
        "(no restart). Changing hour/minute moves the trigger; enabled=false "
        "pauses it. agents/twitter_authors/since_days take effect on the next run."
    ),
)
async def change_collect_scheduler_settings(
    body: ChangeCollectSchedulerSettingsRequest, request: Request
) -> CollectSchedulerSettingsResponse:
    updated = get_settings().model_copy(update=body.model_dump(exclude_none=True))

    unknown = set(updated.agents) - _VALID_AGENTS
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown agents: {sorted(unknown)}")

    scheduler = request.app.state.scheduler
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler is not running")

    save_settings(updated)
    next_run = apply_settings(scheduler, updated)
    return CollectSchedulerSettingsResponse(settings=updated, next_run_time=next_run)

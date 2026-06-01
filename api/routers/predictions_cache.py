"""Endpoints to inspect and manage the multiagent predictions cache.

Backed by Database_of_cached_results_for_predictions/cached_configs_predictions.db.
Destructive operations (clear config / clear all) and settings changes are guarded
against a concurrently running prediction via multiagent_predictions._prediction_lock.
"""

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from Database_of_cached_results_for_predictions.predictions_database import Database
from api.cache_settings import get_cache_settings, save_cache_settings
from api.routers.multiagent_predictions import _prediction_lock
from api.schemas import (
    CacheClearResponse,
    CacheConfigInfo,
    CacheConfigsResponse,
    CacheHashResponse,
    CachePredictionCacheSettings,
    ChangeCachePredictionCacheSettingsRequest,
    MultiagentPredictionsRequest,
)

router = APIRouter(prefix="/api/cache", tags=["predictions_cache"])


def _guard_no_prediction_running() -> None:
    if _prediction_lock.locked():
        raise HTTPException(
            status_code=409,
            detail="A multiagent prediction is currently running; retry once it finishes",
        )


@router.get(
    "/settings",
    response_model=CachePredictionCacheSettings,
    summary="Read predictions-cache settings",
    description="Returns current settings from configs/cache_settings.json (or the dev.env default if absent).",
)
async def cache_settings() -> CachePredictionCacheSettings:
    return get_cache_settings()


@router.put(
    "/settings",
    response_model=CachePredictionCacheSettings,
    summary="Change predictions-cache settings",
    description=(
        "Partial update — send only the fields you want to change. Persists to "
        "configs/cache_settings.json and applies the new retention window immediately "
        "(prunes entries older than today-N). 409 while a prediction is running."
    ),
)
async def change_cache_settings(
    body: ChangeCachePredictionCacheSettingsRequest,
) -> CachePredictionCacheSettings:
    _guard_no_prediction_running()
    updated = get_cache_settings().model_copy(update=body.model_dump(exclude_none=True))
    save_cache_settings(updated)
    await run_in_threadpool(Database().clean_old_records, updated.save_n_last_days)
    return updated


@router.get(
    "/configs",
    response_model=CacheConfigsResponse,
    summary="List cached configs with their cached-date counts",
    description="Returns every config known to the cache registry with the number of cached forecast dates and their date range.",
)
async def list_cache_configs() -> CacheConfigsResponse:
    rows = await run_in_threadpool(Database().list_configs)
    return CacheConfigsResponse(
        total_configs=len(rows),
        configs=[CacheConfigInfo(**r) for r in rows],
    )


@router.get(
    "/configs/{config_hash}",
    response_model=CacheConfigInfo,
    summary="Cached-date count and range for one config",
    description="Returns the number of cached forecast dates (and their range) for the given config_hash. 404 if unknown.",
)
async def get_cache_config(config_hash: str) -> CacheConfigInfo:
    info = await run_in_threadpool(Database().get_config_info, config_hash)
    if info is None:
        raise HTTPException(status_code=404, detail=f"No cached config with hash {config_hash!r}")
    return CacheConfigInfo(**info)


@router.delete(
    "/configs/{config_hash}",
    response_model=CacheClearResponse,
    summary="Clear the cache for one config",
    description="Drops the per-config table and its registry row. 404 if unknown, 409 while a prediction is running.",
)
async def clear_cache_config(config_hash: str) -> CacheClearResponse:
    _guard_no_prediction_running()
    try:
        cleared = await run_in_threadpool(Database().reset_by_config, config_hash)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not cleared:
        raise HTTPException(status_code=404, detail=f"No cached config with hash {config_hash!r}")
    return CacheClearResponse(
        cleared=True,
        dropped_configs=1,
        detail=f"Cleared cache for config {config_hash}",
    )


@router.delete(
    "",
    response_model=CacheClearResponse,
    summary="Clear the entire predictions cache",
    description="Drops all per-config tables and empties the registry. 409 while a prediction is running.",
)
async def clear_cache_all() -> CacheClearResponse:
    _guard_no_prediction_running()
    dropped = await run_in_threadpool(Database().full_reset_database)
    return CacheClearResponse(
        cleared=dropped > 0,
        dropped_configs=dropped,
        detail=f"Dropped {dropped} cached config(s)",
    )


@router.post(
    "/hash",
    response_model=CacheHashResponse,
    summary="Compute the config_hash for a prediction config",
    description=(
        "Returns the config_hash for the supplied body (same shape as "
        "/api/multiagent_predictions). Use it to target /api/cache/configs/{config_hash}. "
        "forecast_start_date / n_last_dates / force_recompute do not affect the hash."
    ),
)
async def compute_cache_hash(request: MultiagentPredictionsRequest) -> CacheHashResponse:
    config = request.model_dump(exclude={"n_last_dates", "force_recompute"})
    config_hash = Database().convert_config_json_into_hash(config)
    return CacheHashResponse(config_hash=config_hash)

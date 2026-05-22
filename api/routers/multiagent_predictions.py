"""Multiagent prediction endpoints router."""

import asyncio
import os
from fastapi import APIRouter, Body, HTTPException
from fastapi.concurrency import run_in_threadpool
import pandas as pd

from MultiagentSystem.multiagent_predictions_module import make_prediction_for_last_N_days, add_y_true
from MultiagentSystem.multiagent_system_main import app as multiagent_app
from MultiagentSystem.agents.news_analyser.news_collector import collect_news
from MultiagentSystem.agents.news_analyser.news_archive_database_manipulator import get_latest_date as news_get_latest_date
from MultiagentSystem.agents.economic_calendar_analyser.calendar_collector import collect_calendar_events
from MultiagentSystem.agents.economic_calendar_analyser.economic_calendar_database_manipulator import get_latest_date as calendar_get_latest_date
from MultiagentSystem.agents.twitter_analyser.twitter_scrapper.twitter_db import get_latest_date as twitter_get_latest_date
from MultiagentSystem.agents.twitter_analyser.full_scrapping_pipeline import (
    run_fetch_only as twitter_fetch,
    run_classify_unclassified as twitter_classify,
)
from MultiagentSystem.agents.twitter_analyser.twitter_scrapper.chrome_login_before_scrapping import (
    check_twitter_auth,
    save_cookies_from_upload,
)
from api.schemas import (
    MultiagentPredictionsRequest,
    MultiagentPredictionsResponse,
    MultiagentSinglePrediction,
    AgentPredictionDetail,
    PredictionMetrics,
    CollectAgentDataRequest,
    CollectAgentDataResponse,
    CollectAgentDataResult,
    AgentsDataStatusResponse,
    TwitterAuthStatusResponse,
    TwitterCookiesUploadRequest,
    TwitterCookiesUploadResponse,
)


_BOOL_PRED_MAP = {True: 1, False: 0}


def _agent_short_name(agent_name: str) -> str:
    return agent_name.replace("agent_for_", "").replace("agent_for_analysing_", "")


def _compute_metrics(predictions: list[MultiagentSinglePrediction]) -> PredictionMetrics:
    tp = tn = fp = fn = 0
    for p in predictions:
        if p.y_true is None or p.y_prediction is None:
            continue
        if   p.y_prediction == 1 and p.y_true == 1: tp += 1
        elif p.y_prediction == 0 and p.y_true == 0: tn += 1
        elif p.y_prediction == 1 and p.y_true == 0: fp += 1
        elif p.y_prediction == 0 and p.y_true == 1: fn += 1
    evaluable = tp + tn + fp + fn
    accuracy  = (tp + tn) / evaluable if evaluable        else None
    precision = tp / (tp + fp)        if (tp + fp)        else None
    recall    = tp / (tp + fn)        if (tp + fn)        else None
    return PredictionMetrics(
        evaluable_dates=evaluable,
        skipped_dates=len(predictions) - evaluable,
        tp=tp, tn=tn, fp=fp, fn=fn,
        accuracy=accuracy, precision=precision, recall=recall,
    )


router = APIRouter(prefix="/api", tags=["multiagent_predictions"])

_prediction_lock = asyncio.Lock()

_collection_locks: dict[str, asyncio.Lock] = {
    "news_analyser": asyncio.Lock(),
    "economic_calendar_analyser": asyncio.Lock(),
    "twitter_analyser": asyncio.Lock(),
}


@router.post(
    "/multiagent_predictions",
    response_model=MultiagentPredictionsResponse,
    summary="Run multiagent predictions",
    description=(
        "Runs the multiagent system for last N eligible dates using a request body "
        "shaped like multiagent_config.json. "
        "Top-level fields: forecast_start_date, horizon, n_last_dates, "
        "neutral_threshold, agent_envolved_in_prediction, agent_settings. "
        "agent_settings accepts per-agent blocks keyed by agent name (e.g. "
        "agent_for_analysing_tech_indicators, agent_for_twitter_analysis, "
        "agent_for_news_analysis, agent_for_economic_calendar_analysis, "
        "agent_for_analysing_onchain_indicators) plus the reserved "
        "'verdicts_validator' block. Each block may override llm_model, "
        "system_prompt_file, window_to_analysis, base_feats, decay_rate, "
        "decay_start_day, initial_weight, authors — anything the agent's "
        "runtime reads from state['config']. Unknown keys inside a block are "
        "forwarded to the agent as-is (agent_settings is dict[str, dict[str, Any]])."
    ),
)
async def multiagent_predictions(request: MultiagentPredictionsRequest) -> MultiagentPredictionsResponse:
    """Run multiagent predictions using request config.

    The request body is passed through to make_prediction_for_last_N_days as
    a plain dict (n_last_dates stripped). agent_settings is schema-free by
    design, so adding new agent keys or per-agent overrides (e.g. llm_model,
    verdicts_validator) requires no changes here — it propagates to the
    LangGraph nodes via state['config'].
    """
    if _prediction_lock.locked():
        raise HTTPException(status_code=409, detail="Multiagent prediction is already running")

    collecting = sorted(name for name, lock in _collection_locks.items() if lock.locked())
    if collecting:
        raise HTTPException(
            status_code=409,
            detail=f"Data collection in progress for {collecting}; predictions are paused until it finishes",
        )

    async with _prediction_lock:
        try:
            config = request.model_dump(exclude={"n_last_dates"})
            results_df = await run_in_threadpool(
                make_prediction_for_last_N_days,
                multiagent_app,
                config,
                request.n_last_dates,
            )
            results_df = add_y_true(results_df, request.horizon)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to run multiagent predictions: {exc}") from exc

    _DIRECTION_MAP = {"LONG": 1, "SHORT": 0}
    predictions: list[MultiagentSinglePrediction] = []

    for _, row in results_df.iterrows():
        raw_date = row["forecast_start_date"]
        if hasattr(raw_date, "strftime"):
            date_value = raw_date.strftime("%Y-%m-%d")
        else:
            date_value = str(raw_date)

        raw_price = row.get("btc_bybit_close_price")
        base_price = float(raw_price) if raw_price is not None and pd.notna(raw_price) else None

        agents_block: dict[str, AgentPredictionDetail] = {}
        for agent_name in request.agent_envolved_in_prediction:
            short = _agent_short_name(agent_name)
            raw_pred = row.get(f"{short}__prediction")
            pred_int = _BOOL_PRED_MAP.get(raw_pred) if raw_pred in (True, False) else None
            raw_avg_score = row.get(f"{short}__avg_score")
            avg_score = float(raw_avg_score) if raw_avg_score is not None and pd.notna(raw_avg_score) else None
            agents_block[agent_name] = AgentPredictionDetail(
                prediction=pred_int,
                confidence=row.get(f"{short}__confidence"),
                avg_score=avg_score,
                summary=row.get(f"{short}__summary"),
                reasoning=row.get(f"{short}__reasoning"),
                risks=row.get(f"{short}__risks"),
            )

        predictions.append(
            MultiagentSinglePrediction(
                date=date_value,
                base_price=base_price,
                y_true=_DIRECTION_MAP.get(row.get("y_true")),
                y_prediction=_DIRECTION_MAP.get(row.get("y_predict")),
                confidence_score=row.get("y_predict_confidence"),
                agents=agents_block,
            )
        )

    return MultiagentPredictionsResponse(
        requested_forecast_start_date=request.forecast_start_date,
        requested_horizon=request.horizon,
        requested_n_last_dates=request.n_last_dates,
        rows_returned=len(predictions),
        predictions=predictions,
        metrics=_compute_metrics(predictions),
    )


_AGENT_COLLECTORS = {
    "news_analyser": collect_news,
    "economic_calendar_analyser": collect_calendar_events,
    "twitter_analyser": twitter_fetch,
}


async def collect_agent_data_core(request: CollectAgentDataRequest) -> CollectAgentDataResponse:
    """Run incremental data collection for the requested agents.

    Shared by the HTTP endpoint and the daily scheduler job (api.scheduler) so both
    paths go through the same per-agent locks (_collection_locks) and stats shaping.
    """
    unknown = set(request.agents) - _AGENT_COLLECTORS.keys()
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown agents: {sorted(unknown)}")

    results = []
    for agent_name in request.agents:
        lock = _collection_locks[agent_name]
        if lock.locked():
            raise HTTPException(status_code=409, detail=f"Collection for '{agent_name}' is already running")
        async with lock:
            try:
                if agent_name == "twitter_analyser":
                    raw = await run_in_threadpool(
                        twitter_fetch,
                        request.twitter_since_date,
                        request.twitter_until_date,
                        request.twitter_authors,
                    )
                    if request.twitter_since_date and request.twitter_until_date:
                        await run_in_threadpool(
                            twitter_classify,
                            request.twitter_since_date,
                            request.twitter_until_date,
                            request.twitter_authors,
                        )
                else:
                    raw = await run_in_threadpool(_AGENT_COLLECTORS[agent_name])
                if agent_name == "twitter_analyser":
                    stats = {
                        "before": raw.get("db_total", 0) - raw.get("tweets_inserted", 0),
                        "fetched": raw.get("tweets_fetched", 0),
                        "new": raw.get("tweets_inserted", 0),
                        "after": raw.get("db_total", 0),
                        "date_range": raw.get("db_range"),
                    }
                else:
                    stats = raw
                results.append(CollectAgentDataResult(agent=agent_name, **stats))
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Collection failed for '{agent_name}': {exc}") from exc

    return CollectAgentDataResponse(results=results)


@router.post(
    "/system/collect_agent_data",
    response_model=CollectAgentDataResponse,
    summary="Collect latest data for news and calendar agents",
    description="Triggers incremental data collection for the specified agents, appending new records to their SQLite archives.",
)
async def collect_agent_data(request: CollectAgentDataRequest) -> CollectAgentDataResponse:
    return await collect_agent_data_core(request)


@router.get(
    "/agents/data-status",
    response_model=AgentsDataStatusResponse,
    summary="Last fetched date per agent archive",
    description="Returns MAX(date) from each agent's SQLite archive. Useful to check how fresh the stored data is before triggering collection.",
)
async def agents_data_status() -> AgentsDataStatusResponse:
    try:
        news_date, calendar_date, twitter_date = await run_in_threadpool(
            lambda: (
                news_get_latest_date(),
                calendar_get_latest_date(),
                twitter_get_latest_date(),
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to query agent archives: {exc}") from exc

    return AgentsDataStatusResponse(
        news_analyser=news_date,
        economic_calendar_analyser=calendar_date,
        twitter_analyser=twitter_date,
    )


@router.get(
    "/agents/twitter-auth-status",
    response_model=TwitterAuthStatusResponse,
    summary="Twitter session / cookie health check",
    description=(
        "Lightweight check (no Chrome launch). Inspects twitter_cookies.json for "
        "session cookies (auth_token, ct0) and verifies credentials are configured. "
        "If relogin_required=true, run: "
        "python -m MultiagentSystem.agents.twitter_analyser.twitter_scrapper.chrome_login_before_scrapping --login"
    ),
)
async def twitter_auth_status() -> TwitterAuthStatusResponse:
    try:
        info = await run_in_threadpool(check_twitter_auth)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to check Twitter auth: {exc}") from exc

    relogin_required = not info["session_cookies_ok"] or not info["credentials_configured"]

    return TwitterAuthStatusResponse(
        **info,
        relogin_required=relogin_required,
    )


@router.post(
    "/agents/twitter-upload-cookies",
    response_model=TwitterCookiesUploadResponse,
    summary="Upload Twitter cookies for re-login without stopping the API",
    description=(
        "Replaces twitter_cookies.json with the uploaded cookies. "
        "Use when session expires and manual re-login via GUI is unavailable. "
        "How to get cookies: open x.com in your browser → DevTools (F12) → "
        "Application → Cookies → x.com, then export with EditThisCookie extension "
        "or copy manually. Must include auth_token and ct0."
    ),
)
async def twitter_upload_cookies(
    request: TwitterCookiesUploadRequest = Body(...),
) -> TwitterCookiesUploadResponse:
    expected_key = os.getenv("TWITTER_UPLOAD_KEY", "")
    if not expected_key or request.upload_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid upload key")
    cookies = request.cookies
    try:
        info = await run_in_threadpool(save_cookies_from_upload, cookies)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save cookies: {exc}") from exc

    return TwitterCookiesUploadResponse(
        saved=info["cookies_count"],
        session_cookies_ok=info["session_cookies_ok"],
        cookies_path=info["cookies_path"],
    )

"""
LangGraph node: analyze macro-economic calendar events for BTC signal.

Unlike the news agent, there is NO pre-classification step.
All filtered events (Major all countries + Medium US only) are sent
to the LLM in a single prompt. The LLM aggregates and returns a verdict.

Flow:
    calendar_archive -> filter -> format prompt -> LLM -> AgentSignal
"""

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
import os

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from llm_factory import make_chat_llm
from multiagent_types import AgentState, get_agent_settings
from .calendar_collector import get_events_in_range


AGENT_DIR = Path(__file__).parent
LOG_TAG = "[agent_for_economic_calendar_analysis]"
AGENT_NAME = "agent_for_economic_calendar_analysis"

class CalendarVerdict(BaseModel):
    direction: Literal["bullish", "bearish", "neutral"]
    confidence: Literal["high", "medium", "low"]
    reasoning: str = Field(description="2-3 sentences explaining which events dominate and why")


# -- Helpers -------------------------------------------------------------------

def _parse_forecast_window(
    forecast_date: str | date | datetime,
    window_days: int,
) -> tuple[datetime, datetime, datetime]:
    """Convert forecast_date + window into (window_start, forecast_end_date, window_end_exclusive)."""
    
    if isinstance(forecast_date, str):
        forecast_end_date = datetime.strptime(forecast_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        forecast_end_date = datetime.combine(forecast_date, datetime.min.time()).replace(tzinfo=timezone.utc)

    window_end_exclusive = forecast_end_date + timedelta(days=1)
    window_start = window_end_exclusive - timedelta(days=window_days)
    return window_start, forecast_end_date, window_end_exclusive


def _is_us_event(event: dict) -> bool:
    code = str(event.get("country_code", "")).upper()
    if code == "US":
        return True
    name = str(event.get("country_name", "")).lower()
    return "united states" in name or "usa" in name or "u.s." in name


def _filter_events_by_importance(events: list[dict]) -> list[dict]:
    """
    Keep:
      - Major events (importance >= 3) for all countries
      - Medium events (importance == 2) for US only
    Exclude:
      - Waiting events
      - events without published value
    """
    filtered = []
    for e in events:
        if not e.get("published_value") or e.get("data_effect") == "Waiting":
            continue
        imp = int(e.get("importance_level", 0) or 0)
        if imp >= 3 or (imp == 2 and _is_us_event(e)):
            filtered.append(e)
    return filtered


def _format_event(event: dict) -> str:
    """Format a single event as a compact string for the LLM prompt."""
    name = event.get("calendar_name", "?")
    country = event.get("country_name", "?")
    imp = "Major" if event.get("importance_level", 0) >= 3 else "Medium"
    actual = event.get("published_value", "—")
    forecast = event.get("forecast_value", "") or "—"
    previous = event.get("previous_value", "") or "—"
    effect = event.get("data_effect", "—")
    dt = event.get("date", "?")
    return (
        f"[{imp}] {dt} [{country}] {name}\n"
        f"  actual: {actual} | forecast: {forecast} | previous: {previous}\n"
        f"  data_effect: {effect}"
    )


def _format_all_events(events: list[dict]) -> str:
    """Format all events into a single text block."""
    return "\n\n".join(_format_event(e) for e in events)


def _event_effect_balance(events: list[dict]) -> tuple[int, int]:
    """
    Count bullish vs bearish data_effect labels.
    Used only as a confidence guardrail (not a directional model).
    """
    bullish = 0
    bearish = 0
    for e in events:
        eff = str(e.get("data_effect", "")).strip().lower()
        if "bull" in eff:
            bullish += 1
        elif "bear" in eff:
            bearish += 1
    return bullish, bearish


def _save_prediction_debug(
    forecast_date,
    horizon: int,
    window_days: int,
    events: list[dict],
    verdict: CalendarVerdict | None,
) -> None:
    """Save debug artifact for post-mortem analysis."""
    debug = {
        "date": str(forecast_date),
        "horizon": horizon,
        "window": window_days,
        "total_events": len(events),
        "events": [
            {
                "date": e.get("date", "?"),
                "name": e.get("calendar_name", "?"),
                "country": e.get("country_name", "?"),
                "importance": e.get("importance_level", 0),
                "actual": e.get("published_value", ""),
                "forecast": e.get("forecast_value", ""),
                "previous": e.get("previous_value", ""),
                "data_effect": e.get("data_effect", ""),
            }
            for e in events
        ],
        "verdict": {
            "direction": verdict.direction,
            "confidence": verdict.confidence,
            "reasoning": verdict.reasoning,
        } if verdict else None,
    }
    (AGENT_DIR / "calendar_predict.json").write_text(
        json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def get_system_prompt()-> str:
    """Getting the prompt txt, which we have in the folder where lay the file"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "system_prompt.txt")
    
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    return text



# -- Main agent function -------------------------------------------------------

def agent_for_economic_calendar_analysis(state: AgentState):
    """LangGraph node: analyze macro-economic calendar events for BTC signal.

    All filtered events are sent to the LLM in a single prompt.
    No pre-classification — the LLM receives raw events and decides.
    """

    if AGENT_NAME not in state.get("agent_envolved_in_prediction", []):
        print(f"{LOG_TAG} Not in agent_envolved_in_prediction — skipping")
        return {}

    my_retry = None
    for r in state.get("retry_agents", []):
        if r["agent_name"] == AGENT_NAME:
            my_retry = r
            break

    if my_retry is not None and my_retry["currents_retry"] >= my_retry["max_retries"]:
        print(f"{LOG_TAG} Retry limit reached ({my_retry['currents_retry']}/{my_retry['max_retries']}) — skipping")
        return {}

    attempt = my_retry["currents_retry"] if my_retry is not None else 0
    print(f"\n{'='*60}")
    print(f"{LOG_TAG} === ATTEMPT #{attempt} ===")
    print(f"{'='*60}")

    # --- Settings ---
    settings = get_agent_settings(state, "agent_for_economic_calendar_analysis")
    horizon = state["horizon"]
    forecast_date = state["forecast_start_date"]
    window_days = settings["window_to_analysis"]
    llm_model = settings.get("llm_model", "gpt-4o-mini")
    print(f"{LOG_TAG} [1/4] Settings | horizon={horizon}d | forecast_date={forecast_date} | window={window_days}d | llm_model={llm_model}")

    # --- Date boundaries ---
    window_start, forecast_end_date, window_end_exclusive = _parse_forecast_window(forecast_date, window_days)
    window_end_inclusive = window_end_exclusive - timedelta(microseconds=1)
    print(f"{LOG_TAG} [2/4] Date window: {window_start.date()} -> {forecast_end_date.date()} (inclusive)")

    # --- Load and filter events ---
    print(f"{LOG_TAG} [3/4] Loading events from archive...")
    all_events = get_events_in_range(dt_from=window_start, dt_to=window_end_inclusive)
    filtered = _filter_events_by_importance(all_events)
    print(
        f"{LOG_TAG}   Raw: {len(all_events)} events | "
        f"After filter (Major all + Medium US): {len(filtered)}"
    )

    if filtered:
        from collections import Counter
        date_counts = Counter(e.get("date", "?") for e in filtered)
        print(f"{LOG_TAG}   Events by date ({len(date_counts)} unique dates):")
        for d in sorted(date_counts):
            print(f"{LOG_TAG}     {d}: {date_counts[d]} event(s)")

        print(f"{LOG_TAG}   Detailed event list:")
        for e in sorted(filtered, key=lambda x: x.get("date", "")):
            d = e.get("date", "?")
            country = e.get("country_name", "?")
            name = e.get("calendar_name", "?")
            actual = e.get("published_value", "—")
            forecast = e.get("forecast_value", "") or "—"
            previous = e.get("previous_value", "") or "—"
            effect = e.get("data_effect", "—")
            print(
                f"{LOG_TAG}     [{d}] [{country}] {name} | "
                f"actual={actual} forecast={forecast} previous={previous} | effect={effect}"
            )

    if not filtered:
        print(f"{LOG_TAG}   No events found — returning neutral stub signal")
        return {"agent_signals": {AGENT_NAME: {
            "reasoning": f"No major macro events in window {window_start.date()} -> {forecast_end_date.date()}.",
            "summary": "No macro data — abstain from voting.",
            "risks": "",
            "prediction": None,
            "confidence": None,
            "description_of_the_reports_problem": [],
        }}}

    # --- LLM call ---
    events_text = _format_all_events(filtered)
    SYSTEM_PROMPT = get_system_prompt()
    system_msg = SYSTEM_PROMPT.format(horizon=horizon, window=window_days)

    print(f"{LOG_TAG}   Sending {len(filtered)} events to LLM...")
    
    llm = make_chat_llm(llm_model, temperature=0.0)
    try:
        verdict = llm.with_structured_output(CalendarVerdict).invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=f"Analyze these {len(filtered)} economic calendar events:\n\n{events_text}"),
        ])
    except Exception as exc:
        err = f"LLM request failed in {AGENT_NAME}: {exc}"
        print(f"{LOG_TAG}   ERROR: {err}")
        return {"agent_signals": {AGENT_NAME: {
            "reasoning": err,
            "summary": "LLM temporarily unavailable — abstain from voting.",
            "risks": "Network/API issue during model call.",
            "prediction": None,
            "confidence": None,
            "description_of_the_reports_problem": [],
        }}}

    final_direction = verdict.direction
    final_confidence = verdict.confidence

    # Post-LLM confidence guardrails:
    # - tiny sample: high confidence is disallowed
    # - mixed bullish/bearish effect labels: confidence is downgraded
    # - tiny and strongly conflicting sample: abstain (neutral)
    bull_cnt, bear_cnt = _event_effect_balance(filtered)
    conflict = bull_cnt > 0 and bear_cnt > 0
    if len(filtered) < 3 and final_confidence == "high":
        final_confidence = "medium"
    if conflict and final_confidence == "high":
        final_confidence = "medium"
    if conflict and len(filtered) <= 2 and abs(bull_cnt - bear_cnt) <= 1:
        final_direction = "neutral"
        final_confidence = "low"

    if final_direction == "bullish":
        prediction = True
        confidence = final_confidence
        prediction_label = "HIGHER"
    elif final_direction == "bearish":
        prediction = False
        confidence = final_confidence
        prediction_label = "LOWER"
    else:
        prediction = None
        confidence = None
        prediction_label = "NEUTRAL"
    print(
        f"{LOG_TAG} [4/4] Verdict: {verdict.direction}->{final_direction} | "
        f"confidence={verdict.confidence}->{final_confidence} | -> {prediction_label}"
    )
    print(f"{LOG_TAG}   Reasoning: {verdict.reasoning}")

    # --- Save debug output ---
    _save_prediction_debug(forecast_date, horizon, window_days, filtered, verdict)
    print(f"{LOG_TAG}   calendar_predict.json saved")

    # --- Build agent signal ---
    summary = (
        f"{len(filtered)} macro events analyzed. "
        f"LLM verdict: {final_direction}, confidence: {final_confidence}."
    )

    return {"agent_signals": {AGENT_NAME: {
        "reasoning": verdict.reasoning,
        "summary": summary,
        "risks": "",
        "prediction": prediction,
        "confidence": confidence,
        "description_of_the_reports_problem": [],
    }}}

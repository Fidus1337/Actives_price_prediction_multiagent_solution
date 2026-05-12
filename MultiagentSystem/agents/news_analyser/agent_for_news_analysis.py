import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from multiagent_types import AgentState, get_agent_settings
from .helpers import parse_release_time
from .news_collector import get_articles_in_range
from .news_classifier import (
    STRENGTH_WEIGHTS,
    classify_articles,
)


AGENT_DIR = Path(__file__).parent
LOG_TAG = "[agent_for_news_analysis]"
AGENT_NAME = "agent_for_news_analysis"


# -- Pure helpers --------------------------------------------------------------

def _distance_to_confidence(distance: float) -> str:
    """Map distance from neutral (0.5) to a confidence label."""
    if distance >= 0.3:
        return "high"
    if distance >= 0.15:
        return "medium"
    return "low"


def _compute_verdict_from_weights(
    bull_weight: float, bear_weight: float,
) -> tuple[bool | None, str | None, float]:
    """Compute prediction from absolute weighted scores.

    Returns (is_bullish, confidence, bull_ratio). When there is no directional
    signal (no bull or bear weight at all), returns (None, None, 0.5) so the
    caller can emit an abstain signal instead of a paragraph SHORT vote.
    """
    total_weight = bull_weight + bear_weight
    if total_weight == 0:
        return None, None, 0.5

    # Neutral point: equal bull and bear weight means no directional signal
    bull_ratio = bull_weight / total_weight
    is_bullish = bull_ratio > 0.5
    confidence = _distance_to_confidence(abs(bull_ratio - 0.5))

    return is_bullish, confidence, bull_ratio


def _compute_verdict_from_normalized_score(
    normalized_score: float,
) -> tuple[bool, str, float]:
    """Compute prediction from normalized score in [-1, 1].

    Returns (is_bullish, confidence, bull_ratio_equivalent).
    """
    bull_ratio_equivalent = (normalized_score + 1.0) / 2.0
    is_bullish = normalized_score > 0
    confidence = _distance_to_confidence(abs(normalized_score) / 2.0)
    return is_bullish, confidence, bull_ratio_equivalent


def _time_decay(
    article: dict,
    reference_date: datetime,
    decay_rate: float,
    decay_start_day: int,
    initial_weight: float,
) -> float:
    """Piecewise decay aligned with twitter-agent schema.

    age < decay_start_day:   weight = 1.0                            (fresh zone)
    age >= decay_start_day:  weight = initial_weight * (1 - decay_rate) ** t
                             where t = age - decay_start_day
    """
    release_time = parse_release_time(article.get("article_release_time"))
    if release_time is None:
        return 1.0
    age_days = max((reference_date - release_time).total_seconds() / 86400.0, 0.0)
    if age_days < decay_start_day:
        return 1.0
    t = age_days - decay_start_day
    return initial_weight * (1 - decay_rate) ** t


# -- Extracted single-responsibility functions ---------------------------------

def _parse_forecast_window(
    forecast_date: str | date | datetime,
    window_days: int,
) -> tuple[datetime, datetime, datetime]:
    """Convert forecast_date + window into (window_start, forecast_end_date, window_end_exclusive).

    Uses exclusive upper bound to avoid double-counting articles published
    exactly at midnight on the boundary date.
    """
    if isinstance(forecast_date, str):
        forecast_end_date = datetime.strptime(forecast_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        forecast_end_date = datetime.combine(forecast_date, datetime.min.time()).replace(tzinfo=timezone.utc)

    window_end_exclusive = forecast_end_date + timedelta(days=1)
    window_start = window_end_exclusive - timedelta(days=window_days)
    return window_start, forecast_end_date, window_end_exclusive


def _load_articles_in_window(
    window_start: datetime,
    window_end_inclusive: datetime,
    window_end_exclusive: datetime,
) -> list[dict]:
    """Fetch pre-classified articles from local archive (with API fallback),
    filter to window. Falls back to LLM classification for any unclassified articles."""
    archive = get_articles_in_range(
        dt_from=window_start,
        dt_to=window_end_inclusive,
        fallback_to_api=True,
    )

    articles_in_window = []
    for item in archive:
        release_time = parse_release_time(item.get("article_release_time"))
        if release_time is None or not (window_start <= release_time < window_end_exclusive):
            continue
        articles_in_window.append(item)

    articles_in_window.sort(
        key=lambda x: parse_release_time(x.get("article_release_time")) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    # Fallback: classify any articles that lack classification
    unclassified = [a for a in articles_in_window if "category" not in a or a.get("category") == "unclassified"]
    if unclassified:
        print(f"{LOG_TAG}   WARNING: {len(unclassified)} unclassified articles found — classifying via LLM fallback")
        classify_articles(unclassified)

    return articles_in_window


def _aggregate_sentiment(
    articles: list[dict],
    reference_date: datetime,
    decay_rate: float,
    decay_start_day: int,
    initial_weight: float,
) -> tuple[float, float, int, int, int]:
    """Aggregate pre-classified articles into weighted sentiment scores.

    Weights are scaled by STRENGTH_WEIGHTS and multiplied by a piecewise
    time-decay factor (see `_time_decay`), so newer articles contribute
    more than older ones according to the configured decay parameters.

    Returns (bull_weight, bear_weight, bull_count, bear_count, not_correlated_count).
    """
    bull_weight = 0.0
    bear_weight = 0.0
    bull_count = 0
    bear_count = 0
    not_correlated_count = 0

    for article in articles:
        category = article.get("category", "not_correlated")
        strength = article.get("strength", "low")
        weight = STRENGTH_WEIGHTS.get(strength, 1) * _time_decay(
            article, reference_date, decay_rate, decay_start_day, initial_weight
        )

        if category == "bull":
            bull_count += 1
            bull_weight += weight
        elif category == "bear":
            bear_count += 1
            bear_weight += weight
        else:
            not_correlated_count += 1

    return bull_weight, bear_weight, bull_count, bear_count, not_correlated_count


def _compute_batch_normalized_score(
    articles: list[dict],
    batch_size: int,
    reference_date: datetime,
    decay_rate: float,
    decay_start_day: int,
    initial_weight: float,
) -> float | None:
    """Per-batch normalization for >60 articles.
    Prevents a single large batch from dominating the signal.
    Uses the same time-decay as _aggregate_sentiment.

    Returns normalized score in [-1, 1], or None if not applicable.
    """
    total = len(articles)
    if total <= 60:
        return None

    effective_batch_size = max(batch_size, 1)
    batch_scores: list[float] = []
    batch_weights: list[float] = []

    for batch_start in range(0, total, effective_batch_size):
        batch = articles[batch_start:batch_start + effective_batch_size]
        batch_bull = 0.0
        batch_bear = 0.0
        for a in batch:
            cat = a.get("category", "not_correlated")
            w = STRENGTH_WEIGHTS.get(a.get("strength", "low"), 1) * _time_decay(
                a, reference_date, decay_rate, decay_start_day, initial_weight
            )
            if cat == "bull":
                batch_bull += w
            elif cat == "bear":
                batch_bear += w
        total_w = batch_bull + batch_bear
        if total_w > 0:
            batch_scores.append((batch_bull - batch_bear) / total_w)
            batch_weights.append(total_w)

    if not batch_scores:
        return None

    weighted_sum = sum(s * w for s, w in zip(batch_scores, batch_weights))
    total_relevant = sum(batch_weights)
    return weighted_sum / total_relevant if total_relevant else 0.0


def _save_prediction_debug(
    forecast_date, horizon: int, window_days: int,
    articles: list[dict],
    bull_count: int, bull_weight: float,
    bear_count: int, bear_weight: float, not_correlated_count: int,
    bull_ratio: float, is_bullish: bool, confidence: str,
) -> None:
    """Debug artifact for post-mortem analysis of classification quality."""
    news_predict = {
        "date": str(forecast_date),
        "horizon": horizon,
        "window": window_days,
        "total_articles": len(articles),
        "bull_count": bull_count,
        "bull_weight": bull_weight,
        "bear_count": bear_count,
        "bear_weight": bear_weight,
        "not_correlated_count": not_correlated_count,
        "bull_ratio_weighted": round(bull_ratio, 4),
        "prediction": is_bullish,
        "confidence": confidence,
        "classifications": [
            {
                "title": a.get("article_title", "—"),
                "category": a.get("category", "?"),
                "strength": a.get("strength", "?"),
                "date": a.get("date", "?"),
            }
            for a in articles
        ],
    }
    (AGENT_DIR / "news_predict.json").write_text(
        json.dumps(news_predict, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# -- Main agent function -------------------------------------------------------

def agent_for_news_analysis(state: AgentState):
    """LangGraph node: aggregate pre-classified crypto news into a bull/bear signal.

    Articles are pre-classified at collection time (see news_collector.py).
    This node only reads classifications and aggregates — no LLM calls
    unless unclassified articles are found (fallback).
    """

    # We take agents envolvedd in pipeline
    if AGENT_NAME not in state.get("agent_envolved_in_prediction", []):
        print(f"{LOG_TAG} Not in agent_envolved_in_prediction — skipping")
        return {}

    # Check if the bot has retries
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
    settings = get_agent_settings(state, "agent_for_news_analysis")
    horizon = state["horizon"]
    forecast_date = state["forecast_start_date"]
    window_days = settings["window_to_analysis"]
    decay_rate = settings["decay_rate"]
    decay_start_day = settings["decay_start_day"]
    initial_weight = settings["initial_weight"]
    print(
        f"{LOG_TAG} [1/4] Settings loaded | horizon={horizon}d | forecast_date={forecast_date} | "
        f"window={window_days}d | decay_rate={decay_rate} | decay_start_day={decay_start_day} | "
        f"initial_weight={initial_weight}"
    )

    # --- Date boundaries ---
    window_start, forecast_end_date, window_end_exclusive = _parse_forecast_window(forecast_date, window_days)
    window_end_inclusive = window_end_exclusive - timedelta(microseconds=1)
    print(f"{LOG_TAG} [2/4] Date window: {window_start.date()} -> {forecast_end_date.date()} (inclusive)")

    # --- Load pre-classified articles ---
    print(f"{LOG_TAG} [3/4] Loading pre-classified articles from archive...")
    articles = _load_articles_in_window(window_start, window_end_inclusive, window_end_exclusive)
    print(f"{LOG_TAG}   Found {len(articles)} articles in window")

    if not articles:
        print(f"{LOG_TAG}   No articles found — returning neutral stub signal")
        return {"agent_signals": {AGENT_NAME: {
            "reasoning": f"No articles found in window {window_start.date()} -> {forecast_end_date.date()}.",
            "summary": "No news data — abstain from voting.",
            "risks": "",
            "prediction": None,
            "confidence": None,
            "description_of_the_reports_problem": [],
        }}}

    # --- Aggregate pre-classified sentiment ---
    bull_weight, bear_weight, bull_count, bear_count, not_correlated_count = _aggregate_sentiment(
        articles, forecast_end_date, decay_rate, decay_start_day, initial_weight
    )

    print(f"{LOG_TAG}   bull={bull_count} (w={bull_weight:.2f}), bear={bear_count} (w={bear_weight:.2f}), neutral={not_correlated_count}")
    for article in articles:
        cat = article.get("category", "not_correlated")
        if cat in ("bull", "bear"):
            title = article.get("article_title", "—")[:80]
            strength = article.get("strength", "?")
            print(f"{LOG_TAG}     [{cat:>4} {strength:>6}] {title}")

    # --- Compute verdict ---
    total_articles = len(articles)
    normalized_score = _compute_batch_normalized_score(
        articles,
        batch_size=30,
        reference_date=forecast_end_date,
        decay_rate=decay_rate,
        decay_start_day=decay_start_day,
        initial_weight=initial_weight,
    )

    if normalized_score is not None:
        is_bullish, confidence, bull_ratio = _compute_verdict_from_normalized_score(normalized_score)
        print(f"{LOG_TAG}   Batch-normalized: score={normalized_score:.3f}")
    else:
        is_bullish, confidence, bull_ratio = _compute_verdict_from_weights(bull_weight, bear_weight)

    if is_bullish is None:
        print(f"{LOG_TAG}   No directional signal (all articles not_correlated) — abstain")
        return {"agent_signals": {AGENT_NAME: {
            "reasoning": (
                f"Out of {total_articles} articles, none were classified as bull/bear "
                f"(not_correlated={not_correlated_count}). No directional signal."
            ),
            "summary": "No directional news signal — abstain from voting.",
            "risks": "",
            "prediction": None,
            "confidence": None,
            "description_of_the_reports_problem": [],
        }}}

    prediction_label = "HIGHER" if is_bullish else "LOWER"
    print(f"{LOG_TAG} [4/4] Verdict: {prediction_label} | confidence={confidence} | bull_ratio={bull_ratio:.2f}")

    # --- Save debug output ---
    _save_prediction_debug(
        forecast_date, horizon, window_days,
        articles,
        bull_count, bull_weight, bear_count, bear_weight, not_correlated_count,
        bull_ratio, is_bullish, confidence,
    )
    print(f"{LOG_TAG}   news_predict.json saved")

    # --- Build agent signal ---
    summary = (
        f"Bull: {bull_count} (w={bull_weight}), Bear: {bear_count} (w={bear_weight}), "
        f"Not correlated: {not_correlated_count}. "
        f"Weighted bull ratio: {bull_ratio:.2f} → {prediction_label}, confidence: {confidence}."
    )
    reasoning = (
        f"Out of {total_articles} articles, {bull_count} are bullish (weight={bull_weight}) "
        f"and {bear_count} are bearish (weight={bear_weight}) "
        f"(+{not_correlated_count} not correlated). "
        f"Weighted bull/bear ratio = {bull_ratio:.2f}."
    )

    return {"agent_signals": {AGENT_NAME: {
        "reasoning": reasoning,
        "summary": summary,
        "risks": "",
        "prediction": is_bullish,
        "confidence": confidence,
        "description_of_the_reports_problem": [],
    }}}

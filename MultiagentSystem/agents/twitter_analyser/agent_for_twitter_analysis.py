import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


try:
    from multiagent_types import AgentState, get_agent_settings
    from agents.twitter_analyser.twitter_scrapper.twitter_db import get_tweets_in_range
except ModuleNotFoundError:
    from MultiagentSystem.multiagent_types import AgentState, get_agent_settings
    from MultiagentSystem.agents.twitter_analyser.twitter_scrapper.twitter_db import get_tweets_in_range


AGENT_DIR = Path(__file__).parent
LOG_TAG = "[agent_for_twitter_analysis]"
AGENT_NAME = "agent_for_twitter_analysis"


# -- Helpers -------------------------------------------------------------------

def _get_window_dates(end_date: str | date | datetime, window_days: int) -> tuple[datetime, datetime]:
    """Calculate dt_from and dt_to for get_tweets_in_range from end date and window size.

    Args:
        end_date:    last day of the window (inclusive), e.g. forecast date
        window_days: how many days back to look

    Returns:
        (dt_from, dt_to) — both timezone-aware UTC datetimes, ready for get_tweets_in_range
    """
    if isinstance(end_date, str):
        dt_to = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    elif isinstance(end_date, date) and not isinstance(end_date, datetime):
        dt_to = datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    else:
        dt_to = end_date.replace(tzinfo=timezone.utc) if end_date.tzinfo is None else end_date

    dt_from = dt_to - timedelta(days=window_days - 1)
    return dt_from, dt_to

def _group_tweets_by_date(tweets: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for tweet in tweets:
        date_key = tweet.get("date") or ""
        if not date_key:
            continue
        result.setdefault(date_key, []).append(tweet)
    return result

def _aggregate_signals_by_author_and_date(
    signals_by_dates: dict[str, list[dict]]
) -> dict[str, dict[str, dict]]:

    SCORE_MAP = {
        ("BULL", "HIGH"): 3, ("BULL", "MIDDLE"): 2, ("BULL", "LOW"): 1,
        ("BEAR", "LOW"): -1, ("BEAR", "MIDDLE"): -2, ("BEAR", "HIGH"): -3,
    }

    result: dict[str, dict[str, dict]] = {}

    for date_key, tweets in signals_by_dates.items():
        author_scores: dict[str, list[float]] = {}
        for tweet in tweets:
            signal_type = (tweet.get("signal_type") or "").upper()
            confidence  = (tweet.get("signal_confidence") or "LOW").upper()
            score = SCORE_MAP.get((signal_type, confidence))
            if score is None:
                continue
            author = (tweet.get("author_username") or "unknown").lower()
            author_scores.setdefault(author, []).append(score)

        result[date_key] = {}
        for author, scores in author_scores.items():
            avg = sum(scores) / len(scores)

            # 2) signal_confidence — floor от абсолютного среднего
            signal_confidence = round(abs(avg))

            # 3) нулевой confidence — автор не даёт сигнала, пропускаем
            if signal_confidence == 0:
                continue

            signal_type = "BULL" if avg > 0 else "BEAR"

            result[date_key][author] = {
                "signal_type": signal_type,
                "signal_confidence": signal_confidence,  # int: 1, 2 или 3
                "avg_score": round(avg, 3),              # 1) float вместо строки
                "tweets_count": len(scores),
            }

    return result

def _merge_authors_signals_in_dates_into_one_signal(
    aggregated: dict[str, dict[str, dict]]
) -> dict[str, dict]:
    """Merge all author signals for each date into one final signal.

    Uses the same averaging principle:
    - BULL author → +signal_confidence (signed score)
    - BEAR author → -signal_confidence (signed score)
    - Average signed scores across all authors
    - Floor abs(avg) → final int confidence
    - If confidence == 0 → no signal for that date

    Returns:
        {
            "2026-03-27": {"signal_type": "BULL", "signal_confidence": 2, "avg_score": 1.8, "authors_count": 4},
            ...
        }
    """
    result: dict[str, dict] = {}

    for date_key, authors in aggregated.items():
        if not authors:
            continue

        signed_scores = []
        for author_signal in authors.values():
            conf = author_signal["signal_confidence"]  # int: 1, 2, 3
            signed = conf if author_signal["signal_type"] == "BULL" else -conf
            signed_scores.append(signed)

        avg = sum(signed_scores) / len(signed_scores)
        signal_confidence = round(abs(avg))

        # Signals cancelled each other out — no actionable signal for this date
        if signal_confidence == 0:
            continue

        result[date_key] = {
            "signal_type": "BULL" if avg > 0 else "BEAR",
            "signal_confidence": signal_confidence,   # int: 1, 2, 3
            "avg_score": round(avg, 3),               # float
            "authors_count": len(signed_scores),
        }

    return result

def _merge_date_signals_into_final_verdict(
    date_signals: dict[str, dict],
    decay_rate: float,
    decay_start_day: int,
    initial_weight: float,
    reference_date: str | date | None = None,
) -> dict | None:
    """Merge one-signal-per-date into a single verdict for the entire window.

    Each day's signal gets a weight based on its age relative to reference_date.
    Weight function is piecewise:
        age < decay_start_day:  weight = 1.0                                    (fresh zone)
        age >= decay_start_day: weight = initial_weight * (1 - decay_rate) ** t
                                where t = age - decay_start_day

    Example (initial_weight=0.8, decay_rate=0.05, decay_start_day=7):
        age 0..6 → 1.00
        age 7    → 0.800
        age 8    → 0.760
        age 14   → 0.559
        age 20   → 0.410 (formula keeps working; no hard cutoff)

    - BULL date → +signal_confidence * weight
    - BEAR date → -signal_confidence * weight
    - Weighted average of signed scores
    - round(abs(avg)) → final int confidence
    - If confidence == 0 → signal_type is None (too weak to vote), avg_score still reported
    - If no valid dates / total_weight == 0 → returns None

    Returns:
        {"signal_type": "BULL", "signal_confidence": 2, "avg_score": 1.5, "dates_count": 7}
        {"signal_type": None,   "signal_confidence": 0, "avg_score": 0.3, "dates_count": 3}  # weak
        or None if no valid dates at all
    """
    if not date_signals:
        return None

    if reference_date is None:
        today = datetime.now(tz=timezone.utc).date()
    elif isinstance(reference_date, str):
        today = datetime.strptime(reference_date, "%Y-%m-%d").date()
    elif isinstance(reference_date, datetime):
        today = reference_date.date()
    else:
        today = reference_date  # already a date object

    weighted_scores = []
    total_weight = 0.0

    for date_key, day_signal in date_signals.items():
        # Вычисляем возраст даты в днях
        try:
            day_date = datetime.strptime(date_key, "%Y-%m-%d").date()
        except ValueError:
            continue
        age_days = max((today - day_date).days, 0)

        # Fresh zone holds weight = 1.0; after decay_start_day apply N0 * (1-r)^t
        if age_days < decay_start_day:
            decay = 1.0
        else:
            t = age_days - decay_start_day
            decay = initial_weight * (1 - decay_rate) ** t

        conf = day_signal["signal_confidence"]   # int: 1, 2, 3
        signed = conf if day_signal["signal_type"] == "BULL" else -conf
        weighted_scores.append(signed * decay)
        total_weight += decay

    if total_weight == 0:
        return None

    avg = sum(weighted_scores) / total_weight
    signal_confidence = round(abs(avg))

    if signal_confidence == 0:
        signal_type = None
    else:
        signal_type = "BULL" if avg > 0 else "BEAR"

    return {
        "signal_type": signal_type,
        "signal_confidence": signal_confidence,
        "avg_score": round(avg, 3),
        "dates_count": len(weighted_scores),
    }


# -- Main agent function -------------------------------------------------------

def agent_for_twitter_analysis(state: AgentState):
    """LangGraph node: aggregate pre-classified Twitter signals into a bull/bear verdict.

    Tweets are pre-classified at collection time (see full_scrapping_pipeline.py).
    Only BULL/BEAR tweets are stored in the SQLite archive.
    This node reads classifications and aggregates — no LLM calls.
    """

    if AGENT_NAME not in state.get("agent_envolved_in_prediction", []):
        print(f"{LOG_TAG} Not in agent_envolved_in_prediction — skipping")
        return {}

    settings = get_agent_settings(state, AGENT_NAME)
    forecast_date = state["forecast_start_date"]
    window_days = settings["window_to_analysis"]
    decay_rate = float(settings["decay_rate"])
    decay_start_day = int(settings["decay_start_day"])
    initial_weight = float(settings["initial_weight"])
    dt_from, dt_to = _get_window_dates(forecast_date, window_days)

    tweets_raw = get_tweets_in_range(dt_from=dt_from, dt_to=dt_to)
    print(f"{LOG_TAG} Fetched {len(tweets_raw)} tweets from DB")

    future_leak = [t for t in tweets_raw if (t.get("date") or "") > str(dt_to.date())]
    if future_leak:
        print(f"{LOG_TAG} !! LOOKAHEAD DETECTED — {len(future_leak)} tweets with date > {dt_to.date()}")
        for t in future_leak[:5]:
            print(f"{LOG_TAG}      date={t['date']}  created_at={t.get('created_at')}  @{t.get('author_username')}")
    else:
        print(f"{LOG_TAG} Lookahead check OK — no tweets beyond {dt_to.date()}")

    allowed_authors = [a.lower() for a in settings.get("authors", [])]
    tweets = tweets_raw
    if allowed_authors:
        tweets = [t for t in tweets if (t.get("author_username") or "").lower() in allowed_authors]
        print(f"{LOG_TAG} After author filter ({allowed_authors}): {len(tweets)} tweets")

    tweets = [
        t for t in tweets
        if (t.get("signal_type") or "").upper() not in ("", "NO_CORRELATION_TO_BTC")
    ]
    print(f"{LOG_TAG} After signal filter: {len(tweets)} actionable tweets")

    for t in sorted(tweets, key=lambda x: x.get("date") or ""):
        text_full = (t.get("text") or "").replace("\n", " ")
        print(f"{LOG_TAG}   [{t.get('date')}] @{t.get('author_username')} | {t.get('signal_type')} {t.get('signal_confidence')}")
        print(f"{LOG_TAG}     {text_full}")

    tweets_by_date = _group_tweets_by_date(tweets)
    by_author = _aggregate_signals_by_author_and_date(tweets_by_date)
    by_date = _merge_authors_signals_in_dates_into_one_signal(by_author)

    verdict = _merge_date_signals_into_final_verdict(
        by_date,
        decay_rate=decay_rate,
        decay_start_day=decay_start_day,
        initial_weight=initial_weight,
        reference_date=forecast_date,
    )

    if verdict is None:
        return {"agent_signals": {AGENT_NAME: {
            "reasoning": "Twitter agent system has no actionable signals in the window.",
            "summary": "No twitter signals in window — abstaining from voting.",
            "risks": "",
            "prediction": None,
            "confidence": None,
            "avg_score": None,
            "description_of_the_reports_problem": [],
        }}}

    if verdict["signal_type"] is None:
        return {"agent_signals": {AGENT_NAME: {
            "reasoning": (
                f"Aggregated twitter signals across window: "
                f"avg_score={verdict['avg_score']}, dates_count={verdict['dates_count']}. "
                f"abs(avg_score) < 0.5 — too weak to vote in the general forecast."
            ),
            "summary": (
                f"Twitter signal too weak | avg_score={verdict['avg_score']} "
                f"over {verdict['dates_count']} days — abstaining from voting."
            ),
            "risks": "",
            "prediction": None,
            "confidence": None,
            "avg_score": verdict["avg_score"],
            "description_of_the_reports_problem": [],
        }}}

    is_bullish = verdict["signal_type"] == "BULL"
    confidence = {3: "high", 2: "medium", 1: "low"}.get(verdict["signal_confidence"], "low")

    return {"agent_signals": {AGENT_NAME: {
    "prediction": is_bullish,                          # bool: True=BULL / False=BEAR
    "confidence": confidence,                          # str: "high" / "medium" / "low"
    "avg_score": verdict["avg_score"],
    "summary": (
        f"{verdict['signal_type']} signal over {verdict['dates_count']} days "
        f"| avg_score={verdict['avg_score']} | confidence={confidence}"
    ),
    "reasoning": (
        f"Aggregated twitter signals across window: "
        f"avg_score={verdict['avg_score']}, dates_count={verdict['dates_count']}"
    ),
    "risks": "",
    "description_of_the_reports_problem": [],
    }}}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    import json

    config_path = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "multiagent_config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    settings     = config["agent_settings"][AGENT_NAME]
    forecast_date = "2026-03-28"
    window_days  = 3
    decay_rate      = float(settings["decay_rate"])
    decay_start_day = int(  settings["decay_start_day"])
    initial_weight  = float(settings["initial_weight"])
    # allowed_authors = [a.lower() for a in settings.get("authors", [])]
    allowed_authors = ["rektcapital"]

    dt_from, dt_to = _get_window_dates(forecast_date, window_days)
    print(f"=== Config ===")
    print(f"  forecast_date   : {forecast_date}")
    print(f"  window          : {window_days} days  ({dt_from.date()} → {dt_to.date()})")
    print(f"  decay_rate      : {decay_rate}")
    print(f"  decay_start_day : {decay_start_day}")
    print(f"  initial_weight  : {initial_weight}")
    print(f"  authors filter  : {allowed_authors or '(all)'}")

    tweets_raw = get_tweets_in_range(dt_from=dt_from, dt_to=dt_to)
    print(f"\n  Total tweets fetched: {len(tweets_raw)}")

    # ── Lookahead check ─────────────────────────────────────────────────────────
    print(f"\n=== Lookahead check (boundary: {dt_to.date()}) ===")
    dates_in_db = sorted({t.get("date") or "" for t in tweets_raw if t.get("date")})
    print(f"  Unique dates in result : {dates_in_db}")
    future_tweets = [t for t in tweets_raw if (t.get("date") or "") > str(dt_to.date())]
    if future_tweets:
        print(f"  !! LOOKAHEAD DETECTED — {len(future_tweets)} tweets with date > dt_to:")
        for t in future_tweets[:10]:
            print(f"     date={t['date']}  created_at={t.get('created_at')}  @{t.get('author_username')}")
    else:
        print(f"  OK — no tweets beyond dt_to ({dt_to.date()})")

    # Check created_at vs date mismatches (timezone drift)
    print(f"\n=== created_at vs date mismatch (sample) ===")
    mismatches = []
    for t in tweets_raw:
        ca = t.get("created_at") or ""
        d  = t.get("date") or ""
        if ca and d:
            ca_date = ca[:10]   # "YYYY-MM-DD" prefix
            if ca_date != d:
                mismatches.append(t)
    print(f"  Tweets where created_at[:10] != date : {len(mismatches)}")
    for t in mismatches[:5]:
        print(f"    created_at={t.get('created_at')}  date={t.get('date')}  @{t.get('author_username')}")
    if not mismatches:
        print("  OK — created_at date matches 'date' field for all fetched tweets")

    # ── Continue filtering ───────────────────────────────────────────────────────
    tweets = tweets_raw
    if allowed_authors:
        tweets = [t for t in tweets if (t.get("author_username") or "").lower() in allowed_authors]
        print(f"\n  After author filter : {len(tweets)} tweets")

    tweets = [t for t in tweets if (t.get("signal_type") or "").upper() not in ("", "NO_CORRELATION_TO_BTC")]
    print(f"  After signal filter : {len(tweets)} tweets with actionable signal")

    # --- Шаг 1: группируем по дате ---
    by_date = _group_tweets_by_date(tweets)
    print("\n=== Step 1: group by date ===")
    for d, day_tweets in sorted(by_date.items()):
        print(f"  {d}: {len(day_tweets)} tweets")

    # --- Шаг 2: агрегируем по автору внутри каждой даты ---
    by_author = _aggregate_signals_by_author_and_date(by_date)
    print("\n=== Step 2: aggregate by author ===")
    for d, authors in sorted(by_author.items()):
        print(f"  {d}:")
        for author, sig in authors.items():
            print(f"    @{author}: {sig['signal_type']} conf={sig['signal_confidence']} avg={sig['avg_score']} ({sig['tweets_count']} tweets)")
        if not authors:
            print("    (no actionable signals)")

    # --- Шаг 3: один сигнал на дату ---
    by_date_merged = _merge_authors_signals_in_dates_into_one_signal(by_author)
    print("\n=== Step 3: one signal per date ===")
    for d, sig in sorted(by_date_merged.items()):
        print(f"  {d}: {sig['signal_type']} conf={sig['signal_confidence']} avg={sig['avg_score']} ({sig['authors_count']} authors)")

    # --- Шаг 4: финальный вердикт по всему окну ---
    verdict = _merge_date_signals_into_final_verdict(
        by_date_merged,
        decay_rate=decay_rate,
        decay_start_day=decay_start_day,
        initial_weight=initial_weight,
        reference_date=forecast_date,
    )
    print("\n=== Step 4: final verdict ===")
    if verdict:
        print(f"  {verdict['signal_type']} | conf={verdict['signal_confidence']} | avg_score={verdict['avg_score']} | over {verdict['dates_count']} days")
    else:
        print("  No actionable signal")


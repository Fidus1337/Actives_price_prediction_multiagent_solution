"""
LLM-based Twitter news classifier for BTC market signals.

Classifies tweets as:
- BEAR
- BULL
- NO_CORRELATION_TO_BTC

and confidence:
- LOW
- MIDDLE
- HIGH

Used by full_scrapping_pipeline.py before writing rows into SQLite.
"""

import json
from typing import Literal, cast
from pathlib import Path
import warnings

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

LOG_TAG = "[twitter_classifier]"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
load_dotenv(_PROJECT_ROOT / "dev.env")

_SETTINGS_PATH = Path(__file__).resolve().parent.parent / "twitter_collector_settings.json"


def _load_classifier_settings() -> dict:
    """Read classifier_settings block from twitter_collector_settings.json."""
    try:
        cfg = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        return cfg.get("classifier_settings", {})
    except Exception:
        return {}


# LangChain structured output can emit noisy pydantic serialization warnings for
# internal `parsed` field metadata; this does not affect classification results.
warnings.filterwarnings(
    "ignore",
    message=r"Pydantic serializer warnings:.*PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"pydantic\.main",
)


class TweetItem(BaseModel):
    tweet_id: str = Field(description="Stable tweet identifier from input payload")
    signal_type: Literal["BEAR", "BULL", "NO_CORRELATION_TO_BTC"]
    confidence: Literal["LOW", "MIDDLE", "HIGH"]


class TweetClassificationResponse(BaseModel):
    classifications: list[TweetItem]


def _build_classifier_prompt(horizon_days: int) -> str:
    """Build the classifier system prompt with the given prediction horizon."""
    return f"""\
You are a BTC market-impact classifier for Twitter posts.
Your goal: identify tweets that signal REAL near-term price impact on BTCUSDT.

For each tweet, classify impact on BTCUSDT:
- "BULL": likely positive for BTC price in the next {horizon_days} day(s).
- "BEAR": likely negative for BTC price in the next {horizon_days} day(s).
- "NO_CORRELATION_TO_BTC": no meaningful BTC price impact.

Confidence scale:
- "HIGH": hard data with direct price catalyst (ETF flows, regulatory decisions,
  large liquidations, major macro events like CPI/FOMC).
- "MIDDLE": meaningful market signal but indirect (on-chain trends,
  funding rate shifts, notable whale moves with clear direction).
- "LOW": weak, speculative, or ambiguous signal.

Critical rules:
1) Distinguish between ANALYTICAL opinions and PROMOTIONAL content:
   - Analytical opinions with specific reasoning, price levels, or data
     references -> classify normally (BULL/BEAR) with LOW or MIDDLE confidence.
   - Cheerleading, hype, and generic promotion without substance
     ("Bitcoin is a gift", "most bullish chart ever", "buy the dip",
     "Bitcoin has been declared dead 470 times") -> NO_CORRELATION_TO_BTC.
2) Treat analytical opinions seriously — even short directional calls from
   experienced analysts carry real market weight. Many tweets reference a chart
   or image that you cannot see; if the text clearly implies a directional
   conclusion ("breaking out", "support lost", "target hit"), treat the
   visual evidence as present and classify accordingly. Do NOT downgrade to
   NO_CORRELATION_TO_BTC just because no image text is visible.
3) On-chain transfers: deposits TO exchanges = potential selling pressure (BEAR).
   Withdrawals FROM exchanges = accumulation (BULL).
   Transfers between unknown wallets = NO_CORRELATION_TO_BTC.
4) Historical facts, memes, quotes, and educational content = NO_CORRELATION_TO_BTC.
5) If a tweet contains both bullish and bearish signals, classify by the
   dominant near-term price impact.
6) Be skeptical - most tweets do NOT move markets. When in doubt,
   classify as NO_CORRELATION_TO_BTC.
7) Classify every item. Return exact enums only. Preserve tweet_id exactly.
"""


def _choose_batch_size(total: int) -> int:
    if total < 20:
        return total
    if total <= 60:
        return 20
    return 30


def _prepare_for_classification(tweets: list[dict]) -> list[dict]:
    prepared = []
    for t in tweets:
        text = (t.get("text") or "").strip()
        prepared.append({
            "tweet_id": str(t.get("tweet_id", "")),
            "author": t.get("author_username", ""),
            "date": t.get("date", ""),
            "likes": int(t.get("likes", 0) or 0),
            "retweets": int(t.get("retweets", 0) or 0),
            "replies": int(t.get("replies", 0) or 0),
            "views": int(t.get("views", 0) or 0),
            "text": text[:500],
        })
    return prepared


def _apply_fallback(tweets: list[dict]) -> None:
    for t in tweets:
        if not t.get("signal_type"):
            t["signal_type"] = "NO_CORRELATION_TO_BTC"
            t["signal_confidence"] = "LOW"


def _short_text(text: str, max_len: int = 90) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def _log_classification(
    idx: int,
    total: int,
    tweet: dict,
    reason: str,
) -> None:
    tweet_id = str(tweet.get("tweet_id", ""))
    author = str(tweet.get("author_username", ""))
    date = str(tweet.get("date", ""))
    sig = str(tweet.get("signal_type", ""))
    conf = str(tweet.get("signal_confidence", ""))
    text = _short_text(str(tweet.get("text", "")))
    print(
        f"{LOG_TAG} [{idx}/{total}] id={tweet_id} author=@{author} date={date} "
        f"signal={sig} confidence={conf} reason={reason} text='{text}'"
    )

def _group_by_date(items: list[dict]) -> list[tuple[str, list[dict]]]:
    groups: dict[str, list[dict]] = {}
    for item in items:
        date_key = str(item.get("date") or "unknown")
        groups.setdefault(date_key, []).append(item)
    return sorted(groups.items(), key=lambda x: x[0], reverse=True)


def classify_tweets(
    tweets: list[dict],
    force_reclassify: bool = False,
    strict: bool = False,
) -> None:
    """Classify tweets in-place.

    Adds:
    - signal_type: BEAR/BULL/NO_CORRELATION_TO_BTC
    - signal_confidence: LOW/MIDDLE/HIGH

    Args:
        tweets: tweet dicts to classify in-place.
        force_reclassify: if True, reclassify every item regardless of existing labels.
        strict: if True, raise on LLM init/batch errors (no fallback-to-neutral).
    """
    if not tweets:
        return

    if force_reclassify:
        for t in tweets:
            t.pop("signal_type", None)
            t.pop("signal_confidence", None)

    total = len(tweets)
    idx_by_obj = {id(t): idx for idx, t in enumerate(tweets, start=1)}

    # Empty-text tweets are not useful for signal extraction.
    for idx, t in enumerate(tweets, start=1):
        if not (t.get("text") or "").strip():
            t["signal_type"] = "NO_CORRELATION_TO_BTC"
            t["signal_confidence"] = "LOW"
            _log_classification(idx, total, t, reason="empty_text_fallback")

    # Items that still do not have a valid label after preprocessing.
    unlabeled = [
        t for t in tweets
        if (t.get("signal_type") not in {"BULL", "BEAR", "NO_CORRELATION_TO_BTC"})
    ]
    if not unlabeled:
        return

    # Pre-filter: only potentially non-neutral items are sent to LLM.
    to_classify = []
    for t in unlabeled:
        to_classify.append(t)

    if not to_classify:
        return

    settings = _load_classifier_settings()
    model = settings.get("model", "gpt-4o-mini")
    horizon_days = int(settings.get("horizon_days", 7))
    classifier_prompt = _build_classifier_prompt(horizon_days)

    try:
        llm = ChatOpenAI(model=model, temperature=0.0)
    except Exception as e:
        msg = f"{LOG_TAG} ERROR init model: {e}"
        print(msg)
        if strict:
            raise RuntimeError(msg) from e
        _apply_fallback(to_classify)
        return

    by_id = {str(t.get("tweet_id", "")): t for t in to_classify}
    idx_by_id = {str(t.get("tweet_id", "")): idx for idx, t in enumerate(tweets, start=1)}
    prepared = _prepare_for_classification(to_classify)
    by_date = _group_by_date(prepared)
    global_batch_id = 0
    running_offset = 0

    # Iterate over tweets grouped by date (sorted newest first)
    for date_key, date_items in by_date:
        batch_size = 1  # Send 1 tweet per LLM call (conservative, avoids misalignment in structured output)
        print(f"{LOG_TAG} Date {date_key}: {len(date_items)} non-neutral candidates, batch_size={batch_size}")

        # Slide a window of batch_size over the day's tweets
        for i in range(0, len(date_items), batch_size):
            global_batch_id += 1                      # Unique ID for logging across all dates
            batch = date_items[i:i + batch_size]      # Current slice (1 tweet here)
            print(f"{LOG_TAG} Batch {global_batch_id} ...")

            try:
                payload = json.dumps(batch, ensure_ascii=False)  # Serialize tweet(s) to JSON string

                # Send to LLM and force the response into TweetClassificationResponse schema
                result = cast(
                    TweetClassificationResponse,
                    llm.with_structured_output(TweetClassificationResponse).invoke([
                        SystemMessage(content=classifier_prompt),           # System rules for classification
                        HumanMessage(content=f"Classify these {len(batch)} tweets:\n{payload}"),  # The tweet data
                    ]),
                )

                # Build a lookup: tweet_id -> TweetItem (from LLM response)
                mapped = {item.tweet_id: item for item in result.classifications}

                # Write classification results back into the original tweet dicts
                for item in batch:
                    tw = by_id.get(item["tweet_id"])   # Get the original dict from the input list
                    if tw is None:
                        continue                        # Should not happen, safety guard

                    idx = idx_by_id.get(item["tweet_id"], 0)
                    cls = mapped.get(item["tweet_id"]) # Look up what the LLM returned for this tweet

                    if cls is None:
                        # LLM returned a response but omitted this tweet_id — fallback to neutral
                        tw["signal_type"] = "NO_CORRELATION_TO_BTC"
                        tw["signal_confidence"] = "LOW"
                        _log_classification(idx, total, tw, reason="missing_llm_item_fallback")
                    else:
                        # Happy path: write LLM classification into the original dict (in-place)
                        tw["signal_type"] = cls.signal_type
                        tw["signal_confidence"] = cls.confidence
                        _log_classification(idx, total, tw, reason="llm")

            except Exception as e:
                # LLM call failed entirely (network error, parsing error, etc.)
                print(f"{LOG_TAG} ERROR batch {global_batch_id}: {e}")
                if strict:
                    raise RuntimeError(...) from e     # Propagate if caller wants hard failure

                # Otherwise fallback: mark every tweet in the failed batch as neutral
                for item in batch:
                    tw = by_id.get(item["tweet_id"])
                    if tw is not None:
                        tw["signal_type"] = "NO_CORRELATION_TO_BTC"
                        tw["signal_confidence"] = "LOW"
                        _log_classification(idx, total, tw, reason="batch_error_fallback")

        running_offset += len(date_items)  # Track position across dates for accurate log counters

"""Unit tests for the four pure aggregation functions used by agent_for_twitter_analysis.

Run:
    .venv/Scripts/python.exe -m unittest MultiagentSystem.tests.test_twitter_aggregation -v
"""
import sys
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

_MULTIAGENT_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _MULTIAGENT_DIR.parent
for _p in (str(_MULTIAGENT_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agents.twitter_analyser.agent_for_twitter_analysis import (
    _get_window_dates,
    _group_tweets_by_date,
    _aggregate_signals_by_author_and_date,
    _merge_authors_signals_in_dates_into_one_signal,
    _merge_date_signals_into_final_verdict,
)


def _tweet(date_str, author, signal_type, signal_confidence):
    return {
        "date": date_str,
        "author_username": author,
        "signal_type": signal_type,
        "signal_confidence": signal_confidence,
    }


class TestGetWindowDates(unittest.TestCase):
    def test_string_input_window_14(self):
        dt_from, dt_to = _get_window_dates("2026-04-26", 14)
        self.assertEqual(dt_to, datetime(2026, 4, 26, tzinfo=timezone.utc))
        self.assertEqual(dt_from, datetime(2026, 4, 13, tzinfo=timezone.utc))

    def test_window_one_day_means_dt_from_equals_dt_to(self):
        dt_from, dt_to = _get_window_dates("2026-04-26", 1)
        self.assertEqual(dt_from, dt_to)

    def test_date_object_input(self):
        dt_from, dt_to = _get_window_dates(date(2026, 4, 26), 7)
        self.assertEqual(dt_to, datetime(2026, 4, 26, tzinfo=timezone.utc))
        self.assertEqual(dt_from, datetime(2026, 4, 20, tzinfo=timezone.utc))

    def test_naive_datetime_gets_utc(self):
        dt_from, dt_to = _get_window_dates(datetime(2026, 4, 26, 12, 0), 3)
        self.assertEqual(dt_to.tzinfo, timezone.utc)
        self.assertEqual(dt_to.hour, 12)
        self.assertEqual(dt_from, dt_to.replace(day=24))

    def test_aware_datetime_preserved(self):
        aware = datetime(2026, 4, 26, 12, 0, tzinfo=timezone.utc)
        _, dt_to = _get_window_dates(aware, 3)
        self.assertIs(dt_to, aware)

    def test_both_returned_dates_are_tz_aware(self):
        dt_from, dt_to = _get_window_dates("2026-04-26", 5)
        self.assertIsNotNone(dt_from.tzinfo)
        self.assertIsNotNone(dt_to.tzinfo)


class TestGroupTweetsByDate(unittest.TestCase):
    def test_empty_list_returns_empty_dict(self):
        self.assertEqual(_group_tweets_by_date([]), {})

    def test_groups_by_date_string(self):
        tweets = [
            _tweet("2026-04-25", "alice", "BULL", "HIGH"),
            _tweet("2026-04-25", "bob", "BEAR", "LOW"),
            _tweet("2026-04-26", "alice", "BULL", "MIDDLE"),
        ]
        grouped = _group_tweets_by_date(tweets)
        self.assertEqual(set(grouped), {"2026-04-25", "2026-04-26"})
        self.assertEqual(len(grouped["2026-04-25"]), 2)
        self.assertEqual(len(grouped["2026-04-26"]), 1)

    def test_tweets_without_date_dropped(self):
        tweets = [
            _tweet(None, "alice", "BULL", "HIGH"),
            _tweet("", "bob", "BULL", "HIGH"),
            _tweet("2026-04-25", "carol", "BULL", "HIGH"),
        ]
        grouped = _group_tweets_by_date(tweets)
        self.assertEqual(set(grouped), {"2026-04-25"})
        self.assertEqual(len(grouped["2026-04-25"]), 1)


class TestAggregateSignalsByAuthorAndDate(unittest.TestCase):
    def test_single_bull_high_tweet_per_author(self):
        grouped = {
            "2026-04-25": [_tweet("2026-04-25", "alice", "BULL", "HIGH")],
        }
        result = _aggregate_signals_by_author_and_date(grouped)
        self.assertEqual(result["2026-04-25"]["alice"]["signal_type"], "BULL")
        self.assertEqual(result["2026-04-25"]["alice"]["signal_confidence"], 3)
        self.assertEqual(result["2026-04-25"]["alice"]["avg_score"], 3.0)
        self.assertEqual(result["2026-04-25"]["alice"]["tweets_count"], 1)

    def test_two_tweets_one_author_averages(self):
        # BULL HIGH (+3) and BEAR LOW (-1) → avg = 1 → BULL/1
        grouped = {
            "2026-04-25": [
                _tweet("2026-04-25", "alice", "BULL", "HIGH"),
                _tweet("2026-04-25", "alice", "BEAR", "LOW"),
            ],
        }
        result = _aggregate_signals_by_author_and_date(grouped)
        self.assertEqual(result["2026-04-25"]["alice"]["signal_type"], "BULL")
        self.assertEqual(result["2026-04-25"]["alice"]["signal_confidence"], 1)
        self.assertEqual(result["2026-04-25"]["alice"]["avg_score"], 1.0)
        self.assertEqual(result["2026-04-25"]["alice"]["tweets_count"], 2)

    def test_zero_confidence_author_dropped(self):
        # BULL LOW (+1) and BEAR LOW (-1) → avg=0, round(abs(0))=0 → drop
        grouped = {
            "2026-04-25": [
                _tweet("2026-04-25", "alice", "BULL", "LOW"),
                _tweet("2026-04-25", "alice", "BEAR", "LOW"),
            ],
        }
        result = _aggregate_signals_by_author_and_date(grouped)
        self.assertEqual(result["2026-04-25"], {})

    def test_unknown_signal_type_dropped(self):
        grouped = {
            "2026-04-25": [
                _tweet("2026-04-25", "alice", "NEUTRAL", "HIGH"),
                _tweet("2026-04-25", "alice", "BULL", "MIDDLE"),
            ],
        }
        result = _aggregate_signals_by_author_and_date(grouped)
        # Only BULL MIDDLE counted: +2
        self.assertEqual(result["2026-04-25"]["alice"]["signal_confidence"], 2)
        self.assertEqual(result["2026-04-25"]["alice"]["tweets_count"], 1)

    def test_two_authors_independent(self):
        grouped = {
            "2026-04-25": [
                _tweet("2026-04-25", "alice", "BULL", "HIGH"),
                _tweet("2026-04-25", "bob", "BEAR", "MIDDLE"),
            ],
        }
        result = _aggregate_signals_by_author_and_date(grouped)
        self.assertEqual(result["2026-04-25"]["alice"]["signal_type"], "BULL")
        self.assertEqual(result["2026-04-25"]["alice"]["signal_confidence"], 3)
        self.assertEqual(result["2026-04-25"]["bob"]["signal_type"], "BEAR")
        self.assertEqual(result["2026-04-25"]["bob"]["signal_confidence"], 2)

    def test_author_username_lowercased(self):
        grouped = {
            "2026-04-25": [_tweet("2026-04-25", "ALICE", "BULL", "HIGH")],
        }
        result = _aggregate_signals_by_author_and_date(grouped)
        self.assertIn("alice", result["2026-04-25"])
        self.assertNotIn("ALICE", result["2026-04-25"])

    def test_missing_signal_confidence_defaults_to_low(self):
        grouped = {
            "2026-04-25": [{
                "date": "2026-04-25",
                "author_username": "alice",
                "signal_type": "BULL",
                "signal_confidence": None,
            }],
        }
        result = _aggregate_signals_by_author_and_date(grouped)
        # BULL + None confidence → defaults to "LOW" → +1
        self.assertEqual(result["2026-04-25"]["alice"]["signal_confidence"], 1)


class TestMergeAuthorsSignalsInDatesIntoOneSignal(unittest.TestCase):
    def test_single_author_per_date_passthrough(self):
        aggregated = {
            "2026-04-25": {
                "alice": {"signal_type": "BULL", "signal_confidence": 3, "avg_score": 3.0, "tweets_count": 1},
            },
        }
        result = _merge_authors_signals_in_dates_into_one_signal(aggregated)
        self.assertEqual(result["2026-04-25"]["signal_type"], "BULL")
        self.assertEqual(result["2026-04-25"]["signal_confidence"], 3)
        self.assertEqual(result["2026-04-25"]["authors_count"], 1)

    def test_two_authors_average(self):
        # +3 (BULL HIGH) and -1 (BEAR LOW) → avg = 1 → BULL conf=1
        aggregated = {
            "2026-04-25": {
                "alice": {"signal_type": "BULL", "signal_confidence": 3, "avg_score": 3.0, "tweets_count": 1},
                "bob":   {"signal_type": "BEAR", "signal_confidence": 1, "avg_score": -1.0, "tweets_count": 1},
            },
        }
        result = _merge_authors_signals_in_dates_into_one_signal(aggregated)
        self.assertEqual(result["2026-04-25"]["signal_type"], "BULL")
        self.assertEqual(result["2026-04-25"]["signal_confidence"], 1)
        self.assertEqual(result["2026-04-25"]["avg_score"], 1.0)
        self.assertEqual(result["2026-04-25"]["authors_count"], 2)

    def test_tied_signals_drop_date(self):
        # +1 and -1 → avg=0 → date dropped
        aggregated = {
            "2026-04-25": {
                "alice": {"signal_type": "BULL", "signal_confidence": 1, "avg_score": 1.0, "tweets_count": 1},
                "bob":   {"signal_type": "BEAR", "signal_confidence": 1, "avg_score": -1.0, "tweets_count": 1},
            },
        }
        result = _merge_authors_signals_in_dates_into_one_signal(aggregated)
        self.assertNotIn("2026-04-25", result)

    def test_empty_authors_dict_dropped(self):
        aggregated = {"2026-04-25": {}, "2026-04-26": {
            "alice": {"signal_type": "BULL", "signal_confidence": 2, "avg_score": 2.0, "tweets_count": 1},
        }}
        result = _merge_authors_signals_in_dates_into_one_signal(aggregated)
        self.assertNotIn("2026-04-25", result)
        self.assertIn("2026-04-26", result)

    def test_three_authors_majority_bear(self):
        # -3, -2, +1 → avg = -4/3 ≈ -1.333 → round(abs)=1 → BEAR
        aggregated = {
            "2026-04-25": {
                "alice":   {"signal_type": "BEAR", "signal_confidence": 3, "avg_score": -3.0, "tweets_count": 1},
                "bob":     {"signal_type": "BEAR", "signal_confidence": 2, "avg_score": -2.0, "tweets_count": 1},
                "charlie": {"signal_type": "BULL", "signal_confidence": 1, "avg_score":  1.0, "tweets_count": 1},
            },
        }
        result = _merge_authors_signals_in_dates_into_one_signal(aggregated)
        self.assertEqual(result["2026-04-25"]["signal_type"], "BEAR")
        self.assertEqual(result["2026-04-25"]["signal_confidence"], 1)
        self.assertEqual(result["2026-04-25"]["authors_count"], 3)


class TestMergeDateSignalsIntoFinalVerdict(unittest.TestCase):
    """The decay function:
        age < decay_start_day:  weight = 1.0
        age >= decay_start_day: weight = initial_weight * (1 - decay_rate) ** (age - decay_start_day)
    """

    def test_empty_signals_returns_none(self):
        result = _merge_date_signals_into_final_verdict(
            {}, decay_rate=0.05, decay_start_day=1, initial_weight=1.0,
            reference_date="2026-04-26",
        )
        self.assertIsNone(result)

    def test_single_fresh_day_passthrough(self):
        # age 0, decay_start_day=1 → fresh zone → weight 1.0
        signals = {
            "2026-04-26": {"signal_type": "BULL", "signal_confidence": 2, "avg_score": 2.0, "authors_count": 1},
        }
        result = _merge_date_signals_into_final_verdict(
            signals, decay_rate=0.05, decay_start_day=1, initial_weight=1.0,
            reference_date="2026-04-26",
        )
        self.assertEqual(result["signal_type"], "BULL")
        self.assertEqual(result["signal_confidence"], 2)
        self.assertEqual(result["dates_count"], 1)
        self.assertAlmostEqual(result["avg_score"], 2.0)

    def test_decay_zone_single_day_uses_initial_weight(self):
        # age 1, decay_start_day=1 → age >= decay_start_day → t=0 → weight = initial_weight * 1.0
        signals = {
            "2026-04-25": {"signal_type": "BULL", "signal_confidence": 2, "avg_score": 2.0, "authors_count": 1},
        }
        result = _merge_date_signals_into_final_verdict(
            signals, decay_rate=0.05, decay_start_day=1, initial_weight=0.5,
            reference_date="2026-04-26",
        )
        # Single day: weighted_score / total_weight = (2 * 0.5) / 0.5 = 2.0
        self.assertEqual(result["signal_type"], "BULL")
        self.assertAlmostEqual(result["avg_score"], 2.0)

    def test_two_opposite_days_fresh_zone_balance(self):
        # +3 and -3 with same weight → avg = 0 → weak verdict (signal_type=None,
        # signal_confidence=0), avg_score still surfaced for downstream visibility.
        signals = {
            "2026-04-26": {"signal_type": "BULL", "signal_confidence": 3, "avg_score": 3.0, "authors_count": 1},
            "2026-04-25": {"signal_type": "BEAR", "signal_confidence": 3, "avg_score": -3.0, "authors_count": 1},
        }
        result = _merge_date_signals_into_final_verdict(
            signals, decay_rate=0.0, decay_start_day=10, initial_weight=1.0,
            reference_date="2026-04-26",
        )
        self.assertIsNotNone(result)
        self.assertIsNone(result["signal_type"])
        self.assertEqual(result["signal_confidence"], 0)
        self.assertAlmostEqual(result["avg_score"], 0.0)
        self.assertEqual(result["dates_count"], 2)

    def test_decay_reduces_old_day_influence(self):
        # Today: +1 (BULL, weight 1.0); 14 days ago: +3 (BULL but heavily decayed)
        # decay_start_day=1, age 14 → t=13 → weight = 1.0 * 0.95^13 ≈ 0.5133
        # avg = (1*1.0 + 3*0.5133) / (1.0 + 0.5133) ≈ 1.678
        signals = {
            "2026-04-26": {"signal_type": "BULL", "signal_confidence": 1, "avg_score": 1.0, "authors_count": 1},
            "2026-04-12": {"signal_type": "BULL", "signal_confidence": 3, "avg_score": 3.0, "authors_count": 1},
        }
        result = _merge_date_signals_into_final_verdict(
            signals, decay_rate=0.05, decay_start_day=1, initial_weight=1.0,
            reference_date="2026-04-26",
        )
        self.assertEqual(result["signal_type"], "BULL")
        self.assertEqual(result["signal_confidence"], 2)
        # Recompute exactly: weight_old = 0.95**13
        weight_old = 0.95 ** 13
        expected_avg = (1.0 * 1.0 + 3.0 * weight_old) / (1.0 + weight_old)
        self.assertAlmostEqual(result["avg_score"], round(expected_avg, 3), places=3)
        self.assertEqual(result["dates_count"], 2)

    def test_zero_avg_returns_weak_verdict(self):
        # Build signals that exactly cancel — function still returns a dict so
        # avg_score is observable downstream, but signal_type is None and
        # signal_confidence is 0 (no vote contributed to general forecast).
        signals = {
            "2026-04-26": {"signal_type": "BULL", "signal_confidence": 2, "avg_score": 2.0, "authors_count": 1},
            "2026-04-25": {"signal_type": "BEAR", "signal_confidence": 2, "avg_score": -2.0, "authors_count": 1},
        }
        result = _merge_date_signals_into_final_verdict(
            signals, decay_rate=0.0, decay_start_day=10, initial_weight=1.0,
            reference_date="2026-04-26",
        )
        self.assertIsNotNone(result)
        self.assertIsNone(result["signal_type"])
        self.assertEqual(result["signal_confidence"], 0)
        self.assertAlmostEqual(result["avg_score"], 0.0)
        self.assertEqual(result["dates_count"], 2)

    def test_invalid_date_string_skipped(self):
        signals = {
            "not-a-date": {"signal_type": "BULL", "signal_confidence": 3, "avg_score": 3.0, "authors_count": 1},
            "2026-04-26": {"signal_type": "BULL", "signal_confidence": 1, "avg_score": 1.0, "authors_count": 1},
        }
        result = _merge_date_signals_into_final_verdict(
            signals, decay_rate=0.05, decay_start_day=1, initial_weight=1.0,
            reference_date="2026-04-26",
        )
        # Only the valid date contributes
        self.assertEqual(result["dates_count"], 1)
        self.assertEqual(result["signal_type"], "BULL")
        self.assertEqual(result["signal_confidence"], 1)

    def test_reference_date_as_date_object(self):
        signals = {
            "2026-04-26": {"signal_type": "BULL", "signal_confidence": 2, "avg_score": 2.0, "authors_count": 1},
        }
        result = _merge_date_signals_into_final_verdict(
            signals, decay_rate=0.05, decay_start_day=1, initial_weight=1.0,
            reference_date=date(2026, 4, 26),
        )
        self.assertEqual(result["signal_type"], "BULL")
        self.assertEqual(result["signal_confidence"], 2)

    def test_reference_date_as_datetime_object(self):
        signals = {
            "2026-04-26": {"signal_type": "BEAR", "signal_confidence": 3, "avg_score": -3.0, "authors_count": 1},
        }
        result = _merge_date_signals_into_final_verdict(
            signals, decay_rate=0.05, decay_start_day=1, initial_weight=1.0,
            reference_date=datetime(2026, 4, 26, 15, 0),
        )
        self.assertEqual(result["signal_type"], "BEAR")
        self.assertEqual(result["signal_confidence"], 3)

    def test_future_date_clamped_to_age_zero(self):
        # date "after" reference_date → max(today-day, 0) = 0 → fresh zone
        signals = {
            "2026-04-30": {"signal_type": "BULL", "signal_confidence": 2, "avg_score": 2.0, "authors_count": 1},
        }
        result = _merge_date_signals_into_final_verdict(
            signals, decay_rate=0.5, decay_start_day=1, initial_weight=0.0,
            reference_date="2026-04-26",
        )
        # age clamped to 0 → fresh zone → weight 1.0
        self.assertEqual(result["signal_confidence"], 2)


if __name__ == "__main__":
    unittest.main()

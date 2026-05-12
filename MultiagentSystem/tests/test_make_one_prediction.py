"""End-to-end integration test for make_one_prediction.

Strategy: enable ONLY the twitter agent (formula-based, no LLM call) and let the
LangGraph DAG run the full supervisor → validator → reports_analyser path.
The validator early-exits when the resolved model is `claude-*` and
CLAUDE_KEY is not set, so no LLM clients are instantiated. The other agent
nodes short-circuit because they are absent from agent_envolved_in_prediction.

Run:
    .venv/Scripts/python.exe -m unittest MultiagentSystem.tests.test_make_one_prediction -v
"""
import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pandas as pd

_MULTIAGENT_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _MULTIAGENT_DIR.parent
for _p in (str(_MULTIAGENT_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _tweet(date_str, author, signal_type, signal_confidence):
    return {
        "date": date_str,
        "author_username": author,
        "signal_type": signal_type,
        "signal_confidence": signal_confidence,
    }


def _base_config(authors_filter, forecast_date):
    return {
        "forecast_start_date": forecast_date,
        "horizon": 1,
        "neutral_threshold": 0.0,
        "agent_envolved_in_prediction": ["agent_for_twitter_analysis"],
        "agent_settings": {
            "agent_for_twitter_analysis": {
                "window_to_analysis": 14,
                "decay_rate": 0.05,
                "decay_start_day": 1,
                "initial_weight": 1.0,
                "authors": authors_filter,
            },
            "verdicts_validator": {"llm_model": "claude-sonnet-4-5"},
        },
    }


class TestMakeOnePrediction(unittest.TestCase):
    """Integration test that walks the full LangGraph DAG end-to-end."""

    @classmethod
    def setUpClass(cls):
        os.environ.pop("CLAUDE_KEY", None)
        # Import graph after env is clean.
        from MultiagentSystem.multiagent_graph import app
        from MultiagentSystem.multiagent_predictions_module import make_one_prediction
        cls.app = app
        cls.make_one_prediction = staticmethod(make_one_prediction)

    def _run_with_tweets(self, tweets, forecast_date, authors_filter):
        """Patch get_tweets_in_range and invoke make_one_prediction."""
        # The function is imported into the agent module; patch it there.
        with mock.patch(
            "MultiagentSystem.agents.twitter_analyser.agent_for_twitter_analysis.get_tweets_in_range",
            return_value=tweets,
        ):
            return self.make_one_prediction(
                self.app,
                _base_config(authors_filter, forecast_date),
                forecast_date,
                cached_dataset=pd.DataFrame(),
            )

    def test_uniform_bull_signals_yield_long(self):
        # 3 BULL HIGH tweets across 3 different days from one author → strong LONG
        forecast = "2026-04-26"
        tweets = [
            _tweet("2026-04-26", "alice", "BULL", "HIGH"),
            _tweet("2026-04-25", "alice", "BULL", "HIGH"),
            _tweet("2026-04-24", "alice", "BULL", "HIGH"),
        ]
        row = self._run_with_tweets(tweets, forecast, ["alice"])

        self.assertEqual(row["forecast_start_date"], forecast)
        self.assertEqual(row["y_predict"], "LONG")
        self.assertGreater(row["y_predict_confidence"], 2.0)
        # Per-agent flatten present
        self.assertEqual(row["twitter_analysis__prediction"], True)
        self.assertEqual(row["twitter_analysis__confidence"], "high")

    def test_uniform_bear_signals_yield_short(self):
        forecast = "2026-04-26"
        tweets = [
            _tweet("2026-04-26", "alice", "BEAR", "HIGH"),
            _tweet("2026-04-25", "alice", "BEAR", "HIGH"),
        ]
        row = self._run_with_tweets(tweets, forecast, ["alice"])
        self.assertEqual(row["y_predict"], "SHORT")
        self.assertLess(row["y_predict_confidence"], -2.0)
        self.assertEqual(row["twitter_analysis__prediction"], False)
        self.assertEqual(row["twitter_analysis__confidence"], "high")

    def test_no_actionable_signals_yields_neutral(self):
        forecast = "2026-04-26"
        # All tweets are NO_CORRELATION → twitter agent abstains
        tweets = [
            _tweet("2026-04-26", "alice", "NO_CORRELATION_TO_BTC", "HIGH"),
            _tweet("2026-04-25", "alice", "NO_CORRELATION_TO_BTC", "MIDDLE"),
        ]
        row = self._run_with_tweets(tweets, forecast, ["alice"])

        # Twitter agent abstained → no votes → neutral verdict
        self.assertIsNone(row["y_predict"])
        self.assertEqual(row["y_predict_confidence"], 0.0)
        self.assertIsNone(row["twitter_analysis__prediction"])
        self.assertIsNone(row["twitter_analysis__confidence"])

    def test_author_filter_excludes_unwanted(self):
        forecast = "2026-04-26"
        # Bob is BEAR HIGH but filtered out; alice is BULL HIGH → LONG
        tweets = [
            _tweet("2026-04-26", "alice", "BULL", "HIGH"),
            _tweet("2026-04-26", "bob",   "BEAR", "HIGH"),
        ]
        row = self._run_with_tweets(tweets, forecast, ["alice"])
        self.assertEqual(row["y_predict"], "LONG")
        self.assertEqual(row["twitter_analysis__prediction"], True)

    def test_neutral_threshold_in_config_filters_low_score(self):
        """When neutral_threshold > |score|, reports_analyser returns NEUTRAL."""
        forecast = "2026-04-26"
        tweets = [_tweet("2026-04-26", "alice", "BULL", "LOW")]  # vote = +1
        cfg = _base_config(["alice"], forecast)
        cfg["neutral_threshold"] = 2.0  # raise threshold above the resulting score

        with mock.patch(
            "MultiagentSystem.agents.twitter_analyser.agent_for_twitter_analysis.get_tweets_in_range",
            return_value=tweets,
        ):
            row = self.make_one_prediction(self.app, cfg, forecast, pd.DataFrame())

        self.assertIsNone(row["y_predict"])  # falls in neutral band

    def test_columns_returned_match_documented_schema(self):
        """Schema regression: row must contain the documented columns."""
        forecast = "2026-04-26"
        tweets = [_tweet("2026-04-26", "alice", "BULL", "HIGH")]
        row = self._run_with_tweets(tweets, forecast, ["alice"])
        for col in (
            "forecast_start_date",
            "y_predict",
            "y_predict_confidence",
            "summary",
            "reasoning",
            "risks",
            "twitter_analysis__prediction",
            "twitter_analysis__confidence",
        ):
            self.assertIn(col, row, f"missing column {col!r} in returned row")


if __name__ == "__main__":
    unittest.main()

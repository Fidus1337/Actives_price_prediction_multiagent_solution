"""Unit tests for compute_confidence_score (the multiagent ensemble aggregator).

Run:
    .venv/Scripts/python.exe -m unittest MultiagentSystem.tests.test_reports_analyser -v
"""
import sys
import unittest
from pathlib import Path

_MULTIAGENT_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _MULTIAGENT_DIR.parent
for _p in (str(_MULTIAGENT_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agents.reports_analyser.agent_for_reports_analysis import (
    compute_confidence_score,
    CONFIDENCE_WEIGHTS,
)


def _signal(prediction, confidence):
    return {"prediction": prediction, "confidence": confidence}


class TestConfidenceWeightsTable(unittest.TestCase):
    def test_weights_match_documented_scale(self):
        # Documented in CLAUDE.md: low=1, medium=2, high=3
        self.assertEqual(CONFIDENCE_WEIGHTS["low"], 1)
        self.assertEqual(CONFIDENCE_WEIGHTS["medium"], 2)
        self.assertEqual(CONFIDENCE_WEIGHTS["high"], 3)


class TestComputeConfidenceScore(unittest.TestCase):
    def test_no_signals_returns_zero_neutral(self):
        score, direction, breakdown = compute_confidence_score({}, neutral_threshold=0.0)
        self.assertEqual(score, 0.0)
        self.assertIsNone(direction)
        self.assertEqual(breakdown, "No real reports")

    def test_all_agents_abstain_returns_zero(self):
        signals = {
            "tech":    _signal(None, None),
            "twitter": _signal(True, None),
            "news":    _signal(None, "high"),
        }
        score, direction, _ = compute_confidence_score(signals, neutral_threshold=0.0)
        self.assertEqual(score, 0.0)
        self.assertIsNone(direction)

    def test_single_agent_long_high(self):
        signals = {"tech": _signal(True, "high")}
        score, direction, _ = compute_confidence_score(signals, neutral_threshold=0.0)
        self.assertEqual(score, 3.0)
        self.assertEqual(direction, "LONG")

    def test_single_agent_short_low(self):
        signals = {"tech": _signal(False, "low")}
        score, direction, _ = compute_confidence_score(signals, neutral_threshold=0.0)
        self.assertEqual(score, -1.0)
        self.assertEqual(direction, "SHORT")

    def test_two_agents_both_long_medium(self):
        # +2 + +2 → mean 2.0 → LONG
        signals = {
            "tech":    _signal(True, "medium"),
            "twitter": _signal(True, "medium"),
        }
        score, direction, _ = compute_confidence_score(signals, neutral_threshold=0.0)
        self.assertEqual(score, 2.0)
        self.assertEqual(direction, "LONG")

    def test_two_agents_long_low_long_medium_yields_one_point_five(self):
        # +1 + +2 → mean 1.5 → LONG (this is the case the user repeatedly hit)
        signals = {
            "twitter": _signal(True, "low"),
            "tech":    _signal(True, "medium"),
        }
        score, direction, _ = compute_confidence_score(signals, neutral_threshold=0.0)
        self.assertEqual(score, 1.5)
        self.assertEqual(direction, "LONG")

    def test_two_agents_opposite_with_threshold_zero(self):
        # +1 (long low) + -2 (short medium) → mean -0.5 → SHORT (threshold=0)
        signals = {
            "twitter": _signal(True, "low"),
            "tech":    _signal(False, "medium"),
        }
        score, direction, _ = compute_confidence_score(signals, neutral_threshold=0.0)
        self.assertEqual(score, -0.5)
        self.assertEqual(direction, "SHORT")

    def test_neutral_threshold_one_filters_low_score(self):
        # Same -0.5 score, but threshold=1.0 → falls in neutral band → None
        signals = {
            "twitter": _signal(True, "low"),
            "tech":    _signal(False, "medium"),
        }
        score, direction, _ = compute_confidence_score(signals, neutral_threshold=1.0)
        self.assertEqual(score, -0.5)
        self.assertIsNone(direction)

    def test_neutral_threshold_strict_gt_lt(self):
        # score == threshold should NOT cross — direction is None at exactly the boundary
        signals = {"tech": _signal(True, "low")}  # score = +1.0
        _, direction, _ = compute_confidence_score(signals, neutral_threshold=1.0)
        self.assertIsNone(direction)

        # score just above threshold → LONG
        signals = {"tech": _signal(True, "medium")}  # score = +2.0
        _, direction, _ = compute_confidence_score(signals, neutral_threshold=1.0)
        self.assertEqual(direction, "LONG")

    def test_abstain_excluded_from_mean(self):
        # +3 (high LONG) + abstain → mean = 3.0 / 1 = 3.0 (NOT 1.5)
        signals = {
            "tech":    _signal(True, "high"),
            "twitter": _signal(None, None),
        }
        score, direction, _ = compute_confidence_score(signals, neutral_threshold=0.0)
        self.assertEqual(score, 3.0)
        self.assertEqual(direction, "LONG")

    def test_three_agents_majority_long(self):
        # +3, +1, -2 → mean = 2/3 ≈ 0.667 → LONG (with threshold=0)
        signals = {
            "tech":    _signal(True, "high"),
            "twitter": _signal(True, "low"),
            "news":    _signal(False, "medium"),
        }
        score, direction, _ = compute_confidence_score(signals, neutral_threshold=0.0)
        self.assertAlmostEqual(score, 2.0 / 3.0, places=6)
        self.assertEqual(direction, "LONG")

    def test_score_stays_in_minus3_plus3_range(self):
        # 3 agents all max → still bounded by ±3
        signals = {
            "a": _signal(True, "high"),
            "b": _signal(True, "high"),
            "c": _signal(True, "high"),
        }
        score, _, _ = compute_confidence_score(signals)
        self.assertEqual(score, 3.0)

        signals = {
            "a": _signal(False, "high"),
            "b": _signal(False, "high"),
            "c": _signal(False, "high"),
        }
        score, _, _ = compute_confidence_score(signals)
        self.assertEqual(score, -3.0)

    def test_unknown_confidence_falls_back_to_weight_one(self):
        # confidence="UNKNOWN" → CONFIDENCE_WEIGHTS.get returns 1
        signals = {"tech": _signal(True, "UNKNOWN")}
        score, _, _ = compute_confidence_score(signals, neutral_threshold=0.0)
        self.assertEqual(score, 1.0)

    def test_confidence_case_and_whitespace_tolerated(self):
        # confidence "  HIGH  " should normalize to "high" → weight 3
        signals = {"tech": _signal(True, "  HIGH  ")}
        score, _, _ = compute_confidence_score(signals, neutral_threshold=0.0)
        self.assertEqual(score, 3.0)

    def test_breakdown_contains_each_agent(self):
        signals = {
            "tech":    _signal(True, "medium"),
            "twitter": _signal(False, "low"),
        }
        _, _, breakdown = compute_confidence_score(signals, neutral_threshold=0.0)
        self.assertIn("tech", breakdown)
        self.assertIn("twitter", breakdown)
        self.assertIn("HIGHER", breakdown)
        self.assertIn("LOWER", breakdown)
        self.assertIn("mean(2 agents)", breakdown)

    def test_breakdown_marks_skipped_agents(self):
        signals = {
            "tech":    _signal(True, "medium"),
            "twitter": _signal(None, None),
        }
        _, _, breakdown = compute_confidence_score(signals, neutral_threshold=0.0)
        self.assertIn("twitter: no vote (skipped)", breakdown)
        self.assertIn("mean(1 agents)", breakdown)


if __name__ == "__main__":
    unittest.main()

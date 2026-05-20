"""Tests for WebShop failure detector rules and judge prompt guidance."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.webshop_failure_detector import JUDGE_PROMPT, WebShopFailureDetector


class WebShopFailureDetectorTests(unittest.TestCase):
    def test_detects_repeated_short_action_pattern(self):
        detector = WebShopFailureDetector(loop_window=3)
        history = [
            "search[capri sun pacific cooler]",
            "click[item - capri sun pacific cooler]",
            "click[back to search]",
            "search[capri sun pacific cooler]",
            "click[item - capri sun pacific cooler]",
            "click[back to search]",
        ]

        result = detector.detect(
            "WebShop\nInstruction: i want capri sun",
            history[-1],
            history,
        )

        self.assertTrue(result.is_failure)
        self.assertEqual(result.failure_type, "action_loop")

    def test_judge_prompt_separates_exploration_from_failures(self):
        self.assertIn("Do NOT mark normal exploration as unproductive", JUDGE_PROMPT)
        self.assertIn("first time", JUDGE_PROMPT)
        self.assertIn("search -> same item -> back to search", JUDGE_PROMPT)


if __name__ == "__main__":
    unittest.main()

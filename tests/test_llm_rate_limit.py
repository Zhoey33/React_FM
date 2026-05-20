"""Tests for shared LLM token-per-minute throttling."""

import unittest
from unittest.mock import patch

from src.llm import RollingTokenRateLimiter, _retry_wait_seconds
from experiments.run_webshop import parse_args, validate_llm_throttle_args


class RollingTokenRateLimiterTests(unittest.TestCase):
    def test_waits_until_window_has_budget(self):
        now = [1000.0]
        sleeps = []

        def fake_time():
            return now[0]

        def fake_sleep(seconds):
            sleeps.append(seconds)
            now[0] += seconds

        limiter = RollingTokenRateLimiter(
            tokens_per_minute=100,
            window_seconds=60,
            time_fn=fake_time,
            sleep_fn=fake_sleep,
        )

        limiter.acquire(80)
        limiter.acquire(30)

        self.assertEqual(sleeps, [60])

    def test_actual_usage_updates_reserved_budget(self):
        now = [1000.0]
        sleeps = []

        def fake_time():
            return now[0]

        def fake_sleep(seconds):
            sleeps.append(seconds)
            now[0] += seconds

        limiter = RollingTokenRateLimiter(
            tokens_per_minute=100,
            window_seconds=60,
            time_fn=fake_time,
            sleep_fn=fake_sleep,
        )

        reservation = limiter.acquire(80)
        limiter.record_actual(reservation, 40)
        limiter.acquire(50)

        self.assertEqual(sleeps, [])

    def test_webshop_defaults_use_conservative_rate_limit_and_action_budget(self):
        with patch("sys.argv", ["run_webshop.py"]):
            args = parse_args()

        self.assertEqual(args.llm_tpm_budget, 12000)
        self.assertEqual(args.agent_max_tokens, 96)

    def test_rate_limit_retries_use_long_backoff(self):
        error = RuntimeError("Error code: 429 - TPM limit reached")

        self.assertEqual(_retry_wait_seconds(error, attempt=0), 30)
        self.assertEqual(_retry_wait_seconds(error, attempt=1), 60)

    def test_webshop_rejects_invalid_throttle_arguments(self):
        with self.assertRaisesRegex(ValueError, "llm-tpm-budget"):
            validate_llm_throttle_args(-1, 96)
        with self.assertRaisesRegex(ValueError, "agent-max-tokens"):
            validate_llm_throttle_args(12000, 0)
        with self.assertRaisesRegex(ValueError, "prompt-history-window"):
            validate_llm_throttle_args(12000, 96, -1)
        with self.assertRaisesRegex(ValueError, "loop-early-stop-cycles"):
            validate_llm_throttle_args(12000, 96, 8, -1)


if __name__ == "__main__":
    unittest.main()

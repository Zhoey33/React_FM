"""Tests for paper-aligned WebShop sampling and metrics."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.memory import FailureMemoryStore
from experiments.run_webshop import (
    compute_summary,
    make_error_episode_result,
    load_memory_for_setting,
    load_or_sample_session_ids,
    memory_scope_for_setting,
    parse_num_products,
    parse_args,
    sample_session_ids,
    validate_memory_requirements,
    validate_session_ids,
)
from prompts.webshop_prompts import build_user_prompt


class WebShopAlignmentTests(unittest.TestCase):
    def test_webshop_summary_uses_exact_success_and_task_score(self):
        results = [
            {"reward": 1.0, "total_tokens": 10},
            {"reward": 0.5, "total_tokens": 20},
            {"reward": 0.0, "total_tokens": 30},
        ]

        summary = compute_summary(results, memory_stats={}, mode="react_fm", memory_setting="online")

        self.assertEqual(summary["total_success"], 1)
        self.assertEqual(summary["success_rate"], 0.3333)
        self.assertEqual(summary["avg_reward"], 0.5)
        self.assertEqual(summary["task_score"], 50.0)
        self.assertEqual(summary["memory_setting"], "online")
        self.assertEqual(summary["success_definition"], "reward == 1.0")

    def test_sample_session_ids_is_deterministic_subset_without_replacement(self):
        first = sample_session_ids(total=500, sample_size=100, seed=42)
        second = sample_session_ids(total=500, sample_size=100, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 100)
        self.assertEqual(len(set(first)), 100)
        self.assertTrue(all(0 <= idx < 500 for idx in first))

    def test_parse_num_products_supports_full_protocol_and_preview_sizes(self):
        self.assertIsNone(parse_num_products("full"))
        self.assertIsNone(parse_num_products("none"))
        self.assertEqual(parse_num_products("1000"), 1000)

    def test_official_eval_and_train_samples_use_split_offsets(self):
        eval_ids = load_or_sample_session_ids(None, split="eval", sample_size=10, seed=42)
        train_ids = load_or_sample_session_ids(None, split="train", sample_size=10, seed=42)

        self.assertTrue(all(500 <= idx < 1500 for idx in eval_ids))
        self.assertTrue(all(1500 <= idx < 12087 for idx in train_ids))

    def test_runner_defaults_to_official_text_rich_observations(self):
        with patch("sys.argv", ["run_webshop.py"]):
            args = parse_args()

        self.assertEqual(args.observation_mode, "text_rich")

    def test_webshop_memory_settings_share_shopping_memories(self):
        self.assertEqual(memory_scope_for_setting("online"), "task_type")
        self.assertEqual(memory_scope_for_setting("frozen"), "task_type")

    def test_loading_env_idx_memory_for_webshop_shared_setting_is_rejected(self):
        with TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "legacy_memory.json"
            memory_path.write_text(
                '{"next_id": 0, "scope": "env_idx", "buckets": {"0": []}}'
            )
            store = FailureMemoryStore(scope="task_type")

            with self.assertRaisesRegex(ValueError, "task_type"):
                load_memory_for_setting(store, str(memory_path), "frozen")

    def test_error_episode_result_counts_as_failed_episode(self):
        result = make_error_episode_result(
            env_idx=17,
            sample_position=4,
            eval_split="test",
            error=RuntimeError("bridge died"),
        )

        self.assertEqual(result["env_idx"], 17)
        self.assertEqual(result["sample_position"], 4)
        self.assertFalse(result["success"])
        self.assertEqual(result["reward"], 0.0)
        self.assertIn("bridge died", result["error"])

    def test_direct_wrapper_actions_are_normalized_to_action_strings(self):
        from experiments.webshop_bridge import normalize_direct_actions

        actions = normalize_direct_actions(
            {
                "has_search_bar": True,
                "clickables": ["search", "next >"],
            }
        )

        self.assertEqual(actions, ["search[product]"])

        actions = normalize_direct_actions(
            {
                "has_search_bar": False,
                "clickables": ["B000123", "Buy Now", "search"],
            }
        )
        self.assertEqual(actions, ["click[B000123]", "click[Buy Now]"])

    def test_frozen_memory_requires_existing_nonempty_offline_memory(self):
        with TemporaryDirectory() as tmp:
            missing_path = str(Path(tmp) / "missing.json")
            with self.assertRaisesRegex(ValueError, "resume-memory"):
                validate_memory_requirements("frozen", None)
            with self.assertRaisesRegex(FileNotFoundError, "missing"):
                validate_memory_requirements("frozen", missing_path)

            empty_path = Path(tmp) / "empty_memory.json"
            empty_path.write_text('{"next_id": 0, "scope": "task_type", "buckets": {"shopping": []}}')
            store = FailureMemoryStore(scope="task_type")
            with self.assertRaisesRegex(ValueError, "non-empty"):
                load_memory_for_setting(store, str(empty_path), "frozen")

    def test_sample_id_file_must_match_split_and_be_unique(self):
        validate_session_ids([0, 42, 499], "test")

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_session_ids([1, 1], "test")
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_session_ids([499, 500], "test")

    def test_valid_actions_are_rendered_after_latest_history(self):
        prompt = build_user_prompt(
            task_type="shopping",
            task_obs="Instruction: find item",
            history=[("search[item]", "Search results page")],
            valid_actions=["click[item - result]", "click[next >]"],
        )

        self.assertLess(prompt.index("Search results page"), prompt.index("Choose exactly one valid action"))
        self.assertLess(prompt.index("click[next >]"), prompt.rindex("> "))


if __name__ == "__main__":
    unittest.main()

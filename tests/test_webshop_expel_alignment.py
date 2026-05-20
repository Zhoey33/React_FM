"""Tests for ExpeL-paper-aligned WebShop runner behavior."""

import unittest
from unittest.mock import patch

from experiments.run_expel_webshop import (
    build_expel_prompt,
    compute_summary,
    load_or_sample_session_ids,
    normalize_expel_action,
    parse_args,
    parse_num_products,
    run_episode,
    sample_session_ids,
)


class FakeTracker:
    """Minimal token tracker used by fake LLM clients in runner tests."""

    def reset(self):
        pass

    def summary(self):
        return {"total_tokens": 0}


class SequenceLLM:
    """Fake LLM that returns fixed completions and records prompts."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.tracker = FakeTracker()

    def complete_text(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.responses.pop(0)


class OneStepWebShopEnv:
    """Fake WebShop env that finishes after one executable action."""

    def __init__(self, reward, actions=None):
        self.reward = reward
        self.actions = []
        self.valid_actions = actions or ["click[Buy Now]"]

    def reset(self, session_idx=0):
        return (
            "Instruction: find an item [SEP] Search",
            "shopping",
            {"available_actions": self.valid_actions},
        )

    def step(self, action):
        self.actions.append(action)
        return "Done", self.reward, True, {"available_actions": []}


class WebShopExpeLAlignmentTests(unittest.TestCase):
    """Paper-alignment checks for the WebShop ExpeL runner."""

    def test_expel_summary_uses_exact_success_and_task_score(self):
        results = [
            {"reward": 1.0, "total_tokens": 10},
            {"reward": 0.5, "total_tokens": 20},
            {"reward": 0.0, "total_tokens": 30},
        ]

        summary = compute_summary(results, insight_stats={}, mode="expel")

        self.assertEqual(summary["total_success"], 1)
        self.assertEqual(summary["success_rate"], 0.3333)
        self.assertEqual(summary["avg_reward"], 0.5)
        self.assertEqual(summary["task_score"], 50.0)
        self.assertEqual(summary["success_definition"], "reward == 1.0")

    def test_reward_below_one_is_not_successful(self):
        env = OneStepWebShopEnv(reward=0.5)
        llm = SequenceLLM(["click[Buy Now]"])

        result = run_episode(
            llm=llm,
            env=env,
            env_idx=0,
            insight_store=None,
            max_steps=1,
            max_inject=3,
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["success_definition"], "reward == 1.0")

    def test_runner_defaults_to_full_webshop_products(self):
        with patch("sys.argv", ["run_expel_webshop.py"]):
            args = parse_args()

        self.assertIsNone(parse_num_products(args.num_products))

    def test_sample_session_ids_is_deterministic_subset_without_replacement(self):
        first = sample_session_ids(total=500, sample_size=100, seed=42)
        second = sample_session_ids(total=500, sample_size=100, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 100)
        self.assertEqual(len(set(first)), 100)
        self.assertTrue(all(0 <= idx < 500 for idx in first))

    def test_official_eval_samples_use_split_offsets(self):
        eval_ids = load_or_sample_session_ids(None, split="eval", sample_size=10, seed=42)
        train_ids = load_or_sample_session_ids(None, split="train", sample_size=10, seed=42)

        self.assertTrue(all(500 <= idx < 1500 for idx in eval_ids))
        self.assertTrue(all(1500 <= idx < 12087 for idx in train_ids))

    def test_valid_actions_are_rendered_and_used_for_action_normalization(self):
        env = OneStepWebShopEnv(
            reward=1.0,
            actions=["click[item - target product]", "click[next >]"],
        )
        llm = SequenceLLM(["click[target product]"])

        result = run_episode(
            llm=llm,
            env=env,
            env_idx=0,
            insight_store=None,
            max_steps=1,
            max_inject=3,
        )

        self.assertTrue(result["success"])
        self.assertEqual(env.actions, ["click[item - target product]"])
        self.assertIn("Choose exactly one valid action", llm.prompts[0])

    def test_direct_wrapper_placeholder_search_does_not_override_query(self):
        action = normalize_expel_action(
            "search[blue cotton shirt]",
            valid_actions=["search[product]"],
        )

        self.assertEqual(action, "search[blue cotton shirt]")

    def test_placeholder_search_action_is_not_rendered_as_prompt_constraint(self):
        prompt = build_expel_prompt(
            task_obs="Instruction: find a blue shirt",
            history=[],
            valid_actions=["search[product]"],
        )

        self.assertNotIn("Choose exactly one valid action", prompt)
        self.assertNotIn("search[product]", prompt)


if __name__ == "__main__":
    unittest.main()

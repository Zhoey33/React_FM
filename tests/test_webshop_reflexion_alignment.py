"""Tests for Reflexion-paper-aligned WebShop behavior."""

import unittest
from unittest.mock import patch

from experiments.run_reflexion_webshop import (
    build_reflexion_prompt,
    compute_summary,
    parse_args,
    run_episode,
)
from prompts.webshop_prompts import normalize_reflexion_action
from src.reflexion_agent import generate_reflection


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

    def __init__(self, reward):
        self.reward = reward
        self.actions = []

    def reset(self, session_idx=0):
        return "Instruction: find an item [SEP] Search", "shopping", {}

    def step(self, action):
        self.actions.append(action)
        return "Done", self.reward, True, {}


class WebShopReflexionAlignmentTests(unittest.TestCase):
    def test_reflexion_runner_defaults_to_four_trials(self):
        with patch("sys.argv", ["run_reflexion_webshop.py"]):
            args = parse_args()

        self.assertEqual(args.num_trials, 4)

    def test_reflexion_summary_uses_exact_success_and_task_score(self):
        results = [
            {"reward": 1.0, "total_tokens": 10},
            {"reward": 0.5, "total_tokens": 20},
            {"reward": 0.0, "total_tokens": 30},
        ]

        summary = compute_summary(results, "reflexion")

        self.assertEqual(summary["total_success"], 1)
        self.assertEqual(summary["success_rate"], 0.3333)
        self.assertEqual(summary["avg_reward"], 0.5)
        self.assertEqual(summary["task_score"], 50.0)
        self.assertEqual(summary["success_definition"], "reward == 1.0")

    def test_reward_below_one_is_not_successful(self):
        env = OneStepWebShopEnv(reward=0.5)
        llm = SequenceLLM(["click[Buy Now]"])

        result = run_episode(llm, env, env_idx=0, memory=[], max_steps=1)

        self.assertFalse(result["success"])
        self.assertEqual(result["success_definition"], "reward == 1.0")

    def test_reflexion_prompt_injects_only_latest_three_memories(self):
        prompt = build_reflexion_prompt(
            task_obs="Instruction: find shampoo",
            history=[],
            memory=["oldest", "second", "third", "newest"],
        )

        self.assertNotIn("oldest", prompt)
        self.assertIn("second", prompt)
        self.assertIn("third", prompt)
        self.assertIn("newest", prompt)

    def test_think_action_is_preserved_and_not_sent_to_environment(self):
        env = OneStepWebShopEnv(reward=1.0)
        llm = SequenceLLM(["think[broaden the search]", "search[deodorant]"])

        result = run_episode(llm, env, env_idx=0, memory=[], max_steps=2)

        self.assertEqual(env.actions, ["search[deodorant]"])
        self.assertEqual(result["steps"][0]["action"], "think[broaden the search]")
        self.assertEqual(result["steps"][0]["observation"], "OK.")
        self.assertTrue(result["success"])

    def test_normalize_reflexion_action_accepts_think_bracket_format(self):
        self.assertEqual(
            normalize_reflexion_action("I should try a broader query. think[search broader]"),
            "think[search broader]",
        )

    def test_webshop_reflection_uses_webshop_examples_and_latest_memory(self):
        llm = SequenceLLM(["Plan: search with broader product keywords next time."])
        log = (
            "Here is the task.\n"
            "Instruction: Find a dairy free snack under 30 dollars\n"
            "> search[chips]\n"
            "No results.\n"
        )

        reflection = generate_reflection(
            llm,
            log,
            ["oldest plan", "second plan", "third plan", "newest plan"],
            domain="webshop",
        )

        self.assertIn("broader product keywords", reflection)
        prompt = llm.prompts[0]
        self.assertIn("There are two examples below.", prompt)
        self.assertIn("search[", prompt)
        self.assertNotIn("oldest plan", prompt)
        self.assertIn("second plan", prompt)
        self.assertIn("third plan", prompt)
        self.assertIn("newest plan", prompt)


if __name__ == "__main__":
    unittest.main()

"""Tests for WebShop in-loop memory retrieval, quality gating, and TTL."""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_webshop import WebShopReActAgent
from src.memory import FailureMemoryEntry, RetrievalResult
from src.webshop_failure_detector import DetectionResult


class FakeTokenTracker:
    def reset(self):
        pass

    def summary(self):
        return {"total_tokens": 0}


class FakeLLM:
    def __init__(self, actions):
        self.actions = list(actions)
        self.tracker = FakeTokenTracker()
        self.prompts = []

    def complete_text(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        return self.actions.pop(0)


class FakeMemory:
    def __init__(self, retrieval_result=None):
        self.calls = []
        self.retrieval_result = retrieval_result or RetrievalResult([], [])

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        return self.retrieval_result


class FakeUnproductiveDetector:
    judge_llm = None

    def detect(self, observation, action, action_history):
        return DetectionResult(is_failure=True, failure_type="unproductive")

    def is_task_complete(self, observation, done, info):
        return done, done


class FakeLoopEnv:
    def reset(self, session_idx=0):
        return "Instruction:\nfind a product", "shopping", {"available_actions": []}

    def step(self, action):
        return "Instruction:\nfind a product", 0.0, False, {"available_actions": []}


class WebShopMemoryGatingTests(unittest.TestCase):
    def setUp(self):
        self.agent = WebShopReActAgent(llm=object(), enable_memory=False)

    def test_unproductive_judge_result_can_trigger_memory_retrieval(self):
        det = DetectionResult(is_failure=True, failure_type="unproductive")

        self.assertTrue(
            self.agent._should_retrieve_memory(det, "search[elastic waist dresses]", set())
        )

    def test_retrieval_is_deduplicated_per_episode_signature(self):
        retrieved_signatures = set()
        det = DetectionResult(is_failure=True, failure_type="unproductive")
        action = "click[back to search]"

        self.assertTrue(self.agent._should_retrieve_memory(det, action, retrieved_signatures))
        retrieved_signatures.add(self.agent._memory_failure_signature(det, action))
        self.assertFalse(self.agent._should_retrieve_memory(det, action, retrieved_signatures))

    def test_explicit_failures_trigger_once_per_episode_signature(self):
        retrieved_signatures = set()
        det = DetectionResult(is_failure=True, failure_type="no_results")
        action = "search[gingko hand painted pillow cover]"

        self.assertTrue(
            self.agent._should_retrieve_memory(det, action, retrieved_signatures)
        )
        retrieved_signatures.add(self.agent._memory_failure_signature(det, action))
        self.assertFalse(
            self.agent._should_retrieve_memory(det, action, retrieved_signatures)
        )

    def test_invalid_action_can_trigger_memory_retrieval(self):
        det = DetectionResult(is_failure=True, failure_type="invalid_action")

        self.assertTrue(
            self.agent._should_retrieve_memory(det, "click[not a valid option]", set())
        )

    def test_relevance_ignores_search_result_titles_in_observation(self):
        entry = FailureMemoryEntry(
            memory_id=1,
            failure_action="search[long lasting spf foundation]",
            failure_observation="Instruction: find long lasting foundation",
            solution_action="click[item - magic cushion cover lasting]",
            task_type="shopping",
        )
        observation = (
            "Instruction:\n"
            "i need a gingko light and 20x20 pillow cover that is hand painted\n"
            "Page 1 [SEP] magic cushion cover lasting high coverage"
        )

        overlap = self.agent._memory_token_overlap(
            "search[gingko light hand painted pillow cover]", observation, entry
        )

        self.assertLess(overlap, self.agent.memory_min_token_overlap)

    def test_low_quality_retrieval_uses_fallback_hint_for_multiple_steps(self):
        memory = FakeMemory()
        llm = FakeLLM(
            [
                "click[back to search]",
                "search[elastic waist dresses]",
                "click[back to search]",
                "search[different elastic waist dress]",
            ]
        )
        agent = WebShopReActAgent(
            llm=llm,
            memory_store=memory,
            failure_detector=FakeUnproductiveDetector(),
            extractor_llm=None,
            max_steps=4,
            memory_injection_steps=3,
        )

        result = agent.run_episode(FakeLoopEnv(), env_idx=0)

        self.assertEqual(result["failures_detected"], 4)
        self.assertEqual(result["memories_retrieved"], 0)
        self.assertEqual(len(memory.calls), 1)
        self.assertNotIn("Try a different search/item strategy", llm.prompts[0])
        self.assertIn("Try a different search/item strategy", llm.prompts[1])
        self.assertIn("Try a different search/item strategy", llm.prompts[2])
        self.assertIn("Try a different search/item strategy", llm.prompts[3])

    def test_high_quality_retrieval_injects_memory_for_multiple_steps(self):
        entry = FailureMemoryEntry(
            memory_id=1,
            failure_action="search[elastic waist dress]",
            failure_observation="Instruction: find a medium elastic waist dress",
            solution_action="search[medium elastic waist dress] -> click[a relevant dress]",
            task_type="shopping",
        )
        memory = FakeMemory(RetrievalResult([entry], [0.0328]))
        llm = FakeLLM(
            [
                "search[elastic waist dresses]",
                "click[item - medium elastic waist dress]",
                "click[medium]",
                "click[buy now]",
            ]
        )
        agent = WebShopReActAgent(
            llm=llm,
            memory_store=memory,
            failure_detector=FakeUnproductiveDetector(),
            extractor_llm=None,
            max_steps=4,
            memory_injection_steps=3,
        )

        result = agent.run_episode(FakeLoopEnv(), env_idx=0)

        self.assertEqual(result["memories_retrieved"], 1)
        self.assertEqual(len(memory.calls), 1)
        self.assertIn("medium elastic waist dress", llm.prompts[1])
        self.assertIn("medium elastic waist dress", llm.prompts[2])
        self.assertIn("medium elastic waist dress", llm.prompts[3])


if __name__ == "__main__":
    unittest.main()

"""Tests for wiring ScienceWorld implicit judge detection into the runner."""

import logging

from experiments import run_scienceworld
from src.memory import RetrievalResult
from src.scienceworld_failure_detector import DetectionResult


class FakeMemoryEntry:
    def __init__(
        self,
        memory_id=12,
        failure_type="syntax_or_parse",
        failure_action="open door to kitchen",
        failure_observation="No known action matches that input.",
        repair_action="open door to kitchen",
        confidence_score=0.9,
    ):
        self.memory_id = memory_id
        self.failure_type = failure_type
        self.failure_action = failure_action
        self.failure_observation = failure_observation
        self.solution_action = repair_action
        self.repair_strategy = "Use a valid action syntax."
        self.repair_tactic = "Avoid repeating commands rejected by the environment."
        self.repair_action = repair_action
        self.confidence_score = confidence_score

    def get_repair_display(self):
        return self.repair_action


class _Tracker:
    def reset(self):
        pass

    def summary(self):
        return {"total_tokens": 0}


class FakeLLM:
    def __init__(self):
        self.tracker = _Tracker()
        self.calls = 0

    def complete_text(self, prompt, stop=None, label="", system=None):
        self.calls += 1
        return "look around"


class FakeExtractorLLM:
    def __init__(self):
        self.tracker = _Tracker()


class FakeDetector:
    judge_llm = None

    def __init__(self):
        self.calls = []

    def detect(self, observation, action, action_history, **kwargs):
        self.calls.append(
            {
                "observation": observation,
                "action": action,
                "action_history": list(action_history),
                **kwargs,
            }
        )
        result = DetectionResult(
            is_failure=True,
            failure_type="implicit_no_progress",
            reason="No new information and score did not change.",
            confidence=0.9,
            detector_source="judge",
        )
        result.productive_signal = "none"
        result.evidence_for_failure = ["No new information."]
        result.evidence_against_failure = []
        result.judge_repair_strategy = "Inspect inventory before repeating exploration."
        result.judge_repair_action = "inventory"
        result.judge_repair_confidence = 0.88
        result.judge_repair_rationale = "Inventory may reveal tools that should guide the next step."
        return result


class FakeMemory:
    def __init__(self, retrieval_result=None):
        self.retrieve_calls = []
        self.add_calls = []
        self.min_score = 0.25
        self.retrieval_mode = "hybrid"
        self.retrieval_result = retrieval_result or RetrievalResult([], [], candidate_count=4)

    def retrieve(self, **kwargs):
        self.retrieve_calls.append(kwargs)
        return self.retrieval_result

    def add(self, **kwargs):
        self.add_calls.append(kwargs)


class FakeEnv:
    def __init__(self):
        self.step_calls = 0

    def reset(self):
        return (
            "Your task is to melt tin.",
            "melt",
            {
                "variation_idx": 3,
                "task_desc": "Your task is to melt tin.",
                "look": "This room is called the kitchen.",
                "inventory": "In your inventory, you see: nothing.",
                "valid_actions": "look around\ninventory",
            },
        )

    def step(self, action):
        self.step_calls += 1
        return (
            "You see the same room.",
            0.0,
            False,
            {
                "score": 0.0,
                "look": "This room is called the kitchen.",
                "inventory": "In your inventory, you see: nothing.",
                "valid_actions": "look around\ninventory\ngo to hallway",
            },
        )


def test_resolve_enable_implicit_failures_prefers_cli_then_config():
    class Args:
        enable_implicit_failures = None

    args = Args()
    assert run_scienceworld._resolve_enable_implicit_failures(
        args,
        {"judge": {"enable_implicit_failures": True}},
    ) is True

    assert run_scienceworld._resolve_enable_implicit_failures(
        args,
        {"judge": {"enable_implicit_failures": False}},
    ) is False

    args.enable_implicit_failures = True
    assert run_scienceworld._resolve_enable_implicit_failures(
        args,
        {"judge": {"enable_implicit_failures": False}},
    ) is True

    args.enable_implicit_failures = False
    assert run_scienceworld._resolve_enable_implicit_failures(
        args,
        {"judge": {"enable_implicit_failures": True}},
    ) is False

    args.enable_implicit_failures = None
    assert run_scienceworld._resolve_enable_implicit_failures(args, {}) is True


def test_judge_detected_failure_triggers_retrieval_and_enters_extraction(monkeypatch):
    captured = {}

    def fake_extract_failure_recoveries(extractor_llm, trajectory, detected_failures=None):
        captured["detected_failures"] = detected_failures
        return [
            {
                "failure_step": 0,
                "failure_action": "look around",
                "failure_observation": "You see the same room.",
                "failure_type": "implicit_no_progress",
                "detector_source": "judge",
                "score_before_action": 0.0,
                "score_after_action": 0.0,
                "score_delta": 0.0,
                "solution_action": "inventory",
                "repair_strategy": "Check inventory before repeating exploration.",
                "repair_tactic": "Inspect carried items, then choose a task-relevant action.",
                "repair_action": "inventory",
                "confidence_score": 0.9,
            }
        ]

    monkeypatch.setattr(
        "src.scienceworld_memory_extractor.extract_failure_recoveries",
        fake_extract_failure_recoveries,
    )

    detector = FakeDetector()
    memory = FakeMemory()
    agent = run_scienceworld.ScienceWorldReActAgent(
        llm=FakeLLM(),
        memory_store=memory,
        failure_detector=detector,
        extractor_llm=FakeExtractorLLM(),
        max_steps=1,
        enable_memory=True,
        inject_mode="in_loop",
    )

    result = agent.run_episode(FakeEnv(), env_idx=7)

    assert detector.calls[0]["task_type"] == "melt"
    assert detector.calls[0]["task_goal"] == "Your task is to melt tin."
    assert detector.calls[0]["recent_history"] == []
    assert detector.calls[0]["step"] == 0
    assert detector.calls[0]["variation_idx"] == 3
    assert detector.calls[0]["score_before_action"] == 0.0
    assert detector.calls[0]["score_after_action"] == 0.0
    assert detector.calls[0]["look_before_action"] == "This room is called the kitchen."
    assert detector.calls[0]["inventory_before_action"] == "In your inventory, you see: nothing."
    assert detector.calls[0]["valid_actions_before_action"] == "look around\ninventory"
    assert detector.calls[0]["look_after_action"] == "This room is called the kitchen."
    assert detector.calls[0]["inventory_after_action"] == "In your inventory, you see: nothing."
    assert detector.calls[0]["valid_actions_after_action"] == "look around\ninventory\ngo to hallway"

    step = result["steps"][0]
    assert step["failure_detected"] is True
    assert step["failure_type"] == "implicit_no_progress"
    assert step["detector_source"] == "judge"
    assert step["failure_reason"] == "No new information and score did not change."
    assert step["failure_confidence"] == 0.9
    assert step["productive_signal"] == "none"
    assert step["evidence_for_failure"] == ["No new information."]
    assert step["judge_repair_strategy"] == "Inspect inventory before repeating exploration."
    assert step["judge_repair_action"] == "inventory"
    assert step["judge_repair_confidence"] == 0.88
    assert step["judge_repair_rationale"] == "Inventory may reveal tools that should guide the next step."
    assert step["judge_advice_injected"] is False

    assert memory.retrieve_calls
    assert memory.retrieve_calls[0]["return_scores"] is True
    assert memory.retrieve_calls[0]["candidate_k"] == 5
    assert memory.retrieve_calls[0]["query_failure_type"] == "implicit_no_progress"
    assert captured["detected_failures"] == [
        {
            "step": 0,
            "action": "look around",
            "observation": "You see the same room.",
            "failure_type": "implicit_no_progress",
            "detector_source": "judge",
            "score_before_action": 0.0,
            "score_after_action": 0.0,
            "score_delta": 0.0,
        }
    ]
    assert memory.add_calls[0]["failure_action"] == "look around"
    assert memory.add_calls[0]["failure_step"] == 0
    assert memory.add_calls[0]["failure_type"] == "implicit_no_progress"
    assert memory.add_calls[0]["detector_source"] == "judge"
    assert memory.add_calls[0]["source_episode_success"] is False
    assert memory.add_calls[0]["source_episode_score"] == 0.0
    assert memory.add_calls[0]["confidence_score"] == 0.9
    assert "question_text" not in memory.add_calls[0]


def test_runner_logs_task_goal_and_retrieved_memory_content(caplog):
    caplog.set_level(logging.INFO)
    detector = FakeDetector()
    memory = FakeMemory(
        RetrievalResult(
            [
                FakeMemoryEntry(
                    memory_id=21,
                    failure_type="implicit_no_progress",
                    failure_action="look around",
                    failure_observation="You see the same room.",
                    repair_action="inventory",
                    confidence_score=0.88,
                )
            ],
            [0.04],
            candidate_count=1,
        )
    )
    agent = run_scienceworld.ScienceWorldReActAgent(
        llm=FakeLLM(),
        memory_store=memory,
        failure_detector=detector,
        extractor_llm=None,
        max_steps=1,
        enable_memory=True,
        inject_mode="in_loop",
    )

    agent.run_episode(FakeEnv(), env_idx=7)

    log_text = caplog.text
    assert "Task goal:" in log_text
    assert "Your task is to melt tin." in log_text
    assert "detector=judge" in log_text
    assert "productive_signal=none" in log_text
    assert "Retrieval candidates ids=[21]" in log_text
    assert "Selected memory #21" in log_text
    assert "repair_action=inventory" in log_text
    assert "Injecting judge advice fallback" not in log_text


def test_retrieval_observability_fields_are_recorded_for_failure_step():
    detector = FakeDetector()
    memory = FakeMemory()
    agent = run_scienceworld.ScienceWorldReActAgent(
        llm=FakeLLM(),
        memory_store=memory,
        failure_detector=detector,
        extractor_llm=None,
        max_steps=1,
        max_memory_inject=1,
        enable_memory=True,
        inject_mode="in_loop",
    )

    result = agent.run_episode(FakeEnv(), env_idx=7)

    step = result["steps"][0]
    assert step["retrieval_attempted"] is True
    assert step["retrieval_mode"] == "hybrid"
    assert step["retrieval_top_k"] == 1
    assert step["retrieval_min_score"] == 0.25
    assert step["retrieval_candidate_count"] == 4
    assert step["retrieved_memory_scores"] == []
    assert step["retrieval_candidate_memory_ids"] == []
    assert step["retrieval_candidate_scores"] == []
    assert step["retrieval_candidate_relevance_scores"] == []
    assert step["retrieval_selected_memory_id"] is None
    assert step["retrieval_relevance_decision"] is False
    assert step["retrieval_rejection_reason"] == "no_candidates"
    assert step["judge_advice_injected"] is False


def test_gated_retrieval_records_candidates_without_injecting_rejected_memory():
    detector = FakeDetector()
    memory = FakeMemory(
        RetrievalResult(
            [FakeMemoryEntry()],
            [0.0327],
            candidate_count=1,
        )
    )
    agent = run_scienceworld.ScienceWorldReActAgent(
        llm=FakeLLM(),
        memory_store=memory,
        failure_detector=detector,
        extractor_llm=None,
        max_steps=1,
        max_memory_inject=1,
        enable_memory=True,
        inject_mode="in_loop",
        retrieval_candidate_k=5,
        relevance_score_threshold=0.45,
    )

    result = agent.run_episode(FakeEnv(), env_idx=7)

    step = result["steps"][0]
    assert step["retrieval_attempted"] is True
    assert step["memory_retrieved"] == 0
    assert step["retrieved_memory_ids"] == []
    assert step["retrieved_memory_scores"] == []
    assert step["retrieval_candidate_memory_ids"] == [12]
    assert step["retrieval_candidate_scores"] == [0.0327]
    assert step["retrieval_candidate_relevance_scores"] == [0.0]
    assert step["retrieval_selected_memory_id"] is None
    assert step["retrieval_relevance_decision"] is False
    assert step["retrieval_rejection_reason"] == "no_type_compatible_candidates"
    assert step["judge_advice_injected"] is False


def test_retrieved_memory_takes_priority_over_judge_advice():
    detector = FakeDetector()
    memory = FakeMemory(
        RetrievalResult(
            [
                FakeMemoryEntry(
                    memory_id=31,
                    failure_type="implicit_no_progress",
                    failure_action="look around",
                    failure_observation="You see the same room.",
                    repair_action="inventory",
                    confidence_score=0.9,
                )
            ],
            [0.04],
            candidate_count=1,
        )
    )
    agent = run_scienceworld.ScienceWorldReActAgent(
        llm=FakeLLM(),
        memory_store=memory,
        failure_detector=detector,
        extractor_llm=None,
        max_steps=1,
        max_memory_inject=1,
        enable_memory=True,
        inject_mode="in_loop",
    )

    result = agent.run_episode(FakeEnv(), env_idx=7)

    step = result["steps"][0]
    assert step["memory_retrieved"] == 1
    assert step["retrieved_memory_ids"] == [31]
    assert step["judge_advice_injected"] is False

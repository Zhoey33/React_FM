"""Tests for wiring ScienceWorld implicit judge detection into the runner."""

from experiments import run_scienceworld
from src.scienceworld_failure_detector import DetectionResult


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
        return DetectionResult(
            is_failure=True,
            failure_type="implicit_no_progress",
            reason="No new information and score did not change.",
            confidence=0.9,
            detector_source="judge",
        )


class FakeMemory:
    def __init__(self):
        self.retrieve_calls = []
        self.add_calls = []

    def retrieve(self, **kwargs):
        self.retrieve_calls.append(kwargs)
        return []

    def add(self, **kwargs):
        self.add_calls.append(kwargs)


class FakeEnv:
    def __init__(self):
        self.step_calls = 0

    def reset(self):
        return (
            "Your task is to melt tin.",
            "melt",
            {"variation_idx": 3},
        )

    def step(self, action):
        self.step_calls += 1
        return (
            "You see the same room.",
            0.0,
            False,
            {"score": 0.0},
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
    assert detector.calls[0]["score_before_action"] == 0.0
    assert detector.calls[0]["score_after_action"] == 0.0

    step = result["steps"][0]
    assert step["failure_detected"] is True
    assert step["failure_type"] == "implicit_no_progress"
    assert step["detector_source"] == "judge"
    assert step["failure_reason"] == "No new information and score did not change."
    assert step["failure_confidence"] == 0.9

    assert memory.retrieve_calls
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

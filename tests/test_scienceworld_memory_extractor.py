"""Tests for ScienceWorld memory extraction prompt, validation, and provenance."""

from src.scienceworld_memory_extractor import build_extractor_prompt, extract_failure_recoveries


class FakeExtractorLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete_text(self, prompt, label="", system=None, max_tokens=None):
        self.calls.append(
            {
                "prompt": prompt,
                "label": label,
                "system": system,
                "max_tokens": max_tokens,
            }
        )
        return self.response


def _trajectory():
    return [
        {
            "step": 0,
            "action": "look around",
            "observation": "You are in the hallway.",
        },
        {
            "step": 1,
            "action": "go to kitchen",
            "observation": "The door is not open.",
        },
        {
            "step": 2,
            "action": "open door to kitchen",
            "observation": "The door is now open.",
        },
        {
            "step": 3,
            "action": "go to kitchen",
            "observation": "You move to the kitchen.",
        },
    ]


def test_prompt_requires_confidence_score_and_excludes_question_text():
    prompt = build_extractor_prompt(
        _trajectory(),
        detected_failures=[
            {
                "step": 1,
                "action": "go to kitchen",
                "observation": "The door is not open.",
                "failure_type": "precondition_blocked",
                "detector_source": "rule",
                "score_before_action": 0.0,
                "score_after_action": 0.0,
            }
        ],
    )

    assert "confidence_score" in prompt
    assert "question_text" not in prompt
    assert 'source: "rule"' in prompt
    assert 'score_delta: "0.0"' in prompt


def test_extractor_overwrites_failure_fields_and_clamps_confidence():
    llm = FakeExtractorLLM(
        """
        [
          {
            "failure_step": "1",
            "failure_action": "LLM rewrote action",
            "failure_observation": "LLM rewrote observation",
            "failure_type": "wrong_type",
            "solution_action": "open door to kitchen -> go to kitchen",
            "repair_strategy": "Open the missing precondition before moving.",
            "repair_tactic": "Open the door, then retry the movement command.",
            "repair_action": "open door to kitchen -> go to kitchen",
            "confidence_score": 1.7,
            "question_text": "old field should be ignored"
          }
        ]
        """
    )

    recoveries = extract_failure_recoveries(
        llm,
        _trajectory(),
        detected_failures=[
            {
                "step": 1,
                "action": "go to kitchen",
                "observation": "The door is not open.",
                "failure_type": "precondition_blocked",
                "detector_source": "rule",
                "score_before_action": 0.0,
                "score_after_action": 0.0,
            }
        ],
    )

    assert len(recoveries) == 1
    recovery = recoveries[0]
    assert recovery["failure_step"] == 1
    assert recovery["failure_action"] == "go to kitchen"
    assert recovery["failure_observation"] == "The door is not open."
    assert recovery["failure_type"] == "precondition_blocked"
    assert recovery["detector_source"] == "rule"
    assert recovery["score_before_action"] == 0.0
    assert recovery["score_after_action"] == 0.0
    assert recovery["score_delta"] == 0.0
    assert recovery["confidence_score"] == 1.0
    assert "question_text" not in recovery


def test_extractor_drops_repair_actions_not_seen_after_failure():
    llm = FakeExtractorLLM(
        """
        [
          {
            "failure_step": "1",
            "failure_action": "go to kitchen",
            "failure_observation": "The door is not open.",
            "failure_type": "precondition_blocked",
            "solution_action": "teleport to kitchen",
            "repair_strategy": "Skip navigation.",
            "repair_tactic": "Teleport.",
            "repair_action": "teleport to kitchen",
            "confidence_score": 0.8
          }
        ]
        """
    )

    recoveries = extract_failure_recoveries(
        llm,
        _trajectory(),
        detected_failures=[
            {
                "step": 1,
                "action": "go to kitchen",
                "observation": "The door is not open.",
                "failure_type": "precondition_blocked",
            }
        ],
    )

    assert recoveries == []


def test_judge_detected_implicit_failure_can_be_extracted():
    trajectory = [
        {
            "step": 0,
            "action": "look around",
            "observation": "You see the same room.",
        },
        {
            "step": 1,
            "action": "inventory",
            "observation": "You are carrying a metal pot.",
        },
    ]
    llm = FakeExtractorLLM(
        """
        [
          {
            "failure_step": "0",
            "failure_action": "look around",
            "failure_observation": "You see the same room.",
            "failure_type": "implicit_no_progress",
            "solution_action": "inventory",
            "repair_strategy": "Inspect carried items instead of repeating exploration.",
            "repair_tactic": "Check inventory to decide a task-relevant next action.",
            "repair_action": "inventory",
            "confidence_score": 0.64
          }
        ]
        """
    )

    recoveries = extract_failure_recoveries(
        llm,
        trajectory,
        detected_failures=[
            {
                "step": 0,
                "action": "look around",
                "observation": "You see the same room.",
                "failure_type": "implicit_no_progress",
                "detector_source": "judge",
            }
        ],
    )

    assert len(recoveries) == 1
    assert recoveries[0]["failure_type"] == "implicit_no_progress"
    assert recoveries[0]["detector_source"] == "judge"

"""Tests for ScienceWorld rule and LLM-judge failure detection."""

from src.scienceworld_failure_detector import ScienceWorldFailureDetector


class FakeJudgeLLM:
    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc
        self.prompts = []

    def complete_text(self, prompt, label="", max_tokens=None):
        self.prompts.append({"prompt": prompt, "label": label, "max_tokens": max_tokens})
        if self.exc is not None:
            raise self.exc
        return self.response


def test_rule_failure_takes_priority_and_does_not_call_judge():
    judge = FakeJudgeLLM('{"is_failure": false}')
    detector = ScienceWorldFailureDetector(
        judge_llm=judge,
        enable_implicit_failures=True,
    )

    result = detector.detect(
        observation="No known action matches that input.",
        action="pick up water",
        action_history=["pick up water"],
    )

    assert result.is_failure is True
    assert result.failure_type == "syntax_or_parse"
    assert result.detector_source == "rule"
    assert judge.prompts == []


def test_judge_is_not_called_when_implicit_failures_are_disabled():
    judge = FakeJudgeLLM('{"is_failure": true, "failure_type": "implicit_no_progress"}')
    detector = ScienceWorldFailureDetector(
        judge_llm=judge,
        enable_implicit_failures=False,
    )

    result = detector.detect(
        observation="You see the same room.",
        action="look around",
        action_history=["look around"],
        task_type="melt",
        task_goal="Your task is to melt tin.",
        recent_history=[("inventory", "You have nothing.")],
        score_before_action=10.0,
        score_after_action=10.0,
    )

    assert result.is_failure is False
    assert judge.prompts == []


def test_enabled_judge_prompt_contains_context_and_parses_failure_json():
    judge = FakeJudgeLLM(
        '{"is_failure": true, "failure_type": "implicit_no_progress", '
        '"confidence": 0.82, "reason": "The action repeated prior exploration."}'
    )
    detector = ScienceWorldFailureDetector(
        judge_llm=judge,
        enable_implicit_failures=True,
    )

    result = detector.detect(
        observation="You see the same room.",
        action="look around",
        action_history=["inventory", "look around"],
        task_type="melt",
        task_goal="Your task is to melt tin.",
        recent_history=[
            ("inventory", "You have a metal pot."),
            ("look around", "You see a kitchen."),
        ],
        score_before_action=10.0,
        score_after_action=10.0,
    )

    assert result.is_failure is True
    assert result.failure_type == "implicit_no_progress"
    assert result.detector_source == "judge"
    assert result.confidence == 0.82
    assert result.reason == "The action repeated prior exploration."

    call = judge.prompts[0]
    prompt = call["prompt"]
    assert call["label"] == "judge"
    assert call["max_tokens"] == 160
    assert "Task type: melt" in prompt
    assert "Task goal / initial observation:" in prompt
    assert "Your task is to melt tin." in prompt
    assert "[history 1] Action: inventory" in prompt
    assert "Current action: look around" in prompt
    assert "Current observation: You see the same room." in prompt
    assert "Score before action: 10.0" in prompt
    assert "Score after action: 10.0" in prompt
    assert "Score delta: 0.0" in prompt


def test_judge_non_failure_json_returns_non_failure():
    judge = FakeJudgeLLM(
        '{"is_failure": false, "failure_type": "productive", '
        '"confidence": 0.7, "reason": "The observation gives new information."}'
    )
    detector = ScienceWorldFailureDetector(
        judge_llm=judge,
        enable_implicit_failures=True,
    )

    result = detector.detect(
        observation="You see a thermometer.",
        action="look around",
        action_history=["look around"],
        task_goal="Your task is to measure temperature.",
        score_before_action=0,
        score_after_action=0,
    )

    assert result.is_failure is False
    assert result.failure_type == ""
    assert result.detector_source == ""


def test_judge_malformed_json_or_exception_safely_returns_non_failure():
    malformed = ScienceWorldFailureDetector(
        judge_llm=FakeJudgeLLM("not json"),
        enable_implicit_failures=True,
    )
    failed_call = ScienceWorldFailureDetector(
        judge_llm=FakeJudgeLLM(exc=RuntimeError("network down")),
        enable_implicit_failures=True,
    )

    for detector in (malformed, failed_call):
        result = detector.detect(
            observation="The room looks unchanged.",
            action="look around",
            action_history=["look around"],
            score_before_action=5,
            score_after_action=5,
        )
        assert result.is_failure is False
        assert result.failure_type == ""

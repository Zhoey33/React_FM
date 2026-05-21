"""Tests for ScienceWorld prompt construction and in-loop memory injection."""

from experiments import run_scienceworld
from prompts.scienceworld_prompts import build_baseline_user_prompt, build_user_prompt
from src.memory import FailureMemoryEntry, RetrievalResult
from src.scienceworld_failure_detector import DetectionResult


def _memory_entry() -> FailureMemoryEntry:
    return FailureMemoryEntry(
        memory_id=1,
        failure_action="go to kitchen",
        failure_observation="The door is not open.",
        solution_action="open door to kitchen -> go to kitchen",
        task_type="melt",
        repair_strategy="Satisfy the missing precondition before retrying movement.",
        repair_tactic="Open the blocked door, then retry going to the destination.",
        repair_action="open door to kitchen -> go to kitchen",
        question_text="legacy question should not be shown",
        confidence_score=0.99,
    )


def _failure_context() -> dict:
    return {
        "action": "go to kitchen",
        "observation": "The door is not open.",
        "failure_type": "precondition_blocked",
        "detector_source": "rule",
        "failure_reason": "Movement was blocked by a closed door.",
    }


def test_baseline_prompt_has_no_memory_or_failure_block():
    prompt = build_baseline_user_prompt(
        task_type="melt",
        task_obs="Your task is to melt tin.",
        history=[("look around", "You are in the hallway.")],
    )

    assert "Here is the task." in prompt
    assert "Retrieved repair memory" not in prompt
    assert "Previous action appears to have failed" not in prompt
    assert "question_text" not in prompt
    assert prompt.endswith("> ")


def test_in_loop_prompt_places_failure_and_memory_after_history():
    prompt = build_user_prompt(
        task_type="melt",
        task_obs="Your task is to melt tin.",
        history=[
            ("look around", "You are in the hallway."),
            ("go to kitchen", "The door is not open."),
        ],
        retrieved_memories=[_memory_entry()],
        failure_context=_failure_context(),
    )

    task_pos = prompt.index("Here is the task.")
    history_pos = prompt.index("> go to kitchen\nThe door is not open.")
    failure_pos = prompt.index("Previous action appears to have failed.")
    memory_pos = prompt.index("Retrieved repair memory:")
    action_cue_pos = prompt.rindex("> ")

    assert task_pos < history_pos < failure_pos < memory_pos < action_cue_pos
    assert "Failed action: go to kitchen" in prompt
    assert "Failure observation: The door is not open." in prompt
    assert "Failure type: precondition_blocked" in prompt
    assert "Detector source: rule" in prompt
    assert "Past failed action: go to kitchen" in prompt
    assert "Past failure observation: The door is not open." in prompt
    assert "Repair strategy: Satisfy the missing precondition" in prompt
    assert "Repair plan: Open the blocked door" in prompt
    assert "Suggested next action: open door to kitchen -> go to kitchen" in prompt
    assert "Use the memory only if it applies to the current state." in prompt
    assert "question_text" not in prompt
    assert "0.99" not in prompt


def test_scienceworld_agent_prompt_history_window_keeps_recent_steps_only():
    agent = run_scienceworld.ScienceWorldReActAgent(
        llm=None,
        prompt_history_window=2,
        enable_memory=False,
        baseline_mode=True,
    )
    history = [
        ("look around", "first room"),
        ("inventory", "empty"),
        ("go to kitchen", "blocked"),
    ]

    assert agent._prompt_history(history) == [
        ("inventory", "empty"),
        ("go to kitchen", "blocked"),
    ]


class _Tracker:
    def reset(self):
        pass

    def summary(self):
        return {"total_tokens": 0}


class _PromptRecordingLLM:
    def __init__(self):
        self.tracker = _Tracker()
        self.prompts = []
        self.responses = ["look around", "inventory", "look around"]

    def complete_text(self, prompt, stop=None, label="", system=None):
        self.prompts.append(prompt)
        return self.responses[len(self.prompts) - 1]


class _DetectorOnce:
    judge_llm = None

    def __init__(self):
        self.calls = 0

    def detect(self, observation, action, action_history, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return DetectionResult(
                is_failure=True,
                failure_type="implicit_no_progress",
                reason="No new information.",
                confidence=0.8,
                detector_source="judge",
            )
        return DetectionResult(is_failure=False)


class _MemoryHit:
    min_score = 0.0
    retrieval_mode = "hybrid"

    def __init__(self):
        self.entry = _memory_entry()

    def retrieve(self, **kwargs):
        return RetrievalResult([self.entry], [0.03], candidate_count=1)


class _NoopEnv:
    def reset(self):
        return "Your task is to melt tin.", "melt", {"variation_idx": 1}

    def step(self, action):
        return "The room is unchanged.", 0.0, False, {"score": 0.0}


def test_runner_injects_failure_memory_prompt_once_after_retrieval():
    llm = _PromptRecordingLLM()
    agent = run_scienceworld.ScienceWorldReActAgent(
        llm=llm,
        memory_store=_MemoryHit(),
        failure_detector=_DetectorOnce(),
        extractor_llm=None,
        max_steps=3,
        prompt_history_window=2,
        enable_memory=True,
        inject_mode="in_loop",
    )

    agent.run_episode(_NoopEnv(), env_idx=1)

    assert "Previous action appears to have failed." not in llm.prompts[0]
    assert "Previous action appears to have failed." in llm.prompts[1]
    assert "Failed action: look around" in llm.prompts[1]
    assert "Retrieved repair memory:" in llm.prompts[1]
    assert "Previous action appears to have failed." not in llm.prompts[2]

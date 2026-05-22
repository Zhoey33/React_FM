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


def test_unknown_action_message_is_rule_failure_and_does_not_call_judge():
    judge = FakeJudgeLLM('{"is_failure": false}')
    detector = ScienceWorldFailureDetector(
        judge_llm=judge,
        enable_implicit_failures=True,
    )

    result = detector.detect(
        observation="Unknown action. Type 'help' for a list of actions, and 'objects' for a list of objects.",
        action="look at drawer (in cupboard, in kitchen)",
        action_history=["look at drawer (in cupboard, in kitchen)"],
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


def test_enabled_compact_judge_prompt_contains_context_and_parses_failure_json():
    judge = FakeJudgeLLM(
        '{"is_failure": true, "failure_type": "implicit_no_progress", '
        '"confidence": 0.82, "reason": "The action repeated prior exploration.", '
        '"evidence_for_failure": ["same action and same observation"], '
        '"evidence_against_failure": [], "productive_signal": "none"}'
    )
    detector = ScienceWorldFailureDetector(
        judge_llm=judge,
        enable_implicit_failures=True,
        implicit_failure_confidence_threshold=0.8,
    )

    result = detector.detect(
        observation="You see the same room.",
        action="look around",
        action_history=["inventory", "look around"],
        task_type="melt",
        task_goal="Your task is to melt tin.",
        recent_history=[
            {"step": 0, "action": "inventory", "observation": "You have a metal pot.", "score": 10.0},
            {"step": 1, "action": "look around", "observation": "You see a kitchen.", "score": 10.0},
        ],
        score_before_action=10.0,
        score_after_action=10.0,
        step=2,
        variation_idx=21,
        look_before_action="This room is called the kitchen.",
        inventory_before_action="In your inventory, you see: a metal pot.",
        valid_actions_before_action=["look around", "inventory", "go to hallway"],
        look_after_action="This room is called the kitchen.",
        inventory_after_action="In your inventory, you see: a metal pot.",
        valid_actions_after_action=["look around", "inventory", "go to hallway"],
    )

    assert result.is_failure is True
    assert result.failure_type == "implicit_no_progress"
    assert result.detector_source == "judge"
    assert result.confidence == 0.82
    assert result.reason == "The action repeated prior exploration."
    assert result.productive_signal == "none"
    assert result.evidence_for_failure == ["same action and same observation"]

    call = judge.prompts[0]
    prompt = call["prompt"]
    assert call["label"] == "judge"
    assert call["max_tokens"] == 128
    assert "Task type: melt" in prompt
    assert "Variation index: 21" in prompt
    assert "Step: 2" in prompt
    assert "Task goal / initial observation:" in prompt
    assert "Your task is to melt tin." in prompt
    assert "[history 1] Action: inventory" in prompt
    assert "Current action: look around" in prompt
    assert "Current observation: You see the same room." in prompt
    assert "Score before action: 10.0" in prompt
    assert "Score after action: 10.0" in prompt
    assert "Score delta: 0.0" in prompt
    assert "Look before action:" in prompt
    assert "Inventory before action:" in prompt
    assert "Valid actions before action:" in prompt
    assert "Look after action:" in prompt
    assert "Inventory after action:" in prompt
    assert "Valid actions after action:" in prompt
    assert "open drawer -> examine drawer" in prompt
    assert "repair_strategy" not in prompt
    assert "repair_action" not in prompt
    assert result.has_judge_repair_advice is False


def test_judge_prompt_includes_five_recent_history_items():
    judge = FakeJudgeLLM(
        '{"is_failure": false, "failure_type": "productive", '
        '"confidence": 0.2, "reason": "Recent exploration is productive.", '
        '"evidence_for_failure": [], "evidence_against_failure": [], '
        '"productive_signal": "new_info"}'
    )
    detector = ScienceWorldFailureDetector(
        judge_llm=judge,
        enable_implicit_failures=True,
    )
    history = [
        {"step": idx, "action": f"action {idx}", "observation": f"obs {idx}", "score": 0}
        for idx in range(12)
    ]

    detector.detect(
        observation="You see a room.",
        action="look around",
        action_history=["look around"],
        recent_history=history,
    )

    prompt = judge.prompts[0]["prompt"]
    assert "Action: action 6" not in prompt
    assert "Action: action 7" in prompt
    assert "Action: action 11" in prompt


def test_valid_actions_prompt_keeps_context_relevant_actions_beyond_default_prefix():
    judge = FakeJudgeLLM(
        '{"is_failure": false, "failure_type": "productive", '
        '"confidence": 0.2, "reason": "No failure.", '
        '"evidence_for_failure": [], "evidence_against_failure": [], '
        '"productive_signal": "none"}'
    )
    detector = ScienceWorldFailureDetector(
        judge_llm=judge,
        enable_implicit_failures=True,
    )
    valid_actions = [f"examine filler {idx}" for idx in range(35)]
    valid_actions.append("open door to workshop")

    detector.detect(
        observation="You are still in the hallway.",
        action="go to workshop",
        action_history=["go to workshop"],
        valid_actions_after_action=valid_actions,
    )

    prompt = judge.prompts[0]["prompt"]
    assert "open door to workshop" in prompt


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


def test_rule_miss_always_calls_judge_for_implicit_detection():
    judge = FakeJudgeLLM(
        '{"is_failure": false, "failure_type": "productive", '
        '"confidence": 0.8, "reason": "This is task-relevant exploration.", '
        '"evidence_for_failure": [], "evidence_against_failure": ["new container information"], '
        '"productive_signal": "new_info"}'
    )
    detector = ScienceWorldFailureDetector(
        judge_llm=judge,
        enable_implicit_failures=True,
    )

    result = detector.detect(
        observation="a drawer. In the drawer is: nothing",
        action="examine drawer",
        action_history=["open drawer", "examine drawer"],
        task_type="melt",
        task_goal="Your task is to melt tin.",
        score_before_action=0,
        score_after_action=0,
    )

    assert result.is_failure is False
    assert len(judge.prompts) == 1


def test_productive_signal_blocks_common_scienceworld_exploration_false_positives():
    examples = [
        ("examine drawer", "a drawer. In the drawer is: nothing", "new_info"),
        ("open freezer", "The freezer is now open.", "state_change"),
        ("examine freezer", "a freezer. The freezer door is open. In the freezer is: nothing.", "new_info"),
        ("open fridge", "The fridge is now open.", "state_change"),
        ("examine fridge", "a fridge. The fridge door is open. In the fridge is: a wood cup.", "new_info"),
        ("examine tin cup", "a tin cup (containing nothing)", "task_relevant_probe"),
        ("pick up thermometer", "You move the thermometer to the inventory.", "state_change"),
        ("use thermometer on metal pot", "the thermometer measures a temperature of 149 degrees celsius", "new_info"),
        ("wait", "You decide to wait for 10 iterations.", "time_progress"),
    ]

    for action, observation, productive_signal in examples:
        judge = FakeJudgeLLM(
            '{"is_failure": true, "failure_type": "irrelevant_action", '
            '"confidence": 0.95, "reason": "No score increase.", '
            '"evidence_for_failure": ["score did not increase"], '
            '"evidence_against_failure": ["the action provides task-relevant information"], '
            f'"productive_signal": "{productive_signal}"' + "}"
        )
        detector = ScienceWorldFailureDetector(
            judge_llm=judge,
            enable_implicit_failures=True,
            implicit_failure_confidence_threshold=0.8,
        )

        result = detector.detect(
            observation=observation,
            action=action,
            action_history=["open drawer", action],
            task_type="boil",
            task_goal="Your task is to boil tin.",
            score_before_action=0,
            score_after_action=0,
        )

        assert result.is_failure is False, action
        assert len(judge.prompts) == 1


def test_low_confidence_or_missing_evidence_judge_failure_is_rejected():
    low_confidence = ScienceWorldFailureDetector(
        judge_llm=FakeJudgeLLM(
            '{"is_failure": true, "failure_type": "implicit_no_progress", '
            '"confidence": 0.6, "reason": "Maybe unhelpful.", '
            '"evidence_for_failure": ["weak guess"], '
            '"evidence_against_failure": [], "productive_signal": "none"}'
        ),
        enable_implicit_failures=True,
        implicit_failure_confidence_threshold=0.8,
    )
    missing_evidence = ScienceWorldFailureDetector(
        judge_llm=FakeJudgeLLM(
            '{"is_failure": true, "failure_type": "implicit_no_progress", '
            '"confidence": 0.95, "reason": "No score increase.", '
            '"evidence_for_failure": [], '
            '"evidence_against_failure": [], "productive_signal": "none"}'
        ),
        enable_implicit_failures=True,
        implicit_failure_confidence_threshold=0.8,
    )

    for detector in (low_confidence, missing_evidence):
        result = detector.detect(
            observation="The room looks unchanged.",
            action="look around",
            action_history=["inventory", "look around"],
            score_before_action=5,
            score_after_action=5,
        )
        assert result.is_failure is False


def test_high_confidence_judge_failure_with_evidence_is_accepted():
    judge = FakeJudgeLLM(
        '{"is_failure": true, "failure_type": "redundant_repeat", '
        '"confidence": 0.94, "reason": "The exact same command repeated with the same result.", '
        '"evidence_for_failure": ["same action repeated three times", "same observation"], '
        '"evidence_against_failure": [], "productive_signal": "none"}'
    )
    detector = ScienceWorldFailureDetector(
        judge_llm=judge,
        enable_implicit_failures=True,
        implicit_failure_confidence_threshold=0.8,
    )

    result = detector.detect(
        observation="The room looks unchanged.",
        action="look around",
        action_history=["inventory", "look around"],
        score_before_action=5,
        score_after_action=5,
    )

    assert result.is_failure is True
    assert result.failure_type == "redundant_repeat"
    assert result.detector_source == "judge"
    assert result.judge_repair_strategy == ""
    assert result.judge_repair_action == ""
    assert result.judge_repair_confidence is None
    assert result.judge_repair_rationale == ""
    assert result.has_judge_repair_advice is False


def test_compact_judge_ignores_repair_fields_in_detection_response():
    detector = ScienceWorldFailureDetector(
        judge_llm=FakeJudgeLLM(
            '{"is_failure": true, "failure_type": "implicit_no_progress", '
            '"confidence": 0.94, "reason": "The action did not move toward the task.", '
            '"evidence_for_failure": ["same state"], '
            '"evidence_against_failure": [], "productive_signal": "none", '
            '"repair_strategy": "This legacy field should be ignored.", '
            '"repair_action": "inventory", '
            '"repair_confidence": 0.9, '
            '"repair_rationale": "This legacy field should be ignored."}'
        ),
        enable_implicit_failures=True,
        implicit_failure_confidence_threshold=0.8,
        repair_confidence_threshold=0.7,
    )

    result = detector.detect(
        observation="The room looks unchanged.",
        action="look around",
        action_history=["inventory", "look around"],
        score_before_action=5,
        score_after_action=5,
    )

    assert result.is_failure is True
    assert result.has_judge_repair_advice is False
    assert result.judge_advice_source == ""


def test_judge_cache_reuses_same_compact_decision_without_second_llm_call():
    judge = FakeJudgeLLM(
        '{"is_failure": true, "failure_type": "implicit_no_progress", '
        '"confidence": 0.91, "reason": "The same state repeated.", '
        '"evidence_for_failure": ["same observation"], '
        '"evidence_against_failure": [], "productive_signal": "none"}'
    )
    detector = ScienceWorldFailureDetector(
        judge_llm=judge,
        enable_implicit_failures=True,
        enable_cache=True,
    )

    first = detector.detect(
        observation="The room looks unchanged.",
        action="look around",
        action_history=["look around"],
        task_type="melt",
        task_goal="Your task is to melt tin.",
        score_before_action=5,
        score_after_action=5,
    )
    second = detector.detect(
        observation="The room looks unchanged.",
        action="look around",
        action_history=["look around"],
        task_type="melt",
        task_goal="Your task is to melt tin.",
        score_before_action=5,
        score_after_action=5,
    )

    assert first.is_failure is True
    assert first.judge_cache_hit is False
    assert first.judge_call_type == "implicit_detector"
    assert second.is_failure is True
    assert second.judge_cache_hit is True
    assert second.judge_call_type == "cache"
    assert len(judge.prompts) == 1


def test_generate_rule_repair_advice_parses_valid_json():
    judge = FakeJudgeLLM(
        '{"repair_strategy": "Open the blocked door before moving.", '
        '"repair_action": "open door to hallway", '
        '"repair_confidence": 0.91, '
        '"repair_rationale": "The movement failed because the door is closed."}'
    )
    detector = ScienceWorldFailureDetector(
        judge_llm=judge,
        repair_confidence_threshold=0.7,
    )

    advice = detector.generate_rule_repair_advice(
        action="go to hallway",
        observation="The door is not open.",
        failure_type="precondition_blocked",
        failure_reason="Observation contains failure indicator: 'the door is not open'",
        task_type="melt",
        task_goal="Your task is to melt tin.",
        recent_history=[{"step": 0, "action": "look around", "observation": "You see a door.", "score": 0}],
        score_before_action=0,
        score_after_action=0,
        step=1,
        variation_idx=3,
        look_after_action="A door to the hallway (that is closed).",
        inventory_after_action="In your inventory, you see: nothing.",
        valid_actions_after_action="open door to hallway\ngo to hallway",
    )

    assert advice.repair_strategy == "Open the blocked door before moving."
    assert advice.repair_action == "open door to hallway"
    assert advice.repair_confidence == 0.91
    assert advice.repair_rationale == "The movement failed because the door is closed."
    assert advice.source == "rule_repair_judge"

    call = judge.prompts[0]
    assert call["label"] == "rule_repair"
    assert call["max_tokens"] == 192
    prompt = call["prompt"]
    assert "Failure type: precondition_blocked" in prompt
    assert "Failed action: go to hallway" in prompt
    assert "Valid actions after action:" in prompt
    assert "open door to hallway" in prompt
    assert "Do not decide whether a failure happened" in prompt


def test_generate_rule_repair_advice_rejects_invalid_outputs():
    responses = [
        '{"repair_strategy": "Maybe try something.", "repair_action": "inventory", '
        '"repair_confidence": 0.2, "repair_rationale": "Weak guess."}',
        '{"repair_strategy": "Try a valid action.", "repair_action": "", '
        '"repair_confidence": 0.9, "repair_rationale": "Missing action."}',
        '{"repair_strategy": "Try a valid action.", '
        '"repair_action": "You should open the door first.", '
        '"repair_confidence": 0.9, "repair_rationale": "Natural language."}',
        '{"repair_strategy": "Open then move.", '
        '"repair_action": "open door to hallway -> go to hallway", '
        '"repair_confidence": 0.9, "repair_rationale": "Multi-action chain."}',
    ]

    for response in responses:
        detector = ScienceWorldFailureDetector(
            judge_llm=FakeJudgeLLM(response),
            repair_confidence_threshold=0.7,
        )

        advice = detector.generate_rule_repair_advice(
            action="go to hallway",
            observation="The door is not open.",
            failure_type="precondition_blocked",
            failure_reason="door closed",
        )

        assert advice is None


def test_generate_rule_repair_advice_requires_valid_action_when_available():
    detector = ScienceWorldFailureDetector(
        judge_llm=FakeJudgeLLM(
            '{"repair_strategy": "Use the environment-supported thermometer action.", '
            '"repair_action": "connect agent to thermometer", '
            '"repair_confidence": 0.9, '
            '"repair_rationale": "The previous action used invalid syntax."}'
        ),
        repair_confidence_threshold=0.7,
    )

    advice = detector.generate_rule_repair_advice(
        action="focus on thermometer",
        observation="No known action matches that input.",
        failure_type="syntax_or_parse",
        failure_reason="Action not recognized",
        valid_actions_after_action=[
            "look around",
            "inventory",
            "focus on substance called thermometer",
        ],
    )

    assert advice is None


def test_generate_rule_repair_advice_rejects_semantically_invalid_connect_action():
    detector = ScienceWorldFailureDetector(
        judge_llm=FakeJudgeLLM(
            '{"repair_strategy": "Use a valid navigation action.", '
            '"repair_action": "connect agent to kitchen", '
            '"repair_confidence": 0.9, '
            '"repair_rationale": "The action appears in the valid action list."}'
        ),
        repair_confidence_threshold=0.7,
    )

    advice = detector.generate_rule_repair_advice(
        action="go to kitchen",
        observation="No known action matches that input.",
        failure_type="syntax_or_parse",
        failure_reason="Action not recognized",
        task_goal="Your task is to melt tin.",
        look_after_action="You are in a hallway. There is a kitchen door.",
        valid_actions_after_action=[
            "connect agent to kitchen",
            "go to kitchen",
            "open door to kitchen",
        ],
    )

    assert advice is None
    prompt = detector.judge_llm.prompts[0]["prompt"]
    assert "connect agent to kitchen" not in prompt
    assert "go to kitchen" in prompt
    assert "open door to kitchen" in prompt


def test_generate_rule_repair_advice_accepts_connect_action_in_circuit_context():
    detector = ScienceWorldFailureDetector(
        judge_llm=FakeJudgeLLM(
            '{"repair_strategy": "Connect the circuit parts.", '
            '"repair_action": "connect red wire to battery", '
            '"repair_confidence": 0.9, '
            '"repair_rationale": "The task context is electrical."}'
        ),
        repair_confidence_threshold=0.7,
    )

    advice = detector.generate_rule_repair_advice(
        action="connect wire",
        observation="No known action matches that input.",
        failure_type="syntax_or_parse",
        failure_reason="Action not recognized",
        task_goal="Your task is to complete the electrical circuit.",
        look_after_action="You see a red wire, a black wire, and a battery terminal.",
        valid_actions_after_action=[
            "connect red wire to battery",
            "connect black wire to battery",
            "look around",
        ],
    )

    assert advice is not None
    assert advice.repair_action == "connect red wire to battery"


def test_generate_rule_repair_advice_accepts_numeric_ambiguity_choice_when_valid():
    detector = ScienceWorldFailureDetector(
        judge_llm=FakeJudgeLLM(
            '{"repair_strategy": "Choose the intended disambiguation option.", '
            '"repair_action": "0", '
            '"repair_confidence": 0.9, '
            '"repair_rationale": "The environment asked for a numeric choice."}'
        ),
        repair_confidence_threshold=0.7,
    )

    advice = detector.generate_rule_repair_advice(
        action="examine drawer",
        observation="Ambiguous request. Please enter the number for the action you intended.",
        failure_type="ambiguity",
        failure_reason="Ambiguous request",
        valid_actions_after_action=["0", "1"],
    )

    assert advice is not None
    assert advice.repair_action == "0"


def test_generate_rule_repair_advice_malformed_json_or_exception_returns_none():
    detectors = [
        ScienceWorldFailureDetector(judge_llm=FakeJudgeLLM("not json")),
        ScienceWorldFailureDetector(judge_llm=FakeJudgeLLM(exc=RuntimeError("network down"))),
    ]

    for detector in detectors:
        advice = detector.generate_rule_repair_advice(
            action="open door",
            observation="Ambiguous request.",
            failure_type="ambiguity",
            failure_reason="ambiguous request",
        )
        assert advice is None


def test_generate_rule_repair_advice_accepts_llm_json_with_raw_newline_in_string():
    detector = ScienceWorldFailureDetector(
        judge_llm=FakeJudgeLLM(
            '{"repair_strategy": "Open the blocked door.", '
            '"repair_action": "open door", '
            '"repair_confidence": 0.91, '
            '"repair_rationale": "The door is closed.\nOpening it satisfies the precondition."}'
        ),
        repair_confidence_threshold=0.7,
    )

    advice = detector.generate_rule_repair_advice(
        action="go to kitchen",
        observation="The door is not open.",
        failure_type="precondition_blocked",
        failure_reason="door closed",
    )

    assert advice is not None
    assert advice.repair_action == "open door"
    assert "Opening it satisfies" in advice.repair_rationale


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


def test_truncated_non_failure_judge_json_uses_conservative_fallback(caplog):
    truncated = (
        '{ "is_failure": false, "failure_type": "productive", '
        '"confidence": 0.95, '
        '"reason": "Examining the open cupboard confirms contents.", '
        '"evidence_for_failure": [], '
        '"evidence_against_failure": ["Cupboard was already open, but examining it confirms '
    )
    detector = ScienceWorldFailureDetector(
        judge_llm=FakeJudgeLLM(truncated),
        enable_implicit_failures=True,
    )

    result = detector.detect(
        observation="a cupboard. The cupboard door is open.",
        action="examine cupboard",
        action_history=["open cupboard", "examine cupboard"],
        score_before_action=0,
        score_after_action=0,
    )

    assert result.is_failure is False
    assert result.confidence == 0.95
    assert result.reason == "Examining the open cupboard confirms contents."
    assert result.productive_signal == "unknown"
    assert "Judge JSON parse failed" not in caplog.text

"""Tests for deterministic ScienceWorld retrieval relevance gating."""

from src.memory import FailureMemoryEntry
from src.scienceworld_retrieval_gate import select_retrieval_memory


def _entry(
    *,
    memory_id=1,
    failure_type="precondition_blocked",
    failure_action="go to workshop",
    failure_observation="The door is not open.",
    repair_action="open door to workshop -> go to workshop",
    confidence_score=0.9,
):
    return FailureMemoryEntry(
        memory_id=memory_id,
        failure_action=failure_action,
        failure_observation=failure_observation,
        solution_action=repair_action,
        task_type="melt",
        repair_strategy="Satisfy the missing precondition before retrying movement.",
        repair_tactic="Open the relevant door, then retry movement.",
        repair_action=repair_action,
        failure_type=failure_type,
        confidence_score=confidence_score,
    )


def test_precondition_blocked_open_door_memory_passes_gate():
    decision = select_retrieval_memory(
        entries=[_entry()],
        retrieval_scores=[0.0327],
        current_failure_type="precondition_blocked",
        failed_action="go to kitchen",
        failure_observation="The door is not open.",
        recent_actions=["look around", "go to kitchen"],
        relevance_score_threshold=0.45,
    )

    assert decision.selected_entry is not None
    assert decision.selected_entry.memory_id == 1
    assert decision.relevance_decision is True
    assert decision.relevance_scores[0] >= 0.45
    assert decision.rejection_reason == ""


def test_precondition_blocked_allows_open_door_then_retry_failed_move():
    decision = select_retrieval_memory(
        entries=[
            _entry(
                failure_action="go to kitchen",
                repair_action="open door to kitchen -> go to kitchen",
            )
        ],
        retrieval_scores=[0.0327],
        current_failure_type="precondition_blocked",
        failed_action="go to kitchen",
        failure_observation="The door is not open.",
        recent_actions=["look around", "go to kitchen"],
        relevance_score_threshold=0.45,
    )

    assert decision.selected_entry is not None
    assert decision.relevance_decision is True
    assert decision.rejection_reason == ""


def test_syntax_parse_rejects_repeating_invalid_action():
    decision = select_retrieval_memory(
        entries=[
            _entry(
                memory_id=2,
                failure_type="syntax_or_parse",
                failure_action="open door to kitchen",
                failure_observation="No known action matches that input.",
                repair_action="open door to kitchen",
            )
        ],
        retrieval_scores=[0.0327],
        current_failure_type="syntax_or_parse",
        failed_action="open door to kitchen",
        failure_observation="No known action matches that input.",
        recent_actions=["open door to kitchen", "open door to kitchen"],
        relevance_score_threshold=0.45,
    )

    assert decision.selected_entry is None
    assert decision.relevance_decision is False
    assert decision.filtered_by_safety_count == 1
    assert decision.rejection_reason == "candidate_would_repeat_failed_action"


def test_action_loop_rejects_recently_repeated_repair_action():
    decision = select_retrieval_memory(
        entries=[
            _entry(
                memory_id=3,
                failure_type="action_loop",
                failure_action="open door",
                failure_observation="The door is already open.",
                repair_action="open door",
            )
        ],
        retrieval_scores=[0.0327],
        current_failure_type="action_loop",
        failed_action="open door",
        failure_observation="The door is already open.",
        recent_actions=["open door", "open door"],
        relevance_score_threshold=0.45,
    )

    assert decision.selected_entry is None
    assert decision.filtered_by_safety_count == 1
    assert decision.rejection_reason == "candidate_would_repeat_recent_action"


def test_incompatible_failure_type_is_filtered_before_scoring():
    decision = select_retrieval_memory(
        entries=[
            _entry(
                memory_id=4,
                failure_type="irrelevant_action",
                failure_action="examine battery",
                failure_observation="a battery",
                repair_action="focus on tin",
            )
        ],
        retrieval_scores=[0.0327],
        current_failure_type="syntax_or_parse",
        failed_action="open door to kitchen",
        failure_observation="No known action matches that input.",
        recent_actions=[],
        relevance_score_threshold=0.45,
    )

    assert decision.selected_entry is None
    assert decision.filtered_by_type_count == 1
    assert decision.rejection_reason == "no_type_compatible_candidates"


def test_navigation_syntax_does_not_match_object_transfer_memory():
    decision = select_retrieval_memory(
        entries=[
            _entry(
                memory_id=5,
                failure_type="syntax_or_parse",
                failure_action="put down unknown substance B in purple box",
                failure_observation="No known action matches that input.",
                repair_action="move unknown substance B to purple box",
            )
        ],
        retrieval_scores=[0.0327],
        current_failure_type="syntax_or_parse",
        failed_action="go to living room",
        failure_observation="No known action matches that input.",
        recent_actions=["look around", "go to living room"],
        relevance_score_threshold=0.45,
    )

    assert decision.selected_entry is None
    assert decision.filtered_by_intent_count == 1
    assert decision.current_intent == "navigation"
    assert decision.candidate_intents == ["object_transfer"]
    assert decision.rejection_reason == "intent_incompatible"


def test_object_transfer_syntax_matches_object_transfer_memory():
    decision = select_retrieval_memory(
        entries=[
            _entry(
                memory_id=6,
                failure_type="syntax_or_parse",
                failure_action="put down unknown substance B in purple box",
                failure_observation="No known action matches that input.",
                repair_action="move unknown substance B to purple box",
            )
        ],
        retrieval_scores=[0.0327],
        current_failure_type="syntax_or_parse",
        failed_action="put down unknown substance B in purple box",
        failure_observation="No known action matches that input.",
        recent_actions=["pick up unknown substance B"],
        relevance_score_threshold=0.45,
    )

    assert decision.selected_entry is not None
    assert decision.selected_entry.memory_id == 6
    assert decision.relevance_decision is True
    assert decision.current_intent == "object_transfer"
    assert decision.candidate_intents == ["object_transfer"]


def test_matching_failure_intent_still_rejects_incompatible_repair_intent():
    decision = select_retrieval_memory(
        entries=[
            _entry(
                memory_id=7,
                failure_type="syntax_or_parse",
                failure_action="go to living room",
                failure_observation="No known action matches that input.",
                repair_action="connect agent to kitchen",
            )
        ],
        retrieval_scores=[0.0327],
        current_failure_type="syntax_or_parse",
        failed_action="go to living room",
        failure_observation="No known action matches that input.",
        recent_actions=["look around"],
        relevance_score_threshold=0.45,
    )

    assert decision.selected_entry is None
    assert decision.filtered_by_intent_count == 1
    assert decision.rejection_reason == "repair_intent_incompatible"

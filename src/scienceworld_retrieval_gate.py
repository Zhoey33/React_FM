"""Deterministic relevance gate for ScienceWorld failure-memory retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.scienceworld_action_intent import (
    are_failure_intents_compatible,
    is_repair_compatible,
    parse_action_intent,
)


_COMPATIBLE_FAILURE_TYPES: dict[str, set[str]] = {
    "precondition_blocked": {"precondition_blocked", "ambiguity", "syntax_or_parse"},
    "syntax_or_parse": {"syntax_or_parse", "ambiguity"},
    "ambiguity": {"ambiguity", "syntax_or_parse"},
    "redundant_repeat": {"redundant_repeat", "action_loop", "implicit_no_progress"},
    "action_loop": {"action_loop", "redundant_repeat", "syntax_or_parse"},
    "implicit_no_progress": {
        "implicit_no_progress",
        "irrelevant_action",
        "precondition_blocked",
    },
    "irrelevant_action": {"irrelevant_action", "implicit_no_progress"},
    "premature_action": {"premature_action", "precondition_blocked"},
}

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "between",
    "in",
    "is",
    "it",
    "not",
    "of",
    "on",
    "or",
    "that",
    "the",
    "there",
    "to",
    "with",
    "you",
}


@dataclass
class RetrievalGateDecision:
    """Decision object returned by deterministic ScienceWorld retrieval gating."""

    selected_entry: Any | None
    candidate_memory_ids: list[int]
    candidate_scores: list[float]
    relevance_scores: list[float]
    selected_memory_id: int | None = None
    relevance_decision: bool = False
    rejection_reason: str = ""
    filtered_by_type_count: int = 0
    filtered_by_safety_count: int = 0
    filtered_by_intent_count: int = 0
    current_intent: str = ""
    candidate_intents: list[str] | None = None


def _normalize_action(action: str) -> str:
    action = action.lower().strip()
    action = action.replace("→", "->")
    action = re.sub(r"[^a-z0-9 ]+", " ", action)
    return re.sub(r"\s+", " ", action).strip()


def _split_repair_actions(repair_action: str) -> list[str]:
    normalized = repair_action.replace("→", "->")
    parts = re.split(r"\s*->\s*|\s*\n\s*", normalized)
    return [_normalize_action(part) for part in parts if _normalize_action(part)]


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOPWORDS and len(token) > 1
    }


def _overlap_score(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _is_type_compatible(current_type: str, memory_type: str) -> bool:
    if not current_type or not memory_type:
        return True
    if current_type == memory_type:
        return True
    return memory_type in _COMPATIBLE_FAILURE_TYPES.get(current_type, {current_type})


def _safety_rejection(
    entry: Any,
    *,
    current_failure_type: str,
    failed_action: str,
    failure_observation: str,
    recent_actions: list[str],
) -> str:
    repair_action = str(
        getattr(entry, "repair_action", "") or getattr(entry, "solution_action", "")
    )
    repair_actions = _split_repair_actions(repair_action)
    if not repair_actions:
        return "candidate_missing_repair_action"

    normalized_failed = _normalize_action(failed_action)
    is_precondition_retry_repair = (
        current_failure_type == "precondition_blocked"
        and "door is not open" in failure_observation.lower()
        and any(action.startswith("open door") for action in repair_actions)
    )
    if current_failure_type in {"action_loop", "redundant_repeat"}:
        recent = {_normalize_action(action) for action in recent_actions[-2:]}
        if any(action in recent for action in repair_actions):
            return "candidate_would_repeat_recent_action"

    if (
        normalized_failed
        and normalized_failed in repair_actions
        and not is_precondition_retry_repair
    ):
        return "candidate_would_repeat_failed_action"

    if "no known action matches that input" in failure_observation.lower():
        if normalized_failed and normalized_failed in repair_actions:
            return "candidate_would_repeat_failed_action"

    return ""


def _candidate_repair_frames(entry: Any, *, context: str) -> list[Any]:
    repair_action = str(
        getattr(entry, "repair_action", "") or getattr(entry, "solution_action", "")
    )
    return [
        parse_action_intent(action, context=context)
        for action in _split_repair_actions(repair_action)
    ]


def _entry_intent_context(entry: Any, extra_context: str = "") -> str:
    return " ".join(
        [
            str(extra_context or ""),
            str(getattr(entry, "failure_action", "") or ""),
            str(getattr(entry, "failure_observation", "") or ""),
            str(getattr(entry, "repair_action", "") or ""),
            str(getattr(entry, "solution_action", "") or ""),
        ]
    )


def _candidate_intent_label(entry: Any, *, context: str) -> str:
    entry_context = _entry_intent_context(entry, context)
    frames = _candidate_repair_frames(entry, context=entry_context)
    for frame in frames:
        if frame.intent not in {"unknown", "examine_info", "wait"}:
            return frame.intent
    failure_frame = parse_action_intent(
        str(getattr(entry, "failure_action", "") or ""),
        entry_context,
    )
    return failure_frame.intent


def _intent_rejection(
    entry: Any,
    *,
    current_failure_type: str,
    failed_action: str,
    failure_observation: str,
) -> str:
    context = " ".join(
        [
            str(current_failure_type or ""),
            str(failed_action or ""),
            str(failure_observation or ""),
            str(getattr(entry, "failure_action", "") or ""),
            str(getattr(entry, "failure_observation", "") or ""),
            str(getattr(entry, "repair_action", "") or ""),
            str(getattr(entry, "solution_action", "") or ""),
        ]
    )
    current_frame = parse_action_intent(failed_action, context=context)
    memory_frame = parse_action_intent(str(getattr(entry, "failure_action", "") or ""), context)
    failure_intents_compatible = are_failure_intents_compatible(
        current_frame,
        memory_frame,
        current_failure_type=current_failure_type,
        observation=failure_observation,
    )

    repair_frames = _candidate_repair_frames(entry, context=context)
    if any(
        is_repair_compatible(
            current_failure_type,
            current_frame,
            repair_frame,
            observation=failure_observation,
            context=context,
        )
        for repair_frame in repair_frames
    ):
        return ""
    if failure_intents_compatible and repair_frames:
        return "repair_intent_incompatible"
    return "intent_incompatible"


def _relevance_score(
    entry: Any,
    *,
    current_failure_type: str,
    failed_action: str,
    failure_observation: str,
) -> float:
    memory_type = str(getattr(entry, "failure_type", "") or "")
    score = 0.0

    if current_failure_type and memory_type:
        score += 0.3 if current_failure_type == memory_type else 0.18

    score += min(_overlap_score(failed_action, getattr(entry, "failure_action", "")), 1.0) * 0.2
    score += min(
        _overlap_score(failure_observation, getattr(entry, "failure_observation", "")),
        1.0,
    ) * 0.2

    repair_text = str(getattr(entry, "repair_action", "") or getattr(entry, "solution_action", ""))
    repair_overlap = _overlap_score(failed_action, repair_text)
    if (
        current_failure_type == "precondition_blocked"
        and "door is not open" in failure_observation.lower()
        and "open door" in repair_text.lower()
    ):
        repair_overlap = max(repair_overlap, 1.0)
    score += min(repair_overlap, 1.0) * 0.2

    confidence = getattr(entry, "confidence_score", None)
    if isinstance(confidence, int | float):
        score += max(0.0, min(float(confidence), 1.0)) * 0.1

    return round(max(0.0, min(score, 1.0)), 4)


def select_retrieval_memory(
    *,
    entries: list[Any],
    retrieval_scores: list[float],
    current_failure_type: str,
    failed_action: str,
    failure_observation: str,
    recent_actions: list[str],
    relevance_score_threshold: float = 0.45,
    enable_failure_type_filter: bool = True,
    enable_safety_gate: bool = True,
    enable_intent_gate: bool = True,
) -> RetrievalGateDecision:
    """Select one memory for injection using deterministic relevance checks."""
    candidate_ids = [int(getattr(entry, "memory_id", -1)) for entry in entries]
    candidate_scores = [float(score) for score in retrieval_scores]
    relevance_scores = [0.0 for _ in entries]
    current_intent = parse_action_intent(
        failed_action,
        context=" ".join([str(current_failure_type or ""), str(failure_observation or "")]),
    ).intent
    candidate_intents = [_candidate_intent_label(entry, context=failure_observation) for entry in entries]

    filtered_by_type = 0
    filtered_by_safety = 0
    filtered_by_intent = 0
    safety_rejection = ""
    intent_rejection = ""
    scored: list[tuple[float, int, Any]] = []

    for index, entry in enumerate(entries):
        memory_type = str(getattr(entry, "failure_type", "") or "")
        if enable_failure_type_filter and not _is_type_compatible(current_failure_type, memory_type):
            filtered_by_type += 1
            continue

        if enable_intent_gate:
            rejection = _intent_rejection(
                entry,
                current_failure_type=current_failure_type,
                failed_action=failed_action,
                failure_observation=failure_observation,
            )
            if rejection:
                filtered_by_intent += 1
                intent_rejection = intent_rejection or rejection
                continue

        if enable_safety_gate:
            rejection = _safety_rejection(
                entry,
                current_failure_type=current_failure_type,
                failed_action=failed_action,
                failure_observation=failure_observation,
                recent_actions=recent_actions,
            )
            if rejection:
                filtered_by_safety += 1
                safety_rejection = safety_rejection or rejection
                continue

        score = _relevance_score(
            entry,
            current_failure_type=current_failure_type,
            failed_action=failed_action,
            failure_observation=failure_observation,
        )
        relevance_scores[index] = score
        scored.append((score, index, entry))

    if not scored:
        if entries and filtered_by_type == len(entries):
            reason = "no_type_compatible_candidates"
        elif entries and filtered_by_intent == len(entries) - filtered_by_type:
            reason = intent_rejection or "intent_incompatible"
        elif entries and filtered_by_safety == len(entries) - filtered_by_type - filtered_by_intent:
            reason = safety_rejection or "no_safety_compatible_candidates"
        else:
            reason = "no_candidates"
        return RetrievalGateDecision(
            selected_entry=None,
            candidate_memory_ids=candidate_ids,
            candidate_scores=candidate_scores,
            relevance_scores=relevance_scores,
            rejection_reason=reason,
            filtered_by_type_count=filtered_by_type,
            filtered_by_safety_count=filtered_by_safety,
            filtered_by_intent_count=filtered_by_intent,
            current_intent=current_intent,
            candidate_intents=candidate_intents,
        )

    scored.sort(
        key=lambda item: (
            item[0],
            retrieval_scores[item[1]] if item[1] < len(retrieval_scores) else 0.0,
        ),
        reverse=True,
    )
    best_score, _, best_entry = scored[0]
    if best_score < relevance_score_threshold:
        return RetrievalGateDecision(
            selected_entry=None,
            candidate_memory_ids=candidate_ids,
            candidate_scores=candidate_scores,
            relevance_scores=relevance_scores,
            rejection_reason="below_relevance_threshold",
            filtered_by_type_count=filtered_by_type,
            filtered_by_safety_count=filtered_by_safety,
            filtered_by_intent_count=filtered_by_intent,
            current_intent=current_intent,
            candidate_intents=candidate_intents,
        )

    return RetrievalGateDecision(
        selected_entry=best_entry,
        selected_memory_id=int(getattr(best_entry, "memory_id", -1)),
        candidate_memory_ids=candidate_ids,
        candidate_scores=candidate_scores,
        relevance_scores=relevance_scores,
        relevance_decision=True,
        filtered_by_type_count=filtered_by_type,
        filtered_by_safety_count=filtered_by_safety,
        filtered_by_intent_count=filtered_by_intent,
        current_intent=current_intent,
        candidate_intents=candidate_intents,
    )

"""Rule-based + optional LLM judge failure detection for ScienceWorld."""

import json
import logging
from dataclasses import dataclass
from typing import Any

from src.llm import LLMClient

logger = logging.getLogger(__name__)


_EXPLICIT_FAILURE_PATTERNS: list[tuple[str, str]] = [
    ("unknown action", "syntax_or_parse"),
    ("ambiguous request", "ambiguity"),
    ("please enter the number for the action you intended", "ambiguity"),
    ("the door is not open", "precondition_blocked"),
    ("you don't have", "precondition_blocked"),
    ("you can't", "physics_or_affordance"),
    ("that doesn't seem to work", "no_effect_or_other"),
    ("nothing happens", "no_effect_or_other"),
    ("i'm not sure what you mean", "syntax_or_parse"),
]

ALLOWED_JUDGE_FAILURE_TYPES = {
    "implicit_no_progress",
    "redundant_repeat",
    "irrelevant_action",
    "premature_action",
}

_SCIENCEWORLD_ACTION_PREFIXES = (
    "go to", "pick up", "put down", "open", "close", "activate", "deactivate",
    "use", "pour", "mix", "focus on", "wait", "look around", "look", "inventory",
    "examine", "read", "connect", "move", "teleport to", "dunk", "eat", "drink",
    "flush", "reset task",
)

JUDGE_PROMPT = """You are judging whether one ScienceWorld agent interaction is an implicit failure.

Use the task goal, state context, recent history, current action/observation, and score delta.
The rule-based detector has already checked explicit environment errors. You are the implicit
failure judge and must be conservative: only flag a subtle failure when there is concrete
evidence that the action wasted the step or moves away from the task.

Task type: {task_type}
Variation index: {variation_idx}
Step: {step}

Task goal / initial observation:
{task_goal}

Recent history:
{recent_history}

Look before action:
{look_before_action}

Inventory before action:
{inventory_before_action}

Valid actions before action:
{valid_actions_before_action}

Current action: {action}
Current observation: {observation}
Score before action: {score_before_action}
Score after action: {score_after_action}
Score delta: {score_delta}

Look after action:
{look_after_action}

Inventory after action:
{inventory_after_action}

Valid actions after action:
{valid_actions_after_action}

Guidance:
- Many necessary ScienceWorld actions do not immediately increase score.
- Searching for objects, opening containers, examining empty containers, examining target objects,
  measuring temperature, and waiting during heating/cooling/growth should NOT be failures solely
  because score_delta is 0.
- open drawer -> examine drawer is an information-gathering chain, not redundant_repeat.
- redundant_repeat requires the same action or same target to be retried with the same outcome.
- A legal action can still be an implicit failure only if it is irrelevant, premature, redundant,
  or makes no progress according to concrete context evidence.

Return ONLY valid JSON with this schema:
{{
  "is_failure": true or false,
  "failure_type": "implicit_no_progress" | "redundant_repeat" | "irrelevant_action" | "premature_action" | "productive",
  "confidence": number between 0 and 1,
  "reason": "short explanation",
  "evidence_for_failure": ["concrete evidence"],
  "evidence_against_failure": ["counter-evidence"],
  "productive_signal": "new_info" | "state_change" | "task_relevant_probe" | "time_progress" | "none",
  "repair_strategy": "high-level immediate repair direction, or empty string",
  "repair_action": "one valid next ScienceWorld action, or empty string",
  "repair_confidence": number between 0 and 1,
  "repair_rationale": "short explanation for the suggested repair"
}}

Keep reason/rationale under 25 words. Use at most 2 evidence items per list.
If is_failure is false, set repair_strategy, repair_action, and repair_rationale to empty strings,
and repair_confidence to 0.0."""


RULE_REPAIR_PROMPT = """You are generating one immediate repair suggestion for a ScienceWorld agent.

Do not decide whether a failure happened. The rule-based detector already found a failure.
Your job is only to suggest one valid next ScienceWorld action that may recover from it.

Task type: {task_type}
Variation index: {variation_idx}
Step: {step}

Task goal / initial observation:
{task_goal}

Recent history:
{recent_history}

Failed action: {action}
Failure observation: {observation}
Failure type: {failure_type}
Detector reason: {failure_reason}
Score before action: {score_before_action}
Score after action: {score_after_action}
Score delta: {score_delta}

Look after action:
{look_after_action}

Inventory after action:
{inventory_after_action}

Valid actions after action:
{valid_actions_after_action}

Guidance:
- For precondition_blocked, satisfy the blocked precondition before retrying.
- If a door is not open, suggest the most specific valid open-door action from valid actions.
- For syntax_or_parse, do not repeat the invalid command; choose the closest valid action.
- For ambiguity, avoid vague object names; use a specific valid action or the requested numeric choice.
- For action_loop, stop repeating and suggest look around, inventory, or a different task-relevant action.
- Suggest exactly one next action, not a multi-step plan.

Return ONLY valid JSON with this schema:
{{
  "repair_strategy": "short high-level repair direction",
  "repair_action": "one valid next ScienceWorld action",
  "repair_confidence": number between 0 and 1,
  "repair_rationale": "short explanation"
}}

Keep repair_strategy and repair_rationale under 25 words each."""


@dataclass
class DetectionResult:
    is_failure: bool
    failure_type: str = ""
    reason: str = ""
    confidence: float | None = None
    detector_source: str = ""
    evidence_for_failure: list[str] | None = None
    evidence_against_failure: list[str] | None = None
    productive_signal: str = ""
    judge_repair_strategy: str = ""
    judge_repair_action: str = ""
    judge_repair_confidence: float | None = None
    judge_repair_rationale: str = ""
    judge_advice_source: str = ""

    @property
    def has_judge_repair_advice(self) -> bool:
        return bool(self.judge_repair_strategy and self.judge_repair_action)


@dataclass(frozen=True)
class RepairAdvice:
    repair_strategy: str
    repair_action: str
    repair_confidence: float
    repair_rationale: str
    source: str = "rule_repair_judge"


class ScienceWorldFailureDetector:
    def __init__(
        self,
        loop_window: int = 3,
        judge_llm: LLMClient | None = None,
        enable_implicit_failures: bool = False,
        implicit_failure_confidence_threshold: float = 0.8,
        judge_max_tokens: int = 512,
        repair_confidence_threshold: float = 0.7,
    ):
        self.loop_window = loop_window
        self.judge_llm = judge_llm
        self.enable_implicit_failures = enable_implicit_failures
        self.implicit_failure_confidence_threshold = implicit_failure_confidence_threshold
        self.judge_max_tokens = judge_max_tokens
        self.repair_confidence_threshold = repair_confidence_threshold

    def detect(
        self,
        observation: str,
        action: str,
        action_history: list[str],
        task_type: str = "",
        task_goal: str = "",
        recent_history: list[tuple[str, str]] | list[dict[str, Any]] | None = None,
        score_before_action: float | int | None = None,
        score_after_action: float | int | None = None,
        step: int | None = None,
        variation_idx: int | None = None,
        look_before_action: str = "",
        inventory_before_action: str = "",
        valid_actions_before_action: str | list[str] = "",
        look_after_action: str = "",
        inventory_after_action: str = "",
        valid_actions_after_action: str | list[str] = "",
    ) -> DetectionResult:
        obs = observation.strip().lower()

        # Rule 1: Unknown action
        if "no known action matches that input" in obs:
            return DetectionResult(
                is_failure=True,
                failure_type="syntax_or_parse",
                reason=f"Action '{action}' not recognized by environment",
                detector_source="rule",
            )

        # Rule 2: Empty observation
        if not obs:
            return DetectionResult(
                is_failure=True,
                failure_type="empty_obs",
                reason="Empty observation returned",
                detector_source="rule",
            )

        # Rule 3: Action loop
        if len(action_history) >= self.loop_window:
            recent = action_history[-self.loop_window:]
            if all(a == action for a in recent):
                return DetectionResult(
                    is_failure=True,
                    failure_type="action_loop",
                    reason=f"Same action '{action}' repeated {self.loop_window} times",
                    detector_source="rule",
                )

        # Rule 4: Explicit failure indicators
        for phrase, failure_type in _EXPLICIT_FAILURE_PATTERNS:
            if phrase in obs:
                return DetectionResult(
                    is_failure=True,
                    failure_type=failure_type,
                    reason=f"Observation contains failure indicator: '{phrase}'",
                    detector_source="rule",
                )

        # Rule 5: LLM judge
        if self.enable_implicit_failures and self.judge_llm is not None:
            return self._judge_intent(
                action=action,
                observation=observation,
                task_type=task_type,
                task_goal=task_goal,
                recent_history=recent_history or [],
                score_before_action=score_before_action,
                score_after_action=score_after_action,
                step=step,
                variation_idx=variation_idx,
                look_before_action=look_before_action,
                inventory_before_action=inventory_before_action,
                valid_actions_before_action=valid_actions_before_action,
                look_after_action=look_after_action,
                inventory_after_action=inventory_after_action,
                valid_actions_after_action=valid_actions_after_action,
            )

        return DetectionResult(is_failure=False)

    def _judge_intent(
        self,
        *,
        action: str,
        observation: str,
        task_type: str = "",
        task_goal: str = "",
        recent_history: list[tuple[str, str]] | list[dict[str, Any]] | None = None,
        score_before_action: float | int | None = None,
        score_after_action: float | int | None = None,
        step: int | None = None,
        variation_idx: int | None = None,
        look_before_action: str = "",
        inventory_before_action: str = "",
        valid_actions_before_action: str | list[str] = "",
        look_after_action: str = "",
        inventory_after_action: str = "",
        valid_actions_after_action: str | list[str] = "",
    ) -> DetectionResult:
        prompt = JUDGE_PROMPT.format(
            action=action,
            observation=observation,
            task_type=task_type or "unknown",
            variation_idx=variation_idx if variation_idx is not None else "unknown",
            step=step if step is not None else "unknown",
            task_goal=_compact_text(task_goal or "unknown"),
            recent_history=_format_recent_history(recent_history or []),
            score_before_action=score_before_action,
            score_after_action=score_after_action,
            score_delta=_score_delta(score_before_action, score_after_action),
            look_before_action=_compact_text(look_before_action or "unknown"),
            inventory_before_action=_compact_text(inventory_before_action or "unknown"),
            valid_actions_before_action=_format_valid_actions(valid_actions_before_action),
            look_after_action=_compact_text(look_after_action or "unknown"),
            inventory_after_action=_compact_text(inventory_after_action or "unknown"),
            valid_actions_after_action=_format_valid_actions(valid_actions_after_action),
        )
        try:
            response = self.judge_llm.complete_text(
                prompt,
                label="judge",
                max_tokens=self.judge_max_tokens,
            )
            try:
                payload = _parse_judge_json(response)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "Judge JSON parse failed: %s response=%s",
                    e,
                    _compact_text(response, max_chars=500),
                )
                return DetectionResult(is_failure=False)
            is_failure = _as_bool(payload.get("is_failure"))
            confidence = _as_confidence(payload.get("confidence"))
            evidence_for = _as_string_list(payload.get("evidence_for_failure"))
            evidence_against = _as_string_list(payload.get("evidence_against_failure"))
            productive_signal = str(payload.get("productive_signal", "") or "none").strip()
            reason = str(payload.get("reason", "")).strip()
            repair_strategy, repair_action, repair_confidence, repair_rationale = (
                _parse_repair_advice(
                    payload,
                    confidence_threshold=self.repair_confidence_threshold,
                    valid_actions=valid_actions_after_action,
                )
            )
            if not is_failure:
                logger.info(
                    "Judge result: non_failure confidence=%s productive_signal=%s reason=%s",
                    confidence,
                    productive_signal,
                    _compact_text(reason, max_chars=180),
                )
                return DetectionResult(is_failure=False)

            failure_type = str(payload.get("failure_type", "")).strip()
            if failure_type not in ALLOWED_JUDGE_FAILURE_TYPES:
                failure_type = "implicit_no_progress"
            if not _accept_judge_failure(
                confidence=confidence,
                evidence_for_failure=evidence_for,
                productive_signal=productive_signal,
                threshold=self.implicit_failure_confidence_threshold,
            ):
                logger.info(
                    "Judge result rejected: type=%s confidence=%s productive_signal=%s "
                    "evidence_for=%s reason=%s",
                    failure_type,
                    confidence,
                    productive_signal,
                    evidence_for,
                    _compact_text(reason, max_chars=180),
                )
                return DetectionResult(is_failure=False)
            logger.info(
                "Judge result accepted: type=%s confidence=%s productive_signal=%s reason=%s",
                failure_type,
                confidence,
                productive_signal,
                _compact_text(reason, max_chars=180),
            )
            return DetectionResult(
                is_failure=True,
                failure_type=failure_type,
                reason=reason,
                confidence=confidence,
                detector_source="judge",
                evidence_for_failure=evidence_for,
                evidence_against_failure=evidence_against,
                productive_signal=productive_signal,
                judge_repair_strategy=repair_strategy,
                judge_repair_action=repair_action,
                judge_repair_confidence=repair_confidence,
                judge_repair_rationale=repair_rationale,
                judge_advice_source="implicit_judge" if repair_strategy and repair_action else "",
            )
        except Exception as e:
            logger.warning(f"Judge LLM call failed: {e}")
        return DetectionResult(is_failure=False)

    def generate_rule_repair_advice(
        self,
        *,
        action: str,
        observation: str,
        failure_type: str,
        failure_reason: str,
        task_type: str = "",
        task_goal: str = "",
        recent_history: list[tuple[str, str]] | list[dict[str, Any]] | None = None,
        score_before_action: float | int | None = None,
        score_after_action: float | int | None = None,
        step: int | None = None,
        variation_idx: int | None = None,
        look_after_action: str = "",
        inventory_after_action: str = "",
        valid_actions_after_action: str | list[str] = "",
    ) -> RepairAdvice | None:
        """Ask the judge LLM for one repair action after a rule-detected failure."""
        if self.judge_llm is None:
            return None
        prompt = RULE_REPAIR_PROMPT.format(
            action=action,
            observation=observation,
            failure_type=failure_type,
            failure_reason=failure_reason,
            task_type=task_type or "unknown",
            variation_idx=variation_idx if variation_idx is not None else "unknown",
            step=step if step is not None else "unknown",
            task_goal=_compact_text(task_goal or "unknown"),
            recent_history=_format_recent_history(recent_history or []),
            score_before_action=score_before_action,
            score_after_action=score_after_action,
            score_delta=_score_delta(score_before_action, score_after_action),
            look_after_action=_compact_text(look_after_action or "unknown"),
            inventory_after_action=_compact_text(inventory_after_action or "unknown"),
            valid_actions_after_action=_format_valid_actions(valid_actions_after_action),
        )
        try:
            response = self.judge_llm.complete_text(
                prompt,
                label="rule_repair",
                max_tokens=self.judge_max_tokens,
            )
            try:
                payload = _parse_judge_json(response)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "Rule repair JSON parse failed: %s response=%s",
                    e,
                    _compact_text(response, max_chars=500),
                )
                return None
            strategy, repair_action, confidence, rationale = _parse_repair_advice(
                payload,
                confidence_threshold=self.repair_confidence_threshold,
                valid_actions=valid_actions_after_action,
            )
            if not strategy or not repair_action or confidence is None:
                logger.info(
                    "Rule repair advice rejected: confidence=%s action=%s rationale=%s",
                    confidence,
                    _compact_text(str(payload.get("repair_action", "")), max_chars=120),
                    _compact_text(str(payload.get("repair_rationale", "")), max_chars=180),
                )
                return None
            logger.info(
                "Rule repair advice accepted: action=%s confidence=%s rationale=%s",
                repair_action,
                confidence,
                _compact_text(rationale, max_chars=180),
            )
            return RepairAdvice(
                repair_strategy=strategy,
                repair_action=repair_action,
                repair_confidence=confidence,
                repair_rationale=rationale,
            )
        except Exception as e:
            logger.warning(f"Rule repair LLM call failed: {e}")
        return None

    def is_task_complete(self, observation: str, done: bool, info: dict) -> tuple[bool, bool]:
        """Check if episode is over and whether it succeeded.
        ScienceWorld API returns score as round(100 * getScore()).
        Full success is 100, but terminal failures can produce negative scores.
        """
        score = info.get("score", 0.0)
        is_success = score >= 100.0
        return done, is_success


def _compact_text(text: str, max_chars: int = 600) -> str:
    compact = " ".join(str(text).replace("\t", " ").split())
    if len(compact) > max_chars:
        return compact[: max_chars - 3] + "..."
    return compact


def _format_recent_history(
    recent_history: list[tuple[str, str]] | list[dict[str, Any]],
    *,
    max_items: int = 5,
) -> str:
    if not recent_history:
        return "(none)"

    lines = []
    for idx, item in enumerate(recent_history[-max_items:], start=1):
        if isinstance(item, dict):
            step = item.get("step", idx - 1)
            action = item.get("action", "")
            observation = item.get("observation", "")
            score = item.get("score", item.get("score_after_action", ""))
        else:
            step = idx - 1
            action, observation = item
            score = ""
        lines.append(
            f"[history {idx}] Step: {step}\n"
            f"[history {idx}] Action: {_compact_text(str(action), max_chars=160)}\n"
            f"[history {idx}] Observation: {_compact_text(str(observation), max_chars=240)}\n"
            f"[history {idx}] Score: {score}"
        )
    return "\n".join(lines)


def _format_valid_actions(valid_actions: str | list[str], max_items: int = 30) -> str:
    if isinstance(valid_actions, str):
        items = [line.strip() for line in valid_actions.splitlines() if line.strip()]
    else:
        items = [str(item).strip() for item in valid_actions if str(item).strip()]
    if not items:
        return "unknown"
    text = "\n".join(items[:max_items])
    if len(items) > max_items:
        text += f"\n... ({len(items) - max_items} more)"
    return text


def _valid_action_items(valid_actions: str | list[str]) -> list[str]:
    if isinstance(valid_actions, str):
        items = [line.strip() for line in valid_actions.splitlines() if line.strip()]
    else:
        items = [str(item).strip() for item in valid_actions if str(item).strip()]
    return [item for item in items if item.lower() != "unknown"]


def _normalize_action_text(action: str) -> str:
    return " ".join(action.strip().lower().split())


def _matches_available_action(action: str, valid_actions: str | list[str]) -> bool:
    items = _valid_action_items(valid_actions)
    if not items:
        return True
    normalized = _normalize_action_text(action)
    return any(normalized == _normalize_action_text(item) for item in items)


def _score_delta(
    score_before_action: float | int | None,
    score_after_action: float | int | None,
) -> float | None:
    if score_before_action is None or score_after_action is None:
        return None
    try:
        return round(float(score_after_action) - float(score_before_action), 4)
    except (TypeError, ValueError):
        return None


def _strip_code_fences(response: str) -> str:
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _extract_json_object_text(response: str) -> str:
    text = _strip_code_fences(response)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return text


def _parse_judge_json(response: str) -> dict[str, Any]:
    payload = json.loads(_extract_json_object_text(response))
    if not isinstance(payload, dict):
        raise ValueError("Judge response is not a JSON object")
    return payload


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return None


def _as_confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, confidence))


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _parse_repair_advice(
    payload: dict[str, Any],
    *,
    confidence_threshold: float,
    valid_actions: str | list[str] = "",
) -> tuple[str, str, float | None, str]:
    strategy = str(payload.get("repair_strategy", "") or "").strip()
    action = str(payload.get("repair_action", "") or "").strip()
    confidence = _as_confidence(payload.get("repair_confidence"))
    rationale = str(payload.get("repair_rationale", "") or "").strip()
    if confidence is None or confidence < confidence_threshold:
        return "", "", confidence, rationale
    if not strategy or not action:
        return "", "", confidence, rationale
    if not _looks_like_scienceworld_action(action):
        return "", "", confidence, rationale
    if not _matches_available_action(action, valid_actions):
        return "", "", confidence, rationale
    return strategy, action, confidence, rationale


def _looks_like_scienceworld_action(action: str) -> bool:
    normalized = action.strip().lower()
    if "\n" in normalized or "->" in normalized or ";" in normalized:
        return False
    return any(normalized == prefix or normalized.startswith(f"{prefix} ") for prefix in _SCIENCEWORLD_ACTION_PREFIXES)


def _accept_judge_failure(
    *,
    confidence: float | None,
    evidence_for_failure: list[str],
    productive_signal: str,
    threshold: float,
) -> bool:
    if confidence is None or confidence < threshold:
        return False
    if not evidence_for_failure:
        return False
    signal = (productive_signal or "none").strip().lower()
    if signal in {"new_info", "state_change", "task_relevant_probe", "time_progress"}:
        evidence_text = " ".join(evidence_for_failure).lower()
        strong_failure_cues = (
            "same action",
            "same observation",
            "repeat",
            "repeated",
            "unrelated",
            "off task",
            "premature",
            "contradict",
        )
        return any(cue in evidence_text for cue in strong_failure_cues)
    return True

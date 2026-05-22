"""Rule-based + optional LLM judge failure detection for ScienceWorld."""

import json
import logging
from dataclasses import dataclass
from typing import Any

from src.llm import LLMClient

logger = logging.getLogger(__name__)


_EXPLICIT_FAILURE_PATTERNS: list[tuple[str, str]] = [
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
  "productive_signal": "new_info" | "state_change" | "task_relevant_probe" | "time_progress" | "none"
}}"""


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


class ScienceWorldFailureDetector:
    def __init__(
        self,
        loop_window: int = 3,
        judge_llm: LLMClient | None = None,
        enable_implicit_failures: bool = False,
        implicit_failure_confidence_threshold: float = 0.8,
        judge_max_tokens: int = 256,
    ):
        self.loop_window = loop_window
        self.judge_llm = judge_llm
        self.enable_implicit_failures = enable_implicit_failures
        self.implicit_failure_confidence_threshold = implicit_failure_confidence_threshold
        self.judge_max_tokens = judge_max_tokens

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
            payload = _parse_judge_json(response)
            is_failure = _as_bool(payload.get("is_failure"))
            confidence = _as_confidence(payload.get("confidence"))
            evidence_for = _as_string_list(payload.get("evidence_for_failure"))
            evidence_against = _as_string_list(payload.get("evidence_against_failure"))
            productive_signal = str(payload.get("productive_signal", "") or "none").strip()
            reason = str(payload.get("reason", "")).strip()
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
            )
        except Exception as e:
            logger.warning(f"Judge LLM call failed: {e}")
        return DetectionResult(is_failure=False)

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

"""Lightweight action-intent parsing and compatibility checks for ScienceWorld."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


_STOPWORDS = {
    "a",
    "an",
    "and",
    "called",
    "in",
    "into",
    "is",
    "of",
    "on",
    "the",
    "to",
    "with",
}

_CIRCUIT_TERMS = {
    "battery",
    "black",
    "circuit",
    "connect",
    "electrical",
    "lead",
    "red",
    "terminal",
    "wire",
}

_BAD_CONNECT_OBJECTS = {
    "agent",
    "air",
    "bathroom",
    "bedroom",
    "cupboard",
    "door",
    "drawer",
    "freezer",
    "fridge",
    "hallway",
    "kitchen",
    "living",
    "room",
    "stove",
    "workshop",
}

_SAFE_FALLBACK_ACTIONS = {"inventory", "look", "look around"}


@dataclass(frozen=True)
class ActionIntentFrame:
    """Coarse semantic frame for a ScienceWorld action."""

    intent: str
    verb: str
    primary_object: str = ""
    target: str = ""
    tokens: frozenset[str] = frozenset()


def parse_action_intent(action: str, context: str = "") -> ActionIntentFrame:
    """Parse a ScienceWorld action into a coarse intent frame."""
    normalized = _normalize(action)
    tokens = frozenset(_tokens(normalized))
    if not normalized:
        return ActionIntentFrame("unknown", "", tokens=tokens)

    if normalized.isdigit():
        return ActionIntentFrame("generic_tool_use", normalized, tokens=tokens)

    if normalized in {"wait", "look", "look around", "inventory"}:
        intent = "wait" if normalized == "wait" else "examine_info"
        return ActionIntentFrame(intent, normalized, tokens=tokens)

    verb = _verb(normalized)
    primary, target = _objects_for_action(normalized, verb)

    if normalized.startswith(("go to ", "teleport to ")):
        return ActionIntentFrame("navigation", verb, primary, target, tokens)
    if normalized.startswith(("open ", "close ")):
        if "door" in tokens:
            return ActionIntentFrame("door_operation", verb, primary, target, tokens)
        return ActionIntentFrame("container_use", verb, primary, target, tokens)
    if normalized.startswith("pick up "):
        return ActionIntentFrame("object_pickup", verb, primary, target, tokens)
    if normalized.startswith(("move ", "put ", "put down ")):
        return ActionIntentFrame("object_transfer", verb, primary, target, tokens)
    if normalized.startswith(("pour ", "mix ", "dunk ")):
        return ActionIntentFrame("container_use", verb, primary, target, tokens)
    if normalized.startswith(("examine ", "read ", "focus on ")):
        return ActionIntentFrame("examine_info", verb, primary, target, tokens)
    if "thermometer" in tokens or normalized.startswith("measure "):
        return ActionIntentFrame("measurement", verb, primary, target, tokens)
    if normalized.startswith(("activate ", "deactivate ")) or (tokens & {"stove", "burner"}):
        return ActionIntentFrame("heating", verb, primary, target, tokens)
    if normalized.startswith("connect "):
        if _is_circuit_context(" ".join([normalized, context])):
            return ActionIntentFrame("circuit_connection", verb, primary, target, tokens)
        return ActionIntentFrame("generic_tool_use", verb, primary, target, tokens)
    if normalized.startswith("use "):
        if "thermometer" in tokens:
            return ActionIntentFrame("measurement", verb, primary, target, tokens)
        return ActionIntentFrame("generic_tool_use", verb, primary, target, tokens)

    return ActionIntentFrame("unknown", verb, primary, target, tokens)


def is_repair_compatible(
    current_failure_type: str,
    current_frame: ActionIntentFrame,
    repair_frame: ActionIntentFrame,
    *,
    observation: str = "",
    context: str = "",
) -> bool:
    """Return whether a repair action is semantically plausible for a current failure."""
    if repair_frame.verb.isdigit() and current_failure_type == "ambiguity":
        return True
    if repair_frame.intent == "circuit_connection":
        return _is_circuit_context(context) and not _has_bad_connect_object(repair_frame)
    if repair_frame.verb == "connect":
        return False
    if repair_frame.intent in {"examine_info", "wait"}:
        return repair_frame.verb in _SAFE_FALLBACK_ACTIONS or current_frame.intent in {
            "examine_info",
            "wait",
        }

    obs = observation.lower()
    if (
        current_failure_type == "precondition_blocked"
        and "door is not open" in obs
        and repair_frame.intent == "door_operation"
    ):
        return True

    compatible_repairs = {
        "navigation": {"navigation", "door_operation", "examine_info"},
        "door_operation": {"door_operation", "navigation", "examine_info"},
        "object_pickup": {"object_pickup", "examine_info"},
        "object_transfer": {"object_transfer", "object_pickup", "container_use"},
        "container_use": {"container_use", "object_pickup", "examine_info"},
        "examine_info": {"examine_info", "object_pickup", "measurement"},
        "measurement": {"measurement", "object_pickup", "examine_info"},
        "heating": {"heating", "container_use", "examine_info", "wait"},
        "circuit_connection": {"circuit_connection", "examine_info"},
        "wait": {"wait", "examine_info"},
        "generic_tool_use": {"generic_tool_use", "examine_info"},
        "unknown": {"examine_info"},
    }
    allowed = compatible_repairs.get(current_frame.intent, {"examine_info"})
    if repair_frame.intent not in allowed:
        return False

    if current_failure_type == "syntax_or_parse" and current_frame.intent != "unknown":
        return repair_frame.intent == current_frame.intent or repair_frame.intent in allowed
    return True


def are_failure_intents_compatible(
    current_frame: ActionIntentFrame,
    memory_frame: ActionIntentFrame,
    *,
    current_failure_type: str,
    observation: str = "",
) -> bool:
    """Return whether two failed actions describe the same repair problem class."""
    if current_frame.intent == "unknown" or memory_frame.intent == "unknown":
        return True
    if current_frame.intent == memory_frame.intent:
        return True
    if (
        current_failure_type == "precondition_blocked"
        and "door is not open" in observation.lower()
        and {current_frame.intent, memory_frame.intent} <= {"navigation", "door_operation"}
    ):
        return True
    return False


def filter_valid_actions_for_repair(
    valid_actions: str | Iterable[str],
    current_frame: ActionIntentFrame,
    *,
    failure_type: str,
    context: str = "",
) -> list[str]:
    """Filter ScienceWorld valid actions to intent-relevant repair candidates."""
    actions = _action_items(valid_actions)
    if not actions:
        return []
    if failure_type == "ambiguity" and all(action.strip().isdigit() for action in actions):
        return actions

    filtered: list[str] = []
    for action in actions:
        frame = parse_action_intent(action, context=context)
        if action in _SAFE_FALLBACK_ACTIONS:
            filtered.append(action)
            continue
        if is_repair_compatible(
            failure_type,
            current_frame,
            frame,
            observation=context,
            context=context,
        ):
            filtered.append(action)

    return filtered or [
        action
        for action in actions
        if _normalize(action) in _SAFE_FALLBACK_ACTIONS
    ]


def _normalize(text: str) -> str:
    text = str(text).lower().strip().replace("→", "->")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


def _verb(action: str) -> str:
    for prefix in (
        "pick up",
        "put down",
        "go to",
        "look around",
        "focus on",
        "teleport to",
    ):
        if action == prefix or action.startswith(f"{prefix} "):
            return prefix
    return action.split(" ", 1)[0]


def _objects_for_action(action: str, verb: str) -> tuple[str, str]:
    remainder = action[len(verb):].strip() if action.startswith(verb) else action
    for separator in (" into ", " in ", " to ", " on ", " with "):
        if separator in remainder:
            left, right = remainder.split(separator, 1)
            return left.strip(), right.strip()
    return remainder.strip(), ""


def _is_circuit_context(context: str) -> bool:
    return bool(_tokens(context) & _CIRCUIT_TERMS)


def _has_bad_connect_object(frame: ActionIntentFrame) -> bool:
    return bool(frame.tokens & _BAD_CONNECT_OBJECTS)


def _action_items(valid_actions: str | Iterable[str]) -> list[str]:
    if isinstance(valid_actions, str):
        return [line.strip() for line in valid_actions.splitlines() if line.strip()]
    return [str(item).strip() for item in valid_actions if str(item).strip()]

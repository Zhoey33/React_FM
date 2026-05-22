"""Tests for lightweight ScienceWorld action intent parsing."""

from src.scienceworld_action_intent import (
    filter_valid_actions_for_repair,
    is_repair_compatible,
    parse_action_intent,
)


def test_parse_common_scienceworld_action_intents():
    assert parse_action_intent("go to living room").intent == "navigation"
    assert parse_action_intent("open door to hallway").intent == "door_operation"
    assert parse_action_intent("move unknown substance B to purple box").intent == "object_transfer"
    assert parse_action_intent("use thermometer on metal pot").intent == "measurement"
    assert parse_action_intent("connect red wire to battery").intent == "circuit_connection"


def test_connect_requires_circuit_context_for_repair_compatibility():
    current = parse_action_intent("go to kitchen")
    connect_room = parse_action_intent("connect agent to kitchen")

    assert not is_repair_compatible(
        "syntax_or_parse",
        current,
        connect_room,
        observation="No known action matches that input.",
        context="Your task is to melt tin. You are in the kitchen.",
    )

    connect_wire = parse_action_intent(
        "connect red wire to battery",
        context="The circuit has a red wire, black wire, and battery terminal.",
    )
    assert is_repair_compatible(
        "syntax_or_parse",
        parse_action_intent("connect wire"),
        connect_wire,
        observation="No known action matches that input.",
        context="The circuit has a red wire, black wire, and battery terminal.",
    )


def test_filter_valid_actions_keeps_intent_relevant_candidates():
    actions = [
        "connect agent to kitchen",
        "open door to hallway",
        "go to hallway",
        "move cup to drawer",
        "inventory",
    ]

    filtered = filter_valid_actions_for_repair(
        actions,
        parse_action_intent("go to hallway"),
        failure_type="precondition_blocked",
        context="The door is not open.",
    )

    assert "open door to hallway" in filtered
    assert "go to hallway" in filtered
    assert "inventory" in filtered
    assert "connect agent to kitchen" not in filtered
    assert "move cup to drawer" not in filtered

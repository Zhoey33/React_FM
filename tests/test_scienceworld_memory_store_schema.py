"""Tests for ScienceWorld memory provenance fields in the shared memory store."""

import json

import numpy as np

from src.memory import FailureMemoryStore


def test_memory_store_saves_and_loads_scienceworld_provenance_fields(tmp_path):
    store = FailureMemoryStore(scope="task_type")
    store._embed = lambda text: np.array([0.1, 0.2], dtype=np.float32)

    store.add(
        failure_action="go to kitchen",
        failure_observation="The door is not open.",
        solution_action="open door to kitchen -> go to kitchen",
        task_type="melt",
        env_idx=7,
        repair_strategy="Open blocked doors before moving.",
        repair_tactic="Open the specific door, then retry movement.",
        repair_action="open door to kitchen -> go to kitchen",
        failure_step=3,
        failure_type="precondition_blocked",
        detector_source="rule",
        score_before_action=0.0,
        score_after_action=0.0,
        score_delta=0.0,
        source_episode_success=False,
        source_episode_score=35.0,
        confidence_score=0.88,
    )

    memory_path = tmp_path / "memory.json"
    store.save(str(memory_path))

    payload = json.loads(memory_path.read_text(encoding="utf-8"))
    entry = payload["buckets"]["melt"][0]
    assert entry["failure_step"] == 3
    assert entry["failure_type"] == "precondition_blocked"
    assert entry["detector_source"] == "rule"
    assert entry["confidence_score"] == 0.88
    assert "question_text" not in entry

    loaded = FailureMemoryStore(scope="task_type")
    loaded.load(str(memory_path))
    loaded_entry = loaded.get_all(task_type="melt")[0]

    assert loaded_entry.failure_step == 3
    assert loaded_entry.failure_type == "precondition_blocked"
    assert loaded_entry.detector_source == "rule"
    assert loaded_entry.source_episode_success is False
    assert loaded_entry.source_episode_score == 35.0
    assert loaded_entry.confidence_score == 0.88

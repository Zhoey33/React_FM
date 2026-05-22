"""Tests for ScienceWorld memory provenance and retrieval observability."""

import json

import numpy as np

from src.memory import FailureMemoryStore, RetrievalResult


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


def test_memory_retrieve_return_scores_tracks_top_k_and_candidate_count():
    store = FailureMemoryStore(scope="task_type", retrieval_mode="hybrid", top_k=3)
    store._embed = lambda text: np.array([1.0, 0.0], dtype=np.float32)

    store.add("go to kitchen", "The door is not open.", "open door to kitchen", task_type="melt")
    store.add("pick up pot", "You are not near the pot.", "go to kitchen", task_type="melt")

    result = store.retrieve(
        query_action="go to kitchen",
        query_observation="The door is not open.",
        task_type="melt",
        top_k=1,
        return_scores=True,
    )

    assert isinstance(result, RetrievalResult)
    assert len(result.entries) == 1
    assert len(result.rrf_scores) == 1
    assert result.candidate_count == 2
    assert result.rrf_scores[0] > 0


def test_memory_retrieve_candidate_k_can_return_more_than_final_top_k():
    store = FailureMemoryStore(scope="task_type", retrieval_mode="hybrid", top_k=1)
    store._embed = lambda text: np.array([1.0, 0.0], dtype=np.float32)

    for idx in range(3):
        store.add(
            f"go to room {idx}",
            "The door is not open.",
            f"open door to room {idx}",
            task_type="melt",
            failure_type="precondition_blocked",
        )

    result = store.retrieve(
        query_action="go to kitchen",
        query_observation="The door is not open.",
        query_failure_type="precondition_blocked",
        task_type="melt",
        top_k=1,
        candidate_k=3,
        return_scores=True,
    )

    assert isinstance(result, RetrievalResult)
    assert len(result.entries) == 3
    assert len(result.rrf_scores) == 3
    assert result.candidate_count == 3


def test_memory_retrieval_text_includes_failure_type():
    text = FailureMemoryStore._build_query_text(
        "go to kitchen",
        "The door is not open.",
        failure_type="precondition_blocked",
    )

    assert "precondition_blocked" in text
    assert "go to kitchen" in text


def test_memory_retrieve_hybrid_min_score_filters_low_rrf_results():
    store = FailureMemoryStore(scope="task_type", retrieval_mode="hybrid", top_k=3, min_score=1.0)
    store._embed = lambda text: np.array([1.0, 0.0], dtype=np.float32)

    store.add("go to kitchen", "The door is not open.", "open door to kitchen", task_type="melt")

    result = store.retrieve(
        query_action="go to kitchen",
        query_observation="The door is not open.",
        task_type="melt",
        return_scores=True,
    )

    assert result.entries == []
    assert result.rrf_scores == []
    assert result.candidate_count == 1

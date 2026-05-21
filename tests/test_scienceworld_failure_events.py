"""Tests for deriving ScienceWorld failure-event JSONL rows from episodes."""

from src.scienceworld_failure_events import build_failure_events


def test_build_failure_events_records_recovery_window():
    episode = {
        "env_idx": 7,
        "task_type": "melt",
        "variation_idx": 3,
        "steps": [
            {"step": 0, "action": "look around", "score_after_action": 10.0},
            {
                "step": 1,
                "action": "move to workshop",
                "observation": "You move to the workshop.",
                "failure_detected": True,
                "failure_type": "implicit_no_progress",
                "memory_retrieved": 1,
                "retrieved_memory_ids": [17],
                "retrieved_memory_scores": [0.0325],
                "retrieval_mode": "hybrid",
                "retrieval_top_k": 1,
                "retrieval_min_score": 0.0,
                "retrieval_candidate_count": 6,
                "injected_memory_text": "go to kitchen first",
                "score_before_action": 10.0,
                "score_after_action": 10.0,
            },
            {"step": 2, "action": "go to kitchen", "score_after_action": 10.0},
            {"step": 3, "action": "pick up metal pot", "score_after_action": 18.0},
            {"step": 4, "action": "activate stove", "score_after_action": 16.0},
        ],
    }

    events = build_failure_events(episode, memory_mode="in_loop")

    assert events == [
        {
            "episode_id": "sw_007",
            "env_idx": 7,
            "task_type": "melt",
            "variation_idx": 3,
            "step": 1,
            "failure_type": "implicit_no_progress",
            "failed_action": "move to workshop",
            "failure_observation": "You move to the workshop.",
            "memory_mode": "in_loop",
            "retrieval_attempted": True,
            "retrieval_hit": True,
            "retrieved_memory_ids": [17],
            "retrieved_memory_scores": [0.0325],
            "retrieval_mode": "hybrid",
            "retrieval_top_k": 1,
            "retrieval_min_score": 0.0,
            "retrieval_candidate_count": 6,
            "injected_memory_text": "go to kitchen first",
            "next_action": "go to kitchen",
            "score_before_failure": 10.0,
            "score_after_1_step": 10.0,
            "score_after_2_steps": 18.0,
            "score_after_3_steps": 16.0,
            "recovered_within_1_step": False,
            "recovered_within_3_steps": True,
        }
    ]

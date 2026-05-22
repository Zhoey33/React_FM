"""Tests for ScienceWorld result summaries and protocol metadata."""

from src.scienceworld_reporting import build_protocol_metadata, compute_summary


def test_summary_reports_only_raw_score_as_primary_score():
    results = [
        {
            "task_type": "boil",
            "success": True,
            "score": 100.0,
            "total_steps": 4,
            "total_tokens": 10,
            "agent_tokens": 7,
            "judge_tokens": 1,
            "extractor_tokens": 2,
        },
        {
            "task_type": "boil",
            "success": False,
            "score": -50.0,
            "total_steps": 8,
            "total_tokens": 20,
            "agent_tokens": 20,
        },
    ]
    protocol = build_protocol_metadata(
        split="test",
        tasks=["boil"],
        max_variations=10,
        step_limit=100,
        test_time_writable=False,
    )

    summary = compute_summary(
        results,
        mode="react_baseline",
        memory_stats={},
        protocol=protocol,
    )

    assert summary["avg_raw_score"] == 25.0
    assert summary["avg_score"] == 25.0
    assert "avg_normalized_score" not in summary
    assert "avg_clamped_score" not in summary
    assert summary["avg_steps_per_episode"] == 6.0
    assert summary["agent_tokens"] == 27
    assert summary["judge_tokens"] == 1
    assert summary["extractor_tokens"] == 2
    assert summary["protocol"]["split"] == "test"
    assert summary["protocol"]["tasks"] == ["boil"]
    assert summary["protocol"]["test_time_writable"] is False


def test_summary_distinguishes_candidate_hits_from_memory_injections():
    results = [
        {
            "task_type": "melt",
            "success": False,
            "score": 10.0,
            "total_steps": 3,
            "total_tokens": 0,
            "steps": [
                {
                    "retrieval_attempted": True,
                    "retrieval_candidate_count": 2,
                    "memory_retrieved": 0,
                },
                {
                    "retrieval_attempted": True,
                    "retrieval_candidate_count": 1,
                    "memory_retrieved": 1,
                },
                {
                    "retrieval_attempted": False,
                    "memory_retrieved": 0,
                },
            ],
        }
    ]

    summary = compute_summary(
        results,
        mode="react_fm",
        memory_stats={
            "total_entries": 3,
            "total_retrievals": 2,
            "total_hits": 2,
        },
    )

    memory_stats = summary["memory_stats"]
    assert memory_stats["candidate_retrievals"] == 2
    assert memory_stats["candidate_hits"] == 2
    assert memory_stats["in_loop_retrieval_attempts"] == 2
    assert memory_stats["injection_hits"] == 1


def test_summary_uses_step_records_for_current_run_retrieval_observability():
    results = [
        {
            "task_type": "melt",
            "success": False,
            "score": 10.0,
            "total_steps": 2,
            "total_tokens": 0,
            "steps": [
                {
                    "retrieval_attempted": True,
                    "retrieval_candidate_count": 0,
                    "memory_retrieved": 0,
                    "memory_injected": False,
                },
                {
                    "retrieval_attempted": True,
                    "retrieval_candidate_count": 1,
                    "memory_retrieved": 1,
                    "memory_injected": True,
                },
            ],
        }
    ]

    summary = compute_summary(
        results,
        mode="react_fm",
        memory_stats={
            "total_entries": 99,
            "total_retrievals": 50,
            "total_hits": 40,
        },
    )

    memory_stats = summary["memory_stats"]
    assert memory_stats["candidate_retrievals"] == 2
    assert memory_stats["candidate_hits"] == 1
    assert memory_stats["injection_hits"] == 1
    assert memory_stats["store_total_retrievals"] == 50
    assert memory_stats["store_total_hits"] == 40


def test_summary_reports_judge_call_and_token_breakdown():
    results = [
        {
            "task_type": "melt",
            "success": False,
            "score": 10.0,
            "total_steps": 3,
            "total_tokens": 110,
            "agent_tokens": 20,
            "judge_tokens": 80,
            "judge_detector_tokens": 50,
            "judge_repair_tokens": 30,
            "steps": [
                {"judge_call_type": "implicit_detector", "judge_cache_hit": False},
                {"judge_call_type": "cache", "judge_cache_hit": True},
                {"judge_call_type": "repair_advice", "judge_cache_hit": False},
            ],
        }
    ]

    summary = compute_summary(results, mode="react_fm", memory_stats={})

    assert summary["judge_tokens"] == 80
    assert summary["judge_detector_tokens"] == 50
    assert summary["judge_repair_tokens"] == 30
    assert summary["judge_calls"] == 2
    assert summary["judge_cache_hits"] == 1

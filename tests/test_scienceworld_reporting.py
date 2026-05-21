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

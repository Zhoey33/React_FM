"""Tests for ScienceWorld recovered@k aggregation from failure-event logs."""

import json

from analysis.scienceworld_recovery_stats import analyze_failure_event_files


def test_analyze_failure_event_files_reports_recovered_at_k_by_group(tmp_path):
    event_path = tmp_path / "failure_events.jsonl"
    events = [
        {
            "task_type": "melt",
            "failure_type": "implicit_no_progress",
            "memory_mode": "in_loop",
            "retrieval_hit": True,
            "score_before_failure": 10.0,
            "score_after_1_step": 12.0,
            "score_after_3_steps": 13.0,
            "recovered_within_1_step": True,
            "recovered_within_3_steps": True,
        },
        {
            "task_type": "melt",
            "failure_type": "implicit_no_progress",
            "memory_mode": "in_loop",
            "retrieval_hit": False,
            "score_before_failure": 5.0,
            "score_after_1_step": 5.0,
            "score_after_3_steps": 7.0,
            "recovered_within_1_step": False,
            "recovered_within_3_steps": True,
        },
        {
            "task_type": "boil",
            "failure_type": "invalid_action",
            "memory_mode": "baseline",
            "retrieval_hit": False,
            "score_before_failure": 20.0,
            "score_after_1_step": None,
            "score_after_3_steps": None,
            "recovered_within_1_step": False,
            "recovered_within_3_steps": False,
        },
        {
            "task_type": "boil",
            "failure_type": "implicit_no_progress",
            "memory_mode": "in_loop",
            "retrieval_hit": True,
            "score_before_failure": 20.0,
            "score_after_1_step": 20.0,
            "score_after_2_steps": 24.0,
            "score_after_3_steps": 19.0,
        },
    ]
    event_path.write_text(
        "\n".join(json.dumps(event) for event in events),
        encoding="utf-8",
    )

    summary = analyze_failure_event_files([event_path])

    assert summary["source_files"] == [str(event_path)]
    assert summary["overall"]["total_events"] == 4
    assert summary["overall"]["recovered_at_1"] == 1
    assert summary["overall"]["recovered_at_3"] == 3
    assert summary["overall"]["recovered_at_1_rate"] == 0.25
    assert summary["overall"]["recovered_at_3_rate"] == 0.75
    assert summary["overall"]["mean_score_delta_at_1"] == 0.6667
    assert summary["overall"]["mean_score_delta_at_3"] == 3.0

    assert summary["by_task_type"]["melt"]["total_events"] == 2
    assert summary["by_task_type"]["melt"]["recovered_at_3_rate"] == 1.0
    assert summary["by_failure_type"]["implicit_no_progress"]["recovered_at_1_rate"] == 0.3333
    assert summary["by_memory_mode"]["baseline"]["recovered_at_3_rate"] == 0.0
    assert summary["by_retrieval_hit"]["true"]["recovered_at_1_rate"] == 0.5
    assert summary["by_retrieval_hit"]["false"]["recovered_at_3_rate"] == 0.5


def test_analyze_failure_event_files_uses_boolean_recovery_when_window_scores_are_partial(
    tmp_path,
):
    event_path = tmp_path / "legacy_failure_events.jsonl"
    event_path.write_text(
        json.dumps(
            {
                "task_type": "melt",
                "failure_type": "implicit_no_progress",
                "memory_mode": "in_loop",
                "retrieval_hit": True,
                "score_before_failure": 10.0,
                "score_after_1_step": 10.0,
                "recovered_within_3_steps": True,
            }
        ),
        encoding="utf-8",
    )

    summary = analyze_failure_event_files([event_path])

    assert summary["overall"]["recovered_at_1"] == 0
    assert summary["overall"]["recovered_at_3"] == 1

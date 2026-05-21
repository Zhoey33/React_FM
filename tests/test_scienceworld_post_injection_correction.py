"""Tests for ScienceWorld post-injection correction annotation templates."""

import json

import pytest

from analysis.scienceworld_post_injection_correction import (
    make_post_injection_template,
    summarize_post_injection_files,
)


def _write_failure_events(path):
    events = [
        {
            "episode_id": "sw_001",
            "env_idx": 1,
            "task_type": "melt",
            "variation_idx": 3,
            "step": 4,
            "failure_type": "precondition_blocked",
            "failed_action": "go to workshop",
            "failure_observation": "The door is not open.",
            "memory_mode": "in_loop",
            "retrieval_attempted": True,
            "retrieval_hit": True,
            "retrieved_memory_ids": [12],
            "injected_memory_text": "Open the workshop door before moving.",
            "next_action": "open workshop door",
            "score_before_failure": 10.0,
            "score_after_1_step": 10.0,
            "score_after_2_steps": 14.0,
            "score_after_3_steps": 14.0,
            "recovered_within_1_step": False,
            "recovered_within_3_steps": True,
        },
        {
            "episode_id": "sw_002",
            "env_idx": 2,
            "task_type": "boil",
            "variation_idx": 4,
            "step": 8,
            "failure_type": "no_effect_or_other",
            "failed_action": "activate stove",
            "failure_observation": "Nothing happens.",
            "memory_mode": "in_loop",
            "retrieval_attempted": True,
            "retrieval_hit": "false",
            "retrieved_memory_ids": [],
            "injected_memory_text": "Put the pot on the stove first.",
            "next_action": "put pot on stove",
            "score_before_failure": 20.0,
            "score_after_1_step": 25.0,
            "score_after_2_steps": 25.0,
            "score_after_3_steps": 25.0,
            "recovered_within_1_step": True,
            "recovered_within_3_steps": True,
        },
        {
            "episode_id": "sw_003",
            "env_idx": 3,
            "task_type": "melt",
            "variation_idx": 5,
            "step": 10,
            "failure_type": "implicit_no_progress",
            "failed_action": "look around",
            "failure_observation": "You see the same room.",
            "memory_mode": "baseline",
            "retrieval_attempted": False,
            "retrieval_hit": False,
            "retrieved_memory_ids": [],
            "injected_memory_text": "",
            "next_action": "look around",
            "score_before_failure": 0.0,
            "score_after_1_step": 0.0,
            "score_after_2_steps": 0.0,
            "score_after_3_steps": 0.0,
            "recovered_within_1_step": False,
            "recovered_within_3_steps": False,
        },
    ]
    path.write_text(
        "\n".join(json.dumps(event) for event in events),
        encoding="utf-8",
    )


def test_make_post_injection_template_samples_injected_events_only(tmp_path):
    event_path = tmp_path / "failure_events.jsonl"
    _write_failure_events(event_path)

    first = make_post_injection_template([event_path], max_samples=10, seed=5)
    second = make_post_injection_template([event_path], max_samples=10, seed=5)

    assert first == second
    assert len(first) == 2
    assert {row["episode_id"] for row in first} == {"sw_001", "sw_002"}

    row = next(item for item in first if item["episode_id"] == "sw_001")
    assert set(row) == {
        "sample_id",
        "source_file",
        "episode_id",
        "env_idx",
        "task_type",
        "variation_idx",
        "step",
        "failure_type",
        "failed_action",
        "failure_observation",
        "retrieved_memory_ids",
        "injected_memory_text",
        "next_action",
        "memory_mode",
        "retrieval_attempted",
        "retrieval_hit",
        "score_before_failure",
        "score_after_1_step",
        "score_after_2_steps",
        "score_after_3_steps",
        "recovered_within_1_step",
        "recovered_within_3_steps",
        "gold_corrected_next_action",
        "gold_used_memory",
        "gold_injection_harmful",
        "annotation_notes",
    }
    assert row["sample_id"].startswith("sw_post_injection_")
    assert row["source_file"] == str(event_path)
    assert row["retrieval_hit"] is True
    assert row["gold_corrected_next_action"] is None
    assert row["gold_used_memory"] is None
    assert row["gold_injection_harmful"] is None
    assert row["annotation_notes"] == ""


def test_make_post_injection_template_respects_max_samples(tmp_path):
    event_path = tmp_path / "failure_events.jsonl"
    _write_failure_events(event_path)

    rows = make_post_injection_template([event_path], max_samples=1, seed=0)

    assert len(rows) == 1


def test_summarize_post_injection_files_reports_metrics_by_group(tmp_path):
    annotation_path = tmp_path / "post_injection_annotations.jsonl"
    rows = [
        {
            "task_type": "melt",
            "failure_type": "precondition_blocked",
            "memory_mode": "in_loop",
            "retrieval_hit": True,
            "score_before_failure": 10.0,
            "score_after_1_step": 10.0,
            "score_after_2_steps": 14.0,
            "score_after_3_steps": 14.0,
            "recovered_within_1_step": False,
            "recovered_within_3_steps": True,
            "gold_corrected_next_action": True,
            "gold_used_memory": "yes",
            "gold_injection_harmful": False,
        },
        {
            "task_type": "melt",
            "failure_type": "precondition_blocked",
            "memory_mode": "in_loop",
            "retrieval_hit": "true",
            "score_before_failure": 5.0,
            "score_after_1_step": 8.0,
            "score_after_3_steps": 8.0,
            "gold_corrected_next_action": "false",
            "gold_used_memory": "no",
            "gold_injection_harmful": "1",
        },
        {
            "task_type": "boil",
            "failure_type": "no_effect_or_other",
            "memory_mode": "in_loop",
            "retrieval_hit": "false",
            "score_before_failure": 20.0,
            "score_after_1_step": 20.0,
            "score_after_2_steps": 22.0,
            "score_after_3_steps": 19.0,
            "gold_corrected_next_action": True,
            "gold_used_memory": True,
            "gold_injection_harmful": False,
        },
    ]
    annotation_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    summary = summarize_post_injection_files([annotation_path])

    assert summary["source_files"] == [str(annotation_path)]
    assert summary["overall"]["total_samples"] == 3
    assert summary["overall"]["corrected_next_action_count"] == 2
    assert summary["overall"]["corrected_next_action_rate"] == 0.6667
    assert summary["overall"]["used_memory_count"] == 2
    assert summary["overall"]["used_memory_rate"] == 0.6667
    assert summary["overall"]["harmful_count"] == 1
    assert summary["overall"]["harmful_rate"] == 0.3333
    assert summary["overall"]["score_recovered_at_1_count"] == 1
    assert summary["overall"]["score_recovered_at_1_rate"] == 0.3333
    assert summary["overall"]["score_recovered_at_3_count"] == 3
    assert summary["overall"]["score_recovered_at_3_rate"] == 1.0
    assert summary["overall"]["mean_score_delta_at_1"] == 1.0
    assert summary["overall"]["mean_score_delta_at_3"] == 3.0
    assert summary["by_task_type"]["melt"]["harmful_rate"] == 0.5
    assert summary["by_failure_type"]["no_effect_or_other"]["corrected_next_action_rate"] == 1.0
    assert summary["by_memory_mode"]["in_loop"]["total_samples"] == 3
    assert summary["by_retrieval_hit"]["true"]["used_memory_rate"] == 0.5
    assert summary["by_gold_used_memory"]["true"]["corrected_next_action_rate"] == 1.0


def test_summarize_post_injection_files_requires_corrected_label_unless_allowed(tmp_path):
    annotation_path = tmp_path / "post_injection_annotations.jsonl"
    rows = [
        {"gold_corrected_next_action": True, "gold_used_memory": True},
        {"gold_corrected_next_action": None, "gold_used_memory": False},
    ]
    annotation_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing gold_corrected_next_action"):
        summarize_post_injection_files([annotation_path])

    summary = summarize_post_injection_files([annotation_path], allow_unlabeled=True)

    assert summary["overall"]["total_samples"] == 1
    assert summary["skipped_unlabeled"] == 1

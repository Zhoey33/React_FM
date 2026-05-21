"""Tests for ScienceWorld memory generation quality annotation templates."""

import json

import pytest

from analysis.scienceworld_memory_quality import (
    make_memory_quality_template,
    summarize_memory_quality_files,
)


def _write_memory_store(path):
    payload = {
        "next_id": 4,
        "scope": "task_type",
        "buckets": {
            "melt": [
                {
                    "memory_id": 1,
                    "failure_action": "go to kitchen",
                    "failure_observation": "The door is not open.",
                    "solution_action": "open door to kitchen -> go to kitchen",
                    "task_type": "melt",
                    "env_idx": 7,
                    "created_at": "2026-04-01T00:00:00",
                    "question_text": "What precondition is missing?",
                    "repair_strategy": "Open blocked doors before moving.",
                    "repair_tactic": "Open the specific door, then retry movement.",
                    "repair_action": "open door to kitchen -> go to kitchen",
                    "embedding": [0.1, 0.2],
                },
                {
                    "memory_id": 2,
                    "failure_action": "pour water into pot",
                    "failure_observation": "No known action matches that input.",
                    "solution_action": "pour cup into pot",
                    "task_type": "melt",
                    "env_idx": 8,
                    "embedding": [0.3, 0.4],
                },
            ],
            "boil": [
                {
                    "memory_id": 3,
                    "failure_action": "activate stove",
                    "failure_observation": "Nothing happens.",
                    "solution_action": "put pot on stove -> activate stove",
                    "task_type": "boil",
                    "env_idx": 9,
                    "repair_action": "put pot on stove -> activate stove",
                }
            ],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_result_json(path):
    payload = {
        "episodes": [
            {"env_idx": 7, "success": True, "score": 100.0},
            {"env_idx": 8, "success": False, "score": -20.0},
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_make_memory_quality_template_reads_buckets_and_enriches_episode_fields(tmp_path):
    memory_path = tmp_path / "sw_memory.json"
    result_path = tmp_path / "sw_result.json"
    _write_memory_store(memory_path)
    _write_result_json(result_path)

    first = make_memory_quality_template(
        [memory_path],
        result_paths=[result_path],
        max_samples=3,
        seed=11,
    )
    second = make_memory_quality_template(
        [memory_path],
        result_paths=[result_path],
        max_samples=3,
        seed=11,
    )

    assert first == second
    assert len(first) == 3
    assert {row["task_type"] for row in first} == {"melt", "boil"}
    assert all("embedding" not in row for row in first)

    row = next(item for item in first if item["memory_id"] == 1)
    assert set(row) == {
        "sample_id",
        "source_file",
        "memory_id",
        "bucket",
        "scope",
        "task_type",
        "env_idx",
        "created_at",
        "failure_action",
        "failure_observation",
        "solution_action",
        "repair_strategy",
        "repair_tactic",
        "repair_action",
        "question_text",
        "source_episode_success",
        "source_episode_score",
        "quality_label",
        "failure_action_accurate",
        "failure_observation_has_evidence",
        "repair_strategy_reasonable",
        "repair_action_executable",
        "overly_state_bound",
        "annotation_notes",
    }
    assert row["sample_id"].startswith("sw_memory_")
    assert row["source_file"] == str(memory_path)
    assert row["bucket"] == "melt"
    assert row["scope"] == "task_type"
    assert row["source_episode_success"] is True
    assert row["source_episode_score"] == 100.0
    assert row["quality_label"] == ""
    assert row["failure_action_accurate"] is None
    assert row["annotation_notes"] == ""


def test_make_memory_quality_template_reads_legacy_task_types_and_envs(tmp_path):
    task_types_path = tmp_path / "legacy_task_types.json"
    task_types_path.write_text(
        json.dumps(
            {
                "task_types": {
                    "melt": [
                        {
                            "memory_id": 10,
                            "failure_action": "go workshop",
                            "failure_observation": "No known action matches that input.",
                            "solution_action": "go to workshop",
                            "task_type": "melt",
                            "env_idx": 1,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    envs_path = tmp_path / "legacy_envs.json"
    envs_path.write_text(
        json.dumps(
            {
                "envs": {
                    "2": [
                        {
                            "memory_id": 11,
                            "failure_action": "activate stove",
                            "failure_observation": "Nothing happens.",
                            "solution_action": "put pot on stove",
                            "env_idx": 2,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    rows = make_memory_quality_template([task_types_path, envs_path], max_samples=10, seed=0)

    assert len(rows) == 2
    assert {row["memory_id"] for row in rows} == {10, 11}
    assert {row["scope"] for row in rows} == {"task_type", "env_idx"}


def test_summarize_memory_quality_files_reports_quality_metrics_by_group(tmp_path):
    annotation_path = tmp_path / "memory_quality.jsonl"
    rows = [
        {
            "task_type": "melt",
            "source_episode_success": True,
            "quality_label": "high_quality",
            "failure_action_accurate": True,
            "failure_observation_has_evidence": True,
            "repair_strategy_reasonable": True,
            "repair_action_executable": True,
            "overly_state_bound": False,
        },
        {
            "task_type": "melt",
            "source_episode_success": "false",
            "quality_label": "usable_but_weak",
            "failure_action_accurate": "yes",
            "failure_observation_has_evidence": "true",
            "repair_strategy_reasonable": "no",
            "repair_action_executable": "yes",
            "overly_state_bound": "1",
        },
        {
            "task_type": "boil",
            "source_episode_success": None,
            "quality_label": "invalid",
            "failure_action_accurate": False,
            "failure_observation_has_evidence": True,
            "repair_strategy_reasonable": False,
            "repair_action_executable": False,
            "overly_state_bound": False,
        },
        {
            "task_type": "boil",
            "quality_label": "duplicate_or_redundant",
            "failure_action_accurate": True,
            "failure_observation_has_evidence": False,
            "repair_strategy_reasonable": True,
            "repair_action_executable": True,
            "overly_state_bound": False,
        },
    ]
    annotation_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    summary = summarize_memory_quality_files([annotation_path])

    assert summary["source_files"] == [str(annotation_path)]
    assert summary["overall"]["total_memories"] == 4
    assert summary["overall"]["high_quality_count"] == 1
    assert summary["overall"]["usable_but_weak_count"] == 1
    assert summary["overall"]["invalid_count"] == 1
    assert summary["overall"]["duplicate_or_redundant_count"] == 1
    assert summary["overall"]["usable_rate"] == 0.5
    assert summary["overall"]["invalid_rate"] == 0.25
    assert summary["overall"]["duplicate_rate"] == 0.25
    assert summary["overall"]["failure_action_accuracy"] == 0.75
    assert summary["overall"]["failure_observation_evidence_rate"] == 0.75
    assert summary["overall"]["repair_strategy_reasonable_rate"] == 0.5
    assert summary["overall"]["repair_action_executable_rate"] == 0.75
    assert summary["overall"]["overly_state_bound_rate"] == 0.25
    assert summary["by_task_type"]["melt"]["usable_rate"] == 1.0
    assert summary["by_quality_label"]["invalid"]["invalid_rate"] == 1.0
    assert summary["by_source_episode_success"]["false"]["usable_rate"] == 1.0


def test_summarize_memory_quality_files_requires_quality_label_unless_allowed(tmp_path):
    annotation_path = tmp_path / "memory_quality.jsonl"
    rows = [
        {"quality_label": "high_quality", "failure_action_accurate": True},
        {"quality_label": "", "failure_action_accurate": False},
    ]
    annotation_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing quality_label"):
        summarize_memory_quality_files([annotation_path])

    summary = summarize_memory_quality_files([annotation_path], allow_unlabeled=True)

    assert summary["overall"]["total_memories"] == 1
    assert summary["skipped_unlabeled"] == 1

"""Tests for ScienceWorld detector quality annotation templates and metrics."""

import json

import pytest

from analysis.scienceworld_detector_quality import (
    make_annotation_template,
    summarize_annotation_files,
)


def _write_result_json(path):
    payload = {
        "episodes": [
            {
                "env_idx": 1,
                "task_type": "melt",
                "variation_idx": 3,
                "steps": [
                    {
                        "step": 0,
                        "action": "look around",
                        "observation": "You see a kitchen.",
                        "score_before_action": 0.0,
                        "score_after_action": 0.0,
                        "failure_detected": False,
                    },
                    {
                        "step": 1,
                        "action": "move to workshop",
                        "observation": "The door is not open.",
                        "score_before_action": 0.0,
                        "score_after_action": 0.0,
                        "failure_detected": True,
                        "failure_type": "precondition_blocked",
                        "detector_source": "rule",
                        "failure_reason": "Observation contains failure indicator.",
                        "failure_confidence": None,
                    },
                    {
                        "step": 2,
                        "action": "think: I should open the door",
                        "observation": "OK.",
                        "is_think": True,
                        "failure_detected": True,
                        "failure_type": "unproductive",
                    },
                ],
            },
            {
                "env_idx": 2,
                "task_type": "boil",
                "variation_idx": 4,
                "steps": [
                    {
                        "step": 0,
                        "action": "pick up pot",
                        "observation": "You pick up the pot.",
                        "score_before_action": 5.0,
                        "score_after_action": 8.0,
                        "failure_detected": False,
                    },
                    {
                        "step": 1,
                        "action": "activate stove",
                        "observation": "Nothing happens.",
                        "score_before_action": 8.0,
                        "score_after_action": 8.0,
                        "failure_detected": True,
                        "failure_type": "no_effect_or_other",
                        "detector_source": "judge",
                        "failure_reason": "No new progress.",
                        "failure_confidence": 0.82,
                    },
                ],
            },
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_make_annotation_template_balances_predictions_and_skips_think_steps(tmp_path):
    result_path = tmp_path / "sw_result.json"
    _write_result_json(result_path)

    first = make_annotation_template(
        [result_path],
        max_samples=4,
        positive_ratio=0.5,
        seed=7,
    )
    second = make_annotation_template(
        [result_path],
        max_samples=4,
        positive_ratio=0.5,
        seed=7,
    )

    assert first == second
    assert len(first) == 4
    assert sum(1 for row in first if row["detector_prediction"]) == 2
    assert sum(1 for row in first if not row["detector_prediction"]) == 2
    assert all("think:" not in row["action"] for row in first)

    row = first[0]
    assert set(row) == {
        "sample_id",
        "source_file",
        "episode_id",
        "env_idx",
        "task_type",
        "variation_idx",
        "step",
        "action",
        "observation",
        "score_before_action",
        "score_after_action",
        "detector_prediction",
        "detector_failure_type",
        "detector_source",
        "failure_reason",
        "failure_confidence",
        "gold_is_failure",
        "gold_failure_type",
        "gold_needs_repair",
        "annotation_notes",
    }
    assert row["source_file"] == str(result_path)
    assert row["sample_id"].startswith("sw_detector_")
    assert row["gold_is_failure"] is None
    assert row["gold_failure_type"] == ""
    assert row["gold_needs_repair"] is None
    assert row["annotation_notes"] == ""


def test_summarize_annotation_files_reports_detector_quality_metrics(tmp_path):
    annotation_path = tmp_path / "annotations.jsonl"
    rows = [
        {
            "task_type": "melt",
            "detector_prediction": True,
            "detector_failure_type": "precondition_blocked",
            "detector_source": "rule",
            "gold_is_failure": True,
            "gold_failure_type": "precondition_blocked",
        },
        {
            "task_type": "melt",
            "detector_prediction": True,
            "detector_failure_type": "no_effect_or_other",
            "detector_source": "judge",
            "gold_is_failure": False,
            "gold_failure_type": "",
        },
        {
            "task_type": "boil",
            "detector_prediction": False,
            "detector_failure_type": "",
            "detector_source": "",
            "gold_is_failure": True,
            "gold_failure_type": "wrong_location",
        },
        {
            "task_type": "boil",
            "detector_prediction": "false",
            "detector_failure_type": "",
            "detector_source": "",
            "gold_is_failure": False,
            "gold_failure_type": "",
        },
        {
            "task_type": "boil",
            "detector_prediction": True,
            "detector_failure_type": "syntax_or_parse",
            "detector_source": "rule",
            "gold_is_failure": True,
            "gold_failure_type": "wrong_location",
        },
    ]
    annotation_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    summary = summarize_annotation_files([annotation_path])

    assert summary["source_files"] == [str(annotation_path)]
    assert summary["overall"] == {
        "total_samples": 5,
        "tp": 2,
        "fp": 1,
        "fn": 1,
        "tn": 1,
        "precision": 0.6667,
        "recall": 0.6667,
        "f1": 0.6667,
        "type_accuracy": 0.5,
        "false_positive_rate": 0.5,
        "false_negative_rate": 0.3333,
    }
    assert summary["by_task_type"]["melt"]["precision"] == 0.5
    assert summary["by_task_type"]["boil"]["recall"] == 0.5
    assert summary["by_detector_source"]["rule"]["precision"] == 1.0
    assert summary["by_detector_source"]["judge"]["precision"] == 0.0
    assert summary["by_detector_failure_type"]["precondition_blocked"]["type_accuracy"] == 1.0
    assert summary["by_gold_failure_type"]["wrong_location"]["recall"] == 0.5


def test_summarize_annotation_files_requires_gold_labels_unless_allowed(tmp_path):
    annotation_path = tmp_path / "annotations.jsonl"
    rows = [
        {"detector_prediction": True, "gold_is_failure": True},
        {"detector_prediction": False, "gold_is_failure": None},
    ]
    annotation_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing gold_is_failure"):
        summarize_annotation_files([annotation_path])

    summary = summarize_annotation_files([annotation_path], allow_unlabeled=True)

    assert summary["overall"]["total_samples"] == 1
    assert summary["skipped_unlabeled"] == 1

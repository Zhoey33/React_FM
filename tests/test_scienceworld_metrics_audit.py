"""Tests for recomputing ScienceWorld metrics from saved result JSON."""

import json

from analysis.scienceworld_metrics_audit import recompute_result_file


def test_recompute_result_file_ignores_stale_summary_avg_score(tmp_path):
    result_path = tmp_path / "sw_result.json"
    result_path.write_text(
        json.dumps(
            {
                "summary": {"avg_score": 0.99},
                "episodes": [
                    {
                        "task_type": "boil",
                        "success": True,
                        "score": 100,
                        "total_steps": 5,
                        "total_tokens": 11,
                    },
                    {
                        "task_type": "boil",
                        "success": False,
                        "score": -100,
                        "total_steps": 7,
                        "total_tokens": 13,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    recomputed = recompute_result_file(result_path)

    assert recomputed["source_summary_avg_score"] == 0.99
    assert recomputed["avg_raw_score"] == 0.0
    assert recomputed["avg_score"] == 0.0
    assert "avg_normalized_score" not in recomputed
    assert "avg_clamped_score" not in recomputed
    assert recomputed["total_envs"] == 2

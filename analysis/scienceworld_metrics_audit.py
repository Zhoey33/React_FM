"""Recompute ScienceWorld metrics from episode JSON instead of trusting summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scienceworld_reporting import compute_summary


def recompute_result_file(path: str | Path) -> dict[str, Any]:
    """Load a ScienceWorld result JSON and recompute metrics from episodes."""
    result_path = Path(path)
    with result_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    episodes = payload.get("episodes", [])
    source_summary = payload.get("summary", {})
    recomputed = compute_summary(
        episodes,
        mode=source_summary.get("mode", "unknown"),
        memory_stats=source_summary.get("memory_stats"),
        insight_stats=source_summary.get("insight_stats"),
        protocol=source_summary.get("protocol"),
    )
    recomputed["source_file"] = str(result_path)
    recomputed["source_summary_avg_score"] = source_summary.get("avg_score")
    recomputed["source_summary_avg_raw_score"] = source_summary.get("avg_raw_score")
    recomputed["score_mismatch"] = (
        source_summary.get("avg_score") is not None
        and source_summary.get("avg_score") != recomputed["avg_score"]
    )
    return recomputed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute ScienceWorld raw score metrics from result JSON."
    )
    parser.add_argument("result_json", nargs="+", help="ScienceWorld result JSON file(s)")
    parser.add_argument("--indent", type=int, default=2, help="JSON output indentation")
    args = parser.parse_args()

    results = [recompute_result_file(path) for path in args.result_json]
    output: Any = results[0] if len(results) == 1 else results
    print(json.dumps(output, indent=args.indent, ensure_ascii=False))


if __name__ == "__main__":
    main()

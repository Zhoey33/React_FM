"""Sample a balanced checkpoint subset for branched rollouts.

Default strategy for ScienceWorld:
- keep only midband checkpoints with score in [1, 99]
- sample up to N checkpoints per task_type
- prefer scores in [20, 79], then [1, 19], then [80, 99]
- preserve original order within each priority bucket

Optional failure-aware strategy:
- within each task, reserve up to K slots for non-syntax failures
- then fill the remaining slots with syntax_or_parse checkpoints

Usage:
    python experiments/gate/sample_checkpoints.py \
        --input checkpoints/scienceworld_core_train_checkpoints_deduped.json \
        --output checkpoints/scienceworld_core_train_checkpoints_sampled.json \
        --max-per-task 8
"""

import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Balanced checkpoint sampling")
    parser.add_argument("--input", required=True, help="Input deduplicated checkpoint JSON")
    parser.add_argument("--output", required=True, help="Output sampled checkpoint JSON")
    parser.add_argument("--max-per-task", type=int, default=8, help="Max sampled checkpoints per task type")
    parser.add_argument(
        "--max-per-variation",
        type=int,
        default=None,
        help="Optional max sampled checkpoints per (task_type, variation_idx)",
    )
    parser.add_argument("--min-score", type=float, default=1.0, help="Minimum score to keep")
    parser.add_argument("--max-score", type=float, default=99.0, help="Maximum score to keep")
    parser.add_argument(
        "--failure-aware",
        action="store_true",
        help="Prefer keeping non-syntax failures within each task before filling with syntax_or_parse",
    )
    parser.add_argument(
        "--max-non-syntax-per-task",
        type=int,
        default=4,
        help="When --failure-aware is set, keep up to this many non-syntax checkpoints per task",
    )
    return parser.parse_args()


def score_bucket(score: float) -> int:
    """Priority bucket: lower is preferred."""
    if 20 <= score <= 79:
        return 0
    if 1 <= score <= 19:
        return 1
    if 80 <= score <= 99:
        return 2
    return 3


def variation_idx(cp: dict) -> int | None:
    env_state = cp.get("env_state") or {}
    return env_state.get("variation_idx")


def take_with_variation_cap(
    candidates: list[dict],
    limit: int,
    max_per_variation: int | None,
    per_variation: Counter | None = None,
) -> list[dict]:
    if limit <= 0:
        return []
    if max_per_variation is None:
        return candidates[:limit]

    chosen = []
    if per_variation is None:
        per_variation = Counter()

    for cp in candidates:
        var_idx = variation_idx(cp)
        if var_idx is not None and per_variation[var_idx] >= max_per_variation:
            continue
        chosen.append(cp)
        if var_idx is not None:
            per_variation[var_idx] += 1
        if len(chosen) >= limit:
            return chosen

    return chosen


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    with open(input_path) as f:
        data = json.load(f)
    checkpoints = data.get("checkpoints", [])

    filtered = [
        cp for cp in checkpoints
        if args.min_score <= float(cp.get("score_at_checkpoint", 0.0) or 0.0) <= args.max_score
    ]

    by_task = defaultdict(list)
    for cp in filtered:
        by_task[cp.get("task_type", "")].append(cp)

    sampled = []
    for task_type in sorted(by_task):
        candidates = by_task[task_type]
        if args.failure_aware:
            non_syntax = [cp for cp in candidates if cp.get("failure_type") != "syntax_or_parse"]
            syntax = [cp for cp in candidates if cp.get("failure_type") == "syntax_or_parse"]
            sort_key = lambda cp: (
                score_bucket(float(cp.get("score_at_checkpoint", 0.0) or 0.0)),
                cp.get("step_idx", 0),
                cp.get("checkpoint_id", ""),
            )
            non_syntax.sort(key=sort_key)
            syntax.sort(key=sort_key)
            per_variation = Counter()
            chosen = take_with_variation_cap(
                non_syntax,
                args.max_non_syntax_per_task,
                args.max_per_variation,
                per_variation=per_variation,
            )
            remaining = max(args.max_per_task - len(chosen), 0)
            if remaining:
                chosen.extend(
                    take_with_variation_cap(
                        syntax,
                        remaining,
                        args.max_per_variation,
                        per_variation=per_variation,
                    )
                )
            sampled.extend(chosen[: args.max_per_task])
        else:
            candidates.sort(
                key=lambda cp: (
                    score_bucket(float(cp.get("score_at_checkpoint", 0.0) or 0.0)),
                    cp.get("step_idx", 0),
                    cp.get("checkpoint_id", ""),
                )
            )
            sampled.extend(
                take_with_variation_cap(
                    candidates,
                    args.max_per_task,
                    args.max_per_variation,
                )
            )

    output = {
        "next_id": data.get("next_id", 0),
        "sampling_config": {
            "source": str(input_path),
            "min_score": args.min_score,
            "max_score": args.max_score,
            "max_per_task": args.max_per_task,
            "max_per_variation": args.max_per_variation,
            "priority": ["20-79", "1-19", "80-99"],
            "failure_aware": args.failure_aware,
            "max_non_syntax_per_task": args.max_non_syntax_per_task if args.failure_aware else None,
        },
        "checkpoints": sampled,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    task_counts = Counter(cp.get("task_type", "") for cp in sampled)
    print(f"Filtered midband checkpoints: {len(filtered)}")
    print(f"Sampled checkpoints: {len(sampled)}")
    print(f"Task distribution: {dict(task_counts)}")


if __name__ == "__main__":
    main()

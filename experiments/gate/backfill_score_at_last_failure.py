"""Backfill env_state.score_at_last_failure for checkpoint JSON files.

This repairs older checkpoint collections that were generated before
ScienceWorld stored the previous failure score in env_state.

Usage:
    python experiments/gate/backfill_score_at_last_failure.py \
        --input checkpoints/scienceworld_core_train_checkpoints.json

By default this:
1. creates a .bak backup alongside the input file
2. rewrites the input file in place with the backfilled field
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Backfill score_at_last_failure in checkpoint JSON")
    parser.add_argument("--input", required=True, help="Checkpoint JSON to repair in place")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write a .bak backup before rewriting the input file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.input)
    with open(path) as f:
        data = json.load(f)
    original_data = json.loads(json.dumps(data))

    checkpoints = data.get("checkpoints", [])
    by_env = defaultdict(list)
    for cp in checkpoints:
        by_env[cp.get("env_idx", 0)].append(cp)

    updated = 0
    env_count = 0
    for env_idx, cps in by_env.items():
        env_count += 1
        cps.sort(key=lambda cp: (cp.get("step_idx", 0), cp.get("checkpoint_id", "")))
        previous_score = 0.0
        for cp in cps:
            env_state = cp.setdefault("env_state", {})
            env_state["score_at_last_failure"] = previous_score
            updated += 1
            previous_score = float(cp.get("score_at_checkpoint", 0.0) or 0.0)

    if not args.no_backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        with open(backup_path, "w") as f:
            json.dump(original_data, f, indent=2, ensure_ascii=False)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Backfilled {updated} checkpoints across {env_count} envs: {path}")
    if not args.no_backup:
        print(f"Backup written: {backup_path}")


if __name__ == "__main__":
    main()

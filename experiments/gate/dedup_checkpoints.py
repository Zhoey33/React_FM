"""Deduplicate checkpoint JSON files by failure signature.

Usage:
    python experiments/gate/dedup_checkpoints.py \
        --input checkpoints/scienceworld_core_train_checkpoints.json \
        --output checkpoints/scienceworld_core_train_checkpoints_deduped.json \
        --max-per-signature 3
"""

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.gate.checkpoint import CheckpointStore


def parse_args():
    parser = argparse.ArgumentParser(description="Deduplicate checkpoint store by failure signature")
    parser.add_argument("--input", required=True, help="Input checkpoint JSON")
    parser.add_argument("--output", required=True, help="Output deduplicated checkpoint JSON")
    parser.add_argument(
        "--max-per-signature",
        type=int,
        default=3,
        help="Keep at most this many checkpoints per failure signature",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    store = CheckpointStore()
    store.load(args.input)
    deduped = store.deduplicate(max_per_signature=args.max_per_signature)
    deduped.save(args.output)
    print(
        f"Deduplicated checkpoints: {len(store)} -> {len(deduped)} "
        f"(max_per_signature={args.max_per_signature})"
    )


if __name__ == "__main__":
    main()

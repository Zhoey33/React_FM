"""Train a two-stage logistic baseline."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gate.checkpoint import CheckpointStore
from src.gate.train_gate_two_stage_logistic import TwoStageLogisticGate
from src.log_utils import setup_logging
from experiments.gate.run_gate_training import (
    build_training_data,
    load_rollout_results,
    oracle_analysis,
)
from experiments.gate.run_gate_training_two_stage import _always_none_baseline

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train two-stage logistic baseline")
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--bypass-threshold", type=float, default=0.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--max-per-sig", type=int, default=3)
    parser.add_argument(
        "--progress-signal",
        choices=["env", "llm", "combined"],
        default="combined",
    )
    parser.add_argument("--combined-llm-weight", type=float, default=None)
    parser.add_argument("--intervene-threshold", type=float, default=0.5)
    parser.add_argument("--margin-eps", type=float, default=0.01)
    parser.add_argument("--run-name", default="gate_train_two_stage_logistic_")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(log_level="INFO", log_dir="logs", run_name=args.run_name.rstrip("_"))

    cp_store = CheckpointStore()
    cp_store.load(args.checkpoints)
    logger.info("Loaded %s checkpoints", len(cp_store))

    deduped = cp_store.deduplicate(max_per_signature=args.max_per_sig)
    rollout_data = load_rollout_results(args.rollouts)
    rollout_cp_count = sum(1 for cp_id in rollout_data if not cp_id.startswith("__"))
    logger.info("Loaded rollouts for %s checkpoints", rollout_cp_count)

    all_data = build_training_data(
        deduped.checkpoints,
        rollout_data,
        beta=args.beta,
        progress_signal=args.progress_signal,
        combined_llm_weight=args.combined_llm_weight,
    )
    logger.info("Built %s training examples (%s progress)", len(all_data), args.progress_signal)
    oracle_results = oracle_analysis(all_data)

    train_store, test_store = deduped.split(
        train_ratio=args.train_ratio,
        group_by="episode",
        stratify_by_task=True,
    )
    train_ids = {cp.checkpoint_id for cp in train_store.checkpoints}
    test_ids = {cp.checkpoint_id for cp in test_store.checkpoints}
    train_data = [d for d in all_data if d["checkpoint_id"] in train_ids]
    test_data = [d for d in all_data if d["checkpoint_id"] in test_ids]

    logger.info("Train: %s, Test: %s", len(train_data), len(test_data))

    gate = TwoStageLogisticGate(
        bypass_threshold=args.bypass_threshold,
        beta=args.beta,
        intervene_threshold=args.intervene_threshold,
        margin_eps=args.margin_eps,
    )
    train_stats = gate.train(train_data, test_data)
    baseline_metrics = _always_none_baseline(test_data)

    logger.info("\nTwo-Stage Logistic Results:")
    for key, value in train_stats.items():
        logger.info("  %s: %s", key, value)

    gate.save(args.output)

    analysis = {
        "oracle_analysis": oracle_results,
        "train_stats": train_stats,
        "always_none_test_baseline": baseline_metrics,
        "config": {
            "beta": args.beta,
            "bypass_threshold": args.bypass_threshold,
            "progress_signal": args.progress_signal,
            "combined_llm_weight": args.combined_llm_weight,
            "train_ratio": args.train_ratio,
            "margin_eps": args.margin_eps,
            "intervene_threshold": args.intervene_threshold,
            "n_checkpoints_raw": len(cp_store),
            "n_checkpoints_deduped": len(deduped),
            "n_train": len(train_data),
            "n_test": len(test_data),
        },
    }
    analysis_path = args.output + "_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    logger.info("Analysis saved: %s", analysis_path)


if __name__ == "__main__":
    main()

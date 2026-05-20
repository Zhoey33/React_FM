"""Train a two-stage intervention gate.

Stage A: none vs intervene
Stage B: question vs repair
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gate.checkpoint import CheckpointStore
from src.gate.train_gate_two_stage import TwoStageInterventionGate
from src.log_utils import setup_logging
from experiments.gate.run_gate_training import (
    build_training_data,
    load_rollout_results,
    oracle_analysis,
)

logger = logging.getLogger(__name__)


def _parse_threshold_candidates(raw: str) -> list[float]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = float(chunk)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {value}")
        values.append(round(value, 4))
    if not values:
        raise ValueError("At least one threshold candidate is required")
    return sorted(set(values))


def _always_none_baseline(data: list[dict]) -> dict:
    if not data:
        return {
            "n_test_checkpoints": 0,
            "intervene_threshold": 1.0,
            "two_stage_accuracy": 0.0,
            "binary_intervene_accuracy": 0.0,
            "intervene_precision": 0.0,
            "intervene_recall": 0.0,
            "question_vs_repair_accuracy_on_oracle_intervene": 0.0,
            "oracle_intervene_rate": 0.0,
            "predicted_intervene_rate": 0.0,
            "avg_selected_utility": 0.0,
            "avg_oracle_utility": 0.0,
            "avg_none_utility": 0.0,
            "avg_regret": 0.0,
            "avg_gain_over_none": 0.0,
            "gate_arm_distribution": {"none": 0, "question": 0, "repair": 0},
        }

    oracle_arms = [max(item["arm_utilities"], key=item["arm_utilities"].get) for item in data]
    selected_utils = [float(item["arm_utilities"]["none"]) for item in data]
    oracle_utils = [float(max(item["arm_utilities"].values())) for item in data]
    oracle_intervene = [arm != "none" for arm in oracle_arms]
    n = len(data)

    return {
        "n_test_checkpoints": n,
        "intervene_threshold": 1.0,
        "two_stage_accuracy": round(sum(arm == "none" for arm in oracle_arms) / n, 4),
        "binary_intervene_accuracy": round(sum(not flag for flag in oracle_intervene) / n, 4),
        "intervene_precision": 0.0,
        "intervene_recall": 0.0,
        "question_vs_repair_accuracy_on_oracle_intervene": 0.0,
        "oracle_intervene_rate": round(sum(oracle_intervene) / n, 4),
        "predicted_intervene_rate": 0.0,
        "avg_selected_utility": round(float(sum(selected_utils) / n), 4),
        "avg_oracle_utility": round(float(sum(oracle_utils) / n), 4),
        "avg_none_utility": round(float(sum(selected_utils) / n), 4),
        "avg_regret": round(float(sum(o - s for o, s in zip(oracle_utils, selected_utils)) / n), 4),
        "avg_gain_over_none": 0.0,
        "gate_arm_distribution": {"none": n, "question": 0, "repair": 0},
    }


def _pick_best_threshold(sweep_rows: list[dict], metric: str) -> dict:
    if metric == "avg_regret":
        return min(
            sweep_rows,
            key=lambda row: (
                row["avg_regret"],
                row["predicted_intervene_rate"],
                -row["avg_gain_over_none"],
            ),
        )
    return min(
        sweep_rows,
        key=lambda row: (
            -row[metric],
            row["avg_regret"],
            row["predicted_intervene_rate"],
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train two-stage intervention gate")
    parser.add_argument("--checkpoints", required=True, help="Checkpoint store JSON")
    parser.add_argument("--rollouts", required=True, help="Judged rollout JSON")
    parser.add_argument("--output", required=True, help="Output model path (without extension)")
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--bypass-threshold", type=float, default=0.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--max-per-sig", type=int, default=3)
    parser.add_argument(
        "--progress-signal",
        choices=["env", "llm", "combined"],
        default="combined",
    )
    parser.add_argument(
        "--combined-llm-weight",
        type=float,
        default=None,
        help="If set with --progress-signal combined, recompute combined progress offline",
    )
    parser.add_argument("--intervene-threshold", type=float, default=0.5)
    parser.add_argument("--margin-eps", type=float, default=0.01)
    parser.add_argument("--stage-a-positive-weight", type=float, default=1.0)
    parser.add_argument("--stage-a-negative-weight", type=float, default=1.0)
    parser.add_argument("--stage-a-margin-weight-alpha", type=float, default=0.0)
    parser.add_argument(
        "--threshold-metric",
        choices=["avg_gain_over_none", "avg_regret", "binary_intervene_accuracy", "two_stage_accuracy"],
        default="avg_gain_over_none",
    )
    parser.add_argument(
        "--threshold-candidates",
        default="0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95",
    )
    parser.add_argument("--run-name", default="gate_train_two_stage_")
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

    if len(all_data) < 10:
        logger.error("Too few training examples! Need at least 10 checkpoints.")
        return

    oracle_results = oracle_analysis(all_data)

    train_store, test_store = deduped.split(
        train_ratio=args.train_ratio,
        group_by="episode",
        stratify_by_task=True,
    )
    test_ids = {cp.checkpoint_id for cp in test_store.checkpoints}
    fit_store = train_store
    val_store = CheckpointStore()
    if args.val_ratio > 0.0:
        fit_store, val_store = train_store.split(
            train_ratio=1.0 - args.val_ratio,
            group_by="episode",
            stratify_by_task=True,
        )

    fit_ids = {cp.checkpoint_id for cp in fit_store.checkpoints}
    val_ids = {cp.checkpoint_id for cp in val_store.checkpoints}
    test_data = [d for d in all_data if d["checkpoint_id"] in test_ids]
    train_data = [d for d in all_data if d["checkpoint_id"] in fit_ids]
    val_data = [d for d in all_data if d["checkpoint_id"] in val_ids]

    logger.info("Train-fit: %s, Val: %s, Test: %s", len(train_data), len(val_data), len(test_data))

    gate = TwoStageInterventionGate(
        bypass_threshold=args.bypass_threshold,
        beta=args.beta,
        intervene_threshold=args.intervene_threshold,
        margin_eps=args.margin_eps,
        stage_a_positive_weight=args.stage_a_positive_weight,
        stage_a_negative_weight=args.stage_a_negative_weight,
        stage_a_margin_weight_alpha=args.stage_a_margin_weight_alpha,
    )
    train_stats = gate.train(train_data)

    threshold_candidates = _parse_threshold_candidates(args.threshold_candidates)
    threshold_sweep = []
    selected_threshold = args.intervene_threshold
    val_metrics = None
    if val_data:
        for threshold in threshold_candidates:
            metrics = gate.evaluate(val_data, intervene_threshold=threshold)
            threshold_sweep.append(metrics)
        best_row = _pick_best_threshold(threshold_sweep, args.threshold_metric)
        selected_threshold = float(best_row["intervene_threshold"])
        gate.intervene_threshold = selected_threshold
        val_metrics = gate.evaluate(val_data)
    else:
        gate.intervene_threshold = args.intervene_threshold

    test_metrics = gate.evaluate(test_data)
    baseline_metrics = _always_none_baseline(test_data)
    train_stats.update(test_metrics)
    train_stats["selected_threshold_metric"] = args.threshold_metric
    train_stats["selected_intervene_threshold"] = round(gate.intervene_threshold, 4)
    train_stats["n_val_checkpoints"] = len(val_data)
    train_stats["threshold_tuned_on_validation"] = bool(val_data)
    if val_metrics is not None:
        for key, value in val_metrics.items():
            train_stats[f"val_{key}"] = value

    logger.info("\nTwo-Stage Gate Training Results:")
    for key, value in train_stats.items():
        logger.info("  %s: %s", key, value)

    gate.save(args.output)

    analysis = {
        "oracle_analysis": oracle_results,
        "train_stats": train_stats,
        "feature_importance": gate.feature_importance(),
        "config": {
            "beta": args.beta,
            "bypass_threshold": args.bypass_threshold,
            "progress_signal": args.progress_signal,
            "combined_llm_weight": args.combined_llm_weight,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "margin_eps": args.margin_eps,
            "intervene_threshold": args.intervene_threshold,
            "selected_intervene_threshold": round(gate.intervene_threshold, 4),
            "threshold_metric": args.threshold_metric,
            "threshold_candidates": threshold_candidates,
            "stage_a_positive_weight": args.stage_a_positive_weight,
            "stage_a_negative_weight": args.stage_a_negative_weight,
            "stage_a_margin_weight_alpha": args.stage_a_margin_weight_alpha,
            "n_checkpoints_raw": len(cp_store),
            "n_checkpoints_deduped": len(deduped),
            "n_train": len(train_data),
            "n_val": len(val_data),
            "n_test": len(test_data),
        },
        "validation_threshold_sweep": threshold_sweep,
        "always_none_test_baseline": baseline_metrics,
    }
    analysis_path = args.output + "_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    logger.info("Analysis saved: %s", analysis_path)


if __name__ == "__main__":
    main()

"""Run repeated 80/20 clean validation across multiple outer seeds."""

import argparse
import json
import logging
import statistics
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.gate.run_gate_training_two_stage_regression_cleanval import run_cleanval
from src.log_utils import setup_logging

logger = logging.getLogger(__name__)


def _parse_int_candidates(raw: str) -> list[int]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("At least one seed is required")
    return values


def _mean(values: list[float]) -> float:
    return round(float(statistics.mean(values)), 4)


def _stdev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return round(float(statistics.stdev(values)), 4)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-seed clean validation for two-stage regression gate")
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--output-prefix", required=True, help="Per-seed model prefix base")
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--bypass-threshold", type=float, default=0.0)
    parser.add_argument("--max-per-sig", type=int, default=3)
    parser.add_argument(
        "--progress-signal",
        choices=["env", "llm", "combined"],
        default="combined",
    )
    parser.add_argument("--combined-llm-weight", type=float, default=0.4)
    parser.add_argument("--margin-eps", type=float, default=0.01)
    parser.add_argument("--categorical-encoding", choices=["int", "onehot"], default="int")
    parser.add_argument("--drop-features", default="")
    parser.add_argument("--outer-train-ratio", type=float, default=0.8)
    parser.add_argument("--inner-train-ratio", type=float, default=0.8)
    parser.add_argument("--outer-seeds", default="123,231,341,451,561")
    parser.add_argument("--inner-seed-base", type=int, default=456)
    parser.add_argument(
        "--selection-metric",
        choices=["avg_gain_over_none", "avg_regret", "binary_intervene_accuracy", "two_stage_accuracy"],
        default="avg_gain_over_none",
    )
    parser.add_argument("--stage-a-min-abs-margin-candidates", default="0.00,0.01,0.02,0.03,0.04,0.05")
    parser.add_argument("--intervene-margin-threshold-candidates", default="0.00,0.01,0.02,0.03")
    parser.add_argument("--repair-margin-threshold-candidates", default="0.00,0.02,0.04,0.06,0.08")
    parser.add_argument("--stage-a-negative-weight-candidates", default="1.0,1.25,1.5,2.0")
    parser.add_argument("--stage-a-positive-weight", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--run-name", default="gate_train_two_stage_regression_cleanval_multiseed_")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(log_level="INFO", log_dir="logs", run_name=args.run_name.rstrip("_"))

    outer_seeds = _parse_int_candidates(args.outer_seeds)
    seed_results = []

    for idx, outer_seed in enumerate(outer_seeds):
        inner_seed = args.inner_seed_base + idx
        output = f"{args.output_prefix}_seed{outer_seed}"
        run_args = Namespace(
            checkpoints=args.checkpoints,
            rollouts=args.rollouts,
            output=output,
            beta=args.beta,
            bypass_threshold=args.bypass_threshold,
            max_per_sig=args.max_per_sig,
            progress_signal=args.progress_signal,
            combined_llm_weight=args.combined_llm_weight,
            margin_eps=args.margin_eps,
            categorical_encoding=args.categorical_encoding,
            drop_features=args.drop_features,
            outer_train_ratio=args.outer_train_ratio,
            inner_train_ratio=args.inner_train_ratio,
            outer_seed=outer_seed,
            inner_seed=inner_seed,
            selection_metric=args.selection_metric,
            stage_a_min_abs_margin_candidates=args.stage_a_min_abs_margin_candidates,
            intervene_margin_threshold_candidates=args.intervene_margin_threshold_candidates,
            repair_margin_threshold_candidates=args.repair_margin_threshold_candidates,
            stage_a_negative_weight_candidates=args.stage_a_negative_weight_candidates,
            stage_a_positive_weight=args.stage_a_positive_weight,
            top_k=args.top_k,
            run_name=f"{args.run_name.rstrip('_')}_seed{outer_seed}_",
        )
        logger.info("Running seed %s (inner_seed=%s)", outer_seed, inner_seed)
        analysis = run_cleanval(run_args)
        test_metrics = analysis["test_metrics"]
        baseline = analysis["test_always_none_baseline"]
        seed_results.append(
            {
                "outer_seed": outer_seed,
                "inner_seed": inner_seed,
                "artifact_prefix": output,
                "selected_config": analysis["selected_config"],
                "n_train_checkpoints": test_metrics["n_train_checkpoints"],
                "n_test_checkpoints": test_metrics["n_test_checkpoints"],
                "avg_gain_over_none": test_metrics["avg_gain_over_none"],
                "avg_regret": test_metrics["avg_regret"],
                "two_stage_accuracy": test_metrics["two_stage_accuracy"],
                "binary_intervene_accuracy": test_metrics["binary_intervene_accuracy"],
                "intervene_precision": test_metrics["intervene_precision"],
                "intervene_recall": test_metrics["intervene_recall"],
                "baseline_avg_gain_over_none": baseline["avg_gain_over_none"],
                "baseline_avg_regret": baseline["avg_regret"],
            }
        )

    summary = {
        "protocol": "multi-seed repeated clean holdout for two-stage regression gate",
        "config": {
            "outer_train_ratio": args.outer_train_ratio,
            "inner_train_ratio": args.inner_train_ratio,
            "outer_seeds": outer_seeds,
            "inner_seed_base": args.inner_seed_base,
            "selection_metric": args.selection_metric,
            "progress_signal": args.progress_signal,
            "combined_llm_weight": args.combined_llm_weight,
            "beta": args.beta,
            "categorical_encoding": args.categorical_encoding,
            "drop_features": args.drop_features,
        },
        "per_seed": seed_results,
        "aggregate": {
            "n_seeds": len(seed_results),
            "positive_gain_seed_count": sum(r["avg_gain_over_none"] > 0 for r in seed_results),
            "avg_gain_over_none_mean": _mean([r["avg_gain_over_none"] for r in seed_results]),
            "avg_gain_over_none_std": _stdev([r["avg_gain_over_none"] for r in seed_results]),
            "avg_regret_mean": _mean([r["avg_regret"] for r in seed_results]),
            "avg_regret_std": _stdev([r["avg_regret"] for r in seed_results]),
            "two_stage_accuracy_mean": _mean([r["two_stage_accuracy"] for r in seed_results]),
            "binary_intervene_accuracy_mean": _mean(
                [r["binary_intervene_accuracy"] for r in seed_results]
            ),
            "intervene_precision_mean": _mean([r["intervene_precision"] for r in seed_results]),
            "intervene_recall_mean": _mean([r["intervene_recall"] for r in seed_results]),
        },
    }

    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved multi-seed summary to %s", summary_path)


if __name__ == "__main__":
    main()

"""Run clean train/val/test selection for the two-stage regression gate.

Protocol:
1. Outer grouped split -> trainval/test
2. Inner grouped split inside trainval -> fit/val
3. Search config on val only
4. Retrain best config on full trainval
5. Evaluate once on untouched outer test
"""

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.gate.run_gate_training import (
    build_training_data,
    load_rollout_results,
    oracle_analysis,
)
from experiments.gate.run_gate_training_two_stage import _always_none_baseline
from src.gate.checkpoint import CheckpointState, CheckpointStore
from src.gate.feature_transform import (
    FeatureTransformConfig,
    parse_drop_features,
    transform_training_data,
)
from src.gate.train_gate_two_stage_regression import TwoStageRegressionGate
from src.log_utils import setup_logging

logger = logging.getLogger(__name__)


def _parse_float_candidates(raw: str) -> list[float]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(round(float(chunk), 4))
    if not values:
        raise ValueError("At least one candidate value is required")
    return sorted(set(values))


def _pick_best_result(rows: list[dict], metric: str) -> dict:
    if metric == "avg_regret":
        return min(
            rows,
            key=lambda row: (
                row["avg_regret"],
                row["predicted_intervene_rate"],
                -row["avg_gain_over_none"],
            ),
        )
    return min(
        rows,
        key=lambda row: (
            -row[metric],
            row["avg_regret"],
            row["predicted_intervene_rate"],
        ),
    )


def _task_checkpoint_counts(checkpoints: list[CheckpointState]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cp in checkpoints:
        counts[cp.task_type] = counts.get(cp.task_type, 0) + 1
    return dict(sorted(counts.items()))


def parse_args():
    parser = argparse.ArgumentParser(description="Clean validation for two-stage regression gate")
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--output", required=True, help="Output model path without extension")
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
    parser.add_argument("--outer-seed", type=int, default=123)
    parser.add_argument("--inner-seed", type=int, default=456)
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
    parser.add_argument("--run-name", default="gate_train_two_stage_regression_cleanval_")
    return parser.parse_args()


def _filter_examples(all_data: list[dict], ids: set[str]) -> list[dict]:
    return [row for row in all_data if row["checkpoint_id"] in ids]


def run_cleanval(args) -> dict:
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
    feature_config = FeatureTransformConfig(
        categorical_encoding=args.categorical_encoding,
        drop_features=parse_drop_features(args.drop_features),
    )
    all_data, feature_names = transform_training_data(all_data, feature_config)
    if len(all_data) < 10:
        raise ValueError(f"Too few training examples: {len(all_data)}")
    logger.info("Built %s training examples (%s progress)", len(all_data), args.progress_signal)
    logger.info(
        "Feature transform: categorical=%s, dropped=%s, dim=%s",
        feature_config.categorical_encoding,
        list(feature_config.drop_features),
        len(feature_names),
    )

    outer_trainval, outer_test = deduped.split(
        train_ratio=args.outer_train_ratio,
        seed=args.outer_seed,
        group_by="episode",
        stratify_by_task=True,
    )
    inner_fit, inner_val = outer_trainval.split(
        train_ratio=args.inner_train_ratio,
        seed=args.inner_seed,
        group_by="episode",
        stratify_by_task=True,
    )

    trainval_ids = {cp.checkpoint_id for cp in outer_trainval.checkpoints}
    fit_ids = {cp.checkpoint_id for cp in inner_fit.checkpoints}
    val_ids = {cp.checkpoint_id for cp in inner_val.checkpoints}
    test_ids = {cp.checkpoint_id for cp in outer_test.checkpoints}

    fit_data = _filter_examples(all_data, fit_ids)
    val_data = _filter_examples(all_data, val_ids)
    trainval_data = _filter_examples(all_data, trainval_ids)
    test_data = _filter_examples(all_data, test_ids)

    logger.info(
        "Split sizes: fit=%s val=%s trainval=%s test=%s",
        len(fit_data),
        len(val_data),
        len(trainval_data),
        len(test_data),
    )

    stage_a_min_abs_margin_candidates = _parse_float_candidates(args.stage_a_min_abs_margin_candidates)
    intervene_margin_threshold_candidates = _parse_float_candidates(
        args.intervene_margin_threshold_candidates
    )
    repair_margin_threshold_candidates = _parse_float_candidates(args.repair_margin_threshold_candidates)
    stage_a_negative_weight_candidates = _parse_float_candidates(args.stage_a_negative_weight_candidates)

    search_rows = []
    for (
        stage_a_min_abs_margin,
        intervene_margin_threshold,
        repair_margin_threshold,
        stage_a_negative_weight,
    ) in itertools.product(
        stage_a_min_abs_margin_candidates,
        intervene_margin_threshold_candidates,
        repair_margin_threshold_candidates,
        stage_a_negative_weight_candidates,
    ):
        gate = TwoStageRegressionGate(
            bypass_threshold=args.bypass_threshold,
            beta=args.beta,
            margin_eps=args.margin_eps,
            stage_a_min_abs_margin=stage_a_min_abs_margin,
            intervene_margin_threshold=intervene_margin_threshold,
            repair_margin_threshold=repair_margin_threshold,
            stage_a_positive_weight=args.stage_a_positive_weight,
            stage_a_negative_weight=stage_a_negative_weight,
            feature_names=feature_names,
        )
        row = gate.train(fit_data, val_data)
        row.update(
            {
                "stage_a_min_abs_margin": stage_a_min_abs_margin,
                "intervene_margin_threshold": intervene_margin_threshold,
                "repair_margin_threshold": repair_margin_threshold,
                "stage_a_positive_weight": args.stage_a_positive_weight,
                "stage_a_negative_weight": stage_a_negative_weight,
            }
        )
        search_rows.append(row)

    best_row = _pick_best_result(search_rows, args.selection_metric)
    selected_config = {
        "stage_a_min_abs_margin": best_row["stage_a_min_abs_margin"],
        "intervene_margin_threshold": best_row["intervene_margin_threshold"],
        "repair_margin_threshold": best_row["repair_margin_threshold"],
        "stage_a_positive_weight": best_row["stage_a_positive_weight"],
        "stage_a_negative_weight": best_row["stage_a_negative_weight"],
    }

    final_gate = TwoStageRegressionGate(
        bypass_threshold=args.bypass_threshold,
        beta=args.beta,
        margin_eps=args.margin_eps,
        stage_a_min_abs_margin=selected_config["stage_a_min_abs_margin"],
        intervene_margin_threshold=selected_config["intervene_margin_threshold"],
        repair_margin_threshold=selected_config["repair_margin_threshold"],
        stage_a_positive_weight=selected_config["stage_a_positive_weight"],
        stage_a_negative_weight=selected_config["stage_a_negative_weight"],
        feature_names=feature_names,
    )
    test_metrics = final_gate.train(trainval_data, test_data)
    baseline_metrics = _always_none_baseline(test_data)
    oracle_results = oracle_analysis(all_data)
    final_gate.save(args.output)

    sorted_search_rows = sorted(
        search_rows,
        key=lambda row: (
            -row["avg_gain_over_none"],
            row["avg_regret"],
            row["predicted_intervene_rate"],
        ),
    )

    analysis = {
        "protocol": "clean train/val/test selection with untouched outer test",
        "data_config": {
            "progress_signal": args.progress_signal,
            "combined_llm_weight": args.combined_llm_weight,
            "beta": args.beta,
            "categorical_encoding": feature_config.categorical_encoding,
            "drop_features": list(feature_config.drop_features),
            "feature_dim": len(feature_names),
        },
        "split_config": {
            "outer_seed": args.outer_seed,
            "inner_seed": args.inner_seed,
            "group_by": "episode",
            "stratify_by_task": True,
            "n_total": len(all_data),
            "n_fit": len(fit_data),
            "n_val": len(val_data),
            "n_trainval": len(trainval_data),
            "n_test": len(test_data),
            "outer_train_ratio": args.outer_train_ratio,
            "inner_train_ratio": args.inner_train_ratio,
            "outer_task_counts_trainval": _task_checkpoint_counts(outer_trainval.checkpoints),
            "outer_task_counts_test": _task_checkpoint_counts(outer_test.checkpoints),
            "inner_task_counts_fit": _task_checkpoint_counts(inner_fit.checkpoints),
            "inner_task_counts_val": _task_checkpoint_counts(inner_val.checkpoints),
        },
        "oracle_analysis": oracle_results,
        "search_space": {
            "stage_a_min_abs_margin": stage_a_min_abs_margin_candidates,
            "intervene_margin_threshold": intervene_margin_threshold_candidates,
            "repair_margin_threshold": repair_margin_threshold_candidates,
            "stage_a_negative_weight": stage_a_negative_weight_candidates,
            "stage_a_positive_weight": args.stage_a_positive_weight,
        },
        "selection_metric": args.selection_metric,
        "val_leaderboard_topk": sorted_search_rows[: args.top_k],
        "selected_config": selected_config,
        "test_metrics": test_metrics,
        "test_always_none_baseline": baseline_metrics,
        "artifact_prefix": args.output,
    }
    analysis_path = args.output + "_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    logger.info("Saved analysis to %s", analysis_path)
    return analysis


def main():
    args = parse_args()
    setup_logging(log_level="INFO", log_dir="logs", run_name=args.run_name.rstrip("_"))
    run_cleanval(args)


if __name__ == "__main__":
    main()

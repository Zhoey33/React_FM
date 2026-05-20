"""Postmortem analysis for a trained two-stage ScienceWorld gate."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gate.checkpoint import CheckpointState, CheckpointStore
from src.gate.features import extract_features_from_checkpoint
from src.gate.train_gate_two_stage import (
    TwoStageInterventionGate,
    derive_two_stage_targets,
)
from experiments.gate.run_gate_training import build_training_data, load_rollout_results


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze trained two-stage gate postmortem slices")
    parser.add_argument("--checkpoints", required=True, help="Checkpoint JSON used for training")
    parser.add_argument("--rollouts", required=True, help="Judged rollout JSON")
    parser.add_argument("--model", required=True, help="Trained two-stage gate path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument(
        "--progress-signal",
        choices=["env", "llm", "combined"],
        default="combined",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--top-misses", type=int, default=12)
    return parser.parse_args()


def _summarize_slice(rows: list[dict]) -> dict:
    if not rows:
        return {
            "count": 0,
            "accuracy": 0.0,
            "binary_intervene_accuracy": 0.0,
            "avg_oracle_margin": 0.0,
            "oracle_distribution": {},
            "pred_distribution": {},
        }
    return {
        "count": len(rows),
        "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4),
        "binary_intervene_accuracy": round(
            sum(r["intervene_true"] == r["intervene_pred"] for r in rows) / len(rows), 4
        ),
        "avg_oracle_margin": round(float(np.mean([r["oracle_margin"] for r in rows])), 4),
        "oracle_distribution": dict(Counter(r["true_best"] for r in rows)),
        "pred_distribution": dict(Counter(r["pred_best"] for r in rows)),
    }


def main():
    args = parse_args()

    cp_store = CheckpointStore()
    cp_store.load(args.checkpoints)
    rollout_data = load_rollout_results(args.rollouts)
    all_data = build_training_data(
        cp_store.checkpoints,
        rollout_data,
        beta=args.beta,
        progress_signal=args.progress_signal,
    )
    train_store, test_store = cp_store.split(
        train_ratio=args.train_ratio,
        group_by="episode",
        stratify_by_task=True,
    )
    train_ids = {cp.checkpoint_id for cp in train_store.checkpoints}
    test_ids = {cp.checkpoint_id for cp in test_store.checkpoints}

    gate = TwoStageInterventionGate()
    gate.load(args.model)
    cp_map = {cp.checkpoint_id: cp for cp in cp_store.checkpoints}

    rows = []
    for item in all_data:
        cp_id = item["checkpoint_id"]
        cp_dict = rollout_data.get("__resolved_checkpoints__", {}).get(cp_id)
        cp = CheckpointState.from_dict(cp_dict) if cp_dict is not None else cp_map[cp_id]
        features = extract_features_from_checkpoint(cp)
        pred = gate.predict_details(features)
        target = derive_two_stage_targets(item)
        true_utils = item["arm_utilities"]
        ranked = sorted(true_utils.items(), key=lambda kv: kv[1], reverse=True)

        rows.append({
            "checkpoint_id": cp_id,
            "split": "train" if cp_id in train_ids else "test",
            "task": cp.task_type,
            "variation": cp.env_state.get("variation_idx"),
            "env_idx": cp.env_idx,
            "failure_type": cp.failure_type,
            "step_idx": cp.step_idx,
            "retrieval_rrf_score": cp.retrieval_rrf_score,
            "retrieval_margin": cp.retrieval_margin,
            "memory_entry_count": cp.memory_entry_count,
            "true_best": str(target["oracle_arm"]),
            "pred_best": str(pred["pred_arm"]),
            "oracle_margin": float(ranked[0][1] - ranked[1][1]),
            "intervene_true": str(target["oracle_arm"]) != "none",
            "intervene_pred": str(pred["pred_arm"]) != "none",
            "stage_a_margin": float(target["stage_a_margin"]),
            "stage_b_margin": float(target["stage_b_margin"]),
            "stage_a_prob": float(pred["p_intervene"]),
            "stage_b_repair_prob": float(pred["p_repair_given_intervene"]),
            "true_utilities": {k: round(float(v), 4) for k, v in true_utils.items()},
        })

    test_rows = [r for r in rows if r["split"] == "test"]
    train_rows = [r for r in rows if r["split"] == "train"]
    for row in rows:
        row["correct"] = row["true_best"] == row["pred_best"]

    confusion = {}
    for gold in ["none", "question", "repair"]:
        confusion[gold] = dict(Counter(r["pred_best"] for r in test_rows if r["true_best"] == gold))

    q_or_r_subset = [r for r in test_rows if r["true_best"] in {"question", "repair"}]
    q_restricted = [r for r in test_rows if r["intervene_pred"] and r["true_best"] in {"question", "repair"}]
    tp = sum(r["intervene_true"] and r["intervene_pred"] for r in test_rows)
    fp = sum((not r["intervene_true"]) and r["intervene_pred"] for r in test_rows)
    fn = sum(r["intervene_true"] and (not r["intervene_pred"]) for r in test_rows)

    per_task = {}
    for task in sorted({r["task"] for r in test_rows}):
        per_task[task] = _summarize_slice([r for r in test_rows if r["task"] == task])

    per_failure = {}
    for failure_type in sorted({r["failure_type"] for r in test_rows}):
        per_failure[failure_type] = _summarize_slice(
            [r for r in test_rows if r["failure_type"] == failure_type]
        )

    hardest_misses = sorted(
        [r for r in test_rows if not r["correct"]],
        key=lambda r: (-r["oracle_margin"], r["task"], r["checkpoint_id"]),
    )[: args.top_misses]

    result = {
        "config": {
            "checkpoints": args.checkpoints,
            "rollouts": args.rollouts,
            "model": args.model,
            "beta": args.beta,
            "progress_signal": args.progress_signal,
            "train_ratio": args.train_ratio,
        },
        "split": {
            "n_total": len(rows),
            "n_train": len(train_rows),
            "n_test": len(test_rows),
            "test_checkpoint_ids": [r["checkpoint_id"] for r in test_rows],
        },
        "overall": {
            "test_accuracy": round(sum(r["correct"] for r in test_rows) / len(test_rows), 4),
            "binary_intervene_accuracy": round(
                sum(r["intervene_true"] == r["intervene_pred"] for r in test_rows) / len(test_rows), 4
            ),
            "intervene_precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0,
            "intervene_recall": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0,
            "oracle_intervene_rate": round(
                sum(r["intervene_true"] for r in test_rows) / len(test_rows), 4
            ),
            "predicted_intervene_rate": round(
                sum(r["intervene_pred"] for r in test_rows) / len(test_rows), 4
            ),
            "question_vs_repair_accuracy_on_oracle_intervene": round(
                sum(r["pred_best"] == r["true_best"] for r in q_or_r_subset) / len(q_or_r_subset), 4
            ) if q_or_r_subset else 0.0,
            "question_vs_repair_accuracy_when_predicted_intervene": round(
                sum(r["pred_best"] == r["true_best"] for r in q_restricted) / len(q_restricted), 4
            ) if q_restricted else 0.0,
            "confusion_matrix": confusion,
        },
        "per_task_test": per_task,
        "per_failure_test": per_failure,
        "hardest_test_misses": hardest_misses,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved postmortem to {output_path}")
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()

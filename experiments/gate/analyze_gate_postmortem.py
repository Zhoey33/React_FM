"""Postmortem analysis for a trained ScienceWorld gate model.

Builds the same offline training data used by run_gate_training.py, replays the
deterministic split, and emits detailed slice metrics for error analysis.
"""

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gate.branched_rollout import GATE_ARMS
from src.gate.checkpoint import CheckpointState, CheckpointStore
from src.gate.features import extract_features_from_checkpoint
from src.gate.train_gate import ARM_ONEHOT
from experiments.gate.run_gate_training import build_training_data, load_rollout_results


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze trained gate postmortem slices")
    parser.add_argument("--checkpoints", required=True, help="Checkpoint JSON used for training")
    parser.add_argument("--rollouts", required=True, help="Judged rollout JSON")
    parser.add_argument("--model", required=True, help="Trained gate pickle path")
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


def _predict_utilities(model, features: np.ndarray) -> dict[str, float]:
    preds = {}
    for arm in GATE_ARMS:
        row = np.concatenate(
            [features, np.array(ARM_ONEHOT[arm], dtype=np.float32)]
        ).reshape(1, -1)
        preds[arm] = float(model.predict(row)[0])
    return preds


def _round_dict(d: dict[str, float], digits: int = 4) -> dict[str, float]:
    return {k: round(float(v), digits) for k, v in d.items()}


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
        "avg_oracle_margin": round(
            float(np.mean([r["oracle_margin"] for r in rows])), 4
        ),
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

    model = pickle.load(open(args.model, "rb"))
    cp_map = {cp.checkpoint_id: cp for cp in cp_store.checkpoints}

    rows = []
    for item in all_data:
        cp_id = item["checkpoint_id"]
        cp_dict = rollout_data.get("__resolved_checkpoints__", {}).get(cp_id)
        cp = CheckpointState.from_dict(cp_dict) if cp_dict is not None else cp_map[cp_id]
        features = extract_features_from_checkpoint(cp)
        pred = _predict_utilities(model, features)
        true = item["arm_utilities"]
        ranked = sorted(true.items(), key=lambda kv: kv[1], reverse=True)

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
            "true_best": max(true, key=true.get),
            "pred_best": max(pred, key=pred.get),
            "oracle_margin": float(ranked[0][1] - ranked[1][1]),
            "intervene_true": max(true, key=true.get) != "none",
            "intervene_pred": max(pred, key=pred.get) != "none",
            "cue_utility": item.get("cue_utility"),
            "true_utilities": _round_dict(true),
            "pred_utilities": _round_dict(pred),
        })

    test_rows = [r for r in rows if r["split"] == "test"]
    train_rows = [r for r in rows if r["split"] == "train"]
    for row in rows:
        row["correct"] = row["true_best"] == row["pred_best"]

    confusion = {}
    for gold in GATE_ARMS:
        confusion[gold] = dict(Counter(r["pred_best"] for r in test_rows if r["true_best"] == gold))

    oracle_intervene = [r for r in test_rows if r["intervene_true"]]
    predicted_intervene = [r for r in test_rows if r["intervene_pred"]]
    q_or_r_subset = [r for r in test_rows if r["true_best"] in {"question", "repair"}]
    q_restricted = [
        r for r in test_rows
        if r["true_best"] in {"question", "repair"} and r["pred_best"] in {"question", "repair"}
    ]

    per_task = {}
    for task in sorted({r["task"] for r in test_rows}):
        per_task[task] = _summarize_slice([r for r in test_rows if r["task"] == task])

    per_failure = {}
    for failure_type in sorted({r["failure_type"] for r in test_rows}):
        per_failure[failure_type] = _summarize_slice(
            [r for r in test_rows if r["failure_type"] == failure_type]
        )

    question_vs_cue_all = [r for r in rows if r["cue_utility"] is not None]
    question_vs_cue_by_task = {}
    for task in sorted({r["task"] for r in question_vs_cue_all}):
        task_rows = [r for r in question_vs_cue_all if r["task"] == task]
        diffs = [r["true_utilities"]["question"] - r["cue_utility"] for r in task_rows]
        question_vs_cue_by_task[task] = {
            "count": len(task_rows),
            "question_better": sum(d > 0 for d in diffs),
            "cue_better": sum(d < 0 for d in diffs),
            "ties": sum(d == 0 for d in diffs),
            "avg_question_minus_cue": round(float(np.mean(diffs)), 4),
        }

    train_coverage = defaultdict(Counter)
    for row in train_rows:
        train_coverage[row["task"]][row["true_best"]] += 1

    hardest_misses = sorted(
        [r for r in test_rows if not r["correct"]],
        key=lambda r: (-r["oracle_margin"], r["task"], r["checkpoint_id"]),
    )[: args.top_misses]

    intervene_margins = [
        max(r["true_utilities"]["question"], r["true_utilities"]["repair"]) - r["true_utilities"]["none"]
        for r in rows
    ]

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
            "oracle_intervene_rate": round(
                sum(r["intervene_true"] for r in test_rows) / len(test_rows), 4
            ),
            "predicted_intervene_rate": round(
                sum(r["intervene_pred"] for r in test_rows) / len(test_rows), 4
            ),
            "question_vs_repair_accuracy_on_oracle_intervene": round(
                sum(r["pred_best"] == r["true_best"] for r in q_or_r_subset) / len(q_or_r_subset), 4
            ) if q_or_r_subset else 0.0,
            "question_vs_repair_accuracy_when_both_predicted_interventions": round(
                sum(r["pred_best"] == r["true_best"] for r in q_restricted) / len(q_restricted), 4
            ) if q_restricted else 0.0,
            "confusion_matrix": confusion,
        },
        "intervene_margin": {
            "mean_best_intervention_minus_none": round(float(np.mean(intervene_margins)), 4),
            "median_best_intervention_minus_none": round(float(np.median(intervene_margins)), 4),
            "positive_count": int(sum(m > 0 for m in intervene_margins)),
            "strong_positive_gt_0_05": int(sum(m > 0.05 for m in intervene_margins)),
        },
        "per_task_test": per_task,
        "per_failure_test": per_failure,
        "question_vs_cue": {
            "all": {
                "count": len(question_vs_cue_all),
                "question_better": int(sum(
                    r["true_utilities"]["question"] > r["cue_utility"] for r in question_vs_cue_all
                )),
                "cue_better": int(sum(
                    r["cue_utility"] > r["true_utilities"]["question"] for r in question_vs_cue_all
                )),
                "ties": int(sum(
                    r["cue_utility"] == r["true_utilities"]["question"] for r in question_vs_cue_all
                )),
                "avg_question_minus_cue": round(float(np.mean([
                    r["true_utilities"]["question"] - r["cue_utility"] for r in question_vs_cue_all
                ])), 4),
            },
            "by_task": question_vs_cue_by_task,
        },
        "train_coverage_by_task_oracle": {
            task: dict(counter) for task, counter in sorted(train_coverage.items())
        },
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

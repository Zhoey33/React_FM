"""End-to-end gate training pipeline.

1. Load checkpoints from collect_checkpoints output
2. Deduplicate
3. Load/run branched rollouts
4. Compute utilities
5. Train XGBoost gate
6. Evaluate on held-out split
7. Run oracle analysis

Usage:
    python experiments/gate/run_gate_training.py \
        --checkpoints checkpoints/scienceworld_checkpoints.json \
        --rollouts rollouts/scienceworld_rollouts.json \
        --output gate_models/scienceworld_gate \
        --beta 0.3
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gate.checkpoint import CheckpointState, CheckpointStore
from src.gate.features import extract_features_from_checkpoint
from src.gate.branched_rollout import GATE_ARMS, ROLLOUT_ARMS, compute_arm_utilities, find_oracle_arm
from src.gate.train_gate import InterventionGate
from src.log_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train intervention gate")
    parser.add_argument("--checkpoints", required=True, help="Checkpoint store JSON")
    parser.add_argument("--rollouts", required=True, help="Rollout results JSON")
    parser.add_argument("--output", required=True, help="Output gate model path (without extension)")
    parser.add_argument("--beta", type=float, default=0.3, help="Disruption penalty weight")
    parser.add_argument("--bypass-threshold", type=float, default=0.0, help="RRF bypass threshold (0.0 = disabled)")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train/test split ratio")
    parser.add_argument("--max-per-sig", type=int, default=3, help="Max checkpoints per signature in dedup")
    parser.add_argument("--use-llm-judge", action="store_true",
                        help="Use llm_judge_progress instead of env-based progress (requires judged rollouts)")
    parser.add_argument(
        "--progress-signal",
        choices=["env", "llm", "combined"],
        default=None,
        help="Which replay progress signal to use for utility computation",
    )
    parser.add_argument(
        "--combined-llm-weight",
        type=float,
        default=None,
        help="If set with --progress-signal combined, recompute combined progress offline",
    )
    parser.add_argument("--run-name", default="gate_train_")
    return parser.parse_args()


def load_rollout_results(filepath: str) -> dict:
    """Load rollout results from JSON.

    Expected format: {
        checkpoint_id: {
            arm: [
                {progress, score_before, score_after, ...},  # replay 0
                ...
            ]
        }
    }
    """
    with open(filepath) as f:
        data = json.load(f)
    return data


def _validate_arm_rollouts(checkpoint_id: str, arm_rollouts: dict) -> None:
    missing = [arm for arm in GATE_ARMS if not arm_rollouts.get(arm)]
    if missing:
        raise ValueError(f"Checkpoint {checkpoint_id} missing rollout arms: {missing}")

    gate_counts = {arm: len(arm_rollouts[arm]) for arm in GATE_ARMS}
    if len(set(gate_counts.values())) != 1:
        raise ValueError(f"Checkpoint {checkpoint_id} has inconsistent replay counts: {gate_counts}")


def build_training_data(
    checkpoints: list,
    rollout_data: dict,
    beta: float = 0.3,
    progress_signal: str = "env",
    combined_llm_weight: float | None = None,
) -> list[dict]:
    """Build training data from checkpoints + rollout results.

    Args:
        progress_signal: env | llm | combined
        combined_llm_weight: if set and progress_signal=combined, recompute
            combined progress as:
            max(env_progress, (1-w)*env_progress + w*llm_judge_progress)

    Returns list of {
        "checkpoint_id": str,
        "features": np.ndarray (11,),
        "arm_utilities": {arm: float},
        "oracle_arm": str,
    }
    """
    progress_key = {
        "env": "progress",
        "llm": "llm_judge_progress",
        "combined": "combined_progress",
    }[progress_signal]
    fallback_key = "progress"  # always available

    if progress_signal == "llm":
        logger.info("Using LLM judge progress for utility computation")
    elif progress_signal == "combined":
        if combined_llm_weight is None:
            logger.info("Using combined env+LLM progress for utility computation")
        else:
            logger.info(
                "Using recomputed combined env+LLM progress for utility computation "
                f"(llm_weight={combined_llm_weight:.2f})"
            )

    training_data = []
    selected_missing = 0
    selected_used = 0

    for cp in checkpoints:
        cp_id = cp.checkpoint_id
        if cp_id not in rollout_data:
            logger.warning(f"No rollouts for checkpoint {cp_id}, skipping")
            continue

        arm_rollouts = rollout_data[cp_id]
        _validate_arm_rollouts(cp_id, arm_rollouts)

        # Extract features
        resolved_cp_dict = rollout_data.get("__resolved_checkpoints__", {}).get(cp_id)
        if resolved_cp_dict is not None:
            feature_cp = CheckpointState.from_dict(resolved_cp_dict)
        else:
            feature_cp = cp
        features = extract_features_from_checkpoint(feature_cp)

        # Helper to get progress from a replay dict
        def _get_progress(r):
            nonlocal selected_missing, selected_used
            if progress_signal == "combined" and combined_llm_weight is not None:
                env_val = float(r.get("progress", 0.0))
                llm_val = r.get("llm_judge_progress")
                if llm_val is None:
                    selected_missing += 1
                    return env_val
                selected_used += 1
                llm_val = float(llm_val)
                return max(
                    env_val,
                    (1.0 - combined_llm_weight) * env_val + combined_llm_weight * llm_val,
                )
            val = r.get(progress_key)
            if progress_signal == "env":
                return float(val if val is not None else 0.0)
            if val is not None:
                selected_used += 1
                return float(val)
            selected_missing += 1
            return float(r.get(fallback_key, 0.0))

        # Compute utilities from rollouts using replay-matched disruption
        # Index none replays by replay_idx for matched disruption
        none_replays = arm_rollouts.get("none", [])
        none_by_idx = {r.get("replay_idx", i): _get_progress(r)
                       for i, r in enumerate(none_replays)}

        # Compute progress averages per arm (including cue for analysis)
        arm_progress_avg = {}
        for arm in ROLLOUT_ARMS:
            if arm not in arm_rollouts or not arm_rollouts[arm]:
                continue
            replays = arm_rollouts[arm]
            progress_values = [_get_progress(r) for r in replays]
            arm_progress_avg[arm] = float(np.mean(progress_values))

        # Compute disruption-adjusted utility for gate arms using per-replay matching
        gate_utilities = {}
        for arm in GATE_ARMS:
            if arm not in arm_rollouts or not arm_rollouts[arm]:
                continue
            replays = arm_rollouts[arm]
            progress_values = [_get_progress(r) for r in replays]
            progress_avg = float(np.mean(progress_values))

            if arm == "none":
                disruption_avg = 0.0
            else:
                disruption_values = []
                for i, r in enumerate(replays):
                    ridx = r.get("replay_idx", i)
                    none_progress = none_by_idx.get(ridx, 0.0)
                    disruption_values.append(max(0, none_progress - _get_progress(r)))
                disruption_avg = float(np.mean(disruption_values)) if disruption_values else 0.0

            utility = progress_avg - beta * disruption_avg
            gate_utilities[arm] = utility

        # Compute cue utility the same way for question vs cue analysis
        cue_utility = None
        if "cue" in arm_rollouts and arm_rollouts["cue"]:
            cue_replays = arm_rollouts["cue"]
            cue_progress = float(np.mean([_get_progress(r) for r in cue_replays]))
            cue_disrupt = []
            for i, r in enumerate(cue_replays):
                ridx = r.get("replay_idx", i)
                cue_disrupt.append(max(0, none_by_idx.get(ridx, 0.0) - _get_progress(r)))
            cue_utility = cue_progress - beta * float(np.mean(cue_disrupt)) if cue_disrupt else cue_progress

        if not gate_utilities:
            continue

        oracle_arm = max(gate_utilities, key=gate_utilities.get)

        training_data.append({
            "checkpoint_id": cp_id,
            "features": features,
            "arm_utilities": gate_utilities,
            "oracle_arm": oracle_arm,
            # Cue utility (disruption-adjusted) for question vs cue analysis
            "cue_utility": cue_utility,
        })

    if progress_signal in {"llm", "combined"}:
        total_reads = selected_used + selected_missing
        logger.info(f"{progress_signal} progress: {selected_used}/{total_reads} reads used {progress_key}")
        if selected_missing > 0:
            logger.warning(
                f"{progress_key} MISSING: {selected_missing}/{total_reads} reads fell back to env progress! "
                f"This means {selected_missing} replays lack {progress_key} — "
                f"re-run run_llm_judge_scoring.py to fill gaps."
            )

    return training_data


def oracle_analysis(training_data: list[dict]) -> dict:
    """Analyze oracle arm distribution and heterogeneity."""
    oracle_counts = {arm: 0 for arm in GATE_ARMS}
    total = len(training_data)

    for item in training_data:
        oracle_counts[item["oracle_arm"]] += 1

    oracle_fractions = {arm: count / total for arm, count in oracle_counts.items()} if total > 0 else {}
    heterogeneity_index = 1 - max(oracle_fractions.values()) if oracle_fractions else 0.0

    # Conditional analysis: when oracle=none, what is always-repair utility?
    conditional_penalties = {}
    for oracle_arm in GATE_ARMS:
        subset = [d for d in training_data if d["oracle_arm"] == oracle_arm]
        if subset:
            repair_utils = [d["arm_utilities"].get("repair", 0.0) for d in subset]
            conditional_penalties[f"repair_utility_when_oracle={oracle_arm}"] = round(
                float(np.mean(repair_utils)), 4
            )

    # question vs cue analysis
    question_better = 0
    cue_better = 0
    comparable = 0

    for item in training_data:
        q_util = item["arm_utilities"].get("question", None)
        c_util = item.get("cue_utility", None)
        if q_util is not None and c_util is not None:
            comparable += 1
            if q_util > c_util + 0.01:
                question_better += 1
            elif c_util > q_util + 0.01:
                cue_better += 1

    result = {
        "total_checkpoints": total,
        "oracle_counts": oracle_counts,
        "oracle_fractions": {k: round(v, 4) for k, v in oracle_fractions.items()},
        "heterogeneity_index": round(heterogeneity_index, 4),
        "conditional_penalties": conditional_penalties,
        "question_vs_cue": {
            "comparable": comparable,
            "question_better": question_better,
            "cue_better": cue_better,
            "question_win_rate": round(question_better / comparable, 4) if comparable > 0 else 0.0,
        },
    }

    logger.info(f"\nOracle Analysis:")
    logger.info(f"  Distribution: {oracle_counts}")
    logger.info(f"  Fractions: {result['oracle_fractions']}")
    logger.info(f"  Heterogeneity index: {result['heterogeneity_index']:.3f} (>0.3 needed)")
    logger.info(f"  Question vs Cue: Q wins {question_better}/{comparable}")

    return result


def main():
    args = parse_args()
    setup_logging(log_level="INFO", log_dir="logs", run_name=args.run_name.rstrip("_"))
    progress_signal = args.progress_signal or ("llm" if args.use_llm_judge else "env")

    # 1. Load checkpoints
    cp_store = CheckpointStore()
    cp_store.load(args.checkpoints)
    logger.info(f"Loaded {len(cp_store)} checkpoints")

    # 2. Deduplicate
    deduped = cp_store.deduplicate(max_per_signature=args.max_per_sig)

    # 3. Load rollout results
    rollout_data = load_rollout_results(args.rollouts)
    rollout_cp_count = sum(1 for cp_id in rollout_data if not cp_id.startswith("__"))
    logger.info(f"Loaded rollouts for {rollout_cp_count} checkpoints")

    # 4. Build training data
    all_data = build_training_data(
        deduped.checkpoints, rollout_data,
        beta=args.beta,
        progress_signal=progress_signal,
        combined_llm_weight=args.combined_llm_weight,
    )
    logger.info(f"Built {len(all_data)} training examples"
                f"{'' if progress_signal == 'env' else f' ({progress_signal} progress)'}")

    if len(all_data) < 10:
        logger.error("Too few training examples! Need at least 10 checkpoints.")
        return

    # 5. Oracle analysis (on all data, before split)
    oracle_results = oracle_analysis(all_data)

    # Check heterogeneity abort condition
    if oracle_results["heterogeneity_index"] < 0.3:
        logger.warning(
            f"LOW HETEROGENEITY: index={oracle_results['heterogeneity_index']:.3f} < 0.3. "
            "Consider aborting gate paper."
        )

    # 6. Split train/test
    # Keep checkpoints from the same episode together to avoid leakage.
    train_store, test_store = deduped.split(
        train_ratio=args.train_ratio,
        group_by="episode",
        stratify_by_task=True,
    )
    train_ids = {cp.checkpoint_id for cp in train_store.checkpoints}
    test_ids = {cp.checkpoint_id for cp in test_store.checkpoints}

    train_data = [d for d in all_data if d["checkpoint_id"] in train_ids]
    test_data = [d for d in all_data if d["checkpoint_id"] in test_ids]

    logger.info(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # 7. Train gate
    gate = InterventionGate(bypass_threshold=args.bypass_threshold, beta=args.beta)
    train_stats = gate.train(train_data, test_data)

    logger.info(f"\nGate Training Results:")
    for k, v in train_stats.items():
        logger.info(f"  {k}: {v}")

    # 8. Save
    gate.save(args.output)

    # 9. Save full analysis
    analysis = {
        "oracle_analysis": oracle_results,
        "train_stats": train_stats,
        "feature_importance": gate.feature_importance(),
        "config": {
            "beta": args.beta,
            "bypass_threshold": args.bypass_threshold,
            "progress_signal": progress_signal,
            "combined_llm_weight": args.combined_llm_weight,
            "train_ratio": args.train_ratio,
            "n_checkpoints_raw": len(cp_store),
            "n_checkpoints_deduped": len(deduped),
            "n_train": len(train_data),
            "n_test": len(test_data),
        },
    }
    analysis_path = args.output + "_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"Analysis saved: {analysis_path}")


if __name__ == "__main__":
    main()

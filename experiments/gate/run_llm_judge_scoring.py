"""Score ScienceWorld rollout trajectories using LLM-as-judge.

Reads existing rollout data and checkpoint data, scores each trajectory
with an LLM, and outputs an augmented rollout JSON with llm_judge_score
and llm_judge_progress fields.

Usage:
    python experiments/gate/run_llm_judge_scoring.py \
        --rollouts rollouts/scienceworld_main_rollouts.json \
        --checkpoints checkpoints/scienceworld_checkpoints_midband.json \
        --gold-paths rollouts/scienceworld_gold_paths.json \
        --config config_scienceworld.yaml \
        --output rollouts/scienceworld_rollouts_judged.json

    # Quick test on 2 CPs:
    python experiments/gate/run_llm_judge_scoring.py \
        --rollouts rollouts/scienceworld_main_rollouts.json \
        --checkpoints checkpoints/scienceworld_checkpoints_midband.json \
        --gold-paths rollouts/scienceworld_gold_paths.json \
        --config config_scienceworld.yaml \
        --output rollouts/scienceworld_rollouts_judged.json \
        --max-checkpoints 2
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.llm import LLMClient
from src.gate.llm_judge import (
    combine_progress_scienceworld,
    score_trajectory,
    get_gold_path_for_checkpoint,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LLM judge scoring for SW rollouts")
    parser.add_argument("--rollouts", required=True, help="Input rollout JSON")
    parser.add_argument("--checkpoints", required=True, help="Checkpoint store JSON")
    parser.add_argument("--gold-paths", required=True, help="Gold paths JSON")
    parser.add_argument("--config", default="config_scienceworld.yaml")
    parser.add_argument("--output", required=True, help="Output judged rollout JSON")
    parser.add_argument("--max-checkpoints", type=int, default=None,
                        help="Limit CPs for testing")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output (skip already-scored)")
    parser.add_argument("--arms", nargs="*", default=None,
                        help="Only score these arms (default: all)")
    parser.add_argument(
        "--llm-weight",
        type=float,
        default=0.6,
        help="Weight of LLM progress in combined_progress; env progress remains the floor",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load data
    with open(args.rollouts) as f:
        rollout_data = json.load(f)
    passthrough_meta = {
        k: v for k, v in rollout_data.items()
        if k.startswith("__")
    }
    passthrough_meta.setdefault("__meta__", {})
    passthrough_meta["__meta__"]["judge_config"] = {
        "llm_weight": args.llm_weight,
        "combined_progress_formula": "max(env_progress, (1-llm_weight)*env_progress + llm_weight*llm_judge_progress)",
    }
    with open(args.checkpoints) as f:
        cp_data = json.load(f)
    with open(args.gold_paths) as f:
        gold_paths = json.load(f)
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Build CP lookup
    cp_map = {c["checkpoint_id"]: c for c in cp_data["checkpoints"]}

    # LLM client (use extractor config — same model, higher tokens)
    ext_cfg = config["extractor"]
    llm = LLMClient(
        model=ext_cfg["model"],
        base_url=ext_cfg["base_url"],
        api_key=ext_cfg.get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
        temperature=0.0,
        max_tokens=256,
    )

    # Resume support
    existing_output = dict(passthrough_meta)
    if args.resume and Path(args.output).exists():
        with open(args.output) as f:
            existing_output = json.load(f)
        for k, v in passthrough_meta.items():
            existing_output.setdefault(k, v)
        scored_cp_count = sum(1 for cp_id in existing_output if not cp_id.startswith("__"))
        logger.info(f"Resuming: {scored_cp_count} CPs already scored")

    # Filter CPs
    cp_ids = [cp_id for cp_id in rollout_data.keys() if not cp_id.startswith("__") and cp_id in cp_map]
    if args.max_checkpoints:
        cp_ids = cp_ids[:args.max_checkpoints]

    arms_filter = set(args.arms) if args.arms else None

    # Stats
    total_scored = 0
    total_skipped = 0
    no_gold_path = 0
    score_sum = 0.0

    logger.info(f"LLM Judge Scoring: {len(cp_ids)} CPs, "
                f"arms={args.arms or 'all'}")

    t0 = time.time()

    for cp_idx, cp_id in enumerate(cp_ids):
        cp = cp_map[cp_id]
        arms_data = rollout_data[cp_id]

        # Get gold path
        gold_actions = get_gold_path_for_checkpoint(cp, gold_paths)
        if gold_actions is None:
            logger.warning(f"No gold path for {cp_id} ({cp['task_type']}), "
                           "using score-based progress only")
            no_gold_path += 1
            # Copy as-is
            if cp_id not in existing_output:
                existing_output[cp_id] = arms_data
            continue

        # Task description
        task_desc = cp.get("init_obs", "")
        if "\n\n" in task_desc:
            task_desc = task_desc.split("\n\n")[0]

        # Context history
        context_history = list(cp.get("history", []))
        failure_action = cp.get("failure_action", "")
        failure_observation = cp.get("failure_observation", "")
        if failure_action or failure_observation:
            context_history.append((failure_action, failure_observation))

        # Check if already scored
        if cp_id in existing_output:
            # Check if all replays have llm_judge_score
            all_scored = True
            for arm, replays in existing_output[cp_id].items():
                for r in replays:
                    if "llm_judge_score" not in r:
                        all_scored = False
                        break
            if all_scored:
                total_skipped += sum(
                    len(replays) for replays in existing_output[cp_id].values()
                )
                continue

        cp_output = {}

        for arm, replays in arms_data.items():
            if arms_filter and arm not in arms_filter:
                cp_output[arm] = replays  # copy as-is
                continue

            scored_replays = []
            for replay in replays:
                traj_actions = replay.get("actions", [])
                traj_obs = replay.get("observations_full") or replay.get("observations", [])

                # Score with LLM
                label = f"judge_{cp_id}_{arm}_r{replay.get('replay_idx', 0)}"
                result = score_trajectory(
                    llm=llm,
                    task_desc=task_desc,
                    gold_actions=gold_actions,
                    context_history=context_history,
                    traj_actions=traj_actions,
                    traj_observations=traj_obs,
                    label=label,
                )

                # Augment replay with judge scores
                scored_replay = dict(replay)
                env_progress = float(replay.get("progress", 0.0))
                combined_progress = combine_progress_scienceworld(
                    env_progress=env_progress,
                    llm_progress=result["progress"],
                    llm_weight=args.llm_weight,
                )
                scored_replay["llm_judge_score"] = result["score"]
                scored_replay["llm_judge_progress"] = result["progress"]
                scored_replay["combined_progress"] = combined_progress
                scored_replay["llm_judge_reasoning"] = result.get("reasoning", "")
                scored_replay["judge_context_includes_failure_step"] = True

                scored_replays.append(scored_replay)
                total_scored += 1
                if result["score"] is not None:
                    score_sum += result["score"]

                logger.debug(
                    f"  {cp_id}/{arm}/r{replay.get('replay_idx', 0)}: "
                    f"score={result['score']}, "
                    f"env_progress={replay.get('progress', 0):.3f}, "
                    f"llm_progress={result['progress']:.2f}, "
                    f"combined_progress={combined_progress:.2f}"
                )

            cp_output[arm] = scored_replays

        existing_output[cp_id] = cp_output

        # Progress log
        elapsed = time.time() - t0
        avg_time = elapsed / (cp_idx + 1)
        remaining = avg_time * (len(cp_ids) - cp_idx - 1)
        logger.info(
            f"[{cp_idx + 1}/{len(cp_ids)}] {cp_id} ({cp['task_type']}): "
            f"scored {sum(len(v) for v in cp_output.values())} trajectories, "
            f"ETA {remaining / 60:.1f}min"
        )

        # Save intermediate every 5 CPs
        if (cp_idx + 1) % 5 == 0:
            _save(existing_output, args.output.replace(".json", "_intermediate.json"))

    elapsed = time.time() - t0

    # Final save
    _save(existing_output, args.output)

    # Summary
    avg_score = score_sum / total_scored if total_scored > 0 else 0
    logger.info(f"\n{'=' * 60}")
    logger.info(f"LLM Judge Scoring Complete")
    logger.info(f"  CPs processed: {len(cp_ids)}")
    logger.info(f"  Trajectories scored: {total_scored}")
    logger.info(f"  Skipped (already scored): {total_skipped}")
    logger.info(f"  No gold path: {no_gold_path}")
    logger.info(f"  Average LLM score: {avg_score:.2f}/10")
    logger.info(f"  Elapsed: {elapsed:.0f}s ({elapsed / max(total_scored, 1):.1f}s/traj)")
    logger.info(f"  Output: {args.output}")
    logger.info(f"{'=' * 60}")

    # Quick comparison: env-based vs LLM-based progress
    _print_comparison(existing_output)


def _save(data: dict, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _print_comparison(data: dict):
    """Print comparison of env-based vs LLM-based progress."""
    logger.info(f"\n--- Env vs Judge Progress Comparison ---")

    blind_count = 0
    llm_diff_count = 0
    combined_diff_count = 0

    for cp_id, arms in data.items():
        if cp_id.startswith("__"):
            continue
        none_replays = arms.get("none", [])
        repair_replays = arms.get("repair", [])
        if not none_replays or not repair_replays:
            continue

        env_none = sum(r.get("progress", 0) for r in none_replays) / len(none_replays)
        env_repair = sum(r.get("progress", 0) for r in repair_replays) / len(repair_replays)

        llm_none_scores = [r.get("llm_judge_progress", None) for r in none_replays]
        llm_repair_scores = [r.get("llm_judge_progress", None) for r in repair_replays]
        combined_none_scores = [r.get("combined_progress", None) for r in none_replays]
        combined_repair_scores = [r.get("combined_progress", None) for r in repair_replays]

        if any(s is None for s in llm_none_scores + llm_repair_scores + combined_none_scores + combined_repair_scores):
            continue

        llm_none = sum(llm_none_scores) / len(llm_none_scores)
        llm_repair = sum(llm_repair_scores) / len(llm_repair_scores)
        combined_none = sum(combined_none_scores) / len(combined_none_scores)
        combined_repair = sum(combined_repair_scores) / len(combined_repair_scores)

        is_env_blind = abs(env_none - env_repair) < 0.001
        is_llm_diff = abs(llm_none - llm_repair) > 0.05
        is_combined_diff = abs(combined_none - combined_repair) > 0.05

        if is_env_blind:
            blind_count += 1
            if is_llm_diff:
                llm_diff_count += 1
            if is_combined_diff:
                combined_diff_count += 1
            if is_llm_diff or is_combined_diff:
                better = "repair" if llm_repair > llm_none else "none"
                logger.info(
                    f"  {cp_id}: env_blind=True, "
                    f"llm none={llm_none:.2f} vs repair={llm_repair:.2f}, "
                    f"combined none={combined_none:.2f} vs repair={combined_repair:.2f} "
                    f"({better} better by llm)"
                )

    if blind_count > 0:
        logger.info(
            f"\n  LLM detection rate: {llm_diff_count}/{blind_count} "
            f"({llm_diff_count / blind_count * 100:.0f}%) of env-blind CPs"
        )
        logger.info(
            f"  Combined detection rate: {combined_diff_count}/{blind_count} "
            f"({combined_diff_count / blind_count * 100:.0f}%) of env-blind CPs"
        )


if __name__ == "__main__":
    main()

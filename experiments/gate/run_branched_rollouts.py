"""Stage 2: Collect branched rollout data from failure checkpoints.

For each checkpoint, reset the environment, replay to the checkpoint state,
then branch into 4 arms × N steps × M replays. Saves results as JSON
for gate training (run_gate_training.py).

Usage:
    python experiments/gate/run_branched_rollouts.py \
        --benchmark scienceworld \
        --checkpoints checkpoints/scienceworld_checkpoints.json \
        --memory memory/scienceworld_memory.json \
        --config config.yaml \
        --output rollouts/scienceworld_rollouts.json \
        --n-steps 5 --n-replays 3
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llm import LLMClient
from src.memory import FailureMemoryStore
from src.gate.checkpoint import CheckpointState, CheckpointStore
from src.gate.branched_rollout import (
    ROLLOUT_ARMS, run_branched_rollout, replay_to_checkpoint,
)
from src.log_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect branched rollout data")
    parser.add_argument("--benchmark", required=True,
                        choices=["scienceworld", "alfworld", "webshop"])
    parser.add_argument("--checkpoints", required=True, help="Checkpoint store JSON")
    parser.add_argument("--memory", required=True, help="Canonicalized memory store JSON")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", required=True, help="Output rollout results JSON")
    parser.add_argument("--n-steps", type=int, default=5, help="Steps forward per rollout")
    parser.add_argument("--n-replays", type=int, default=3, help="Replays per arm")
    parser.add_argument("--arms", nargs="*", default=None,
                        help="Arms to run (default: all 4). E.g., --arms none repair")
    parser.add_argument("--max-checkpoints", type=int, default=None,
                        help="Limit number of checkpoints (for testing)")
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save intermediate rollout payload every N completed checkpoints",
    )
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from partial rollout JSON")
    parser.add_argument(
        "--resolved-checkpoints-output",
        type=str,
        default=None,
        help="Optional path to save resolved checkpoint features used during rollout",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Keep partial checkpoints after replay failures instead of failing fast",
    )
    parser.add_argument(
        "--reject-numeric-action-history",
        action="store_true",
        help="Drop checkpoints whose action_history contains raw numeric menu selections like '0'/'1'",
    )
    parser.add_argument(
        "--replay-precheck",
        action="store_true",
        help="Before rollout, replay each remaining ScienceWorld checkpoint once and drop mismatches",
    )
    parser.add_argument(
        "--replay-mismatch-tolerance",
        type=float,
        default=1.0,
        help="Allowed abs(replay_score - saved_score) during replay precheck",
    )
    parser.add_argument(
        "--rejected-checkpoints-output",
        type=str,
        default=None,
        help="Optional path to save checkpoints rejected before rollout with reasons",
    )
    parser.add_argument("--run-name", default="rollout_")
    return parser.parse_args()


def lookup_memory_entries(memory_store: FailureMemoryStore, checkpoint, cross_env: bool = False):
    """Look up the retrieved memory entries for a checkpoint.

    Also backfills checkpoint retrieval features (rrf_score, margin, top1 memory_id,
    entry_count) so gate training features match the actual memory state.

    Args:
        cross_env: Whether to search memory across all envs. Default False
            (per-env isolation). All three benchmarks have per-env memory and
            don't need cross-env lookup when each env has its own entries.
    """
    # Always resolve against the rollout-time memory store so:
    # 1) the injected entry matches the memory actually available now
    # 2) retrieval features used for gate training match the rollout labels
    ret = memory_store.retrieve(
        query_action=checkpoint.failure_action,
        query_observation=checkpoint.failure_observation,
        task_type=checkpoint.task_type,
        top_k=memory_store.top_k,
        env_idx=checkpoint.env_idx,
        cross_env=cross_env,
        return_scores=True,
    )
    checkpoint.memory_entry_count = memory_store.entry_count(
        task_type=checkpoint.task_type,
        env_idx=checkpoint.env_idx,
        cross_env=cross_env,
    )
    if ret.entries:
        checkpoint.retrieved_memory_id = ret.entries[0].memory_id
        checkpoint.retrieval_rrf_score = ret.rrf_scores[0]
        checkpoint.retrieval_margin = (
            ret.rrf_scores[0] - ret.rrf_scores[1]
            if len(ret.rrf_scores) > 1 else ret.rrf_scores[0]
        )
        return ret.entries
    checkpoint.retrieved_memory_id = -1
    checkpoint.retrieval_rrf_score = 0.0
    checkpoint.retrieval_margin = 0.0
    return []


def _merge_results_payload(cp_results: dict, resolved_checkpoints: dict, meta: dict) -> dict:
    payload = dict(cp_results)
    payload["__meta__"] = meta
    payload["__resolved_checkpoints__"] = resolved_checkpoints
    return payload


def _rollout_meta_from_args(args) -> dict:
    return {
        "benchmark": args.benchmark,
        "n_steps": args.n_steps,
        "n_replays": args.n_replays,
        "arms": args.arms or ROLLOUT_ARMS,
    }


def _checkpoint_ids_fingerprint(checkpoints: list[CheckpointState]) -> str:
    material = "\n".join(cp.checkpoint_id for cp in checkpoints)
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


def _build_rollout_meta(args, checkpoints: list[CheckpointState], memory_store: FailureMemoryStore) -> dict:
    meta = _rollout_meta_from_args(args)
    meta.update({
        "checkpoints_path": str(Path(args.checkpoints).resolve()),
        "memory_path": str(Path(args.memory).resolve()),
        "config_path": str(Path(args.config).resolve()),
        "output_path": str(Path(args.output).resolve()),
        "resolved_checkpoints_output": str(
            Path(
                args.resolved_checkpoints_output
                or args.output.replace(".json", "_resolved_checkpoints.json")
            ).resolve()
        ),
        "save_every": args.save_every,
        "checkpoint_count": len(checkpoints),
        "checkpoint_ids_sha1": _checkpoint_ids_fingerprint(checkpoints),
        "memory_scope": memory_store.scope,
        "memory_size": memory_store.size(),
        "memory_top_k": memory_store.top_k,
    })
    return meta


def _validate_resume_meta(current_meta: dict, resume_meta: dict) -> None:
    required_keys = [
        "benchmark",
        "n_steps",
        "n_replays",
        "arms",
        "checkpoints_path",
        "memory_path",
        "config_path",
        "checkpoint_count",
        "checkpoint_ids_sha1",
        "memory_scope",
        "memory_top_k",
    ]
    missing = [k for k in required_keys if k not in resume_meta]
    if missing:
        raise RuntimeError(
            "Resume file is missing required metadata for safe resume: "
            f"{missing}"
        )

    mismatches = []
    for key in required_keys:
        if current_meta.get(key) != resume_meta.get(key):
            mismatches.append((key, resume_meta.get(key), current_meta.get(key)))

    if mismatches:
        lines = [
            f"{key}: resume={old!r}, current={new!r}"
            for key, old, new in mismatches
        ]
        raise RuntimeError(
            "Resume metadata mismatch. Refusing to merge partial rollout data:\n"
            + "\n".join(lines)
        )


def _resolved_checkpoint_snapshots(checkpoints: list, included_ids: set[str]) -> dict:
    return {
        cp.checkpoint_id: cp.to_dict()
        for cp in checkpoints
        if cp.checkpoint_id in included_ids
    }


def _assert_complete_arm_results(cp_id: str, arm: str, arm_results: list[dict], expected_replays: int) -> None:
    if len(arm_results) != expected_replays:
        raise RuntimeError(
            f"Incomplete rollout for {cp_id}/{arm}: expected {expected_replays}, got {len(arm_results)}"
        )


def _rejected_output_path(args) -> str:
    return args.rejected_checkpoints_output or args.output.replace(
        ".json", "_rejected_checkpoints.json"
    )


def _save_rejected_checkpoints(rejected: list[dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"rejected": rejected}, f, indent=2)
    logger.info(f"Saved rejected checkpoints to {output_path}")


def _append_rejected(rejected: list[dict], checkpoint: CheckpointState, reason: str, **extra) -> None:
    item = checkpoint.to_dict()
    item["reject_reason"] = reason
    item.update(extra)
    rejected.append(item)


def _has_numeric_only_action(action: str) -> bool:
    return bool(re.fullmatch(r"\d+", str(action).strip()))


def _reject_numeric_history_checkpoints(
    checkpoints: list[CheckpointState],
    rejected: list[dict],
) -> list[CheckpointState]:
    kept = []
    for cp in checkpoints:
        numeric_actions = [a for a in cp.action_history if _has_numeric_only_action(a)]
        if numeric_actions:
            _append_rejected(
                rejected,
                cp,
                reason="numeric_action_history",
                numeric_actions=numeric_actions,
                numeric_action_count=len(numeric_actions),
            )
            continue
        kept.append(cp)
    return kept


def _prevalidate_scienceworld_checkpoints(
    checkpoints: list[CheckpointState],
    tolerance: float,
    rejected: list[dict],
) -> list[CheckpointState]:
    import warnings
    warnings.filterwarnings("ignore", message=".*camel case.*")
    from scienceworld import ScienceWorldEnv as SWEnv

    sw_env = SWEnv("", envStepLimit=100)

    class SWAdapter:
        def __init__(self, raw_env):
            self.raw_env = raw_env

        def step(self, action):
            obs, reward, done, info = self.raw_env.step(action)
            return obs, float(reward), bool(done), {"score": info.get("score", 0.0)}

    adapter = SWAdapter(sw_env)
    kept = []

    try:
        for idx, cp in enumerate(checkpoints, start=1):
            task_name = cp.env_state.get("task_name", "")
            var_idx = cp.env_state.get("variation_idx", 0)
            if not task_name:
                _append_rejected(rejected, cp, reason="missing_task_name")
                continue

            try:
                sw_env.load(task_name, var_idx)
                sw_env.reset()
                replay_score = replay_to_checkpoint(cp, adapter)
                score_diff = abs(replay_score - cp.score_at_checkpoint)
                if score_diff > tolerance:
                    _append_rejected(
                        rejected,
                        cp,
                        reason="replay_score_mismatch",
                        replay_score=replay_score,
                        saved_score=cp.score_at_checkpoint,
                        score_diff=score_diff,
                    )
                    continue
            except Exception as e:
                _append_rejected(
                    rejected,
                    cp,
                    reason="replay_exception",
                    error=str(e),
                )
                continue

            kept.append(cp)
            if idx % 25 == 0:
                logger.info(
                    f"Replay precheck progress: {idx}/{len(checkpoints)} "
                    f"(kept={len(kept)}, rejected={len(rejected)})"
                )
    finally:
        sw_env.close()

    return kept


def run_scienceworld_rollouts(args, config, cp_store, memory_store, rollout_meta):
    """Run branched rollouts on ScienceWorld checkpoints."""
    import warnings
    warnings.filterwarnings("ignore", message=".*camel case.*")
    from scienceworld import ScienceWorldEnv as SWEnv
    from experiments.run_scienceworld import _extract_action
    from prompts.scienceworld_prompts import build_user_prompt, SYSTEM_PROMPT_FM

    llm = LLMClient(
        model=config["llm"]["model"],
        base_url=config["llm"]["base_url"],
        api_key=config["llm"].get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
    )

    arms = args.arms or ROLLOUT_ARMS
    n_steps = args.n_steps
    n_replays = args.n_replays

    # Create ScienceWorld env (raw, not wrapper — we control task loading)
    sw_env = SWEnv("", envStepLimit=100)
    logger.info(f"ScienceWorld JVM started for rollouts")

    # Create a thin env adapter for replay_to_checkpoint and run_branched_rollout
    class SWAdapter:
        """Adapts raw ScienceWorld env to the (obs, reward, done, info) interface."""
        def __init__(self, raw_env):
            self.raw_env = raw_env

        def step(self, action):
            obs, reward, done, info = self.raw_env.step(action)
            score = info.get("score", 0.0)
            info_dict = {"score": score}
            return obs, float(reward), bool(done), info_dict

    adapter = SWAdapter(sw_env)

    checkpoints = cp_store.checkpoints
    if args.max_checkpoints:
        checkpoints = checkpoints[:args.max_checkpoints]

    all_results = {}
    for cp_idx, cp in enumerate(checkpoints):
        cp_id = cp.checkpoint_id
        task_name = cp.env_state.get("task_name", "")
        var_idx = cp.env_state.get("variation_idx", 0)

        if not task_name:
            logger.warning(f"Checkpoint {cp_id} missing task_name, skipping")
            continue

        # Look up memory entry for question/repair arms
        mem_entries = lookup_memory_entries(memory_store, cp)

        cp_results = {}

        for arm in arms:
            arm_results = []

            for replay_idx in range(n_replays):
                try:
                    # 1. Reset env to task/variation
                    sw_env.load(task_name, var_idx)
                    sw_env.reset()

                    # 2. Replay to checkpoint state
                    replay_score = replay_to_checkpoint(cp, adapter)
                    score_diff = abs(replay_score - cp.score_at_checkpoint)
                    if score_diff > 1.0:
                        raise RuntimeError(
                            "Replay score mismatch before rollout: "
                            f"{cp_id}, replay={replay_score:.1f}, "
                            f"saved={cp.score_at_checkpoint:.1f}, diff={score_diff:.1f}"
                        )

                    # 3. Run branched rollout
                    result = run_branched_rollout(
                        checkpoint=cp,
                        arm=arm,
                        replay_idx=replay_idx,
                        llm=llm,
                        env=adapter,
                        build_prompt_fn=build_user_prompt,
                        n_steps=n_steps,
                        memory_entries=mem_entries,
                        system_prompt=SYSTEM_PROMPT_FM,
                        action_postprocess=_extract_action,
                    )
                    arm_results.append(result.to_dict())

                except Exception as e:
                    logger.error(f"Rollout failed: {cp_id}/{arm}/r{replay_idx}: {e}")
                    if not args.allow_incomplete:
                        raise RuntimeError(f"Aborting on rollout failure: {cp_id}/{arm}/r{replay_idx}") from e

            if not args.allow_incomplete:
                _assert_complete_arm_results(cp_id, arm, arm_results, n_replays)
            cp_results[arm] = arm_results

        all_results[cp_id] = cp_results

        # Progress
        logger.info(
            f"[{cp_idx + 1}/{len(checkpoints)}] {cp_id}: "
            f"{sum(len(v) for v in cp_results.values())} rollouts completed"
        )

        # Intermediate save every 5 checkpoints
        if args.save_every > 0 and (cp_idx + 1) % args.save_every == 0:
            _save_rollouts(
                all_results,
                args.output.replace(".json", "_intermediate.json"),
                resolved_checkpoints=_resolved_checkpoint_snapshots(checkpoints, set(all_results.keys())),
                meta=rollout_meta,
            )

    sw_env.close()
    return all_results


def run_alfworld_rollouts(args, config, cp_store, memory_store, rollout_meta):
    """Run branched rollouts on ALFWorld checkpoints.

    ALFWorld envs are sequential — to reach env N, we must reset() N times.
    To avoid re-creating the env for each replay (which causes broken pipe
    due to importlib.reload), we create the env once per (cp, arm, replay)
    and skip to the right env_idx.
    """
    from src.alfworld_env import ALFWorldEnv, translate_action
    from prompts.alfworld_prompts import build_user_prompt, SYSTEM_PROMPT_FM

    llm = LLMClient(
        model=config["llm"]["model"],
        base_url=config["llm"]["base_url"],
        api_key=config["llm"].get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
    )

    arms = args.arms or ROLLOUT_ARMS
    n_steps = args.n_steps
    n_replays = args.n_replays

    checkpoints = cp_store.checkpoints
    if args.max_checkpoints:
        checkpoints = checkpoints[:args.max_checkpoints]

    all_results = {}
    total = len(checkpoints) * len(arms) * n_replays

    for cp_idx, cp in enumerate(checkpoints):
        cp_id = cp.checkpoint_id
        env_idx = cp.env_idx  # 1-based
        mem_entries = lookup_memory_entries(memory_store, cp)
        cp_results = {}

        for arm in arms:
            arm_results = []
            for replay_idx in range(n_replays):
                env = None
                try:
                    # Create fresh env and skip to the right env_idx
                    env = ALFWorldEnv(
                        split=config.get("alfworld", {}).get("split", "eval_out_of_distribution")
                    )
                    env.setup()
                    for _ in range(env_idx):
                        env.reset()

                    replay_score = replay_to_checkpoint(cp, env)

                    result = run_branched_rollout(
                        checkpoint=cp,
                        arm=arm,
                        replay_idx=replay_idx,
                        llm=llm,
                        env=env,
                        build_prompt_fn=build_user_prompt,
                        n_steps=n_steps,
                        memory_entries=mem_entries,
                        system_prompt=SYSTEM_PROMPT_FM,
                    )
                    arm_results.append(result.to_dict())

                except Exception as e:
                    logger.error(f"Rollout failed: {cp_id}/{arm}/r{replay_idx}: {e}")
                    if not args.allow_incomplete:
                        raise RuntimeError(f"Aborting on rollout failure: {cp_id}/{arm}/r{replay_idx}") from e
                finally:
                    # Clean up env to avoid resource leaks
                    if env is not None:
                        try:
                            env.env = None
                        except Exception:
                            pass

            if not args.allow_incomplete:
                _assert_complete_arm_results(cp_id, arm, arm_results, n_replays)
            cp_results[arm] = arm_results

        all_results[cp_id] = cp_results
        logger.info(
            f"[{cp_idx + 1}/{len(checkpoints)}] {cp_id}: "
            f"{sum(len(v) for v in cp_results.values())} rollouts"
        )

        # Intermediate save every 5 checkpoints
        if args.save_every > 0 and (cp_idx + 1) % args.save_every == 0:
            _save_rollouts(
                all_results,
                args.output.replace(".json", "_intermediate.json"),
                resolved_checkpoints=_resolved_checkpoint_snapshots(checkpoints, set(all_results.keys())),
                meta=rollout_meta,
            )

    return all_results


def run_webshop_rollouts(args, config, cp_store, memory_store, rollout_meta):
    """Run branched rollouts on WebShop checkpoints."""
    from src.webshop_env import WebShopEnv
    from prompts.webshop_prompts import build_user_prompt, SYSTEM_PROMPT_FM

    llm = LLMClient(
        model=config["llm"]["model"],
        base_url=config["llm"]["base_url"],
        api_key=config["llm"].get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
    )

    arms = args.arms or ROLLOUT_ARMS
    n_steps = args.n_steps
    n_replays = args.n_replays

    checkpoints = cp_store.checkpoints
    if args.max_checkpoints:
        checkpoints = checkpoints[:args.max_checkpoints]

    all_results = {}

    for cp_idx, cp in enumerate(checkpoints):
        cp_id = cp.checkpoint_id
        session_idx = cp.env_state.get("session_idx", cp.env_idx - 1)
        mem_entries = lookup_memory_entries(memory_store, cp)
        cp_results = {}

        for arm in arms:
            arm_results = []
            for replay_idx in range(n_replays):
                try:
                    # WebShop supports resetting to specific session
                    ws_env = WebShopEnv()
                    ws_env.setup()
                    ws_env.reset(session_idx=session_idx)

                    replay_score = replay_to_checkpoint(cp, ws_env)

                    result = run_branched_rollout(
                        checkpoint=cp,
                        arm=arm,
                        replay_idx=replay_idx,
                        llm=llm,
                        env=ws_env,
                        build_prompt_fn=build_user_prompt,
                        n_steps=n_steps,
                        memory_entries=mem_entries,
                        system_prompt=SYSTEM_PROMPT_FM,
                    )
                    arm_results.append(result.to_dict())
                    ws_env.close()

                except Exception as e:
                    logger.error(f"Rollout failed: {cp_id}/{arm}/r{replay_idx}: {e}")
                    if not args.allow_incomplete:
                        raise RuntimeError(f"Aborting on rollout failure: {cp_id}/{arm}/r{replay_idx}") from e

            if not args.allow_incomplete:
                _assert_complete_arm_results(cp_id, arm, arm_results, n_replays)
            cp_results[arm] = arm_results

        all_results[cp_id] = cp_results
        logger.info(
            f"[{cp_idx + 1}/{len(checkpoints)}] {cp_id}: "
            f"{sum(len(v) for v in cp_results.values())} rollouts"
        )

        if args.save_every > 0 and (cp_idx + 1) % args.save_every == 0:
            _save_rollouts(
                all_results,
                args.output.replace(".json", "_intermediate.json"),
                resolved_checkpoints=_resolved_checkpoint_snapshots(checkpoints, set(all_results.keys())),
                meta=rollout_meta,
            )

    return all_results


def _save_rollouts(results: dict, filepath: str, resolved_checkpoints: dict | None = None, meta: dict | None = None):
    """Save rollout results to JSON."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    payload = _merge_results_payload(results, resolved_checkpoints or {}, meta or {})
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Saved rollouts to {filepath}")


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    setup_logging(log_level="INFO", log_dir="logs", run_name=args.run_name.rstrip("_"))

    # Load checkpoints
    cp_store = CheckpointStore()
    cp_store.load(args.checkpoints)
    if args.max_checkpoints:
        cp_store.checkpoints = cp_store.checkpoints[:args.max_checkpoints]
    logger.info(f"Loaded {len(cp_store)} checkpoints")

    # Load canonicalized memory store
    memory_store = FailureMemoryStore(
        embedding_model_name=config["memory"]["embedding_model"],
        max_entries=config["memory"]["max_entries"],
        top_k=int(config.get("agent", {}).get("max_memory_inject", config["memory"]["retrieval_top_k"])),
        scope="task_type" if args.benchmark == "scienceworld" else "env_idx",
    )
    memory_store.load(args.memory)
    logger.info(f"Loaded memory store: {memory_store.size()} entries")

    rejected_checkpoints = []
    existing_results = {}
    existing_resolved_checkpoints = {}

    if args.reject_numeric_action_history:
        before = len(cp_store.checkpoints)
        cp_store.checkpoints = _reject_numeric_history_checkpoints(
            cp_store.checkpoints,
            rejected_checkpoints,
        )
        logger.info(
            "Rejected checkpoints with numeric action history: "
            f"{before - len(cp_store.checkpoints)} dropped, {len(cp_store.checkpoints)} remain"
        )

    if args.replay_precheck:
        if args.benchmark != "scienceworld":
            raise ValueError("--replay-precheck is currently only supported for scienceworld")
        before = len(cp_store.checkpoints)
        cp_store.checkpoints = _prevalidate_scienceworld_checkpoints(
            cp_store.checkpoints,
            tolerance=args.replay_mismatch_tolerance,
            rejected=rejected_checkpoints,
        )
        logger.info(
            "Replay precheck complete: "
            f"{before - len(cp_store.checkpoints)} dropped, {len(cp_store.checkpoints)} remain"
        )

    if rejected_checkpoints:
        _save_rejected_checkpoints(rejected_checkpoints, _rejected_output_path(args))

    # Resume from partial results
    rollout_meta = _build_rollout_meta(args, cp_store.checkpoints, memory_store)
    if args.resume:
        try:
            with open(args.resume) as f:
                resume_payload = json.load(f)
            resume_results = {
                cp_id: cp_data for cp_id, cp_data in resume_payload.items()
                if not cp_id.startswith("__")
            }
            resume_resolved_checkpoints = resume_payload.get("__resolved_checkpoints__", {})
            resume_meta = resume_payload.get("__meta__", {})
            _validate_resume_meta(rollout_meta, resume_meta)
            existing_results = resume_results
            existing_resolved_checkpoints = resume_resolved_checkpoints
            logger.info(f"Resuming from {len(existing_results)} completed checkpoints")
            # Filter out already-completed checkpoints
            done_ids = set(existing_results.keys())
            cp_store.checkpoints = [
                cp for cp in cp_store.checkpoints
                if cp.checkpoint_id not in done_ids
            ]
            logger.info(f"Remaining: {len(cp_store)} checkpoints")
        except Exception as e:
            raise RuntimeError(
                f"Could not safely resume from {args.resume}: {e}"
            ) from e

    arms_str = ", ".join(args.arms) if args.arms else "all"
    logger.info(
        f"Branched rollouts: {args.benchmark}, "
        f"{len(cp_store)} checkpoints, arms=[{arms_str}], "
        f"{args.n_steps} steps × {args.n_replays} replays"
    )
    if rejected_checkpoints:
        logger.info(f"Pre-run rejected checkpoints: {len(rejected_checkpoints)}")

    t0 = time.time()

    if args.benchmark == "scienceworld":
        results = run_scienceworld_rollouts(args, config, cp_store, memory_store, rollout_meta)
    elif args.benchmark == "alfworld":
        results = run_alfworld_rollouts(args, config, cp_store, memory_store, rollout_meta)
    elif args.benchmark == "webshop":
        results = run_webshop_rollouts(args, config, cp_store, memory_store, rollout_meta)

    # Merge with existing results (resume)
    results.update(existing_results)

    elapsed = time.time() - t0

    # Save final results
    resolved_checkpoints = dict(existing_resolved_checkpoints)
    for cp in cp_store.checkpoints:
        if cp.checkpoint_id in results:
            resolved_checkpoints[cp.checkpoint_id] = cp.to_dict()

    _save_rollouts(results, args.output, resolved_checkpoints=resolved_checkpoints, meta=rollout_meta)

    resolved_output = args.resolved_checkpoints_output or args.output.replace(
        ".json", "_resolved_checkpoints.json"
    )
    resolved_store = CheckpointStore()
    resolved_store.checkpoints = [
        CheckpointState.from_dict(resolved_checkpoints[cp_id])
        for cp_id in sorted(resolved_checkpoints)
    ]
    resolved_store._next_id = cp_store._next_id
    resolved_store.save(resolved_output)

    # Summary
    total_rollouts = sum(
        sum(len(arm_list) for arm_list in cp_data.values())
        for cp_data in results.values()
    )
    logger.info(f"\n{'='*60}")
    logger.info(f"Branched rollout collection complete: {args.benchmark}")
    logger.info(f"  Checkpoints: {len(results)}")
    logger.info(f"  Total rollouts: {total_rollouts}")
    logger.info(f"  Elapsed: {elapsed:.0f}s")
    logger.info(f"  Saved: {args.output}")
    logger.info(f"  Resolved checkpoints: {resolved_output}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

"""Collect failure checkpoints from React_FM E1 runs.

Runs React_FM with checkpoint logging enabled, saving CheckpointState
at every failure detection point. This produces the raw data for
branched rollout collection.

Usage:
    python experiments/gate/collect_checkpoints.py \
        --benchmark scienceworld \
        --config config.yaml \
        --output checkpoints/scienceworld_checkpoints.json \
        --max-envs 50
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llm import LLMClient
from src.memory import FailureMemoryStore, RetrievalResult
from src.gate.checkpoint import CheckpointState, CheckpointStore
from src.log_utils import setup_logging

logger = logging.getLogger(__name__)


def _configured_retrieval_top_k(config: dict) -> int:
    return int(config.get("agent", {}).get("max_memory_inject", config["memory"]["retrieval_top_k"]))


def parse_args():
    parser = argparse.ArgumentParser(description="Collect failure checkpoints")
    parser.add_argument("--benchmark", required=True, choices=["scienceworld", "alfworld", "webshop"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", required=True, help="Output checkpoint store JSON")
    parser.add_argument("--max-envs", type=int, default=None)
    parser.add_argument("--memory-file", type=str, default=None, help="Load existing memory store")
    parser.add_argument("--inject-mode", default="in_loop")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test",
                        help="Official split to collect checkpoints from")
    parser.add_argument("--tasks", nargs="+", default=None, help="Specific task names to run")
    parser.add_argument("--max-variations", type=int, default=5, help="Max variations per task")
    parser.add_argument("--run-name", default="cp_collect_")
    return parser.parse_args()


def collect_scienceworld(args, config):
    """Collect checkpoints from ScienceWorld runs."""
    from src.scienceworld_env import ScienceWorldEnv, DEFAULT_EVAL_TASKS
    from src.scienceworld_failure_detector import ScienceWorldFailureDetector
    from prompts.scienceworld_prompts import build_user_prompt, SYSTEM_PROMPT_FM

    llm = LLMClient(
        model=config["llm"]["model"],
        base_url=config["llm"]["base_url"],
        api_key=config["llm"].get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
    )

    memory_store = FailureMemoryStore(
        embedding_model_name=config["memory"]["embedding_model"],
        max_entries=config["memory"]["max_entries"],
        top_k=_configured_retrieval_top_k(config),
        scope="task_type",
    )
    if args.memory_file:
        memory_store.load(args.memory_file)

    judge_llm = None
    if "judge" in config:
        judge_cfg = config["judge"]
        judge_llm = LLMClient(
            model=judge_cfg["model"], base_url=judge_cfg["base_url"],
            api_key=judge_cfg.get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
            temperature=0.0, max_tokens=16,
        )

    detector = ScienceWorldFailureDetector(judge_llm=judge_llm)
    extractor_llm = None
    if "extractor" in config:
        ext_cfg = config["extractor"]
        extractor_llm = LLMClient(
            model=ext_cfg["model"], base_url=ext_cfg["base_url"],
            api_key=ext_cfg.get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
            temperature=0.0, max_tokens=512,
        )

    task_names = args.tasks or DEFAULT_EVAL_TASKS
    env = ScienceWorldEnv(
        task_names=task_names,
        split=args.split,
        max_variations_per_task=args.max_variations,
    )
    env.setup()

    cp_store = CheckpointStore()
    max_envs = args.max_envs or env.total_episodes
    retrieval_top_k = _configured_retrieval_top_k(config)

    for env_count in range(max_envs):
        try:
            init_obs, task_type, info = env.reset()
        except StopIteration:
            break

        env_idx = env_count + 1
        history = []
        action_history = []
        steps = []
        env_history_records = []
        current_retrieved = None
        failures_detected = 0
        current_score = 0.0
        score_at_last_failure = 0.0

        llm.tracker.reset()

        logger.info(f"[{env_idx}/{max_envs}] task={task_type}")

        for step_num in range(config["agent"]["max_steps"]):
            prompt = build_user_prompt(
                task_type=task_type, task_obs=init_obs, history=history,
                retrieved_memories=current_retrieved, memory_style="original",
            )
            response = llm.complete_text(prompt, stop=["\n"], label=f"step_{step_num}", system=SYSTEM_PROMPT_FM)
            action = response.strip().split("\n")[0].strip()
            current_retrieved = None

            if action.startswith("> "):
                action = action[2:]

            from experiments.run_scienceworld import _extract_action
            action = _extract_action(action)
            if not action:
                action = "look around"

            is_think = action.startswith("think:") or action.startswith("think ")
            if is_think:
                history.append((action, "OK."))
                steps.append({"step": step_num, "action": action, "observation": "OK.",
                              "is_think": True, "failure_detected": False})
                continue

            observation, reward, done, step_info = env.step(action)
            action_history.append(action)
            current_score = step_info.get("score", current_score)
            env_history_records.append(
                {"step": step_num, "action": action, "observation": observation}
            )

            is_done, is_success = detector.is_task_complete(observation, done, step_info)

            record = {"step": step_num, "action": action, "observation": observation[:200],
                      "is_think": False, "failure_detected": False}

            if not is_done:
                det = detector.detect(observation, action, action_history)
                if det.is_failure:
                    record["failure_detected"] = True
                    record["failure_type"] = det.failure_type
                    failures_detected += 1

                    # Retrieve with scores for checkpoint
                    ret_result = memory_store.retrieve(
                        query_action=action, query_observation=observation,
                        task_type=task_type, top_k=retrieval_top_k, env_idx=env_idx,
                        return_scores=True,
                    )
                    if isinstance(ret_result, RetrievalResult):
                        rrf_score = ret_result.top1_score
                        rrf_margin = ret_result.margin
                        retrieved = ret_result.entries
                        mem_id = retrieved[0].memory_id if retrieved else -1
                    else:
                        rrf_score = 0.0
                        rrf_margin = 0.0
                        retrieved = ret_result
                        mem_id = -1

                    # Save checkpoint
                    cp = CheckpointState(
                        env_idx=env_idx,
                        task_type=task_type,
                        benchmark="scienceworld",
                        step_idx=step_num,
                        max_steps=config["agent"]["max_steps"],
                        failure_type=det.failure_type,
                        failure_action=action,
                        failure_observation=observation[:300],
                        history=list(history),
                        action_history=list(action_history),
                        init_obs=init_obs[:500],
                        retrieval_rrf_score=rrf_score,
                        retrieval_margin=rrf_margin,
                        retrieved_memory_id=mem_id,
                        memory_entry_count=memory_store.entry_count(task_type=task_type, env_idx=env_idx),
                        score_at_checkpoint=current_score,
                        failures_so_far=failures_detected,
                        env_state={
                            "task_name": info.get("task_name", ""),
                            "variation_idx": info.get("variation_idx", 0),
                            "score": current_score,
                            "score_at_last_failure": score_at_last_failure,
                            "failures_in_last_5": sum(
                                1 for s in steps[-5:] if s.get("failure_detected")
                            ),
                            "same_failure_count": sum(
                                1 for s in steps if s.get("failure_type") == det.failure_type
                            ),
                        },
                    )
                    cp_store.add(cp)
                    logger.info(f"  Checkpoint saved: {cp.checkpoint_id} ({det.failure_type})")
                    score_at_last_failure = current_score

                    if retrieved:
                        current_retrieved = retrieved

            steps.append(record)
            history.append((action, observation))

            if is_done:
                # Mark all checkpoints from this episode with success/fail
                for cp in cp_store.checkpoints:
                    if cp.env_idx == env_idx and cp.episode_success is None:
                        cp.episode_success = is_success
                break

        # Post-episode: extract memories (same as React_FM)
        if extractor_llm is not None:
            from src.scienceworld_memory_extractor import extract_failure_recoveries
            detected_failures = [
                {
                    "step": step["step"],
                    "action": step["action"],
                    "observation": step["observation"],
                    "failure_type": step.get("failure_type", ""),
                }
                for step in steps
                if step.get("failure_detected")
                and not step.get("is_think")
                and step.get("failure_type") != "unproductive"
            ]
            recoveries = extract_failure_recoveries(
                extractor_llm,
                env_history_records,
                detected_failures=detected_failures,
            )
            for rec in recoveries:
                memory_store.add(
                    failure_action=rec["failure_action"],
                    failure_observation=rec["failure_observation"],
                    solution_action=rec["solution_action"],
                    task_type=task_type, env_idx=env_idx,
                )

        # Intermediate save
        if (env_count + 1) % 10 == 0:
            cp_store.save(args.output.replace(".json", "_intermediate.json"))

    env.close()
    return cp_store, memory_store


def collect_alfworld(args, config):
    """Collect checkpoints from ALFWorld runs."""
    from src.alfworld_env import ALFWorldEnv
    from src.failure_detector import ALFWorldFailureDetector
    from src.memory_extractor import extract_failure_recoveries
    from prompts.alfworld_prompts import build_user_prompt, SYSTEM_PROMPT_FM

    llm = LLMClient(
        model=config["llm"]["model"],
        base_url=config["llm"]["base_url"],
        api_key=config["llm"].get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
    )

    memory_store = FailureMemoryStore(
        embedding_model_name=config["memory"]["embedding_model"],
        max_entries=config["memory"]["max_entries"],
        top_k=config["memory"]["retrieval_top_k"],
    )
    if args.memory_file:
        memory_store.load(args.memory_file)

    judge_llm = None
    if "judge" in config:
        judge_cfg = config["judge"]
        judge_llm = LLMClient(
            model=judge_cfg["model"], base_url=judge_cfg["base_url"],
            api_key=judge_cfg.get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
            temperature=0.0, max_tokens=16,
        )

    detector = ALFWorldFailureDetector(judge_llm=judge_llm)
    extractor_llm = None
    if "extractor" in config:
        ext_cfg = config["extractor"]
        extractor_llm = LLMClient(
            model=ext_cfg["model"], base_url=ext_cfg["base_url"],
            api_key=ext_cfg.get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
            temperature=0.0, max_tokens=512,
        )

    env = ALFWorldEnv(split=config.get("alfworld", {}).get("split", "eval_out_of_distribution"))
    env.setup()

    cp_store = CheckpointStore()
    max_envs = args.max_envs or 134

    for env_count in range(max_envs):
        try:
            init_obs, task_type, info = env.reset()
        except StopIteration:
            break

        env_idx = env_count + 1
        history = []
        action_history = []
        steps = []
        current_retrieved = None
        failures_detected = 0
        score_at_last_failure = 0.0

        llm.tracker.reset()
        logger.info(f"[{env_idx}/{max_envs}] task={task_type}")

        # Extract gamefile for env restoration
        gamefile = ""
        if isinstance(info.get("extra.gamefile"), list):
            gamefile = info["extra.gamefile"][0] if info["extra.gamefile"] else ""
        else:
            gamefile = info.get("extra.gamefile", "")

        for step_num in range(config["agent"]["max_steps"]):
            prompt = build_user_prompt(
                task_type=task_type, task_obs=init_obs, history=history,
                retrieved_memories=current_retrieved, memory_style="original",
            )
            response = llm.complete_text(prompt, stop=["\n"], label=f"step_{step_num}", system=SYSTEM_PROMPT_FM)
            action = response.strip().split("\n")[0].strip()
            current_retrieved = None

            if action.startswith("> "):
                action = action[2:]
            if not action:
                action = "look"

            is_think = action.startswith("think:") or action.startswith("think ")
            if is_think:
                history.append((action, "OK."))
                steps.append({"step": step_num, "action": action, "observation": "OK.",
                              "is_think": True, "failure_detected": False})
                continue

            observation, reward, done, step_info = env.step(action)
            action_history.append(action)

            is_done, is_success = detector.is_task_complete(observation, done, step_info)

            record = {"step": step_num, "action": action, "observation": observation[:200],
                      "is_think": False, "failure_detected": False}

            if not is_done:
                det = detector.detect(observation, action, action_history)
                if det.is_failure:
                    record["failure_detected"] = True
                    record["failure_type"] = det.failure_type
                    failures_detected += 1

                    ret_result = memory_store.retrieve(
                        query_action=action, query_observation=observation,
                        task_type=task_type, top_k=3, env_idx=env_idx,
                        return_scores=True,
                    )
                    if isinstance(ret_result, RetrievalResult):
                        rrf_score = ret_result.top1_score
                        rrf_margin = ret_result.margin
                        retrieved = ret_result.entries
                        mem_id = retrieved[0].memory_id if retrieved else -1
                    else:
                        rrf_score = 0.0
                        rrf_margin = 0.0
                        retrieved = ret_result
                        mem_id = -1

                    cp = CheckpointState(
                        env_idx=env_idx,
                        task_type=task_type,
                        benchmark="alfworld",
                        step_idx=step_num,
                        max_steps=config["agent"]["max_steps"],
                        failure_type=det.failure_type,
                        failure_action=action,
                        failure_observation=observation[:300],
                        history=list(history),
                        action_history=list(action_history),
                        init_obs=init_obs[:500],
                        retrieval_rrf_score=rrf_score,
                        retrieval_margin=rrf_margin,
                        retrieved_memory_id=mem_id,
                        memory_entry_count=memory_store.size(),
                        score_at_checkpoint=0.0,  # ALFWorld: binary, no partial score
                        failures_so_far=failures_detected,
                        env_state={
                            "gamefile": gamefile,
                            "score_at_last_failure": score_at_last_failure,
                            "failures_in_last_5": sum(
                                1 for s in steps[-5:] if s.get("failure_detected")
                            ),
                            "same_failure_count": sum(
                                1 for s in steps if s.get("failure_type") == det.failure_type
                            ),
                        },
                    )
                    cp_store.add(cp)
                    logger.info(f"  Checkpoint saved: {cp.checkpoint_id} ({det.failure_type})")

                    if retrieved:
                        current_retrieved = retrieved

            steps.append(record)
            history.append((action, observation))

            if is_done:
                for cp_item in cp_store.checkpoints:
                    if cp_item.env_idx == env_idx and cp_item.episode_success is None:
                        cp_item.episode_success = is_success
                break

        # Post-episode: extract memories
        if extractor_llm is not None:
            env_history = [(a, o) for a, o in history if not a.startswith("think")]
            recoveries = extract_failure_recoveries(extractor_llm, env_history)
            for rec in recoveries:
                memory_store.add(
                    failure_action=rec["failure_action"],
                    failure_observation=rec["failure_observation"],
                    solution_action=rec["solution_action"],
                    task_type=task_type, env_idx=env_idx,
                )

        if (env_count + 1) % 10 == 0:
            cp_store.save(args.output.replace(".json", "_intermediate.json"))

    return cp_store, memory_store


def collect_webshop(args, config):
    """Collect checkpoints from WebShop runs."""
    from src.webshop_env import WebShopEnv
    from src.webshop_failure_detector import WebShopFailureDetector
    from src.webshop_memory_extractor import extract_failure_recoveries
    from prompts.webshop_prompts import build_user_prompt, SYSTEM_PROMPT_FM

    llm = LLMClient(
        model=config["llm"]["model"],
        base_url=config["llm"]["base_url"],
        api_key=config["llm"].get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
    )

    memory_store = FailureMemoryStore(
        embedding_model_name=config["memory"]["embedding_model"],
        max_entries=config["memory"]["max_entries"],
        top_k=config["memory"]["retrieval_top_k"],
    )
    if args.memory_file:
        memory_store.load(args.memory_file)

    judge_llm = None
    if "judge" in config:
        judge_cfg = config["judge"]
        judge_llm = LLMClient(
            model=judge_cfg["model"], base_url=judge_cfg["base_url"],
            api_key=judge_cfg.get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
            temperature=0.0, max_tokens=16,
        )

    detector = WebShopFailureDetector(judge_llm=judge_llm)
    extractor_llm = None
    if "extractor" in config:
        ext_cfg = config["extractor"]
        extractor_llm = LLMClient(
            model=ext_cfg["model"], base_url=ext_cfg["base_url"],
            api_key=ext_cfg.get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
            temperature=0.0, max_tokens=512,
        )

    env = WebShopEnv()
    env.setup()

    cp_store = CheckpointStore()
    max_envs = args.max_envs or 100

    for env_count in range(max_envs):
        try:
            init_obs, task_type, info = env.reset()
        except Exception:
            break

        env_idx = env_count + 1
        history = []
        action_history = []
        steps = []
        current_retrieved = None
        failures_detected = 0
        current_reward = 0.0
        score_at_last_failure = 0.0

        llm.tracker.reset()
        session_idx = info.get("session_idx", env_count)
        logger.info(f"[{env_idx}/{max_envs}] session={session_idx}")

        for step_num in range(config["agent"]["max_steps"]):
            prompt = build_user_prompt(
                task_type=task_type, task_obs=init_obs, history=history,
                retrieved_memories=current_retrieved, memory_style="original",
            )
            response = llm.complete_text(prompt, stop=["\n"], label=f"step_{step_num}", system=SYSTEM_PROMPT_FM)
            action = response.strip().split("\n")[0].strip()
            current_retrieved = None

            if action.startswith("> "):
                action = action[2:]
            if not action:
                action = "search[product]"

            is_think = action.startswith("think:") or action.startswith("think ")
            if is_think:
                history.append((action, "OK."))
                steps.append({"step": step_num, "action": action, "observation": "OK.",
                              "is_think": True, "failure_detected": False})
                continue

            observation, reward, done, step_info = env.step(action)
            action_history.append(action)
            if reward > 0:
                current_reward = reward

            is_done, is_success = detector.is_task_complete(observation, done, step_info)

            record = {"step": step_num, "action": action, "observation": observation[:200],
                      "is_think": False, "failure_detected": False}

            if not is_done:
                det = detector.detect(observation, action, action_history)
                if det.is_failure:
                    record["failure_detected"] = True
                    record["failure_type"] = det.failure_type
                    failures_detected += 1

                    ret_result = memory_store.retrieve(
                        query_action=action, query_observation=observation,
                        task_type=task_type, top_k=3, env_idx=env_idx,
                        return_scores=True,
                    )
                    if isinstance(ret_result, RetrievalResult):
                        rrf_score = ret_result.top1_score
                        rrf_margin = ret_result.margin
                        retrieved = ret_result.entries
                        mem_id = retrieved[0].memory_id if retrieved else -1
                    else:
                        rrf_score = 0.0
                        rrf_margin = 0.0
                        retrieved = ret_result
                        mem_id = -1

                    cp = CheckpointState(
                        env_idx=env_idx,
                        task_type=task_type,
                        benchmark="webshop",
                        step_idx=step_num,
                        max_steps=config["agent"]["max_steps"],
                        failure_type=det.failure_type,
                        failure_action=action,
                        failure_observation=observation[:300],
                        history=list(history),
                        action_history=list(action_history),
                        init_obs=init_obs[:500],
                        retrieval_rrf_score=rrf_score,
                        retrieval_margin=rrf_margin,
                        retrieved_memory_id=mem_id,
                        memory_entry_count=memory_store.size(),
                        score_at_checkpoint=current_reward,
                        failures_so_far=failures_detected,
                        env_state={
                            "session_idx": session_idx,
                            "score_at_last_failure": score_at_last_failure,
                            "failures_in_last_5": sum(
                                1 for s in steps[-5:] if s.get("failure_detected")
                            ),
                            "same_failure_count": sum(
                                1 for s in steps if s.get("failure_type") == det.failure_type
                            ),
                        },
                    )
                    cp_store.add(cp)
                    score_at_last_failure = current_reward
                    logger.info(f"  Checkpoint saved: {cp.checkpoint_id} ({det.failure_type})")

                    if retrieved:
                        current_retrieved = retrieved

            steps.append(record)
            history.append((action, observation))

            if is_done:
                for cp_item in cp_store.checkpoints:
                    if cp_item.env_idx == env_idx and cp_item.episode_success is None:
                        cp_item.episode_success = is_success
                break

        # Post-episode: extract memories
        if extractor_llm is not None:
            env_history = [(a, o) for a, o in history if not a.startswith("think")]
            recoveries = extract_failure_recoveries(extractor_llm, env_history)
            for rec in recoveries:
                memory_store.add(
                    failure_action=rec["failure_action"],
                    failure_observation=rec["failure_observation"],
                    solution_action=rec["solution_action"],
                    task_type=task_type, env_idx=env_idx,
                )

        if (env_count + 1) % 10 == 0:
            cp_store.save(args.output.replace(".json", "_intermediate.json"))

    env.close()
    return cp_store, memory_store


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    setup_logging(log_level="INFO", log_dir="logs", run_name=args.run_name.rstrip("_"))

    logger.info(f"Collecting checkpoints for {args.benchmark}")

    if args.benchmark == "scienceworld":
        cp_store, memory_store = collect_scienceworld(args, config)
    elif args.benchmark == "alfworld":
        cp_store, memory_store = collect_alfworld(args, config)
    elif args.benchmark == "webshop":
        cp_store, memory_store = collect_webshop(args, config)

    # Save final results
    cp_store.save(args.output)

    # Also save memory store
    mem_path = args.output.replace("checkpoints", "memory").replace("_checkpoints", "_memory")
    Path(mem_path).parent.mkdir(parents=True, exist_ok=True)
    memory_store.save(mem_path)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Checkpoint collection complete: {args.benchmark}")
    logger.info(f"  Total checkpoints: {len(cp_store)}")
    logger.info(f"  Unique signatures: {len(set(cp.failure_signature for cp in cp_store.checkpoints))}")
    logger.info(f"  Memory entries: {memory_store.size()}")
    logger.info(f"  Saved to: {args.output}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

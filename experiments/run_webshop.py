"""Run React_FM (or baseline ReAct) on WebShop."""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import logging
import sys
import random
import time
from pathlib import Path

import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import LLMClient
from src.memory import FailureMemoryStore
from src.webshop_failure_detector import WebShopFailureDetector
from src.webshop_env import WebShopEnv
from src.log_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run React_FM on WebShop")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--max-envs", type=int, default=None, help="Max sessions")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--baseline", action="store_true", help="Vanilla ReAct")
    parser.add_argument("--resume-memory", type=str, default=None)
    parser.add_argument("--resume-results", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--inject-mode", choices=["in_loop", "episode", "none"], default="in_loop")
    parser.add_argument("--memory-style", choices=["original", "factual", "reflexion", "hint"], default="original")
    parser.add_argument("--memory-format", choices=["failure_recovery", "success_trajectory", "reflexion_reflection"],
                        default="failure_recovery", help="Memory storage format for ablation")
    parser.add_argument("--retrieval-mode", choices=["hybrid", "bm25_only", "embedding_only", "random"],
                        default="hybrid", help="Retrieval method for ablation")
    parser.add_argument("--cross-env", action="store_true",
                        help="Enable cross-env memory sharing (retrieve from all envs)")
    parser.add_argument("--num-products", type=int, default=1000, help="Number of products")
    parser.add_argument("--max-steps", type=int, default=15, help="Max steps per episode")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def save_results(results: list[dict], summary: dict, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    data = {"summary": summary, "episodes": results}
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def compute_summary(results: list[dict], memory_stats: dict, mode: str) -> dict:
    total = len(results)
    # WebShop uses continuous reward, not binary success
    rewards = [r.get("reward", 0.0) for r in results]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    successes = sum(1 for r in rewards if r >= 0.5)

    total_tokens = sum(r["total_tokens"] for r in results)
    agent_tokens = sum(r.get("agent_tokens", r["total_tokens"]) for r in results)
    judge_tokens = sum(r.get("judge_tokens", 0) for r in results)
    extractor_tokens = sum(r.get("extractor_tokens", 0) for r in results)

    return {
        "mode": mode,
        "benchmark": "webshop",
        "total_envs": total,
        "total_success": successes,
        "success_rate": round(successes / total, 4) if total > 0 else 0,
        "avg_reward": round(avg_reward, 4),
        "total_tokens": total_tokens,
        "agent_tokens": agent_tokens,
        "judge_tokens": judge_tokens,
        "extractor_tokens": extractor_tokens,
        "avg_tokens_per_episode": round(total_tokens / total) if total > 0 else 0,
        "memory_stats": memory_stats,
    }


def log_summary(summary: dict):
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  {summary['mode']} Results (WebShop)")
    logger.info("=" * 60)
    logger.info(f"  Success (reward≥0.5): {summary['total_success']}/{summary['total_envs']} "
                f"({summary['success_rate']:.1%})")
    logger.info(f"  Avg Reward: {summary['avg_reward']:.4f}")
    logger.info("-" * 60)
    logger.info(f"  Tokens: {summary['total_tokens']:,} total, "
                f"{summary['avg_tokens_per_episode']:,} avg/episode")
    if summary["memory_stats"]:
        ms = summary["memory_stats"]
        logger.info(f"  Memory: {ms.get('total_entries', 0)} entries, "
                    f"{ms.get('total_retrievals', 0)} retrievals, "
                    f"{ms.get('total_hits', 0)} hits")
    logger.info("=" * 60)


class WebShopReActAgent:
    """ReAct agent for WebShop."""

    def __init__(
        self,
        llm: LLMClient,
        memory_store: FailureMemoryStore | None = None,
        failure_detector: WebShopFailureDetector | None = None,
        extractor_llm: LLMClient | None = None,
        max_steps: int = 15,
        max_memory_inject: int = 3,
        enable_memory: bool = True,
        inject_mode: str = "in_loop",
        memory_style: str = "original",
        memory_format: str = "failure_recovery",
        cross_env: bool = False,
    ):
        self.llm = llm
        self.memory = memory_store
        self.detector = failure_detector or WebShopFailureDetector()
        self.extractor_llm = extractor_llm
        self.max_steps = max_steps
        self.max_memory_inject = max_memory_inject
        self.enable_memory = enable_memory and (memory_store is not None)
        self.inject_mode = inject_mode
        self.memory_style = memory_style
        self.memory_format = memory_format
        self.cross_env = cross_env

    def run_episode(self, env, env_idx: int = 0) -> dict:
        from prompts.webshop_prompts import build_user_prompt, SYSTEM_PROMPT_FM
        from src.webshop_memory_extractor import extract_failure_recoveries

        t0 = time.time()
        self.llm.tracker.reset()
        if self.detector.judge_llm is not None:
            self.detector.judge_llm.tracker.reset()
        if self.extractor_llm is not None:
            self.extractor_llm.tracker.reset()

        init_obs, task_type, info = env.reset(session_idx=env_idx)

        history: list[tuple[str, str]] = []
        action_history: list[str] = []
        steps: list[dict] = []
        final_reward = 0.0

        current_retrieved = None
        failures_detected = 0
        memories_retrieved_total = 0

        episode_memories = None
        if self.enable_memory and self.inject_mode == "episode":
            all_mem = self.memory.get_all(env_idx=env_idx, cross_env=self.cross_env)
            if all_mem:
                episode_memories = all_mem[:self.max_memory_inject]
                memories_retrieved_total = len(episode_memories)

        for step_num in range(self.max_steps):
            if self.inject_mode == "episode":
                prompt = build_user_prompt(
                    task_type=task_type, task_obs=init_obs, history=history,
                    retrieved_memories=episode_memories, memory_style=self.memory_style,
                )
            else:
                prompt = build_user_prompt(
                    task_type=task_type, task_obs=init_obs, history=history,
                    retrieved_memories=current_retrieved, memory_style=self.memory_style,
                )

            response = self.llm.complete_text(
                prompt, stop=["\n"], label=f"step_{step_num}",
                system=SYSTEM_PROMPT_FM,
            )
            action = response.strip().split("\n")[0].strip()

            if self.inject_mode == "in_loop":
                current_retrieved = None

            if action.startswith("> "):
                action = action[2:]

            # Extract search[...] or click[...] from response
            import re
            search_match = re.search(r'search\[([^\]]+)\]', action)
            click_match = re.search(r'click\[([^\]]+)\]', action)
            if search_match:
                action = f"search[{search_match.group(1)}]"
            elif click_match:
                action = f"click[{click_match.group(1)}]"
            elif not action or not (action.startswith("search[") or action.startswith("click[")):
                # If LLM didn't produce a valid action, default
                if "search" not in action.lower():
                    action = "search[product]"

            logger.info(f"  Step {step_num}: {action[:80]}")

            # Handle think
            is_think = action.startswith("think:") or action.startswith("think ")
            if is_think:
                observation = "OK."
                steps.append({
                    "step": step_num, "action": action, "observation": observation,
                    "is_think": True, "failure_detected": False, "memory_retrieved": 0,
                })
                history.append((action, observation))
                continue

            observation, reward, done, step_info = env.step(action)
            logger.info(f"    obs: {observation[:80]}")
            action_history.append(action)
            final_reward = max(final_reward, reward)

            is_done, _ = self.detector.is_task_complete(observation, done, step_info)

            record = {
                "step": step_num, "action": action, "observation": observation[:200],
                "is_think": False, "failure_detected": False, "memory_retrieved": 0,
            }

            if self.enable_memory and self.inject_mode in ("in_loop", "none") and not is_done:
                det = self.detector.detect(observation, action, action_history)
                if det.is_failure:
                    record["failure_detected"] = True
                    failures_detected += 1
                    logger.info(f"    FAILURE detected: {det.failure_type}")

                    if self.inject_mode == "in_loop":
                        retrieved = self.memory.retrieve(
                            query_action=action, query_observation=observation,
                            task_type=task_type, top_k=self.max_memory_inject,
                            env_idx=env_idx, cross_env=self.cross_env,
                        )
                        record["memory_retrieved"] = len(retrieved)
                        memories_retrieved_total += len(retrieved)
                        if retrieved:
                            current_retrieved = retrieved

            steps.append(record)
            history.append((action, observation))

            if is_done:
                break

        # Post-episode memory extraction
        memories_stored = 0
        if self.enable_memory and self.extractor_llm is not None:
            env_history = [(a, o) for a, o in history if not a.startswith("think")]
            if self.memory_format == "success_trajectory":
                from src.memory_extractor_ablation import extract_success_trajectories
                recoveries = extract_success_trajectories(self.extractor_llm, env_history, final_reward >= 0.5)
            elif self.memory_format == "reflexion_reflection":
                from src.memory_extractor_ablation import extract_reflexion_reflections
                recoveries = extract_reflexion_reflections(self.extractor_llm, env_history)
            else:
                recoveries = extract_failure_recoveries(self.extractor_llm, env_history)
            for rec in recoveries:
                self.memory.add(
                    failure_action=rec["failure_action"],
                    failure_observation=rec["failure_observation"],
                    solution_action=rec["solution_action"],
                    task_type=task_type, env_idx=env_idx,
                )
                memories_stored += 1

        wall_time = time.time() - t0
        agent_stats = self.llm.tracker.summary()
        judge_tokens = 0
        if self.detector.judge_llm is not None:
            judge_tokens = self.detector.judge_llm.tracker.summary()["total_tokens"]
        extractor_tokens = 0
        if self.extractor_llm is not None:
            extractor_tokens = self.extractor_llm.tracker.summary()["total_tokens"]
        total_tokens = agent_stats["total_tokens"] + judge_tokens + extractor_tokens

        result = {
            "env_idx": env_idx,
            "task_type": task_type,
            "task_description": init_obs[:200],
            "success": final_reward >= 0.5,
            "reward": final_reward,
            "total_steps": len(steps),
            "total_tokens": total_tokens,
            "agent_tokens": agent_stats["total_tokens"],
            "judge_tokens": judge_tokens,
            "extractor_tokens": extractor_tokens,
            "failures_detected": failures_detected,
            "memories_retrieved": memories_retrieved_total,
            "memories_stored": memories_stored,
            "wall_time_s": round(wall_time, 2),
            "skipped": False,
            "steps": steps,
        }

        status = "SUCCESS" if final_reward >= 0.5 else "FAIL"
        logger.info(
            f"Env #{env_idx} [{task_type}] {status} reward={final_reward:.4f} in {len(steps)} steps"
        )
        return result


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = args.seed or config["experiment"]["seed"]
    set_seed(seed)
    num_epochs = args.epochs or config["experiment"].get("num_epochs", 1)
    is_baseline = args.baseline
    mode = "react_baseline" if is_baseline else "react_fm"
    run_name = args.run_name or f"ws_{mode}_"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = f"{config['experiment']['results_dir']}/{timestamp}"
    memory_dir = f"{config['memory']['persist_dir']}/{timestamp}" if not is_baseline else None

    setup_logging(
        log_level=config["experiment"]["log_level"],
        log_dir="logs",
        run_name=run_name.rstrip("_"),
    )

    llm = LLMClient(
        model=config["llm"]["model"],
        base_url=config["llm"]["base_url"],
        api_key=config["llm"].get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
    )

    memory_store = None
    if not is_baseline:
        memory_store = FailureMemoryStore(
            embedding_model_name=config["memory"]["embedding_model"],
            max_entries=config["memory"]["max_entries"],
            top_k=config["memory"]["retrieval_top_k"],
            retrieval_mode=args.retrieval_mode,
        )
        if args.resume_memory:
            memory_store.load(args.resume_memory)

    judge_llm = None
    if not is_baseline and "judge" in config:
        judge_cfg = config["judge"]
        judge_llm = LLMClient(
            model=judge_cfg["model"],
            base_url=judge_cfg["base_url"],
            api_key=judge_cfg.get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
            temperature=judge_cfg.get("temperature", 0.0),
            max_tokens=judge_cfg.get("max_tokens", 16),
        )

    detector = WebShopFailureDetector(judge_llm=judge_llm)

    extractor_llm = None
    if not is_baseline and "extractor" in config:
        ext_cfg = config["extractor"]
        extractor_llm = LLMClient(
            model=ext_cfg["model"],
            base_url=ext_cfg["base_url"],
            api_key=ext_cfg.get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
            temperature=ext_cfg.get("temperature", 0.0),
            max_tokens=ext_cfg.get("max_tokens", 512),
        )

    agent = WebShopReActAgent(
        llm=llm,
        memory_store=memory_store,
        failure_detector=detector,
        extractor_llm=extractor_llm,
        max_steps=args.max_steps,
        max_memory_inject=config["agent"]["max_memory_inject"],
        enable_memory=not is_baseline,
        inject_mode=args.inject_mode,
        memory_style=args.memory_style,
        memory_format=args.memory_format,
        cross_env=args.cross_env,
    )

    max_envs = args.max_envs or 200
    env_success: dict[int, float] = {}

    if args.resume_results:
        with open(args.resume_results) as f:
            prev_data = json.load(f)
        for ep in prev_data.get("episodes", []):
            idx = ep["env_idx"]
            if ep.get("reward", 0) >= 0.5:
                env_success[idx] = ep["reward"]

    for epoch in range(1, num_epochs + 1):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  Epoch {epoch}/{num_epochs} — {mode} (WebShop)")
        logger.info("=" * 60)

        env = WebShopEnv(num_products=args.num_products, max_sessions=max_envs)
        env.setup()

        episode_results = []
        env_count = 0
        num_skipped = 0

        while env_count < max_envs:
            try:
                if epoch > 1 and env_count in env_success:
                    env.skip()
                    episode_results.append({
                        "env_idx": env_count,
                        "task_type": "shopping",
                        "task_description": "",
                        "success": True,
                        "reward": env_success[env_count],
                        "total_steps": 0,
                        "total_tokens": 0,
                        "agent_tokens": 0,
                        "judge_tokens": 0,
                        "extractor_tokens": 0,
                        "failures_detected": 0,
                        "memories_retrieved": 0,
                        "memories_stored": 0,
                        "wall_time_s": 0,
                        "skipped": True,
                        "steps": [],
                    })
                    num_skipped += 1
                    env_count += 1
                    continue

                result = agent.run_episode(env, env_idx=env_count)
                episode_results.append(result)

                if result["reward"] >= 0.5:
                    env_success[env_count] = result["reward"]

                env_count += 1

                rewards = [r.get("reward", 0) for r in episode_results]
                avg_r = sum(rewards) / len(rewards)
                logger.info(f"  [{env_count}] reward={result['reward']:.4f} "
                            f"running_avg={avg_r:.4f}")

                if env_count % 10 == 0:
                    mem_stats = memory_store.stats() if memory_store else {}
                    summary = compute_summary(episode_results, mem_stats, mode)
                    save_results(
                        episode_results, summary,
                        f"{results_dir}/{run_name}{epoch}_intermediate.json"
                    )
                    if memory_store:
                        memory_store.save(f"{memory_dir}/ws_epoch{epoch}_intermediate.json")

            except Exception as e:
                logger.error(f"Error on env #{env_count}: {e}", exc_info=True)
                env_count += 1

        if num_skipped > 0:
            logger.info(f"  Skipped {num_skipped} already-succeeded envs")

        mem_stats = memory_store.stats() if memory_store else {}
        summary = compute_summary(episode_results, mem_stats, mode)
        log_summary(summary)

        result_path = f"{results_dir}/{run_name}{epoch}.json"
        save_results(episode_results, summary, result_path)
        logger.info(f"Results saved: {result_path}")

        if memory_store:
            mem_path = f"{memory_dir}/ws_epoch{epoch}.json"
            memory_store.save(mem_path)
            logger.info(f"Memory saved: {mem_path} ({memory_store.size()} entries)")

        env.close()

    logger.info("Done!")


if __name__ == "__main__":
    main()

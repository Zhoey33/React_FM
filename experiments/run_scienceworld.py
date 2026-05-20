"""Run React_FM (or baseline ReAct) on ScienceWorld."""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure Java is on PATH
_JAVA_HOME = "/opt/homebrew/opt/openjdk/bin"
if _JAVA_HOME not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _JAVA_HOME + ":" + os.environ.get("PATH", "")

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
from src.scienceworld_failure_detector import ScienceWorldFailureDetector
from src.scienceworld_env import ScienceWorldEnv, DEFAULT_EVAL_TASKS
from src.log_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run React_FM on ScienceWorld")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--max-envs", type=int, default=None, help="Max environments per epoch")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--baseline", action="store_true",
                        help="Run as pure ReAct (no memory, no failure detection)")
    parser.add_argument("--resume-memory", type=str, default=None, help="Load memory from file")
    parser.add_argument("--resume-results", type=str, default=None, help="Load previous results for skip-on-success")
    parser.add_argument("--run-name", type=str, default=None, help="Custom run name")
    parser.add_argument("--inject-mode", choices=["in_loop", "episode", "none"], default="in_loop")
    parser.add_argument("--memory-style", choices=["original", "factual", "reflexion", "hint"], default="original")
    parser.add_argument("--memory-format", choices=["failure_recovery", "success_trajectory", "reflexion_reflection"],
                        default="failure_recovery", help="Memory storage format for ablation")
    parser.add_argument("--retrieval-mode", choices=["hybrid", "bm25_only", "embedding_only", "random"],
                        default="hybrid", help="Retrieval method for ablation")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test",
                        help="Official ScienceWorld split to run")
    parser.add_argument("--tasks", nargs="+", default=None, help="Specific task names to run")
    parser.add_argument("--max-variations", type=int, default=5, help="Max variations per task")
    parser.add_argument("--step-limit", type=int, default=100, help="Max steps per episode")
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
    successes = sum(1 for r in results if r["success"])

    by_type: dict[str, dict] = {}
    for r in results:
        tt = r["task_type"]
        if tt not in by_type:
            by_type[tt] = {"total": 0, "success": 0}
        by_type[tt]["total"] += 1
        if r["success"]:
            by_type[tt]["success"] += 1

    for v in by_type.values():
        v["rate"] = round(v["success"] / v["total"], 4) if v["total"] > 0 else 0

    # ScienceWorld API reports score as round(100 * getScore()).
    # Full success is 100, but the environment can also return negative scores.
    scores = [r.get("score", 0.0) / 100.0 for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    total_tokens = sum(r["total_tokens"] for r in results)
    agent_tokens = sum(r.get("agent_tokens", r["total_tokens"]) for r in results)
    judge_tokens = sum(r.get("judge_tokens", 0) for r in results)
    extractor_tokens = sum(r.get("extractor_tokens", 0) for r in results)

    return {
        "mode": mode,
        "benchmark": "scienceworld",
        "total_envs": total,
        "total_success": successes,
        "success_rate": round(successes / total, 4) if total > 0 else 0,
        "avg_score": round(avg_score, 4),
        "by_task_type": by_type,
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
    logger.info(f"  {summary['mode']} Results (ScienceWorld)")
    logger.info("=" * 60)
    logger.info(f"  Total: {summary['total_success']}/{summary['total_envs']} "
                f"({summary['success_rate']:.1%})")
    logger.info(f"  Avg Score: {summary['avg_score']:.2%}")
    logger.info("-" * 60)
    logger.info(f"  {'Task Name':<40} {'Success':>8} {'Total':>8} {'Rate':>8}")
    logger.info("-" * 60)
    for tt, stats in sorted(summary["by_task_type"].items()):
        logger.info(f"  {tt:<40} {stats['success']:>8} {stats['total']:>8} {stats['rate']:>7.1%}")
    logger.info("-" * 60)
    logger.info(f"  Tokens: {summary['total_tokens']:,} total, "
                f"{summary['avg_tokens_per_episode']:,} avg/episode "
                f"(agent={summary.get('agent_tokens', 0):,}, "
                f"judge={summary.get('judge_tokens', 0):,}, "
                f"ext={summary.get('extractor_tokens', 0):,})")
    if summary["memory_stats"]:
        ms = summary["memory_stats"]
        logger.info(f"  Memory: {ms.get('total_entries', 0)} entries, "
                    f"{ms.get('total_retrievals', 0)} retrievals, "
                    f"{ms.get('total_hits', 0)} hits")
    logger.info("=" * 60)


import re

# Valid ScienceWorld action prefixes
_SW_ACTION_PREFIXES = (
    "go to", "pick up", "put down", "open", "close", "activate", "deactivate",
    "use", "pour", "mix", "focus on", "wait", "look around", "look", "inventory",
    "examine", "read", "connect", "move", "teleport to", "dunk", "eat", "drink",
    "flush", "reset task", "think:",  "think ",
)


def _extract_action(raw: str) -> str:
    """Extract a valid ScienceWorld action from possibly verbose LLM output."""
    raw = raw.strip()
    if not raw:
        return ""

    # If it already starts with a valid prefix, use as-is
    lower = raw.lower()
    for prefix in _SW_ACTION_PREFIXES:
        if lower.startswith(prefix):
            return raw

    # Try to find a valid action within the text
    for prefix in _SW_ACTION_PREFIXES:
        idx = lower.find(prefix)
        if idx >= 0:
            # Extract from this prefix to end of line or period
            rest = raw[idx:]
            # Take up to first period or end
            end = rest.find(".")
            if end > 0 and end < 80:
                return rest[:end].strip()
            return rest[:80].strip()

    # Fallback: return first 80 chars
    return raw[:80]


def _is_task_complete(done: bool, info: dict) -> tuple[bool, bool]:
    """Check episode termination using ScienceWorld's score convention."""
    score = info.get("score", 0.0)
    return done, score >= 100.0


def _infer_task_name_from_env(env) -> str:
    """Best-effort task name recovery for exception accounting."""
    schedule = getattr(env, "_schedule", None)
    schedule_idx = getattr(env, "_schedule_idx", 0)
    if not schedule:
        return "unknown"
    if schedule_idx > 0 and schedule_idx - 1 < len(schedule):
        return schedule[schedule_idx - 1][0]
    if schedule_idx < len(schedule):
        return schedule[schedule_idx][0]
    return "unknown"


class ScienceWorldReActAgent:
    """ReAct agent adapted for ScienceWorld."""

    def __init__(
        self,
        llm: LLMClient,
        memory_store: FailureMemoryStore | None = None,
        failure_detector: ScienceWorldFailureDetector | None = None,
        extractor_llm: LLMClient | None = None,
        max_steps: int = 50,
        max_memory_inject: int = 3,
        enable_memory: bool = True,
        inject_mode: str = "in_loop",
        memory_style: str = "original",
        memory_format: str = "failure_recovery",
        baseline_mode: bool = False,
    ):
        self.llm = llm
        self.memory = memory_store
        self.baseline_mode = baseline_mode
        self.detector = None if baseline_mode else (failure_detector or ScienceWorldFailureDetector())
        self.extractor_llm = extractor_llm
        self.max_steps = max_steps
        self.max_memory_inject = max_memory_inject
        self.enable_memory = enable_memory and (memory_store is not None)
        self.inject_mode = inject_mode
        self.memory_style = memory_style
        self.memory_format = memory_format

    def run_episode(self, env, env_idx: int = 0) -> dict:
        """Run one ScienceWorld episode. Returns result dict."""
        from prompts.scienceworld_prompts import (
            build_baseline_user_prompt,
            build_user_prompt,
            SYSTEM_PROMPT_BASE,
            SYSTEM_PROMPT_FM,
        )
        from src.scienceworld_memory_extractor import extract_failure_recoveries

        t0 = time.time()
        self.llm.tracker.reset()
        if self.detector is not None and self.detector.judge_llm is not None:
            self.detector.judge_llm.tracker.reset()
        if self.extractor_llm is not None:
            self.extractor_llm.tracker.reset()

        init_obs, task_type, info = env.reset()

        history: list[tuple[str, str]] = []
        action_history: list[str] = []
        steps: list[dict] = []
        env_history_records: list[dict] = []
        success = False
        final_score = 0.0

        current_retrieved = None
        failures_detected = 0
        memories_retrieved_total = 0
        consecutive_thinks = 0
        max_consecutive_thinks = 3

        episode_memories = None
        if self.enable_memory and self.inject_mode == "episode":
            all_mem = self.memory.get_all(env_idx=env_idx, task_type=task_type)
            if all_mem:
                episode_memories = all_mem[:self.max_memory_inject]
                memories_retrieved_total = len(episode_memories)

        for step_num in range(self.max_steps):
            if self.baseline_mode:
                prompt = build_baseline_user_prompt(
                    task_type=task_type,
                    task_obs=init_obs,
                    history=history,
                )
                system_prompt = SYSTEM_PROMPT_BASE
            elif self.inject_mode == "episode":
                prompt = build_user_prompt(
                    task_type=task_type, task_obs=init_obs, history=history,
                    retrieved_memories=episode_memories, memory_style=self.memory_style,
                )
                system_prompt = SYSTEM_PROMPT_FM
            else:
                prompt = build_user_prompt(
                    task_type=task_type, task_obs=init_obs, history=history,
                    retrieved_memories=current_retrieved, memory_style=self.memory_style,
                )
                system_prompt = SYSTEM_PROMPT_FM

            response = self.llm.complete_text(
                prompt, stop=["\n"], label=f"step_{step_num}",
                system=system_prompt,
            )
            action = response.strip().split("\n")[0].strip()

            if self.inject_mode == "in_loop":
                current_retrieved = None

            if action.startswith("> "):
                action = action[2:]

            # Post-process: extract valid action from verbose LLM output
            action = _extract_action(action)
            if not action:
                action = "look around"

            logger.info(f"  Step {step_num}: {action[:80]}")

            is_think = action.startswith("think:") or action.startswith("think ")
            if is_think:
                consecutive_thinks += 1
                if consecutive_thinks > max_consecutive_thinks:
                    # Force an action to break think loops
                    action = "look around"
                    is_think = False
                    consecutive_thinks = 0
                else:
                    observation = "OK."
                    steps.append({
                        "step": step_num, "action": action, "observation": observation,
                        "is_think": True, "failure_detected": False, "failure_type": "",
                        "memory_retrieved": 0,
                    })
                    history.append((action, observation))
                    continue
            else:
                consecutive_thinks = 0

            observation, reward, done, step_info = env.step(action)
            logger.info(f"    obs: {observation[:80]}")
            action_history.append(action)
            env_history_records.append(
                {"step": step_num, "action": action, "observation": observation}
            )

            final_score = step_info.get("score", final_score)
            is_done, is_success = _is_task_complete(done, step_info)

            record = {
                "step": step_num, "action": action, "observation": observation[:200],
                "is_think": False, "failure_detected": False, "failure_type": "",
                "memory_retrieved": 0,
            }

            if (
                self.detector is not None
                and self.enable_memory
                and self.inject_mode in ("in_loop", "none")
                and not is_done
            ):
                det = self.detector.detect(observation, action, action_history)
                if det.is_failure:
                    record["failure_detected"] = True
                    record["failure_type"] = det.failure_type
                    failures_detected += 1
                    logger.info(f"    FAILURE detected: {det.failure_type}")

                    if self.inject_mode == "in_loop":
                        retrieved = self.memory.retrieve(
                            query_action=action, query_observation=observation,
                            task_type=task_type, top_k=self.max_memory_inject, env_idx=env_idx,
                        )
                        record["memory_retrieved"] = len(retrieved)
                        memories_retrieved_total += len(retrieved)
                        if retrieved:
                            current_retrieved = retrieved
                            logger.info(f"    Retrieved {len(retrieved)} memories")

            steps.append(record)
            history.append((action, observation))

            if is_done:
                success = is_success
                break

        # Post-episode: extract and store memories
        memories_stored = 0
        if self.enable_memory and self.extractor_llm is not None:
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
            if self.memory_format == "success_trajectory":
                from src.memory_extractor_ablation import extract_success_trajectories
                recoveries = extract_success_trajectories(self.extractor_llm, env_history, success)
            elif self.memory_format == "reflexion_reflection":
                from src.memory_extractor_ablation import extract_reflexion_reflections
                recoveries = extract_reflexion_reflections(self.extractor_llm, env_history)
            else:
                recoveries = extract_failure_recoveries(
                    self.extractor_llm,
                    env_history_records,
                    detected_failures=detected_failures,
                )
            for rec in recoveries:
                self.memory.add(
                    failure_action=rec["failure_action"],
                    failure_observation=rec["failure_observation"],
                    solution_action=rec["solution_action"],
                    task_type=task_type, env_idx=env_idx,
                    question_text=rec.get("question_text", ""),
                    repair_strategy=rec.get("repair_strategy", ""),
                    repair_tactic=rec.get("repair_tactic", ""),
                    repair_action=rec.get("repair_action", ""),
                )
                memories_stored += 1

        wall_time = time.time() - t0
        agent_stats = self.llm.tracker.summary()
        judge_tokens = 0
        if self.detector is not None and self.detector.judge_llm is not None:
            judge_tokens = self.detector.judge_llm.tracker.summary()["total_tokens"]
        extractor_tokens = 0
        if self.extractor_llm is not None:
            extractor_tokens = self.extractor_llm.tracker.summary()["total_tokens"]
        total_tokens = agent_stats["total_tokens"] + judge_tokens + extractor_tokens

        result = {
            "env_idx": env_idx,
            "task_type": task_type,
            "task_description": init_obs[:200],
            "success": success,
            "score": final_score,
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

        status = "SUCCESS" if success else "FAIL"
        logger.info(
            f"Env #{env_idx} [{task_type}] {status} score={final_score:.2f} in {len(steps)} steps, "
            f"tokens={total_tokens}, failures={failures_detected}, mem={memories_retrieved_total}/{memories_stored}"
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
    run_name = args.run_name or f"sw_{mode}_"

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
            scope="task_type",
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

    detector = None if is_baseline else ScienceWorldFailureDetector(judge_llm=judge_llm)

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

    agent = ScienceWorldReActAgent(
        llm=llm,
        memory_store=memory_store,
        failure_detector=detector,
        extractor_llm=extractor_llm,
        max_steps=min(args.step_limit, config["agent"]["max_steps"]),
        max_memory_inject=config["agent"]["max_memory_inject"],
        enable_memory=not is_baseline,
        inject_mode=args.inject_mode,
        memory_style=args.memory_style,
        memory_format=args.memory_format,
        baseline_mode=is_baseline,
    )

    task_names = args.tasks or DEFAULT_EVAL_TASKS

    # Track per-env success across epochs: env_idx -> (task_type, score)
    env_success: dict[int, tuple[str, float]] = {}

    if args.resume_results:
        with open(args.resume_results) as f:
            prev_data = json.load(f)
        for ep in prev_data.get("episodes", []):
            idx = ep["env_idx"]
            if ep.get("success"):
                env_success[idx] = (ep.get("task_type", "unknown"), ep.get("score", 100))
        logger.info(f"Loaded {len(env_success)} succeeded envs from {args.resume_results}")

    for epoch in range(1, num_epochs + 1):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  Epoch {epoch}/{num_epochs} — {mode} (ScienceWorld)")
        logger.info("=" * 60)

        env = ScienceWorldEnv(
            task_names=task_names,
            split=args.split,
            max_variations_per_task=args.max_variations,
            env_step_limit=args.step_limit,
        )
        env.setup()

        episode_results = []
        max_envs = args.max_envs or env.total_episodes
        env_count = 0
        num_skipped = 0

        while env_count < max_envs:
            try:
                # Skip already-succeeded envs in later epochs
                if epoch > 1 and (env_count + 1) in env_success:
                    env.skip()
                    prev_task_type, prev_score = env_success[env_count + 1]
                    episode_results.append({
                        "env_idx": env_count + 1,
                        "task_type": prev_task_type,
                        "task_description": "",
                        "success": True,
                        "score": prev_score,
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
                    logger.info(f"  [{env_count}] SKIP (already succeeded)")
                    continue

                result = agent.run_episode(env, env_idx=env_count + 1)
                episode_results.append(result)

                if result["success"]:
                    env_success[env_count + 1] = (result["task_type"], result.get("score", 100))

                env_count += 1

                successes = sum(1 for r in episode_results if r["success"])
                rate = successes / len(episode_results)
                logger.info(f"  [{env_count}] {result['task_type']:<30} "
                            f"{'OK' if result['success'] else 'FAIL':>4} "
                            f"score={result['score']:.2f} "
                            f"steps={result['total_steps']:<3} "
                            f"running={rate:.1%} ({successes}/{env_count})")

                # Intermediate save every 10 envs
                if env_count % 10 == 0:
                    mem_stats = memory_store.stats() if memory_store else {}
                    summary = compute_summary(episode_results, mem_stats, mode)
                    save_results(
                        episode_results, summary,
                        f"{results_dir}/{run_name}{epoch}_intermediate.json"
                    )
                    if memory_store:
                        memory_store.save(f"{memory_dir}/sw_epoch{epoch}_intermediate.json")

            except StopIteration:
                break
            except Exception as e:
                logger.error(f"Error on env #{env_count + 1}: {e}", exc_info=True)
                episode_results.append({
                    "env_idx": env_count + 1,
                    "task_type": _infer_task_name_from_env(env),
                    "task_description": "",
                    "success": False,
                    "score": -100.0,
                    "total_steps": 0,
                    "total_tokens": 0,
                    "agent_tokens": 0,
                    "judge_tokens": 0,
                    "extractor_tokens": 0,
                    "failures_detected": 0,
                    "memories_retrieved": 0,
                    "memories_stored": 0,
                    "wall_time_s": 0,
                    "skipped": False,
                    "steps": [],
                })
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
            mem_path = f"{memory_dir}/sw_epoch{epoch}.json"
            memory_store.save(mem_path)
            logger.info(f"Memory saved: {mem_path} ({memory_store.size()} entries)")

        # Close env to free JVM
        env.close()

    logger.info("Done!")


if __name__ == "__main__":
    main()

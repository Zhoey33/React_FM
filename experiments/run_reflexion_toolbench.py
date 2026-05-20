"""Run Reflexion baseline on ToolBench (StableToolBench).

Reflexion (Shinn et al., 2023) flow:
  Trial 0: Run ReAct, collect trajectories
  Post-trial: Generate free-text reflections for failed episodes
  Trial 1: Inject reflections at episode start, run ReAct
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import logging
import re
import sys
import random
import time
from pathlib import Path

import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import LLMClient
from src.toolbench_env import ToolBenchEnv
from src.reflexion_agent import generate_reflection
from src.log_utils import setup_logging
from prompts.toolbench_prompts import FEWSHOT_EXAMPLE, SYSTEM_PROMPT_BASE

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Reflexion on ToolBench")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--num-trials", type=int, default=2)
    parser.add_argument("--max-envs", type=int, default=765)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="tb_reflexion_")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to intermediate JSON to resume from")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def save_results(results, summary, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump({"summary": summary, "episodes": results}, f, indent=2, ensure_ascii=False)


def compute_summary(results, mode):
    total = len(results)
    successes = sum(1 for r in results if r.get("success"))
    total_tokens = sum(r.get("total_tokens", 0) for r in results)
    by_subset = {}
    for r in results:
        s = r.get("subset", "unknown")
        if s not in by_subset:
            by_subset[s] = {"total": 0, "success": 0}
        by_subset[s]["total"] += 1
        if r.get("success"):
            by_subset[s]["success"] += 1
    for v in by_subset.values():
        v["rate"] = round(v["success"] / v["total"], 4) if v["total"] > 0 else 0
    return {
        "mode": mode,
        "benchmark": "toolbench",
        "total_envs": total,
        "total_success": successes,
        "success_rate": round(successes / total, 4) if total > 0 else 0,
        "by_subset": by_subset,
        "total_tokens": total_tokens,
        "avg_tokens_per_episode": round(total_tokens / total) if total > 0 else 0,
    }


def _parse_thought_action(response: str) -> tuple[str, str]:
    thought = ""
    action = ""
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.lower().startswith("thought:"):
            thought = line[len("thought:"):].strip()
        elif line.lower().startswith("action:"):
            action = line[len("action:"):].strip()
    if not action:
        m = re.search(r'(\w+)\(\{.*?\}\)', response, re.DOTALL)
        if m:
            action = m.group(0)
        else:
            m = re.search(r'Finish\(.+?\)', response, re.DOTALL)
            if m:
                action = m.group(0)
    return thought, action


def build_reflexion_prompt(task_obs, history, memory=None):
    """Build prompt with Reflexion memory injected at episode start."""
    sections = [FEWSHOT_EXAMPLE]
    if memory:
        lines = ["Your memory for the task below:"]
        for i, m in enumerate(memory):
            lines.append(f"Trial {i}:\n{m.strip()}")
        sections.append("\n".join(lines))
    sections.append(task_obs)
    prompt = "\n\n".join(sections) + "\n"
    for thought, action, observation in history:
        if thought:
            prompt += f"Thought: {thought}\n"
        prompt += f"Action: {action}\nObservation: {observation}\n"
    return prompt


def run_episode(llm, env, env_idx, memory, max_steps):
    """Run one ToolBench episode with optional Reflexion memory."""
    t0 = time.time()
    llm.tracker.reset()

    init_obs, task_type, info = env.reset()

    history = []
    action_history = []
    steps = []
    final_answer = ""
    success = False

    for step_num in range(max_steps):
        prompt = build_reflexion_prompt(init_obs, history, memory)
        response = llm.complete_text(prompt, label=f"step_{step_num}",
                                     system=SYSTEM_PROMPT_BASE)
        thought, action = _parse_thought_action(response)
        if not action:
            action = 'Finish({"return_type": "give_up_and_restart", "final_answer": ""})'

        logger.info(f"  Step {step_num}: {action[:80]}")

        observation, reward, done, step_info = env.step(action)
        logger.info(f"    obs: {observation[:80]}")
        action_history.append(action)

        steps.append({"step": step_num, "thought": thought, "action": action,
                       "observation": observation[:300]})
        history.append((thought, action, observation))

        if done:
            final_answer = step_info.get("final_answer", "")
            success = step_info.get("return_type", "") == "give_answer" and len(final_answer) > 0
            break

    wall_time = time.time() - t0
    agent_stats = llm.tracker.summary()
    log_str = build_reflexion_prompt(init_obs, history, memory)

    return {
        "env_idx": env_idx,
        "subset": info.get("subset", ""),
        "query_id": info.get("query_id", ""),
        "query": info["query"][:200],
        "final_answer": final_answer[:200],
        "success": success,
        "total_steps": len(steps),
        "total_tokens": agent_stats["total_tokens"],
        "wall_time_s": round(wall_time, 2),
        "skipped": False,
        "log_str": log_str,
    }


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = args.seed or config["experiment"]["seed"]
    set_seed(seed)

    # Resume: reuse existing results_dir; fresh: create new timestamp
    if args.resume:
        results_dir = str(Path(args.resume).parent)
        logger.info(f"Resuming from {args.resume}")
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_dir = f"{config['experiment']['results_dir']}/{timestamp}"

    setup_logging(
        log_level=config["experiment"]["log_level"],
        log_dir="logs",
        run_name=args.run_name.rstrip("_"),
    )

    llm = LLMClient(
        model=config["llm"]["model"],
        base_url=config["llm"]["base_url"],
        api_key=config["llm"].get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
    )

    max_envs = args.max_envs
    env_configs = {}

    # Load resume state
    resume_trial = 0
    resume_env_count = 0
    resume_logs = []
    if args.resume:
        with open(args.resume) as f:
            data = json.load(f)
        resume_logs = data.get("episodes", [])
        resume_env_count = len(resume_logs)
        # Reconstruct env_configs from loaded results
        for ep in resume_logs:
            eidx = ep["env_idx"]
            if eidx not in env_configs:
                env_configs[eidx] = {"memory": [], "is_success": False}
            if ep.get("success"):
                env_configs[eidx]["is_success"] = True
        logger.info(f"Resumed {resume_env_count} episodes from checkpoint")

    for trial_idx in range(args.num_trials):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  Trial {trial_idx}/{args.num_trials} — Reflexion (ToolBench)")
        logger.info("=" * 60)

        env = ToolBenchEnv(subsets=args.subsets, max_queries=max_envs)
        env.setup()

        trial_logs = []
        env_count = 0
        num_success = 0

        # If resuming trial 0, fast-forward env and reload prior results
        skip_to = 0
        if trial_idx == resume_trial and resume_env_count > 0 and args.resume:
            skip_to = resume_env_count
            trial_logs = resume_logs
            num_success = sum(1 for r in resume_logs if r.get("success"))
            # Fast-forward env to skip_to
            for _ in range(skip_to):
                env.skip()
            env_count = skip_to
            logger.info(f"  Fast-forwarded to env #{skip_to} ({num_success} successes)")
            # Clear resume so it doesn't apply to trial 1
            args.resume = None
            resume_env_count = 0
            resume_logs = []

        while env_count < max_envs:
            try:
                if env_count not in env_configs:
                    env_configs[env_count] = {"memory": [], "is_success": False}

                if env_configs[env_count]["is_success"]:
                    env.skip()
                    num_success += 1
                    trial_logs.append({
                        "env_idx": env_count, "subset": "",
                        "query_id": "", "query": "", "final_answer": "",
                        "success": True, "total_steps": 0, "total_tokens": 0,
                        "wall_time_s": 0, "skipped": True,
                    })
                    logger.info(f"  [{env_count + 1}] SKIP (already succeeded)")
                    env_count += 1
                    continue

                result = run_episode(
                    llm, env, env_count,
                    env_configs[env_count]["memory"], args.max_steps,
                )
                trial_logs.append(result)

                if result["success"]:
                    env_configs[env_count]["is_success"] = True
                    num_success += 1

                env_count += 1
                rate = num_success / env_count
                status = "OK" if result["success"] else "FAIL"
                logger.info(f"  [{env_count}] {status} running={rate:.1%}")

                if env_count % 10 == 0:
                    summary = compute_summary(trial_logs, "reflexion")
                    save_results(trial_logs, summary,
                                 f"{results_dir}/{args.run_name}trial{trial_idx}_intermediate.json")

            except StopIteration:
                break
            except Exception as e:
                logger.error(f"Error on env #{env_count}: {e}", exc_info=True)
                env_count += 1

        # Generate reflections for failed envs
        if trial_idx < args.num_trials - 1:
            logger.info("Generating reflections for failed episodes...")
            num_reflected = 0
            for log in trial_logs:
                eidx = log["env_idx"]
                if not env_configs[eidx]["is_success"] and not log.get("skipped"):
                    log_str = log.get("log_str", "")
                    if log_str:
                        reflection = generate_reflection(
                            llm, log_str, env_configs[eidx]["memory"],
                            domain="toolbench",
                        )
                        env_configs[eidx]["memory"].append(reflection)
                        num_reflected += 1
            logger.info(f"Generated {num_reflected} reflections")

        summary = compute_summary(trial_logs, "reflexion")
        logger.info(f"  Trial {trial_idx}: {summary['total_success']}/{summary['total_envs']} "
                    f"({summary['success_rate']:.1%})")
        for s, stats in sorted(summary.get("by_subset", {}).items()):
            logger.info(f"    {s:<20} {stats['success']}/{stats['total']} ({stats['rate']:.1%})")

        for log in trial_logs:
            log.pop("log_str", None)
        result_path = f"{results_dir}/{args.run_name}trial{trial_idx}.json"
        save_results(trial_logs, summary, result_path)
        logger.info(f"Results saved: {result_path}")

        env.close()

    logger.info("Done!")


if __name__ == "__main__":
    main()

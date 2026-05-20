"""Run Reflexion baseline on WebShop.

Reflexion (Shinn et al., 2023) flow:
  Trial 0: Run ReAct, collect trajectories
  Post-trial: Generate free-text reflections for failed sessions
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
from src.webshop_env import WebShopEnv
from src.reflexion_agent import generate_reflection
from src.log_utils import setup_logging
from prompts.webshop_prompts import FEWSHOT_EXAMPLE, SYSTEM_PROMPT_BASE

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Reflexion on WebShop")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--num-trials", type=int, default=2)
    parser.add_argument("--max-envs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="ws_reflexion_")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--num-products", type=int, default=1000)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def build_reflexion_prompt(task_obs, history, memory=None):
    """Build prompt with Reflexion memory injected at episode start."""
    sections = [FEWSHOT_EXAMPLE]
    if memory:
        lines = ["Your memory for the task below:"]
        for i, m in enumerate(memory):
            lines.append(f"Trial {i}:\n{m.strip()}")
        sections.append("\n".join(lines))
    sections.append("Here is the task.\n" + task_obs)
    prompt = "\n\n".join(sections) + "\n"
    for action, obs in history:
        prompt += f"> {action}\n{obs}\n"
    prompt += "> "
    return prompt


def run_episode(llm, env, env_idx, memory, max_steps):
    """Run one WebShop episode with optional Reflexion memory."""
    t0 = time.time()
    llm.tracker.reset()
    init_obs, task_type, info = env.reset(session_idx=env_idx)

    history = []
    action_history = []
    steps = []
    final_reward = 0.0

    for step_num in range(max_steps):
        prompt = build_reflexion_prompt(init_obs, history, memory)
        response = llm.complete_text(prompt, stop=["\n"], label=f"step_{step_num}",
                                     system=SYSTEM_PROMPT_BASE)
        action = response.strip().split("\n")[0].strip()
        if action.startswith("> "):
            action = action[2:]

        # Extract search[...] or click[...]
        search_match = re.search(r'search\[([^\]]+)\]', action)
        click_match = re.search(r'click\[([^\]]+)\]', action)
        if search_match:
            action = f"search[{search_match.group(1)}]"
        elif click_match:
            action = f"click[{click_match.group(1)}]"
        elif not action or not (action.startswith("search[") or action.startswith("click[")):
            action = "search[product]"

        logger.info(f"  Step {step_num}: {action[:80]}")

        if action.startswith("think:") or action.startswith("think "):
            steps.append({"step": step_num, "action": action, "observation": "OK."})
            history.append((action, "OK."))
            continue

        observation, reward, done, step_info = env.step(action)
        logger.info(f"    obs: {observation[:80]}")
        action_history.append(action)
        final_reward = max(final_reward, reward)

        steps.append({"step": step_num, "action": action, "observation": observation[:200]})
        history.append((action, observation))

        if done:
            break

    wall_time = time.time() - t0
    agent_stats = llm.tracker.summary()

    # Build log string for reflection generation
    log_str = build_reflexion_prompt(init_obs, history, memory)

    return {
        "env_idx": env_idx,
        "task_type": task_type,
        "task_description": init_obs[:200],
        "success": final_reward >= 0.5,
        "reward": final_reward,
        "total_steps": len(steps),
        "total_tokens": agent_stats["total_tokens"],
        "wall_time_s": round(wall_time, 2),
        "skipped": False,
        "steps": steps,
        "log_str": log_str,
    }


def save_results(results, summary, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump({"summary": summary, "episodes": results}, f, indent=2, ensure_ascii=False)


def compute_summary(results, mode):
    total = len(results)
    rewards = [r.get("reward", 0.0) for r in results]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    successes = sum(1 for r in rewards if r >= 0.5)
    total_tokens = sum(r.get("total_tokens", 0) for r in results)
    return {
        "mode": mode,
        "benchmark": "webshop",
        "total_envs": total,
        "total_success": successes,
        "success_rate": round(successes / total, 4) if total > 0 else 0,
        "avg_reward": round(avg_reward, 4),
        "total_tokens": total_tokens,
        "avg_tokens_per_episode": round(total_tokens / total) if total > 0 else 0,
    }


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = args.seed or config["experiment"]["seed"]
    set_seed(seed)

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

    # Per-session memory
    env_configs = [{"memory": [], "is_success": False, "reward": 0.0} for _ in range(args.max_envs)]

    for trial_idx in range(args.num_trials):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  Trial {trial_idx}/{args.num_trials} — Reflexion (WebShop)")
        logger.info("=" * 60)

        env = WebShopEnv(num_products=args.num_products, max_sessions=args.max_envs)
        env.setup()

        trial_logs = []
        num_success = 0

        for z in range(args.max_envs):
            if env_configs[z]["is_success"]:
                env.skip()
                num_success += 1
                trial_logs.append({
                    "env_idx": z, "task_type": "shopping", "success": True,
                    "reward": env_configs[z]["reward"], "skipped": True,
                    "total_tokens": 0, "total_steps": 0, "wall_time_s": 0,
                })
                logger.info(f"  [{z+1}] SKIP (already succeeded)")
                continue

            try:
                result = run_episode(llm, env, z, env_configs[z]["memory"], args.max_steps)
                trial_logs.append(result)

                if result["success"]:
                    env_configs[z]["is_success"] = True
                    env_configs[z]["reward"] = result["reward"]
                    num_success += 1

                rate = num_success / (z + 1)
                status = "OK" if result["success"] else "FAIL"
                logger.info(f"  [{z+1}] {status:>4} reward={result['reward']:.4f} "
                            f"running={rate:.1%} ({num_success}/{z+1})")
            except Exception as e:
                logger.error(f"Error on session #{z}: {e}", exc_info=True)
                trial_logs.append({
                    "env_idx": z, "task_type": "shopping", "success": False,
                    "reward": 0.0, "skipped": False, "total_tokens": 0,
                    "total_steps": 0, "wall_time_s": 0,
                })

        # Generate reflections for failed sessions
        if trial_idx < args.num_trials - 1:
            logger.info("Generating reflections for failed sessions...")
            num_reflected = 0
            for z in range(args.max_envs):
                if not env_configs[z]["is_success"] and not trial_logs[z].get("skipped"):
                    log_str = trial_logs[z].get("log_str", "")
                    if log_str:
                        reflection = generate_reflection(
                            llm, log_str, env_configs[z]["memory"], domain="webshop",
                        )
                        env_configs[z]["memory"].append(reflection)
                        num_reflected += 1
            logger.info(f"Generated {num_reflected} reflections")

        # Trial summary
        summary = compute_summary(trial_logs, "reflexion")
        logger.info(f"  Trial {trial_idx}: {summary['total_success']}/{summary['total_envs']} "
                    f"({summary['success_rate']:.1%}), avg_reward={summary['avg_reward']:.4f}")

        # Save (remove log_str to save space)
        for log in trial_logs:
            log.pop("log_str", None)
            log.pop("steps", None)
        result_path = f"{results_dir}/{args.run_name}trial{trial_idx}.json"
        save_results(trial_logs, summary, result_path)
        logger.info(f"Results saved: {result_path}")

        env.close()

    logger.info("Done!")


if __name__ == "__main__":
    main()

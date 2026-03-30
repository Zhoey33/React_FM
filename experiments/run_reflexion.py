"""Run Reflexion on ALFWorld — reproduced from github.com/noahshinn/reflexion."""

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

from prompts.alfworld_prompts import build_fewshot_prefix
from src.llm import LLMClient
from src.alfworld_env import ALFWorldEnv, get_task_type
from src.reflexion_agent import (
    PREFIXES,
    run_reflexion_episode,
    generate_reflection,
)
from src.log_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Reflexion on ALFWorld")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--num-trials", type=int, default=2, help="Number of trials (retry rounds)")
    parser.add_argument("--max-envs", type=int, default=134, help="Number of environments")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--run-name", type=str, default="reflexion_", help="Custom run name")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = args.seed or config["experiment"]["seed"]
    random.seed(seed)
    np.random.seed(seed)

    # Timestamp-based output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = f"{config['experiment']['results_dir']}/{timestamp}"

    # Logging: console (INFO) + file (DEBUG)
    setup_logging(
        log_level=config["experiment"]["log_level"],
        log_dir="logs",
        run_name=args.run_name.rstrip("_"),
    )

    # Initialize LLM (same as agent LLM)
    llm = LLMClient(
        model=config["llm"]["model"],
        base_url=config["llm"]["base_url"],
        api_key=config["llm"].get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
    )

    # Initialize env configs (same structure as Reflexion repo)
    env_configs = []
    for i in range(args.max_envs):
        env_configs.append({
            "name": f"env_{i}",
            "memory": [],
            "is_success": False,
        })

    # Run trials
    for trial_idx in range(args.num_trials):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  Trial {trial_idx}/{args.num_trials} — Reflexion")
        logger.info("=" * 60)

        env = ALFWorldEnv(split=config["alfworld"]["split"])
        env.setup()

        trial_logs = []
        num_success = 0

        for z in range(args.max_envs):
            # Reset to next env
            ob, task_type, info = env.reset()
            gamefile = info.get("extra.gamefile", [""])[0] if isinstance(info.get("extra.gamefile"), list) else info.get("extra.gamefile", "")

            # Get task prefix for few-shot selection
            name = "/".join(gamefile.split("/")[-3:-1])
            task_prefix = None
            for k, v in PREFIXES.items():
                if name.startswith(k):
                    task_prefix = v
                    break
            if task_prefix is None:
                task_prefix = task_type

            # Skip already-succeeded envs (core Reflexion mechanism)
            if env_configs[z]["is_success"]:
                num_success += 1
                trial_logs.append({"env_idx": z, "task_type": task_prefix, "success": True, "skipped": True})
                logger.info(f"  [{z+1}] {task_prefix:<8} SKIP (already succeeded)")
                continue

            # Build base prompt with 2 few-shot examples
            base_prompt = build_fewshot_prefix(task_prefix)

            # Run episode with memory from past trials
            t0 = time.time()
            log_str, is_success = run_reflexion_episode(
                env=env,
                llm=llm,
                base_prompt=base_prompt,
                ob=ob,
                memory=env_configs[z]["memory"],
                max_steps=config["agent"]["max_steps"],
            )
            wall_time = time.time() - t0

            if is_success:
                env_configs[z]["is_success"] = True
                num_success += 1

            trial_logs.append({
                "env_idx": z,
                "task_type": task_prefix,
                "success": is_success,
                "skipped": False,
                "wall_time_s": round(wall_time, 2),
                "log": log_str,  # full log for reflection generation
            })

            rate = num_success / (z + 1)
            status = "OK" if is_success else "FAIL"
            logger.info(f"  [{z+1}] {task_prefix:<8} {status:>4} running={rate:.1%} ({num_success}/{z+1})")

        # Generate reflections for failed envs
        if trial_idx < args.num_trials - 1:  # no need to reflect after last trial
            logger.info("Generating reflections for failed envs...")
            num_reflected = 0
            for z in range(args.max_envs):
                if not env_configs[z]["is_success"] and not trial_logs[z].get("skipped"):
                    log_str = trial_logs[z].get("log", "")
                    if log_str:
                        reflection = generate_reflection(
                            llm, log_str, env_configs[z]["memory"]
                        )
                        env_configs[z]["memory"].append(reflection)
                        num_reflected += 1
            logger.info(f"Generated {num_reflected} reflections")

        # Trial summary
        total = args.max_envs
        by_type = {}
        for log in trial_logs:
            tt = log["task_type"]
            if tt not in by_type:
                by_type[tt] = {"total": 0, "success": 0}
            by_type[tt]["total"] += 1
            if log["success"]:
                by_type[tt]["success"] += 1

        token_stats = llm.tracker.summary()

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  Trial {trial_idx} Results: {num_success}/{total} ({num_success/total:.1%})")
        logger.info("-" * 60)
        logger.info(f"  {'Task Type':<12} {'Success':>8} {'Total':>8} {'Rate':>8}")
        logger.info("-" * 60)
        for tt, stats in sorted(by_type.items()):
            rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
            logger.info(f"  {tt:<12} {stats['success']:>8} {stats['total']:>8} {rate:>7.1%}")
        logger.info("-" * 60)
        logger.info(f"  Tokens: {token_stats['total_tokens']:,} total")
        logger.info("=" * 60)

        # Save trial results
        trial_result = {
            "trial": trial_idx,
            "total_envs": total,
            "total_success": num_success,
            "success_rate": round(num_success / total, 4),
            "by_task_type": {tt: {**s, "rate": round(s["success"]/s["total"], 4) if s["total"] > 0 else 0} for tt, s in by_type.items()},
            "total_tokens": token_stats["total_tokens"],
            "episodes": trial_logs,
        }

        Path(results_dir).mkdir(parents=True, exist_ok=True)
        result_path = f"{results_dir}/{args.run_name}trial{trial_idx}.json"
        with open(result_path, "w") as f:
            json.dump(trial_result, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved: {result_path}")

    # Save env configs (with memory) for analysis
    config_path = f"{results_dir}/{args.run_name}env_configs.json"
    with open(config_path, "w") as f:
        json.dump(env_configs, f, indent=2, ensure_ascii=False)
    logger.info(f"Env configs saved: {config_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()

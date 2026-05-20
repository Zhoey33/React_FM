"""Run ExpeL baseline on WebShop.

ExpeL (Zhao et al., 2024) flow:
  Epoch 1: Run ReAct, collect trajectories
  Post-E1: Extract cross-task insights from success/failure pairs
  Epoch 2: Inject relevant insights at episode start, run ReAct
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
from src.expel_memory import InsightMemoryStore
from src.expel_extractor import extract_insights_from_pair, extract_insights_from_failure
from src.webshop_env import WebShopEnv
from src.log_utils import setup_logging
from prompts.webshop_prompts import FEWSHOT_EXAMPLE, SYSTEM_PROMPT_BASE

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ExpeL on WebShop")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-envs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="ws_expel_")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--max-insights-inject", type=int, default=5)
    parser.add_argument("--num-products", type=int, default=1000)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def save_results(results, summary, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump({"summary": summary, "episodes": results}, f, indent=2, ensure_ascii=False)


def compute_summary(results, insight_stats, mode):
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
        "insight_stats": insight_stats,
    }


def build_expel_prompt(task_obs, history, insights=None):
    """Build prompt with ExpeL insights injected at episode start."""
    sections = [FEWSHOT_EXAMPLE]
    if insights:
        lines = ["Based on past experience, keep these rules in mind:"]
        for entry in insights:
            lines.append(f"- {entry.rule}")
        sections.append("\n".join(lines))
    sections.append("Here is the task.\n" + task_obs)
    prompt = "\n\n".join(sections) + "\n"
    for action, obs in history:
        prompt += f"> {action}\n{obs}\n"
    prompt += "> "
    return prompt


def run_episode(llm, env, env_idx, insight_store, max_steps, max_inject):
    """Run one WebShop episode with optional ExpeL insight injection."""
    t0 = time.time()
    llm.tracker.reset()
    init_obs, task_type, info = env.reset(session_idx=env_idx)

    insights = None
    if insight_store and insight_store.size() > 0:
        insights = insight_store.retrieve(task_type, top_k=max_inject)
        if insights:
            logger.info(f"  ExpeL: injecting {len(insights)} insights")

    history = []
    action_history = []
    steps = []
    final_reward = 0.0

    for step_num in range(max_steps):
        prompt = build_expel_prompt(init_obs, history, insights)
        response = llm.complete_text(prompt, stop=["\n"], label=f"step_{step_num}",
                                     system=SYSTEM_PROMPT_BASE)
        action = response.strip().split("\n")[0].strip()
        if action.startswith("> "):
            action = action[2:]

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
    env_history = [(a, o) for a, o in history if not a.startswith("think")]

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
        "history": env_history,
    }


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = args.seed or config["experiment"]["seed"]
    set_seed(seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = f"{config['experiment']['results_dir']}/{timestamp}"
    insight_dir = f"{config['memory']['persist_dir']}/{timestamp}"

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

    extractor_llm = None
    if "extractor" in config:
        ext_cfg = config["extractor"]
        extractor_llm = LLMClient(
            model=ext_cfg["model"],
            base_url=ext_cfg["base_url"],
            api_key=ext_cfg.get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
            temperature=ext_cfg.get("temperature", 0.0),
            max_tokens=ext_cfg.get("max_tokens", 512),
        )

    insight_store = InsightMemoryStore(
        embedding_model_name=config["memory"]["embedding_model"],
    )

    env_success: dict[int, float] = {}
    all_trajectories: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  ExpeL Epoch {epoch}/{args.epochs} (WebShop)")
        logger.info("=" * 60)

        env = WebShopEnv(num_products=args.num_products, max_sessions=args.max_envs)
        env.setup()

        episode_results = []
        env_count = 0
        num_skipped = 0

        while env_count < args.max_envs:
            try:
                if epoch > 1 and env_count in env_success:
                    env.skip()
                    episode_results.append({
                        "env_idx": env_count, "task_type": "shopping",
                        "task_description": "", "success": True,
                        "reward": env_success[env_count],
                        "total_steps": 0, "total_tokens": 0,
                        "wall_time_s": 0, "skipped": True,
                    })
                    num_skipped += 1
                    env_count += 1
                    continue

                store_to_use = insight_store if epoch > 1 else None
                result = run_episode(
                    llm, env, env_count, store_to_use,
                    args.max_steps, args.max_insights_inject,
                )
                episode_results.append(result)

                if result["success"]:
                    env_success[env_count] = result["reward"]

                if epoch == 1:
                    all_trajectories.append(result)

                env_count += 1
                rewards = [r.get("reward", 0) for r in episode_results]
                avg_r = sum(rewards) / len(rewards)
                status = "OK" if result["success"] else "FAIL"
                logger.info(f"  [{env_count}] {status} reward={result['reward']:.4f} "
                            f"running_avg={avg_r:.4f}")

                if env_count % 10 == 0:
                    summary = compute_summary(episode_results, insight_store.stats(), "expel")
                    save_results(episode_results, summary,
                                 f"{results_dir}/{args.run_name}{epoch}_intermediate.json")

            except Exception as e:
                logger.error(f"Error on session #{env_count}: {e}", exc_info=True)
                env_count += 1

        if num_skipped > 0:
            logger.info(f"  Skipped {num_skipped} already-succeeded sessions")

        # Post-epoch: extract insights (ExpeL core mechanism)
        if extractor_llm is not None and epoch == 1:
            logger.info("")
            logger.info("=" * 60)
            logger.info("  ExpeL: Extracting insights from epoch 1 trajectories...")
            logger.info("=" * 60)

            extractor_llm.tracker.reset()
            total_insights = 0

            successes_t = [t for t in all_trajectories if t["success"]]
            failures_t = [t for t in all_trajectories if not t["success"]]

            logger.info(f"  {len(successes_t)} success, {len(failures_t)} failure trajectories")

            for fail_t in failures_t:
                fail_hist = fail_t.get("history", [])
                fail_desc = fail_t.get("task_description", "")
                if not fail_hist:
                    continue

                if successes_t:
                    succ_t = random.choice(successes_t)
                    succ_hist = succ_t.get("history", [])
                    succ_desc = succ_t.get("task_description", "")
                    insights = extract_insights_from_pair(
                        extractor_llm, fail_hist, fail_desc,
                        succ_hist, succ_desc, "shopping", domain="webshop",
                    )
                else:
                    insights = extract_insights_from_failure(
                        extractor_llm, fail_hist, fail_desc,
                        "shopping", domain="webshop",
                    )

                for ins in insights:
                    insight_store.add(
                        rule=ins["rule"],
                        task_type=ins.get("task_type", "shopping"),
                        source_env_idx=fail_t["env_idx"],
                    )
                    total_insights += 1

            ext_stats = extractor_llm.tracker.summary()
            logger.info(f"  ExpeL: extracted {total_insights} insights, "
                        f"stored {insight_store.size()} (after dedup), "
                        f"extractor tokens: {ext_stats['total_tokens']:,}")
            insight_store.save(f"{insight_dir}/ws_expel_insights.json")

        # Epoch summary
        summary = compute_summary(episode_results, insight_store.stats(), "expel")
        logger.info(f"  Epoch {epoch}: {summary['total_success']}/{summary['total_envs']} "
                    f"({summary['success_rate']:.1%}), avg_reward={summary['avg_reward']:.4f}")

        # Remove history before saving
        for ep in episode_results:
            ep.pop("history", None)
        result_path = f"{results_dir}/{args.run_name}{epoch}.json"
        save_results(episode_results, summary, result_path)
        logger.info(f"Results saved: {result_path}")

        env.close()

    logger.info("Done!")


if __name__ == "__main__":
    main()

"""Run ExpeL baseline on ALFWorld.

ExpeL (Zhao et al., 2024) flow:
  Epoch 1: Run ReAct, collect trajectories (success + failure)
  Post-E1: Extract cross-task insights by comparing success/failure pairs
  Epoch 2: Inject relevant insights at episode start, run ReAct

This reproduces ExpeL's core mechanism using our same LLM backbone
(Qwen3.5-9B) for fair comparison with React_FM.
"""

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
from src.expel_memory import InsightMemoryStore
from src.expel_extractor import extract_insights_from_pair, extract_insights_from_failure
from src.failure_detector import ALFWorldFailureDetector
from src.alfworld_env import ALFWorldEnv
from src.log_utils import setup_logging
from prompts.alfworld_prompts import build_fewshot_prefix, format_step

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ExpeL on ALFWorld")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-envs", type=int, default=134)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="expel_")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-insights-inject", type=int, default=5)
    parser.add_argument("--resume-insights", type=str, default=None)
    parser.add_argument("--resume-results", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def save_results(results: list[dict], summary: dict, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    data = {"summary": summary, "episodes": results}
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def compute_summary(results: list[dict], insight_stats: dict, mode: str) -> dict:
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
    total_tokens = sum(r.get("total_tokens", 0) for r in results)
    return {
        "mode": mode,
        "total_envs": total,
        "total_success": successes,
        "success_rate": round(successes / total, 4) if total > 0 else 0,
        "by_task_type": by_type,
        "total_tokens": total_tokens,
        "avg_tokens_per_episode": round(total_tokens / total) if total > 0 else 0,
        "insight_stats": insight_stats,
    }


def build_expel_prompt(task_type, task_obs, history, insights=None):
    """Build prompt with ExpeL insights injected at episode start."""
    sections = [build_fewshot_prefix(task_type)]
    if insights:
        lines = ["Based on past experience, keep these rules in mind:"]
        for entry in insights:
            lines.append(f"- {entry.rule}")
        sections.append("\n".join(lines))
    sections.append("Here is the task.\n" + task_obs)
    prompt = "\n\n".join(sections) + "\n"
    for action, obs in history:
        prompt += format_step(action, obs)
    prompt += "> "
    return prompt


def run_episode(llm, env, env_idx, detector, insight_store, max_steps, max_inject):
    """Run one ALFWorld episode with optional ExpeL insight injection."""
    t0 = time.time()
    llm.tracker.reset()
    init_obs, task_type, info = env.reset()

    insights = None
    if insight_store and insight_store.size() > 0:
        insights = insight_store.retrieve(task_type, top_k=max_inject)
        if insights:
            logger.info(f"  ExpeL: injecting {len(insights)} insights for [{task_type}]")

    history = []
    action_history = []
    steps = []
    success = False

    for step_num in range(max_steps):
        prompt = build_expel_prompt(task_type, init_obs, history, insights)
        response = llm.complete_text(prompt, stop=["\n"], label=f"step_{step_num}")
        action = response.strip().split("\n")[0].strip()
        if action.startswith("> "):
            action = action[2:]
        if not action:
            action = "look"

        logger.info(f"  Step {step_num}: {action[:80]}")

        if action.startswith("think:") or action.startswith("think "):
            steps.append({"step": step_num, "action": action, "observation": "OK."})
            history.append((action, "OK."))
            continue

        observation, reward, done, step_info = env.step(action)
        logger.info(f"    obs: {observation[:80]}")
        action_history.append(action)
        is_done, is_success = detector.is_task_complete(observation, done, step_info)
        steps.append({"step": step_num, "action": action, "observation": observation[:200]})
        history.append((action, observation))

        if is_done:
            success = is_success
            break

    wall_time = time.time() - t0
    agent_stats = llm.tracker.summary()
    env_history = [(a, o) for a, o in history if not a.startswith("think")]

    return {
        "env_idx": env_idx,
        "task_type": task_type,
        "task_description": init_obs[:200],
        "success": success,
        "total_steps": len(steps),
        "total_tokens": agent_stats["total_tokens"],
        "wall_time_s": round(wall_time, 2),
        "skipped": False,
        "steps": steps,
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
    if args.resume_insights:
        insight_store.load(args.resume_insights)

    detector = ALFWorldFailureDetector()

    # Track per-env success for skip-on-success
    env_success = [None] * args.max_envs

    if args.resume_results:
        with open(args.resume_results) as f:
            prev_data = json.load(f)
        for ep in prev_data.get("episodes", []):
            idx = ep["env_idx"] - 1
            if 0 <= idx < len(env_success) and ep.get("success"):
                env_success[idx] = ep.get("task_type", "unknown")

    # Collect trajectories per task_type for insight extraction
    trajectories_by_type: dict[str, list[dict]] = {}

    for epoch in range(1, args.epochs + 1):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  ExpeL Epoch {epoch}/{args.epochs}")
        logger.info("=" * 60)

        env = ALFWorldEnv(split=config["alfworld"]["split"])
        env.setup()

        episode_results = []
        env_count = 0
        num_skipped = 0

        while True:
            try:
                # Skip already-succeeded envs in epoch > 1
                if epoch > 1 and env_success[env_count] is not None:
                    env.skip()
                    episode_results.append({
                        "env_idx": env_count + 1,
                        "task_type": env_success[env_count],
                        "task_description": "",
                        "success": True,
                        "total_steps": 0,
                        "total_tokens": 0,
                        "wall_time_s": 0,
                        "skipped": True,
                        "steps": [],
                    })
                    num_skipped += 1
                    env_count += 1
                    if env_count >= args.max_envs:
                        break
                    continue

                # Use insights only in epoch > 1
                store_to_use = insight_store if epoch > 1 else None
                result = run_episode(
                    llm, env, env_count + 1, detector,
                    store_to_use, args.max_steps, args.max_insights_inject,
                )
                episode_results.append(result)

                if result["success"]:
                    env_success[env_count] = result["task_type"]

                # Collect trajectory for insight extraction
                tt = result["task_type"]
                if tt not in trajectories_by_type:
                    trajectories_by_type[tt] = []
                trajectories_by_type[tt].append(result)

                env_count += 1
                successes = sum(1 for r in episode_results if r["success"])
                rate = successes / len(episode_results)
                status = "OK" if result["success"] else "FAIL"
                logger.info(f"  [{env_count}] {result['task_type']:<8} {status:>4} "
                            f"running={rate:.1%} ({successes}/{env_count})")

                if env_count % 10 == 0:
                    summary = compute_summary(
                        episode_results, insight_store.stats(), "expel",
                    )
                    save_results(
                        episode_results, summary,
                        f"{results_dir}/{args.run_name}{epoch}_intermediate.json",
                    )

                if env_count >= args.max_envs:
                    break

            except StopIteration:
                break
            except Exception as e:
                logger.error(f"Error on env #{env_count + 1}: {e}", exc_info=True)
                env_count += 1
                if env_count >= args.max_envs:
                    break

        if num_skipped > 0:
            logger.info(f"  Skipped {num_skipped} already-succeeded envs")

        # --- Post-epoch: extract insights (ExpeL's core mechanism) ---
        if extractor_llm is not None and epoch == 1:
            logger.info("")
            logger.info("=" * 60)
            logger.info("  ExpeL: Extracting insights from epoch 1 trajectories...")
            logger.info("=" * 60)

            extractor_llm.tracker.reset()
            total_insights = 0

            for tt, trajs in trajectories_by_type.items():
                successes_t = [t for t in trajs if t["success"]]
                failures_t = [t for t in trajs if not t["success"]]

                logger.info(f"  [{tt}] {len(successes_t)} success, {len(failures_t)} failure")

                # ExpeL: compare each failure with a success of same type
                for fail_t in failures_t:
                    fail_hist = fail_t.get("history", [])
                    fail_desc = fail_t.get("task_description", "")
                    if not fail_hist:
                        continue

                    if successes_t:
                        # Pick a success trajectory for comparison
                        succ_t = random.choice(successes_t)
                        succ_hist = succ_t.get("history", [])
                        succ_desc = succ_t.get("task_description", "")
                        insights = extract_insights_from_pair(
                            extractor_llm, fail_hist, fail_desc,
                            succ_hist, succ_desc, tt,
                        )
                    else:
                        # No success for this type — extract from failure alone
                        insights = extract_insights_from_failure(
                            extractor_llm, fail_hist, fail_desc, tt,
                        )

                    for ins in insights:
                        insight_store.add(
                            rule=ins["rule"],
                            task_type=ins.get("task_type", tt),
                            source_env_idx=fail_t["env_idx"],
                        )
                        total_insights += 1

            ext_stats = extractor_llm.tracker.summary()
            logger.info(f"  ExpeL: extracted {total_insights} insights, "
                        f"stored {insight_store.size()} (after dedup), "
                        f"extractor tokens: {ext_stats['total_tokens']:,}")

            insight_store.save(f"{insight_dir}/expel_insights.json")

        # Epoch summary
        summary = compute_summary(episode_results, insight_store.stats(), "expel")
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  ExpeL Epoch {epoch} Results")
        logger.info("=" * 60)
        logger.info(f"  Total: {summary['total_success']}/{summary['total_envs']} "
                    f"({summary['success_rate']:.1%})")
        for tt, stats in sorted(summary["by_task_type"].items()):
            logger.info(f"  {tt:<12} {stats['success']:>4}/{stats['total']:<4} "
                        f"{stats['rate']:>7.1%}")
        logger.info(f"  Tokens: {summary['total_tokens']:,}")
        logger.info(f"  Insights: {insight_store.size()} stored")
        logger.info("=" * 60)

        result_path = f"{results_dir}/{args.run_name}{epoch}.json"
        # Remove history from saved results (too large)
        for ep in episode_results:
            ep.pop("history", None)
        save_results(episode_results, summary, result_path)
        logger.info(f"Results saved: {result_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()

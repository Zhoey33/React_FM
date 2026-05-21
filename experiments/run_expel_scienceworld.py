"""Run ExpeL baseline on ScienceWorld.

ExpeL (Zhao et al., 2024) flow:
  Epoch 1: Run ReAct, collect trajectories
  Post-E1: Extract cross-task insights by comparing success/failure pairs
  Epoch 2: Inject relevant insights at episode start, run ReAct
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
from src.expel_memory import InsightMemoryStore
from src.expel_extractor import extract_insights_from_pair, extract_insights_from_failure
from src.scienceworld_env import ScienceWorldEnv, DEFAULT_EVAL_TASKS
from src.scienceworld_reporting import build_protocol_metadata
from src.scienceworld_reporting import compute_summary as compute_scienceworld_summary
from src.log_utils import setup_logging
from prompts.scienceworld_prompts import get_fewshot_examples, SYSTEM_PROMPT_BASE

logger = logging.getLogger(__name__)

_SW_ACTION_PREFIXES = (
    "go to", "pick up", "put down", "open", "close", "activate", "deactivate",
    "use", "pour", "mix", "focus on", "wait", "look around", "look", "inventory",
    "examine", "read", "connect", "move", "teleport to", "dunk", "eat", "drink",
    "flush", "reset task", "think:", "think ",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ExpeL on ScienceWorld")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-envs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="sw_expel_")
    parser.add_argument("--step-limit", type=int, default=30)
    parser.add_argument("--max-insights-inject", type=int, default=5)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--max-variations", type=int, default=5)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def _extract_action(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return ""
    lower = raw.lower()
    for prefix in _SW_ACTION_PREFIXES:
        if lower.startswith(prefix):
            return raw
    for prefix in _SW_ACTION_PREFIXES:
        idx = lower.find(prefix)
        if idx >= 0:
            rest = raw[idx:]
            end = rest.find(".")
            if 0 < end < 80:
                return rest[:end].strip()
            return rest[:80].strip()
    return raw[:80]


def save_results(results, summary, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump({"summary": summary, "episodes": results}, f, indent=2, ensure_ascii=False)


def compute_summary(results, insight_stats, mode, extractor_tokens: int = 0, protocol=None):
    return compute_scienceworld_summary(
        results,
        mode=mode,
        insight_stats=insight_stats,
        protocol=protocol,
        extra_tokens={"extractor_tokens": extractor_tokens},
    )


def build_expel_prompt(task_type, task_obs, history, insights=None):
    """Build prompt with ExpeL insights injected at episode start."""
    sections = [get_fewshot_examples(task_type)]
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


def run_episode(llm, env, env_idx, task_type, variation_idx, init_obs, insight_store, max_steps, max_inject):
    """Run one ScienceWorld episode with optional ExpeL insight injection."""
    t0 = time.time()
    llm.tracker.reset()

    insights = None
    if insight_store and insight_store.size() > 0:
        insights = insight_store.retrieve(task_type, top_k=max_inject)
        if insights:
            logger.info(f"  ExpeL: injecting {len(insights)} insights for [{task_type}]")

    history = []
    action_history = []
    steps = []
    success = False
    final_score = 0.0
    consecutive_thinks = 0

    for step_num in range(max_steps):
        prompt = build_expel_prompt(task_type, init_obs, history, insights)
        response = llm.complete_text(prompt, stop=["\n"], label=f"step_{step_num}",
                                     system=SYSTEM_PROMPT_BASE)
        action = response.strip().split("\n")[0].strip()
        if action.startswith("> "):
            action = action[2:]
        action = _extract_action(action)
        if not action:
            action = "look around"

        logger.info(f"  Step {step_num}: {action[:80]}")

        is_think = action.startswith("think:") or action.startswith("think ")
        if is_think:
            consecutive_thinks += 1
            if consecutive_thinks > 3:
                action = "look around"
                is_think = False
                consecutive_thinks = 0
            else:
                steps.append({"step": step_num, "action": action, "observation": "OK."})
                history.append((action, "OK."))
                continue
        else:
            consecutive_thinks = 0

        observation, reward, done, step_info = env.step(action)
        logger.info(f"    obs: {observation[:80]}")
        action_history.append(action)
        final_score = step_info.get("score", final_score)

        steps.append({"step": step_num, "action": action, "observation": observation[:200]})
        history.append((action, observation))

        if done or final_score >= 100.0:
            success = final_score >= 100.0
            break

    wall_time = time.time() - t0
    agent_stats = llm.tracker.summary()
    env_history = [(a, o) for a, o in history if not a.startswith("think")]

    return {
        "env_idx": env_idx,
        "task_type": task_type,
        "variation_idx": variation_idx,
        "task_description": init_obs[:200],
        "success": success,
        "score": final_score,
        "total_steps": len(steps),
        "total_tokens": agent_stats["total_tokens"],
        "agent_tokens": agent_stats["total_tokens"],
        "extractor_tokens": 0,
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

    llm_cfg = config["llm"]
    llm = LLMClient(
        model=llm_cfg["model"],
        base_url=llm_cfg["base_url"],
        api_key=llm_cfg.get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
        temperature=llm_cfg["temperature"],
        max_tokens=llm_cfg["max_tokens"],
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

    task_names = args.tasks or DEFAULT_EVAL_TASKS
    env_success: dict[int, str] = {}
    trajectories_by_type: dict[str, list[dict]] = {}
    cumulative_extractor_tokens = 0

    for epoch in range(1, args.epochs + 1):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  ExpeL Epoch {epoch}/{args.epochs} (ScienceWorld)")
        logger.info("=" * 60)

        env = ScienceWorldEnv(
            task_names=task_names, split=args.split,
            max_variations_per_task=args.max_variations,
            env_step_limit=args.step_limit,
        )
        env.setup()

        episode_results = []
        max_envs = args.max_envs or env.total_episodes
        protocol = build_protocol_metadata(
            split=args.split,
            tasks=task_names,
            max_variations=args.max_variations,
            step_limit=args.step_limit,
            test_time_writable=bool(args.split == "test"),
            memory_scope="expel_insight",
            max_envs=max_envs,
        )
        env_count = 0
        num_skipped = 0
        epoch_extractor_tokens = 0

        while env_count < max_envs:
            try:
                eidx = env_count + 1

                if epoch > 1 and eidx in env_success:
                    schedule = getattr(env, "_schedule", [])
                    variation_idx = schedule[eidx - 1][1] if eidx - 1 < len(schedule) else None
                    env.skip()
                    episode_results.append({
                        "env_idx": eidx, "task_type": env_success[eidx],
                        "variation_idx": variation_idx,
                        "task_description": "", "success": True, "score": 100.0,
                        "total_steps": 0, "total_tokens": 0,
                        "agent_tokens": 0, "extractor_tokens": 0,
                        "wall_time_s": 0, "skipped": True,
                    })
                    num_skipped += 1
                    env_count += 1
                    continue

                init_obs, task_type, info = env.reset()
                store_to_use = insight_store if epoch > 1 else None
                result = run_episode(
                    llm, env, eidx, task_type, info.get("variation_idx"), init_obs,
                    store_to_use, args.step_limit, args.max_insights_inject,
                )
                episode_results.append(result)

                if result["success"]:
                    env_success[eidx] = task_type

                # Collect trajectories for insight extraction
                if epoch == 1:
                    tt = result["task_type"]
                    if tt not in trajectories_by_type:
                        trajectories_by_type[tt] = []
                    trajectories_by_type[tt].append(result)

                env_count += 1
                successes = sum(1 for r in episode_results if r["success"])
                rate = successes / len(episode_results)
                status = "OK" if result["success"] else "FAIL"
                logger.info(f"  [{env_count}] {task_type:<30} {status:>4} "
                            f"score={result['score']:.2f} running={rate:.1%}")

                if env_count % 10 == 0:
                    summary = compute_summary(
                        episode_results,
                        insight_store.stats(),
                        "expel",
                        protocol=protocol,
                    )
                    save_results(episode_results, summary,
                                 f"{results_dir}/{args.run_name}{epoch}_intermediate.json")

            except StopIteration:
                break
            except Exception as e:
                logger.error(f"Error on env #{env_count + 1}: {e}", exc_info=True)
                task_type = "unknown"
                schedule = getattr(env, "_schedule", None)
                if schedule and env_count < len(schedule):
                    task_type = schedule[env_count][0]
                episode_results.append({
                    "env_idx": env_count + 1,
                    "task_type": task_type,
                    "variation_idx": (
                        schedule[env_count][1]
                        if schedule and env_count < len(schedule)
                        else None
                    ),
                    "task_description": "",
                    "success": False,
                    "score": -100.0,
                    "total_steps": 0,
                    "total_tokens": 0,
                    "agent_tokens": 0,
                    "extractor_tokens": 0,
                    "wall_time_s": 0.0,
                    "skipped": False,
                    "error": str(e),
                })
                env_count += 1

        if num_skipped > 0:
            logger.info(f"  Skipped {num_skipped} already-succeeded envs")

        # Post-epoch: extract insights
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
                            succ_hist, succ_desc, tt, domain="scienceworld",
                        )
                    else:
                        insights = extract_insights_from_failure(
                            extractor_llm, fail_hist, fail_desc,
                            tt, domain="scienceworld",
                        )

                    for ins in insights:
                        insight_store.add(
                            rule=ins["rule"],
                            task_type=tt,
                            source_env_idx=fail_t["env_idx"],
                        )
                        total_insights += 1

            ext_stats = extractor_llm.tracker.summary()
            epoch_extractor_tokens = ext_stats["total_tokens"]
            cumulative_extractor_tokens += epoch_extractor_tokens
            logger.info(f"  ExpeL: extracted {total_insights} insights, "
                        f"stored {insight_store.size()} (after dedup), "
                        f"extractor tokens: {ext_stats['total_tokens']:,}")
            insight_store.save(f"{insight_dir}/sw_expel_insights.json")

        # Epoch summary
        summary = compute_summary(
            episode_results,
            insight_store.stats(),
            "expel",
            extractor_tokens=cumulative_extractor_tokens,
            protocol=protocol,
        )
        logger.info(f"  Epoch {epoch}: {summary['total_success']}/{summary['total_envs']} "
                    f"({summary['success_rate']:.1%}), avg_score={summary['avg_score']:.2f}")
        for tt, stats in sorted(summary["by_task_type"].items()):
            logger.info(f"    {tt:<30} {stats['success']}/{stats['total']} ({stats['rate']:.1%})")

        for ep in episode_results:
            ep.pop("history", None)
        result_path = f"{results_dir}/{args.run_name}{epoch}.json"
        save_results(episode_results, summary, result_path)
        logger.info(f"Results saved: {result_path}")

        env.close()

    logger.info("Done!")


if __name__ == "__main__":
    main()

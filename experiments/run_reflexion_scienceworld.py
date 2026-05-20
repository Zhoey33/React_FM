"""Run online Reflexion on ScienceWorld.

Online Reflexion flow:
  Pass 0: Run one episode, collect trajectory
  Post-pass: If failed, immediately generate a reflection
  Pass 1: Re-run the same episode with the new reflection injected
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
from src.scienceworld_env import ScienceWorldEnv, DEFAULT_EVAL_TASKS
from src.reflexion_agent import generate_reflection
from src.log_utils import setup_logging
from prompts.scienceworld_prompts import get_fewshot_examples, SYSTEM_PROMPT_BASE

logger = logging.getLogger(__name__)

# Valid action prefixes (same as run_scienceworld.py)
_SW_ACTION_PREFIXES = (
    "go to", "pick up", "put down", "open", "close", "activate", "deactivate",
    "use", "pour", "mix", "focus on", "wait", "look around", "look", "inventory",
    "examine", "read", "connect", "move", "teleport to", "dunk", "eat", "drink",
    "flush", "reset task", "think:", "think ",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Reflexion on ScienceWorld")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--num-trials",
        type=int,
        default=2,
        help="Number of per-episode passes (kept for CLI compatibility; 2 is the intended setting).",
    )
    parser.add_argument("--max-envs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="sw_reflexion_")
    parser.add_argument("--step-limit", type=int, default=30)
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


def build_reflexion_prompt(task_type, task_obs, history, memory=None):
    """Build prompt with Reflexion memory injected at episode start."""
    sections = [get_fewshot_examples(task_type)]
    if memory:
        lines = ["Your memory for the task below:"]
        recent_memory = memory[-3:] if len(memory) > 3 else memory
        for i, m in enumerate(recent_memory):
            lines.append(f"Trial {i}:\n{m.strip()}")
        sections.append("\n".join(lines))
    sections.append("Here is the task.\n" + task_obs)
    prompt = "\n\n".join(sections) + "\n"
    for action, obs in history:
        prompt += f"> {action}\n{obs}\n"
    prompt += "> "
    return prompt


def run_episode(llm, env, env_idx, pass_idx, total_passes, task_type, init_obs, memory, max_steps):
    """Run one ScienceWorld episode with optional Reflexion memory."""
    t0 = time.time()
    llm.tracker.reset()

    injected_memory = memory[-3:] if len(memory) > 3 else memory
    logger.info(
        f"  Env #{env_idx} pass {pass_idx + 1}/{total_passes} memory injection: "
        f"{len(injected_memory)} item(s)"
    )

    history = []
    steps = []
    success = False
    final_score = 0.0
    consecutive_thinks = 0
    last_action = ""

    for step_num in range(max_steps):
        prompt = build_reflexion_prompt(task_type, init_obs, history, memory)
        response = llm.complete_text(prompt, stop=["\n"], label=f"step_{step_num}",
                                     system=SYSTEM_PROMPT_BASE)
        action = response.strip().split("\n")[0].strip()
        if action.startswith("> "):
            action = action[2:]
        action = _extract_action(action)
        if not action:
            action = "look around"

        logger.info(f"  Step {step_num}: {action[:80]}")

        # Match official Reflexion's repeated-action exhaustion behavior.
        if action == last_action:
            logger.info(f"  Episode exhausted (repeated action) at step {step_num + 1}")
            break
        last_action = action

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
        final_score = step_info.get("score", final_score)

        steps.append({"step": step_num, "action": action, "observation": observation[:200]})
        history.append((action, observation))

        if done or final_score >= 100.0:
            success = final_score >= 100.0
            break

    wall_time = time.time() - t0
    agent_stats = llm.tracker.summary()
    log_str = build_reflexion_prompt(task_type, init_obs, history, memory)

    return {
        "env_idx": env_idx,
        "pass_idx": pass_idx,
        "task_type": task_type,
        "task_description": init_obs[:200],
        "success": success,
        "score": final_score,
        "total_steps": len(steps),
        "total_tokens": agent_stats["total_tokens"],
        "wall_time_s": round(wall_time, 2),
        "skipped": False,
        "memory_items_used": len(injected_memory),
        "reflection_generated": False,
        "reflection_error": False,
        "memory_count_after": len(memory),
        "log_str": log_str,
    }


def save_results(results, summary, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump({"summary": summary, "episodes": results}, f, indent=2, ensure_ascii=False)


def strip_prompt_logs(results):
    """Return a copy of results without the large per-episode prompt transcript."""
    return [{k: v for k, v in r.items() if k != "log_str"} for r in results]


def save_run_report(payload, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def compute_summary(results, mode):
    total = len(results)
    successes = sum(1 for r in results if r["success"])
    # ScienceWorld may return negative scores; keep the official scale so
    # Reflexion is directly comparable with the main ScienceWorld runners.
    scores = [r.get("score", 0.0) / 100.0 for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    by_type = {}
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
        "benchmark": "scienceworld",
        "total_envs": total,
        "total_success": successes,
        "success_rate": round(successes / total, 4) if total > 0 else 0,
        "avg_score": round(avg_score, 4),
        "by_task_type": by_type,
        "total_tokens": total_tokens,
        "avg_tokens_per_episode": round(total_tokens / total) if total > 0 else 0,
    }


def compute_comparison(pass_results, reflection_stats):
    if len(pass_results) < 2:
        return {
            "num_passes": len(pass_results),
            "reflection_stats": reflection_stats,
        }

    base_results = pass_results[0]
    final_results = pass_results[-1]
    by_task_type = {}
    improved_to_success = 0
    regressed_from_success = 0
    score_improved = 0
    score_regressed = 0

    for base, final in zip(base_results, final_results):
        task_type = final.get("task_type") or base.get("task_type") or "unknown"
        entry = by_task_type.setdefault(task_type, {
            "total": 0,
            "pass0_success": 0,
            "passN_success": 0,
            "pass0_score_sum": 0.0,
            "passN_score_sum": 0.0,
            "improved_to_success": 0,
            "regressed_from_success": 0,
            "score_improved": 0,
            "score_regressed": 0,
        })
        entry["total"] += 1
        entry["pass0_success"] += int(base["success"])
        entry["passN_success"] += int(final["success"])
        entry["pass0_score_sum"] += base.get("score", 0.0)
        entry["passN_score_sum"] += final.get("score", 0.0)

        if (not base["success"]) and final["success"]:
            improved_to_success += 1
            entry["improved_to_success"] += 1
        if base["success"] and (not final["success"]):
            regressed_from_success += 1
            entry["regressed_from_success"] += 1
        if final.get("score", 0.0) > base.get("score", 0.0):
            score_improved += 1
            entry["score_improved"] += 1
        elif final.get("score", 0.0) < base.get("score", 0.0):
            score_regressed += 1
            entry["score_regressed"] += 1

    for entry in by_task_type.values():
        total = entry["total"]
        entry["pass0_success_rate"] = round(entry["pass0_success"] / total, 4) if total else 0.0
        entry["passN_success_rate"] = round(entry["passN_success"] / total, 4) if total else 0.0
        entry["success_rate_delta"] = round(
            entry["passN_success_rate"] - entry["pass0_success_rate"], 4
        )
        entry["pass0_avg_score"] = round((entry["pass0_score_sum"] / total) / 100.0, 4) if total else 0.0
        entry["passN_avg_score"] = round((entry["passN_score_sum"] / total) / 100.0, 4) if total else 0.0
        entry["avg_score_delta"] = round(entry["passN_avg_score"] - entry["pass0_avg_score"], 4)
        del entry["pass0_score_sum"]
        del entry["passN_score_sum"]

    pass0_summary = compute_summary(base_results, "reflexion_online_pass0")
    passn_summary = compute_summary(final_results, f"reflexion_online_pass{len(pass_results) - 1}")
    total = len(base_results)

    return {
        "num_passes": len(pass_results),
        "pass0_success_rate": pass0_summary["success_rate"],
        "passN_success_rate": passn_summary["success_rate"],
        "success_rate_delta": round(passn_summary["success_rate"] - pass0_summary["success_rate"], 4),
        "pass0_avg_score": pass0_summary["avg_score"],
        "passN_avg_score": passn_summary["avg_score"],
        "avg_score_delta": round(passn_summary["avg_score"] - pass0_summary["avg_score"], 4),
        "improved_to_success": improved_to_success,
        "regressed_from_success": regressed_from_success,
        "score_improved": score_improved,
        "score_regressed": score_regressed,
        "score_unchanged": max(total - score_improved - score_regressed, 0),
        "reflection_stats": reflection_stats,
        "by_task_type": by_task_type,
    }


def build_run_payload(pass_results, episode_records, reflection_stats):
    pass_summaries = {
        f"pass{pass_idx}": compute_summary(results, f"reflexion_online_pass{pass_idx}")
        for pass_idx, results in enumerate(pass_results)
    }
    return {
        "mode": "reflexion_online",
        "benchmark": "scienceworld",
        "num_passes": len(pass_results),
        "pass_summaries": pass_summaries,
        "comparison": compute_comparison(pass_results, reflection_stats),
        "episodes": episode_records,
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

    # ScienceWorld Reflexion should use the main agent model, not the extractor model.
    if "llm" in config:
        llm_cfg = config["llm"]
        llm = LLMClient(
            model=llm_cfg["model"],
            base_url=llm_cfg["base_url"],
            api_key=llm_cfg.get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
            temperature=llm_cfg["temperature"],
            max_tokens=llm_cfg["max_tokens"],
        )
    else:
        llm = LLMClient(
            model=config["llm"]["model"],
            base_url=config["llm"]["base_url"],
            api_key=config["llm"].get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
            temperature=config["llm"]["temperature"],
            max_tokens=config["llm"]["max_tokens"],
        )

    task_names = args.tasks or DEFAULT_EVAL_TASKS

    if args.num_trials < 1:
        raise ValueError("--num-trials must be at least 1")

    if args.num_trials != 2:
        logger.warning(
            f"Online Reflexion is usually evaluated with 2 passes; got --num-trials={args.num_trials}"
        )

    env = ScienceWorldEnv(
        task_names=task_names, split=args.split,
        max_variations_per_task=args.max_variations,
        env_step_limit=args.step_limit,
    )
    env.setup()

    max_envs = min(args.max_envs or env.total_episodes, env.total_episodes)
    env_configs = {
        eidx: {"memory": [], "task_type": "", "task_description": "", "reflections_generated": 0}
        for eidx in range(1, max_envs + 1)
    }
    pass_results = [[] for _ in range(args.num_trials)]
    episode_records = []
    reflection_stats = {"attempted": 0, "generated": 0, "errors": 0}
    checkpoint_path = f"{results_dir}/{args.run_name}online_checkpoint.json"

    for env_idx in range(1, max_envs + 1):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  Episode {env_idx}/{max_envs} — Online Reflexion")
        logger.info("=" * 60)

        episode_passes = []
        for pass_idx in range(args.num_trials):
            logger.info(f"  Starting pass {pass_idx + 1}/{args.num_trials} for env #{env_idx}")
            current_memory = list(env_configs[env_idx]["memory"])
            try:
                init_obs, task_type, info = env.reset_to_episode(env_idx - 1)
                env_configs[env_idx]["task_type"] = task_type
                env_configs[env_idx]["task_description"] = init_obs[:200]
                result = run_episode(
                    llm, env, env_idx, pass_idx, args.num_trials, task_type, init_obs,
                    current_memory, args.step_limit,
                )
            except Exception as e:
                task_type = env_configs[env_idx]["task_type"] or "unknown"
                logger.error(f"Error on env #{env_idx} pass {pass_idx}: {e}", exc_info=True)
                result = {
                    "env_idx": env_idx,
                    "pass_idx": pass_idx,
                    "task_type": task_type,
                    "task_description": env_configs[env_idx]["task_description"],
                    "success": False,
                    "score": 0.0,
                    "total_steps": 0,
                    "total_tokens": 0,
                    "wall_time_s": 0.0,
                    "skipped": False,
                    "memory_items_used": min(len(current_memory), 3),
                    "reflection_generated": False,
                    "reflection_error": False,
                    "memory_count_after": len(current_memory),
                    "error": str(e),
                    "log_str": "",
                }

            if (pass_idx < args.num_trials - 1) and (not result["success"]):
                reflection_stats["attempted"] += 1
                logger.info(
                    f"  Generating reflection for env #{env_idx} after pass {pass_idx} "
                    f"(prior_memory={len(env_configs[env_idx]['memory'])})"
                )
                if result.get("log_str"):
                    try:
                        reflection = generate_reflection(
                            llm, result["log_str"], env_configs[env_idx]["memory"], domain="scienceworld",
                        )
                    except Exception as e:
                        reflection_stats["errors"] += 1
                        result["reflection_error"] = True
                        result["reflection_error_message"] = str(e)
                        logger.error(
                            f"Reflection failed for env #{env_idx} ({task_type}) after pass {pass_idx}: {e}",
                            exc_info=True,
                        )
                    else:
                        if reflection:
                            env_configs[env_idx]["memory"].append(reflection)
                            env_configs[env_idx]["reflections_generated"] += 1
                            reflection_stats["generated"] += 1
                            result["reflection_generated"] = True
                        else:
                            logger.warning(
                                f"Empty reflection for env #{env_idx} ({task_type}) after pass {pass_idx}"
                            )
                else:
                    logger.warning(
                        f"Skipping reflection for env #{env_idx} ({task_type}) after pass {pass_idx}: "
                        "missing log_str"
                    )

            result["memory_count_after"] = len(env_configs[env_idx]["memory"])
            pass_results[pass_idx].append(result)
            episode_passes.append(result)

            summary = compute_summary(pass_results[pass_idx], f"reflexion_online_pass{pass_idx}")
            status = "OK" if result["success"] else "FAIL"
            logger.info(
                f"  [{env_idx}/{max_envs}] pass{pass_idx} {task_type:<30} {status:>4} "
                f"score={result['score']:.2f} running={summary['success_rate']:.1%}"
            )

        episode_records.append({
            "env_idx": env_idx,
            "task_type": env_configs[env_idx]["task_type"],
            "task_description": env_configs[env_idx]["task_description"],
            "memory": list(env_configs[env_idx]["memory"]),
            "reflections_generated": env_configs[env_idx]["reflections_generated"],
            "passes": strip_prompt_logs(episode_passes),
        })

        payload = build_run_payload(pass_results, episode_records, reflection_stats)
        save_run_report(payload, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    env.close()

    for pass_idx, results in enumerate(pass_results):
        summary = compute_summary(results, f"reflexion_online_pass{pass_idx}")
        result_path = f"{results_dir}/{args.run_name}pass{pass_idx}.json"
        save_results(strip_prompt_logs(results), summary, result_path)
        logger.info(
            f"  Pass {pass_idx}: {summary['total_success']}/{summary['total_envs']} "
            f"({summary['success_rate']:.1%}), avg_score={summary['avg_score']:.2%}"
        )
        logger.info(f"Results saved: {result_path}")

    final_payload = build_run_payload(pass_results, episode_records, reflection_stats)
    comparison_path = f"{results_dir}/{args.run_name}comparison.json"
    save_run_report(final_payload, comparison_path)
    logger.info(f"Comparison saved: {comparison_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()

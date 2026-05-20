"""Run Reflexion baseline on HotPotQA.

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
from src.hotpotqa_env import HotPotQAEnv, exact_match_score, f1_score
from src.reflexion_agent import generate_reflection
from src.log_utils import setup_logging
from prompts.hotpotqa_prompts import FEWSHOT_EXAMPLE, SYSTEM_PROMPT_BASE

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Reflexion on HotPotQA")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--num-trials", type=int, default=2)
    parser.add_argument("--max-envs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="hqa_reflexion_")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--num-examples", type=int, default=500)
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
    em_scores = [r.get("em", 0.0) for r in results]
    f1_scores_list = [r.get("f1", 0.0) for r in results]
    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    avg_f1 = sum(f1_scores_list) / len(f1_scores_list) if f1_scores_list else 0.0
    successes = sum(1 for r in results if r.get("em", 0.0) > 0)
    total_tokens = sum(r.get("total_tokens", 0) for r in results)
    return {
        "mode": mode,
        "benchmark": "hotpotqa",
        "total_envs": total,
        "total_success": successes,
        "success_rate": round(successes / total, 4) if total > 0 else 0,
        "avg_em": round(avg_em, 4),
        "avg_f1": round(avg_f1, 4),
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
        m = re.search(r'(Search\[.+?\]|Lookup\[.+?\]|Finish\[.+?\])', response)
        if m:
            action = m.group(1)
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
    """Run one HotPotQA episode with optional Reflexion memory."""
    t0 = time.time()
    llm.tracker.reset()

    init_obs, task_type, info = env.reset()
    gold_answer = info["gold_answer"]

    history = []
    action_history = []
    steps = []
    final_em = 0.0
    final_f1 = 0.0
    final_answer = ""

    for step_num in range(max_steps):
        prompt = build_reflexion_prompt(init_obs, history, memory)
        response = llm.complete_text(prompt, label=f"step_{step_num}",
                                     system=SYSTEM_PROMPT_BASE)
        thought, action = _parse_thought_action(response)
        if not action:
            action = "Finish[unknown]"

        logger.info(f"  Step {step_num}: {action[:80]}")

        observation, reward, done, step_info = env.step(action)
        logger.info(f"    obs: {observation[:80]}")
        action_history.append(action)

        steps.append({"step": step_num, "thought": thought, "action": action,
                       "observation": observation[:300]})
        history.append((thought, action, observation))

        if done:
            final_em = step_info.get("em", 0.0)
            final_f1 = step_info.get("f1", 0.0)
            final_answer = action[7:-1] if action.startswith("Finish[") else observation
            break

    wall_time = time.time() - t0
    agent_stats = llm.tracker.summary()
    log_str = build_reflexion_prompt(init_obs, history, memory)

    return {
        "env_idx": env_idx,
        "task_type": task_type,
        "question": info["question"],
        "gold_answer": gold_answer,
        "predicted_answer": final_answer,
        "em": final_em,
        "f1": final_f1,
        "success": final_em > 0,
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

    max_envs = args.max_envs or args.num_examples

    # Per-env memory: keyed by env_idx
    env_configs = {}

    for trial_idx in range(args.num_trials):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  Trial {trial_idx}/{args.num_trials} — Reflexion (HotPotQA)")
        logger.info("=" * 60)

        env = HotPotQAEnv(num_examples=max_envs, seed=seed)
        env.setup()

        trial_logs = []
        env_count = 0
        num_success = 0

        while env_count < max_envs:
            try:
                if env_count not in env_configs:
                    env_configs[env_count] = {"memory": [], "is_success": False}

                if env_configs[env_count]["is_success"]:
                    env.skip()
                    num_success += 1
                    trial_logs.append({
                        "env_idx": env_count, "task_type": "qa",
                        "question": "", "gold_answer": "", "predicted_answer": "",
                        "em": 1.0, "f1": 1.0, "success": True,
                        "total_steps": 0, "total_tokens": 0,
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

                if result["em"] > 0:
                    env_configs[env_count]["is_success"] = True
                    num_success += 1

                env_count += 1
                rate = num_success / env_count
                status = "OK" if result["em"] > 0 else "FAIL"
                logger.info(f"  [{env_count}] {status} em={result['em']:.1f} "
                            f"f1={result['f1']:.2f} running={rate:.1%}")

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
                            domain="hotpotqa",
                        )
                        env_configs[eidx]["memory"].append(reflection)
                        num_reflected += 1
            logger.info(f"Generated {num_reflected} reflections")

        summary = compute_summary(trial_logs, "reflexion")
        logger.info(f"  Trial {trial_idx}: EM={summary['avg_em']:.3f} F1={summary['avg_f1']:.3f} "
                    f"({summary['total_success']}/{summary['total_envs']})")

        for log in trial_logs:
            log.pop("log_str", None)
        result_path = f"{results_dir}/{args.run_name}trial{trial_idx}.json"
        save_results(trial_logs, summary, result_path)
        logger.info(f"Results saved: {result_path}")

        env.close()

    logger.info("Done!")


if __name__ == "__main__":
    main()

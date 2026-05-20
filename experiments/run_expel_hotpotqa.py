"""Run ExpeL baseline on HotPotQA.

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
from src.hotpotqa_env import HotPotQAEnv, exact_match_score, f1_score
from src.log_utils import setup_logging
from prompts.hotpotqa_prompts import FEWSHOT_EXAMPLE, SYSTEM_PROMPT_BASE

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ExpeL on HotPotQA")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-envs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="hqa_expel_")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-insights-inject", type=int, default=5)
    parser.add_argument("--num-examples", type=int, default=500)
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
        "insight_stats": insight_stats,
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


def build_expel_prompt(task_obs, history, insights=None):
    """Build prompt with ExpeL insights injected at episode start."""
    sections = [FEWSHOT_EXAMPLE]
    if insights:
        lines = ["Based on past experience, keep these rules in mind:"]
        for entry in insights:
            lines.append(f"- {entry.rule}")
        sections.append("\n".join(lines))
    sections.append(task_obs)
    prompt = "\n\n".join(sections) + "\n"
    for thought, action, observation in history:
        if thought:
            prompt += f"Thought: {thought}\n"
        prompt += f"Action: {action}\nObservation: {observation}\n"
    return prompt


def run_episode(llm, env, env_idx, insight_store, max_steps, max_inject):
    """Run one HotPotQA episode with optional ExpeL insight injection."""
    t0 = time.time()
    llm.tracker.reset()

    init_obs, task_type, info = env.reset()
    gold_answer = info["gold_answer"]

    insights = None
    if insight_store and insight_store.size() > 0:
        insights = insight_store.retrieve(task_type, top_k=max_inject)
        if insights:
            logger.info(f"  ExpeL: injecting {len(insights)} insights")

    history = []
    action_history = []
    steps = []
    final_em = 0.0
    final_f1 = 0.0
    final_answer = ""

    for step_num in range(max_steps):
        prompt = build_expel_prompt(init_obs, history, insights)
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
        "history": [(a, o) for _, a, o in history],
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

    max_envs = args.max_envs or args.num_examples
    env_success: dict[int, float] = {}
    all_trajectories: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  ExpeL Epoch {epoch}/{args.epochs} (HotPotQA)")
        logger.info("=" * 60)

        env = HotPotQAEnv(num_examples=max_envs, seed=seed)
        env.setup()

        episode_results = []
        env_count = 0
        num_skipped = 0

        while env_count < max_envs:
            try:
                if epoch > 1 and env_count in env_success:
                    env.skip()
                    episode_results.append({
                        "env_idx": env_count, "task_type": "qa",
                        "question": "", "gold_answer": "", "predicted_answer": "",
                        "em": env_success[env_count], "f1": env_success[env_count],
                        "success": True, "total_steps": 0, "total_tokens": 0,
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

                if result["em"] > 0:
                    env_success[env_count] = result["em"]

                if epoch == 1:
                    all_trajectories.append(result)

                env_count += 1
                em_scores = [r.get("em", 0) for r in episode_results]
                avg_em = sum(em_scores) / len(em_scores)
                status = "OK" if result["em"] > 0 else "FAIL"
                logger.info(f"  [{env_count}] {status} em={result['em']:.1f} "
                            f"running_em={avg_em:.3f}")

                if env_count % 10 == 0:
                    summary = compute_summary(episode_results, insight_store.stats(), "expel")
                    save_results(episode_results, summary,
                                 f"{results_dir}/{args.run_name}{epoch}_intermediate.json")

            except StopIteration:
                break
            except Exception as e:
                logger.error(f"Error on env #{env_count}: {e}", exc_info=True)
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

            successes_t = [t for t in all_trajectories if t["success"]]
            failures_t = [t for t in all_trajectories if not t["success"]]

            logger.info(f"  {len(successes_t)} success, {len(failures_t)} failure trajectories")

            for fail_t in failures_t:
                fail_hist = fail_t.get("history", [])
                fail_desc = fail_t.get("question", "")
                if not fail_hist:
                    continue

                if successes_t:
                    succ_t = random.choice(successes_t)
                    succ_hist = succ_t.get("history", [])
                    succ_desc = succ_t.get("question", "")
                    insights = extract_insights_from_pair(
                        extractor_llm, fail_hist, fail_desc,
                        succ_hist, succ_desc, "qa", domain="hotpotqa",
                    )
                else:
                    insights = extract_insights_from_failure(
                        extractor_llm, fail_hist, fail_desc,
                        "qa", domain="hotpotqa",
                    )

                for ins in insights:
                    insight_store.add(
                        rule=ins["rule"],
                        task_type=ins.get("task_type", "qa"),
                        source_env_idx=fail_t["env_idx"],
                    )
                    total_insights += 1

            ext_stats = extractor_llm.tracker.summary()
            logger.info(f"  ExpeL: extracted {total_insights} insights, "
                        f"stored {insight_store.size()} (after dedup), "
                        f"extractor tokens: {ext_stats['total_tokens']:,}")
            insight_store.save(f"{insight_dir}/hqa_expel_insights.json")

        # Epoch summary
        summary = compute_summary(episode_results, insight_store.stats(), "expel")
        logger.info(f"  Epoch {epoch}: EM={summary['avg_em']:.3f} F1={summary['avg_f1']:.3f} "
                    f"({summary['total_success']}/{summary['total_envs']})")

        for ep in episode_results:
            ep.pop("history", None)
        result_path = f"{results_dir}/{args.run_name}{epoch}.json"
        save_results(episode_results, summary, result_path)
        logger.info(f"Results saved: {result_path}")

        env.close()

    logger.info("Done!")


if __name__ == "__main__":
    main()

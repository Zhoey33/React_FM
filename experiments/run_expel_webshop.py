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
from collections.abc import Sequence
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
    parser.add_argument("--max-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="ws_expel_")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--max-insights-inject", type=int, default=5)
    parser.add_argument("--num-products", type=str, default="full")
    parser.add_argument("--eval-split", choices=["test", "eval", "train"], default="test")
    parser.add_argument("--eval-sample-size", type=int, default=100)
    parser.add_argument("--eval-sample-seed", type=int, default=42)
    parser.add_argument("--sample-ids", type=str, default=None)
    parser.add_argument("--observation-mode", choices=["html", "text", "text_rich", "url"], default="text_rich")
    parser.add_argument("--human-goals", type=int, default=1)
    parser.add_argument("--webshop-wrapper", choices=["official", "direct"], default="official")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def save_results(results, summary, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump({"summary": summary, "episodes": results}, f, indent=2, ensure_ascii=False)


def save_sample_ids(session_ids: Sequence[int], filepath: str):
    """Save fixed WebShop session IDs for reproducible evaluation."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(list(session_ids), f, indent=2)


def parse_num_products(value: str) -> int | None:
    """Parse product-count values, where None means full WebShop."""
    lowered = str(value).lower()
    if lowered in {"full", "all", "none", "null"}:
        return None
    return int(value)


def sample_session_ids(total: int, sample_size: int, seed: int) -> list[int]:
    """Sample a deterministic official-split subset without replacement."""
    if sample_size > total:
        raise ValueError(f"sample_size={sample_size} exceeds split size {total}")
    rng = random.Random(seed)
    return rng.sample(range(total), sample_size)


def official_split_session_ids(split: str) -> list[int]:
    """Return official WebShop shuffled-goal indices for a split."""
    if split == "test":
        return list(range(500))
    if split == "eval":
        return list(range(500, 1500))
    if split == "train":
        return list(range(1500, 12087))
    raise ValueError(f"Unsupported WebShop split: {split}")


def validate_session_ids(session_ids: list[int], split: str) -> None:
    """Validate WebShop session IDs against the requested split."""
    if len(set(session_ids)) != len(session_ids):
        raise ValueError("Duplicate WebShop session IDs are not allowed")
    allowed = set(official_split_session_ids(split))
    outside = [idx for idx in session_ids if idx not in allowed]
    if outside:
        raise ValueError(f"Sample IDs outside official {split} split: {outside[:10]}")


def load_or_sample_session_ids(
    sample_ids_path: str | None,
    split: str,
    sample_size: int,
    seed: int,
) -> list[int]:
    """Load session IDs from JSON or sample from the official WebShop split."""
    if sample_ids_path:
        with open(sample_ids_path) as f:
            ids = json.load(f)
        if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
            raise ValueError(f"Sample ID file must be a JSON list of ints: {sample_ids_path}")
        validate_session_ids(ids, split)
        return ids

    candidates = official_split_session_ids(split)
    sampled_offsets = sample_session_ids(len(candidates), sample_size, seed)
    ids = [candidates[offset] for offset in sampled_offsets]
    validate_session_ids(ids, split)
    return ids


def is_exact_success(reward: float) -> bool:
    """Return paper-aligned WebShop success: reward must be exactly 1.0."""
    return reward == 1.0


def compute_summary(results, insight_stats, mode):
    total = len(results)
    rewards = [r.get("reward", 0.0) for r in results]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    successes = sum(1 for r in rewards if is_exact_success(r))
    total_tokens = sum(r.get("total_tokens", 0) for r in results)
    return {
        "mode": mode,
        "benchmark": "webshop",
        "success_definition": "reward == 1.0",
        "total_envs": total,
        "total_success": successes,
        "success_rate": round(successes / total, 4) if total > 0 else 0,
        "avg_reward": round(avg_reward, 4),
        "task_score": round(100 * avg_reward, 2),
        "total_tokens": total_tokens,
        "avg_tokens_per_episode": round(total_tokens / total) if total > 0 else 0,
        "insight_stats": insight_stats,
    }


def build_expel_prompt(task_obs, history, insights=None, valid_actions=None):
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
    rendered_actions = [
        action for action in (valid_actions or [])
        if action != "search[product]"
    ]
    if rendered_actions:
        action_lines = "\n".join(f"- {action}" for action in rendered_actions)
        prompt += (
            "\nChoose exactly one valid action from this list and output only that action.\n"
            + action_lines
            + "\n"
        )
    prompt += "> "
    return prompt


def normalize_expel_action(action: str, valid_actions: list[str] | dict | None) -> str:
    """Coerce model output to one official WebShop valid action when available."""
    if isinstance(valid_actions, dict):
        valid_list = []
    else:
        valid_list = list(valid_actions or [])

    action = action.strip().split("\n")[0].strip()
    if action.startswith("> "):
        action = action[2:].strip()
    if action.lower().startswith("action:"):
        action = action.split(":", 1)[1].strip()

    search_match = re.search(r"search\[([^\]]+)\]", action, flags=re.IGNORECASE)
    click_match = re.search(r"click\[([^\]]+)\]", action, flags=re.IGNORECASE)
    if search_match:
        action = f"search[{search_match.group(1)}]"
    elif click_match:
        action = f"click[{click_match.group(1)}]"

    if not valid_list:
        if not action or not (action.startswith("search[") or action.startswith("click[")):
            return "search[product]"
        return action

    if action in valid_list:
        return action

    lowered = action.lower()
    for candidate in valid_list:
        if candidate.lower() == lowered:
            return candidate

    if action.startswith("search["):
        if valid_list == ["search[product]"]:
            return action
        search_actions = [candidate for candidate in valid_list if candidate.startswith("search[")]
        if search_actions:
            return search_actions[-1]

    if action.startswith("click["):
        target = action[6:-1].lower()
        for candidate in valid_list:
            if candidate.startswith("click[") and target in candidate.lower():
                return candidate

    return valid_list[0]


def run_episode(llm, env, env_idx, insight_store, max_steps, max_inject):
    """Run one WebShop episode with optional ExpeL insight injection."""
    t0 = time.time()
    llm.tracker.reset()
    init_obs, task_type, info = env.reset(session_idx=env_idx)
    valid_actions = info.get("available_actions", [])

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
        prompt = build_expel_prompt(init_obs, history, insights, valid_actions=valid_actions)
        response = llm.complete_text(prompt, stop=["\n"], label=f"step_{step_num}",
                                     system=SYSTEM_PROMPT_BASE)
        action = normalize_expel_action(response, valid_actions)

        logger.info(f"  Step {step_num}: {action[:80]}")

        if action.startswith("think:") or action.startswith("think "):
            steps.append({"step": step_num, "action": action, "observation": "OK."})
            history.append((action, "OK."))
            continue

        observation, reward, done, step_info = env.step(action)
        valid_actions = step_info.get("available_actions", valid_actions)
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
        "success": is_exact_success(final_reward),
        "success_definition": "reward == 1.0",
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
    num_products = parse_num_products(args.num_products)
    max_envs = args.max_envs or args.eval_sample_size
    session_ids = load_or_sample_session_ids(
        args.sample_ids,
        split=args.eval_split,
        sample_size=max_envs,
        seed=args.eval_sample_seed,
    )

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

        env = WebShopEnv(
            num_products=num_products,
            observation_mode=args.observation_mode,
            max_sessions=len(session_ids),
            human_goals=args.human_goals,
            split=args.eval_split,
            step_limit=args.max_steps,
            wrapper=args.webshop_wrapper,
        )
        env.setup()

        episode_results = []
        sample_pos = 0
        num_skipped = 0

        while sample_pos < len(session_ids):
            session_id = session_ids[sample_pos]
            try:
                if epoch > 1 and session_id in env_success:
                    env.skip()
                    episode_results.append({
                        "env_idx": session_id, "task_type": "shopping",
                        "task_description": "", "success": True,
                        "success_definition": "reward == 1.0",
                        "reward": env_success[session_id],
                        "total_steps": 0, "total_tokens": 0,
                        "wall_time_s": 0, "skipped": True,
                    })
                    num_skipped += 1
                    sample_pos += 1
                    continue

                store_to_use = insight_store if epoch > 1 else None
                result = run_episode(
                    llm, env, session_id, store_to_use,
                    args.max_steps, args.max_insights_inject,
                )
                result["sample_position"] = sample_pos
                result["eval_split"] = args.eval_split
                episode_results.append(result)

                if result["success"]:
                    env_success[session_id] = result["reward"]

                if epoch == 1:
                    all_trajectories.append(result)

                sample_pos += 1
                rewards = [r.get("reward", 0) for r in episode_results]
                avg_r = sum(rewards) / len(rewards)
                status = "OK" if result["success"] else "FAIL"
                logger.info(f"  [{sample_pos}/{len(session_ids)}] session={session_id} "
                            f"{status} reward={result['reward']:.4f} "
                            f"running_avg={avg_r:.4f}")

                if sample_pos % 10 == 0:
                    summary = compute_summary(episode_results, insight_store.stats(), "expel")
                    save_results(episode_results, summary,
                                 f"{results_dir}/{args.run_name}{epoch}_intermediate.json")
                    save_sample_ids(session_ids, f"{results_dir}/{args.run_name}{epoch}_sample_ids.json")

            except Exception as e:
                logger.error(f"Error on session #{session_id}: {e}", exc_info=True)
                episode_results.append({
                    "env_idx": session_id,
                    "sample_position": sample_pos,
                    "eval_split": args.eval_split,
                    "task_type": "shopping",
                    "task_description": "",
                    "success": False,
                    "success_definition": "reward == 1.0",
                    "reward": 0.0,
                    "total_steps": 0,
                    "total_tokens": 0,
                    "wall_time_s": 0,
                    "skipped": False,
                    "error": str(e),
                })
                sample_pos += 1

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
        summary.update({
            "eval_split": args.eval_split,
            "eval_sample_size": len(session_ids),
            "eval_sample_seed": args.eval_sample_seed,
            "num_products": "full" if num_products is None else num_products,
            "human_goals": args.human_goals,
            "max_steps": args.max_steps,
            "webshop_wrapper": args.webshop_wrapper,
        })
        logger.info(f"  Epoch {epoch}: {summary['total_success']}/{summary['total_envs']} "
                    f"({summary['success_rate']:.1%}), avg_reward={summary['avg_reward']:.4f}")

        # Remove history before saving
        for ep in episode_results:
            ep.pop("history", None)
        result_path = f"{results_dir}/{args.run_name}{epoch}.json"
        save_results(episode_results, summary, result_path)
        save_sample_ids(session_ids, f"{results_dir}/{args.run_name}{epoch}_sample_ids.json")
        logger.info(f"Results saved: {result_path}")

        env.close()

    logger.info("Done!")


if __name__ == "__main__":
    main()

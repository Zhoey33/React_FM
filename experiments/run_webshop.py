"""Run paper-aligned React_FM (or baseline ReAct) on WebShop."""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import logging
import sys
import random
import time
from collections.abc import Sequence
from pathlib import Path

import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import LLMClient, RollingTokenRateLimiter
from src.memory import FailureMemoryStore
from src.webshop_failure_detector import WebShopFailureDetector
from src.webshop_env import WebShopEnv
from src.log_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run React_FM on WebShop")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--max-envs", type=int, default=None, help="Max sessions")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--baseline", action="store_true", help="Vanilla ReAct")
    parser.add_argument("--resume-memory", type=str, default=None)
    parser.add_argument("--resume-results", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--inject-mode", choices=["in_loop", "episode", "none"], default="in_loop")
    parser.add_argument("--memory-style", choices=["original", "factual", "reflexion", "hint"], default="original")
    parser.add_argument("--memory-format", choices=["failure_recovery", "success_trajectory", "reflexion_reflection"],
                        default="failure_recovery", help="Memory storage format for ablation")
    parser.add_argument("--retrieval-mode", choices=["hybrid", "bm25_only", "embedding_only", "random"],
                        default="hybrid", help="Retrieval method for ablation")
    parser.add_argument("--cross-env", action="store_true",
                        help="Enable cross-env memory sharing (retrieve from all envs)")
    parser.add_argument("--num-products", type=str, default="full",
                        help="Product count: 'full' for paper-aligned full WebShop, or preview sizes such as 1000")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--eval-split", choices=["test", "eval", "train"], default="test")
    parser.add_argument("--eval-sample-size", type=int, default=100,
                        help="Fixed subset size from the official split")
    parser.add_argument("--eval-sample-seed", type=int, default=42,
                        help="Seed for the fixed official-split subset")
    parser.add_argument("--memory-setting", choices=["online", "frozen"], default="online",
                        help="Online writes test-time memory; frozen is read-only during test")
    parser.add_argument("--human-goals", type=int, default=1, help="Use crowd-sourced human instructions")
    parser.add_argument("--observation-mode", choices=["html", "text", "text_rich", "url"], default="text_rich",
                        help="WebShop observation mode; official baseline uses text_rich")
    parser.add_argument("--webshop-wrapper", choices=["official", "direct"], default="official",
                        help="Use official WebShop wrapper/valid-action interface or legacy direct text env")
    parser.add_argument("--sample-ids", type=str, default=None,
                        help="Optional JSON list of official split session IDs to run")
    parser.add_argument("--llm-tpm-budget", type=int, default=12000,
                        help="Shared LLM token-per-minute budget across agent/judge/extractor; 0 disables")
    parser.add_argument("--agent-max-tokens", type=int, default=96,
                        help="Max completion tokens for each WebShop action call")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def validate_llm_throttle_args(llm_tpm_budget: int, agent_max_tokens: int) -> None:
    """Validate WebShop LLM throttling CLI arguments."""
    if llm_tpm_budget < 0:
        raise ValueError("--llm-tpm-budget must be 0 or a positive integer")
    if agent_max_tokens <= 0:
        raise ValueError("--agent-max-tokens must be a positive integer")


def save_results(results: list[dict], summary: dict, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    data = {"summary": summary, "episodes": results}
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_num_products(value: str) -> int | None:
    """Parse product-count values, where None means full WebShop."""
    lowered = str(value).lower()
    if lowered in {"full", "all", "none", "null"}:
        return None
    return int(value)


def memory_scope_for_setting(memory_setting: str) -> str:
    """Return WebShop memory scope; both online and frozen share shopping memory."""
    if memory_setting not in {"online", "frozen"}:
        raise ValueError(f"Unsupported memory setting: {memory_setting}")
    return "task_type"


def load_memory_for_setting(
    memory_store: FailureMemoryStore,
    memory_path: str,
    memory_setting: str,
) -> None:
    """Load WebShop memory and reject non-shared legacy scopes."""
    validate_memory_requirements(memory_setting, memory_path)
    expected_scope = memory_scope_for_setting(memory_setting)
    memory_store.load(memory_path)
    if memory_store.scope != expected_scope:
        raise ValueError(
            "WebShop memory must use shared task_type scope "
            f"for {memory_setting}; got scope={memory_store.scope!r} from {memory_path}"
        )
    bucket_keys = memory_store.get_env_ids()
    if memory_store.size() > 0 and "shopping" not in bucket_keys:
        raise ValueError(
            "WebShop memory must be stored under task_type='shopping'; "
            f"got buckets={sorted(bucket_keys)} from {memory_path}"
        )
    if memory_setting == "frozen" and memory_store.size() == 0:
        raise ValueError(
            "Frozen WebShop memory must be non-empty; "
            f"loaded 0 entries from {memory_path}"
        )


def validate_memory_requirements(memory_setting: str, memory_path: str | None) -> None:
    """Validate memory-file requirements for WebShop memory settings."""
    if memory_setting != "frozen":
        return
    if not memory_path:
        raise ValueError("Frozen WebShop runs require --resume-memory with offline shopping memory")
    if not Path(memory_path).exists():
        raise FileNotFoundError(f"Frozen WebShop resume-memory file is missing: {memory_path}")


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
    """Validate user-provided WebShop session IDs against the requested split."""
    if len(set(session_ids)) != len(session_ids):
        raise ValueError("Duplicate WebShop session IDs are not allowed")
    allowed = set(official_split_session_ids(split))
    outside = [idx for idx in session_ids if idx not in allowed]
    if outside:
        preview = outside[:10]
        raise ValueError(
            f"Sample IDs outside official {split} split: {preview}"
        )


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


def save_sample_ids(session_ids: Sequence[int], filepath: str):
    """Save the fixed WebShop session list for reproducibility."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(list(session_ids), f, indent=2)


def is_exact_success(reward: float) -> bool:
    """Return paper-aligned WebShop success: reward must be exactly 1.0."""
    return reward == 1.0


def compute_summary(
    results: list[dict],
    memory_stats: dict,
    mode: str,
    memory_setting: str = "online",
) -> dict:
    total = len(results)
    rewards = [r.get("reward", 0.0) for r in results]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    successes = sum(1 for r in rewards if is_exact_success(r))

    total_tokens = sum(r["total_tokens"] for r in results)
    agent_tokens = sum(r.get("agent_tokens", r["total_tokens"]) for r in results)
    judge_tokens = sum(r.get("judge_tokens", 0) for r in results)
    extractor_tokens = sum(r.get("extractor_tokens", 0) for r in results)

    return {
        "mode": mode,
        "benchmark": "webshop",
        "memory_setting": memory_setting,
        "success_definition": "reward == 1.0",
        "total_envs": total,
        "total_success": successes,
        "success_rate": round(successes / total, 4) if total > 0 else 0,
        "avg_reward": round(avg_reward, 4),
        "task_score": round(100 * avg_reward, 2),
        "total_tokens": total_tokens,
        "agent_tokens": agent_tokens,
        "judge_tokens": judge_tokens,
        "extractor_tokens": extractor_tokens,
        "avg_tokens_per_episode": round(total_tokens / total) if total > 0 else 0,
        "memory_stats": memory_stats,
    }


def make_error_episode_result(
    env_idx: int,
    sample_position: int,
    eval_split: str,
    error: Exception,
) -> dict:
    """Create a failed episode record for infrastructure/runtime errors."""
    return {
        "env_idx": env_idx,
        "sample_position": sample_position,
        "eval_split": eval_split,
        "task_type": "shopping",
        "task_description": "",
        "success": False,
        "success_definition": "reward == 1.0",
        "reward": 0.0,
        "total_steps": 0,
        "total_tokens": 0,
        "agent_tokens": 0,
        "judge_tokens": 0,
        "extractor_tokens": 0,
        "failures_detected": 0,
        "memories_retrieved": 0,
        "memories_stored": 0,
        "wall_time_s": 0,
        "error": str(error),
        "steps": [],
    }


def log_summary(summary: dict):
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  {summary['mode']} Results (WebShop)")
    logger.info("=" * 60)
    logger.info(f"  Success (reward == 1.0): {summary['total_success']}/{summary['total_envs']} "
                f"({summary['success_rate']:.1%})")
    logger.info(f"  Avg Reward: {summary['avg_reward']:.4f}")
    logger.info(f"  Task Score: {summary['task_score']:.2f}")
    logger.info("-" * 60)
    logger.info(f"  Tokens: {summary['total_tokens']:,} total, "
                f"{summary['avg_tokens_per_episode']:,} avg/episode")
    if summary["memory_stats"]:
        ms = summary["memory_stats"]
        logger.info(f"  Memory: {ms.get('total_entries', 0)} entries, "
                    f"{ms.get('total_retrievals', 0)} retrievals, "
                    f"{ms.get('total_hits', 0)} hits")
    logger.info("=" * 60)


class WebShopReActAgent:
    """ReAct agent for WebShop."""

    def __init__(
        self,
        llm: LLMClient,
        memory_store: FailureMemoryStore | None = None,
        failure_detector: WebShopFailureDetector | None = None,
        extractor_llm: LLMClient | None = None,
        max_steps: int = 15,
        max_memory_inject: int = 3,
        enable_memory: bool = True,
        inject_mode: str = "in_loop",
        memory_style: str = "original",
        memory_format: str = "failure_recovery",
        cross_env: bool = False,
        allow_memory_updates: bool = True,
    ):
        self.llm = llm
        self.memory = memory_store
        self.detector = failure_detector or WebShopFailureDetector()
        self.extractor_llm = extractor_llm
        self.max_steps = max_steps
        self.max_memory_inject = max_memory_inject
        self.enable_memory = enable_memory and (memory_store is not None)
        self.inject_mode = inject_mode
        self.memory_style = memory_style
        self.memory_format = memory_format
        self.cross_env = cross_env
        self.allow_memory_updates = allow_memory_updates

    def run_episode(self, env, env_idx: int = 0) -> dict:
        from prompts.webshop_prompts import build_user_prompt, SYSTEM_PROMPT_FM
        from src.webshop_memory_extractor import extract_failure_recoveries

        t0 = time.time()
        self.llm.tracker.reset()
        if self.detector.judge_llm is not None:
            self.detector.judge_llm.tracker.reset()
        if self.extractor_llm is not None:
            self.extractor_llm.tracker.reset()

        init_obs, task_type, info = env.reset(session_idx=env_idx)
        valid_actions = info.get("available_actions", [])

        history: list[tuple[str, str]] = []
        action_history: list[str] = []
        steps: list[dict] = []
        final_reward = 0.0

        current_retrieved = None
        failures_detected = 0
        memories_retrieved_total = 0

        episode_memories = None
        if self.enable_memory and self.inject_mode == "episode":
            all_mem = self.memory.get_all(
                env_idx=env_idx,
                task_type=task_type,
                cross_env=self.cross_env,
            )
            if all_mem:
                episode_memories = all_mem[:self.max_memory_inject]
                memories_retrieved_total = len(episode_memories)

        for step_num in range(self.max_steps):
            if self.inject_mode == "episode":
                prompt = build_user_prompt(
                    task_type=task_type, task_obs=init_obs, history=history,
                    retrieved_memories=episode_memories, memory_style=self.memory_style,
                    valid_actions=valid_actions,
                )
            else:
                prompt = build_user_prompt(
                    task_type=task_type, task_obs=init_obs, history=history,
                    retrieved_memories=current_retrieved, memory_style=self.memory_style,
                    valid_actions=valid_actions,
                )

            response = self.llm.complete_text(
                prompt, stop=["\n"], label=f"step_{step_num}",
                system=SYSTEM_PROMPT_FM,
            )
            action = response.strip().split("\n")[0].strip()

            if self.inject_mode == "in_loop":
                current_retrieved = None

            if action.startswith("> "):
                action = action[2:]

            action = self._normalize_action(action, valid_actions)

            logger.info(f"  Step {step_num}: {action[:80]}")

            # Handle think
            is_think = action.startswith("think:") or action.startswith("think ")
            if is_think:
                observation = "OK."
                steps.append({
                    "step": step_num, "action": action, "observation": observation,
                    "is_think": True, "failure_detected": False, "memory_retrieved": 0,
                })
                history.append((action, observation))
                continue

            observation, reward, done, step_info = env.step(action)
            valid_actions = step_info.get("available_actions", valid_actions)
            logger.info(f"    obs: {observation[:80]}")
            action_history.append(action)
            final_reward = max(final_reward, reward)

            is_done, _ = self.detector.is_task_complete(observation, done, step_info)

            record = {
                "step": step_num, "action": action, "observation": observation[:200],
                "is_think": False, "failure_detected": False, "memory_retrieved": 0,
            }

            if self.enable_memory and self.inject_mode in ("in_loop", "none") and not is_done:
                det = self.detector.detect(observation, action, action_history)
                if det.is_failure:
                    record["failure_detected"] = True
                    failures_detected += 1
                    logger.info(f"    FAILURE detected: {det.failure_type}")

                    if self.inject_mode == "in_loop":
                        retrieved = self.memory.retrieve(
                            query_action=action, query_observation=observation,
                            task_type=task_type, top_k=self.max_memory_inject,
                            env_idx=env_idx, cross_env=self.cross_env,
                        )
                        record["memory_retrieved"] = len(retrieved)
                        memories_retrieved_total += len(retrieved)
                        if retrieved:
                            current_retrieved = retrieved

            steps.append(record)
            history.append((action, observation))

            if is_done:
                break

        # Post-episode memory extraction
        memories_stored = 0
        if self.enable_memory and self.extractor_llm is not None and self.allow_memory_updates:
            env_history = [(a, o) for a, o in history if not a.startswith("think")]
            if self.memory_format == "success_trajectory":
                from src.memory_extractor_ablation import extract_success_trajectories
                recoveries = extract_success_trajectories(self.extractor_llm, env_history, is_exact_success(final_reward))
            elif self.memory_format == "reflexion_reflection":
                from src.memory_extractor_ablation import extract_reflexion_reflections
                recoveries = extract_reflexion_reflections(self.extractor_llm, env_history)
            else:
                recoveries = extract_failure_recoveries(self.extractor_llm, env_history)
            for rec in recoveries:
                self.memory.add(
                    failure_action=rec["failure_action"],
                    failure_observation=rec["failure_observation"],
                    solution_action=rec["solution_action"],
                    task_type=task_type, env_idx=env_idx,
                )
                memories_stored += 1

        wall_time = time.time() - t0
        agent_stats = self.llm.tracker.summary()
        judge_tokens = 0
        if self.detector.judge_llm is not None:
            judge_tokens = self.detector.judge_llm.tracker.summary()["total_tokens"]
        extractor_tokens = 0
        if self.extractor_llm is not None:
            extractor_tokens = self.extractor_llm.tracker.summary()["total_tokens"]
        total_tokens = agent_stats["total_tokens"] + judge_tokens + extractor_tokens

        result = {
            "env_idx": env_idx,
            "task_type": task_type,
            "task_description": init_obs[:200],
            "success": is_exact_success(final_reward),
            "success_definition": "reward == 1.0",
            "reward": final_reward,
            "total_steps": len(steps),
            "total_tokens": total_tokens,
            "agent_tokens": agent_stats["total_tokens"],
            "judge_tokens": judge_tokens,
            "extractor_tokens": extractor_tokens,
            "failures_detected": failures_detected,
            "memories_retrieved": memories_retrieved_total,
            "memories_stored": memories_stored,
            "wall_time_s": round(wall_time, 2),
            "skipped": False,
            "steps": steps,
        }

        status = "SUCCESS" if is_exact_success(final_reward) else "FAIL"
        logger.info(
            f"Env #{env_idx} [{task_type}] {status} reward={final_reward:.4f} in {len(steps)} steps"
        )
        return result

    @staticmethod
    def _normalize_action(action: str, valid_actions: list[str] | dict | None) -> str:
        """Coerce model output to one of the official valid actions when available."""
        import re

        if isinstance(valid_actions, dict):
            valid_list = []
        else:
            valid_list = list(valid_actions or [])

        search_match = re.search(r"search\[([^\]]+)\]", action)
        click_match = re.search(r"click\[([^\]]+)\]", action)
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
            search_actions = [candidate for candidate in valid_list if candidate.startswith("search[")]
            if search_actions:
                return search_actions[-1]

        if action.startswith("click["):
            target = action[6:-1].lower()
            for candidate in valid_list:
                if candidate.startswith("click[") and target in candidate.lower():
                    return candidate

        return valid_list[0]


def main():
    args = parse_args()
    validate_llm_throttle_args(args.llm_tpm_budget, args.agent_max_tokens)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = args.seed or config["experiment"]["seed"]
    set_seed(seed)
    num_epochs = args.epochs or config["experiment"].get("num_epochs", 1)
    is_baseline = args.baseline
    mode = "react_baseline" if is_baseline else "react_fm"
    run_name = args.run_name or f"ws_{mode}_"
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
    memory_dir = f"{config['memory']['persist_dir']}/{timestamp}" if not is_baseline else None

    setup_logging(
        log_level=config["experiment"]["log_level"],
        log_dir="logs",
        run_name=run_name.rstrip("_"),
    )

    rate_limiter = None
    if args.llm_tpm_budget and args.llm_tpm_budget > 0:
        rate_limiter = RollingTokenRateLimiter(tokens_per_minute=args.llm_tpm_budget)
        logger.info("LLM TPM limiter enabled: %s tokens/minute", args.llm_tpm_budget)

    agent_max_tokens = args.agent_max_tokens or config["llm"]["max_tokens"]
    logger.info("WebShop agent max_tokens: %s", agent_max_tokens)

    llm = LLMClient(
        model=config["llm"]["model"],
        base_url=config["llm"]["base_url"],
        api_key=config["llm"].get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
        temperature=config["llm"]["temperature"],
        max_tokens=agent_max_tokens,
        rate_limiter=rate_limiter,
    )

    memory_store = None
    if not is_baseline:
        validate_memory_requirements(args.memory_setting, args.resume_memory)
        memory_store = FailureMemoryStore(
            embedding_model_name=config["memory"]["embedding_model"],
            max_entries=config["memory"]["max_entries"],
            top_k=config["memory"]["retrieval_top_k"],
            retrieval_mode=args.retrieval_mode,
            scope=memory_scope_for_setting(args.memory_setting),
        )
        if args.resume_memory:
            load_memory_for_setting(memory_store, args.resume_memory, args.memory_setting)

    judge_llm = None
    if not is_baseline and "judge" in config:
        judge_cfg = config["judge"]
        judge_llm = LLMClient(
            model=judge_cfg["model"],
            base_url=judge_cfg["base_url"],
            api_key=judge_cfg.get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
            temperature=judge_cfg.get("temperature", 0.0),
            max_tokens=judge_cfg.get("max_tokens", 16),
            rate_limiter=rate_limiter,
        )

    detector = WebShopFailureDetector(judge_llm=judge_llm)

    extractor_llm = None
    if not is_baseline and "extractor" in config:
        ext_cfg = config["extractor"]
        extractor_llm = LLMClient(
            model=ext_cfg["model"],
            base_url=ext_cfg["base_url"],
            api_key=ext_cfg.get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
            temperature=ext_cfg.get("temperature", 0.0),
            max_tokens=ext_cfg.get("max_tokens", 512),
            rate_limiter=rate_limiter,
        )

    agent = WebShopReActAgent(
        llm=llm,
        memory_store=memory_store,
        failure_detector=detector,
        extractor_llm=extractor_llm,
        max_steps=args.max_steps,
        max_memory_inject=config["agent"]["max_memory_inject"],
        enable_memory=not is_baseline,
        inject_mode=args.inject_mode,
        memory_style=args.memory_style,
        memory_format=args.memory_format,
        cross_env=args.cross_env,
        allow_memory_updates=args.memory_setting == "online",
    )
    run_metadata = {
        "llm_tpm_budget": args.llm_tpm_budget,
        "agent_max_tokens": agent_max_tokens,
    }

    env_success: dict[int, float] = {}

    if args.resume_results:
        with open(args.resume_results) as f:
            prev_data = json.load(f)
        for ep in prev_data.get("episodes", []):
            idx = ep["env_idx"]
            if is_exact_success(ep.get("reward", 0)):
                env_success[idx] = ep["reward"]

    for epoch in range(1, num_epochs + 1):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  Epoch {epoch}/{num_epochs} — {mode} (WebShop)")
        logger.info("=" * 60)

        env = WebShopEnv(
            num_products=num_products,
            observation_mode=args.observation_mode,
            max_sessions=max_envs,
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
                        "env_idx": session_id,
                        "task_type": "shopping",
                        "task_description": "",
                        "success": True,
                        "success_definition": "reward == 1.0",
                        "reward": env_success[session_id],
                        "total_steps": 0,
                        "total_tokens": 0,
                        "agent_tokens": 0,
                        "judge_tokens": 0,
                        "extractor_tokens": 0,
                        "failures_detected": 0,
                        "memories_retrieved": 0,
                        "memories_stored": 0,
                        "wall_time_s": 0,
                        "skipped": True,
                        "steps": [],
                    })
                    num_skipped += 1
                    sample_pos += 1
                    continue

                result = agent.run_episode(env, env_idx=session_id)
                result["sample_position"] = sample_pos
                result["eval_split"] = args.eval_split
                episode_results.append(result)

                if is_exact_success(result["reward"]):
                    env_success[session_id] = result["reward"]

                sample_pos += 1

                rewards = [r.get("reward", 0) for r in episode_results]
                avg_r = sum(rewards) / len(rewards)
                logger.info(f"  [{sample_pos}/{len(session_ids)}] session={session_id} "
                            f"reward={result['reward']:.4f} "
                            f"running_avg={avg_r:.4f}")

                if sample_pos % 10 == 0:
                    mem_stats = memory_store.stats() if memory_store else {}
                    summary = compute_summary(
                        episode_results,
                        mem_stats,
                        mode,
                        memory_setting=args.memory_setting,
                    )
                    summary.update(run_metadata)
                    save_results(
                        episode_results, summary,
                        f"{results_dir}/{run_name}{epoch}_intermediate.json"
                    )
                    save_sample_ids(session_ids, f"{results_dir}/{run_name}{epoch}_sample_ids.json")
                    if memory_store:
                        memory_store.save(f"{memory_dir}/ws_epoch{epoch}_intermediate.json")

            except Exception as e:
                logger.error(f"Error on session #{session_id}: {e}", exc_info=True)
                episode_results.append(
                    make_error_episode_result(
                        env_idx=session_id,
                        sample_position=sample_pos,
                        eval_split=args.eval_split,
                        error=e,
                    )
                )
                sample_pos += 1

        if num_skipped > 0:
            logger.info(f"  Skipped {num_skipped} already-succeeded envs")

        mem_stats = memory_store.stats() if memory_store else {}
        summary = compute_summary(
            episode_results,
            mem_stats,
            mode,
            memory_setting=args.memory_setting,
        )
        summary.update({
            "eval_split": args.eval_split,
            "eval_sample_size": len(session_ids),
            "eval_sample_seed": args.eval_sample_seed,
            "num_products": "full" if num_products is None else num_products,
            "human_goals": args.human_goals,
            "max_steps": args.max_steps,
            "webshop_wrapper": args.webshop_wrapper,
        })
        summary.update(run_metadata)
        log_summary(summary)

        result_path = f"{results_dir}/{run_name}{epoch}.json"
        save_results(episode_results, summary, result_path)
        sample_path = f"{results_dir}/{run_name}{epoch}_sample_ids.json"
        save_sample_ids(session_ids, sample_path)
        logger.info(f"Results saved: {result_path}")
        logger.info(f"Sample IDs saved: {sample_path}")

        if memory_store:
            mem_path = f"{memory_dir}/ws_epoch{epoch}.json"
            memory_store.save(mem_path)
            logger.info(f"Memory saved: {mem_path} ({memory_store.size()} entries)")

        env.close()

    logger.info("Done!")


if __name__ == "__main__":
    main()

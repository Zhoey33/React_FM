"""Run React_FM (or baseline ReAct) on HotPotQA."""

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
from src.memory import FailureMemoryStore
from src.hotpotqa_failure_detector import HotPotQAFailureDetector
from src.hotpotqa_env import HotPotQAEnv, exact_match_score, f1_score
from src.log_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run React_FM on HotPotQA")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--inject-mode", choices=["in_loop", "episode", "none"], default="in_loop")
    parser.add_argument("--memory-style", choices=["original", "factual", "reflexion", "hint"], default="original")
    parser.add_argument("--memory-format", choices=["failure_recovery", "success_trajectory", "reflexion_reflection"],
                        default="failure_recovery")
    parser.add_argument("--retrieval-mode", choices=["hybrid", "bm25_only", "embedding_only", "random"],
                        default="hybrid")
    parser.add_argument("--cross-env", action="store_true",
                        help="Enable cross-env memory sharing (retrieve from all envs)")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--num-examples", type=int, default=500)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def save_results(results: list[dict], summary: dict, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump({"summary": summary, "episodes": results}, f, indent=2, ensure_ascii=False)


def compute_summary(results: list[dict], memory_stats: dict, mode: str) -> dict:
    total = len(results)
    em_scores = [r.get("em", 0.0) for r in results]
    f1_scores = [r.get("f1", 0.0) for r in results]
    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    successes = sum(1 for r in results if r.get("em", 0.0) > 0)
    total_tokens = sum(r.get("total_tokens", 0) for r in results)
    agent_tokens = sum(r.get("agent_tokens", r.get("total_tokens", 0)) for r in results)
    judge_tokens = sum(r.get("judge_tokens", 0) for r in results)
    extractor_tokens = sum(r.get("extractor_tokens", 0) for r in results)
    return {
        "mode": mode,
        "benchmark": "hotpotqa",
        "total_envs": total,
        "total_success": successes,
        "success_rate": round(successes / total, 4) if total > 0 else 0,
        "avg_em": round(avg_em, 4),
        "avg_f1": round(avg_f1, 4),
        "total_tokens": total_tokens,
        "agent_tokens": agent_tokens,
        "judge_tokens": judge_tokens,
        "extractor_tokens": extractor_tokens,
        "avg_tokens_per_episode": round(total_tokens / total) if total > 0 else 0,
        "memory_stats": memory_stats,
    }


def _parse_thought_action(response: str) -> tuple[str, str]:
    """Parse LLM response into (thought, action)."""
    thought = ""
    action = ""
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.lower().startswith("thought:"):
            thought = line[len("thought:"):].strip()
        elif line.lower().startswith("action:"):
            action = line[len("action:"):].strip()
    if not action:
        # Try to extract action pattern directly
        m = re.search(r'(Search\[.+?\]|Lookup\[.+?\]|Finish\[.+?\])', response)
        if m:
            action = m.group(1)
    return thought, action


class HotPotQAReActAgent:
    """ReAct agent for HotPotQA."""

    def __init__(
        self,
        llm: LLMClient,
        memory_store: FailureMemoryStore | None = None,
        failure_detector: HotPotQAFailureDetector | None = None,
        extractor_llm: LLMClient | None = None,
        max_steps: int = 8,
        max_memory_inject: int = 3,
        enable_memory: bool = True,
        inject_mode: str = "in_loop",
        memory_style: str = "original",
        memory_format: str = "failure_recovery",
        cross_env: bool = False,
    ):
        self.llm = llm
        self.memory = memory_store
        self.detector = failure_detector or HotPotQAFailureDetector()
        self.extractor_llm = extractor_llm
        self.max_steps = max_steps
        self.max_memory_inject = max_memory_inject
        self.enable_memory = enable_memory and (memory_store is not None)
        self.inject_mode = inject_mode
        self.memory_style = memory_style
        self.memory_format = memory_format
        self.cross_env = cross_env

    def run_episode(self, env: HotPotQAEnv, env_idx: int = 0) -> dict:
        from prompts.hotpotqa_prompts import build_user_prompt, SYSTEM_PROMPT_FM
        from src.hotpotqa_memory_extractor import extract_failure_recoveries

        t0 = time.time()
        self.llm.tracker.reset()
        if self.detector.judge_llm is not None:
            self.detector.judge_llm.tracker.reset()
        if self.extractor_llm is not None:
            self.extractor_llm.tracker.reset()

        init_obs, task_type, info = env.reset()
        gold_answer = info["gold_answer"]

        history: list[tuple[str, str, str]] = []  # (thought, action, observation)
        action_history: list[str] = []
        steps: list[dict] = []
        final_em = 0.0
        final_f1 = 0.0
        final_answer = ""

        current_retrieved = None
        failures_detected = 0
        memories_retrieved_total = 0

        episode_memories = None
        if self.enable_memory and self.inject_mode == "episode":
            all_mem = self.memory.get_all(env_idx=env_idx, cross_env=self.cross_env)
            if all_mem:
                episode_memories = all_mem[:self.max_memory_inject]
                memories_retrieved_total = len(episode_memories)

        for step_num in range(self.max_steps):
            if self.inject_mode == "episode":
                prompt = build_user_prompt(
                    task_obs=init_obs, history=history,
                    retrieved_memories=episode_memories, memory_style=self.memory_style,
                )
            else:
                prompt = build_user_prompt(
                    task_obs=init_obs, history=history,
                    retrieved_memories=current_retrieved, memory_style=self.memory_style,
                )

            response = self.llm.complete_text(
                prompt, label=f"step_{step_num}", system=SYSTEM_PROMPT_FM,
            )
            thought, action = _parse_thought_action(response)

            if self.inject_mode == "in_loop":
                current_retrieved = None

            if not action:
                action = "Finish[unknown]"

            logger.info(f"  Step {step_num}: {action[:80]}")

            observation, reward, done, step_info = env.step(action)
            logger.info(f"    obs: {observation[:80]}")
            action_history.append(action)

            record = {
                "step": step_num, "thought": thought, "action": action,
                "observation": observation[:300],
                "failure_detected": False, "memory_retrieved": 0,
            }

            if done:
                final_answer = step_info.get("gold_answer", observation)
                final_em = step_info.get("em", 0.0)
                final_f1 = step_info.get("f1", 0.0)
                final_answer = action[7:-1] if action.startswith("Finish[") else observation
                steps.append(record)
                history.append((thought, action, observation))
                break

            # Failure detection + memory retrieval
            if self.enable_memory and self.inject_mode in ("in_loop", "none") and not done:
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
            history.append((thought, action, observation))

        # If agent didn't finish, score is 0
        if not any(s["action"].startswith("Finish[") for s in steps):
            final_em = 0.0
            final_f1 = 0.0

        # Post-episode memory extraction
        memories_stored = 0
        if self.enable_memory and self.extractor_llm is not None:
            recoveries = extract_failure_recoveries(self.extractor_llm, history)
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
            "question": info["question"],
            "gold_answer": gold_answer,
            "predicted_answer": final_answer,
            "em": final_em,
            "f1": final_f1,
            "success": final_em > 0,
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
        }

        status = "OK" if final_em > 0 else "FAIL"
        logger.info(f"Env #{env_idx} [{task_type}] {status} em={final_em:.1f} f1={final_f1:.2f}")
        return result


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = args.seed or config["experiment"]["seed"]
    set_seed(seed)
    num_epochs = args.epochs or config["experiment"].get("num_epochs", 1)
    is_baseline = args.baseline
    mode = "react_baseline" if is_baseline else "react_fm"
    run_name = args.run_name or f"hqa_{mode}_"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = f"{config['experiment']['results_dir']}/{timestamp}"
    memory_dir = f"{config['memory']['persist_dir']}/{timestamp}" if not is_baseline else None

    setup_logging(
        log_level=config["experiment"]["log_level"],
        log_dir="logs",
        run_name=run_name.rstrip("_"),
    )

    llm = LLMClient(
        model=config["llm"]["model"],
        base_url=config["llm"]["base_url"],
        api_key=config["llm"].get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
    )

    memory_store = None
    if not is_baseline:
        memory_store = FailureMemoryStore(
            embedding_model_name=config["memory"]["embedding_model"],
            max_entries=config["memory"]["max_entries"],
            top_k=config["memory"]["retrieval_top_k"],
            retrieval_mode=args.retrieval_mode,
        )

    judge_llm = None
    if not is_baseline and "judge" in config:
        judge_cfg = config["judge"]
        judge_llm = LLMClient(
            model=judge_cfg["model"],
            base_url=judge_cfg["base_url"],
            api_key=judge_cfg.get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
            temperature=judge_cfg.get("temperature", 0.0),
            max_tokens=judge_cfg.get("max_tokens", 16),
        )

    detector = HotPotQAFailureDetector(judge_llm=judge_llm)

    extractor_llm = None
    if not is_baseline and "extractor" in config:
        ext_cfg = config["extractor"]
        extractor_llm = LLMClient(
            model=ext_cfg["model"],
            base_url=ext_cfg["base_url"],
            api_key=ext_cfg.get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
            temperature=ext_cfg.get("temperature", 0.0),
            max_tokens=ext_cfg.get("max_tokens", 512),
        )

    agent = HotPotQAReActAgent(
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
    )

    max_envs = args.max_envs or args.num_examples
    env_success: dict[int, float] = {}

    for epoch in range(1, num_epochs + 1):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  Epoch {epoch}/{num_epochs} — {mode} (HotPotQA)")
        logger.info("=" * 60)

        env = HotPotQAEnv(
            num_examples=max_envs, seed=seed,
        )
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
                        "agent_tokens": 0, "judge_tokens": 0, "extractor_tokens": 0,
                        "failures_detected": 0, "memories_retrieved": 0,
                        "memories_stored": 0, "wall_time_s": 0, "skipped": True,
                    })
                    num_skipped += 1
                    env_count += 1
                    continue

                result = agent.run_episode(env, env_idx=env_count)
                episode_results.append(result)

                if result["em"] > 0:
                    env_success[env_count] = result["em"]

                env_count += 1

                em_scores = [r.get("em", 0) for r in episode_results]
                f1_scores_list = [r.get("f1", 0) for r in episode_results]
                avg_em = sum(em_scores) / len(em_scores)
                avg_f1 = sum(f1_scores_list) / len(f1_scores_list)
                logger.info(f"  [{env_count}] running_em={avg_em:.3f} running_f1={avg_f1:.3f}")

                if env_count % 10 == 0:
                    mem_stats = memory_store.stats() if memory_store else {}
                    summary = compute_summary(episode_results, mem_stats, mode)
                    save_results(
                        episode_results, summary,
                        f"{results_dir}/{run_name}{epoch}_intermediate.json"
                    )
                    if memory_store:
                        memory_store.save(f"{memory_dir}/hqa_epoch{epoch}_intermediate.json")

            except StopIteration:
                break
            except Exception as e:
                logger.error(f"Error on env #{env_count}: {e}", exc_info=True)
                env_count += 1

        if num_skipped > 0:
            logger.info(f"  Skipped {num_skipped} already-succeeded envs")

        mem_stats = memory_store.stats() if memory_store else {}
        summary = compute_summary(episode_results, mem_stats, mode)
        logger.info(f"  Epoch {epoch}: EM={summary['avg_em']:.3f} F1={summary['avg_f1']:.3f} "
                    f"({summary['total_success']}/{summary['total_envs']})")

        result_path = f"{results_dir}/{run_name}{epoch}.json"
        save_results(episode_results, summary, result_path)
        logger.info(f"Results saved: {result_path}")

        if memory_store:
            mem_path = f"{memory_dir}/hqa_epoch{epoch}.json"
            memory_store.save(mem_path)
            logger.info(f"Memory saved: {mem_path} ({memory_store.size()} entries)")

        env.close()

    logger.info("Done!")


if __name__ == "__main__":
    main()

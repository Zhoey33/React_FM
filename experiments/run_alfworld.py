"""Run React_FM (or baseline ReAct) on ALFWorld."""

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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import LLMClient
from src.memory import FailureMemoryStore
from src.failure_detector import ALFWorldFailureDetector
from src.agent import ReactFMAgent
from src.alfworld_env import ALFWorldEnv
from src.log_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run React_FM on ALFWorld")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--max-envs", type=int, default=None, help="Max environments per epoch (for quick tests)")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--baseline", action="store_true", help="Run as vanilla ReAct (no memory)")
    parser.add_argument("--resume-memory", type=str, default=None, help="Load memory from file before starting")
    parser.add_argument("--resume", action="store_true", help="Resume from last intermediate checkpoint")
    parser.add_argument("--run-name", type=str, default=None, help="Custom run name for results")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def save_results(results: list[dict], summary: dict, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    data = {"summary": summary, "episodes": results}
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def compute_summary(results: list[dict], memory_stats: dict, mode: str) -> dict:
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

    total_tokens = sum(r["total_tokens"] for r in results)
    total_failures = sum(r["failures_detected"] for r in results)

    return {
        "mode": mode,
        "total_envs": total,
        "total_success": successes,
        "success_rate": round(successes / total, 4) if total > 0 else 0,
        "by_task_type": by_type,
        "total_tokens": total_tokens,
        "avg_tokens_per_episode": round(total_tokens / total) if total > 0 else 0,
        "total_failures_detected": total_failures,
        "memory_stats": memory_stats,
    }


def log_summary(summary: dict):
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  {summary['mode']} Results")
    logger.info("=" * 60)
    logger.info(f"  Total: {summary['total_success']}/{summary['total_envs']} "
                f"({summary['success_rate']:.1%})")
    logger.info("-" * 60)
    logger.info(f"  {'Task Type':<12} {'Success':>8} {'Total':>8} {'Rate':>8}")
    logger.info("-" * 60)
    for tt, stats in sorted(summary["by_task_type"].items()):
        logger.info(f"  {tt:<12} {stats['success']:>8} {stats['total']:>8} {stats['rate']:>7.1%}")
    logger.info("-" * 60)
    logger.info(f"  Tokens: {summary['total_tokens']:,} total, "
                f"{summary['avg_tokens_per_episode']:,} avg/episode")
    if summary["memory_stats"]:
        ms = summary["memory_stats"]
        logger.info(f"  Memory: {ms.get('total_entries', 0)} entries, "
                    f"{ms.get('total_retrievals', 0)} retrievals, "
                    f"{ms.get('total_hits', 0)} hits")
    logger.info("=" * 60)


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup
    seed = args.seed or config["experiment"]["seed"]
    set_seed(seed)
    num_epochs = args.epochs or config["experiment"]["num_epochs"]
    results_dir = config["experiment"]["results_dir"]
    is_baseline = args.baseline
    mode = "react_baseline" if is_baseline else "react_fm"
    run_name = args.run_name or f"{mode}_epoch"

    # Logging: console (INFO) + file (DEBUG)
    setup_logging(
        log_level=config["experiment"]["log_level"],
        log_dir="logs",
        run_name=run_name.rstrip("_"),
    )

    # Initialize components
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
        )
        if args.resume_memory:
            memory_store.load(args.resume_memory)

    # Initialize judge LLM for implicit failure detection (React_FM only)
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

    detector = ALFWorldFailureDetector(judge_llm=judge_llm)

    # Initialize extractor LLM for post-episode memory extraction (React_FM only)
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

    agent = ReactFMAgent(
        llm=llm,
        memory_store=memory_store,
        failure_detector=detector,
        extractor_llm=extractor_llm,
        max_steps=config["agent"]["max_steps"],
        max_memory_inject=config["agent"]["max_memory_inject"],
        enable_memory=not is_baseline,
    )

    # Track per-env success across epochs (for skip-on-success)
    env_success = [False] * (args.max_envs or 134)

    # Run epochs
    for epoch in range(1, num_epochs + 1):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  Epoch {epoch}/{num_epochs} — {mode}")
        logger.info("=" * 60)

        env = ALFWorldEnv(split=config["alfworld"]["split"])
        env.setup()

        episode_results = []
        max_envs = args.max_envs or 134  # ALFWorld valid_unseen has 134 games
        skip_envs = 0

        # Resume from checkpoint if requested
        if args.resume:
            ckpt_path = f"{results_dir}/{run_name}{epoch}_intermediate.json"
            if Path(ckpt_path).exists():
                with open(ckpt_path) as f:
                    ckpt_data = json.load(f)
                episode_results = ckpt_data.get("episodes", [])
                skip_envs = len(episode_results)
                logger.info(f"Resuming epoch {epoch} from checkpoint: {skip_envs} envs already done")

                # Load corresponding memory checkpoint
                if memory_store:
                    mem_ckpt = f"{config['memory']['persist_dir']}/epoch{epoch}_intermediate.json"
                    if Path(mem_ckpt).exists():
                        memory_store.load(mem_ckpt)
                        logger.info(f"Loaded memory checkpoint: {memory_store.size()} entries")

        env_count = 0
        num_skipped = 0
        while True:
            try:
                # Skip already-completed envs when resuming
                if env_count < skip_envs:
                    env.skip()  # advance env without running
                    env_count += 1
                    continue

                # Skip envs that succeeded in previous epochs
                if epoch > 1 and env_success[env_count]:
                    env.skip()
                    # Record as skipped success
                    episode_results.append({
                        "env_idx": env_count + 1,
                        "task_type": "",
                        "task_description": "",
                        "success": True,
                        "total_steps": 0,
                        "total_tokens": 0,
                        "failures_detected": 0,
                        "memories_retrieved": 0,
                        "memories_stored": 0,
                        "wall_time_s": 0,
                        "skipped": True,
                        "steps": [],
                    })
                    num_skipped += 1
                    env_count += 1
                    logger.info(f"  [{env_count}] {'':8} SKIP (already succeeded)")
                    if env_count >= max_envs:
                        break
                    continue

                result = agent.run_episode(env, env_idx=env_count + 1)
                result_dict = result.to_dict()
                result_dict["skipped"] = False
                episode_results.append(result_dict)

                # Update success tracker
                if result.success:
                    env_success[env_count] = True

                env_count += 1

                # Progress
                successes = sum(1 for r in episode_results if r["success"])
                rate = successes / len(episode_results)
                logger.info(f"  [{env_count}] {result.task_type:<8} "
                            f"{'OK' if result.success else 'FAIL':>4} "
                            f"steps={result.total_steps:<3} "
                            f"running={rate:.1%} ({successes}/{env_count})")

                # Intermediate save every 10 envs
                if env_count % 10 == 0:
                    mem_stats = memory_store.stats() if memory_store else {}
                    summary = compute_summary(episode_results, mem_stats, mode)
                    save_results(
                        episode_results, summary,
                        f"{results_dir}/{run_name}{epoch}_intermediate.json"
                    )
                    if memory_store:
                        memory_store.save(f"{config['memory']['persist_dir']}/epoch{epoch}_intermediate.json")

                if env_count >= max_envs:
                    break

            except StopIteration:
                break
            except Exception as e:
                logger.error(f"Error on env #{env_count + 1}: {e}")
                env_count += 1
                if env_count >= max_envs:
                    break

        if num_skipped > 0:
            logger.info(f"  Skipped {num_skipped} already-succeeded envs")

        # Epoch summary
        mem_stats = memory_store.stats() if memory_store else {}
        summary = compute_summary(episode_results, mem_stats, mode)
        log_summary(summary)

        # Save epoch results
        result_path = f"{results_dir}/{run_name}{epoch}.json"
        save_results(episode_results, summary, result_path)
        logger.info(f"Results saved: {result_path}")

        # Save memory
        if memory_store:
            mem_path = f"{config['memory']['persist_dir']}/epoch{epoch}.json"
            memory_store.save(mem_path)
            logger.info(f"Memory saved: {mem_path} ({memory_store.size()} entries)")

    logger.info("Done!")


if __name__ == "__main__":
    main()

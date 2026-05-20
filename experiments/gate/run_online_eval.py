"""Online evaluation with the intervention gate.

Runs full episodes with the trained gate selecting arms at each failure
checkpoint. Compares: ReAct, Always-Repair, Binary-Gate, Learned-Gate, Oracle-Gate.

Usage:
    python experiments/gate/run_online_eval.py \
        --benchmark scienceworld \
        --system learned-gate \
        --gate-model gate_models/scienceworld_gate \
        --config config.yaml \
        --max-envs 50 \
        --output results/gate_eval/
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llm import LLMClient
from src.memory import FailureMemoryStore, RetrievalResult
from src.gate.train_gate import InterventionGate
from src.gate.train_gate_two_stage import TwoStageInterventionGate
from src.gate.train_gate_two_stage_regression import TwoStageRegressionGate
from src.gate.features import extract_features_from_agent_state
from src.log_utils import setup_logging

logger = logging.getLogger(__name__)

SYSTEMS = ["react", "always-repair", "binary-gate", "learned-gate", "oracle-gate"]
SCIENCEWORLD_RRF_K = 60
SCIENCEWORLD_MAX_RRF = 2.0 / (SCIENCEWORLD_RRF_K + 1)


def load_gate_model(filepath: str):
    path = Path(filepath)
    with open(path.with_suffix(".json")) as f:
        config = json.load(f)

    if "intervene_margin_threshold" in config and "repair_margin_threshold" in config:
        gate = TwoStageRegressionGate()
    elif "intervene_threshold" in config:
        gate = TwoStageInterventionGate()
    else:
        gate = InterventionGate()

    gate.load(filepath)
    return gate


def build_arm_injection(
    arm: str,
    retrieved: list | None,
) -> tuple[list | None, str | None]:
    """Map a selected arm to a one-step prompt injection payload.

    Returns:
        (retrieved_memories, hint_text)
    """
    if arm == "none" or not retrieved:
        return None, None

    mem = retrieved[0]
    if arm == "question":
        if mem.question_text:
            return None, mem.question_text
        return None, f"What went wrong when you tried '{mem.failure_action}'?"

    if arm == "repair":
        return retrieved, None

    return None, None


def _configured_retrieval_top_k(config: dict) -> int:
    return int(config.get("agent", {}).get("max_memory_inject", config["memory"]["retrieval_top_k"]))


def _memory_bucket_entry_count(memory_store: FailureMemoryStore, task_type: str, env_idx: int) -> int:
    return memory_store.entry_count(task_type=task_type, env_idx=env_idx)


def parse_args():
    parser = argparse.ArgumentParser(description="Online evaluation with gate")
    parser.add_argument("--benchmark", required=True, choices=["scienceworld", "alfworld", "webshop"])
    parser.add_argument("--system", required=True, choices=SYSTEMS)
    parser.add_argument("--gate-model", type=str, default=None, help="Path to trained gate model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--memory-file", type=str, default=None, help="Pre-loaded memory store")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--max-envs", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test",
                        help="Official split to evaluate on")
    parser.add_argument("--tasks", nargs="+", default=None, help="Specific task names to run")
    parser.add_argument("--max-variations", type=int, default=5, help="Max variations per task")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


class GateAgent:
    """Agent with intervention gate for online evaluation.

    Wraps the benchmark-specific agent logic with gate-controlled
    memory injection at failure checkpoints.
    """

    def __init__(
        self,
        system: str,
        gate: InterventionGate | None = None,
        oracle_data: dict | None = None,
    ):
        self.system = system
        self.gate = gate
        self.oracle_data = oracle_data or {}

        # Per-episode tracking
        self.gate_decisions = []
        self.bypass_count = 0
        self.abstention_count = 0

    def bypass_threshold(self) -> float:
        """Return the effective bypass threshold used for stats and logging."""
        if self.gate is None:
            return 0.0
        return float(self.gate.bypass_threshold)

    def select_arm(
        self,
        features: np.ndarray,
        retrieval_rrf_score: float,
        checkpoint_key: str = "",
    ) -> str:
        """Select intervention arm based on system type.

        Args:
            features: 11-dim feature vector
            retrieval_rrf_score: top-1 RRF score
            checkpoint_key: for oracle lookup

        Returns:
            arm: "none", "question", or "repair"
        """
        if self.system == "react":
            return "none"

        if self.system == "always-repair":
            return "repair"

        if self.system == "binary-gate":
            if self.gate is None:
                raise RuntimeError("binary-gate requires a trained gate")
            arm = self.gate.predict_arm(features, retrieval_rrf_score)
            # Binary: collapse question → repair
            if arm == "question":
                arm = "repair"
            return arm

        if self.system == "learned-gate":
            if self.gate is None:
                raise RuntimeError("learned-gate requires a trained gate")
            return self.gate.predict_arm(features, retrieval_rrf_score)

        if self.system == "oracle-gate":
            # Look up oracle arm from pre-computed data
            if checkpoint_key in self.oracle_data:
                return self.oracle_data[checkpoint_key]
            # Fallback: use gate if available, else repair
            if self.gate:
                return self.gate.predict_arm(features, retrieval_rrf_score)
            return "repair"

        raise ValueError(f"Unknown system: {self.system}")


def run_scienceworld_eval(args, config, gate_agent: GateAgent):
    """Run online evaluation on ScienceWorld with gate."""
    from src.scienceworld_env import ScienceWorldEnv, DEFAULT_EVAL_TASKS
    from src.scienceworld_failure_detector import ScienceWorldFailureDetector
    from src.scienceworld_memory_extractor import extract_failure_recoveries
    from prompts.scienceworld_prompts import build_user_prompt, SYSTEM_PROMPT_FM
    from src.gate.canonicalize import CUE_TEXT

    llm = LLMClient(
        model=config["llm"]["model"],
        base_url=config["llm"]["base_url"],
        api_key=config["llm"].get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
    )

    memory_store = FailureMemoryStore(
        embedding_model_name=config["memory"]["embedding_model"],
        max_entries=config["memory"]["max_entries"],
        top_k=_configured_retrieval_top_k(config),
        scope="task_type",
    )
    if args.memory_file:
        memory_store.load(args.memory_file)

    judge_llm = None
    if "judge" in config:
        jc = config["judge"]
        judge_llm = LLMClient(model=jc["model"], base_url=jc["base_url"],
            api_key=jc.get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
            temperature=0.0, max_tokens=16)

    detector = ScienceWorldFailureDetector(judge_llm=judge_llm)

    extractor_llm = None
    if "extractor" in config and args.system != "react":
        ec = config["extractor"]
        extractor_llm = LLMClient(model=ec["model"], base_url=ec["base_url"],
            api_key=ec.get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
            temperature=0.0, max_tokens=512)

    enable_memory = args.system != "react"
    max_steps = config["agent"]["max_steps"]
    retrieval_top_k = _configured_retrieval_top_k(config)

    all_results = []
    gate_stats = {"decisions": [], "bypass": 0, "abstain": 0, "total_failures": 0}
    task_names = args.tasks or DEFAULT_EVAL_TASKS

    for epoch in range(1, args.epochs + 1):
        env = ScienceWorldEnv(
            task_names=task_names,
            split=args.split,
            max_variations_per_task=args.max_variations,
        )
        env.setup()
        max_envs = args.max_envs or env.total_episodes

        for env_count in range(max_envs):
            try:
                init_obs, task_type, info = env.reset()
            except StopIteration:
                break

            env_idx = env_count + 1
            history = []
            action_history = []
            steps = []
            env_history_records = []
            current_score = 0.0
            failures_detected = 0
            success = False
            next_retrieved = None
            next_hint = None
            score_at_last_failure = 0.0  # for feature 8

            llm.tracker.reset()

            for step_num in range(max_steps):
                # Build prompt with at most one-step gate injection.
                prompt = build_user_prompt(
                    task_type=task_type, task_obs=init_obs, history=history,
                    retrieved_memories=next_retrieved, memory_style="original",
                    hint_text=next_hint,
                )
                next_retrieved = None
                next_hint = None
                response = llm.complete_text(prompt, stop=["\n"], label=f"step_{step_num}", system=SYSTEM_PROMPT_FM)
                action = response.strip().split("\n")[0].strip()
                if action.startswith("> "):
                    action = action[2:]

                from experiments.run_scienceworld import _extract_action
                action = _extract_action(action)
                if not action:
                    action = "look around"

                is_think = action.startswith("think:") or action.startswith("think ")
                if is_think:
                    history.append((action, "OK."))
                    steps.append({"step": step_num, "action": action, "observation": "OK.",
                                  "is_think": True, "failure_detected": False, "gate_arm": None})
                    continue

                observation, reward, done, step_info = env.step(action)
                action_history.append(action)
                current_score = step_info.get("score", current_score)
                env_history_records.append(
                    {"step": step_num, "action": action, "observation": observation}
                )
                is_done, is_success = detector.is_task_complete(observation, done, step_info)

                record = {"step": step_num, "action": action, "observation": observation[:200],
                          "is_think": False, "failure_detected": False, "gate_arm": None}

                if enable_memory and not is_done:
                    det = detector.detect(observation, action, action_history)
                    if det.is_failure:
                        record["failure_detected"] = True
                        record["failure_type"] = det.failure_type
                        failures_detected += 1
                        gate_stats["total_failures"] += 1

                        # Retrieve with scores
                        ret = memory_store.retrieve(
                            query_action=action, query_observation=observation,
                            task_type=task_type, top_k=retrieval_top_k, env_idx=env_idx,
                            return_scores=True,
                        )
                        if isinstance(ret, RetrievalResult):
                            rrf_score = ret.top1_score
                            rrf_margin = ret.margin
                            retrieved = ret.entries
                        else:
                            rrf_score = 0.0
                            rrf_margin = 0.0
                            retrieved = ret

                        # Extract features
                        features = extract_features_from_agent_state(
                            failure_type=det.failure_type,
                            failure_action=action,
                            task_type=task_type,
                            step_idx=step_num,
                            max_steps=max_steps,
                            action_history=action_history,
                            history=history,
                            retrieval_rrf_score=rrf_score,
                            retrieval_margin=rrf_margin,
                            memory_entry_count=_memory_bucket_entry_count(memory_store, task_type, env_idx),
                            score_at_checkpoint=current_score,
                            step_records=steps,
                            score_at_last_failure=score_at_last_failure,
                        )
                        score_at_last_failure = current_score  # update for next failure

                        # Gate decision
                        arm = gate_agent.select_arm(features, rrf_score)
                        record["gate_arm"] = arm

                        gate_stats["decisions"].append({
                            "env_idx": env_idx, "step": step_num,
                            "arm": arm, "rrf_score": rrf_score,
                        })

                        # Track bypass vs abstain correctly
                        if rrf_score < gate_agent.bypass_threshold():
                            gate_stats["bypass"] += 1
                        if arm == "none":
                            gate_stats["abstain"] += 1

                        next_retrieved, next_hint = build_arm_injection(arm, retrieved)
                        gate_stats["decisions"][-1].update({
                            "injected": bool(next_retrieved or next_hint),
                            "injection_mode": (
                                "memory_block" if next_retrieved else
                                ("hint_text" if next_hint else "none")
                            ),
                        })

                steps.append(record)
                history.append((action, observation))

                if is_done:
                    success = is_success
                    break

            # Post-episode: extract memories
            if enable_memory and extractor_llm is not None:
                detected_failures = [
                    {
                        "step": step["step"],
                        "action": step["action"],
                        "observation": step["observation"],
                        "failure_type": step.get("failure_type", ""),
                    }
                    for step in steps
                    if step.get("failure_detected")
                    and not step.get("is_think")
                    and step.get("failure_type") != "unproductive"
                ]
                recoveries = extract_failure_recoveries(
                    extractor_llm,
                    env_history_records,
                    detected_failures=detected_failures,
                )
                for rec in recoveries:
                    memory_store.add(
                        failure_action=rec["failure_action"],
                        failure_observation=rec["failure_observation"],
                        solution_action=rec["solution_action"],
                        task_type=task_type, env_idx=env_idx,
                    )

            agent_stats = llm.tracker.summary()
            all_results.append({
                "env_idx": env_idx,
                "task_type": task_type,
                "success": success,
                "score": current_score,
                "total_steps": len(steps),
                "total_tokens": agent_stats["total_tokens"],
                "failures_detected": failures_detected,
                "epoch": epoch,
            })

            successes = sum(1 for r in all_results if r["success"])
            logger.info(
                f"  [{env_idx}] {task_type:<30} "
                f"{'OK' if success else 'FAIL':>4} score={current_score:.2f} "
                f"running={successes}/{len(all_results)}"
            )

        env.close()

    return all_results, gate_stats


def run_alfworld_eval(args, config, gate_agent: GateAgent):
    """Run online evaluation on ALFWorld with gate."""
    from src.alfworld_env import ALFWorldEnv
    from src.failure_detector import ALFWorldFailureDetector
    from src.memory_extractor import extract_failure_recoveries
    from prompts.alfworld_prompts import build_user_prompt, SYSTEM_PROMPT_FM

    llm = LLMClient(
        model=config["llm"]["model"],
        base_url=config["llm"]["base_url"],
        api_key=config["llm"].get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
    )

    memory_store = FailureMemoryStore(
        embedding_model_name=config["memory"]["embedding_model"],
        max_entries=config["memory"]["max_entries"],
        top_k=_configured_retrieval_top_k(config),
    )
    if args.memory_file:
        memory_store.load(args.memory_file)

    judge_llm = None
    if "judge" in config:
        jc = config["judge"]
        judge_llm = LLMClient(model=jc["model"], base_url=jc["base_url"],
            api_key=jc.get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
            temperature=0.0, max_tokens=16)

    detector = ALFWorldFailureDetector(judge_llm=judge_llm)

    extractor_llm = None
    if "extractor" in config and args.system != "react":
        ec = config["extractor"]
        extractor_llm = LLMClient(model=ec["model"], base_url=ec["base_url"],
            api_key=ec.get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
            temperature=0.0, max_tokens=512)

    enable_memory = args.system != "react"
    max_steps = config["agent"]["max_steps"]
    retrieval_top_k = _configured_retrieval_top_k(config)

    all_results = []
    gate_stats = {"decisions": [], "bypass": 0, "abstain": 0, "total_failures": 0}

    for epoch in range(1, args.epochs + 1):
        env = ALFWorldEnv(split=config.get("alfworld", {}).get("split", "eval_out_of_distribution"))
        env.setup()
        max_envs = args.max_envs or 134

        for env_count in range(max_envs):
            try:
                init_obs, task_type, info = env.reset()
            except StopIteration:
                break

            env_idx = env_count + 1
            history = []
            action_history = []
            steps = []
            failures_detected = 0
            success = False
            next_retrieved = None
            next_hint = None
            score_at_last_failure = 0.0

            llm.tracker.reset()

            for step_num in range(max_steps):
                prompt = build_user_prompt(
                    task_type=task_type, task_obs=init_obs, history=history,
                    retrieved_memories=next_retrieved, memory_style="original",
                    hint_text=next_hint,
                )
                next_retrieved = None
                next_hint = None

                response = llm.complete_text(prompt, stop=["\n"], label=f"step_{step_num}", system=SYSTEM_PROMPT_FM)
                action = response.strip().split("\n")[0].strip()
                if action.startswith("> "):
                    action = action[2:]
                if not action:
                    action = "look"

                is_think = action.startswith("think:") or action.startswith("think ")
                if is_think:
                    history.append((action, "OK."))
                    steps.append({"step": step_num, "action": action, "observation": "OK.",
                                  "is_think": True, "failure_detected": False, "gate_arm": None})
                    continue

                observation, reward, done, step_info = env.step(action)
                action_history.append(action)
                env_history_records.append(
                    {"step": step_num, "action": action, "observation": observation}
                )
                is_done, is_success = detector.is_task_complete(observation, done, step_info)

                record = {"step": step_num, "action": action, "observation": observation[:200],
                          "is_think": False, "failure_detected": False, "gate_arm": None}

                if enable_memory and not is_done:
                    det = detector.detect(observation, action, action_history)
                    if det.is_failure:
                        record["failure_detected"] = True
                        record["failure_type"] = det.failure_type
                        failures_detected += 1
                        gate_stats["total_failures"] += 1

                        ret = memory_store.retrieve(
                            query_action=action, query_observation=observation,
                            task_type=task_type, top_k=retrieval_top_k, env_idx=env_idx,
                            return_scores=True,
                        )
                        if isinstance(ret, RetrievalResult):
                            rrf_score = ret.top1_score
                            retrieved = ret.entries
                        else:
                            rrf_score = 0.0
                            retrieved = ret

                        rrf_margin = ret.margin if isinstance(ret, RetrievalResult) else 0.0
                        features = extract_features_from_agent_state(
                            failure_type=det.failure_type,
                            failure_action=action,
                            task_type=task_type,
                            step_idx=step_num,
                            max_steps=max_steps,
                            action_history=action_history,
                            history=history,
                            retrieval_rrf_score=rrf_score,
                            retrieval_margin=rrf_margin,
                            memory_entry_count=_memory_bucket_entry_count(memory_store, task_type, env_idx),
                            score_at_checkpoint=0.0,
                            step_records=steps,
                            score_at_last_failure=score_at_last_failure,
                        )

                        arm = gate_agent.select_arm(features, rrf_score)
                        record["gate_arm"] = arm

                        gate_stats["decisions"].append({
                            "env_idx": env_idx, "step": step_num,
                            "arm": arm, "rrf_score": rrf_score,
                        })

                        if rrf_score < gate_agent.bypass_threshold():
                            gate_stats["bypass"] += 1
                        if arm == "none":
                            gate_stats["abstain"] += 1

                        next_retrieved, next_hint = build_arm_injection(arm, retrieved)
                        gate_stats["decisions"][-1].update({
                            "injected": bool(next_retrieved or next_hint),
                            "injection_mode": (
                                "memory_block" if next_retrieved else
                                ("hint_text" if next_hint else "none")
                            ),
                        })

                steps.append(record)
                history.append((action, observation))

                if is_done:
                    success = is_success
                    break

            # Post-episode: extract memories
            if enable_memory and extractor_llm is not None:
                detected_failures = [
                    {
                        "step": step["step"],
                        "action": step["action"],
                        "observation": step["observation"],
                        "failure_type": step.get("failure_type", ""),
                    }
                    for step in steps
                    if step.get("failure_detected")
                    and not step.get("is_think")
                    and step.get("failure_type") != "unproductive"
                ]
                recoveries = extract_failure_recoveries(
                    extractor_llm,
                    env_history_records,
                    detected_failures=detected_failures,
                )
                for rec in recoveries:
                    memory_store.add(
                        failure_action=rec["failure_action"],
                        failure_observation=rec["failure_observation"],
                        solution_action=rec["solution_action"],
                        task_type=task_type, env_idx=env_idx,
                    )

            agent_stats = llm.tracker.summary()
            all_results.append({
                "env_idx": env_idx,
                "task_type": task_type,
                "success": success,
                "score": 1.0 if success else 0.0,
                "total_steps": len(steps),
                "total_tokens": agent_stats["total_tokens"],
                "failures_detected": failures_detected,
                "epoch": epoch,
            })

            successes = sum(1 for r in all_results if r["success"])
            logger.info(
                f"  [{env_idx}] {task_type:<12} "
                f"{'OK' if success else 'FAIL':>4} "
                f"running={successes}/{len(all_results)}"
            )

    return all_results, gate_stats


def run_webshop_eval(args, config, gate_agent: GateAgent):
    """Run online evaluation on WebShop with gate."""
    from src.webshop_env import WebShopEnv
    from src.webshop_failure_detector import WebShopFailureDetector
    from src.webshop_memory_extractor import extract_failure_recoveries
    from prompts.webshop_prompts import build_user_prompt, SYSTEM_PROMPT_FM

    llm = LLMClient(
        model=config["llm"]["model"],
        base_url=config["llm"]["base_url"],
        api_key=config["llm"].get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
    )

    memory_store = FailureMemoryStore(
        embedding_model_name=config["memory"]["embedding_model"],
        max_entries=config["memory"]["max_entries"],
        top_k=_configured_retrieval_top_k(config),
    )
    if args.memory_file:
        memory_store.load(args.memory_file)

    judge_llm = None
    if "judge" in config:
        jc = config["judge"]
        judge_llm = LLMClient(model=jc["model"], base_url=jc["base_url"],
            api_key=jc.get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
            temperature=0.0, max_tokens=16)

    detector = WebShopFailureDetector(judge_llm=judge_llm)

    extractor_llm = None
    if "extractor" in config and args.system != "react":
        ec = config["extractor"]
        extractor_llm = LLMClient(model=ec["model"], base_url=ec["base_url"],
            api_key=ec.get("api_key") or os.environ.get("SILICONFLOW_API_KEY", ""),
            temperature=0.0, max_tokens=512)

    enable_memory = args.system != "react"
    max_steps = config["agent"]["max_steps"]
    retrieval_top_k = _configured_retrieval_top_k(config)

    all_results = []
    gate_stats = {"decisions": [], "bypass": 0, "abstain": 0, "total_failures": 0}

    for epoch in range(1, args.epochs + 1):
        env = WebShopEnv()
        env.setup()
        max_envs = args.max_envs or 100

        for env_count in range(max_envs):
            try:
                init_obs, task_type, info = env.reset()
            except Exception:
                break

            env_idx = env_count + 1
            history = []
            action_history = []
            steps = []
            failures_detected = 0
            success = False
            next_retrieved = None
            next_hint = None
            current_reward = 0.0
            score_at_last_failure = 0.0

            llm.tracker.reset()

            for step_num in range(max_steps):
                prompt = build_user_prompt(
                    task_type=task_type, task_obs=init_obs, history=history,
                    retrieved_memories=next_retrieved, memory_style="original",
                    hint_text=next_hint,
                )
                next_retrieved = None
                next_hint = None

                response = llm.complete_text(prompt, stop=["\n"], label=f"step_{step_num}", system=SYSTEM_PROMPT_FM)
                action = response.strip().split("\n")[0].strip()
                if action.startswith("> "):
                    action = action[2:]
                if not action:
                    action = "search[product]"

                is_think = action.startswith("think:") or action.startswith("think ")
                if is_think:
                    history.append((action, "OK."))
                    steps.append({"step": step_num, "action": action, "observation": "OK.",
                                  "is_think": True, "failure_detected": False, "gate_arm": None})
                    continue

                observation, reward, done, step_info = env.step(action)
                action_history.append(action)
                if reward > 0:
                    current_reward = reward

                is_done, is_success = detector.is_task_complete(observation, done, step_info)

                record = {"step": step_num, "action": action, "observation": observation[:200],
                          "is_think": False, "failure_detected": False, "gate_arm": None}

                if enable_memory and not is_done:
                    det = detector.detect(observation, action, action_history)
                    if det.is_failure:
                        record["failure_detected"] = True
                        failures_detected += 1
                        gate_stats["total_failures"] += 1

                        ret = memory_store.retrieve(
                            query_action=action, query_observation=observation,
                            task_type=task_type, top_k=retrieval_top_k, env_idx=env_idx,
                            return_scores=True,
                        )
                        if isinstance(ret, RetrievalResult):
                            rrf_score = ret.top1_score
                            retrieved = ret.entries
                        else:
                            rrf_score = 0.0
                            retrieved = ret

                        rrf_margin = ret.margin if isinstance(ret, RetrievalResult) else 0.0
                        features = extract_features_from_agent_state(
                            failure_type=det.failure_type,
                            failure_action=action,
                            task_type=task_type,
                            step_idx=step_num,
                            max_steps=max_steps,
                            action_history=action_history,
                            history=history,
                            retrieval_rrf_score=rrf_score,
                            retrieval_margin=rrf_margin,
                            memory_entry_count=_memory_bucket_entry_count(memory_store, task_type, env_idx),
                            score_at_checkpoint=current_reward,
                            step_records=steps,
                            score_at_last_failure=score_at_last_failure,
                        )
                        score_at_last_failure = current_reward

                        arm = gate_agent.select_arm(features, rrf_score)
                        record["gate_arm"] = arm

                        gate_stats["decisions"].append({
                            "env_idx": env_idx, "step": step_num,
                            "arm": arm, "rrf_score": rrf_score,
                        })

                        if rrf_score < gate_agent.bypass_threshold():
                            gate_stats["bypass"] += 1
                        if arm == "none":
                            gate_stats["abstain"] += 1

                        next_retrieved, next_hint = build_arm_injection(arm, retrieved)
                        gate_stats["decisions"][-1].update({
                            "injected": bool(next_retrieved or next_hint),
                            "injection_mode": (
                                "memory_block" if next_retrieved else
                                ("hint_text" if next_hint else "none")
                            ),
                        })

                steps.append(record)
                history.append((action, observation))

                if is_done:
                    success = is_success
                    break

            # Post-episode: extract memories
            if enable_memory and extractor_llm is not None:
                env_history = [(a, o) for a, o in history if not a.startswith("think")]
                recoveries = extract_failure_recoveries(extractor_llm, env_history)
                for rec in recoveries:
                    memory_store.add(
                        failure_action=rec["failure_action"],
                        failure_observation=rec["failure_observation"],
                        solution_action=rec["solution_action"],
                        task_type=task_type, env_idx=env_idx,
                    )

            agent_stats = llm.tracker.summary()
            all_results.append({
                "env_idx": env_idx,
                "task_type": task_type,
                "success": success,
                "score": current_reward,
                "total_steps": len(steps),
                "total_tokens": agent_stats["total_tokens"],
                "failures_detected": failures_detected,
                "epoch": epoch,
            })

            successes = sum(1 for r in all_results if r["success"])
            logger.info(
                f"  [{env_idx}] reward={current_reward:.2f} "
                f"{'OK' if success else 'FAIL':>4} "
                f"running={successes}/{len(all_results)}"
            )

        env.close()

    return all_results, gate_stats


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_name = args.run_name or f"{args.benchmark}_{args.system}_"
    setup_logging(log_level="INFO", log_dir="logs", run_name=run_name.rstrip("_"))

    # Load gate model if needed
    gate = None
    if args.gate_model and args.system in ("binary-gate", "learned-gate", "oracle-gate"):
        gate = load_gate_model(args.gate_model)
        logger.info(f"Gate loaded from {args.gate_model}")
        if args.benchmark == "scienceworld" and gate.bypass_threshold > SCIENCEWORLD_MAX_RRF:
            raise ValueError(
                "Refusing to run ScienceWorld online eval with a legacy gate model whose "
                f"bypass_threshold={gate.bypass_threshold:.4f} exceeds max retrieval_rrf_score="
                f"{SCIENCEWORLD_MAX_RRF:.4f}. Re-train or use a gate model with bypass_threshold=0.0."
            )

    gate_agent = GateAgent(system=args.system, gate=gate)

    logger.info(f"Online evaluation: {args.benchmark} / {args.system}")

    if args.benchmark == "scienceworld":
        results, gate_stats = run_scienceworld_eval(args, config, gate_agent)
    elif args.benchmark == "alfworld":
        results, gate_stats = run_alfworld_eval(args, config, gate_agent)
    elif args.benchmark == "webshop":
        results, gate_stats = run_webshop_eval(args, config, gate_agent)
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary
    total = len(results)
    successes = sum(1 for r in results if r["success"])
    avg_score = np.mean([r["score"] for r in results])
    total_tokens = sum(r["total_tokens"] for r in results)

    summary = {
        "system": args.system,
        "benchmark": args.benchmark,
        "total": total,
        "successes": successes,
        "success_rate": round(successes / total, 4) if total > 0 else 0,
        "avg_score": round(float(avg_score), 4),
        "total_tokens": total_tokens,
        "gate_stats": {
            "total_failures": gate_stats["total_failures"],
            "bypass_count": gate_stats["bypass"],
            "abstain_count": gate_stats["abstain"],
            "bypass_rate": round(gate_stats["bypass"] / max(gate_stats["total_failures"], 1), 4),
            "abstention_rate": round(gate_stats["abstain"] / max(gate_stats["total_failures"], 1), 4),
            "decision_count": len(gate_stats["decisions"]),
            "arm_counts": {
                arm: sum(1 for d in gate_stats["decisions"] if d["arm"] == arm)
                for arm in ("none", "question", "repair")
            },
        },
    }

    result_path = output_dir / f"{args.benchmark}_{args.system}.json"
    with open(result_path, "w") as f:
        json.dump(
            {
                "summary": summary,
                "episodes": results,
                "gate_decisions": gate_stats["decisions"],
            },
            f,
            indent=2,
        )

    logger.info(f"\n{'='*60}")
    logger.info(f"  {args.system} on {args.benchmark}")
    logger.info(f"  Success: {successes}/{total} ({summary['success_rate']:.1%})")
    logger.info(f"  Avg Score: {avg_score:.2f}")
    logger.info(f"  Tokens: {total_tokens:,}")
    if gate_stats["total_failures"] > 0:
        logger.info(f"  Bypass: {gate_stats['bypass']}/{gate_stats['total_failures']}")
        logger.info(f"  Abstain: {gate_stats['abstain']}/{gate_stats['total_failures']}")
    logger.info(f"  Saved: {result_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

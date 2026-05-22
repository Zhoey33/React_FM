"""Run React_FM (or baseline ReAct) on ScienceWorld."""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure Java is on PATH
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
from src.memory import FailureMemoryStore, RetrievalResult
from src.scienceworld_retrieval_gate import select_retrieval_memory
from src.scienceworld_failure_detector import RepairAdvice, ScienceWorldFailureDetector
from src.scienceworld_env import ScienceWorldEnv, DEFAULT_EVAL_TASKS
from src.scienceworld_failure_events import write_failure_events_jsonl
from src.scienceworld_reporting import build_protocol_metadata
from src.scienceworld_reporting import compute_summary as compute_scienceworld_summary
from src.log_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run React_FM on ScienceWorld")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--max-envs", type=int, default=None, help="Max environments per epoch")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--baseline", action="store_true",
                        help="Run as pure ReAct (no memory, no failure detection)")
    parser.add_argument("--resume-memory", type=str, default=None, help="Load memory from file")
    parser.add_argument("--resume-results", type=str, default=None, help="Load previous results for skip-on-success")
    parser.add_argument("--run-name", type=str, default=None, help="Custom run name")
    parser.add_argument("--inject-mode", choices=["in_loop", "episode", "none"], default="in_loop")
    parser.add_argument("--memory-style", choices=["original", "factual", "reflexion", "hint"], default="original")
    parser.add_argument("--memory-format", choices=["failure_recovery"],
                        default="failure_recovery", help="ScienceWorld memory storage format")
    parser.add_argument("--retrieval-mode", choices=["hybrid", "bm25_only", "embedding_only", "random"],
                        default="hybrid", help="Retrieval method for ablation")
    parser.add_argument("--enable-implicit-failures", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Enable or disable LLM judge implicit failure detection after rule checks")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test",
                        help="Official ScienceWorld split to run")
    parser.add_argument("--tasks", nargs="+", default=None, help="Specific task names to run")
    parser.add_argument("--max-variations", type=int, default=5, help="Max variations per task")
    parser.add_argument("--step-limit", type=int, default=100, help="Max steps per episode")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def save_results(results: list[dict], summary: dict, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    data = {"summary": summary, "episodes": results}
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def compute_summary(
    results: list[dict],
    memory_stats: dict,
    mode: str,
    protocol: dict | None = None,
) -> dict:
    return compute_scienceworld_summary(
        results,
        mode=mode,
        memory_stats=memory_stats,
        protocol=protocol,
    )


def _resolve_enable_implicit_failures(args, config: dict) -> bool:
    """Resolve ScienceWorld implicit judge detection from CLI first, then config."""
    cli_value = getattr(args, "enable_implicit_failures", None)
    if cli_value is not None:
        return bool(cli_value)
    return bool(config.get("judge", {}).get("enable_implicit_failures", True))


def log_summary(summary: dict):
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  {summary['mode']} Results (ScienceWorld)")
    logger.info("=" * 60)
    logger.info(f"  Total: {summary['total_success']}/{summary['total_envs']} "
                f"({summary['success_rate']:.1%})")
    logger.info(
        f"  Avg Score: {summary['avg_score']:.2f}"
    )
    logger.info("-" * 60)
    logger.info(f"  {'Task Name':<40} {'Success':>8} {'Total':>8} {'Rate':>8}")
    logger.info("-" * 60)
    for tt, stats in sorted(summary["by_task_type"].items()):
        logger.info(f"  {tt:<40} {stats['success']:>8} {stats['total']:>8} {stats['rate']:>7.1%}")
    logger.info("-" * 60)
    logger.info(f"  Tokens: {summary['total_tokens']:,} total, "
                f"{summary['avg_tokens_per_episode']:,} avg/episode "
                f"(agent={summary.get('agent_tokens', 0):,}, "
                f"judge={summary.get('judge_tokens', 0):,}, "
                f"ext={summary.get('extractor_tokens', 0):,})")
    if summary["memory_stats"]:
        ms = summary["memory_stats"]
        logger.info(f"  Memory: {ms.get('total_entries', 0)} entries, "
                    f"{ms.get('candidate_retrievals', ms.get('total_retrievals', 0))} retrievals, "
                    f"{ms.get('candidate_hits', ms.get('total_hits', 0))} candidate hits, "
                    f"{ms.get('injection_hits', 0)} injection hits")
    logger.info("=" * 60)


import re

# Valid ScienceWorld action prefixes
_SW_ACTION_PREFIXES = (
    "go to", "pick up", "put down", "open", "close", "activate", "deactivate",
    "use", "pour", "mix", "focus on", "wait", "look around", "look", "inventory",
    "examine", "read", "connect", "move", "teleport to", "dunk", "eat", "drink",
    "flush", "reset task", "think:",  "think ",
)


def _extract_action(raw: str) -> str:
    """Extract a valid ScienceWorld action from possibly verbose LLM output."""
    raw = raw.strip()
    if not raw:
        return ""

    # If it already starts with a valid prefix, use as-is
    lower = raw.lower()
    for prefix in _SW_ACTION_PREFIXES:
        if lower.startswith(prefix):
            return raw

    # Try to find a valid action within the text
    for prefix in _SW_ACTION_PREFIXES:
        idx = lower.find(prefix)
        if idx >= 0:
            # Extract from this prefix to end of line or period
            rest = raw[idx:]
            # Take up to first period or end
            end = rest.find(".")
            if end > 0 and end < 80:
                return rest[:end].strip()
            return rest[:80].strip()

    # Fallback: return first 80 chars
    return raw[:80]


def _is_task_complete(done: bool, info: dict) -> tuple[bool, bool]:
    """Check episode termination using ScienceWorld's score convention."""
    score = info.get("score", 0.0)
    return done, score >= 100.0


def _score_delta(before, after) -> float | None:
    """Compute a compact score delta for memory provenance."""
    if before is None or after is None:
        return None
    try:
        return round(float(after) - float(before), 4)
    except (TypeError, ValueError):
        return None


def _compact_log_text(text, max_chars: int = 240) -> str:
    """Compact long ScienceWorld text for readable INFO logs."""
    compact = " ".join(str(text or "").replace("\t", " ").split())
    if len(compact) > max_chars:
        return compact[: max_chars - 3] + "..."
    return compact


def _infer_task_name_from_env(env) -> str:
    """Best-effort task name recovery for exception accounting."""
    schedule = getattr(env, "_schedule", None)
    schedule_idx = getattr(env, "_schedule_idx", 0)
    if not schedule:
        return "unknown"
    if schedule_idx > 0 and schedule_idx - 1 < len(schedule):
        return schedule[schedule_idx - 1][0]
    if schedule_idx < len(schedule):
        return schedule[schedule_idx][0]
    return "unknown"


def _infer_variation_idx_from_env(env) -> int | None:
    """Best-effort variation recovery for exception and skip accounting."""
    schedule = getattr(env, "_schedule", None)
    schedule_idx = getattr(env, "_schedule_idx", 0)
    if not schedule:
        return None
    if schedule_idx > 0 and schedule_idx - 1 < len(schedule):
        return schedule[schedule_idx - 1][1]
    if schedule_idx < len(schedule):
        return schedule[schedule_idx][1]
    return None


def _format_retrieved_memory_text(entries: list) -> str:
    """Compact retrieved memories for failure-event logs."""
    lines = []
    for entry in entries:
        repair = entry.get_repair_display() if hasattr(entry, "get_repair_display") else ""
        lines.append(f"{entry.memory_id}: {repair}")
    return "\n".join(lines)


def _format_memory_for_log(entry) -> str:
    """Compact one retrieved memory for human audit logs."""
    return (
        f"failure_type={getattr(entry, 'failure_type', '')}; "
        f"failure_action={_compact_log_text(getattr(entry, 'failure_action', ''), 120)}; "
        f"failure_observation={_compact_log_text(getattr(entry, 'failure_observation', ''), 160)}; "
        f"repair_strategy={_compact_log_text(getattr(entry, 'repair_strategy', ''), 160)}; "
        f"repair_tactic={_compact_log_text(getattr(entry, 'repair_tactic', ''), 160)}; "
        f"repair_action={_compact_log_text(getattr(entry, 'repair_action', '') or getattr(entry, 'solution_action', ''), 160)}; "
        f"confidence_score={getattr(entry, 'confidence_score', None)}"
    )


def _unpack_retrieval_result(result) -> tuple[list, list[float], int]:
    """Normalize scored and legacy memory retrieval return values."""
    if isinstance(result, RetrievalResult):
        return result.entries, result.rrf_scores, result.candidate_count
    return result, [], len(result or [])


def _memory_scope_for_protocol(memory_store) -> str | None:
    """Return the active memory scope saved in ScienceWorld protocol metadata."""
    if memory_store is None:
        return None
    return getattr(memory_store, "scope", "task_type")


class ScienceWorldReActAgent:
    """ReAct agent adapted for ScienceWorld."""

    def __init__(
        self,
        llm: LLMClient,
        memory_store: FailureMemoryStore | None = None,
        failure_detector: ScienceWorldFailureDetector | None = None,
        extractor_llm: LLMClient | None = None,
        max_steps: int = 50,
        max_memory_inject: int = 1,
        enable_memory: bool = True,
        inject_mode: str = "in_loop",
        memory_style: str = "original",
        memory_format: str = "failure_recovery",
        baseline_mode: bool = False,
        prompt_history_window: int = 10,
        retrieval_candidate_k: int = 5,
        enable_failure_type_filter: bool = True,
        enable_safety_gate: bool = True,
        relevance_score_threshold: float = 0.45,
        enable_relevance_judge: bool = False,
        enable_rule_repair_advice: bool = True,
        enable_intent_gate: bool = True,
    ):
        self.llm = llm
        self.memory = memory_store
        self.baseline_mode = baseline_mode
        self.detector = None if baseline_mode else (failure_detector or ScienceWorldFailureDetector())
        self.extractor_llm = extractor_llm
        self.max_steps = max_steps
        self.max_memory_inject = max_memory_inject
        self.enable_memory = enable_memory and (memory_store is not None)
        self.inject_mode = inject_mode
        self.memory_style = memory_style
        self.memory_format = memory_format
        self.prompt_history_window = prompt_history_window
        self.retrieval_candidate_k = retrieval_candidate_k
        self.enable_failure_type_filter = enable_failure_type_filter
        self.enable_safety_gate = enable_safety_gate
        self.relevance_score_threshold = relevance_score_threshold
        self.enable_relevance_judge = enable_relevance_judge
        self.enable_rule_repair_advice = enable_rule_repair_advice
        self.enable_intent_gate = enable_intent_gate

    def _prompt_history(self, history: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Return the recent history slice used in ScienceWorld prompts."""
        if self.prompt_history_window <= 0:
            return history
        return history[-self.prompt_history_window:]

    def run_episode(self, env, env_idx: int = 0) -> dict:
        """Run one ScienceWorld episode. Returns result dict."""
        from prompts.scienceworld_prompts import (
            build_baseline_user_prompt,
            build_user_prompt,
            SYSTEM_PROMPT_BASE,
            SYSTEM_PROMPT_FM,
        )
        from src.scienceworld_memory_extractor import extract_failure_recoveries

        t0 = time.time()
        self.llm.tracker.reset()
        if self.detector is not None and self.detector.judge_llm is not None:
            self.detector.judge_llm.tracker.reset()
        if self.detector is not None and hasattr(self.detector, "reset_judge_cache"):
            self.detector.reset_judge_cache()
        if self.extractor_llm is not None:
            self.extractor_llm.tracker.reset()

        init_obs, task_type, info = env.reset()
        variation_idx = info.get("variation_idx")
        task_goal = info.get("task_desc") or init_obs

        history: list[tuple[str, str]] = []
        action_history: list[str] = []
        steps: list[dict] = []
        env_history_records: list[dict] = []
        detector_history: list[dict] = []
        success = False
        final_score = 0.0
        current_look = info.get("look", "")
        current_inventory = info.get("inventory", "")
        current_valid_actions = info.get("valid_actions", "")

        logger.info(
            "Env #%s task=%s variation=%s",
            env_idx,
            task_type,
            variation_idx,
        )
        logger.info("  Task goal: %s", _compact_log_text(task_goal, 500))

        current_retrieved = None
        current_failure_context = None
        current_judge_advice = None
        pending_memory_record = None
        pending_judge_advice_record = None
        failures_detected = 0
        memories_retrieved_total = 0
        consecutive_thinks = 0
        max_consecutive_thinks = 3

        episode_memories = None
        if self.enable_memory and self.inject_mode == "episode":
            all_mem = self.memory.get_all(env_idx=env_idx, task_type=task_type)
            if all_mem:
                episode_memories = all_mem[:self.max_memory_inject]
                memories_retrieved_total = len(episode_memories)

        for step_num in range(self.max_steps):
            if self.baseline_mode:
                prompt = build_baseline_user_prompt(
                    task_type=task_type,
                    task_obs=init_obs,
                    history=self._prompt_history(history),
                )
                system_prompt = SYSTEM_PROMPT_BASE
            elif self.inject_mode == "episode":
                prompt = build_user_prompt(
                    task_type=task_type, task_obs=init_obs, history=self._prompt_history(history),
                    retrieved_memories=episode_memories, memory_style=self.memory_style,
                )
                system_prompt = SYSTEM_PROMPT_FM
            else:
                inject_judge_advice_now = bool(current_failure_context and current_judge_advice)
                inject_memory_now = bool(current_failure_context and current_retrieved)
                prompt = build_user_prompt(
                    task_type=task_type, task_obs=init_obs, history=self._prompt_history(history),
                    retrieved_memories=current_retrieved, memory_style=self.memory_style,
                    failure_context=current_failure_context,
                    judge_advice=current_judge_advice,
                )
                system_prompt = SYSTEM_PROMPT_FM
                if inject_memory_now and pending_memory_record is not None:
                    pending_memory_record["memory_retrieved"] = len(current_retrieved)
                    pending_memory_record["memory_injected"] = True
                    pending_memory_record["injected_memory_text"] = _format_retrieved_memory_text(
                        current_retrieved
                    )
                    memories_retrieved_total += len(current_retrieved)
                    pending_memory_record = None
                if inject_judge_advice_now and pending_judge_advice_record is not None:
                    pending_judge_advice_record["judge_advice_injected"] = True
                    logger.info(
                        "    Injecting judge advice fallback: strategy=%s action=%s "
                        "confidence=%s rationale=%s",
                        _compact_log_text(current_judge_advice.get("repair_strategy", ""), 160),
                        _compact_log_text(current_judge_advice.get("repair_action", ""), 120),
                        current_judge_advice.get("repair_confidence"),
                        _compact_log_text(current_judge_advice.get("repair_rationale", ""), 180),
                    )
                    pending_judge_advice_record = None

            response = self.llm.complete_text(
                prompt, stop=["\n"], label=f"step_{step_num}",
                system=system_prompt,
            )
            action = response.strip().split("\n")[0].strip()

            if self.inject_mode == "in_loop":
                current_retrieved = None
                current_failure_context = None
                current_judge_advice = None

            if action.startswith("> "):
                action = action[2:]

            # Post-process: extract valid action from verbose LLM output
            action = _extract_action(action)
            if not action:
                action = "look around"

            logger.info(f"  Step {step_num}: {action[:80]}")

            is_think = action.startswith("think:") or action.startswith("think ")
            if is_think:
                consecutive_thinks += 1
                if consecutive_thinks > max_consecutive_thinks:
                    # Force an action to break think loops
                    action = "look around"
                    is_think = False
                    consecutive_thinks = 0
                else:
                    observation = "OK."
                    steps.append({
                        "step": step_num, "action": action, "observation": observation,
                        "is_think": True, "failure_detected": False, "failure_type": "",
                        "memory_retrieved": 0,
                    })
                    history.append((action, observation))
                    continue
            else:
                consecutive_thinks = 0

            score_before_action = final_score
            look_before_action = current_look
            inventory_before_action = current_inventory
            valid_actions_before_action = current_valid_actions
            observation, reward, done, step_info = env.step(action)
            logger.info(f"    obs: {observation[:80]}")
            action_history.append(action)
            env_history_records.append(
                {"step": step_num, "action": action, "observation": observation}
            )

            final_score = step_info.get("score", final_score)
            look_after_action = step_info.get("look", current_look)
            inventory_after_action = step_info.get("inventory", current_inventory)
            valid_actions_after_action = step_info.get("valid_actions", current_valid_actions)
            is_done, is_success = _is_task_complete(done, step_info)

            record = {
                "step": step_num, "action": action, "observation": observation[:200],
                "is_think": False, "failure_detected": False, "failure_type": "",
                "memory_retrieved": 0,
                "memory_injected": False,
                "retrieval_attempted": False,
                "retrieval_mode": getattr(self.memory, "retrieval_mode", "") if self.memory else "",
                "retrieval_top_k": self.max_memory_inject,
                "retrieval_min_score": getattr(self.memory, "min_score", 0.0) if self.memory else 0.0,
                "retrieval_candidate_count": 0,
                "retrieval_candidate_memory_ids": [],
                "retrieval_candidate_scores": [],
                "retrieval_candidate_relevance_scores": [],
                "retrieval_selected_memory_id": None,
                "retrieval_relevance_decision": False,
                "retrieval_rejection_reason": "",
                "retrieval_filtered_by_type_count": 0,
                "retrieval_filtered_by_safety_count": 0,
                "retrieval_filtered_by_intent_count": 0,
                "retrieval_current_intent": "",
                "retrieval_candidate_intents": [],
                "retrieved_memory_ids": [],
                "retrieved_memory_scores": [],
                "injected_memory_text": "",
                "score_before_action": score_before_action,
                "score_after_action": final_score,
                "detector_source": "",
                "failure_reason": "",
                "failure_confidence": None,
                "productive_signal": "",
                "evidence_for_failure": [],
                "evidence_against_failure": [],
                "judge_repair_strategy": "",
                "judge_repair_action": "",
                "judge_repair_confidence": None,
                "judge_repair_rationale": "",
                "judge_advice_source": "",
                "judge_advice_injected": False,
                "judge_cache_hit": False,
                "judge_call_type": "",
            }

            if (
                self.detector is not None
                and self.enable_memory
                and self.inject_mode in ("in_loop", "none")
                and not is_done
            ):
                det = self.detector.detect(
                    observation,
                    action,
                    action_history,
                    task_type=task_type,
                    task_goal=task_goal,
                    recent_history=detector_history[-10:],
                    score_before_action=score_before_action,
                    score_after_action=final_score,
                    step=step_num,
                    variation_idx=variation_idx,
                    look_before_action=look_before_action,
                    inventory_before_action=inventory_before_action,
                    valid_actions_before_action=valid_actions_before_action,
                    look_after_action=look_after_action,
                    inventory_after_action=inventory_after_action,
                    valid_actions_after_action=valid_actions_after_action,
                )
                record["judge_cache_hit"] = bool(getattr(det, "judge_cache_hit", False))
                record["judge_call_type"] = getattr(det, "judge_call_type", "") or ""
                if det.is_failure:
                    record["failure_detected"] = True
                    record["failure_type"] = det.failure_type
                    record["detector_source"] = det.detector_source
                    record["failure_reason"] = det.reason
                    record["failure_confidence"] = det.confidence
                    record["productive_signal"] = det.productive_signal
                    record["evidence_for_failure"] = det.evidence_for_failure or []
                    record["evidence_against_failure"] = det.evidence_against_failure or []
                    record["judge_repair_strategy"] = det.judge_repair_strategy
                    record["judge_repair_action"] = det.judge_repair_action
                    record["judge_repair_confidence"] = det.judge_repair_confidence
                    record["judge_repair_rationale"] = det.judge_repair_rationale
                    record["judge_advice_source"] = det.judge_advice_source
                    failures_detected += 1
                    logger.info(
                        "    FAILURE detected: %s detector=%s confidence=%s "
                        "productive_signal=%s score=%s->%s reason=%s",
                        det.failure_type,
                        det.detector_source,
                        det.confidence,
                        det.productive_signal,
                        score_before_action,
                        final_score,
                        _compact_log_text(det.reason, 180),
                    )

                    if self.inject_mode == "in_loop":
                        record["retrieval_attempted"] = True
                        retrieval_result = self.memory.retrieve(
                            query_action=action, query_observation=observation,
                            query_failure_type=det.failure_type,
                            task_type=task_type,
                            top_k=self.max_memory_inject,
                            candidate_k=self.retrieval_candidate_k,
                            env_idx=env_idx,
                            return_scores=True,
                        )
                        candidates, candidate_scores, candidate_count = _unpack_retrieval_result(
                            retrieval_result
                        )
                        decision = select_retrieval_memory(
                            entries=candidates,
                            retrieval_scores=candidate_scores,
                            current_failure_type=det.failure_type,
                            failed_action=action,
                            failure_observation=observation,
                            recent_actions=action_history,
                            relevance_score_threshold=self.relevance_score_threshold,
                            enable_failure_type_filter=self.enable_failure_type_filter,
                            enable_safety_gate=self.enable_safety_gate,
                            enable_intent_gate=self.enable_intent_gate,
                        )
                        retrieved = [decision.selected_entry] if decision.selected_entry else []
                        selected_scores = []
                        if decision.selected_memory_id in decision.candidate_memory_ids:
                            selected_index = decision.candidate_memory_ids.index(
                                decision.selected_memory_id
                            )
                            if selected_index < len(decision.candidate_scores):
                                selected_scores = [decision.candidate_scores[selected_index]]
                        record["retrieval_candidate_count"] = candidate_count
                        record["retrieval_candidate_memory_ids"] = decision.candidate_memory_ids
                        record["retrieval_candidate_scores"] = decision.candidate_scores
                        record["retrieval_candidate_relevance_scores"] = decision.relevance_scores
                        record["retrieval_selected_memory_id"] = decision.selected_memory_id
                        record["retrieval_relevance_decision"] = decision.relevance_decision
                        record["retrieval_rejection_reason"] = decision.rejection_reason
                        record["retrieval_filtered_by_type_count"] = (
                            decision.filtered_by_type_count
                        )
                        record["retrieval_filtered_by_safety_count"] = (
                            decision.filtered_by_safety_count
                        )
                        record["retrieval_filtered_by_intent_count"] = (
                            decision.filtered_by_intent_count
                        )
                        record["retrieval_current_intent"] = decision.current_intent
                        record["retrieval_candidate_intents"] = (
                            decision.candidate_intents or []
                        )
                        record["retrieved_memory_scores"] = selected_scores
                        record["retrieved_memory_ids"] = [
                            entry.memory_id for entry in retrieved
                        ]
                        logger.info(
                            "    Retrieval candidates ids=%s scores=%s relevance=%s "
                            "intents=%s current_intent=%s selected=%s reason=%s",
                            decision.candidate_memory_ids,
                            decision.candidate_scores,
                            decision.relevance_scores,
                            decision.candidate_intents,
                            decision.current_intent,
                            decision.selected_memory_id,
                            decision.rejection_reason,
                        )
                        if retrieved:
                            current_retrieved = retrieved
                            current_failure_context = {
                                "action": action,
                                "observation": observation,
                                "failure_type": det.failure_type,
                                "detector_source": det.detector_source,
                                "failure_reason": det.reason,
                            }
                            logger.info(f"    Retrieved {len(retrieved)} memories")
                            for entry in retrieved:
                                logger.info(
                                    "    Selected memory #%s: %s",
                                    entry.memory_id,
                                    _format_memory_for_log(entry),
                                )
                            pending_memory_record = record
                        elif (
                            self.enable_rule_repair_advice
                            and det.detector_source in ("judge", "rule")
                            and (
                                det.detector_source != "judge"
                                or step_num < self.max_steps - 1
                            )
                        ):
                            advice_source = (
                                "implicit_judge_repair"
                                if det.detector_source == "judge"
                                else "rule_repair_judge"
                            )
                            generator = getattr(self.detector, "generate_repair_advice", None)
                            if generator is None and det.detector_source == "rule":
                                generator = getattr(
                                    self.detector,
                                    "generate_rule_repair_advice",
                                    None,
                                )
                            advice = None
                            if generator is not None:
                                advice = generator(
                                    action=action,
                                    observation=observation,
                                    failure_type=det.failure_type,
                                    failure_reason=det.reason,
                                    advice_source=advice_source,
                                    task_type=task_type,
                                    task_goal=task_goal,
                                    recent_history=detector_history[-10:],
                                    score_before_action=score_before_action,
                                    score_after_action=final_score,
                                    step=step_num,
                                    variation_idx=variation_idx,
                                    look_after_action=look_after_action,
                                    inventory_after_action=inventory_after_action,
                                    valid_actions_after_action=valid_actions_after_action,
                                )
                            elif (
                                det.detector_source == "judge"
                                and det.has_judge_repair_advice
                            ):
                                # Compatibility for older test doubles/results; the compact
                                # detector itself no longer produces repair advice inline.
                                advice = RepairAdvice(
                                    repair_strategy=det.judge_repair_strategy,
                                    repair_action=det.judge_repair_action,
                                    repair_confidence=det.judge_repair_confidence or 0.0,
                                    repair_rationale=det.judge_repair_rationale,
                                    source=det.judge_advice_source or advice_source,
                                )
                            if advice is not None:
                                if not record["judge_call_type"]:
                                    record["judge_call_type"] = "repair_advice"
                                current_failure_context = {
                                    "action": action,
                                    "observation": observation,
                                    "failure_type": det.failure_type,
                                    "detector_source": det.detector_source,
                                    "failure_reason": det.reason,
                                }
                                current_judge_advice = {
                                    "repair_strategy": advice.repair_strategy,
                                    "repair_action": advice.repair_action,
                                    "repair_confidence": advice.repair_confidence,
                                    "repair_rationale": advice.repair_rationale,
                                }
                                record["judge_repair_strategy"] = advice.repair_strategy
                                record["judge_repair_action"] = advice.repair_action
                                record["judge_repair_confidence"] = advice.repair_confidence
                                record["judge_repair_rationale"] = advice.repair_rationale
                                record["judge_advice_source"] = advice.source
                                pending_judge_advice_record = record
                                logger.info(
                                    "    Scheduling %s advice fallback: strategy=%s action=%s "
                                    "confidence=%s rationale=%s",
                                    "judge" if det.detector_source == "judge" else "rule repair",
                                    _compact_log_text(advice.repair_strategy, 160),
                                    _compact_log_text(advice.repair_action, 120),
                                    advice.repair_confidence,
                                    _compact_log_text(advice.repair_rationale, 180),
                                )

            steps.append(record)
            history.append((action, observation))
            detector_history.append(
                {
                    "step": step_num,
                    "action": action,
                    "observation": observation,
                    "score": final_score,
                }
            )
            current_look = look_after_action
            current_inventory = inventory_after_action
            current_valid_actions = valid_actions_after_action

            if is_done:
                success = is_success
                break

        # Post-episode: extract and store memories
        memories_stored = 0
        if self.enable_memory and self.extractor_llm is not None:
            detected_failures = [
                {
                    "step": step["step"],
                    "action": step["action"],
                    "observation": step["observation"],
                    "failure_type": step.get("failure_type", ""),
                    "detector_source": step.get("detector_source", ""),
                    "score_before_action": step.get("score_before_action"),
                    "score_after_action": step.get("score_after_action"),
                    "score_delta": _score_delta(
                        step.get("score_before_action"),
                        step.get("score_after_action"),
                    ),
                }
                for step in steps
                if step.get("failure_detected")
                and not step.get("is_think")
                and step.get("failure_type") != "unproductive"
            ]
            recoveries = extract_failure_recoveries(
                self.extractor_llm,
                env_history_records,
                detected_failures=detected_failures,
            )
            for rec in recoveries:
                self.memory.add(
                    failure_action=rec["failure_action"],
                    failure_observation=rec["failure_observation"],
                    solution_action=rec["solution_action"],
                    task_type=task_type, env_idx=env_idx,
                    repair_strategy=rec.get("repair_strategy", ""),
                    repair_tactic=rec.get("repair_tactic", ""),
                    repair_action=rec.get("repair_action", ""),
                    failure_step=rec.get("failure_step"),
                    failure_type=rec.get("failure_type", ""),
                    detector_source=rec.get("detector_source", ""),
                    score_before_action=rec.get("score_before_action"),
                    score_after_action=rec.get("score_after_action"),
                    score_delta=rec.get("score_delta"),
                    source_episode_success=success,
                    source_episode_score=final_score,
                    confidence_score=rec.get("confidence_score"),
                )
                memories_stored += 1

        wall_time = time.time() - t0
        agent_stats = self.llm.tracker.summary()
        judge_tokens = 0
        judge_detector_tokens = 0
        judge_repair_tokens = 0
        judge_calls = 0
        if self.detector is not None and self.detector.judge_llm is not None:
            judge_tracker = self.detector.judge_llm.tracker
            judge_tokens = judge_tracker.summary()["total_tokens"]
            for call in getattr(judge_tracker, "call_log", []):
                call_tokens = int(call.get("input_tokens", 0)) + int(
                    call.get("output_tokens", 0)
                )
                label = call.get("label", "")
                if label == "judge":
                    judge_detector_tokens += call_tokens
                    judge_calls += 1
                elif label in {"rule_repair", "repair_advice"}:
                    judge_repair_tokens += call_tokens
                    judge_calls += 1
            if judge_tokens and not getattr(judge_tracker, "call_log", []):
                judge_detector_tokens = judge_tokens
                judge_calls = judge_tracker.summary().get("total_calls", 0)
        extractor_tokens = 0
        if self.extractor_llm is not None:
            extractor_tokens = self.extractor_llm.tracker.summary()["total_tokens"]
        total_tokens = agent_stats["total_tokens"] + judge_tokens + extractor_tokens

        result = {
            "env_idx": env_idx,
            "task_type": task_type,
            "variation_idx": variation_idx,
            "task_description": init_obs[:200],
            "success": success,
            "score": final_score,
            "total_steps": len(steps),
            "total_tokens": total_tokens,
            "agent_tokens": agent_stats["total_tokens"],
            "judge_tokens": judge_tokens,
            "judge_detector_tokens": judge_detector_tokens,
            "judge_repair_tokens": judge_repair_tokens,
            "judge_calls": judge_calls,
            "judge_cache_hits": sum(
                1 for step in steps if step.get("judge_cache_hit")
            ),
            "extractor_tokens": extractor_tokens,
            "failures_detected": failures_detected,
            "memories_retrieved": memories_retrieved_total,
            "memories_stored": memories_stored,
            "wall_time_s": round(wall_time, 2),
            "skipped": False,
            "steps": steps,
        }

        status = "SUCCESS" if success else "FAIL"
        logger.info(
            f"Env #{env_idx} [{task_type}] {status} score={final_score:.2f} in {len(steps)} steps, "
            f"tokens={total_tokens}, failures={failures_detected}, mem={memories_retrieved_total}/{memories_stored}"
        )
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
    run_name = args.run_name or f"sw_{mode}_"

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
            min_score=config["memory"].get("min_score", 0.0),
            retrieval_mode=args.retrieval_mode,
            scope="task_type",
        )
        if args.resume_memory:
            memory_store.load(args.resume_memory)

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

    enable_implicit_failures = _resolve_enable_implicit_failures(args, config)
    detector = None if is_baseline else ScienceWorldFailureDetector(
        judge_llm=judge_llm,
        enable_implicit_failures=enable_implicit_failures,
        implicit_failure_confidence_threshold=config.get("judge", {}).get(
            "implicit_failure_confidence_threshold",
            0.8,
        ),
        judge_max_tokens=config.get("judge", {}).get("max_tokens", 512),
        detector_max_tokens=config.get("judge", {}).get("detector_max_tokens", 128),
        repair_max_tokens=config.get("judge", {}).get("repair_max_tokens", 192),
        repair_confidence_threshold=config.get("judge", {}).get(
            "repair_confidence_threshold",
            0.7,
        ),
        enable_cache=config.get("judge", {}).get("enable_cache", True),
        enable_repair_intent_filter=config.get("judge", {}).get(
            "enable_repair_intent_filter",
            True,
        ),
    )

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

    agent = ScienceWorldReActAgent(
        llm=llm,
        memory_store=memory_store,
        failure_detector=detector,
        extractor_llm=extractor_llm,
        max_steps=min(args.step_limit, config["agent"]["max_steps"]),
        max_memory_inject=config["agent"]["max_memory_inject"],
        enable_memory=not is_baseline,
        inject_mode=args.inject_mode,
        memory_style=args.memory_style,
        memory_format=args.memory_format,
        baseline_mode=is_baseline,
        prompt_history_window=config["agent"].get("prompt_history_window", 10),
        retrieval_candidate_k=config["memory"].get("retrieval_candidate_k", 5),
        enable_failure_type_filter=config["memory"].get("enable_failure_type_filter", True),
        enable_safety_gate=config["memory"].get("enable_safety_gate", True),
        relevance_score_threshold=config["memory"].get("relevance_score_threshold", 0.45),
        enable_relevance_judge=config["memory"].get("enable_relevance_judge", False),
        enable_intent_gate=config["memory"].get("enable_intent_gate", True),
        enable_rule_repair_advice=config.get("judge", {}).get(
            "enable_rule_repair_advice",
            True,
        ),
    )

    task_names = args.tasks or DEFAULT_EVAL_TASKS

    # Track per-env success across epochs: env_idx -> (task_type, variation_idx, score)
    env_success: dict[int, tuple[str, int | None, float]] = {}

    if args.resume_results:
        with open(args.resume_results) as f:
            prev_data = json.load(f)
        for ep in prev_data.get("episodes", []):
            idx = ep["env_idx"]
            if ep.get("success"):
                env_success[idx] = (
                    ep.get("task_type", "unknown"),
                    ep.get("variation_idx"),
                    ep.get("score", 100),
                )
        logger.info(f"Loaded {len(env_success)} succeeded envs from {args.resume_results}")

    for epoch in range(1, num_epochs + 1):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  Epoch {epoch}/{num_epochs} — {mode} (ScienceWorld)")
        logger.info("=" * 60)

        env = ScienceWorldEnv(
            task_names=task_names,
            split=args.split,
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
            test_time_writable=bool(not is_baseline and args.split == "test" and extractor_llm is not None),
            memory_scope=_memory_scope_for_protocol(memory_store),
            max_envs=max_envs,
        )
        env_count = 0
        num_skipped = 0

        while env_count < max_envs:
            try:
                # Skip already-succeeded envs in later epochs
                if epoch > 1 and (env_count + 1) in env_success:
                    env.skip()
                    prev_task_type, prev_variation_idx, prev_score = env_success[env_count + 1]
                    episode_results.append({
                        "env_idx": env_count + 1,
                        "task_type": prev_task_type,
                        "variation_idx": prev_variation_idx,
                        "task_description": "",
                        "success": True,
                        "score": prev_score,
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
                    env_count += 1
                    logger.info(f"  [{env_count}] SKIP (already succeeded)")
                    continue

                result = agent.run_episode(env, env_idx=env_count + 1)
                episode_results.append(result)

                if result["success"]:
                    env_success[env_count + 1] = (
                        result["task_type"],
                        result.get("variation_idx"),
                        result.get("score", 100),
                    )

                env_count += 1

                successes = sum(1 for r in episode_results if r["success"])
                rate = successes / len(episode_results)
                logger.info(f"  [{env_count}] {result['task_type']:<30} "
                            f"{'OK' if result['success'] else 'FAIL':>4} "
                            f"score={result['score']:.2f} "
                            f"steps={result['total_steps']:<3} "
                            f"running={rate:.1%} ({successes}/{env_count})")

                # Intermediate save every 10 envs
                if env_count % 10 == 0:
                    mem_stats = memory_store.stats() if memory_store else {}
                    summary = compute_summary(episode_results, mem_stats, mode, protocol)
                    save_results(
                        episode_results, summary,
                        f"{results_dir}/{run_name}{epoch}_intermediate.json"
                    )
                    if memory_store:
                        memory_store.save(f"{memory_dir}/sw_epoch{epoch}_intermediate.json")

            except StopIteration:
                break
            except Exception as e:
                logger.error(f"Error on env #{env_count + 1}: {e}", exc_info=True)
                episode_results.append({
                    "env_idx": env_count + 1,
                    "task_type": _infer_task_name_from_env(env),
                    "variation_idx": _infer_variation_idx_from_env(env),
                    "task_description": "",
                    "success": False,
                    "score": -100.0,
                    "total_steps": 0,
                    "total_tokens": 0,
                    "agent_tokens": 0,
                    "judge_tokens": 0,
                    "extractor_tokens": 0,
                    "failures_detected": 0,
                    "memories_retrieved": 0,
                    "memories_stored": 0,
                    "wall_time_s": 0,
                    "skipped": False,
                    "steps": [],
                })
                env_count += 1

        if num_skipped > 0:
            logger.info(f"  Skipped {num_skipped} already-succeeded envs")

        mem_stats = memory_store.stats() if memory_store else {}
        summary = compute_summary(episode_results, mem_stats, mode, protocol)
        log_summary(summary)

        result_path = f"{results_dir}/{run_name}{epoch}.json"
        save_results(episode_results, summary, result_path)
        logger.info(f"Results saved: {result_path}")

        failure_event_path = f"{results_dir}/{run_name}{epoch}_failure_events.jsonl"
        failure_event_count = write_failure_events_jsonl(
            episode_results,
            failure_event_path,
            memory_mode="baseline" if is_baseline else args.inject_mode,
        )
        logger.info(f"Failure events saved: {failure_event_path} ({failure_event_count} rows)")

        if memory_store:
            mem_path = f"{memory_dir}/sw_epoch{epoch}.json"
            memory_store.save(mem_path)
            logger.info(f"Memory saved: {mem_path} ({memory_store.size()} entries)")

        # Close env to free JVM
        env.close()

    logger.info("Done!")


if __name__ == "__main__":
    main()

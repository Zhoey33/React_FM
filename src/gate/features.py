"""11-dimensional feature extractor for the intervention gate.

Extracts features from agent state at a failure checkpoint.
All features are observable at inference time — no privileged environment state.
"""

import logging
from dataclasses import dataclass

import numpy as np

from src.gate.checkpoint import CheckpointState

logger = logging.getLogger(__name__)

# Feature names in order (for XGBoost + SHAP)
FEATURE_NAMES = [
    "failure_type",           # 0: categorical → int encoding
    "task_type",              # 1: categorical → int encoding
    "step_index",             # 2: normalized by max_steps
    "action_repetition_count",  # 3: how many times current action repeated consecutively
    "retrieval_rrf_score",    # 4: top-1 RRF score
    "retrieval_margin",       # 5: top-1 minus top-2 RRF
    "memory_entry_count",     # 6: total memories available
    "history_token_count",    # 7: approximate token count of history
    "progress_since_last_failure",  # 8: binary — any progress since last failure?
    "failures_in_last_5_steps",  # 9: count of failures in last 5 non-think steps
    "same_failure_recurrence_count",  # 10: how many times same failure type recurred
]

# Encoding maps (built from training data, can be extended)
FAILURE_TYPE_MAP = {
    "nothing_happens": 0,
    "empty_obs": 1,
    "action_loop": 2,
    "unproductive": 3,
    "unknown_action": 4,
    "action_failed": 5,
    "intent_mismatch": 6,
    # ScienceWorld middle-grain failure types
    "syntax_or_parse": 7,
    "ambiguity": 8,
    "precondition_blocked": 9,
    "physics_or_affordance": 10,
    "no_effect_or_other": 11,
}

# Task types across benchmarks
TASK_TYPE_MAP = {
    # ALFWorld (long names from task_type detection)
    "pick_and_place": 0, "pick_clean_then_place": 1, "pick_heat_then_place": 2,
    "pick_cool_then_place": 3, "look_at_obj": 4, "pick_two_obj": 5,
    # ALFWorld (short names emitted by alfworld_env.py)
    "put": 0, "clean": 1, "heat": 2, "cool": 3, "examine": 4, "puttwo": 5,
    # WebShop
    "shopping": 6,
    # ScienceWorld
    "boil": 7, "melt": 8, "freeze": 9, "change-the-state-of-matter-of": 10,
    "use-thermometer": 11, "measure-melting-point-known-substance": 12,
    "measure-melting-point-unknown-substance": 13,
    "power-component": 14, "power-component-renewable-vs-nonrenewable-energy": 15,
    "test-conductivity": 16, "test-conductivity-of-unknown-substances": 17,
    "find-animal": 18, "find-living-thing": 19, "find-non-living-thing": 20,
    "find-plant": 21, "grow-plant": 22, "grow-fruit": 23,
    "chemistry-mix": 24, "chemistry-mix-paint-secondary-color": 25,
    "chemistry-mix-paint-tertiary-color": 26,
    "lifespan-longest-lived": 27, "lifespan-shortest-lived": 28,
    "lifespan-longest-lived-then-shortest-lived": 29,
    "identify-life-stages-1": 30, "identify-life-stages-2": 31,
    "inclined-plane-determine-angle": 32, "inclined-plane-friction-named-surfaces": 33,
    "inclined-plane-friction-unnamed-surfaces": 34,
    "mendelian-genetics-known-plant": 35, "mendelian-genetics-unknown-plant": 36,
}


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return len(text) // 4


def _count_action_repetitions(action: str, action_history: list[str]) -> int:
    """Count how many times the current action appears consecutively at the end."""
    count = 0
    for a in reversed(action_history):
        if a == action:
            count += 1
        else:
            break
    return count


def _count_failures_in_window(
    step_records: list[dict],
    window: int = 5,
) -> int:
    """Count failures in last `window` non-think steps."""
    non_think = [s for s in step_records if not s.get("is_think", False)]
    recent = non_think[-window:]
    return sum(1 for s in recent if s.get("failure_detected", False))


def _count_same_failure_recurrence(
    failure_type: str,
    step_records: list[dict],
) -> int:
    """Count how many previous failures had the same failure_type."""
    count = 0
    for s in step_records:
        if s.get("failure_detected") and s.get("failure_type") == failure_type:
            count += 1
    return count


def extract_features_from_checkpoint(cp: CheckpointState) -> np.ndarray:
    """Extract 11-dim feature vector from a CheckpointState.

    Returns:
        np.ndarray of shape (11,) with float values.
    """
    features = np.zeros(11, dtype=np.float32)

    # 0: failure_type (categorical → int)
    features[0] = FAILURE_TYPE_MAP.get(cp.failure_type, len(FAILURE_TYPE_MAP))

    # 1: task_type (categorical → int)
    features[1] = TASK_TYPE_MAP.get(cp.task_type, len(TASK_TYPE_MAP))

    # 2: step_index (normalized)
    features[2] = cp.step_idx / max(cp.max_steps, 1)

    # 3: action_repetition_count
    features[3] = _count_action_repetitions(cp.failure_action, cp.action_history)

    # 4: retrieval_rrf_score (top-1)
    features[4] = cp.retrieval_rrf_score

    # 5: retrieval_margin (top-1 minus top-2)
    features[5] = cp.retrieval_margin

    # 6: memory_entry_count
    features[6] = cp.memory_entry_count

    # 7: history_token_count (approximate)
    history_text = " ".join(f"{a} {o}" for a, o in cp.history)
    features[7] = _estimate_tokens(history_text)

    # 8: progress_since_last_failure (binary)
    # True if score increased since the previous failure checkpoint
    score_at_last_failure = cp.env_state.get("score_at_last_failure", 0.0)
    features[8] = 1.0 if cp.score_at_checkpoint > score_at_last_failure else 0.0

    # 9: failures_in_last_5_steps (consistent: count from step records)
    features[9] = cp.env_state.get("failures_in_last_5", 0)

    # 10: same_failure_recurrence_count
    features[10] = cp.env_state.get("same_failure_count", 0)

    return features


def extract_features_from_agent_state(
    failure_type: str,
    failure_action: str,
    task_type: str,
    step_idx: int,
    max_steps: int,
    action_history: list[str],
    history: list[tuple[str, str]],
    retrieval_rrf_score: float,
    retrieval_margin: float,
    memory_entry_count: int,
    score_at_checkpoint: float,
    step_records: list[dict],
    score_at_last_failure: float = 0.0,
) -> np.ndarray:
    """Extract 11-dim feature vector directly from live agent state.

    This is the version used during online evaluation (inference time).
    """
    features = np.zeros(11, dtype=np.float32)

    features[0] = FAILURE_TYPE_MAP.get(failure_type, len(FAILURE_TYPE_MAP))
    features[1] = TASK_TYPE_MAP.get(task_type, len(TASK_TYPE_MAP))
    features[2] = step_idx / max(max_steps, 1)
    features[3] = _count_action_repetitions(failure_action, action_history)
    features[4] = retrieval_rrf_score
    features[5] = retrieval_margin
    features[6] = memory_entry_count

    history_text = " ".join(f"{a} {o}" for a, o in history)
    features[7] = _estimate_tokens(history_text)

    # Feature 8: progress since LAST failure (not cumulative score)
    features[8] = 1.0 if score_at_checkpoint > score_at_last_failure else 0.0
    # Feature 9: use same non-think window as offline
    features[9] = _count_failures_in_window(step_records)
    features[10] = _count_same_failure_recurrence(failure_type, step_records)

    return features

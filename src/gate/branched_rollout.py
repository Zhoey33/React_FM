"""Branched rollout pipeline for gate training data collection.

Given a failure checkpoint, branch into multiple arms (none, cue, question, repair),
run N steps forward with M replays, and compute replay-averaged progress + disruption.

This is the core data collection mechanism for the intervention gate.
"""

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from src.gate.checkpoint import CheckpointState
from src.gate.progress import compute_progress, compute_utility
from src.gate.canonicalize import CUE_TEXT
from src.memory import FailureMemoryEntry

logger = logging.getLogger(__name__)

# The 4 arms used in branched rollouts
# (cue is for scientific analysis only, NOT in gate action space)
ROLLOUT_ARMS = ["none", "cue", "question", "repair"]
GATE_ARMS = ["none", "question", "repair"]  # arms the gate can select

ARM_INDEX = {arm: i for i, arm in enumerate(ROLLOUT_ARMS)}


@dataclass
class RolloutResult:
    """Result of a single branched rollout (one arm, one replay)."""
    checkpoint_id: str
    arm: str
    replay_idx: int
    steps_taken: int
    score_before: float
    score_after: float
    score_max: float = 0.0  # max score seen during rollout window
    done_before: bool = False
    done_after: bool = False
    reward_after: float = 0.0
    progress: float = 0.0
    actions: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "checkpoint_id": self.checkpoint_id,
            "arm": self.arm,
            "replay_idx": self.replay_idx,
            "steps_taken": self.steps_taken,
            "score_before": self.score_before,
            "score_after": self.score_after,
            "score_max": self.score_max,
            "done_before": self.done_before,
            "done_after": self.done_after,
            "reward_after": self.reward_after,
            "progress": self.progress,
            "actions": self.actions,
            "observations": [o[:200] for o in self.observations],
            "observations_full": self.observations,
        }


@dataclass
class ArmUtility:
    """Aggregated utility for one (checkpoint, arm) pair over replays."""
    checkpoint_id: str
    arm: str
    replays: list[RolloutResult] = field(default_factory=list)
    progress_avg: float = 0.0
    disruption_avg: float = 0.0
    utility: float = 0.0

    def to_dict(self) -> dict:
        return {
            "checkpoint_id": self.checkpoint_id,
            "arm": self.arm,
            "n_replays": len(self.replays),
            "progress_avg": round(self.progress_avg, 4),
            "disruption_avg": round(self.disruption_avg, 4),
            "utility": round(self.utility, 4),
        }


def build_injection_payload(
    arm: str,
    memory_entries: list[FailureMemoryEntry] | None,
) -> tuple[list[FailureMemoryEntry] | None, str | None]:
    """Return one-step prompt injection payload for a given arm.

    Returns:
        (retrieved_memories, hint_text)
    """
    if arm == "none":
        return None, None

    if arm == "cue":
        return None, CUE_TEXT

    if not memory_entries:
        logger.warning(f"No memory entry for arm={arm}, falling back to none")
        return None, None

    top1_entry = memory_entries[0]

    if arm == "question":
        if top1_entry.question_text:
            return None, top1_entry.question_text
        return None, f"What went wrong when you tried '{top1_entry.failure_action}'?"

    if arm == "repair":
        return memory_entries, None

    raise ValueError(f"Unknown arm: {arm}")


def run_branched_rollout(
    checkpoint: CheckpointState,
    arm: str,
    replay_idx: int,
    llm,  # LLMClient
    env,  # environment wrapper
    build_prompt_fn,  # function(task_type, init_obs, history, hint_text) -> prompt
    n_steps: int = 5,
    memory_entries: list[FailureMemoryEntry] | None = None,
    hint_steps: int = 1,
    system_prompt: str | None = None,
    action_postprocess: Callable[[str], str] | None = None,
) -> RolloutResult:
    """Run a single branched rollout from a checkpoint.

    IMPORTANT: The caller must ensure the environment is at the checkpoint
    state before calling this function. For ScienceWorld, this means:
    1. Reset the env to the same task/variation
    2. Replay the action_history from the checkpoint deterministically
    3. Then call this function to branch forward

    This function resumes from the checkpoint's agent state (history),
    injects the arm content into the first hint_steps prompts, then
    continues without injection for remaining steps.

    Args:
        checkpoint: the saved state to resume from
        arm: "none", "cue", "question", or "repair"
        replay_idx: which replay this is (for logging)
        llm: LLM client for agent completions
        env: environment (must already be at the checkpoint state)
        build_prompt_fn: function to build agent prompt
        n_steps: number of steps to run forward
        memory_entries: retrieved memories for question/repair arms
        hint_steps: number of steps to persist the injection (default 1)

    Returns:
        RolloutResult with progress information
    """
    # Build one-step injection payload.
    injected_memories, injected_hint = build_injection_payload(arm, memory_entries)

    # Copy history from checkpoint — INCLUDE the failure step that triggered
    # the checkpoint, so the agent's prompt matches the real decision point
    history = list(checkpoint.history)
    history.append((checkpoint.failure_action, checkpoint.failure_observation))
    action_history = list(checkpoint.action_history)

    score_before = checkpoint.score_at_checkpoint
    score_after = score_before
    score_max = score_before  # track window max for ScienceWorld
    done_before = False
    done_after = False
    reward_after = 0.0

    actions_taken = []
    observations_seen = []

    for step_i in range(n_steps):
        # Build prompt — inject hint for first hint_steps steps
        if step_i < hint_steps and (injected_memories or injected_hint):
            prompt = build_prompt_fn(
                task_type=checkpoint.task_type,
                task_obs=checkpoint.init_obs,
                history=history,
                retrieved_memories=injected_memories,
                memory_style="original",
                hint_text=injected_hint,
            )
        else:
            prompt = build_prompt_fn(
                task_type=checkpoint.task_type,
                task_obs=checkpoint.init_obs,
                history=history,
                retrieved_memories=None,
                memory_style="original",
                hint_text=None,
            )

        # Get agent response
        try:
            response = llm.complete_text(
                prompt, stop=["\n"],
                label=f"rollout_{checkpoint.checkpoint_id}_{arm}_{replay_idx}_s{step_i}",
                system=system_prompt,
            )
        except Exception as e:
            logger.warning(f"LLM call failed in rollout: {e}")
            break

        action = response.strip().split("\n")[0].strip()
        if action.startswith("> "):
            action = action[2:]
        if action_postprocess is not None:
            action = action_postprocess(action)
        if not action:
            action = "look" if checkpoint.benchmark == "alfworld" else "look around"

        # Handle think actions
        is_think = action.startswith("think:") or action.startswith("think ")
        if is_think:
            observation = "OK."
            history.append((action, observation))
            actions_taken.append(action)
            observations_seen.append(observation)
            continue

        # Execute in environment
        try:
            observation, reward, done, step_info = env.step(action)
        except Exception as e:
            logger.warning(f"Env step failed in rollout: {e}")
            break

        action_history.append(action)
        actions_taken.append(action)
        observations_seen.append(observation)
        history.append((action, observation))

        # Update score tracking
        if checkpoint.benchmark == "scienceworld":
            score_after = step_info.get("score", score_after)
            score_max = max(score_max, score_after)
        elif checkpoint.benchmark == "webshop":
            if reward > 0:
                score_after = reward
                reward_after = reward
                score_max = max(score_max, score_after)
        elif checkpoint.benchmark == "alfworld":
            reward_after = reward

        if done:
            done_after = True
            break

    # Compute progress — use score_max for ScienceWorld (max-in-window),
    # score_after for other benchmarks
    progress_score = score_max if checkpoint.benchmark == "scienceworld" else score_after

    # For ALFWorld, pass action+observation histories for milestone-based scoring.
    # Filter out think actions from milestone counting.
    # Build paired (action, observation) lists so the milestone scorer can check
    # whether each action succeeded (observation != "Nothing happens").
    env_actions_before = []
    env_obs_before = []
    for a, o in checkpoint.history:
        if not a.startswith("think"):
            env_actions_before.append(a)
            env_obs_before.append(o)
    # Also include the failure action+observation from the checkpoint itself
    if not checkpoint.failure_action.startswith("think"):
        env_actions_before.append(checkpoint.failure_action)
        env_obs_before.append(checkpoint.failure_observation)

    env_actions_after = list(env_actions_before)
    env_obs_after = list(env_obs_before)
    for a, o in zip(actions_taken, observations_seen):
        if not a.startswith("think"):
            env_actions_after.append(a)
            env_obs_after.append(o)

    progress = compute_progress(
        benchmark=checkpoint.benchmark,
        score_before=score_before,
        score_after=progress_score,
        done_before=done_before,
        done_after=done_after,
        reward_after=reward_after,
        task_type=checkpoint.task_type,
        actions_before=env_actions_before,
        actions_after=env_actions_after,
        observations_before=env_obs_before,
        observations_after=env_obs_after,
    )

    result = RolloutResult(
        checkpoint_id=checkpoint.checkpoint_id,
        arm=arm,
        replay_idx=replay_idx,
        steps_taken=len(actions_taken),
        score_before=score_before,
        score_after=score_after,
        score_max=score_max,
        done_before=done_before,
        done_after=done_after,
        reward_after=reward_after,
        progress=progress,
        actions=actions_taken,
        observations=observations_seen,
    )

    logger.debug(
        f"Rollout {checkpoint.checkpoint_id}/{arm}/r{replay_idx}: "
        f"progress={progress:.4f}, steps={len(actions_taken)}"
    )
    return result


def compute_arm_utilities(
    rollouts_by_arm: dict[str, list[RolloutResult]],
    checkpoint_id: str,
    beta: float = 0.3,
) -> dict[str, ArmUtility]:
    """Compute replay-averaged utility for each arm.

    Disruption is computed per-replay by matching replay_idx:
      disruption_i = max(0, progress(none, replay_i) - progress(arm, replay_i))
    Then disruption_avg = mean over replays.

    For arm=none, disruption is always 0.

    Args:
        rollouts_by_arm: {arm_name: [RolloutResult, ...]} for N replays
        checkpoint_id: ID of the checkpoint
        beta: disruption penalty weight

    Returns:
        {arm_name: ArmUtility}
    """
    results = {}

    # Index none replays by replay_idx for matched disruption
    none_replays = rollouts_by_arm.get("none", [])
    none_by_idx = {r.replay_idx: r.progress for r in none_replays}

    for arm, replays in rollouts_by_arm.items():
        if not replays:
            continue

        progress_values = [r.progress for r in replays]
        progress_avg = float(np.mean(progress_values))

        if arm == "none":
            # No disruption for the no-intervention baseline
            disruption_avg = 0.0
        else:
            # Per-replay matched disruption
            disruption_values = []
            for r in replays:
                none_progress = none_by_idx.get(r.replay_idx, 0.0)
                disruption_values.append(max(0, none_progress - r.progress))
            disruption_avg = float(np.mean(disruption_values)) if disruption_values else 0.0

        utility = compute_utility(progress_avg, disruption_avg, beta=beta)

        results[arm] = ArmUtility(
            checkpoint_id=checkpoint_id,
            arm=arm,
            replays=replays,
            progress_avg=progress_avg,
            disruption_avg=disruption_avg,
            utility=utility,
        )

    return results


def find_oracle_arm(
    arm_utilities: dict[str, ArmUtility],
    gate_arms_only: bool = True,
) -> str:
    """Find the arm with highest utility (oracle selection).

    Args:
        arm_utilities: {arm_name: ArmUtility}
        gate_arms_only: if True, only consider GATE_ARMS (exclude cue)
    """
    best_arm = "none"
    best_utility = float("-inf")

    for arm, au in arm_utilities.items():
        if gate_arms_only and arm not in GATE_ARMS:
            continue
        if au.utility > best_utility:
            best_utility = au.utility
            best_arm = arm

    return best_arm


def replay_to_checkpoint(
    checkpoint: CheckpointState,
    env,
) -> float:
    """Replay stored actions to reach the checkpoint state in the environment.

    The environment must already be reset to the correct task/variation.
    This replays the action_history deterministically to reach the
    checkpoint's env state.

    Args:
        checkpoint: checkpoint to replay to
        env: environment (already reset to correct task/variation)

    Returns:
        Current score after replay

    Raises:
        RuntimeError: If replay cannot reliably reproduce the checkpoint state.
    """
    current_score = 0.0
    for action_idx, action in enumerate(checkpoint.action_history):
        try:
            obs, reward, done, info = env.step(action)
            if checkpoint.benchmark == "scienceworld":
                current_score = info.get("score", current_score)
            elif checkpoint.benchmark == "webshop":
                if reward > 0:
                    current_score = reward
            if done:
                raise RuntimeError(
                    "Replay terminated before reaching checkpoint state: "
                    f"action_idx={action_idx}, action={action[:50]!r}"
                )
        except Exception as e:
            raise RuntimeError(
                "Replay step failed before reaching checkpoint state: "
                f"action_idx={action_idx}, action={action[:50]!r}, error={e}"
            ) from e

    return current_score

"""LLM-as-judge scoring for ScienceWorld rollout trajectories.

Problem: ScienceWorld's built-in score has 37.5% "evaluation blindness" —
different arms produce different trajectories but identical progress scores (0).

Solution: Use an LLM to score each trajectory against the gold path on a 0-10
scale, then normalize to [0, 1] for use as the progress signal in utility
computation.

Integration:
    1. run_branched_rollouts.py collects rollouts (actions, observations, scores)
    2. run_llm_judge_scoring.py calls this module to score each rollout via LLM
    3. run_gate_training.py uses LLM judge scores as the progress signal
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Single-trajectory judge prompt ──────────────────────────────────────

SINGLE_TRAJ_JUDGE_PROMPT = """\
You are evaluating an AI agent's progress toward completing a task in a text-based science simulation.

## Task Description
{task_desc}

## Gold (Correct) Action Sequence
The following is a correct sequence of key actions to complete this task:
{gold_path}

## Agent's Trajectory
The agent started from this context, then took the following actions:

Previous context (last few steps before evaluation window):
{context}

Evaluation window (the actions being scored):
{trajectory}

## Scoring Instructions
Score the agent's trajectory on a 0-10 scale based on how much meaningful progress it makes toward the goal:

- 0: Agent takes no useful actions, completely stuck, repeating failed actions, or producing gibberish
- 1-2: Agent takes valid actions but in completely wrong direction (wrong room, wrong object)
- 3-4: Agent takes some reasonable actions but doesn't approach any gold path milestone
- 5-6: Agent moves in the right direction (correct room, relevant objects) but doesn't complete a key step
- 7-8: Agent completes one or more key steps from the gold path (picks up right object, reaches right location, performs correct action)
- 9-10: Agent completes multiple key gold path steps in sequence

Focus on:
1. Does the agent take actions that align with gold path key steps?
2. Does the agent make meaningful progress (even if the environment score doesn't change)?
3. Does the agent avoid wasting steps on invalid/unproductive actions?

Output in this EXACT format (two lines only):
SCORE: [0-10]
REASONING: [1-2 sentences]"""


def combine_progress_scienceworld(
    env_progress: float,
    llm_progress: float,
    llm_weight: float = 0.6,
) -> float:
    """Blend ScienceWorld env progress with LLM judge progress.

    Design:
    - env progress is treated as the authoritative floor
    - LLM progress adds soft credit in score-blind windows
    - the final value stays in [0, 1]
    """
    env_clamped = min(max(float(env_progress), 0.0), 1.0)
    llm_clamped = min(max(float(llm_progress), 0.0), 1.0)
    blended = (1.0 - llm_weight) * env_clamped + llm_weight * llm_clamped
    return max(env_clamped, min(blended, 1.0))


def _extract_key_gold_actions(gold_actions: list[str], max_display: int = 30) -> str:
    """Extract key actions from gold path, filtering redundant ones."""
    key_actions = []
    for i, action in enumerate(gold_actions):
        # Skip repeated thermometer checks
        if action in ("use thermometer in inventory on", "examine thermometer"):
            if not key_actions or not key_actions[-1].startswith("(repeated"):
                key_actions.append("(repeated temperature checks)")
            continue
        # Skip repeated "look around" after navigation
        if action == "look around" and i > 0 and gold_actions[i - 1].startswith("go to"):
            continue
        key_actions.append(f"  {i + 1}. {action}")
        if len(key_actions) >= max_display:
            key_actions.append(f"  ... ({len(gold_actions) - i - 1} more steps)")
            break
    return "\n".join(key_actions)


def _format_trajectory(actions: list[str], observations: list[str]) -> str:
    """Format a trajectory for the judge prompt."""
    if not actions:
        return "  (no actions taken)"
    lines = []
    for i, (act, obs) in enumerate(zip(actions, observations)):
        if act.startswith("think:") and len(act) > 200:
            act = act[:200] + "..."
        obs_short = obs[:150] + "..." if len(obs) > 150 else obs
        lines.append(f"  {i + 1}. Action: {act}")
        lines.append(f"     Observation: {obs_short}")
    return "\n".join(lines)


def _format_context(history: list, max_recent: int = 5) -> str:
    """Format checkpoint history as context."""
    if not history:
        return "  (start of episode)"
    recent = history[-max_recent:]
    lines = []
    for action, obs in recent:
        if isinstance(action, str) and isinstance(obs, str):
            act_short = action[:100] + "..." if len(action) > 100 else action
            obs_short = obs[:100] + "..." if len(obs) > 100 else obs
            lines.append(f"  > {act_short} → {obs_short}")
    if len(history) > max_recent:
        lines.insert(0, f"  (... {len(history) - max_recent} earlier steps omitted ...)")
    return "\n".join(lines) if lines else "  (start of episode)"


def build_single_traj_prompt(
    task_desc: str,
    gold_actions: list[str],
    context_history: list,
    traj_actions: list[str],
    traj_observations: list[str],
) -> str:
    """Build LLM judge prompt for a single trajectory."""
    return SINGLE_TRAJ_JUDGE_PROMPT.format(
        task_desc=task_desc,
        gold_path=_extract_key_gold_actions(gold_actions),
        context=_format_context(context_history),
        trajectory=_format_trajectory(traj_actions, traj_observations),
    )


def parse_judge_score(response_text: str) -> dict:
    """Parse structured score from LLM judge response."""
    result = {"score": None, "reasoning": None, "raw": response_text}
    for line in response_text.split("\n"):
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            try:
                score_str = line.split(":")[1].strip()
                # Handle "7/10" or "7" format
                if "/" in score_str:
                    score_str = score_str.split("/")[0].strip()
                result["score"] = int(score_str)
            except (ValueError, IndexError):
                pass
        elif line.upper().startswith("REASONING:"):
            result["reasoning"] = line.split(":", 1)[1].strip()
    return result


def score_trajectory(
    llm,
    task_desc: str,
    gold_actions: list[str],
    context_history: list,
    traj_actions: list[str],
    traj_observations: list[str],
    label: str = "llm_judge",
) -> dict:
    """Score a single trajectory using the LLM judge.

    Args:
        llm: LLMClient instance
        task_desc: task description from init_obs
        gold_actions: gold path action sequence
        context_history: [(action, obs), ...] before the evaluation window
        traj_actions: actions in the evaluation window
        traj_observations: observations in the evaluation window
        label: label for LLM tracking

    Returns:
        {"score": 0-10, "reasoning": str, "progress": 0.0-1.0}
    """
    prompt = build_single_traj_prompt(
        task_desc=task_desc,
        gold_actions=gold_actions,
        context_history=context_history,
        traj_actions=traj_actions,
        traj_observations=traj_observations,
    )

    try:
        response = llm.chat(
            messages=[
                {"role": "system", "content": "You are an expert evaluator of AI agent behavior in interactive environments. Be precise and consistent in your scoring."},
                {"role": "user", "content": prompt},
            ],
            label=label,
        )
        result = parse_judge_score(response.content)
        if result["score"] is not None:
            result["progress"] = result["score"] / 10.0  # normalize to [0, 1]
        else:
            result["progress"] = 0.0
            logger.warning(f"Failed to parse LLM judge score, defaulting to 0")
        return result

    except Exception as e:
        logger.error(f"LLM judge call failed: {e}")
        return {"score": None, "reasoning": f"LLM call failed: {e}", "progress": 0.0}


def get_gold_path_for_checkpoint(cp_data: dict, gold_paths: dict) -> list[str] | None:
    """Find gold path actions for a checkpoint.

    Args:
        cp_data: checkpoint dict with task_type and env_state.variation_idx
        gold_paths: {task_var_key: {task_name, variation_idx, gold_actions}}

    Returns:
        list of gold actions, or None if not found
    """
    task = cp_data.get("task_type", "")
    var_idx = cp_data.get("env_state", {}).get("variation_idx")

    if var_idx is not None:
        key = f"{task}_{var_idx}"
        if key in gold_paths:
            return gold_paths[key]["gold_actions"]
    logger.warning(
        "Missing exact gold path for checkpoint task=%s variation=%s; refusing task-level fallback",
        task,
        var_idx,
    )
    return None

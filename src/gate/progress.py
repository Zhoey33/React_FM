"""Benchmark-native progress signal extractors.

Used for OFFLINE utility labeling only — NOT inference-time inputs.
Each benchmark has a different notion of "progress":
  - ScienceWorld: normalized score delta (official API score / 100)
  - ALFWorld: milestone-based subgoal progress (fractional)
  - WebShop: reward delta (attribute matching score)
"""

import logging

logger = logging.getLogger(__name__)


def compute_progress_scienceworld(
    score_before: float,
    score_after: float,
) -> float:
    """ScienceWorld progress = normalized score delta.

    ScienceWorld reports score as round(100 * getScore()).
    Full success is 100, but terminal failures can be negative.
    Returns score delta divided by 100.
    """
    delta = (score_after - score_before) / 100.0
    return delta


def _is_successful_action(action: str, observation: str | None) -> bool:
    """Check if an ALFWorld action succeeded based on its observation.

    Failed actions return "Nothing happens." in ALFWorld.
    If no observation is provided, assume success (backward compat).
    """
    if observation is None:
        return True
    return "Nothing happens" not in observation


def _count_alfworld_milestones(
    task_type: str,
    actions: list[str],
    done: bool,
    reward: float,
    observations: list[str] | None = None,
) -> tuple[int, int]:
    """Count milestones achieved from an action sequence.

    Returns (achieved, total) milestone counts.

    Only actions that SUCCEEDED (observation != "Nothing happens") count
    toward milestones. This prevents counting failed attempts (e.g.,
    "clean X with sinkbasin" that returned "Nothing happens").

    Milestone templates per task family:
      put:     take → place                        (2 milestones)
      clean:   take → clean with sinkbasin → place (3 milestones)
      heat:    take → heat with microwave → place  (3 milestones)
      cool:    take → cool with fridge → place     (3 milestones)
      examine: take → use lamp                     (2 milestones)
      puttwo:  take1 → place1 → take2 → place2    (4 milestones)
    """
    completed = done and reward > 0

    # Build list of successful actions only
    obs_list = observations or [None] * len(actions)
    successful = [
        a for a, o in zip(actions, obs_list)
        if _is_successful_action(a, o)
    ]

    if task_type == "put":
        total = 2
        has_taken = any(a.startswith("take ") for a in successful)
        achieved = int(has_taken) + int(completed)

    elif task_type in ("clean", "heat", "cool"):
        total = 3
        has_taken = any(a.startswith("take ") for a in successful)
        verb = task_type
        has_verbed = any(a.startswith(f"{verb} ") for a in successful)
        achieved = int(has_taken) + int(has_verbed) + int(completed)

    elif task_type == "examine":
        total = 2
        has_taken = any(a.startswith("take ") for a in successful)
        has_lamp = any(
            a.startswith("use desklamp") or a.startswith("use floorlamp")
            for a in successful
        )
        achieved = int(has_taken) + int(has_lamp)

    elif task_type == "puttwo":
        total = 4
        # Track take→place pairs sequentially
        take_count = 0
        place_count = 0
        holding = False
        for a in successful:
            if a.startswith("take ") and take_count < 2:
                take_count += 1
                holding = True
            elif holding and (a.startswith("put ") or a.startswith("move ")):
                place_count += 1
                holding = False
        achieved = take_count + place_count
        if completed:
            achieved = total

    else:
        logger.warning(f"Unknown ALFWorld task type: {task_type}")
        return int(completed), 1

    return achieved, total


def compute_progress_alfworld(
    done_before: bool,
    done_after: bool,
    reward_after: float,
    task_type: str | None = None,
    actions_before: list[str] | None = None,
    actions_after: list[str] | None = None,
    observations_before: list[str] | None = None,
    observations_after: list[str] | None = None,
) -> float:
    """ALFWorld progress = milestone delta.

    When task_type and action histories are provided, uses milestone-based
    scoring that detects subgoal completion (take, verb, place). Progress is
    the fraction of new milestones achieved during the rollout window.

    Only successful actions (observation != "Nothing happens") count toward
    milestones.

    Falls back to binary completion when milestone info is unavailable.
    """
    # Milestone-based scoring when action histories are available
    if task_type and actions_after is not None:
        before_actions = actions_before or []
        before_obs = observations_before
        ms_before, total = _count_alfworld_milestones(
            task_type, before_actions, done_before, 0.0,
            observations=before_obs,
        )
        ms_after, total = _count_alfworld_milestones(
            task_type, actions_after, done_after, reward_after,
            observations=observations_after,
        )
        if total == 0:
            return 0.0
        return (ms_after - ms_before) / total

    # Fallback: binary completion
    if not done_before and done_after and reward_after > 0:
        return 1.0
    return 0.0


def compute_progress_webshop(
    reward_before: float,
    reward_after: float,
) -> float:
    """WebShop progress = reward delta.

    WebShop gives reward at buy action based on attribute matching (0-1).
    Progress = reward improvement over the window.
    """
    return reward_after - reward_before


def compute_progress(
    benchmark: str,
    score_before: float,
    score_after: float,
    done_before: bool = False,
    done_after: bool = False,
    reward_after: float = 0.0,
    task_type: str | None = None,
    actions_before: list[str] | None = None,
    actions_after: list[str] | None = None,
    observations_before: list[str] | None = None,
    observations_after: list[str] | None = None,
) -> float:
    """Unified progress computation dispatching to benchmark-specific functions.

    Args:
        benchmark: "scienceworld", "alfworld", or "webshop"
        score_before: score/reward at checkpoint
        score_after: score/reward after N steps
        done_before: was episode done at checkpoint?
        done_after: is episode done after N steps?
        reward_after: final reward (for ALFWorld binary)
        task_type: task family (for ALFWorld milestone scoring)
        actions_before: action history at checkpoint (for ALFWorld milestones)
        actions_after: full action history after rollout (for ALFWorld milestones)
        observations_before: observations at checkpoint (for ALFWorld milestone success check)
        observations_after: full observations after rollout (for ALFWorld milestone success check)
    """
    if benchmark == "scienceworld":
        return compute_progress_scienceworld(score_before, score_after)
    elif benchmark == "alfworld":
        return compute_progress_alfworld(
            done_before, done_after, reward_after,
            task_type=task_type,
            actions_before=actions_before,
            actions_after=actions_after,
            observations_before=observations_before,
            observations_after=observations_after,
        )
    elif benchmark == "webshop":
        return compute_progress_webshop(score_before, score_after)
    else:
        logger.warning(f"Unknown benchmark: {benchmark}, using raw score delta")
        return score_after - score_before


def compute_utility(
    progress_avg: float,
    disruption_avg: float,
    beta: float = 0.3,
) -> float:
    """Compute utility for a (checkpoint, arm) pair.

    utility = progress_avg - β · disruption_avg

    Args:
        progress_avg: mean progress over N replays for this arm
        disruption_avg: mean disruption over N replays (max(0, progress_none - progress_arm))
        beta: disruption penalty weight (default 0.3)
    """
    return progress_avg - beta * disruption_avg

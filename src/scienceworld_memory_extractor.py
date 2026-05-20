"""Extract failure-recovery pairs from ScienceWorld trajectories using LLM.

V2: Directly extracts tiered repair fields (strategy/tactic/action/question)
in a single LLM call, eliminating the separate canonicalize step.
"""

import json
import logging
from src.llm import LLMClient

logger = logging.getLogger(__name__)

EXTRACTOR_SYSTEM_PROMPT = """\
You extract failure-recovery patterns from science experiment task trajectories.

Your goal is precision, not coverage. Only extract a memory when the trajectory
shows a real, reusable method for resolving the failure. If the recovery is not
clear, not reusable, or only looks like a temporary local patch, omit it.

A failure-recovery pattern is when:
1. An action fails (e.g., "No known action matches that input", "You can't do that", action has no effect)
2. The agent takes one or more corrective steps
3. The original goal is achieved either:
   - through a different approach, or
   - by satisfying a missing precondition and retrying the blocked action
4. The trajectory provides enough evidence to state HOW the failure was actually resolved

For each pattern found, output EIGHT fields:
- "failure_step": the numbered trajectory step where the failed action happened
- "failure_action": the action that failed
- "failure_observation": a SHORT single-line summary of the environment response
- "solution_action": the corrective steps that fixed the problem, joined with " → "
- "repair_strategy": a 1-sentence high-level guidance about WHAT the agent should do and WHERE to go
- "repair_tactic": a 1-2 sentence specific plan for the next few steps
- "repair_action": the exact corrective action(s) using valid environment commands
- "question_text": a diagnostic question that hints at the problem WITHOUT revealing the answer

Formatting requirements:
- Output valid JSON only
- Every field value must be a single-line string
- "solution_action" and "repair_action" must be copied exactly from actions that appear later in the trajectory
- Every action listed in "solution_action" must occur strictly AFTER "failure_step"
- Do NOT use actions from before the failure
- Extract the method that solves the failure, not just the next token or local interaction after the failure
- The memory should still make sense as advice for another variation of the same task type
- Prefer stable environment commands or short command sequences over transient UI/menu responses
- Treat "fix precondition, then retry" as a valid recovery pattern
- If the failure says a door is not open, a valid recovery is usually "open door ... → go to ..."
- If the failure says the agent does not have an item, a valid recovery can be "pick up ... → retry"
- If the later trajectory only shows trial-and-error, confusion, or a local patch with no reusable method, omit that pattern
- If the agent never actually recovers after a failure, omit that pattern
- If you are unsure whether a recovery is real and reusable, omit it
- Do NOT copy menus, tabs, or multiline observations verbatim
- Summarize ambiguous/unknown-action observations briefly
- Keep each string concise

If no failure-recovery patterns exist, output [].
Output ONLY the JSON array, nothing else."""

EXTRACTOR_FEW_SHOT = """Here are examples:

<example_trajectory>
> pick up metal pot from stove
No known action matches that input.
> pick up metal pot
You pick up the metal pot from the stove.
> pour water into metal pot
No known action matches that input.
> pour glass cup into metal pot
You pour the glass cup into the metal pot. The metal pot now contains: water.
</example_trajectory>

<example_output>
[{"failure_step": "0", "failure_action": "pick up metal pot from stove", "failure_observation": "No known action matches that input.", "solution_action": "pick up metal pot", "repair_strategy": "Use simplified action syntax — do not specify the location when picking up objects.", "repair_tactic": "Try 'pick up <object>' without specifying 'from <location>'. The environment will pick up the object from its current location.", "repair_action": "pick up metal pot", "question_text": "Does the environment support specifying the source location in pick up commands?"}, {"failure_step": "2", "failure_action": "pour water into metal pot", "failure_observation": "No known action matches that input.", "solution_action": "pour glass cup into metal pot", "repair_strategy": "Pour the container holding the liquid, not the liquid itself — the environment operates on containers.", "repair_tactic": "Identify which container holds the water and use 'pour <container> into <target>' instead of naming the liquid directly.", "repair_action": "pour glass cup into metal pot", "question_text": "Are you trying to pour the liquid directly, or the container that holds it?"}]
</example_output>

<example_trajectory>
> go to kitchen
The door is not open.
> open door to kitchen
The door is now open.
> go to kitchen
You move to the kitchen.
</example_trajectory>

<example_output>
[{"failure_step": "0", "failure_action": "go to kitchen", "failure_observation": "The door is not open.", "solution_action": "open door to kitchen → go to kitchen", "repair_strategy": "Satisfy the missing precondition before retrying movement actions.", "repair_tactic": "If movement is blocked because a door is closed, open that specific door first and then retry going to the destination room.", "repair_action": "open door to kitchen → go to kitchen", "question_text": "What precondition must be satisfied before going to the kitchen?"}]
</example_output>

<example_trajectory>
> look around
This room is called the greenhouse. In it, you see: a flower pot (containing soil and a seed), a jug (containing water), a thermometer.
> pick up jug
You pick up the jug (containing water).
> pour jug into flower pot
You pour the jug into the flower pot. The flower pot now contains: soil, a seed, water.
> wait
Time passes.
> wait
Time passes. The seed starts to grow.
</example_trajectory>

<example_output>
[]
</example_output>"""


def _normalize_text(text: str, max_chars: int = 240) -> str:
    """Collapse multiline observations into a compact single-line form."""
    compact = " ".join(text.replace("\t", " ").split())
    if len(compact) > max_chars:
        return compact[: max_chars - 3] + "..."
    return compact


def _strip_code_fences(response: str) -> str:
    response = response.strip()
    if response.startswith("```"):
        response = response.split("\n", 1)[1]
        response = response.rsplit("```", 1)[0]
    return response.strip()


def _extract_json_array_text(response: str) -> str:
    """Best-effort extraction of the outer JSON array text."""
    start = response.find("[")
    end = response.rfind("]")
    if start >= 0 and end > start:
        return response[start:end + 1]
    return response


def _normalize_action_text(text: str) -> str:
    return " ".join(str(text).split()).strip().lower()


def _normalize_trajectory_steps(trajectory: list[tuple[str, str]] | list[dict]) -> list[dict]:
    steps: list[dict] = []
    for idx, item in enumerate(trajectory):
        if isinstance(item, dict):
            action = str(item.get("action", ""))
            observation = str(item.get("observation", ""))
            step_num = item.get("step", idx)
        else:
            action, observation = item
            step_num = idx
        steps.append(
            {
                "step": int(step_num),
                "action": action,
                "observation": observation,
            }
        )
    return steps


def _find_matching_step(
    trajectory_steps: list[dict],
    action: str,
    observation: str,
    start_idx: int = 0,
) -> int | None:
    norm_action = _normalize_action_text(action)
    norm_obs = _normalize_text(observation, max_chars=500).lower()
    for idx in range(start_idx, len(trajectory_steps)):
        step = trajectory_steps[idx]
        if _normalize_action_text(step["action"]) != norm_action:
            continue
        step_obs = _normalize_text(step["observation"], max_chars=500).lower()
        if norm_obs == step_obs or norm_obs in step_obs or step_obs in norm_obs:
            return idx
    return None


def _normalize_detected_failures(
    trajectory_steps: list[dict],
    detected_failures: list[tuple[str, str]] | list[dict] | None,
) -> list[dict]:
    if not detected_failures:
        return []

    normalized: list[dict] = []
    search_start = 0
    for item in detected_failures:
        if isinstance(item, dict):
            action = str(item.get("action", ""))
            observation = str(item.get("observation", ""))
            failure_type = str(item.get("failure_type", ""))
            step_num = item.get("step")
            trajectory_index = None
            if step_num is not None:
                for idx, step in enumerate(trajectory_steps):
                    if step["step"] == int(step_num):
                        trajectory_index = idx
                        break
            if trajectory_index is None:
                trajectory_index = _find_matching_step(
                    trajectory_steps, action, observation, start_idx=search_start
                )
        else:
            action, observation = item
            failure_type = ""
            trajectory_index = _find_matching_step(
                trajectory_steps, action, observation, start_idx=search_start
            )

        if trajectory_index is None:
            logger.warning(
                "Could not align detected ScienceWorld failure to trajectory: action=%r",
                action,
            )
            continue

        step = trajectory_steps[trajectory_index]
        normalized.append(
            {
                "step": step["step"],
                "trajectory_index": trajectory_index,
                "action": step["action"],
                "observation": step["observation"],
                "failure_type": failure_type,
            }
        )
        search_start = trajectory_index + 1

    return normalized


def _split_action_sequence(text: str) -> list[str]:
    if not text:
        return []
    normalized = text.replace("->", "→")
    return [part.strip() for part in normalized.split("→") if part.strip()]


def _actions_occur_after(
    actions: list[str],
    trajectory_steps: list[dict],
    failure_trajectory_index: int,
) -> bool:
    if not actions:
        return False

    search_start = failure_trajectory_index + 1
    for action in actions:
        norm_action = _normalize_action_text(action)
        found = False
        for idx in range(search_start, len(trajectory_steps)):
            if _normalize_action_text(trajectory_steps[idx]["action"]) == norm_action:
                found = True
                search_start = idx + 1
                break
        if not found:
            return False
    return True


def _validate_recovery(
    item: dict,
    explicit_failures: list[dict],
    trajectory_steps: list[dict],
) -> dict | None:
    failure_step_raw = str(item.get("failure_step", "")).strip().strip("[]")
    failure_action = str(item.get("failure_action", "")).strip()
    if not failure_step_raw or not failure_action:
        return None

    try:
        failure_step = int(failure_step_raw)
    except ValueError:
        return None

    failure_match = None
    for failure in explicit_failures:
        if failure["step"] != failure_step:
            continue
        if _normalize_action_text(failure["action"]) != _normalize_action_text(failure_action):
            continue
        failure_match = failure
        break

    if failure_match is None:
        return None

    solution_actions = _split_action_sequence(str(item.get("solution_action", "")))
    if not _actions_occur_after(solution_actions, trajectory_steps, failure_match["trajectory_index"]):
        return None

    repair_action = str(item.get("repair_action", "")).strip()
    repair_actions = _split_action_sequence(repair_action)
    if repair_action and not _actions_occur_after(
        repair_actions, trajectory_steps, failure_match["trajectory_index"]
    ):
        item["repair_action"] = item["solution_action"]

    item["failure_step"] = str(failure_step)
    item["failure_action"] = failure_match["action"]
    return item


def build_extractor_prompt(
    trajectory: list[tuple[str, str]] | list[dict],
    detected_failures: list[tuple[str, str]] | list[dict] | None = None,
) -> str:
    trajectory_steps = _normalize_trajectory_steps(trajectory)
    traj_str = ""
    for step in trajectory_steps:
        traj_str += (
            f'[{step["step"]}] > {step["action"]}\n'
            f'{_normalize_text(step["observation"], max_chars=320)}\n'
        )

    failure_block = ""
    explicit_failures = _normalize_detected_failures(trajectory_steps, detected_failures)
    if explicit_failures:
        lines = [
            "The following interactions were explicitly detected as failures during the run.",
            "Only extract patterns for these failed steps.",
            "For each output item, failure_step must match one of the failed steps below.",
            "solution_action and repair_action must be exact later actions from the trajectory, after the failed step.",
            "A valid repair may be a precondition fix followed by retrying the blocked action.",
            "If no later action actually repairs a failed step, omit it.",
            "If the trajectory does not reveal a reusable method for fixing the failure, omit it.",
            "Do not force coverage. Fewer high-quality memories are better than weak or local ones.",
        ]
        for failure in explicit_failures:
            extra = f' | type: "{failure["failure_type"]}"' if failure["failure_type"] else ""
            lines.append(
                f'- failed step: "{failure["step"]}" | action: "{failure["action"]}"'
                f' | observation: "{_normalize_text(failure["observation"], max_chars=180)}"{extra}'
            )
        failure_block = "\n\n" + "\n".join(lines)

    return f"""{EXTRACTOR_FEW_SHOT}

Now extract failure-recovery patterns from this trajectory:

<trajectory>
{traj_str.rstrip()}
</trajectory>
{failure_block}

Output ONLY the JSON array."""


# Fields required from extractor output
_REQUIRED_FIELDS = ("failure_action", "failure_observation", "solution_action")
_TIERED_FIELDS = ("repair_strategy", "repair_tactic", "repair_action", "question_text")


def extract_failure_recoveries(
    extractor_llm: LLMClient,
    trajectory: list[tuple[str, str]] | list[dict],
    detected_failures: list[tuple[str, str]] | list[dict] | None = None,
) -> list[dict]:
    """Extract failure-recovery pairs from a ScienceWorld episode trajectory.

    Returns list of dicts with 7 fields:
        failure_action, failure_observation, solution_action,
        repair_strategy, repair_tactic, repair_action, question_text
    """
    trajectory_steps = _normalize_trajectory_steps(trajectory)
    if not trajectory_steps:
        return []

    explicit_failures = _normalize_detected_failures(trajectory_steps, detected_failures)
    if not explicit_failures:
        return []

    # Check if there are any failures worth extracting
    failure_indicators = [
        "no known action",
        "unknown action",
        "you can't",
        "nothing happens",
        "that doesn't seem to work",
        "i'm not sure what you mean",
        "ambiguous request",
        "please enter the number for the action you intended",
        "the door is not open",
    ]
    has_failure = any(
        any(ind in obs.lower() for ind in failure_indicators)
        for _, obs in [(step["action"], step["observation"]) for step in trajectory_steps]
    )
    if not has_failure and not explicit_failures:
        return []

    prompt = build_extractor_prompt(trajectory_steps, detected_failures=explicit_failures)
    try:
        response = extractor_llm.complete_text(
            prompt,
            label="extractor",
            system=EXTRACTOR_SYSTEM_PROMPT,
            max_tokens=1024,
        )

        response = _extract_json_array_text(_strip_code_fences(response))

        results = json.loads(response)
        if not isinstance(results, list):
            logger.warning(f"Extractor returned non-list: {type(results)}")
            return []

        valid = []
        for item in results:
            if all(k in item for k in _REQUIRED_FIELDS):
                item = _validate_recovery(item, explicit_failures, trajectory_steps)
                if item is None:
                    logger.info("Dropped invalid ScienceWorld recovery that was not observed post-failure")
                    continue
                # Ensure tiered fields exist with fallbacks
                for tf in _TIERED_FIELDS:
                    if tf not in item or not item[tf]:
                        if tf == "question_text":
                            item[tf] = f"What went wrong when you tried '{item['failure_action']}'?"
                        elif tf == "repair_action":
                            item[tf] = item["solution_action"]
                        else:
                            item[tf] = ""
                valid.append(item)

        logger.info(f"Extractor found {len(valid)} failure-recovery patterns (tiered)")
        return valid

    except json.JSONDecodeError as e:
        truncated = ""
        if "]" not in response:
            truncated = " (response appears truncated)"
        logger.warning(f"Extractor JSON parse error: {e}{truncated}, response: {response[:200]}")
        return []
    except Exception as e:
        logger.warning(f"Extractor call failed: {e}")
        return []

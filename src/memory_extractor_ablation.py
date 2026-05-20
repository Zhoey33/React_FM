"""Alternative memory extraction modes for ablation study.

Three formats:
1. failure_recovery (default) — our (failure_action, failure_obs, solution_action) pairs
2. success_trajectory — store key steps from successful episodes
3. reflexion_reflection — store free-text reflections (Reflexion-style)
"""

import json
import logging
from src.llm import LLMClient

logger = logging.getLogger(__name__)

# ============================================================
# Mode 2: Success Trajectory Extraction
# ============================================================

SUCCESS_SYSTEM_PROMPT = """You extract key successful action patterns from household robot task trajectories.

For each successful action sequence, output the key steps that led to task completion.

Output a JSON array. Each element has:
- "failure_action": (use the task goal as a summary, e.g., "clean cloth")
- "failure_observation": "success_pattern"
- "solution_action": the key successful steps joined with " → "

If the trajectory failed (task not completed), output [].
Output ONLY the JSON array, nothing else."""

SUCCESS_FEW_SHOT = """Here are examples:

<example_trajectory>
> go to cabinet 1
You arrive at cabinet 1. The cabinet 1 is closed.
> open cabinet 1
You open the cabinet 1. In it, you see a cloth 1.
> take cloth 1 from cabinet 1
You pick up the cloth 1 from the cabinet 1.
> go to sinkbasin 1
You arrive at sinkbasin 1.
> clean cloth 1 with sinkbasin 1
You clean the cloth 1 using the sinkbasin 1.
> go to toilet 1
You arrive at toilet 1.
> put cloth 1 in/on toilet 1
You put the cloth 1 in/on the toilet 1.
</example_trajectory>

<example_output>
[{"failure_action": "clean cloth and place on toilet", "failure_observation": "success_pattern", "solution_action": "open cabinet 1 → take cloth 1 from cabinet 1 → go to sinkbasin 1 → clean cloth 1 → go to toilet 1 → put cloth 1 in/on toilet 1"}]
</example_output>

<example_trajectory>
> go to countertop 1
You arrive at countertop 1. On the countertop 1, you see a bread 1.
> take bread 1 from countertop 1
You pick up the bread 1.
> go to fridge 1
You arrive at fridge 1.
> cool bread 1 with fridge 1
You cool the bread 1 using the fridge 1.
> go to countertop 1
You arrive at countertop 1.
> put bread 1 in/on countertop 1
You put the bread 1 in/on the countertop 1.
</example_trajectory>

<example_output>
[{"failure_action": "cool bread and place on countertop", "failure_observation": "success_pattern", "solution_action": "take bread 1 → go to fridge 1 → cool bread 1 → go to countertop 1 → put bread 1"}]
</example_output>"""


# ============================================================
# Mode 3: Reflexion-style Free-text Reflection
# ============================================================

REFLEXION_SYSTEM_PROMPT = """You generate a brief self-reflection about a household robot task trajectory, summarizing what went wrong and what should be done differently.

Output a JSON array with exactly one element:
- "failure_action": (a brief summary of the main mistake, e.g., "tried to take from closed container")
- "failure_observation": "reflection"
- "solution_action": a free-text reflection sentence about what to do differently

If the trajectory was perfect with no issues, output [].
Output ONLY the JSON array, nothing else."""

REFLEXION_FEW_SHOT = """Here are examples:

<example_trajectory>
> go to fridge 1
You arrive at fridge 1.
> take apple 1 from fridge 1
Nothing happens
> open fridge 1
You open the fridge 1. In it, you see a apple 1.
> take apple 1 from fridge 1
You pick up the apple 1 from the fridge 1.
> go to countertop 1
You arrive at countertop 1.
> put apple 1 in/on countertop 1
You put the apple 1 in/on the countertop 1.
</example_trajectory>

<example_output>
[{"failure_action": "took from closed fridge", "failure_observation": "reflection", "solution_action": "Always check if a container (fridge, cabinet, drawer) is open before trying to take items from it. Open it first, then take the item."}]
</example_output>

<example_trajectory>
> go to countertop 1
You arrive at countertop 1. On the countertop 1, you see a bread 1.
> take bread 1 from countertop 1
You pick up the bread 1.
> go to fridge 1
You arrive at fridge 1.
> cool bread 1 with fridge 1
You cool the bread 1 using the fridge 1.
> go to countertop 1
You arrive at countertop 1.
> put bread 1 in/on countertop 1
You put the bread 1 in/on the countertop 1.
</example_trajectory>

<example_output>
[]
</example_output>"""


def _build_prompt(trajectory: list[tuple[str, str]], few_shot: str) -> str:
    traj_str = ""
    for action, obs in trajectory:
        traj_str += f"> {action}\n{obs}\n"
    return f"""{few_shot}

Now analyze this trajectory:

<trajectory>
{traj_str.rstrip()}
</trajectory>

Output ONLY the JSON array."""


def _call_extractor(extractor_llm: LLMClient, prompt: str, system: str) -> list[dict]:
    try:
        response = extractor_llm.complete_text(prompt, label="extractor", system=system)
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1]
            response = response.rsplit("```", 1)[0]
            response = response.strip()

        results = json.loads(response)
        if not isinstance(results, list):
            return []

        valid = []
        for item in results:
            if all(k in item for k in ("failure_action", "failure_observation", "solution_action")):
                valid.append(item)
        return valid

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Extractor call failed: {e}")
        return []


def extract_success_trajectories(
    extractor_llm: LLMClient,
    trajectory: list[tuple[str, str]],
    success: bool,
) -> list[dict]:
    """Extract key steps from successful trajectories only."""
    if not trajectory or not success:
        return []

    prompt = _build_prompt(trajectory, SUCCESS_FEW_SHOT)
    results = _call_extractor(extractor_llm, prompt, SUCCESS_SYSTEM_PROMPT)
    logger.info(f"Success extractor found {len(results)} patterns")
    return results


def extract_reflexion_reflections(
    extractor_llm: LLMClient,
    trajectory: list[tuple[str, str]],
) -> list[dict]:
    """Extract Reflexion-style free-text reflections."""
    if not trajectory:
        return []

    # Only extract if there were failures
    has_failure = any("nothing happens" in obs.lower() for _, obs in trajectory)
    if not has_failure:
        return []

    prompt = _build_prompt(trajectory, REFLEXION_FEW_SHOT)
    results = _call_extractor(extractor_llm, prompt, REFLEXION_SYSTEM_PROMPT)
    logger.info(f"Reflexion extractor found {len(results)} reflections")
    return results

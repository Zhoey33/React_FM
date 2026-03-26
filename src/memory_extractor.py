"""Extract failure-recovery pairs from episode trajectories using LLM."""

import json
import logging
from src.llm import LLMClient

logger = logging.getLogger(__name__)

EXTRACTOR_SYSTEM_PROMPT = """You extract failure-recovery patterns from household robot task trajectories.

A failure-recovery pattern is when:
1. An action fails (typically "Nothing happens")
2. The agent takes one or more corrective steps
3. The original failed action (or a similar one) succeeds

For each pattern found, output the failed action, what happened, and the fix steps.

Output a JSON array. Each element has:
- "failure_action": the action that failed
- "failure_observation": the environment response (e.g. "Nothing happens")
- "solution_action": the corrective steps that fixed the problem, joined with " → "

If no failure-recovery patterns exist, output [].
Output ONLY the JSON array, nothing else."""

EXTRACTOR_FEW_SHOT = """Here are examples:

<example_trajectory>
> go to fridge 1
You arrive at fridge 1.
> take apple 1 from fridge 1
Nothing happens
> open fridge 1
You open the fridge 1. The fridge 1 is open. In it, you see a apple 1, a egg 1.
> take apple 1 from fridge 1
You pick up the apple 1 from the fridge 1.
> go to countertop 1
You arrive at countertop 1.
> put apple 1 in/on countertop 1
You put the apple 1 in/on the countertop 1.
</example_trajectory>

<example_output>
[{"failure_action": "take apple 1 from fridge 1", "failure_observation": "Nothing happens", "solution_action": "open fridge 1 → take apple 1 from fridge 1"}]
</example_output>

<example_trajectory>
> go to cabinet 1
You arrive at cabinet 1. The cabinet 1 is closed.
> take cloth 1 from cabinet 1
Nothing happens
> open cabinet 1
You open the cabinet 1. The cabinet 1 is open. In it, you see a cloth 1.
> take cloth 1 from cabinet 1
You pick up the cloth 1 from the cabinet 1.
> go to sinkbasin 1
You arrive at sinkbasin 1.
> clean cloth 1 with sinkbasin 1
You clean the cloth 1 using the sinkbasin 1.
> go to toilet 1
You arrive at toilet 1.
> put cloth 1 in/on toilet 1
Nothing happens
> examine toilet 1
On the toilet 1, you see a soapbottle 1.
> put cloth 1 in/on toilet 1
You put the cloth 1 in/on the toilet 1.
</example_trajectory>

<example_output>
[{"failure_action": "take cloth 1 from cabinet 1", "failure_observation": "Nothing happens", "solution_action": "open cabinet 1 → take cloth 1 from cabinet 1"}, {"failure_action": "put cloth 1 in/on toilet 1", "failure_observation": "Nothing happens", "solution_action": "examine toilet 1 → put cloth 1 in/on toilet 1"}]
</example_output>

<example_trajectory>
> go to countertop 1
You arrive at countertop 1. On the countertop 1, you see a bread 1, a mug 1.
> take bread 1 from countertop 1
You pick up the bread 1 from the countertop 1.
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


def build_extractor_prompt(trajectory: list[tuple[str, str]]) -> str:
    """Build the user prompt for the extractor LLM.

    Args:
        trajectory: List of (action, observation) tuples from the episode
    """
    traj_str = ""
    for action, obs in trajectory:
        traj_str += f"> {action}\n{obs}\n"

    return f"""{EXTRACTOR_FEW_SHOT}

Now extract failure-recovery patterns from this trajectory:

<trajectory>
{traj_str.rstrip()}
</trajectory>

Output ONLY the JSON array."""


def extract_failure_recoveries(
    extractor_llm: LLMClient,
    trajectory: list[tuple[str, str]],
) -> list[dict]:
    """Extract failure-recovery pairs from an episode trajectory.

    Args:
        extractor_llm: LLM client for extraction (e.g. DeepSeek-V3.2)
        trajectory: List of (action, observation) tuples

    Returns:
        List of dicts with keys: failure_action, failure_observation, solution_action
    """
    if not trajectory:
        return []

    # Check if there are any failures worth extracting
    has_failure = any("nothing happens" in obs.lower() for _, obs in trajectory)
    if not has_failure:
        return []

    prompt = build_extractor_prompt(trajectory)
    try:
        response = extractor_llm.complete_text(
            prompt,
            label="extractor",
            system=EXTRACTOR_SYSTEM_PROMPT,
        )

        # Parse JSON from response
        response = response.strip()
        # Handle markdown code blocks
        if response.startswith("```"):
            response = response.split("\n", 1)[1]
            response = response.rsplit("```", 1)[0]
            response = response.strip()

        results = json.loads(response)
        if not isinstance(results, list):
            logger.warning(f"Extractor returned non-list: {type(results)}")
            return []

        # Validate entries
        valid = []
        for item in results:
            if all(k in item for k in ("failure_action", "failure_observation", "solution_action")):
                valid.append(item)
            else:
                logger.warning(f"Extractor returned incomplete entry: {item}")

        logger.info(f"Extractor found {len(valid)} failure-recovery patterns")
        return valid

    except json.JSONDecodeError as e:
        logger.warning(f"Extractor JSON parse error: {e}, response: {response[:200]}")
        return []
    except Exception as e:
        logger.warning(f"Extractor call failed: {e}")
        return []

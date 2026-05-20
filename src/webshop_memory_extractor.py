"""Extract failure-recovery pairs from WebShop trajectories using LLM."""

import json
import logging
from src.llm import LLMClient

logger = logging.getLogger(__name__)

EXTRACTOR_SYSTEM_PROMPT = """You extract failure-recovery patterns from online shopping task trajectories.

A failure-recovery pattern is when:
1. An action fails (e.g., search returns no results, wrong product selected, wrong option chosen)
2. The agent takes corrective steps (new search, back to search, different product)
3. The agent finds a better approach

For each pattern found, output the failed action, what happened, and the fix steps.

Output a JSON array. Each element has:
- "failure_action": the action that failed
- "failure_observation": what went wrong
- "solution_action": the corrective steps, joined with " → "

If no failure-recovery patterns exist, output [].
Output ONLY the JSON array, nothing else."""

EXTRACTOR_FEW_SHOT = """Here are examples:

<example_trajectory>
> search[red large shirt]
Page 1 (Total results: 0)
> search[red shirt large]
Page 1 (Total results: 8) [SEP] B001234 [SEP] Men's Red Cotton Shirt $29.99
> click[B001234]
Men's Red Cotton Shirt [SEP] Color: Red, Blue [SEP] Size: S, M, L, XL
> click[XL]
You have selected XL.
> click[Back to Search]
> click[B001234]
Men's Red Cotton Shirt [SEP] Color: Red, Blue [SEP] Size: S, M, L, XL
> click[L]
You have selected L.
> click[Buy Now]
Purchase successful!
</example_trajectory>

<example_output>
[{"failure_action": "search[red large shirt]", "failure_observation": "Total results: 0", "solution_action": "search[red shirt large]"}, {"failure_action": "click[XL]", "failure_observation": "Selected wrong size XL instead of L", "solution_action": "click[Back to Search] → click[B001234] → click[L]"}]
</example_output>

<example_trajectory>
> search[blue running shoes]
Page 1 (Total results: 12) [SEP] B005678 [SEP] Nike Blue Running Shoes $89.99
> click[B005678]
Nike Blue Running Shoes [SEP] Color: Blue, Black [SEP] Size: 8, 9, 10, 11
> click[Blue]
You have selected Blue.
> click[10]
You have selected 10.
> click[Buy Now]
Purchase successful!
</example_trajectory>

<example_output>
[]
</example_output>"""


def build_extractor_prompt(trajectory: list[tuple[str, str]]) -> str:
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
    """Extract failure-recovery pairs from a WebShop episode trajectory."""
    if not trajectory:
        return []

    # Check if there are potential failures
    failure_indicators = [
        "total results: 0",
        "no results",
        "back to search",
        "invalid",
    ]
    has_failure = any(
        any(ind in obs.lower() for ind in failure_indicators)
        for _, obs in trajectory
    )
    # Also check for repeated actions (indicates wrong choices)
    actions = [a for a, _ in trajectory]
    has_repetition = len(actions) != len(set(actions))

    if not has_failure and not has_repetition:
        return []

    prompt = build_extractor_prompt(trajectory)
    try:
        response = extractor_llm.complete_text(
            prompt, label="extractor", system=EXTRACTOR_SYSTEM_PROMPT,
        )
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

        logger.info(f"Extractor found {len(valid)} failure-recovery patterns")
        return valid

    except json.JSONDecodeError as e:
        logger.warning(f"Extractor JSON parse error: {e}")
        return []
    except Exception as e:
        logger.warning(f"Extractor call failed: {e}")
        return []

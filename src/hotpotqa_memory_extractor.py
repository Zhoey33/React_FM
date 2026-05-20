"""Extract failure-recovery pairs from HotPotQA trajectories using LLM."""

import json
import logging
from src.llm import LLMClient

logger = logging.getLogger(__name__)

EXTRACTOR_SYSTEM_PROMPT = """You extract failure-recovery patterns from question-answering task trajectories.

A failure-recovery pattern is when:
1. A search or lookup fails (entity not found, irrelevant results, wrong page)
2. The agent corrects by searching a different entity or using a different keyword
3. The agent finds the needed information

For each pattern found, output the failed action, what happened, and the fix steps.

Output a JSON array. Each element has:
- "failure_action": the action that failed
- "failure_observation": what went wrong
- "solution_action": the corrective steps, joined with " → "

If no failure-recovery patterns exist, output [].
Output ONLY the JSON array, nothing else."""

EXTRACTOR_FEW_SHOT = """Here are examples:

<example_trajectory>
Thought: I need to find who directed The Matrix.
Action: Search[The Matrix movie director]
Observation: Could not find [The Matrix movie director]. Similar: ['The Matrix', 'The Matrix (franchise)']
Thought: Let me search for The Matrix directly.
Action: Search[The Matrix]
Observation: The Matrix is a 1999 science fiction action film written and directed by the Wachowskis.
Thought: The Matrix was directed by the Wachowskis.
Action: Finish[the Wachowskis]
</example_trajectory>

<example_output>
[{"failure_action": "Search[The Matrix movie director]", "failure_observation": "Could not find the entity", "solution_action": "Search[The Matrix]"}]
</example_output>

<example_trajectory>
Thought: I need to find the capital of France.
Action: Search[France]
Observation: France, officially the French Republic, is a country located primarily in Western Europe. Its capital is Paris.
Thought: The capital of France is Paris.
Action: Finish[Paris]
</example_trajectory>

<example_output>
[]
</example_output>"""


def build_extractor_prompt(trajectory: list[tuple[str, str, str]]) -> str:
    traj_str = ""
    for thought, action, obs in trajectory:
        if thought:
            traj_str += f"Thought: {thought}\n"
        traj_str += f"Action: {action}\nObservation: {obs}\n"

    return f"""{EXTRACTOR_FEW_SHOT}

Now extract failure-recovery patterns from this trajectory:

<trajectory>
{traj_str.rstrip()}
</trajectory>

Output ONLY the JSON array."""


def extract_failure_recoveries(
    extractor_llm: LLMClient,
    trajectory: list[tuple[str, str, str]],
) -> list[dict]:
    """Extract failure-recovery pairs from a HotPotQA episode trajectory."""
    if not trajectory:
        return []

    # Check if there are potential failures
    failure_indicators = [
        "could not find",
        "no more results",
        "invalid action",
        "search error",
    ]
    has_failure = any(
        any(ind in obs.lower() for ind in failure_indicators)
        for _, _, obs in trajectory
    )
    # Also check for repeated actions
    actions = [a for _, a, _ in trajectory]
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

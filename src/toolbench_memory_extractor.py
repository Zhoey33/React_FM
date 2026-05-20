"""Extract failure-recovery pairs from ToolBench trajectories using LLM."""

import json
import logging
from src.llm import LLMClient

logger = logging.getLogger(__name__)

EXTRACTOR_SYSTEM_PROMPT = """You extract failure-recovery patterns from tool-use task trajectories.

A failure-recovery pattern is when:
1. An API call fails (wrong parameters, wrong API, error response)
2. The agent corrects by calling a different API or fixing parameters
3. The agent gets useful results

For each pattern found, output the failed action, what happened, and the fix steps.

Output a JSON array. Each element has:
- "failure_action": the action that failed
- "failure_observation": what went wrong
- "solution_action": the corrective steps, joined with " → "

If no failure-recovery patterns exist, output [].
Output ONLY the JSON array, nothing else."""

EXTRACTOR_FEW_SHOT = """Here are examples:

<example_trajectory>
Thought: I need to search for weather data.
Action: get_weather({"city_name": "NYC"})
Observation: API Error: got an unexpected keyword argument 'city_name'
Thought: The parameter name is wrong. Let me try 'city'.
Action: get_weather({"city": "New York"})
Observation: {"temperature": 72, "condition": "Sunny"}
Thought: Got the weather data.
Action: Finish({"return_type": "give_answer", "final_answer": "72F and Sunny in New York"})
</example_trajectory>

<example_output>
[{"failure_action": "get_weather({\"city_name\": \"NYC\"})", "failure_observation": "unexpected keyword argument 'city_name'", "solution_action": "get_weather({\"city\": \"New York\"})"}]
</example_output>

<example_trajectory>
Thought: I need to find flight prices.
Action: search_flights({"from": "LAX", "to": "JFK", "date": "2024-01-15"})
Observation: {"flights": [{"price": 299, "airline": "Delta"}, {"price": 350, "airline": "United"}]}
Thought: Found flights. The cheapest is $299 on Delta.
Action: Finish({"return_type": "give_answer", "final_answer": "The cheapest flight is $299 on Delta."})
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
    if not trajectory:
        return []

    failure_indicators = [
        "api error", "error", "unknown function", "invalid action",
        "no cached response", "unexpected keyword", "failed",
    ]
    has_failure = any(
        any(ind in obs.lower() for ind in failure_indicators)
        for _, _, obs in trajectory
    )
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

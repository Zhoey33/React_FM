"""ExpeL-style insight extraction from trajectory pairs.

ExpeL (Zhao et al., 2024) extracts cross-task insights by comparing
success and failure trajectories. The LLM identifies abstract rules
that explain why one trajectory succeeded and another failed.

Supports multiple domains: alfworld, webshop, scienceworld.
"""

import json
import logging
from src.llm import LLMClient

logger = logging.getLogger(__name__)

# --- Domain-specific system prompts ---

INSIGHT_SYSTEM_PROMPTS = {
    "alfworld": (
        "You are an expert at analyzing household robot task trajectories.\n\n"
        "Given a FAILED trajectory and a SUCCESSFUL trajectory for the same type of task,\n"
        "extract general rules/insights that explain why one failed and the other succeeded.\n\n"
        "Each insight should be:\n"
        "- Abstract and generalizable (not specific to exact item names)\n"
        "- Actionable (tells the agent what to do or avoid)\n"
        "- Concise (one sentence)\n\n"
        "Output a JSON array. Each element has:\n"
        '- "rule": the abstract insight/rule\n'
        '- "task_type": the type of task\n\n'
        "If no useful insights can be extracted, output [].\n"
        "Output ONLY the JSON array, nothing else."
    ),
    "webshop": (
        "You are an expert at analyzing online shopping task trajectories.\n\n"
        "Given a FAILED trajectory and a SUCCESSFUL trajectory for shopping tasks,\n"
        "extract general rules/insights that explain why one failed and the other succeeded.\n\n"
        "Each insight should be:\n"
        "- Abstract and generalizable (not specific to exact product names)\n"
        "- Actionable (tells the agent what to do or avoid when searching/selecting products)\n"
        "- Concise (one sentence)\n\n"
        "Output a JSON array. Each element has:\n"
        '- "rule": the abstract insight/rule\n'
        '- "task_type": "shopping"\n\n'
        "If no useful insights can be extracted, output [].\n"
        "Output ONLY the JSON array, nothing else."
    ),
    "scienceworld": (
        "You are an expert at analyzing science experiment task trajectories.\n\n"
        "Given a FAILED trajectory and a SUCCESSFUL trajectory for the same type of task,\n"
        "extract general rules/insights that explain why one failed and the other succeeded.\n\n"
        "Each insight should be:\n"
        "- Abstract and generalizable (not specific to exact object names)\n"
        "- Actionable (tells the agent what to do or avoid in experiments)\n"
        "- Concise (one sentence)\n\n"
        "Output a JSON array. Each element has:\n"
        '- "rule": the abstract insight/rule\n'
        '- "task_type": the type of task\n\n'
        "If no useful insights can be extracted, output [].\n"
        "Output ONLY the JSON array, nothing else."
    ),
    "hotpotqa": (
        "You are an expert at analyzing multi-hop question-answering trajectories.\n\n"
        "Given a FAILED trajectory and a SUCCESSFUL trajectory for QA tasks using Wikipedia,\n"
        "extract general rules/insights that explain why one failed and the other succeeded.\n\n"
        "Each insight should be:\n"
        "- Abstract and generalizable (not specific to exact entities or answers)\n"
        "- Actionable (tells the agent better search/lookup strategies)\n"
        "- Concise (one sentence)\n\n"
        "Output a JSON array. Each element has:\n"
        '- "rule": the abstract insight/rule\n'
        '- "task_type": "qa"\n\n'
        "If no useful insights can be extracted, output [].\n"
        "Output ONLY the JSON array, nothing else."
    ),
    "toolbench": (
        "You are an expert at analyzing API tool-use task trajectories.\n\n"
        "Given a FAILED trajectory and a SUCCESSFUL trajectory for tasks requiring API calls,\n"
        "extract general rules/insights that explain why one failed and the other succeeded.\n\n"
        "Each insight should be:\n"
        "- Abstract and generalizable (not specific to exact API names or parameters)\n"
        "- Actionable (tells the agent better API calling strategies)\n"
        "- Concise (one sentence)\n\n"
        "Output a JSON array. Each element has:\n"
        '- "rule": the abstract insight/rule\n'
        '- "task_type": "tool_use"\n\n'
        "If no useful insights can be extracted, output [].\n"
        "Output ONLY the JSON array, nothing else."
    ),
}

INSIGHT_FROM_FAILURE_PROMPTS = {
    "alfworld": (
        "You are an expert at analyzing task trajectories.\n\n"
        "Given a FAILED trajectory, extract general rules/insights about what went wrong "
        "and what should be done differently in future attempts.\n\n"
        "Each insight should be:\n"
        "- Abstract and generalizable (not specific to exact item names)\n"
        "- Actionable (tells the agent what to do or avoid)\n"
        "- Concise (one sentence)\n\n"
        "Output a JSON array. Each element has:\n"
        '- "rule": the abstract insight/rule\n'
        '- "task_type": the type of task\n\n'
        "If no useful insights can be extracted, output [].\n"
        "Output ONLY the JSON array, nothing else."
    ),
    "webshop": (
        "You are an expert at analyzing online shopping task trajectories.\n\n"
        "Given a FAILED shopping trajectory, extract general rules/insights about what went wrong "
        "and what should be done differently in future shopping attempts.\n\n"
        "Each insight should be:\n"
        "- Abstract and generalizable (not specific to exact product names)\n"
        "- Actionable (tells the agent what to do or avoid)\n"
        "- Concise (one sentence)\n\n"
        "Output a JSON array. Each element has:\n"
        '- "rule": the abstract insight/rule\n'
        '- "task_type": "shopping"\n\n'
        "If no useful insights can be extracted, output [].\n"
        "Output ONLY the JSON array, nothing else."
    ),
    "scienceworld": (
        "You are an expert at analyzing science experiment task trajectories.\n\n"
        "Given a FAILED experiment trajectory, extract general rules/insights about what went wrong "
        "and what should be done differently in future experiments.\n\n"
        "Each insight should be:\n"
        "- Abstract and generalizable (not specific to exact object names)\n"
        "- Actionable (tells the agent what to do or avoid)\n"
        "- Concise (one sentence)\n\n"
        "Output a JSON array. Each element has:\n"
        '- "rule": the abstract insight/rule\n'
        '- "task_type": the type of task\n\n'
        "If no useful insights can be extracted, output [].\n"
        "Output ONLY the JSON array, nothing else."
    ),
    "hotpotqa": (
        "You are an expert at analyzing multi-hop question-answering trajectories.\n\n"
        "Given a FAILED QA trajectory, extract general rules/insights about what went wrong "
        "and what should be done differently in future QA attempts.\n\n"
        "Each insight should be:\n"
        "- Abstract and generalizable (not specific to exact entities)\n"
        "- Actionable (tells the agent better search/lookup strategies)\n"
        "- Concise (one sentence)\n\n"
        "Output a JSON array. Each element has:\n"
        '- "rule": the abstract insight/rule\n'
        '- "task_type": "qa"\n\n'
        "If no useful insights can be extracted, output [].\n"
        "Output ONLY the JSON array, nothing else."
    ),
    "toolbench": (
        "You are an expert at analyzing API tool-use task trajectories.\n\n"
        "Given a FAILED tool-use trajectory, extract general rules/insights about what went wrong "
        "and what should be done differently in future API calling attempts.\n\n"
        "Each insight should be:\n"
        "- Abstract and generalizable (not specific to exact API names)\n"
        "- Actionable (tells the agent better API calling strategies)\n"
        "- Concise (one sentence)\n\n"
        "Output a JSON array. Each element has:\n"
        '- "rule": the abstract insight/rule\n'
        '- "task_type": "tool_use"\n\n'
        "If no useful insights can be extracted, output [].\n"
        "Output ONLY the JSON array, nothing else."
    ),
}

# --- Domain-specific few-shot examples ---

INSIGHT_FEW_SHOTS = {
    "alfworld": """Example:

<failed_trajectory>
Task: put a cool tomato in microwave.
> go to fridge 1
You arrive at fridge 1.
> take tomato 1 from fridge 1
Nothing happens.
> open fridge 1
You open the fridge 1. In it, you see a tomato 1.
> take tomato 1 from fridge 1
You pick up the tomato 1.
> go to microwave 1
You arrive at microwave 1.
> put tomato 1 in/on microwave 1
Nothing happens.
[FAILED - repeated action loop]
</failed_trajectory>

<successful_trajectory>
Task: put a cool apple in countertop.
> go to fridge 1
You arrive at fridge 1.
> open fridge 1
You open the fridge 1. In it, you see an apple 1.
> take apple 1 from fridge 1
You pick up the apple 1.
> cool apple 1 with fridge 1
You cool the apple 1 using the fridge 1.
> go to countertop 1
You arrive at countertop 1.
> put apple 1 in/on countertop 1
You put the apple 1 in/on the countertop 1.
[SUCCESS]
</successful_trajectory>

<output>
[{"rule": "Always open a container (fridge, cabinet, drawer) before trying to take items from it.", "task_type": "cool"}, {"rule": "When putting an item into a receptacle, first open the receptacle if it is closed.", "task_type": "cool"}]
</output>""",

    "webshop": """Example:

<failed_trajectory>
Task: Find me a blue cotton t-shirt, size large, price lower than $30.
> search[blue cotton t-shirt large under $30]
Page 1 (Total results: 0)
> search[blue t-shirt]
Page 1 (Total results: 8) [SEP] B001234 [SEP] Men's Blue Cotton Tee $24.99
> click[B001234]
Men's Blue Cotton Tee [SEP] Price: $24.99 [SEP] Color: Blue, Red [SEP] Size: Small, Medium
> click[Buy Now]
Purchase completed. Reward: 0.3
[FAILED - wrong size, did not select options]
</failed_trajectory>

<successful_trajectory>
Task: Find me a red wool sweater, size medium, price lower than $50.
> search[red wool sweater]
Page 1 (Total results: 12) [SEP] B005678 [SEP] Classic Red Wool Sweater $39.99
> click[B005678]
Classic Red Wool Sweater [SEP] Price: $39.99 [SEP] Color: Red, Navy [SEP] Size: Small, Medium, Large
> click[Red]
You have selected Red.
> click[Medium]
You have selected Medium.
> click[Buy Now]
Purchase successful! Reward: 1.0
[SUCCESS]
</successful_trajectory>

<output>
[{"rule": "Always select all required options (color, size) before clicking Buy Now.", "task_type": "shopping"}, {"rule": "Use simple search keywords instead of overly specific queries to get more results.", "task_type": "shopping"}]
</output>""",

    "scienceworld": """Example:

<failed_trajectory>
Task: Your task is to boil water.
> look around
This room is called the kitchen. You see: a stove, a sink, a table. On the table: a glass cup (containing water), a metal pot.
> activate stove
You activate the stove.
> wait
Time passes. Nothing happens.
> wait
Time passes. Nothing happens.
[FAILED - water not on stove]
</failed_trajectory>

<successful_trajectory>
Task: Your task is to boil water.
> look around
This room is called the kitchen. You see: a stove, a sink, a table. On the table: a glass cup (containing water), a metal pot.
> pick up metal pot
You pick up the metal pot.
> move metal pot to stove
You move the metal pot to the stove.
> pick up glass cup
You pick up the glass cup (containing water).
> pour glass cup into metal pot
You pour the water into the metal pot.
> focus on water in metal pot
You focus on the water.
> activate stove
You activate the stove. The water starts to heat up.
> wait
The water is now boiling!
[SUCCESS]
</successful_trajectory>

<output>
[{"rule": "Place a container on the heat source before activating it - items must be physically on the stove to be heated.", "task_type": "boil"}, {"rule": "Transfer liquids into a heat-safe container (metal pot) before heating.", "task_type": "boil"}]
</output>""",

    "hotpotqa": """Example:

<failed_trajectory>
Task: Question: What nationality were both combatants in the Battle of Isandlwana?
Action: Search[Battle of Isandlwana combatants nationality]
Observation: Could not find [Battle of Isandlwana combatants nationality]. Similar: ['Battle of Isandlwana']
Action: Search[Battle of Isandlwana]
Observation: The Battle of Isandlwana on 22 January 1879 was the first major encounter in the Anglo-Zulu War between the British Empire and the Zulu Kingdom.
Action: Finish[British and Zulu]
[FAILED - answer should be specific nationalities]
</failed_trajectory>

<successful_trajectory>
Task: Question: What war was fought between the British and the Boers?
Action: Search[British Boer war]
Observation: Could not find [British Boer war]. Similar: ['Boer Wars', 'First Boer War']
Action: Search[Boer Wars]
Observation: The Boer Wars were two wars fought between the British Empire and the two Boer states, the South African Republic and the Orange Free State.
Action: Lookup[fought]
Observation: (Result 1/3) The Boer Wars were two wars fought between the British Empire and the two Boer states.
Action: Finish[Boer Wars]
[SUCCESS]
</successful_trajectory>

<output>
[{"rule": "Search for the exact entity name rather than a descriptive phrase - Wikipedia titles are proper nouns.", "task_type": "qa"}, {"rule": "Use Lookup to find specific details on a page rather than relying only on the initial search summary.", "task_type": "qa"}]
</output>""",

    "toolbench": """Example:

<failed_trajectory>
Task: Get the current weather in Tokyo and convert the temperature to Fahrenheit.
> get_weather_for_weather_api({"city": "Tokyo", "units": "fahrenheit"})
API Error: got an unexpected keyword argument 'units'
> get_weather_for_weather_api({"city": "Tokyo"})
{"temperature_celsius": 22, "condition": "Cloudy"}
> convert_temperature_for_unit_converter({"value": "22", "from": "celsius", "to": "fahrenheit"})
API call failed: no cached response available.
> Finish({"return_type": "give_up_and_restart", "final_answer": ""})
[FAILED]
</failed_trajectory>

<successful_trajectory>
Task: Get the current weather in London and convert the temperature to Fahrenheit.
> get_weather_for_weather_api({"city": "London"})
{"temperature_celsius": 15, "condition": "Rainy"}
> Finish({"return_type": "give_answer", "final_answer": "The weather in London is 15°C (59°F) and Rainy."})
[SUCCESS]
</successful_trajectory>

<output>
[{"rule": "Check API parameter names carefully - do not add parameters that are not in the API schema.", "task_type": "tool_use"}, {"rule": "When a conversion API is unavailable, perform simple calculations directly in the final answer instead of giving up.", "task_type": "tool_use"}]
</output>""",
}


def _format_trajectory(history: list[tuple[str, str]], task_desc: str, success: bool) -> str:
    """Format a trajectory for the insight extractor."""
    lines = [f"Task: {task_desc}"]
    for action, obs in history:
        lines.append(f"> {action}")
        lines.append(obs)
    lines.append(f"[{'SUCCESS' if success else 'FAILED'}]")
    return "\n".join(lines)


def extract_insights_from_pair(
    extractor_llm: LLMClient,
    failed_traj: list[tuple[str, str]],
    failed_desc: str,
    success_traj: list[tuple[str, str]],
    success_desc: str,
    task_type: str,
    domain: str = "alfworld",
) -> list[dict]:
    """Extract insights by comparing a failed and successful trajectory pair."""
    failed_str = _format_trajectory(failed_traj, failed_desc, False)
    success_str = _format_trajectory(success_traj, success_desc, True)

    few_shot = INSIGHT_FEW_SHOTS.get(domain, INSIGHT_FEW_SHOTS["alfworld"])
    system = INSIGHT_SYSTEM_PROMPTS.get(domain, INSIGHT_SYSTEM_PROMPTS["alfworld"])

    prompt = f"""{few_shot}

Now extract insights from these trajectories:

<failed_trajectory>
{failed_str}
</failed_trajectory>

<successful_trajectory>
{success_str}
</successful_trajectory>

Output ONLY the JSON array."""

    return _call_insight_extractor(extractor_llm, prompt, system, task_type)


def extract_insights_from_failure(
    extractor_llm: LLMClient,
    failed_traj: list[tuple[str, str]],
    failed_desc: str,
    task_type: str,
    domain: str = "alfworld",
) -> list[dict]:
    """Extract insights from a single failed trajectory."""
    failed_str = _format_trajectory(failed_traj, failed_desc, False)
    system = INSIGHT_FROM_FAILURE_PROMPTS.get(domain, INSIGHT_FROM_FAILURE_PROMPTS["alfworld"])

    prompt = f"""Analyze this failed trajectory and extract general rules:

<failed_trajectory>
{failed_str}
</failed_trajectory>

Output ONLY the JSON array."""

    return _call_insight_extractor(extractor_llm, prompt, system, task_type)


def _call_insight_extractor(
    extractor_llm: LLMClient, prompt: str, system: str, task_type: str,
) -> list[dict]:
    """Call LLM and parse insight JSON."""
    try:
        response = extractor_llm.complete_text(
            prompt, label="expel_extractor", system=system,
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
            if "rule" in item:
                item.setdefault("task_type", task_type)
                valid.append(item)

        logger.info(f"ExpeL extractor found {len(valid)} insights")
        return valid

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"ExpeL extractor failed: {e}")
        return []

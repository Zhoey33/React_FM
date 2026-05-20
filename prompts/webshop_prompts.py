"""WebShop prompts for action-only agents and Reflexion-style runs."""

import re

SYSTEM_PROMPT_BASE = """You are an online shopping agent. Each turn, output ONLY one action. No explanations.

Actions:
search[keywords] — search for products
click[element] — click a product ID, option, or button

Output format: search[...] or click[...]
Nothing else. No text before or after the action."""

SYSTEM_PROMPT_FM = SYSTEM_PROMPT_BASE

REFLEXION_SYSTEM_PROMPT = """You are an online shopping agent. Each turn, output ONLY one action.

Actions:
think[reasoning] - reason privately about search, product, option, or budget choices
search[keywords] - search for products
click[element] - click a product ID, option, navigation element, or Buy Now

Output format: think[...], search[...], or click[...].
Nothing else. No text before or after the action."""

FEWSHOT_EXAMPLE = """Here is an example of solving a shopping task.

Instruction: Find me a red cotton shirt, size medium, price lower than $50.

> search[red cotton shirt]
Page 1 (Total results: 15) [SEP] B001234 [SEP] Men's Classic Red Cotton Shirt $29.99 [SEP] B001235 [SEP] Women's Red Silk Blouse $45.00
> click[B001234]
Men's Classic Red Cotton Shirt [SEP] Price: $29.99 [SEP] Color: Red, Blue, White [SEP] Size: Small, Medium, Large
> click[Red]
You have selected Red.
> click[Medium]
You have selected Medium.
> click[Buy Now]
Purchase successful! Reward: 1.0"""

REFLEXION_FEWSHOT_EXAMPLE = """Webshop
Instruction:
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars
[Search]

> search[3 ounce bright citrus deodorant sensitive skin]
[Back to Search]
Page 1 (Total results: 50) [SEP] B078GWRC1J [SEP] Bright Citrus Deodorant by Earth Mama 3-Ounce $10.99 [SEP] B078GTKVXY [SEP] Ginger Fresh Deodorant by Earth Mama 3-Ounce $10.99
> think[B078GWRC1J matches bright citrus, sensitive skin, 3 ounce, and the price is below 50 dollars. I should inspect it and select the required options.]
OK.
> click[B078GWRC1J]
[Back to Search] [SEP] scent [assorted scents][bright citrus][calming lavender][ginger fresh] [SEP] size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)] [SEP] Price: $10.99 [SEP] [Buy Now]
> think[The item has the requested scent and size options, so I should select bright citrus and 3 ounce before buying.]
OK.
> click[bright citrus]
You have clicked bright citrus.
> click[3 ounce (pack of 1)]
You have clicked 3 ounce (pack of 1).
> click[Buy Now]
Purchase successful! Reward: 1.0"""


def format_step(action: str, observation: str) -> str:
    return f"> {action}\n{observation}\n"


def latest_reflexion_memory(memory: list[str] | None, limit: int = 3) -> list[str]:
    """Return the Reflexion paper-aligned sliding memory window."""
    return list(memory or [])[-limit:]


def build_reflexion_prompt(
    task_obs: str,
    history: list[tuple[str, str]],
    memory: list[str] | None = None,
) -> str:
    """Build a ReAct + Reflexion prompt for WebShop."""
    sections = [REFLEXION_FEWSHOT_EXAMPLE]
    recent_memory = latest_reflexion_memory(memory)
    if recent_memory:
        memory_lines = ["Your memory for the task below:"]
        for i, item in enumerate(recent_memory):
            memory_lines.append(f"Trial {i}:\n{item.strip()}")
        sections.append("\n".join(memory_lines))

    sections.append("Here is the task.\n" + task_obs)
    prompt = "\n\n".join(sections) + "\n"
    for action, obs in history:
        prompt += format_step(action, obs)
    prompt += "> "
    return prompt


def normalize_reflexion_action(action: str) -> str:
    """Normalize WebShop Reflexion output while preserving ReAct think actions."""
    cleaned = action.strip().split("\n")[0].strip()
    if cleaned.startswith("> "):
        cleaned = cleaned[2:].strip()
    if cleaned.lower().startswith("action:"):
        cleaned = cleaned.split(":", 1)[1].strip()

    think_match = re.search(r"think\[([^\]]+)\]", cleaned, flags=re.IGNORECASE)
    if think_match:
        return f"think[{think_match.group(1)}]"

    search_match = re.search(r"search\[([^\]]+)\]", cleaned, flags=re.IGNORECASE)
    if search_match:
        return f"search[{search_match.group(1)}]"

    click_match = re.search(r"click\[([^\]]+)\]", cleaned, flags=re.IGNORECASE)
    if click_match:
        return f"click[{click_match.group(1)}]"

    lowered = cleaned.lower()
    if lowered.startswith("think:") or lowered.startswith("think "):
        return cleaned
    return "search[product]"


def build_user_prompt(
    task_type: str,
    task_obs: str,
    history: list[tuple[str, str]],
    retrieved_memories: list | None = None,
    memory_style: str = "original",
    hint_text: str | None = None,
    valid_actions: list[str] | None = None,
) -> str:
    """Build user prompt for WebShop."""
    sections = [FEWSHOT_EXAMPLE]

    # Memory injection
    if retrieved_memories:
        if memory_style == "factual":
            memory_lines = []
            for entry in retrieved_memories:
                memory_lines.append(
                    f'Previously, "{entry.failure_action}" failed. The fix was: {entry.get_repair_display()}'
                )
            sections.append("\n".join(memory_lines))
        elif memory_style == "reflexion":
            memory_lines = ["Your memory for this task:"]
            for entry in retrieved_memories:
                memory_lines.append(
                    f'- "{entry.failure_action}" failed → fix: {entry.get_repair_display()}'
                )
            sections.append("\n".join(memory_lines))
        elif memory_style == "hint":
            hints = []
            for entry in retrieved_memories:
                repair = entry.get_repair_display()
                first_action = repair.split("\n")[0].strip()
                for prefix in ("[Strategy] ", "[Plan] ", "[Next action] "):
                    if first_action.startswith(prefix):
                        first_action = first_action[len(prefix):]
                        break
                hints.append(f'Hint: {first_action}')
            sections.append("\n".join(hints))
        else:  # original
            memory_lines = [
                "You can refer to these past failure-recovery experiences to help decide your next action."
            ]
            for entry in retrieved_memories:
                memory_lines.append(
                    f"- failure: {entry.failure_action}, fix: {entry.get_repair_display()}"
                )
            sections.append("\n".join(memory_lines))

    # Task observation
    sections.append("Here is the task.\n" + task_obs)
    prompt = "\n\n".join(sections) + "\n"

    for action, obs in history:
        prompt += format_step(action, obs)

    if valid_actions:
        action_lines = "\n".join(f"- {action}" for action in valid_actions)
        prompt += (
            "\nChoose exactly one valid action from this list and output only that action.\n"
            + action_lines
            + "\n"
        )

    if hint_text:
        prompt += f"\n{hint_text}\n"

    prompt += "> "
    return prompt

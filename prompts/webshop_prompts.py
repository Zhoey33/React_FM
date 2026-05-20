"""WebShop prompts — system/user split, memory injection."""

SYSTEM_PROMPT_BASE = """You are an online shopping agent. Each turn, output ONLY one action. No explanations.

Actions:
search[keywords] — search for products
click[element] — click a product ID, option, or button

Output format: search[...] or click[...]
Nothing else. No text before or after the action."""

SYSTEM_PROMPT_FM = SYSTEM_PROMPT_BASE

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


def format_step(action: str, observation: str) -> str:
    return f"> {action}\n{observation}\n"


def build_user_prompt(
    task_type: str,
    task_obs: str,
    history: list[tuple[str, str]],
    retrieved_memories: list | None = None,
    memory_style: str = "original",
    hint_text: str | None = None,
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

    if hint_text:
        prompt += f"\n{hint_text}\n"

    prompt += "> "
    return prompt

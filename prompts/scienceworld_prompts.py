"""ScienceWorld prompts — system/user split, memory injection."""

SYSTEM_PROMPT_BASE = """You are a science experiment agent in a text-based simulation. Each turn, output ONLY a single action or think command. No explanations.

Valid actions: go to, pick up, put down, open, close, activate, deactivate, use, pour, mix, focus on, wait, look around, inventory, examine, read, connect, move, teleport to, dunk

Output format — one of:
think: [reasoning]
[action]

Rules:
- Open containers before taking items from them.
- Pick up objects before using them.
- If an action fails, try a different approach.
- Do not repeat failed actions."""

SYSTEM_PROMPT_FM = SYSTEM_PROMPT_BASE

# Few-shot examples by topic
FEWSHOT_EXAMPLES = {
    "matter": """Here is an example of solving a boiling task.

Task: Your task is to boil water. First, focus on the water. Then, heat the water until it changes to a gas.

> look around
This room is called the kitchen. In it, you see: a stove, which is turned off. On the stove is: nothing. You also see: a sink, which is turned off. In the sink is: nothing. a table. On the table is: a glass cup (containing water), a metal pot (containing nothing).
> pick up metal pot
You pick up the metal pot.
> move metal pot to stove
You move the metal pot to: stove.
> pick up glass cup
You pick up the glass cup (containing water).
> pour glass cup into metal pot
You pour the glass cup into the metal pot. The metal pot now contains: water.
> focus on water in metal pot
You focus on the water.
> activate stove
You activate the stove. The water in the metal pot starts to heat up.
> wait
Time passes. The water is heating.
> wait
Time passes. The water is heating.
> wait
Time passes. The water is now boiling! It has changed to steam (gas).

Task completed successfully.""",

    "default": """Here is an example of solving a science task.

Task: Your task is to find a living thing.

> look around
This room is called the outside. In it, you see: a bird, a rock, a tree, a flower pot with a plant.
> focus on bird
You focus on the bird. A bird is a living thing.

Task completed successfully.""",
}


def get_fewshot_examples(task_name: str) -> str:
    """Get few-shot example for a task type."""
    from src.scienceworld_env import TASK_TOPIC_MAP
    topic = TASK_TOPIC_MAP.get(task_name, "default")
    return FEWSHOT_EXAMPLES.get(topic, FEWSHOT_EXAMPLES["default"])


def format_step(action: str, observation: str) -> str:
    return f"> {action}\n{observation}\n"


def build_baseline_user_prompt(
    task_type: str,
    task_obs: str,
    history: list[tuple[str, str]],
    hint_text: str | None = None,
) -> str:
    """Build a pure ReAct prompt with no memory-related sections."""
    sections = [get_fewshot_examples(task_type), "Here is the task.\n" + task_obs]
    prompt = "\n\n".join(sections) + "\n"

    for action, obs in history:
        prompt += format_step(action, obs)

    if hint_text:
        prompt += f"\n{hint_text}\n"

    prompt += "> "
    return prompt


def build_user_prompt(
    task_type: str,
    task_obs: str,
    history: list[tuple[str, str]],
    retrieved_memories: list | None = None,
    memory_style: str = "original",
    hint_text: str | None = None,
) -> str:
    """Build user prompt for ScienceWorld, matching ALFWorld interface."""
    sections = []

    # Few-shot example
    examples = get_fewshot_examples(task_type)
    sections.append(examples)

    # Memory injection (same styles as ALFWorld)
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

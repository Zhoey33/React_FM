"""ALFWorld prompts — system/user split, Reflexion-compatible format."""

import json
from pathlib import Path

# Load Reflexion's few-shot examples
_PROMPT_FILE = Path(__file__).parent / "alfworld_3prompts.json"
with open(_PROMPT_FILE) as f:
    _PROMPTS = json.load(f)

# Task types
TASK_TYPES = ["put", "clean", "heat", "cool", "examine", "puttwo"]

FEWSHOT_INTRO = "Interact with a household to solve a task. Here are two examples."
SYSTEM_PROMPT_BASE = """You are a household robot agent. You complete tasks in a text-based environment by reasoning and choosing actions step by step.

Each turn, output exactly ONE line:
- think: [your reasoning]
- Or a valid action: go to, take, put, open, close, toggle, clean, cool, heat, use, examine, look, inventory

Rules:
- Open closed receptacles before taking items from them.
- You can carry only one object at a time.
- If an action fails ("Nothing happens"), think about why and try a different approach.
- Do not repeat the exact same failed action unless the state has changed."""

SYSTEM_PROMPT_FM = SYSTEM_PROMPT_BASE


def get_fewshot_examples(task_type: str, num_examples: int = 2) -> str:
    """Get few-shot examples for a task type from Reflexion prompts."""
    examples = []
    for i in reversed(range(num_examples)):
        key = f"react_{task_type}_{i}"
        if key in _PROMPTS:
            examples.append(_PROMPTS[key])
    return "\n".join(examples)


def build_fewshot_prefix(task_type: str, num_examples: int = 2) -> str:
    """Build the shared few-shot prefix used by all action prompts."""
    examples = get_fewshot_examples(task_type, num_examples=num_examples)
    return FEWSHOT_INTRO + "\n" + examples


def format_step(action: str, observation: str) -> str:
    """Format a single step in Reflexion's format."""
    return f"> {action}\n{observation}\n"


def build_user_prompt(
    task_type: str,
    task_obs: str,
    history: list[tuple[str, str]],
    retrieved_memories: list | None = None,
) -> str:
    """Build the user message in Reflexion-compatible format.

    Args:
        task_type: Task type for selecting few-shot example
        task_obs: Initial task observation from env.reset()
        history: List of (action, observation) tuples so far
        retrieved_memories: List of FailureMemoryEntry objects (or None)
    """
    sections = [build_fewshot_prefix(task_type)]

    if retrieved_memories:
        memory_lines = [
            "You can refer to these past failure-recovery experiences to help decide your next action."
        ]
        for entry in retrieved_memories:
            memory_lines.append(
                f"- failure: {entry.failure_action}, fix: {entry.solution_action}"
            )
        sections.append("\n".join(memory_lines))

    sections.append("Here is the task.\n" + task_obs)
    prompt = "\n\n".join(sections) + "\n"

    for action, obs in history:
        prompt += format_step(action, obs)

    prompt += "> "
    return prompt

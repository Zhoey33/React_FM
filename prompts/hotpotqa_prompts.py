"""HotPotQA prompts — system/user split, memory injection."""

SYSTEM_PROMPT_BASE = """You are a question-answering agent. Each turn, output a Thought and an Action.

Actions:
Search[entity] — search Wikipedia for an entity
Lookup[keyword] — find next sentence with keyword on current page
Finish[answer] — submit your final answer

Output format:
Thought: your reasoning
Action: Search[...] or Lookup[...] or Finish[...]

Nothing else. One thought and one action per turn."""

SYSTEM_PROMPT_FM = SYSTEM_PROMPT_BASE

FEWSHOT_EXAMPLE = """Here are examples of solving multi-hop questions.

Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought: I need to search Pavel Urysohn and Leonid Levin, and find their types of work.
Action: Search[Pavel Urysohn]
Observation: Pavel Samuilovich Urysohn (February 3, 1898 – August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.
Thought: Pavel Urysohn is a mathematician. I need to search Leonid Levin next.
Action: Search[Leonid Levin]
Observation: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.
Thought: Leonid Levin is a mathematician and computer scientist. Both are known for mathematical work, so yes.
Action: Finish[yes]

Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought: I need to search Colorado orogeny and find the area the eastern sector extends into.
Action: Search[Colorado orogeny]
Observation: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
Thought: It does not mention the eastern sector. I should look up eastern sector.
Action: Lookup[eastern sector]
Observation: (Result 1/1) The eastern sector extends into the High Plains.
Thought: The eastern sector extends into the High Plains. I need to search High Plains and find its elevation range.
Action: Search[High Plains]
Observation: High Plains refers to one of the two distinct land regions of the Great Plains of the United States.
Thought: I need to look up the elevation of the High Plains.
Action: Lookup[elevation]
Observation: (Result 1/2) The High Plains are a subregion of the Great Plains, with elevation ranging from around 1,800 to 7,000 ft (550 to 2,130 m).
Thought: The elevation range is approximately 1,800 to 7,000 ft.
Action: Finish[approximately 1,800 to 7,000 ft]"""


def format_step(thought: str, action: str, observation: str) -> str:
    lines = []
    if thought:
        lines.append(f"Thought: {thought}")
    lines.append(f"Action: {action}")
    lines.append(f"Observation: {observation}")
    return "\n".join(lines) + "\n"


def build_user_prompt(
    task_obs: str,
    history: list[tuple[str, str, str]],
    retrieved_memories: list | None = None,
    memory_style: str = "original",
) -> str:
    """Build user prompt for HotPotQA.

    history: list of (thought, action, observation) tuples
    """
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

    sections.append(task_obs)
    prompt = "\n\n".join(sections) + "\n"

    for thought, action, observation in history:
        prompt += format_step(thought, action, observation)

    return prompt

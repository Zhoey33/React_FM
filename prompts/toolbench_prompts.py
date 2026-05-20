"""ToolBench prompts — system/user split, memory injection."""

SYSTEM_PROMPT_BASE = """You are a tool-use agent. Each turn, output a Thought and an Action.

Actions:
function_name({"param": "value"}) — call an API with JSON parameters
Finish({"return_type": "give_answer", "final_answer": "your answer"}) — submit final answer

Output format:
Thought: your reasoning about which API to call and why
Action: function_name({"param": "value"})

Nothing else. One thought and one action per turn. Parameters must be valid JSON."""

SYSTEM_PROMPT_FM = SYSTEM_PROMPT_BASE

FEWSHOT_EXAMPLE = """Here is an example of solving a task using API tools.

Task: What is the current weather in San Francisco?

Available APIs:
- get_weather_for_weather_api(city: string): Get current weather for a city
- Finish(return_type, final_answer): Submit your final answer.

Thought: I need to get the weather for San Francisco using the weather API.
Action: get_weather_for_weather_api({"city": "San Francisco"})
Observation: {"temperature": 65, "condition": "Partly Cloudy", "humidity": 72}
Thought: I have the weather data. The temperature is 65F and partly cloudy.
Action: Finish({"return_type": "give_answer", "final_answer": "The current weather in San Francisco is 65°F and Partly Cloudy with 72% humidity."})"""


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
    """Build user prompt for ToolBench."""
    sections = [FEWSHOT_EXAMPLE]

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

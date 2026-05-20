"""Reflexion agent for ALFWorld — reproduced from github.com/noahshinn/reflexion."""

import logging
import time
from pathlib import Path

from src.llm import LLMClient
from src.alfworld_env import ALFWorldEnv, process_observation, get_task_type

logger = logging.getLogger(__name__)

# Load few-shot examples for reflection generation
_FEW_SHOT_FILE = Path(__file__).parent.parent / "prompts" / "reflexion_few_shot_examples.txt"
with open(_FEW_SHOT_FILE) as f:
    REFLEXION_FEW_SHOT = f.read()

_WEBSHOP_FEW_SHOT_FILE = Path(__file__).parent.parent / "prompts" / "webshop_reflexion_few_shot_examples.txt"
with open(_WEBSHOP_FEW_SHOT_FILE) as f:
    WEBSHOP_REFLEXION_FEW_SHOT = f.read()

# Task type prefixes (same as alfworld_trial.py)
PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}
class EnvironmentHistory:
    """Tracks the history of actions and observations for one episode.
    Reproduces env_history.py from the Reflexion repo.
    """

    def __init__(self, base_prompt: str, start_info: str, memory: list[str]):
        self._base_query = self._build_base_query(base_prompt, start_info, memory)
        self._history: list[dict] = []
        self._last_action: str = ""
        self._is_exhausted: bool = False

    @staticmethod
    def _build_base_query(base_prompt: str, start_info: str, memory: list[str]) -> str:
        sections = [base_prompt]
        if memory:
            memory_lines = ["Your memory for the task below:"]
            for i, m in enumerate(memory):
                memory_lines.append(f"Trial {i}:\n{m.strip()}")
            sections.append("\n".join(memory_lines))
        sections.append(f"Here is the task:\n{start_info}")
        return "\n\n".join(sections)

    def add(self, label: str, value: str):
        self._history.append({"label": label, "value": value})
        if label == "action":
            if value == self._last_action:
                self._is_exhausted = True
            else:
                self._last_action = value

    def check_is_exhausted(self) -> bool:
        return self._is_exhausted

    def reset(self):
        self._history = []

    def __str__(self) -> str:
        s = self._base_query + "\n"
        for i, item in enumerate(self._history):
            if item["label"] == "action":
                s += f"> {item['value']}"
            elif item["label"] == "observation":
                s += item["value"]
            if i != len(self._history) - 1:
                s += "\n"
        return s


def generate_reflection(llm: LLMClient, log_str: str, memory: list[str], domain: str = "alfworld") -> str:
    """Generate a reflection for a failed episode.
    Reproduces generate_reflections.py from the Reflexion repo.
    Supports domains: alfworld, webshop, scienceworld.
    """
    # Parse scenario from log. Different runners may use either
    # "Here is the task:" or "Here is the task." as the section header.
    scenario = log_str
    for marker in ("Here is the task:", "Here is the task."):
        if marker in log_str:
            scenario = log_str.split(marker, 1)[-1].strip()
            break
    if not scenario or scenario == log_str.strip():
        # Fallback: use full log (for non-ALFWorld domains)
        scenario = log_str[-3000:]  # truncate to avoid token overflow

    domain_desc = {
        "alfworld": "placed in an environment and given a task to complete",
        "webshop": "given a shopping task to find and purchase a product online",
        "scienceworld": "given a science experiment task to complete in a simulated environment",
        "hotpotqa": "given a multi-hop question-answering task using Wikipedia search",
        "toolbench": "given a task that requires calling multiple APIs to gather information and provide an answer",
    }
    desc = domain_desc.get(domain, domain_desc["alfworld"])

    recent_memory = memory[-3:] if len(memory) > 3 else memory
    few_shot = ""
    if domain == "alfworld":
        few_shot = f"\n\n{REFLEXION_FEW_SHOT}\n\n"
    elif domain == "webshop":
        few_shot = f" There are two examples below.\n\n{WEBSHOP_REFLEXION_FEW_SHOT}\n\n"
    else:
        few_shot = "\n\n"

    query = f"""You will be given the history of a past experience in which you were {desc}. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Give your plan after "Plan".{few_shot}{scenario}"""

    if recent_memory:
        query += "\n\nPlans from past attempts:\n"
        for i, m in enumerate(recent_memory):
            query += f"Trial #{i}: {m}\n"

    query += "\n\nNew plan:"
    reflection = llm.complete_text(query, label="reflection", system=None)
    return reflection.strip()


def run_reflexion_episode(
    env: ALFWorldEnv,
    llm: LLMClient,
    base_prompt: str,
    ob: str,
    memory: list[str],
    max_steps: int = 49,
) -> tuple[str, bool]:
    """Run one Reflexion episode. Returns (log_str, is_success).
    Reproduces alfworld_run() from the Reflexion repo.
    """
    env_history = EnvironmentHistory(base_prompt, ob, memory[-3:] if len(memory) > 3 else memory)
    env_history.reset()

    for step in range(max_steps):
        prompt = str(env_history) + ">"
        action = llm.complete_text(prompt, stop=["\n"], label=f"step_{step}")
        action = action.strip()

        # Strip "> " prefix if present
        if action.startswith("> "):
            action = action[2:]

        if not action:
            action = "look"

        logger.info(f"  Step {step}: {action[:80]}")
        env_history.add("action", action)

        if action.startswith("think:") or action.startswith("think "):
            observation = "OK."
        else:
            # Execute (ALFWorldEnv.step handles translate_action internally)
            observation, reward, done, info = env.step(action)
            logger.info(f"    obs: {observation[:80]}")
            won = info.get("won", [False])
            if isinstance(won, list):
                won = won[0]
            if done:
                env_history.add("observation", observation)
                status = "SUCCESS" if won else "FAIL"
                logger.info(f"  Episode done: {status} in {step+1} steps")
                return str(env_history), bool(won)

        env_history.add("observation", observation)

        if env_history.check_is_exhausted():
            logger.info(f"  Episode exhausted (repeated action) at step {step+1}")
            return str(env_history), False

    logger.info(f"  Episode reached max steps ({max_steps})")
    return str(env_history), False

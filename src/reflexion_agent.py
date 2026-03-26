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

# Task type prefixes (same as alfworld_trial.py)
PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}

SYSTEM_PROMPT = """You are a household robot agent. You complete tasks in a text-based environment by reasoning and choosing actions step by step.

Each turn, output exactly ONE line:
- think: [your reasoning]
- Or a valid action: go to, take, put, open, close, toggle, clean, cool, heat, use, examine, look, inventory

Rules:
- Open closed receptacles before taking items from them.
- You can carry only one object at a time.
- If an action fails ("Nothing happens"), think about why and try a different approach.
- Do not repeat the exact same failed action unless the state has changed."""


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
        query = base_prompt
        if memory:
            query += "\n\nYour memory for the task below:"
            for i, m in enumerate(memory):
                query += f"\nTrial {i}:\n{m.strip()}"
        query += f"\nHere is the task:\n{start_info}"
        return query

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


def generate_reflection(llm: LLMClient, log_str: str, memory: list[str]) -> str:
    """Generate a reflection for a failed episode.
    Reproduces generate_reflections.py from the Reflexion repo.
    """
    # Parse scenario from log
    scenario = log_str.split("Here is the task:")[-1].strip()

    query = f"""You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Give your plan after "Plan". Here are two examples:

{REFLEXION_FEW_SHOT}

{scenario}"""

    if memory:
        query += "\n\nPlans from past attempts:\n"
        for i, m in enumerate(memory):
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

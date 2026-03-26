"""ALFWorld environment wrapper."""

import os
import re
import logging
import importlib

logger = logging.getLogger(__name__)


def translate_action(action: str) -> str:
    """Translate human-friendly 'put' actions to ALFWorld programmatic format.

    LLM outputs 'put X on Y', 'put X in Y', or 'put X in/on Y',
    but ALFWorld programmatic interface expects 'move X to Y'.
    """
    m = re.match(r"put (.+?) (?:in/on|in|on) (.+)", action)
    if m:
        return f"move {m.group(1)} to {m.group(2)}"
    return action

# Task type mapping: directory prefix → short name
TASK_TYPE_MAP = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


def process_observation(ob: str) -> str:
    """Clean observation string."""
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2:]
    return ob.strip()


def get_task_type(gamefile: str) -> str:
    """Extract task type from ALFWorld gamefile path.

    Gamefile path looks like: .../pick_cool_then_place_in_recep-Tomato-.../trial_.../game.tw-pddl
    We need to check the grandparent directory name, not the basename.
    """
    # Walk up path components to find a matching prefix
    parts = gamefile.replace("\\", "/").split("/")
    for part in parts:
        for prefix, task_type in TASK_TYPE_MAP.items():
            if part.startswith(prefix):
                return task_type
    return "unknown"


class ALFWorldEnv:
    def __init__(self, split: str = "eval_out_of_distribution"):
        self.split = split
        self.env = None
        self._env_idx = 0

    def setup(self, alfworld_config: str = "alfworld_config.yaml"):
        """Initialize ALFWorld environment. Call once before episodes."""
        import yaml as _yaml

        # Reload to avoid memory leaks (from Reflexion)
        import alfworld
        import alfworld.agents.environment
        importlib.reload(alfworld)
        importlib.reload(alfworld.agents.environment)

        from alfworld.agents.environment import get_environment

        # Load config directly (bypass ALFWorld's argparse-based load_config)
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), alfworld_config)
        with open(config_path) as f:
            config = _yaml.safe_load(f)

        # Expand ~ in paths
        for section in ["dataset", "logic"]:
            if section in config:
                for key, val in config[section].items():
                    if isinstance(val, str) and "~" in val:
                        config[section][key] = os.path.expanduser(val)

        env_type = config["env"]["type"]
        self.env = get_environment(env_type)(config, train_eval=self.split)
        self.env = self.env.init_env(batch_size=1)
        self._env_idx = 0
        logger.info(f"ALFWorld environment initialized (split={self.split})")

    def reset(self) -> tuple[str, str, dict]:
        """Reset to next environment.
        Returns: (initial_observation, task_type, info)
        """
        obs, info = self.env.reset()
        # obs is a list of strings; take first
        ob = obs[0]
        # Parse: first paragraph is boilerplate, rest is task description
        ob = "\n".join(ob.split("\n\n")[1:])
        ob = process_observation(ob)

        # Get task type from gamefile
        gamefile = info.get("extra.gamefile", [""])[0] if isinstance(info.get("extra.gamefile"), list) else info.get("extra.gamefile", "")
        task_type = get_task_type(gamefile)

        self._env_idx += 1
        logger.info(f"Env #{self._env_idx}: task_type={task_type}")
        return ob, task_type, info

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Execute action. Returns (observation, reward, done, info)."""
        action = translate_action(action)
        obs, rewards, dones, infos = self.env.step([action])
        ob = process_observation(obs[0])
        reward = rewards[0] if isinstance(rewards, (list, tuple)) else rewards
        done = dones[0] if isinstance(dones, (list, tuple)) else dones
        return ob, float(reward), bool(done), infos

    def skip(self):
        """Advance to next environment without running it (for resume)."""
        self.reset()
        logger.debug(f"Skipped env #{self._env_idx}")

    @property
    def env_idx(self) -> int:
        return self._env_idx

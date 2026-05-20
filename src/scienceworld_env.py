"""ScienceWorld environment wrapper — unified interface matching ALFWorldEnv."""

import os
import logging

logger = logging.getLogger(__name__)

# Ensure Java is on PATH for py4j
_JAVA_HOME = "/opt/homebrew/opt/openjdk/bin"
if _JAVA_HOME not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _JAVA_HOME + ":" + os.environ.get("PATH", "")

# Official ScienceWorld task types: 30 total.
# The packaged tasks.json uses 7 coarse topics. The mapping below is repo-specific
# and splits Biology into finer labels for analysis convenience.
TASK_TOPIC_MAP = {
    "boil": "matter",
    "melt": "matter",
    "freeze": "matter",
    "change-the-state-of-matter-of": "matter",
    "use-thermometer": "measurement",
    "measure-melting-point-known-substance": "measurement",
    "measure-melting-point-unknown-substance": "measurement",
    "power-component": "electricity",
    "power-component-renewable-vs-nonrenewable-energy": "electricity",
    "test-conductivity": "electricity",
    "test-conductivity-of-unknown-substances": "electricity",
    "find-animal": "classification",
    "find-living-thing": "classification",
    "find-non-living-thing": "classification",
    "find-plant": "classification",
    "grow-plant": "biology-growth",
    "grow-fruit": "biology-growth",
    "chemistry-mix": "chemistry",
    "chemistry-mix-paint-secondary-color": "chemistry",
    "chemistry-mix-paint-tertiary-color": "chemistry",
    "lifespan-longest-lived": "lifespan",
    "lifespan-shortest-lived": "lifespan",
    "lifespan-longest-lived-then-shortest-lived": "lifespan",
    "identify-life-stages-1": "life-stages",
    "identify-life-stages-2": "life-stages",
    "inclined-plane-determine-angle": "forces",
    "inclined-plane-friction-named-surfaces": "forces",
    "inclined-plane-friction-unnamed-surfaces": "forces",
    "mendelian-genetics-known-plant": "genetics",
    "mendelian-genetics-unknown-plant": "genetics",
}

# All official task names in this repository's canonical order.
OFFICIAL_TASKS = list(TASK_TOPIC_MAP.keys())

# Repo default subset for reproducible evaluation.
# This is not the full official benchmark: it picks 10 representative tasks,
# with up to 5 variations each by default, for ~50 episodes per split.
DEFAULT_EVAL_TASKS = [
    "boil",
    "melt",
    "use-thermometer",
    "power-component",
    "test-conductivity",
    "find-living-thing",
    "grow-plant",
    "chemistry-mix",
    "lifespan-longest-lived",
    "mendelian-genetics-known-plant",
]


class ScienceWorldEnv:
    """Wrapper for ScienceWorld with ALFWorld-compatible interface."""

    def __init__(
        self,
        task_names: list[str] | None = None,
        split: str = "test",
        max_variations_per_task: int = 5,
        env_step_limit: int = 100,
    ):
        self.task_names = task_names or DEFAULT_EVAL_TASKS
        self.split = split
        self.max_variations_per_task = max_variations_per_task
        self.env_step_limit = env_step_limit
        self.env = None
        self._env_idx = 0
        self._schedule: list[tuple[str, int]] = []  # (task_name, variation_idx)
        self._schedule_idx = 0

    def _load_scheduled_episode(self, schedule_idx: int, env_idx: int) -> tuple[str, str, dict]:
        """Load a specific scheduled episode by absolute schedule index."""
        if schedule_idx < 0 or schedule_idx >= len(self._schedule):
            raise IndexError(f"Episode index out of range: {schedule_idx}")

        task_name, var_idx = self._schedule[schedule_idx]
        self.env.load(task_name, var_idx)
        obs, info = self.env.reset()

        task_desc = self.env.get_task_description()
        topic = TASK_TOPIC_MAP.get(task_name, task_name)

        self._env_idx = env_idx

        # Build observation with task description
        full_obs = f"{task_desc}\n\n{obs}"

        info_dict = {
            "task_name": task_name,
            "variation_idx": var_idx,
            "topic": topic,
            "task_desc": task_desc,
            "look": self.env.look(),
            "inventory": self.env.inventory(),
            "valid_actions": "\n".join(self.env.get_valid_action_object_combinations()),
        }

        logger.info(f"Env #{env_idx}: task={task_name}, var={var_idx}, topic={topic}")
        return full_obs, task_name, info_dict

    def setup(self):
        """Initialize ScienceWorld JVM and build evaluation schedule."""
        import warnings
        warnings.filterwarnings("ignore", message=".*camel case.*")
        from scienceworld import ScienceWorldEnv as SWEnv

        # Build evaluation schedule by querying variations per task
        self._schedule = []
        for task_name in self.task_names:
            tmp_env = SWEnv(task_name, envStepLimit=self.env_step_limit)
            if self.split == "test":
                variations = tmp_env.get_variations_test()
            elif self.split == "dev":
                variations = tmp_env.get_variations_dev()
            else:
                variations = tmp_env.get_variations_train()
            tmp_env.close()

            selected = variations[:self.max_variations_per_task]
            for var_idx in selected:
                self._schedule.append((task_name, var_idx))

        # Create main env (will load specific tasks on reset)
        self.env = SWEnv("", envStepLimit=self.env_step_limit)
        logger.info(f"ScienceWorld JVM started (step_limit={self.env_step_limit})")

        self._schedule_idx = 0
        logger.info(
            f"ScienceWorld schedule: {len(self._schedule)} episodes "
            f"({len(self.task_names)} tasks × ≤{self.max_variations_per_task} variations, split={self.split})"
        )

    def reset(self) -> tuple[str, str, dict]:
        """Reset to next scheduled task/variation.
        Returns: (initial_observation, task_type, info)
        Raises StopIteration when schedule exhausted.
        """
        if self._schedule_idx >= len(self._schedule):
            raise StopIteration("All scheduled episodes completed")

        env_idx = self._schedule_idx + 1
        result = self._load_scheduled_episode(self._schedule_idx, env_idx)
        self._schedule_idx += 1
        return result

    def reset_to_episode(self, episode_idx: int) -> tuple[str, str, dict]:
        """Reset to a specific scheduled episode without advancing the schedule pointer."""
        return self._load_scheduled_episode(episode_idx, episode_idx + 1)

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Execute action. Returns (observation, reward, done, info)."""
        obs, reward, done, info = self.env.step(action)

        score = info.get("score", 0.0)
        info_dict = {
            "score": score,
            "moves": info.get("moves", 0),
            "look": info.get("look", ""),
            "inventory": info.get("inv", ""),
            "valid_actions": "\n".join(self.env.get_valid_action_object_combinations()),
        }

        return obs, float(reward), bool(done), info_dict

    def skip(self):
        """Advance schedule without running episode."""
        if self._schedule_idx < len(self._schedule):
            self._env_idx += 1
            self._schedule_idx += 1
            logger.debug(f"Skipped env #{self._env_idx}")
        else:
            raise StopIteration("Schedule exhausted")

    def get_possible_actions(self) -> list[str]:
        """Get valid actions for current state."""
        return self.env.get_valid_action_object_combinations()

    def close(self):
        """Shut down JVM."""
        if self.env:
            self.env.close()
            self.env = None

    @property
    def env_idx(self) -> int:
        return self._env_idx

    @property
    def total_episodes(self) -> int:
        return len(self._schedule)

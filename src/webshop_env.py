"""WebShop environment wrapper for paper-aligned subprocess execution.

WebShop requires Python 3.10 with incompatible dependencies, so we run it
in a separate conda environment and communicate via subprocess + JSON.
"""

import json
import logging
import subprocess
import sys
import os
import selectors
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# WebShop bridge script that runs in the conda webshop env
_BRIDGE_SCRIPT = Path(__file__).parent.parent / "experiments" / "webshop_bridge.py"

JAVA_HOME = "/opt/homebrew/Cellar/openjdk/25.0.2/libexec/openjdk.jdk/Contents/Home"
WEBSHOP_PYTHON = "/opt/miniconda3/envs/webshop/bin/python"


class WebShopEnv:
    """Wrapper that runs WebShop in a subprocess via conda env."""

    def __init__(
        self,
        num_products: int | None = None,
        observation_mode: str = "text_rich",
        max_sessions: int = 500,
        human_goals: int = 1,
        split: str = "test",
        step_limit: int = 100,
        wrapper: str = "official",
    ):
        self.num_products = num_products
        self.observation_mode = observation_mode
        self.max_sessions = max_sessions
        self.human_goals = human_goals
        self.split = split
        self.step_limit = step_limit
        self.wrapper = wrapper
        self._proc = None
        self._env_idx = 0
        self._current_instruction = ""

    def setup(self):
        """Start the WebShop bridge subprocess."""
        env = os.environ.copy()
        env["JAVA_HOME"] = JAVA_HOME
        env["PATH"] = f"/opt/homebrew/opt/openjdk/bin:{env.get('PATH', '')}"

        self._proc = subprocess.Popen(
            [
                WEBSHOP_PYTHON,
                str(_BRIDGE_SCRIPT),
                "--num-products", "full" if self.num_products is None else str(self.num_products),
                "--observation-mode", self.observation_mode,
                "--human-goals", str(self.human_goals),
                "--split", self.split,
                "--step-limit", str(self.step_limit),
                "--wrapper", self.wrapper,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=str(Path(__file__).parent.parent),
        )

        # Wait for ready signal (skip non-READY lines) and surface bridge errors.
        selector = selectors.DefaultSelector()
        selector.register(self._proc.stdout, selectors.EVENT_READ)
        selector.register(self._proc.stderr, selectors.EVENT_READ)
        stderr_lines: list[str] = []
        start = time.time()
        while time.time() - start < 300:
            events = selector.select(timeout=0.2)
            if not events and self._proc.poll() is not None:
                break
            for key, _ in events:
                line = key.fileobj.readline()
                if not line:
                    continue
                line = line.strip()
                if key.fileobj is self._proc.stdout:
                    if line == "READY":
                        logger.info(
                            f"WebShop bridge started (products={self.num_products or 'full'}, "
                            f"split={self.split}, wrapper={self.wrapper})"
                        )
                        return
                elif line:
                    stderr_lines.append(line)
                    logger.debug(f"WebShop bridge stderr: {line[:200]}")

        err = "\n".join(stderr_lines[-20:])
        if self._proc.poll() is not None:
            raise RuntimeError(f"WebShop bridge failed to start (rc={self._proc.returncode})\n{err}")
        raise TimeoutError(f"Timed out waiting for WebShop bridge READY.\n{err}")

    def _send_command(self, cmd: dict) -> dict:
        """Send command to bridge and get response."""
        self._proc.stdin.write(json.dumps(cmd) + "\n")
        self._proc.stdin.flush()
        while True:
            line = self._proc.stdout.readline()
            if not line:
                # Process died
                rc = self._proc.poll()
                raise RuntimeError(f"WebShop bridge died (rc={rc})")
            line = line.strip()
            if not line:
                continue
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                # Skip non-JSON output (warnings, etc.)
                logger.debug(f"Bridge non-JSON: {line[:100]}")
                continue
            if "error" in response:
                raise RuntimeError(f"WebShop bridge error: {response['error']}")
            return response

    def reset(self, session_idx: int | None = None) -> tuple[str, str, dict]:
        """Reset to a new shopping session.
        Returns: (observation, task_type, info)
        """
        idx = session_idx if session_idx is not None else self._env_idx
        resp = self._send_command({"cmd": "reset", "session": idx})

        self._env_idx += 1
        obs = resp["observation"]

        # Extract instruction from observation
        # Format: "Instruction: [SEP] ... [SEP] Search"
        instruction = ""
        if "[SEP]" in obs:
            parts = obs.split("[SEP]")
            if len(parts) >= 2:
                instruction = parts[1].strip()
        self._current_instruction = instruction

        info = {
            "session_idx": idx,
            "instruction": instruction,
            "available_actions": resp.get("actions", {}),
            "goal": resp.get("goal", ""),
        }

        logger.info(f"Env #{self._env_idx}: session={idx}, instruction={instruction[:80]}")
        return obs, "shopping", info

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Execute action. Returns (observation, reward, done, info)."""
        resp = self._send_command({"cmd": "step", "action": action})

        info = {
            "available_actions": resp.get("actions", {}),
            "raw_info": resp.get("raw_info", {}),
        }

        return resp["observation"], float(resp["reward"]), bool(resp["done"]), info

    def skip(self):
        """Skip a session."""
        self._env_idx += 1

    def close(self):
        """Shut down bridge."""
        if self._proc:
            try:
                self._send_command({"cmd": "close"})
            except Exception:
                pass
            self._proc.terminate()
            self._proc = None

    @property
    def env_idx(self) -> int:
        return self._env_idx

    @property
    def total_episodes(self) -> int:
        return self.max_sessions

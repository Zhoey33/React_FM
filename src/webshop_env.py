"""WebShop environment wrapper — subprocess bridge to conda webshop env.

WebShop requires Python 3.10 with incompatible dependencies, so we run it
in a separate conda environment and communicate via subprocess + JSON.
"""

import json
import logging
import subprocess
import sys
import os
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
        num_products: int = 1000,
        observation_mode: str = "text",
        max_sessions: int = 500,
    ):
        self.num_products = num_products
        self.observation_mode = observation_mode
        self.max_sessions = max_sessions
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
                "--num-products", str(self.num_products),
                "--observation-mode", self.observation_mode,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=str(Path(__file__).parent.parent),
        )

        # Wait for ready signal (skip non-READY lines)
        while True:
            line = self._proc.stdout.readline()
            if not line:
                rc = self._proc.poll()
                raise RuntimeError(f"WebShop bridge failed to start (rc={rc})")
            line = line.strip()
            if line == "READY":
                break
            # Skip other output lines during startup

        logger.info(f"WebShop bridge started (products={self.num_products})")

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
                return json.loads(line)
            except json.JSONDecodeError:
                # Skip non-JSON output (warnings, etc.)
                logger.debug(f"Bridge non-JSON: {line[:100]}")
                continue

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
        }

        logger.info(f"Env #{self._env_idx}: session={idx}, instruction={instruction[:80]}")
        return obs, "shopping", info

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Execute action. Returns (observation, reward, done, info)."""
        resp = self._send_command({"cmd": "step", "action": action})

        info = {
            "available_actions": resp.get("actions", {}),
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

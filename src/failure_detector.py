"""Rule-based + LLM judge failure detection for ALFWorld."""

import logging
from dataclasses import dataclass

from src.llm import LLMClient

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are judging whether an action succeeded in a household environment.

Action: {action}
Observation: {observation}

Did the action achieve its intended effect? Answer only "yes" or "no"."""


@dataclass
class DetectionResult:
    is_failure: bool
    failure_type: str = ""  # nothing_happens, action_loop, empty_obs, intent_mismatch
    reason: str = ""


class ALFWorldFailureDetector:
    def __init__(self, loop_window: int = 3, judge_llm: LLMClient | None = None):
        self.loop_window = loop_window
        self.judge_llm = judge_llm

    def detect(
        self,
        observation: str,
        action: str,
        action_history: list[str],
    ) -> DetectionResult:
        obs = observation.strip().lower()

        # Rule 1: "Nothing happens" — dominant failure signal
        if "nothing happens" in obs:
            return DetectionResult(
                is_failure=True,
                failure_type="nothing_happens",
                reason=f"Action '{action}' resulted in 'Nothing happens'",
            )

        # Rule 2: Empty or whitespace observation
        if not obs:
            return DetectionResult(
                is_failure=True,
                failure_type="empty_obs",
                reason="Empty observation returned",
            )

        # Rule 3: Action loop — same action repeated
        if len(action_history) >= self.loop_window:
            recent = action_history[-self.loop_window:]
            if all(a == action for a in recent):
                return DetectionResult(
                    is_failure=True,
                    failure_type="action_loop",
                    reason=f"Same action '{action}' repeated {self.loop_window} times",
                )

        # Rule 4: LLM judge — intent-outcome mismatch (only if rules didn't trigger)
        if self.judge_llm is not None:
            return self._judge_intent(action, observation)

        return DetectionResult(is_failure=False)

    def _judge_intent(self, action: str, observation: str) -> DetectionResult:
        """Use a lightweight LLM to judge if action achieved its intent."""
        prompt = JUDGE_PROMPT.format(action=action, observation=observation)
        try:
            response = self.judge_llm.complete_text(prompt, label="judge")
            answer = response.strip().lower()
            if answer.startswith("no"):
                return DetectionResult(
                    is_failure=True,
                    failure_type="intent_mismatch",
                    reason=f"Judge: action '{action}' did not achieve intent (obs: '{observation[:80]}')",
                )
        except Exception as e:
            logger.warning(f"Judge LLM call failed: {e}")
        return DetectionResult(is_failure=False)

    def is_task_complete(self, observation: str, done: bool, info: dict) -> tuple[bool, bool]:
        """Check if episode is over and whether it succeeded.
        Returns: (is_done, is_success)
        """
        won = info.get("won", [False])
        if isinstance(won, list):
            won = won[0]
        return done, bool(won)

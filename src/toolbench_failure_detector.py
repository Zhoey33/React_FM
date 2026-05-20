"""Rule-based + LLM judge failure detection for ToolBench."""

import logging
from dataclasses import dataclass

from src.llm import LLMClient

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are judging whether an agent's API call was productive in a tool-use task.

Action: {action}
Observation: {observation}

An interaction is PRODUCTIVE if the agent:
- Called a relevant API with correct parameters
- Got useful data that helps answer the user's question
- Made progress toward the final answer

An interaction is UNPRODUCTIVE if:
- The API returned an error
- The parameters were wrong or missing
- The response was empty or irrelevant
- The agent called the wrong API for the task

Was this interaction productive? Answer only "yes" or "no"."""


@dataclass
class DetectionResult:
    is_failure: bool
    failure_type: str = ""
    reason: str = ""


class ToolBenchFailureDetector:
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

        # Rule 1: API error
        if obs.startswith("api error") or "error" in obs[:50]:
            return DetectionResult(
                is_failure=True,
                failure_type="api_error",
                reason=f"API call failed: {observation[:100]}",
            )

        # Rule 2: No cache / call failed
        if "no cached response" in obs or "api call to" in obs and "failed" in obs:
            return DetectionResult(
                is_failure=True,
                failure_type="no_cache",
                reason=f"No cached response for: {action[:80]}",
            )

        # Rule 3: Unknown function
        if "unknown function" in obs:
            return DetectionResult(
                is_failure=True,
                failure_type="unknown_function",
                reason=f"Unknown function called: {action[:80]}",
            )

        # Rule 4: Invalid action format
        if "invalid action" in obs:
            return DetectionResult(
                is_failure=True,
                failure_type="invalid_action",
                reason=f"Invalid action format: {action[:80]}",
            )

        # Rule 5: Action loop
        if len(action_history) >= self.loop_window:
            recent = action_history[-self.loop_window:]
            if all(a == action for a in recent):
                return DetectionResult(
                    is_failure=True,
                    failure_type="action_loop",
                    reason=f"Same action repeated {self.loop_window} times",
                )

        # Rule 6: LLM judge
        if self.judge_llm is not None:
            return self._judge_intent(action, observation)

        return DetectionResult(is_failure=False)

    def _judge_intent(self, action: str, observation: str) -> DetectionResult:
        prompt = JUDGE_PROMPT.format(action=action, observation=observation[:500])
        try:
            response = self.judge_llm.complete_text(prompt, label="judge")
            answer = response.strip().lower()
            if answer.startswith("no"):
                return DetectionResult(
                    is_failure=True,
                    failure_type="unproductive",
                    reason="Judge: interaction not productive",
                )
        except Exception as e:
            logger.warning(f"Judge LLM call failed: {e}")
        return DetectionResult(is_failure=False)

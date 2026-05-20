"""Rule-based + LLM judge failure detection for HotPotQA."""

import logging
from dataclasses import dataclass

from src.llm import LLMClient

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are judging whether an agent's interaction was productive in a question-answering task.

Action: {action}
Observation: {observation}

An interaction is PRODUCTIVE if the agent:
- Found relevant information about the question topic
- Discovered useful facts that help answer the question
- Made progress toward finding the answer

An interaction is UNPRODUCTIVE if:
- The search returned no relevant results or the wrong entity
- The lookup found nothing useful
- The agent is searching for the same thing repeatedly
- The agent is going in circles without making progress

Was this interaction productive? Answer only "yes" or "no"."""


@dataclass
class DetectionResult:
    is_failure: bool
    failure_type: str = ""
    reason: str = ""


class HotPotQAFailureDetector:
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

        # Rule 1: Search not found
        if "could not find" in obs:
            return DetectionResult(
                is_failure=True,
                failure_type="search_not_found",
                reason=f"Search '{action}' returned no results",
            )

        # Rule 2: No more lookup results
        if obs == "no more results.":
            return DetectionResult(
                is_failure=True,
                failure_type="lookup_exhausted",
                reason=f"Lookup exhausted for '{action}'",
            )

        # Rule 3: Invalid action
        if obs.startswith("invalid action"):
            return DetectionResult(
                is_failure=True,
                failure_type="invalid_action",
                reason=f"Invalid action: {action}",
            )

        # Rule 4: Action loop
        if len(action_history) >= self.loop_window:
            recent = action_history[-self.loop_window:]
            if all(a == action for a in recent):
                return DetectionResult(
                    is_failure=True,
                    failure_type="action_loop",
                    reason=f"Same action '{action}' repeated {self.loop_window} times",
                )

        # Rule 5: LLM judge
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

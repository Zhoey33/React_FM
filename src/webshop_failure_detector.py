"""Rule-based + LLM judge failure detection for WebShop."""

import logging
from dataclasses import dataclass

from src.llm import LLMClient

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are judging whether an agent's interaction was productive in an online shopping environment.

Action: {action}
Observation: {observation}

An interaction is PRODUCTIVE if the agent:
- Found relevant products matching the shopping goal
- Narrowed down options by selecting appropriate attributes
- Made progress toward purchasing the correct item
- Is exploring a plausible product for the first time
- Is selecting an option that could satisfy the instruction

An interaction is UNPRODUCTIVE if:
- The search returned no relevant results
- The agent clicked an irrelevant product
- The agent selected wrong options (wrong color, size, etc.)
- The agent is going in circles without making progress, such as repeating
  search -> same item -> back to search
- The agent returns to search after inspecting the same unsuitable product
  multiple times

Do NOT mark normal exploration as unproductive. A first time search, first time
product click, or first time plausible option selection is productive unless the
observation clearly shows it cannot satisfy the shopping instruction.

Was this interaction productive? Answer only "yes" or "no"."""


@dataclass
class DetectionResult:
    is_failure: bool
    failure_type: str = ""
    reason: str = ""


class WebShopFailureDetector:
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

        # Rule 1: No results found
        if "total results: 0" in obs or "no results found" in obs:
            return DetectionResult(
                is_failure=True,
                failure_type="no_results",
                reason=f"Search '{action}' returned no results",
            )

        # Rule 2: Action loop
        if len(action_history) >= self.loop_window:
            recent = action_history[-self.loop_window:]
            if all(a == action for a in recent):
                return DetectionResult(
                    is_failure=True,
                    failure_type="action_loop",
                    reason=f"Same action '{action}' repeated {self.loop_window} times",
                )
            repeated_pattern_len = self._repeated_pattern_length(action_history)
            if repeated_pattern_len is not None:
                return DetectionResult(
                    is_failure=True,
                    failure_type="action_loop",
                    reason=f"Repeated action pattern of length {repeated_pattern_len}",
                )

        # Rule 3: Invalid action
        if "invalid action" in obs or "error" in obs[:50]:
            return DetectionResult(
                is_failure=True,
                failure_type="invalid_action",
                reason=f"Invalid action: {action}",
            )

        # Rule 4: LLM judge
        if self.judge_llm is not None:
            return self._judge_intent(action, observation)

        return DetectionResult(is_failure=False)

    def _repeated_pattern_length(self, action_history: list[str]) -> int | None:
        """Detect short repeated WebShop action patterns, e.g. search-item-back."""
        max_pattern = min(self.loop_window, len(action_history) // 2)
        for pattern_len in range(2, max_pattern + 1):
            window = action_history[-pattern_len * 2:]
            if window[:pattern_len] == window[pattern_len:]:
                return pattern_len
        return None

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

    def is_task_complete(self, observation: str, done: bool, info: dict) -> tuple[bool, bool]:
        """Check if episode is over.
        WebShop: done when 'Buy Now' is clicked. Success = reward > 0.
        We check done flag and consider reward > 0.5 as success.
        """
        return done, done  # In WebShop, if done, agent made a purchase

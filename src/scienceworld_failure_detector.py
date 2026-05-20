"""Rule-based + LLM judge failure detection for ScienceWorld."""

import logging
from dataclasses import dataclass

from src.llm import LLMClient

logger = logging.getLogger(__name__)


_EXPLICIT_FAILURE_PATTERNS: list[tuple[str, str]] = [
    ("ambiguous request", "ambiguity"),
    ("please enter the number for the action you intended", "ambiguity"),
    ("the door is not open", "precondition_blocked"),
    ("you don't have", "precondition_blocked"),
    ("you can't", "physics_or_affordance"),
    ("that doesn't seem to work", "no_effect_or_other"),
    ("nothing happens", "no_effect_or_other"),
    ("i'm not sure what you mean", "syntax_or_parse"),
]

JUDGE_PROMPT = """You are judging whether an agent's interaction was productive in a science experiment simulation.

Action: {action}
Observation: {observation}

An interaction is PRODUCTIVE if the agent:
- Gained useful new information about objects, locations, or experiment state
- Changed the environment state (moved, picked up, activated, mixed, heated, etc.)
- Made progress toward the experiment goal

An interaction is UNPRODUCTIVE if:
- The action was not recognized ("No known action matches that input")
- Nothing changed and no new information was gained
- The agent repeated an already-completed step with no effect

Was this interaction productive? Answer only "yes" or "no"."""


@dataclass
class DetectionResult:
    is_failure: bool
    failure_type: str = ""
    reason: str = ""


class ScienceWorldFailureDetector:
    def __init__(
        self,
        loop_window: int = 3,
        judge_llm: LLMClient | None = None,
        enable_implicit_failures: bool = False,
    ):
        self.loop_window = loop_window
        self.judge_llm = judge_llm
        self.enable_implicit_failures = enable_implicit_failures

    def detect(
        self,
        observation: str,
        action: str,
        action_history: list[str],
    ) -> DetectionResult:
        obs = observation.strip().lower()

        # Rule 1: Unknown action
        if "no known action matches that input" in obs:
            return DetectionResult(
                is_failure=True,
                failure_type="syntax_or_parse",
                reason=f"Action '{action}' not recognized by environment",
            )

        # Rule 2: Empty observation
        if not obs:
            return DetectionResult(
                is_failure=True,
                failure_type="empty_obs",
                reason="Empty observation returned",
            )

        # Rule 3: Action loop
        if len(action_history) >= self.loop_window:
            recent = action_history[-self.loop_window:]
            if all(a == action for a in recent):
                return DetectionResult(
                    is_failure=True,
                    failure_type="action_loop",
                    reason=f"Same action '{action}' repeated {self.loop_window} times",
                )

        # Rule 4: Explicit failure indicators
        for phrase, failure_type in _EXPLICIT_FAILURE_PATTERNS:
            if phrase in obs:
                return DetectionResult(
                    is_failure=True,
                    failure_type=failure_type,
                    reason=f"Observation contains failure indicator: '{phrase}'",
                )

        # Rule 5: LLM judge
        if self.enable_implicit_failures and self.judge_llm is not None:
            return self._judge_intent(action, observation)

        return DetectionResult(is_failure=False)

    def _judge_intent(self, action: str, observation: str) -> DetectionResult:
        prompt = JUDGE_PROMPT.format(action=action, observation=observation)
        try:
            response = self.judge_llm.complete_text(prompt, label="judge")
            answer = response.strip().lower()
            if answer.startswith("no"):
                return DetectionResult(
                    is_failure=True,
                    failure_type="unproductive",
                    reason=f"Judge: interaction not productive",
                )
        except Exception as e:
            logger.warning(f"Judge LLM call failed: {e}")
        return DetectionResult(is_failure=False)

    def is_task_complete(self, observation: str, done: bool, info: dict) -> tuple[bool, bool]:
        """Check if episode is over and whether it succeeded.
        ScienceWorld API returns score as round(100 * getScore()).
        Full success is 100, but terminal failures can produce negative scores.
        """
        score = info.get("score", 0.0)
        is_success = score >= 100.0
        return done, is_success

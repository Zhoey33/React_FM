"""ReAct / React_FM agent for ALFWorld."""

import logging
import time
from dataclasses import dataclass, field

from src.llm import LLMClient
from src.memory import FailureMemoryStore
from src.failure_detector import ALFWorldFailureDetector, DetectionResult
from src.memory_extractor import extract_failure_recoveries
from prompts.alfworld_prompts import build_user_prompt

logger = logging.getLogger(__name__)


@dataclass
class StepRecord:
    step: int
    action: str
    observation: str
    is_think: bool = False
    failure_detected: bool = False
    failure_type: str = ""
    memory_retrieved: int = 0


@dataclass
class EpisodeResult:
    env_idx: int
    task_type: str
    task_description: str
    success: bool
    steps: list[StepRecord] = field(default_factory=list)
    total_steps: int = 0
    total_tokens: int = 0
    agent_tokens: int = 0
    judge_tokens: int = 0
    extractor_tokens: int = 0
    failures_detected: int = 0
    memories_retrieved: int = 0
    memories_stored: int = 0
    wall_time_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "env_idx": self.env_idx,
            "task_type": self.task_type,
            "task_description": self.task_description[:200],
            "success": self.success,
            "total_steps": self.total_steps,
            "total_tokens": self.total_tokens,
            "agent_tokens": self.agent_tokens,
            "judge_tokens": self.judge_tokens,
            "extractor_tokens": self.extractor_tokens,
            "failures_detected": self.failures_detected,
            "memories_retrieved": self.memories_retrieved,
            "memories_stored": self.memories_stored,
            "wall_time_s": round(self.wall_time_s, 2),
            "steps": [
                {
                    "step": s.step,
                    "action": s.action,
                    "observation": s.observation[:200],
                    "is_think": s.is_think,
                    "failure_detected": s.failure_detected,
                    "memory_retrieved": s.memory_retrieved,
                }
                for s in self.steps
            ],
        }


class ReactFMAgent:
    def __init__(
        self,
        llm: LLMClient,
        memory_store: FailureMemoryStore | None = None,
        failure_detector: ALFWorldFailureDetector | None = None,
        extractor_llm: LLMClient | None = None,
        max_steps: int = 50,
        max_memory_inject: int = 3,
        enable_memory: bool = True,
        inject_mode: str = "in_loop",
    ):
        self.llm = llm
        self.memory = memory_store
        self.detector = failure_detector or ALFWorldFailureDetector()
        self.extractor_llm = extractor_llm
        self.max_steps = max_steps
        self.max_memory_inject = max_memory_inject
        self.enable_memory = enable_memory and (memory_store is not None)
        self.inject_mode = inject_mode  # "in_loop" or "episode"

    def run_episode(self, env, env_idx: int = 0) -> EpisodeResult:
        """Run one ALFWorld episode."""
        t0 = time.time()
        self.llm.tracker.reset()
        # Reset judge/extractor trackers for per-episode stats
        if self.detector.judge_llm is not None:
            self.detector.judge_llm.tracker.reset()
        if self.extractor_llm is not None:
            self.extractor_llm.tracker.reset()

        # Reset environment
        init_obs, task_type, info = env.reset()

        history: list[tuple[str, str]] = []  # (action, observation)
        action_history: list[str] = []
        steps: list[StepRecord] = []
        success = False

        # Track retrieved memories for prompt injection
        current_retrieved: list | None = None
        failures_detected = 0
        memories_retrieved_total = 0

        # Episode-level injection: retrieve all memories at start
        episode_memories: list | None = None
        if self.enable_memory and self.inject_mode == "episode":
            all_mem = self.memory.get_all(env_idx=env_idx)
            if all_mem:
                episode_memories = all_mem[:self.max_memory_inject]
                memories_retrieved_total = len(episode_memories)
                logger.info(f"  Episode-level: injecting {len(episode_memories)} memories for env {env_idx}")

        for step_num in range(self.max_steps):
            # Build prompt
            if self.inject_mode == "episode":
                # Episode mode: inject episode_memories every step
                prompt = build_user_prompt(
                    task_type=task_type,
                    task_obs=init_obs,
                    history=history,
                    retrieved_memories=episode_memories,
                )
            else:
                # In-loop mode: inject current_retrieved (set on failure)
                prompt = build_user_prompt(
                    task_type=task_type,
                    task_obs=init_obs,
                    history=history,
                    retrieved_memories=current_retrieved,
                )
            response = self.llm.complete_text(
                prompt, stop=["\n"], label=f"step_{step_num}"
            )
            action = response.strip().split("\n")[0].strip()

            # Clear retrieved memories after use (in-loop only)
            if self.inject_mode == "in_loop":
                current_retrieved = None

            # Strip "> " prefix if LLM outputs in Reflexion format
            if action.startswith("> "):
                action = action[2:]

            if not action:
                action = "look"

            logger.info(f"  Step {step_num}: {action[:80]}")

            # Handle think action (same logic as Reflexion/Baseline)
            is_think = action.startswith("think:") or action.startswith("think ")
            if is_think:
                observation = "OK."
                record = StepRecord(
                    step=step_num, action=action, observation=observation, is_think=True
                )
                steps.append(record)
                history.append((action, observation))
                continue

            # Execute action in environment
            observation, reward, done, step_info = env.step(action)
            logger.info(f"    obs: {observation[:80]}")
            action_history.append(action)

            # Check task completion
            is_done, is_success = self.detector.is_task_complete(observation, done, step_info)

            record = StepRecord(step=step_num, action=action, observation=observation)

            # Failure detection + in-loop retrieval (for in_loop and none modes)
            if self.enable_memory and self.inject_mode in ("in_loop", "none") and not is_done:
                det = self.detector.detect(observation, action, action_history)
                if det.is_failure:
                    record.failure_detected = True
                    record.failure_type = det.failure_type
                    failures_detected += 1
                    logger.info(f"    FAILURE detected: {det.failure_type}")

                    # Retrieve memories for next prompt (per-env, skip for "none" mode)
                    if self.inject_mode == "in_loop":
                        retrieved = self.memory.retrieve(
                            query_action=action,
                            query_observation=observation,
                            task_type=task_type,
                            top_k=self.max_memory_inject,
                            env_idx=env_idx,
                        )
                        record.memory_retrieved = len(retrieved)
                        memories_retrieved_total += len(retrieved)

                        if retrieved:
                            current_retrieved = retrieved
                            logger.info(f"    Retrieved {len(retrieved)} memories")

            steps.append(record)
            history.append((action, observation))

            if is_done:
                success = is_success
                break

        # Post-episode: extract failure-recovery pairs and store in memory
        memories_stored = 0
        if self.enable_memory and self.extractor_llm is not None:
            logger.info(f"  Extracting failure-recovery pairs...")
            memories_stored = self._extract_and_store(history, task_type, env_idx)
            logger.info(f"  Extracted and stored {memories_stored} memories")

        wall_time = time.time() - t0

        # Token stats — separate agent / judge / extractor
        agent_stats = self.llm.tracker.summary()
        judge_tokens = 0
        if self.detector.judge_llm is not None:
            judge_tokens = self.detector.judge_llm.tracker.summary()["total_tokens"]
        extractor_tokens = 0
        if self.extractor_llm is not None:
            extractor_tokens = self.extractor_llm.tracker.summary()["total_tokens"]
        total_tokens = agent_stats["total_tokens"] + judge_tokens + extractor_tokens

        result = EpisodeResult(
            env_idx=env_idx,
            task_type=task_type,
            task_description=init_obs[:200],
            success=success,
            steps=steps,
            total_steps=len(steps),
            total_tokens=total_tokens,
            agent_tokens=agent_stats["total_tokens"],
            judge_tokens=judge_tokens,
            extractor_tokens=extractor_tokens,
            failures_detected=failures_detected,
            memories_retrieved=memories_retrieved_total,
            memories_stored=memories_stored,
            wall_time_s=wall_time,
        )

        status = "SUCCESS" if success else "FAIL"
        logger.info(
            f"Env #{env_idx} [{task_type}] {status} in {len(steps)} steps, "
            f"tokens={total_tokens} (agent={agent_stats['total_tokens']}, judge={judge_tokens}, ext={extractor_tokens}), "
            f"failures={failures_detected}, mem_retrieved={memories_retrieved_total}, mem_stored={memories_stored}"
        )
        return result

    def _extract_and_store(self, history: list[tuple[str, str]], task_type: str, env_idx: int = 0) -> int:
        """Extract failure-recovery pairs from trajectory and store in memory."""
        # Filter to only valid env actions — exclude think and invalid LLM outputs
        _VALID_PREFIXES = ("go to", "take", "put", "open", "close", "toggle",
                           "clean", "cool", "heat", "use", "examine", "look", "inventory")
        env_history = [
            (a, o) for a, o in history
            if a.lower().startswith(_VALID_PREFIXES)
        ]
        recoveries = extract_failure_recoveries(self.extractor_llm, env_history)
        stored = 0
        for rec in recoveries:
            self.memory.add(
                failure_action=rec["failure_action"],
                failure_observation=rec["failure_observation"],
                solution_action=rec["solution_action"],
                task_type=task_type,
                env_idx=env_idx,
            )
            stored += 1
        return stored

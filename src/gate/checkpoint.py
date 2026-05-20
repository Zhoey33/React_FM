"""Checkpoint infrastructure for intervention gate experiments.

Captures agent state at failure detection points and supports
branched rollouts from those checkpoints.
"""

import copy
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """Snapshot of agent + environment state at a failure detection point.

    Captures everything needed to:
    1. Extract gate features (11-dim vector)
    2. Resume the episode from this point with an injected arm
    3. Run forward N steps and measure progress
    """
    # Identity
    checkpoint_id: str = ""
    env_idx: int = 0
    task_type: str = ""
    benchmark: str = ""  # "alfworld", "webshop", "scienceworld"

    # Position in episode
    step_idx: int = 0
    max_steps: int = 50

    # Failure info
    failure_type: str = ""       # "nothing_happens", "action_loop", etc.
    failure_action: str = ""
    failure_observation: str = ""

    # Agent state
    history: list[tuple[str, str]] = field(default_factory=list)  # (action, obs) pairs
    action_history: list[str] = field(default_factory=list)       # action-only list
    init_obs: str = ""           # initial observation for prompt building

    # Retrieval state
    retrieval_rrf_score: float = 0.0   # top-1 RRF
    retrieval_margin: float = 0.0      # top-1 - top-2 RRF
    retrieved_memory_id: int = -1      # which memory entry was top-1
    memory_entry_count: int = 0        # total entries available at this point

    # Progress tracking (for utility computation)
    score_at_checkpoint: float = 0.0   # benchmark-native score at this point
    failures_so_far: int = 0           # total failures before this one

    # Environment state (benchmark-specific, for restoring)
    env_state: dict = field(default_factory=dict)

    # Metadata
    created_at: str = ""
    episode_success: bool | None = None  # filled post-episode

    def to_dict(self) -> dict:
        """Serialize for JSON storage."""
        d = {
            "checkpoint_id": self.checkpoint_id,
            "env_idx": self.env_idx,
            "task_type": self.task_type,
            "benchmark": self.benchmark,
            "step_idx": self.step_idx,
            "max_steps": self.max_steps,
            "failure_type": self.failure_type,
            "failure_action": self.failure_action,
            "failure_observation": self.failure_observation,
            "history": self.history,
            "action_history": self.action_history,
            "init_obs": self.init_obs[:500],  # truncate for storage
            "retrieval_rrf_score": self.retrieval_rrf_score,
            "retrieval_margin": self.retrieval_margin,
            "retrieved_memory_id": self.retrieved_memory_id,
            "memory_entry_count": self.memory_entry_count,
            "score_at_checkpoint": self.score_at_checkpoint,
            "failures_so_far": self.failures_so_far,
            "env_state": self.env_state,
            "created_at": self.created_at,
            "episode_success": self.episode_success,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CheckpointState":
        """Deserialize from JSON."""
        # Convert history from list-of-lists to list-of-tuples
        history = [tuple(h) for h in d.get("history", [])]
        return cls(
            checkpoint_id=d.get("checkpoint_id", ""),
            env_idx=d.get("env_idx", 0),
            task_type=d.get("task_type", ""),
            benchmark=d.get("benchmark", ""),
            step_idx=d.get("step_idx", 0),
            max_steps=d.get("max_steps", 50),
            failure_type=d.get("failure_type", ""),
            failure_action=d.get("failure_action", ""),
            failure_observation=d.get("failure_observation", ""),
            history=history,
            action_history=d.get("action_history", []),
            init_obs=d.get("init_obs", ""),
            retrieval_rrf_score=d.get("retrieval_rrf_score", 0.0),
            retrieval_margin=d.get("retrieval_margin", 0.0),
            retrieved_memory_id=d.get("retrieved_memory_id", -1),
            memory_entry_count=d.get("memory_entry_count", 0),
            score_at_checkpoint=d.get("score_at_checkpoint", 0.0),
            failures_so_far=d.get("failures_so_far", 0),
            env_state=d.get("env_state", {}),
            created_at=d.get("created_at", ""),
            episode_success=d.get("episode_success"),
        )

    @property
    def failure_signature(self) -> str:
        """Deduplication key: (failure_type, task_type, failure_action_pattern)."""
        # Normalize action to first 3 words for grouping
        action_words = self.failure_action.lower().split()[:3]
        action_key = " ".join(action_words)
        return f"{self.failure_type}|{self.task_type}|{action_key}"


class CheckpointStore:
    """Collection of checkpoints with save/load and deduplication."""

    def __init__(self):
        self.checkpoints: list[CheckpointState] = []
        self._next_id = 0

    def add(self, cp: CheckpointState) -> CheckpointState:
        """Add a checkpoint with auto-generated ID."""
        if not cp.checkpoint_id:
            cp.checkpoint_id = f"cp_{self._next_id:04d}"
        if not cp.created_at:
            cp.created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._next_id += 1
        self.checkpoints.append(cp)
        return cp

    def deduplicate(self, max_per_signature: int = 3) -> "CheckpointStore":
        """Return deduplicated store keeping at most max_per_signature per failure_signature."""
        sig_counts: dict[str, int] = {}
        deduped = CheckpointStore()
        deduped._next_id = self._next_id

        for cp in self.checkpoints:
            sig = cp.failure_signature
            count = sig_counts.get(sig, 0)
            if count < max_per_signature:
                deduped.checkpoints.append(cp)
                sig_counts[sig] = count + 1

        logger.info(
            f"Dedup: {len(self.checkpoints)} → {len(deduped.checkpoints)} checkpoints "
            f"({len(sig_counts)} unique signatures)"
        )
        return deduped

    def split(
        self,
        train_ratio: float = 0.7,
        seed: int = 42,
        group_by: str = "signature",
        stratify_by_task: bool = False,
    ) -> tuple["CheckpointStore", "CheckpointStore"]:
        """Split into train/test with optional grouping.

        Args:
            group_by:
                - "signature": split by failure_signature (legacy behavior)
                - "episode": keep checkpoints from the same episode together
                - "task_variation": keep checkpoints from the same task/variation together
            stratify_by_task:
                If True, perform the split independently inside each task_type bucket.
                Useful with grouped splits to avoid dropping whole tasks from test.
        """
        import random
        rng = random.Random(seed)

        def _group_key(cp: CheckpointState):
            if group_by == "signature":
                return cp.failure_signature
            if group_by == "episode":
                return (
                    cp.task_type,
                    cp.env_state.get("variation_idx"),
                    cp.env_idx,
                )
            if group_by == "task_variation":
                return (
                    cp.task_type,
                    cp.env_state.get("variation_idx"),
                )
            raise ValueError(f"Unknown split group_by={group_by}")

        def _pick_train_groups(group_keys: list[Any]) -> set[Any]:
            # Sort before shuffling so the seeded split is reproducible across
            # Python processes. Iterating a set directly is hash-order dependent.
            keys = sorted(group_keys, key=repr)
            rng.shuffle(keys)
            if len(keys) <= 1:
                return set(keys)
            split_idx = int(round(len(keys) * train_ratio))
            split_idx = max(1, min(len(keys) - 1, split_idx))
            return set(keys[:split_idx])

        def _pick_train_groups_balanced(task_checkpoints: list[CheckpointState]) -> set[Any]:
            """Pick grouped train keys whose checkpoint count is closest to target.

            This keeps grouped splits leak-free while making per-task train/test
            checkpoint counts much better balanced than a raw group-count split.
            """
            group_counts: dict[Any, int] = {}
            for cp in task_checkpoints:
                key = _group_key(cp)
                group_counts[key] = group_counts.get(key, 0) + 1

            items = sorted(group_counts.items(), key=lambda kv: repr(kv[0]))
            rng.shuffle(items)
            if len(items) <= 1:
                return {key for key, _ in items}

            total = sum(size for _, size in items)
            target = int(round(total * train_ratio))
            target = max(1, min(total - 1, target))
            desired_group_count = int(round(len(items) * train_ratio))
            desired_group_count = max(1, min(len(items) - 1, desired_group_count))

            # Subset-sum DP over small grouped buckets; keep one deterministic mask
            # per reachable sum. This is enough for our task-level grouped split.
            reachable: dict[int, int] = {0: 0}  # sum -> bitmask
            for idx, (_, size) in enumerate(items):
                current = list(reachable.items())
                for subtotal, mask in current:
                    new_total = subtotal + size
                    if new_total not in reachable:
                        reachable[new_total] = mask | (1 << idx)

            valid_sums = [s for s in reachable if 0 < s < total]
            best_sum = min(
                valid_sums,
                key=lambda s: (
                    abs(s - target),
                    abs(reachable[s].bit_count() - desired_group_count),
                    s,
                ),
            )
            best_mask = reachable[best_sum]
            return {
                key
                for idx, (key, _) in enumerate(items)
                if (best_mask >> idx) & 1
            }

        train_groups: set[Any] = set()
        if stratify_by_task:
            task_to_checkpoints: dict[str, list[CheckpointState]] = {}
            for cp in self.checkpoints:
                task_to_checkpoints.setdefault(cp.task_type, []).append(cp)
            for task_type in sorted(task_to_checkpoints):
                train_groups.update(_pick_train_groups_balanced(task_to_checkpoints[task_type]))
        else:
            all_groups = {_group_key(cp) for cp in self.checkpoints}
            train_groups = _pick_train_groups(all_groups)

        train_store = CheckpointStore()
        test_store = CheckpointStore()
        for cp in self.checkpoints:
            if _group_key(cp) in train_groups:
                train_store.checkpoints.append(cp)
            else:
                test_store.checkpoints.append(cp)

        logger.info(
            f"Split(group_by={group_by}, stratify_by_task={stratify_by_task}): "
            f"{len(train_store)} train, {len(test_store)} test"
        )
        return train_store, test_store

    def save(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "next_id": self._next_id,
            "checkpoints": [cp.to_dict() for cp in self.checkpoints],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.checkpoints)} checkpoints to {filepath}")

    def load(self, filepath: str) -> None:
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"Checkpoint file not found: {filepath}")
            return
        with open(path) as f:
            data = json.load(f)
        self._next_id = data.get("next_id", 0)
        self.checkpoints = [
            CheckpointState.from_dict(d) for d in data.get("checkpoints", [])
        ]
        logger.info(f"Loaded {len(self.checkpoints)} checkpoints from {filepath}")

    def __len__(self) -> int:
        return len(self.checkpoints)

    def filter_by_benchmark(self, benchmark: str) -> "CheckpointStore":
        """Return new store with only checkpoints from the given benchmark."""
        filtered = CheckpointStore()
        filtered._next_id = self._next_id
        filtered.checkpoints = [cp for cp in self.checkpoints if cp.benchmark == benchmark]
        return filtered

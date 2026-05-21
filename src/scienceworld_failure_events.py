"""Build ScienceWorld failure-event records for recovered@k analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _score_at(steps: list[dict[str, Any]], failure_index: int, offset: int) -> float | None:
    target_index = failure_index + offset
    if target_index >= len(steps):
        return None
    return steps[target_index].get("score_after_action")


def _recovered_within(
    steps: list[dict[str, Any]],
    failure_index: int,
    *,
    score_before: float,
    window: int,
) -> bool:
    for offset in range(1, window + 1):
        score_after = _score_at(steps, failure_index, offset)
        if score_after is not None and score_after > score_before:
            return True
    return False


def build_failure_events(episode: dict[str, Any], *, memory_mode: str) -> list[dict[str, Any]]:
    """Convert one episode result into failure-event rows."""
    steps = episode.get("steps", [])
    events: list[dict[str, Any]] = []
    env_idx = int(episode.get("env_idx", 0))

    for index, step in enumerate(steps):
        if not step.get("failure_detected"):
            continue

        score_before = step.get("score_before_action", step.get("score_after_action", 0.0))
        score_after_1 = _score_at(steps, index, 1)
        score_after_2 = _score_at(steps, index, 2)
        score_after_3 = _score_at(steps, index, 3)
        retrieval_hit = bool(step.get("memory_retrieved", 0))

        events.append(
            {
                "episode_id": f"sw_{env_idx:03d}",
                "env_idx": env_idx,
                "task_type": episode.get("task_type", "unknown"),
                "variation_idx": episode.get("variation_idx"),
                "step": step.get("step"),
                "failure_type": step.get("failure_type", ""),
                "failed_action": step.get("action", ""),
                "failure_observation": step.get("observation", ""),
                "memory_mode": memory_mode,
                "retrieval_attempted": bool(step.get("retrieval_attempted", retrieval_hit)),
                "retrieval_hit": retrieval_hit,
                "retrieved_memory_ids": step.get("retrieved_memory_ids", []),
                "injected_memory_text": step.get("injected_memory_text", ""),
                "next_action": steps[index + 1].get("action", "") if index + 1 < len(steps) else "",
                "score_before_failure": score_before,
                "score_after_1_step": score_after_1,
                "score_after_2_steps": score_after_2,
                "score_after_3_steps": score_after_3,
                "recovered_within_1_step": _recovered_within(
                    steps, index, score_before=score_before, window=1
                ),
                "recovered_within_3_steps": _recovered_within(
                    steps, index, score_before=score_before, window=3
                ),
            }
        )

    return events


def write_failure_events_jsonl(
    episodes: list[dict[str, Any]],
    filepath: str | Path,
    *,
    memory_mode: str,
) -> int:
    """Write failure-event rows for a set of episodes and return row count."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for episode in episodes:
            for event in build_failure_events(episode, memory_mode=memory_mode):
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
                count += 1
    return count

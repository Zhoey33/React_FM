"""Shared ScienceWorld reporting helpers for protocol metadata and summaries."""

from __future__ import annotations

from typing import Any


def build_protocol_metadata(
    *,
    split: str,
    tasks: list[str],
    max_variations: int,
    step_limit: int,
    test_time_writable: bool,
    memory_scope: str | None = None,
    max_envs: int | None = None,
) -> dict[str, Any]:
    """Build the protocol block saved with every formal ScienceWorld result."""
    metadata: dict[str, Any] = {
        "split": split,
        "tasks": list(tasks),
        "max_variations": max_variations,
        "step_limit": step_limit,
        "test_time_writable": test_time_writable,
    }
    if memory_scope is not None:
        metadata["memory_scope"] = memory_scope
    if max_envs is not None:
        metadata["max_envs"] = max_envs
    return metadata


def _with_retrieval_observability(
    memory_stats: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Add gated ScienceWorld retrieval counters without mutating store stats."""
    stats = dict(memory_stats)
    if "total_retrievals" in memory_stats:
        stats["store_total_retrievals"] = int(memory_stats.get("total_retrievals") or 0)
    if "total_hits" in memory_stats:
        stats["store_total_hits"] = int(memory_stats.get("total_hits") or 0)
    attempted = 0
    injection_hits = 0
    injected_memories = 0
    candidate_hits_from_steps = 0

    for result in results:
        for step in result.get("steps", []):
            if step.get("retrieval_attempted"):
                attempted += 1
            candidate_count = int(step.get("retrieval_candidate_count") or 0)
            if candidate_count > 0:
                candidate_hits_from_steps += 1
            if "memory_injected" in step:
                memory_retrieved = int(step.get("memory_retrieved") or 0) if step.get("memory_injected") else 0
            else:
                memory_retrieved = int(step.get("memory_retrieved") or 0)
            if memory_retrieved > 0:
                injection_hits += 1
                injected_memories += memory_retrieved

    stats["candidate_retrievals"] = attempted
    stats["candidate_hits"] = candidate_hits_from_steps
    stats["in_loop_retrieval_attempts"] = attempted
    stats["injection_hits"] = injection_hits
    stats["injected_memories"] = injected_memories
    return stats


def compute_summary(
    results: list[dict[str, Any]],
    *,
    mode: str,
    memory_stats: dict[str, Any] | None = None,
    insight_stats: dict[str, Any] | None = None,
    protocol: dict[str, Any] | None = None,
    extra_tokens: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Compute ScienceWorld summary metrics with raw score as the only public score."""
    total = len(results)
    successes = sum(1 for r in results if r.get("success"))

    by_type: dict[str, dict[str, Any]] = {}
    for result in results:
        task_type = result.get("task_type", "unknown")
        entry = by_type.setdefault(task_type, {"total": 0, "success": 0})
        entry["total"] += 1
        if result.get("success"):
            entry["success"] += 1

    for entry in by_type.values():
        entry["rate"] = round(entry["success"] / entry["total"], 4) if entry["total"] else 0.0

    raw_scores = [float(r.get("score", 0.0)) for r in results]
    avg_raw_score = sum(raw_scores) / total if total else 0.0
    avg_steps = sum(int(r.get("total_steps", 0)) for r in results) / total if total else 0.0

    extras = extra_tokens or {}
    agent_tokens = sum(int(r.get("agent_tokens", r.get("total_tokens", 0))) for r in results)
    judge_tokens = sum(int(r.get("judge_tokens", 0)) for r in results) + int(extras.get("judge_tokens", 0))
    extractor_tokens = sum(int(r.get("extractor_tokens", 0)) for r in results) + int(
        extras.get("extractor_tokens", 0)
    )
    reflection_tokens = sum(int(r.get("reflection_tokens", 0)) for r in results) + int(
        extras.get("reflection_tokens", 0)
    )
    stored_total_tokens = sum(int(r.get("total_tokens", 0)) for r in results)
    extra_total_tokens = (
        int(extras.get("total_tokens", 0))
        or int(extras.get("judge_tokens", 0))
        + int(extras.get("extractor_tokens", 0))
        + int(extras.get("reflection_tokens", 0))
    )
    total_tokens = stored_total_tokens + extra_total_tokens

    summary: dict[str, Any] = {
        "mode": mode,
        "benchmark": "scienceworld",
        "total_envs": total,
        "total_success": successes,
        "success_rate": round(successes / total, 4) if total else 0.0,
        "avg_raw_score": round(avg_raw_score, 4),
        "avg_score": round(avg_raw_score, 4),
        "avg_steps_per_episode": round(avg_steps, 4),
        "by_task_type": by_type,
        "total_tokens": total_tokens,
        "agent_tokens": agent_tokens,
        "judge_tokens": judge_tokens,
        "extractor_tokens": extractor_tokens,
        "reflection_tokens": reflection_tokens,
        "avg_tokens_per_episode": round(total_tokens / total) if total else 0,
    }

    if protocol is not None:
        summary["protocol"] = protocol
    if memory_stats is not None:
        summary["memory_stats"] = _with_retrieval_observability(memory_stats, results)
    if insight_stats is not None:
        summary["insight_stats"] = insight_stats
        summary["avg_agent_tokens_per_episode"] = round(agent_tokens / total) if total else 0
        summary["avg_extractor_tokens_per_episode"] = round(extractor_tokens / total) if total else 0

    return summary

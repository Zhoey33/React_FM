"""Aggregate ScienceWorld failure-event JSONL logs into recovered@k statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                events.append(json.loads(stripped))
    return events


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_after(event: dict[str, Any], offset: int) -> float | None:
    for key in (f"score_after_{offset}_step", f"score_after_{offset}_steps"):
        if key in event:
            return _as_float(event.get(key))
    return None


def _score_deltas(event: dict[str, Any], *, window: int) -> list[float]:
    before = _as_float(event.get("score_before_failure"))
    if before is None:
        return []
    deltas: list[float] = []
    for offset in range(1, window + 1):
        after = _score_after(event, offset)
        if after is not None:
            deltas.append(after - before)
    return deltas


def _is_recovered(event: dict[str, Any], *, window: int) -> bool:
    deltas = _score_deltas(event, window=window)
    if deltas and max(deltas) > 0:
        return True

    for key in (f"recovered_within_{window}_step", f"recovered_within_{window}_steps"):
        if key in event:
            return bool(event[key])
    return False


def _score_delta(event: dict[str, Any], *, window: int) -> float | None:
    deltas = _score_deltas(event, window=window)
    if not deltas:
        return None
    return max(deltas)


def _summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(events)
    recovered_at_1 = sum(1 for event in events if _is_recovered(event, window=1))
    recovered_at_3 = sum(1 for event in events if _is_recovered(event, window=3))
    deltas_at_1 = [
        delta for event in events if (delta := _score_delta(event, window=1)) is not None
    ]
    deltas_at_3 = [
        delta for event in events if (delta := _score_delta(event, window=3)) is not None
    ]

    return {
        "total_events": total,
        "recovered_at_1": recovered_at_1,
        "recovered_at_3": recovered_at_3,
        "recovered_at_1_rate": round(recovered_at_1 / total, 4) if total else 0.0,
        "recovered_at_3_rate": round(recovered_at_3 / total, 4) if total else 0.0,
        "score_delta_at_1_count": len(deltas_at_1),
        "score_delta_at_3_count": len(deltas_at_3),
        "mean_score_delta_at_1": (
            round(sum(deltas_at_1) / len(deltas_at_1), 4) if deltas_at_1 else None
        ),
        "mean_score_delta_at_3": (
            round(sum(deltas_at_3) / len(deltas_at_3), 4) if deltas_at_3 else None
        ),
    }


def _group_by(events: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        key = str(event.get(field, "unknown")).lower() if field == "retrieval_hit" else str(
            event.get(field, "unknown")
        )
        grouped.setdefault(key, []).append(event)
    return {key: _summarize_events(grouped[key]) for key in sorted(grouped)}


def analyze_failure_event_files(paths: list[str | Path]) -> dict[str, Any]:
    """Read failure-event JSONL files and return recovered@1/@3 summary stats."""
    events: list[dict[str, Any]] = []
    for path in paths:
        events.extend(_read_jsonl(path))

    return {
        "source_files": [str(path) for path in paths],
        "overall": _summarize_events(events),
        "by_task_type": _group_by(events, "task_type"),
        "by_failure_type": _group_by(events, "failure_type"),
        "by_memory_mode": _group_by(events, "memory_mode"),
        "by_retrieval_hit": _group_by(events, "retrieval_hit"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute recovered@1 and recovered@3 from ScienceWorld failure-event JSONL logs."
    )
    parser.add_argument("failure_events_jsonl", nargs="+", help="Failure-event JSONL file(s)")
    parser.add_argument("--indent", type=int, default=2, help="JSON output indentation")
    args = parser.parse_args()

    summary = analyze_failure_event_files(args.failure_events_jsonl)
    print(json.dumps(summary, indent=args.indent, ensure_ascii=False))


if __name__ == "__main__":
    main()

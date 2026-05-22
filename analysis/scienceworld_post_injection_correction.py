"""Create and summarize ScienceWorld post-injection correction annotations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

BOOLEAN_FIELDS = (
    "gold_corrected_next_action",
    "gold_used_memory",
    "gold_injection_harmful",
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(payload: dict[str, Any], path: str | Path, *, indent: int) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_injection(event: dict[str, Any]) -> bool:
    retrieval_hit = _as_bool(event.get("retrieval_hit"))
    injected_text = str(event.get("injected_memory_text", "")).strip()
    judge_advice_injected = _as_bool(event.get("judge_advice_injected"))
    return bool(retrieval_hit) or bool(injected_text) or bool(judge_advice_injected)


def _event_to_annotation(*, source_file: str, event: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": "",
        "source_file": source_file,
        "episode_id": event.get("episode_id", ""),
        "env_idx": event.get("env_idx"),
        "task_type": event.get("task_type", "unknown"),
        "variation_idx": event.get("variation_idx"),
        "step": event.get("step"),
        "failure_type": event.get("failure_type", ""),
        "failed_action": event.get("failed_action", ""),
        "failure_observation": event.get("failure_observation", ""),
        "retrieved_memory_ids": event.get("retrieved_memory_ids", []),
        "retrieved_memory_scores": event.get("retrieved_memory_scores", []),
        "injected_memory_text": event.get("injected_memory_text", ""),
        "next_action": event.get("next_action", ""),
        "memory_mode": event.get("memory_mode", "unknown"),
        "memory_injected": _as_bool(event.get("memory_injected")),
        "retrieval_attempted": _as_bool(event.get("retrieval_attempted")),
        "retrieval_hit": _as_bool(event.get("retrieval_hit")),
        "retrieval_mode": event.get("retrieval_mode", ""),
        "retrieval_top_k": event.get("retrieval_top_k"),
        "retrieval_min_score": event.get("retrieval_min_score"),
        "retrieval_candidate_count": event.get("retrieval_candidate_count"),
        "retrieval_candidate_memory_ids": event.get("retrieval_candidate_memory_ids", []),
        "retrieval_candidate_scores": event.get("retrieval_candidate_scores", []),
        "retrieval_candidate_relevance_scores": event.get(
            "retrieval_candidate_relevance_scores", []
        ),
        "retrieval_selected_memory_id": event.get("retrieval_selected_memory_id"),
        "retrieval_relevance_decision": _as_bool(event.get("retrieval_relevance_decision")),
        "retrieval_rejection_reason": event.get("retrieval_rejection_reason", ""),
        "retrieval_filtered_by_type_count": event.get("retrieval_filtered_by_type_count", 0),
        "retrieval_filtered_by_safety_count": event.get(
            "retrieval_filtered_by_safety_count", 0
        ),
        "judge_advice_source": event.get("judge_advice_source", ""),
        "judge_advice_injected": _as_bool(event.get("judge_advice_injected")),
        "judge_repair_strategy": event.get("judge_repair_strategy", ""),
        "judge_repair_action": event.get("judge_repair_action", ""),
        "judge_repair_confidence": event.get("judge_repair_confidence"),
        "judge_repair_rationale": event.get("judge_repair_rationale", ""),
        "score_before_failure": event.get("score_before_failure"),
        "score_after_1_step": event.get("score_after_1_step"),
        "score_after_2_steps": event.get("score_after_2_steps"),
        "score_after_3_steps": event.get("score_after_3_steps"),
        "recovered_within_1_step": _as_bool(event.get("recovered_within_1_step")),
        "recovered_within_3_steps": _as_bool(event.get("recovered_within_3_steps")),
        "gold_corrected_next_action": None,
        "gold_used_memory": None,
        "gold_injection_harmful": None,
        "annotation_notes": "",
    }


def _collect_template_rows(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for event in _read_jsonl(path):
            if not _has_injection(event):
                continue
            rows.append(_event_to_annotation(source_file=str(path), event=event))
    return rows


def _balanced_sample_by_task(
    rows: list[dict[str, Any]],
    *,
    max_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    if max_samples < 1:
        return []
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("task_type") or "unknown"), []).append(row)
    for values in grouped.values():
        rng.shuffle(values)

    selected: list[dict[str, Any]] = []
    task_keys = sorted(grouped)
    while len(selected) < max_samples and any(grouped.values()):
        for task_key in task_keys:
            if grouped[task_key] and len(selected) < max_samples:
                selected.append(grouped[task_key].pop(0))
    rng.shuffle(selected)
    for index, row in enumerate(selected, start=1):
        row["sample_id"] = f"sw_post_injection_{index:04d}"
    return selected


def make_post_injection_template(
    failure_event_paths: list[str | Path],
    *,
    max_samples: int = 200,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Build post-injection correction annotation rows from failure-event JSONL files."""
    rows = _collect_template_rows(failure_event_paths)
    return _balanced_sample_by_task(rows, max_samples=max_samples, seed=seed)


def _score_after(row: dict[str, Any], offset: int) -> float | None:
    for key in (f"score_after_{offset}_step", f"score_after_{offset}_steps"):
        if key in row:
            return _as_float(row.get(key))
    return None


def _score_deltas(row: dict[str, Any], *, window: int) -> list[float]:
    before = _as_float(row.get("score_before_failure"))
    if before is None:
        return []
    deltas: list[float] = []
    for offset in range(1, window + 1):
        after = _score_after(row, offset)
        if after is not None:
            deltas.append(after - before)
    return deltas


def _score_delta(row: dict[str, Any], *, window: int) -> float | None:
    deltas = _score_deltas(row, window=window)
    if not deltas:
        return None
    return max(deltas)


def _score_recovered(row: dict[str, Any], *, window: int) -> bool:
    deltas = _score_deltas(row, window=window)
    if deltas:
        return max(deltas) > 0
    for key in (f"recovered_within_{window}_step", f"recovered_within_{window}_steps"):
        if key in row:
            return bool(_as_bool(row.get(key)))
    return False


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _bool_count(rows: list[dict[str, Any]], field: str) -> int:
    return sum(1 for row in rows if _as_bool(row.get(field)) is True)


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    corrected = _bool_count(rows, "gold_corrected_next_action")
    used_memory = _bool_count(rows, "gold_used_memory")
    harmful = _bool_count(rows, "gold_injection_harmful")
    recovered_at_1 = sum(1 for row in rows if _score_recovered(row, window=1))
    recovered_at_3 = sum(1 for row in rows if _score_recovered(row, window=3))
    deltas_at_1 = [
        delta for row in rows if (delta := _score_delta(row, window=1)) is not None
    ]
    deltas_at_3 = [
        delta for row in rows if (delta := _score_delta(row, window=3)) is not None
    ]

    return {
        "total_samples": total,
        "corrected_next_action_count": corrected,
        "corrected_next_action_rate": _safe_rate(corrected, total),
        "used_memory_count": used_memory,
        "used_memory_rate": _safe_rate(used_memory, total),
        "harmful_count": harmful,
        "harmful_rate": _safe_rate(harmful, total),
        "score_recovered_at_1_count": recovered_at_1,
        "score_recovered_at_1_rate": _safe_rate(recovered_at_1, total),
        "score_recovered_at_3_count": recovered_at_3,
        "score_recovered_at_3_rate": _safe_rate(recovered_at_3, total),
        "mean_score_delta_at_1": (
            round(sum(deltas_at_1) / len(deltas_at_1), 4) if deltas_at_1 else None
        ),
        "mean_score_delta_at_3": (
            round(sum(deltas_at_3) / len(deltas_at_3), 4) if deltas_at_3 else None
        ),
    }


def _group_key(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "unknown"
    return str(value)


def _group_by(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row.get(field, "unknown")), []).append(row)
    return {key: _summarize_rows(grouped[key]) for key in sorted(grouped)}


def _load_labeled_rows(
    annotation_paths: list[str | Path],
    *,
    allow_unlabeled: bool,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    skipped_unlabeled = 0
    for path in annotation_paths:
        for line_idx, row in enumerate(_read_jsonl(path), start=1):
            corrected = _as_bool(row.get("gold_corrected_next_action"))
            if corrected is None:
                if allow_unlabeled:
                    skipped_unlabeled += 1
                    continue
                raise ValueError(f"{path}:{line_idx} missing gold_corrected_next_action")
            normalized = dict(row)
            normalized["gold_corrected_next_action"] = corrected
            normalized["retrieval_hit"] = _as_bool(row.get("retrieval_hit"))
            for field in BOOLEAN_FIELDS:
                normalized[field] = _as_bool(row.get(field))
            rows.append(normalized)
    return rows, skipped_unlabeled


def summarize_post_injection_files(
    annotation_paths: list[str | Path],
    *,
    allow_unlabeled: bool = False,
) -> dict[str, Any]:
    """Summarize labeled ScienceWorld post-injection correction annotation files."""
    rows, skipped_unlabeled = _load_labeled_rows(
        annotation_paths,
        allow_unlabeled=allow_unlabeled,
    )
    return {
        "source_files": [str(path) for path in annotation_paths],
        "overall": _summarize_rows(rows),
        "by_task_type": _group_by(rows, "task_type"),
        "by_failure_type": _group_by(rows, "failure_type"),
        "by_memory_mode": _group_by(rows, "memory_mode"),
        "by_retrieval_hit": _group_by(rows, "retrieval_hit"),
        "by_judge_advice_source": _group_by(rows, "judge_advice_source"),
        "by_judge_advice_injected": _group_by(rows, "judge_advice_injected"),
        "by_gold_used_memory": _group_by(rows, "gold_used_memory"),
        "skipped_unlabeled": skipped_unlabeled,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create and summarize ScienceWorld post-injection correction annotations."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_parser = subparsers.add_parser("make-template")
    make_parser.add_argument("failure_events_jsonl", nargs="+", help="Failure-event JSONL file(s)")
    make_parser.add_argument("--output", required=True, help="Output annotation JSONL")
    make_parser.add_argument("--max-samples", type=int, default=200)
    make_parser.add_argument("--seed", type=int, default=0)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("annotation_jsonl", nargs="+", help="Annotation JSONL file(s)")
    summarize_parser.add_argument("--output", required=True, help="Output summary JSON")
    summarize_parser.add_argument("--allow-unlabeled", action="store_true")
    summarize_parser.add_argument("--indent", type=int, default=2)

    args = parser.parse_args()
    if args.command == "make-template":
        rows = make_post_injection_template(
            args.failure_events_jsonl,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        _write_jsonl(rows, args.output)
    elif args.command == "summarize":
        summary = summarize_post_injection_files(
            args.annotation_jsonl,
            allow_unlabeled=args.allow_unlabeled,
        )
        _write_json(summary, args.output, indent=args.indent)


if __name__ == "__main__":
    main()

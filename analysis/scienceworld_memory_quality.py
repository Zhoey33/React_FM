"""Create and summarize ScienceWorld memory generation quality annotations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

QUALITY_LABELS = (
    "high_quality",
    "usable_but_weak",
    "invalid",
    "duplicate_or_redundant",
)

BOOLEAN_FIELDS = (
    "failure_action_accurate",
    "failure_observation_has_evidence",
    "repair_strategy_reasonable",
    "repair_action_executable",
    "overly_state_bound",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


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


def _load_episode_index(result_paths: list[str | Path] | None) -> dict[int, dict[str, Any]]:
    index: dict[int, dict[str, Any]] = {}
    for path in result_paths or []:
        payload = _read_json(path)
        for episode in payload.get("episodes", []):
            if "env_idx" not in episode:
                continue
            index[int(episode["env_idx"])] = {
                "success": episode.get("success"),
                "score": episode.get("score"),
            }
    return index


def _resolve_store_buckets(payload: dict[str, Any]) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    raw_buckets = payload.get("buckets")
    if raw_buckets is not None:
        return payload.get("scope", "task_type"), raw_buckets
    if "task_types" in payload:
        return "task_type", payload.get("task_types", {})
    return "env_idx", payload.get("envs", {})


def _memory_entry_to_annotation(
    *,
    source_file: str,
    scope: str,
    bucket: str,
    entry: dict[str, Any],
    episode_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    env_idx = int(entry.get("env_idx", int(bucket) if scope == "env_idx" and str(bucket).isdigit() else 0))
    episode = episode_index.get(env_idx, {})
    return {
        "sample_id": "",
        "source_file": source_file,
        "memory_id": entry.get("memory_id"),
        "bucket": str(bucket),
        "scope": scope,
        "task_type": entry.get("task_type", ""),
        "env_idx": env_idx,
        "created_at": entry.get("created_at", ""),
        "failure_action": entry.get("failure_action", ""),
        "failure_observation": entry.get("failure_observation", ""),
        "solution_action": entry.get("solution_action", ""),
        "repair_strategy": entry.get("repair_strategy", ""),
        "repair_tactic": entry.get("repair_tactic", ""),
        "repair_action": entry.get("repair_action", ""),
        "question_text": entry.get("question_text", ""),
        "source_episode_success": episode.get("success"),
        "source_episode_score": episode.get("score"),
        "quality_label": "",
        "failure_action_accurate": None,
        "failure_observation_has_evidence": None,
        "repair_strategy_reasonable": None,
        "repair_action_executable": None,
        "overly_state_bound": None,
        "annotation_notes": "",
    }


def _collect_memory_rows(
    memory_paths: list[str | Path],
    *,
    episode_index: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in memory_paths:
        payload = _read_json(path)
        scope, buckets = _resolve_store_buckets(payload)
        for bucket, entries in buckets.items():
            for entry in entries:
                rows.append(
                    _memory_entry_to_annotation(
                        source_file=str(path),
                        scope=scope,
                        bucket=str(bucket),
                        entry=entry,
                        episode_index=episode_index,
                    )
                )
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
        row["sample_id"] = f"sw_memory_{index:04d}"
    return selected


def make_memory_quality_template(
    memory_paths: list[str | Path],
    *,
    result_paths: list[str | Path] | None = None,
    max_samples: int = 100,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Build memory quality annotation rows from ScienceWorld memory store JSON files."""
    episode_index = _load_episode_index(result_paths)
    rows = _collect_memory_rows(memory_paths, episode_index=episode_index)
    return _balanced_sample_by_task(rows, max_samples=max_samples, seed=seed)


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _quality_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {label: 0 for label in QUALITY_LABELS}
    for row in rows:
        label = row.get("quality_label", "")
        if label in counts:
            counts[label] += 1
    return counts


def _boolean_rate(rows: list[dict[str, Any]], field: str) -> float:
    values = [_as_bool(row.get(field)) for row in rows]
    labeled_values = [value for value in values if value is not None]
    if not labeled_values:
        return 0.0
    return round(sum(1 for value in labeled_values if value) / len(labeled_values), 4)


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    counts = _quality_counts(rows)
    summary: dict[str, Any] = {"total_memories": total}
    for label in QUALITY_LABELS:
        summary[f"{label}_count"] = counts[label]
        summary[f"{label}_rate"] = _safe_rate(counts[label], total)
    summary["usable_rate"] = _safe_rate(
        counts["high_quality"] + counts["usable_but_weak"], total
    )
    summary["invalid_rate"] = _safe_rate(counts["invalid"], total)
    summary["duplicate_rate"] = _safe_rate(counts["duplicate_or_redundant"], total)
    summary["failure_action_accuracy"] = _boolean_rate(rows, "failure_action_accurate")
    summary["failure_observation_evidence_rate"] = _boolean_rate(
        rows, "failure_observation_has_evidence"
    )
    summary["repair_strategy_reasonable_rate"] = _boolean_rate(
        rows, "repair_strategy_reasonable"
    )
    summary["repair_action_executable_rate"] = _boolean_rate(rows, "repair_action_executable")
    summary["overly_state_bound_rate"] = _boolean_rate(rows, "overly_state_bound")
    return summary


def _group_by(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = row.get(field, "unknown")
        key = str(value).lower() if isinstance(value, bool) else str(value)
        grouped.setdefault(key, []).append(row)
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
            label = str(row.get("quality_label", "")).strip()
            if not label:
                if allow_unlabeled:
                    skipped_unlabeled += 1
                    continue
                raise ValueError(f"{path}:{line_idx} missing quality_label")
            if label not in QUALITY_LABELS:
                raise ValueError(f"{path}:{line_idx} invalid quality_label: {label}")
            normalized = dict(row)
            normalized["quality_label"] = label
            for field in BOOLEAN_FIELDS:
                normalized[field] = _as_bool(row.get(field))
            rows.append(normalized)
    return rows, skipped_unlabeled


def summarize_memory_quality_files(
    annotation_paths: list[str | Path],
    *,
    allow_unlabeled: bool = False,
) -> dict[str, Any]:
    """Summarize labeled ScienceWorld memory quality annotation files."""
    rows, skipped_unlabeled = _load_labeled_rows(
        annotation_paths,
        allow_unlabeled=allow_unlabeled,
    )
    return {
        "source_files": [str(path) for path in annotation_paths],
        "overall": _summarize_rows(rows),
        "by_task_type": _group_by(rows, "task_type"),
        "by_quality_label": _group_by(rows, "quality_label"),
        "by_source_episode_success": _group_by(rows, "source_episode_success"),
        "skipped_unlabeled": skipped_unlabeled,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create and summarize ScienceWorld memory quality annotations."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_parser = subparsers.add_parser("make-template")
    make_parser.add_argument("memory_json", nargs="+", help="ScienceWorld memory JSON file(s)")
    make_parser.add_argument("--output", required=True, help="Output annotation JSONL")
    make_parser.add_argument("--max-samples", type=int, default=100)
    make_parser.add_argument("--seed", type=int, default=0)
    make_parser.add_argument(
        "--result-json",
        nargs="*",
        default=None,
        help="Optional ScienceWorld result JSON file(s) for source episode enrichment",
    )

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("annotation_jsonl", nargs="+", help="Annotation JSONL file(s)")
    summarize_parser.add_argument("--output", required=True, help="Output summary JSON")
    summarize_parser.add_argument("--allow-unlabeled", action="store_true")
    summarize_parser.add_argument("--indent", type=int, default=2)

    args = parser.parse_args()
    if args.command == "make-template":
        rows = make_memory_quality_template(
            args.memory_json,
            result_paths=args.result_json,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        _write_jsonl(rows, args.output)
    elif args.command == "summarize":
        summary = summarize_memory_quality_files(
            args.annotation_jsonl,
            allow_unlabeled=args.allow_unlabeled,
        )
        _write_json(summary, args.output, indent=args.indent)


if __name__ == "__main__":
    main()

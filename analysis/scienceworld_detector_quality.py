"""Create and summarize ScienceWorld detector quality annotation JSONL files."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


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


def _step_to_annotation(
    *,
    source_file: str,
    episode: dict[str, Any],
    step: dict[str, Any],
) -> dict[str, Any]:
    env_idx = int(episode.get("env_idx", 0))
    detector_prediction = bool(step.get("failure_detected", False))
    return {
        "sample_id": "",
        "source_file": source_file,
        "episode_id": f"sw_{env_idx:03d}",
        "env_idx": env_idx,
        "task_type": episode.get("task_type", "unknown"),
        "variation_idx": episode.get("variation_idx"),
        "step": step.get("step"),
        "action": step.get("action", ""),
        "observation": step.get("observation", ""),
        "score_before_action": step.get("score_before_action"),
        "score_after_action": step.get("score_after_action"),
        "detector_prediction": detector_prediction,
        "detector_failure_type": step.get("failure_type", "") if detector_prediction else "",
        "detector_source": step.get("detector_source", "") if detector_prediction else "",
        "failure_reason": step.get("failure_reason", "") if detector_prediction else "",
        "failure_confidence": step.get("failure_confidence") if detector_prediction else None,
        "gold_is_failure": None,
        "gold_failure_type": "",
        "gold_needs_repair": None,
        "annotation_notes": "",
    }


def _collect_template_candidates(
    result_paths: list[str | Path],
    *,
    include_think: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for path in result_paths:
        source_file = str(path)
        payload = _read_json(path)
        for episode in payload.get("episodes", []):
            for step in episode.get("steps", []):
                if step.get("is_think") and not include_think:
                    continue
                row = _step_to_annotation(
                    source_file=source_file,
                    episode=episode,
                    step=step,
                )
                if row["detector_prediction"]:
                    positives.append(row)
                else:
                    negatives.append(row)
    return positives, negatives


def _balanced_sample(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    *,
    max_samples: int,
    positive_ratio: float,
    seed: int,
) -> list[dict[str, Any]]:
    if max_samples < 1:
        return []
    rng = random.Random(seed)
    positives = list(positives)
    negatives = list(negatives)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    positive_target = round(max_samples * positive_ratio)
    negative_target = max_samples - positive_target
    selected = positives[: min(positive_target, len(positives))]
    selected.extend(negatives[: min(negative_target, len(negatives))])

    if len(selected) < max_samples:
        selected_ids = {id(row) for row in selected}
        remainder = [row for row in positives + negatives if id(row) not in selected_ids]
        selected.extend(remainder[: max_samples - len(selected)])

    rng.shuffle(selected)
    for index, row in enumerate(selected, start=1):
        row["sample_id"] = f"sw_detector_{index:04d}"
    return selected


def make_annotation_template(
    result_paths: list[str | Path],
    *,
    max_samples: int = 200,
    positive_ratio: float = 0.5,
    seed: int = 0,
    include_think: bool = False,
) -> list[dict[str, Any]]:
    """Build detector quality annotation rows from ScienceWorld result JSON files."""
    positives, negatives = _collect_template_candidates(
        result_paths,
        include_think=include_think,
    )
    return _balanced_sample(
        positives,
        negatives,
        max_samples=max_samples,
        positive_ratio=positive_ratio,
        seed=seed,
    )


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


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    type_correct = 0
    for row in rows:
        prediction = bool(row.get("detector_prediction", False))
        gold = bool(row.get("gold_is_failure", False))
        if prediction and gold:
            tp += 1
            if row.get("detector_failure_type", "") == row.get("gold_failure_type", ""):
                type_correct += 1
        elif prediction and not gold:
            fp += 1
        elif (not prediction) and gold:
            fn += 1
        else:
            tn += 1

    precision = _safe_rate(tp, tp + fp)
    recall = _safe_rate(tp, tp + fn)
    f1_denominator = (2 * tp) + fp + fn
    f1 = round((2 * tp) / f1_denominator, 4) if f1_denominator else 0.0
    return {
        "total_samples": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "type_accuracy": _safe_rate(type_correct, tp),
        "false_positive_rate": _safe_rate(fp, fp + tn),
        "false_negative_rate": _safe_rate(fn, fn + tp),
    }


def _group_by(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(field, "unknown"))
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
            gold = _as_bool(row.get("gold_is_failure"))
            if gold is None:
                if allow_unlabeled:
                    skipped_unlabeled += 1
                    continue
                raise ValueError(f"{path}:{line_idx} missing gold_is_failure")
            normalized = dict(row)
            normalized["detector_prediction"] = bool(_as_bool(row.get("detector_prediction")))
            normalized["gold_is_failure"] = gold
            rows.append(normalized)
    return rows, skipped_unlabeled


def summarize_annotation_files(
    annotation_paths: list[str | Path],
    *,
    allow_unlabeled: bool = False,
) -> dict[str, Any]:
    """Summarize labeled detector quality annotation JSONL files."""
    rows, skipped_unlabeled = _load_labeled_rows(
        annotation_paths,
        allow_unlabeled=allow_unlabeled,
    )
    return {
        "source_files": [str(path) for path in annotation_paths],
        "overall": _summarize_rows(rows),
        "by_task_type": _group_by(rows, "task_type"),
        "by_detector_source": _group_by(rows, "detector_source"),
        "by_detector_failure_type": _group_by(rows, "detector_failure_type"),
        "by_gold_failure_type": _group_by(rows, "gold_failure_type"),
        "skipped_unlabeled": skipped_unlabeled,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create and summarize ScienceWorld detector quality annotations."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_parser = subparsers.add_parser("make-template")
    make_parser.add_argument("result_json", nargs="+", help="ScienceWorld result JSON file(s)")
    make_parser.add_argument("--output", required=True, help="Output annotation JSONL")
    make_parser.add_argument("--max-samples", type=int, default=200)
    make_parser.add_argument("--positive-ratio", type=float, default=0.5)
    make_parser.add_argument("--seed", type=int, default=0)
    make_parser.add_argument("--include-think", action="store_true")

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("annotation_jsonl", nargs="+", help="Annotation JSONL file(s)")
    summarize_parser.add_argument("--output", required=True, help="Output summary JSON")
    summarize_parser.add_argument("--allow-unlabeled", action="store_true")
    summarize_parser.add_argument("--indent", type=int, default=2)

    args = parser.parse_args()
    if args.command == "make-template":
        rows = make_annotation_template(
            args.result_json,
            max_samples=args.max_samples,
            positive_ratio=args.positive_ratio,
            seed=args.seed,
            include_think=args.include_think,
        )
        _write_jsonl(rows, args.output)
    elif args.command == "summarize":
        summary = summarize_annotation_files(
            args.annotation_jsonl,
            allow_unlabeled=args.allow_unlabeled,
        )
        _write_json(summary, args.output, indent=args.indent)


if __name__ == "__main__":
    main()

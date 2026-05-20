"""Memory canonicalization pipeline for the intervention gate.

Takes existing FailureMemoryEntry entries and generates:
  - question_text: diagnostic question (doesn't reveal the answer)
  - repair_text: full corrective instruction

One LLM call per entry, ~100 tokens per call.
"""

import json
import logging
import time
from pathlib import Path

from src.llm import LLMClient
from src.memory import FailureMemoryStore, FailureMemoryEntry

logger = logging.getLogger(__name__)

CANONICALIZE_PROMPT = """Given this failure-recovery pair from an interactive agent:

Failed action: {failure_action}
Observation after failure: {failure_observation}
Correct solution: {solution_action}

Generate two things:
1. A diagnostic question that hints at the problem WITHOUT revealing the answer. The question should make the agent reconsider its approach.
2. A repair instruction that gives the full fix directly.

Output in this exact format (two lines only):
QUESTION: <your diagnostic question>
REPAIR: <your repair instruction>"""

CUE_TEXT = "Reconsider your approach."


# ── Tiered (v2) canonicalization ────────────────────────────────────────

TIERED_CANONICALIZE_PROMPT = """\
Given this failure-recovery pair from an interactive science simulation agent:

Task type: {task_type}
Failed action: {failure_action}
Observation after failure: {failure_observation}
Known correct approach: {solution_action}

Generate a THREE-LEVEL repair instruction, from high-level strategy to specific action fix:

STRATEGY: A 1-sentence high-level guidance about WHAT the agent should be trying to do and WHERE to go. Address the overall goal, not just the immediate error.
TACTIC: A 1-2 sentence specific plan for the next few steps.
ACTION: The exact corrective action(s) to take right now, using valid environment commands.

Also generate a diagnostic question that hints at the problem WITHOUT revealing the answer.

Output in this exact format (four lines only):
QUESTION: <diagnostic question>
STRATEGY: <high-level guidance>
TACTIC: <step-by-step plan>
ACTION: <exact command(s)>"""


def canonicalize_entry_tiered(
    llm: LLMClient,
    entry: FailureMemoryEntry,
) -> tuple[str, str, str, str]:
    """Generate question + tiered repair (strategy/tactic/action) for one entry.

    Returns:
        (question_text, strategy, tactic, action) tuple
    """
    prompt = TIERED_CANONICALIZE_PROMPT.format(
        task_type=entry.task_type or "unknown",
        failure_action=entry.failure_action,
        failure_observation=entry.failure_observation,
        solution_action=entry.solution_action,
    )

    try:
        response = llm.complete_text(prompt, label="canonicalize_tiered")
        lines = response.strip().split("\n")

        question_text = ""
        strategy = ""
        tactic = ""
        action = ""

        for line in lines:
            line = line.strip()
            up = line.upper()
            if up.startswith("QUESTION:"):
                question_text = line[len("QUESTION:"):].strip()
            elif up.startswith("STRATEGY:"):
                strategy = line[len("STRATEGY:"):].strip()
            elif up.startswith("TACTIC:"):
                tactic = line[len("TACTIC:"):].strip()
            elif up.startswith("ACTION:"):
                action = line[len("ACTION:"):].strip()

        # Fallbacks
        if not question_text:
            question_text = f"What went wrong when you tried '{entry.failure_action}'?"
        if not strategy:
            strategy = ""
        if not tactic:
            tactic = ""
        if not action:
            action = entry.solution_action

        return question_text, strategy, tactic, action

    except Exception as e:
        logger.warning(f"Tiered canonicalization failed for entry {entry.memory_id}: {e}")
        return (
            f"What went wrong when you tried '{entry.failure_action}'?",
            "",
            "",
            entry.solution_action,
        )


def canonicalize_store_tiered(
    llm: LLMClient,
    memory_store: FailureMemoryStore,
    skip_existing: bool = True,
    save_path: str | None = None,
    progress_interval: int = 10,
) -> int:
    """Canonicalize all entries with tiered repair (strategy/tactic/action).

    Also populates the legacy repair_text field by joining all three levels
    for backward compatibility with existing rollout code.
    """
    processed = 0
    skipped = 0
    total_entries = memory_store.size()

    logger.info(f"Tiered canonicalization: {total_entries} entries...")

    for env_idx, entries in memory_store._env_entries.items():
        for entry in entries:
            if skip_existing and entry.repair_strategy:
                skipped += 1
                continue

            question, strategy, tactic, action = canonicalize_entry_tiered(llm, entry)
            entry.question_text = question
            entry.repair_strategy = strategy
            entry.repair_tactic = tactic
            entry.repair_action = action

            # Build legacy repair_text from tiered fields (backward compat)
            parts = []
            if strategy:
                parts.append(f"[Strategy] {strategy}")
            if tactic:
                parts.append(f"[Plan] {tactic}")
            if action:
                parts.append(f"[Action] {action}")
            entry.repair_text = " ".join(parts) if parts else action

            processed += 1

            if processed % progress_interval == 0:
                logger.info(f"  Tiered canonicalized {processed}/{total_entries - skipped}")

    logger.info(f"Tiered canonicalization done: {processed} processed, {skipped} skipped")

    if save_path:
        memory_store.save(save_path)
        logger.info(f"Tiered store saved to {save_path}")

    return processed


def canonicalize_entry(
    llm: LLMClient,
    entry: FailureMemoryEntry,
) -> tuple[str, str]:
    """Generate question_text and repair_text for one memory entry.

    Returns:
        (question_text, repair_text) tuple
    """
    prompt = CANONICALIZE_PROMPT.format(
        failure_action=entry.failure_action,
        failure_observation=entry.failure_observation,
        solution_action=entry.solution_action,
    )

    try:
        response = llm.complete_text(prompt, label="canonicalize")
        lines = response.strip().split("\n")

        question_text = ""
        repair_text = ""

        for line in lines:
            line = line.strip()
            if line.upper().startswith("QUESTION:"):
                question_text = line[len("QUESTION:"):].strip()
            elif line.upper().startswith("REPAIR:"):
                repair_text = line[len("REPAIR:"):].strip()

        # Fallback if parsing fails
        if not question_text:
            question_text = f"What went wrong when you tried '{entry.failure_action}'?"
        if not repair_text:
            repair_text = entry.solution_action

        return question_text, repair_text

    except Exception as e:
        logger.warning(f"Canonicalization failed for entry {entry.memory_id}: {e}")
        # Safe fallback
        return (
            f"What went wrong when you tried '{entry.failure_action}'?",
            entry.solution_action,
        )


def canonicalize_store(
    llm: LLMClient,
    memory_store: FailureMemoryStore,
    skip_existing: bool = True,
    save_path: str | None = None,
    progress_interval: int = 10,
) -> int:
    """Canonicalize all entries in a memory store.

    Args:
        llm: LLM client for canonicalization calls
        memory_store: store to process (modified in place)
        skip_existing: skip entries that already have question_text
        save_path: if provided, save store after processing
        progress_interval: log progress every N entries

    Returns:
        Number of entries processed
    """
    processed = 0
    skipped = 0
    total_entries = memory_store.size()

    logger.info(f"Canonicalizing {total_entries} memory entries...")

    for env_idx, entries in memory_store._env_entries.items():
        for entry in entries:
            if skip_existing and entry.question_text:
                skipped += 1
                continue

            question, repair = canonicalize_entry(llm, entry)
            entry.question_text = question
            entry.repair_text = repair
            processed += 1

            if processed % progress_interval == 0:
                logger.info(f"  Canonicalized {processed}/{total_entries - skipped} entries")

    logger.info(
        f"Canonicalization complete: {processed} processed, {skipped} skipped"
    )

    if save_path:
        memory_store.save(save_path)
        logger.info(f"Canonicalized store saved to {save_path}")

    return processed


def canonicalize_memory_file(
    llm: LLMClient,
    input_path: str,
    output_path: str | None = None,
) -> int:
    """Canonicalize all entries in a memory file.

    Args:
        llm: LLM client
        input_path: path to memory store JSON
        output_path: path to save (defaults to input_path with _canonicalized suffix)

    Returns:
        Number of entries processed
    """
    store = FailureMemoryStore()
    store.load(input_path)

    if output_path is None:
        p = Path(input_path)
        output_path = str(p.with_stem(p.stem + "_canonicalized"))

    count = canonicalize_store(llm, store, save_path=output_path)
    return count

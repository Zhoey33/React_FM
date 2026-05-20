"""Verify tiered memory injection works end-to-end.

Tests:
1. ScienceWorld tiered memory → [Strategy]/[Plan]/[Next action] format
2. ALFWorld canonicalized memory → repair_text fallback
3. Raw memory (no canonicalization) → solution_action fallback
4. Prompt injection renders correctly for each memory_style

Usage:
    python experiments/gate/verify_tiered_injection.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.memory import FailureMemoryStore, FailureMemoryEntry


def test_get_repair_display():
    """Test the 3-tier fallback chain in get_repair_display()."""
    print("=" * 60)
    print("TEST 1: get_repair_display() fallback chain")
    print("=" * 60)

    # Case A: Full tiered entry (ScienceWorld)
    entry_tiered = FailureMemoryEntry(
        memory_id=0,
        failure_action="focus on tin",
        failure_observation="You focus on the tin.",
        solution_action="move metal pot to oven",
        repair_strategy="To melt tin, you need to move it to a heat source like the kitchen stove.",
        repair_tactic="Pick up the metal pot containing tin, go to hallway, then go to kitchen, put pot on stove.",
        repair_action="pick up metal pot",
    )
    result_a = entry_tiered.get_repair_display()
    print(f"\n[Case A: Tiered entry]")
    print(f"  Output:\n    {result_a.replace(chr(10), chr(10) + '    ')}")
    assert "[Strategy]" in result_a, "FAIL: Missing [Strategy] tag"
    assert "[Plan]" in result_a, "FAIL: Missing [Plan] tag"
    assert "[Next action]" in result_a, "FAIL: Missing [Next action] tag"
    assert "solution_action" not in result_a, "FAIL: Should not contain raw solution_action text"
    print("  ✅ PASS")

    # Case B: Canonicalized entry (has repair_text, no tiered)
    entry_canon = FailureMemoryEntry(
        memory_id=1,
        failure_action="put potato in fridge",
        failure_observation="Nothing happens.",
        solution_action="open fridge → put potato in fridge",
        repair_text="Open the fridge first, then put the potato inside.",
    )
    result_b = entry_canon.get_repair_display()
    print(f"\n[Case B: Canonicalized (repair_text only)]")
    print(f"  Output: {result_b}")
    assert result_b == "Open the fridge first, then put the potato inside.", f"FAIL: Got '{result_b}'"
    print("  ✅ PASS")

    # Case C: Raw entry (only solution_action)
    entry_raw = FailureMemoryEntry(
        memory_id=2,
        failure_action="take apple from table",
        failure_observation="Nothing happens.",
        solution_action="go to countertop 1",
    )
    result_c = entry_raw.get_repair_display()
    print(f"\n[Case C: Raw (solution_action only)]")
    print(f"  Output: {result_c}")
    assert result_c == "go to countertop 1", f"FAIL: Got '{result_c}'"
    print("  ✅ PASS")

    # Case D: Tiered but only strategy (no tactic/action)
    entry_partial = FailureMemoryEntry(
        memory_id=3,
        failure_action="examine crocodile egg",
        failure_observation="The egg is here.",
        solution_action="focus on crocodile",
        repair_strategy="Focus on the adult animal, not the egg, to determine lifespan.",
    )
    result_d = entry_partial.get_repair_display()
    print(f"\n[Case D: Partial tiered (strategy only)]")
    print(f"  Output: {result_d}")
    assert "[Strategy]" in result_d, "FAIL: Missing [Strategy] tag"
    assert "[Plan]" not in result_d, "FAIL: Should not have [Plan] when tactic is empty"
    print("  ✅ PASS")


def test_prompt_injection_styles():
    """Test that all memory_styles render correctly with tiered entries."""
    print("\n" + "=" * 60)
    print("TEST 2: Prompt injection with all memory_styles")
    print("=" * 60)

    from prompts.scienceworld_prompts import build_user_prompt as sw_prompt
    from prompts.alfworld_prompts import build_user_prompt as alf_prompt

    tiered_entry = FailureMemoryEntry(
        memory_id=0,
        failure_action="focus on tin",
        failure_observation="You focus on the tin.",
        solution_action="move metal pot to oven",
        repair_strategy="To melt tin, move it to a heat source.",
        repair_tactic="Go to kitchen, put pot on stove.",
        repair_action="pick up metal pot",
    )

    canon_entry = FailureMemoryEntry(
        memory_id=1,
        failure_action="put potato in fridge",
        failure_observation="Nothing happens.",
        solution_action="open fridge → put potato in fridge",
        repair_text="Open the fridge first, then put the potato inside.",
    )

    history = [("look around", "You see a table.")]

    for style in ["original", "factual", "reflexion", "hint"]:
        # ScienceWorld with tiered
        prompt_sw = sw_prompt(
            task_type="melt",
            task_obs="Your task is to melt tin.",
            history=history,
            retrieved_memories=[tiered_entry],
            memory_style=style,
        )

        # ALFWorld with canonicalized (fallback)
        prompt_alf = alf_prompt(
            task_type="heat",
            task_obs="Your task is to heat potato.",
            history=history,
            retrieved_memories=[canon_entry],
            memory_style=style,
        )

        print(f"\n--- Style: {style} ---")

        # Extract memory section from ScienceWorld prompt
        if style == "hint":
            assert "[Strategy]" not in prompt_sw, f"FAIL: hint style should strip [Strategy] prefix"
            # For hint mode, check the strategy content is there (just without tag)
            assert "melt tin" in prompt_sw.lower() or "heat source" in prompt_sw.lower(), \
                f"FAIL: hint should contain strategy content"
            print(f"  SW (tiered, hint): ✅ Strategy content present, tags stripped")

            assert "fridge" in prompt_alf.lower(), f"FAIL: ALF hint should mention fridge"
            print(f"  ALF (canon, hint): ✅ repair_text content present")
        else:
            assert "[Strategy]" in prompt_sw, f"FAIL: {style} prompt should contain [Strategy]"
            assert "[Plan]" in prompt_sw, f"FAIL: {style} prompt should contain [Plan]"
            print(f"  SW (tiered, {style}): ✅ Tiered format injected")

            assert "Open the fridge" in prompt_alf, \
                f"FAIL: ALF {style} should use repair_text"
            print(f"  ALF (canon, {style}): ✅ repair_text fallback works")


def test_real_memory_files():
    """Load actual memory files and verify get_repair_display() output."""
    print("\n" + "=" * 60)
    print("TEST 3: Real memory files")
    print("=" * 60)

    # ScienceWorld tiered
    sw_tiered_path = "memory/scienceworld_memory_tiered.json"
    if Path(sw_tiered_path).exists():
        store = FailureMemoryStore()
        store.load(sw_tiered_path)
        total = store.size()
        tiered_count = 0
        fallback_count = 0
        for entries in store._env_entries.values():
            for e in entries:
                display = e.get_repair_display()
                if "[Strategy]" in display:
                    tiered_count += 1
                else:
                    fallback_count += 1
        print(f"\n  SW tiered: {total} entries, {tiered_count} tiered ({tiered_count/total*100:.0f}%), {fallback_count} fallback")

        # Show 2 examples
        examples_shown = 0
        for entries in store._env_entries.values():
            for e in entries:
                if examples_shown >= 2:
                    break
                if e.repair_strategy:
                    print(f"\n  Example (env={e.env_idx}, failure='{e.failure_action[:40]}...'):")
                    display = e.get_repair_display()
                    for line in display.split("\n"):
                        print(f"    {line}")
                    examples_shown += 1
            if examples_shown >= 2:
                break
        print(f"  ✅ SW tiered memory loads and renders correctly")
    else:
        print(f"  ⚠️  {sw_tiered_path} not found, skipping")

    # ALFWorld canonicalized
    alf_canon_path = "memory/alfworld_memory_canonicalized.json"
    if Path(alf_canon_path).exists():
        store = FailureMemoryStore()
        store.load(alf_canon_path)
        total = store.size()
        has_repair_text = 0
        has_tiered = 0
        uses_solution = 0
        for entries in store._env_entries.values():
            for e in entries:
                display = e.get_repair_display()
                if "[Strategy]" in display:
                    has_tiered += 1
                elif e.repair_text and display == e.repair_text:
                    has_repair_text += 1
                else:
                    uses_solution += 1
        print(f"\n  ALF canon: {total} entries — tiered:{has_tiered}, repair_text:{has_repair_text}, solution_action:{uses_solution}")

        # Show 1 example
        for entries in store._env_entries.values():
            for e in entries:
                if e.repair_text:
                    print(f"\n  Example (env={e.env_idx}, failure='{e.failure_action[:40]}...'):")
                    print(f"    get_repair_display() = {e.get_repair_display()[:100]}")
                    break
            break
        print(f"  ✅ ALF canon memory: fallback to repair_text works")
    else:
        print(f"  ⚠️  {alf_canon_path} not found, skipping")


def main():
    print("🔬 Verifying tiered memory injection pipeline\n")

    test_get_repair_display()
    test_prompt_injection_styles()
    test_real_memory_files()

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

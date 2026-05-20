"""
LLM-as-judge prototype: evaluate whether an intervention moves the agent
closer to the correct (gold) path in ScienceWorld.

Problem: ScienceWorld's built-in score has 37.5% "evaluation blindness" —
different arms produce different trajectories but identical progress scores.

Approach: Give an LLM the task description, gold action sequence, and
agent trajectory for two arms (none vs repair). Ask it to judge which
arm makes more progress toward the goal.

Usage:
    python experiments/gate/test_llm_judge.py [--cp CP_ID] [--all]
"""

import json
import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.llm import LLMClient

# ── Config ──────────────────────────────────────────────────────────────

GOLD_PATHS_FILE = "rollouts/scienceworld_gold_paths.json"
ROLLOUTS_FILE = "rollouts/scienceworld_main_rollouts.json"
CHECKPOINTS_FILE = "checkpoints/scienceworld_checkpoints_midband.json"
CONFIG_FILE = "config_scienceworld.yaml"

# Representative "evaluation-blind" CPs (all zero progress, actions differ)
BLIND_CPS = [
    "cp_0012",  # boil — tin in workshop, agent struggles with heating
    "cp_0019",  # boil — different variation
    "cp_0058",  # melt — tin, repair goes to kitchen (better strategy)
    "cp_0393",  # chemistry-mix — none produces gibberish, repair tries real actions
    "cp_0209",  # test-conductivity
]

# Control CPs (non-blind, have nonzero progress)
CONTROL_CPS = [
    "cp_0006",  # boil — has progress, good control case
]

JUDGE_PROMPT_TEMPLATE = """\
You are evaluating whether an intervention helped an AI agent make progress toward completing a task in a text-based science simulation.

## Task Description
{task_desc}

## Gold (Correct) Action Sequence
The following is the correct sequence of actions to complete this task. Note: the gold path may include redundant steps (like repeated thermometer checks), so focus on the KEY actions (picking up objects, moving to correct rooms, performing core task steps).

{gold_path}

## Agent Context Before Intervention
The agent has already taken these actions before the intervention point:
{history_actions}

The agent failed at step {step_idx} with:
- Failed action: {failure_action}
- Observation: {failure_obs}

## Trajectory A (no intervention — agent continues on its own):
{traj_a}

## Trajectory B (with repair intervention — agent receives guidance):
{traj_b}

## Your Task
Compare trajectories A and B. For each trajectory, assess:
1. Does the agent take actions that align with the gold path's key steps?
2. Does the agent make meaningful progress toward the goal (even if no score change)?
3. Does the agent avoid wasting steps on invalid/unproductive actions?

Then provide your judgment in this EXACT format:

TRAJECTORY_A_SCORE: [0-10]
TRAJECTORY_B_SCORE: [0-10]
BETTER: [A|B|TIE]
REASONING: [1-3 sentences explaining why]

Scoring guide:
- 0: Agent takes no useful actions, completely stuck or producing gibberish
- 1-3: Agent takes some valid actions but wrong direction
- 4-5: Agent takes reasonable actions but doesn't reach any gold path milestone
- 6-7: Agent reaches or approaches a gold path milestone
- 8-10: Agent completes key steps from the gold path
"""


def load_data():
    """Load gold paths, rollouts, and checkpoints."""
    with open(GOLD_PATHS_FILE) as f:
        gold_paths = json.load(f)
    with open(ROLLOUTS_FILE) as f:
        rollouts = json.load(f)
    with open(CHECKPOINTS_FILE) as f:
        checkpoints_data = json.load(f)

    cp_map = {c["checkpoint_id"]: c for c in checkpoints_data["checkpoints"]}
    return gold_paths, rollouts, cp_map


def get_gold_path_for_cp(cp, gold_paths):
    """Find the best matching gold path for a checkpoint."""
    task = cp["task_type"]
    var_idx = cp.get("env_state", {}).get("variation_idx")

    # Try exact match first
    if var_idx is not None:
        key = f"{task}_{var_idx}"
        if key in gold_paths:
            return gold_paths[key]

    # Fall back to any gold path for this task type
    matching = {k: v for k, v in gold_paths.items() if v["task_name"] == task}
    if matching:
        # Pick the shortest one as representative
        return min(matching.values(), key=lambda v: len(v["gold_actions"]))

    return None


def extract_key_actions(gold_actions, max_display=40):
    """Extract key actions from gold path, filtering redundant ones."""
    key_actions = []
    seen_patterns = set()

    for i, action in enumerate(gold_actions):
        # Skip repeated thermometer checks
        if action in ("use thermometer in inventory on", "examine thermometer"):
            pattern = f"thermo_{len(key_actions)}"
            if pattern not in seen_patterns:
                seen_patterns.add(pattern)
                key_actions.append(f"[step {i+1}] (repeated temperature checks omitted)")
            continue

        # Skip repeated "look around" after navigation
        if action == "look around" and i > 0 and gold_actions[i-1].startswith("go to"):
            continue

        key_actions.append(f"[step {i+1}] {action}")

        if len(key_actions) >= max_display:
            key_actions.append(f"... ({len(gold_actions) - i - 1} more steps)")
            break

    return "\n".join(key_actions)


def format_trajectory(replay_data):
    """Format a replay's actions and observations into readable text."""
    lines = []
    actions = replay_data["actions"]
    observations = replay_data["observations"]

    for i, (act, obs) in enumerate(zip(actions, observations)):
        # Truncate very long think actions
        if act.startswith("think:") and len(act) > 300:
            act = act[:300] + "..."
        lines.append(f"  Action {i+1}: {act}")
        # Truncate long observations too
        if len(obs) > 200:
            obs = obs[:200] + "..."
        lines.append(f"  Observation: {obs}")
        lines.append("")

    return "\n".join(lines)


def format_history(history, max_recent=8):
    """Format checkpoint history (action/observation pairs)."""
    if not history:
        return "(no history)"

    recent = history[-max_recent:]
    lines = []
    start_idx = len(history) - len(recent)
    for i, (action, obs) in enumerate(recent):
        if action.startswith("think:") and len(action) > 200:
            action = action[:200] + "..."
        if len(obs) > 150:
            obs = obs[:150] + "..."
        lines.append(f"  [{start_idx + i + 1}] {action} → {obs}")

    if len(history) > max_recent:
        lines.insert(0, f"  (... {len(history) - max_recent} earlier steps omitted ...)")

    return "\n".join(lines)


def build_judge_prompt(cp, gold_path_data, none_replay, repair_replay):
    """Build the LLM judge prompt for a specific CP."""
    # Task description from init_obs (first line)
    task_desc = cp.get("init_obs", "")
    if "\n\n" in task_desc:
        task_desc = task_desc.split("\n\n")[0]  # Just the task instruction

    # Gold path
    gold_actions_str = extract_key_actions(gold_path_data["gold_actions"])

    # History
    history_str = format_history(cp.get("history", []))

    # Trajectories
    traj_a = format_trajectory(none_replay)
    traj_b = format_trajectory(repair_replay)

    return JUDGE_PROMPT_TEMPLATE.format(
        task_desc=task_desc,
        gold_path=gold_actions_str,
        history_actions=history_str,
        step_idx=cp["step_idx"],
        failure_action=cp["failure_action"],
        failure_obs=cp["failure_observation"],
        traj_a=traj_a,
        traj_b=traj_b,
    )


def parse_judge_response(response_text):
    """Parse structured judgment from LLM response."""
    result = {
        "score_a": None,
        "score_b": None,
        "better": None,
        "reasoning": None,
        "raw": response_text,
    }

    for line in response_text.split("\n"):
        line = line.strip()
        if line.startswith("TRAJECTORY_A_SCORE:"):
            try:
                result["score_a"] = int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("TRAJECTORY_B_SCORE:"):
            try:
                result["score_b"] = int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("BETTER:"):
            result["better"] = line.split(":")[1].strip()
        elif line.startswith("REASONING:"):
            result["reasoning"] = line.split(":", 1)[1].strip()

    return result


def run_judge(cp_ids, llm_client):
    """Run the LLM judge on specified CPs."""
    gold_paths, rollouts, cp_map = load_data()

    results = []

    for cp_id in cp_ids:
        if cp_id not in cp_map:
            print(f"⚠️  CP {cp_id} not found, skipping")
            continue
        if cp_id not in rollouts:
            print(f"⚠️  CP {cp_id} has no rollout data, skipping")
            continue

        cp = cp_map[cp_id]
        arms = rollouts[cp_id]

        if "none" not in arms or "repair" not in arms:
            print(f"⚠️  CP {cp_id} missing none/repair arms, skipping")
            continue

        # Get gold path
        gold_path = get_gold_path_for_cp(cp, gold_paths)
        if gold_path is None:
            print(f"⚠️  No gold path for {cp_id} ({cp['task_type']}), skipping")
            continue

        # Use first replay of each arm
        none_replay = arms["none"][0]
        repair_replay = arms["repair"][0]

        # Original score-based progress
        none_progress = none_replay["progress"]
        repair_progress = repair_replay["progress"]

        print(f"\n{'='*60}")
        print(f"📋 {cp_id} | task={cp['task_type']} | step={cp['step_idx']}")
        print(f"   Score-based progress: none={none_progress:.3f}, repair={repair_progress:.3f}")
        print(f"   Gold path: {gold_path['task_name']}_{gold_path['variation_idx']} ({len(gold_path['gold_actions'])} steps)")

        # Build and send prompt
        prompt = build_judge_prompt(cp, gold_path, none_replay, repair_replay)

        print(f"   Prompt length: {len(prompt)} chars")
        print(f"   Calling LLM judge...")

        try:
            # Use the extractor config for the judge (DeepSeek-V3.2, higher max_tokens)
            response = llm_client.chat(
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of AI agent behavior in interactive environments."},
                    {"role": "user", "content": prompt},
                ],
                label="llm_judge",
            )

            judgment = parse_judge_response(response.content)

            print(f"   ✅ LLM Judge result:")
            print(f"      Trajectory A (none):   score={judgment['score_a']}")
            print(f"      Trajectory B (repair): score={judgment['score_b']}")
            print(f"      Better: {judgment['better']}")
            print(f"      Reasoning: {judgment['reasoning']}")

            results.append({
                "cp_id": cp_id,
                "task_type": cp["task_type"],
                "step_idx": cp["step_idx"],
                "score_progress_none": none_progress,
                "score_progress_repair": repair_progress,
                "score_progress_same": abs(none_progress - repair_progress) < 0.001,
                "llm_score_none": judgment["score_a"],
                "llm_score_repair": judgment["score_b"],
                "llm_better": judgment["better"],
                "llm_reasoning": judgment["reasoning"],
            })

        except Exception as e:
            print(f"   ❌ LLM call failed: {e}")
            results.append({
                "cp_id": cp_id,
                "task_type": cp["task_type"],
                "error": str(e),
            })

    return results


def print_summary(results):
    """Print summary of LLM judge results."""
    print(f"\n{'='*60}")
    print("📊 SUMMARY: LLM-as-Judge vs Score-based Progress")
    print(f"{'='*60}")

    valid = [r for r in results if "error" not in r]
    blind = [r for r in valid if r["score_progress_same"]]

    print(f"\nTotal CPs evaluated: {len(valid)}")
    print(f"Score-blind CPs (same progress): {len(blind)}")

    if blind:
        print(f"\n--- Score-blind CPs where LLM detected a difference ---")
        for r in blind:
            diff = "✅ DETECTED" if r["llm_better"] != "TIE" else "❌ SAME"
            print(f"  {r['cp_id']} ({r['task_type']}): "
                  f"none={r['llm_score_none']}, repair={r['llm_score_repair']}, "
                  f"better={r['llm_better']} {diff}")
            print(f"    → {r['llm_reasoning']}")

    # Compute detection rate
    if blind:
        detected = sum(1 for r in blind if r["llm_better"] != "TIE")
        print(f"\n🎯 Detection rate on blind CPs: {detected}/{len(blind)} ({detected/len(blind)*100:.0f}%)")

    # Bias check: does LLM always favor B (repair)?
    if valid:
        favor_a = sum(1 for r in valid if r["llm_better"] == "A")
        favor_b = sum(1 for r in valid if r["llm_better"] == "B")
        ties = sum(1 for r in valid if r["llm_better"] == "TIE")
        print(f"\n📈 Overall LLM preference: A(none)={favor_a}, B(repair)={favor_b}, TIE={ties}")

    return valid


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge for ScienceWorld gate evaluation")
    parser.add_argument("--cp", type=str, nargs="+", help="Specific CP IDs to evaluate")
    parser.add_argument("--all-blind", action="store_true", help="Evaluate all blind CPs")
    parser.add_argument("--with-controls", action="store_true", help="Include control (non-blind) CPs")
    parser.add_argument("--save", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    # Load LLM config
    import yaml
    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    # Use extractor config (DeepSeek-V3.2, bigger context)
    ext_cfg = config["extractor"]
    llm_client = LLMClient(
        model=ext_cfg["model"],
        base_url=ext_cfg["base_url"],
        api_key=ext_cfg["api_key"],
        temperature=0.0,
        max_tokens=512,
    )

    # Determine which CPs to evaluate
    if args.cp:
        cp_ids = args.cp
    elif args.all_blind:
        cp_ids = list(BLIND_CPS)
        if args.with_controls:
            cp_ids.extend(CONTROL_CPS)
    else:
        # Default: 3 blind + 1 control
        cp_ids = BLIND_CPS[:3] + CONTROL_CPS[:1]

    print(f"🔬 LLM-as-Judge Prototype")
    print(f"   Model: {ext_cfg['model']}")
    print(f"   CPs to evaluate: {cp_ids}")

    results = run_judge(cp_ids, llm_client)
    summary = print_summary(results)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to {args.save}")


if __name__ == "__main__":
    main()

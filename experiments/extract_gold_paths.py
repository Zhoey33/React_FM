"""
Extract gold action sequences from ScienceWorld for the (task_name, variation_idx)
pairs present in a checkpoint JSON file.

Run from /Users/zhoey/React_FM/:
    python experiments/extract_gold_paths.py \
        --checkpoints checkpoints/scienceworld_core_train_checkpoints_resolved_v1.json \
        --output rollouts/scienceworld_gold_paths_v1.json
"""

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict

# Ensure Java is on PATH (mirrors src/scienceworld_env.py)
_JAVA_HOME = "/opt/homebrew/opt/openjdk/bin"
if _JAVA_HOME not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _JAVA_HOME + ":" + os.environ.get("PATH", "")

warnings.filterwarnings("ignore", message=".*camel case.*")

DEFAULT_CHECKPOINT_FILE = "checkpoints/scienceworld_checkpoints_midband.json"
DEFAULT_OUTPUT_FILE = "rollouts/scienceworld_gold_paths.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ScienceWorld gold paths for checkpoint task/variation pairs."
    )
    parser.add_argument(
        "--checkpoints",
        default=DEFAULT_CHECKPOINT_FILE,
        help="Checkpoint JSON file containing env_state.task_name and env_state.variation_idx",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help="Output JSON path for extracted gold paths",
    )
    parser.add_argument(
        "--rollouts",
        default="",
        help="Optional rollout JSON for zero-progress analysis",
    )
    return parser.parse_args()


args = parse_args()
CHECKPOINT_FILE = args.checkpoints
OUTPUT_FILE = args.output
ROLLOUT_FILE = args.rollouts

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – load checkpoints
# ─────────────────────────────────────────────────────────────────────────────
with open(CHECKPOINT_FILE) as f:
    data = json.load(f)

if isinstance(data, dict) and "checkpoints" in data:
    checkpoints = data["checkpoints"]
elif isinstance(data, list):
    checkpoints = data
else:
    raise ValueError(
        f"Unsupported checkpoint format in {CHECKPOINT_FILE}; expected list or dict with 'checkpoints'."
    )
print(f"Loaded {len(checkpoints)} checkpoints from {CHECKPOINT_FILE}")

# Build unique (task_name, variation_idx) pairs from env_state
unique_pairs: dict[tuple[str, int], list[str]] = {}   # (task, var) -> [cp_ids]
for cp in checkpoints:
    es = cp["env_state"]
    task = es["task_name"]
    var  = es["variation_idx"]
    key  = (task, var)
    unique_pairs.setdefault(key, []).append(cp["checkpoint_id"])

print(f"Unique (task_name, variation_idx) pairs: {len(unique_pairs)}")
for (task, var), ids in sorted(unique_pairs.items()):
    print(f"  {task:45s} var={var:>4d}  CPs: {ids}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – initialise ScienceWorld and extract gold paths
# ─────────────────────────────────────────────────────────────────────────────
print("\nInitialising ScienceWorld JVM …", flush=True)
from scienceworld import ScienceWorldEnv as SWEnv

gold_paths: dict[str, dict] = {}

# We need to group pairs by task_name so we create one env per task
# (SWEnv constructor takes the task name)
pairs_by_task: dict[str, list[int]] = defaultdict(list)
for (task, var) in unique_pairs:
    pairs_by_task[task].append(var)

for task_name, variations in sorted(pairs_by_task.items()):
    print(f"\n  Task: {task_name}  variations: {sorted(variations)}")
    # One env object per task (py4j JVM reuse)
    env = SWEnv(task_name, envStepLimit=100)

    for var_idx in sorted(variations):
        key = (task_name, var_idx)
        key_str = f"{task_name}_{var_idx}"

        env.load(task_name, var_idx, generateGoldPath=True)
        gold_actions = env.get_gold_action_sequence()

        cp_ids = unique_pairs[key]
        gold_paths[key_str] = {
            "task_name":      task_name,
            "variation_idx":  var_idx,
            "gold_actions":   gold_actions,
            "n_steps":        len(gold_actions),
            "checkpoint_ids": cp_ids,
        }
        print(f"    var={var_idx:>4d}  gold_len={len(gold_actions):>3d}  CPs={cp_ids}")

    env.close()

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – save
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(gold_paths, f, indent=2)
print(f"\nSaved {len(gold_paths)} gold paths → {OUTPUT_FILE}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – summary statistics
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("SUMMARY: unique gold paths extracted")
print("═" * 70)

# 4a. How many unique gold paths
print(f"\nTotal unique gold paths: {len(gold_paths)}")

# 4b. Gold path for 3 different task types
SHOW_TASKS = ["boil", "power-component", "chemistry-mix"]
print("\n" + "─" * 70)
print("Sample gold paths (first 10 actions shown):")
for key_str, gp in gold_paths.items():
    if gp["task_name"] in SHOW_TASKS:
        SHOW_TASKS.remove(gp["task_name"])          # show one per task type
        print(f"\n  [{gp['task_name']}  var={gp['variation_idx']}]  "
              f"gold_len={gp['n_steps']}  CPs={gp['checkpoint_ids']}")
        for i, a in enumerate(gp["gold_actions"][:10], 1):
            print(f"    {i:>3}. {a}")
        if gp["n_steps"] > 10:
            print(f"    … ({gp['n_steps'] - 10} more actions)")

# 4c. Average gold path length per task type
print("\n" + "─" * 70)
print("Average gold path length per task type:")
lengths_by_task: dict[str, list[int]] = defaultdict(list)
for gp in gold_paths.values():
    lengths_by_task[gp["task_name"]].append(gp["n_steps"])

rows = []
for task, lens in sorted(lengths_by_task.items()):
    avg = sum(lens) / len(lens)
    rows.append((task, len(lens), min(lens), max(lens), avg))

print(f"  {'task_name':<45}  vars  min   max   avg")
print(f"  {'─'*45}  ────  ───   ───   ─────")
for task, n, lo, hi, avg in rows:
    print(f"  {task:<45}  {n:>4}  {lo:>3}  {hi:>3}  {avg:>5.1f}")

if ROLLOUT_FILE and os.path.exists(ROLLOUT_FILE):
    # ─────────────────────────────────────────────────────────────────────────
    # Optional Step 5 – load rollouts and analyse the zero-progress checkpoints
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("ZERO-PROGRESS CHECKPOINT ANALYSIS")
    print("═" * 70)

    with open(ROLLOUT_FILE) as f:
        rollouts = json.load(f)

    zero_cp_ids: list[str] = []
    for cp_id, arms in rollouts.items():
        max_prog = max(
            r.get("progress", 0)
            for replays in arms.values()
            for r in replays
        )
        if max_prog == 0:
            zero_cp_ids.append(cp_id)

    print(f"\nZero-progress checkpoints: {len(zero_cp_ids)}  ({zero_cp_ids})")

    cp_to_pair: dict[str, tuple[str, int]] = {}
    for cp in checkpoints:
        es = cp["env_state"]
        cp_to_pair[cp["checkpoint_id"]] = (es["task_name"], es["variation_idx"])

    cp_to_action_hist: dict[str, list[str]] = {}
    for cp in checkpoints:
        cp_to_action_hist[cp["checkpoint_id"]] = cp.get("action_history", [])

    def _longest_prefix_match(gold: list[str], history: list[str]) -> int:
        matched = 0
        gold_norm = [a.strip().lower() for a in gold]
        history_norm = [a.strip().lower() for a in history]
        history_set = set(history_norm)
        for i, ga in enumerate(gold_norm):
            if ga in history_set:
                matched = i + 1
            else:
                break
        return matched

    print("\n" + f"  {'cp_id':<10}  {'task_name':<35}  {'var':>4}  "
          f"{'hist_len':>8}  {'gold_len':>8}  {'prefix_match':>12}  {'%':>5}")
    print("  " + "─" * 85)

    for cp_id in sorted(zero_cp_ids):
        task_name, var_idx = cp_to_pair.get(cp_id, ("unknown", -1))
        key_str = f"{task_name}_{var_idx}"

        action_hist = cp_to_action_hist.get(cp_id, [])
        gp = gold_paths.get(key_str)

        if gp is None:
            print(f"  {cp_id:<10}  {task_name:<35}  var={var_idx:>4}  "
                  f"(no gold path found)")
            continue

        gold = gp["gold_actions"]
        gold_len = gp["n_steps"]
        hist_len = len(action_hist)
        prefix = _longest_prefix_match(gold, action_hist)
        pct = 100.0 * prefix / gold_len if gold_len else 0.0

        print(f"  {cp_id:<10}  {task_name:<35}  {var_idx:>4}  "
              f"{hist_len:>8}  {gold_len:>8}  {prefix:>12}  {pct:>4.1f}%")

print("\nDone.")

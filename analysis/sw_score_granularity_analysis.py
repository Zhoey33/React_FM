#!/usr/bin/env python3
"""
ScienceWorld rollout analysis: "different behavior, same score" diagnosis.

Determines whether the problem is evaluation granularity or genuinely no progress.
"""

import json
import sys
from collections import defaultdict, Counter

# ── Load data ──────────────────────────────────────────────────────────────────
with open("rollouts/scienceworld_main_rollouts.json") as f:
    rollouts = json.load(f)

with open("checkpoints/scienceworld_checkpoints_midband.json") as f:
    cp_data = json.load(f)

with open("memory/scienceworld_memory_canonicalized.json") as f:
    mem_data = json.load(f)

ARMS = ["none", "cue", "question", "repair"]
EPS = 0.001

# Build checkpoint lookup
cp_lookup = {cp["checkpoint_id"]: cp for cp in cp_data["checkpoints"]}

# ── Helper: compute per-arm average progress ──────────────────────────────────
def arm_avg_progress(cp_id):
    """Returns dict arm -> avg progress across replays."""
    result = {}
    for arm in ARMS:
        replays = rollouts[cp_id][arm]
        result[arm] = sum(r["progress"] for r in replays) / len(replays)
    return result

def arm_avg_score_after(cp_id):
    """Returns dict arm -> avg score_after across replays."""
    result = {}
    for arm in ARMS:
        replays = rollouts[cp_id][arm]
        result[arm] = sum(r["score_after"] for r in replays) / len(replays)
    return result

# ── Classify checkpoints ──────────────────────────────────────────────────────
same_score_cps = []
diff_score_cps = []

for cp_id in rollouts:
    avg_prog = arm_avg_progress(cp_id)
    values = list(avg_prog.values())
    # "Same score" = all arms within epsilon of each other
    if max(values) - min(values) < EPS:
        same_score_cps.append(cp_id)
    else:
        diff_score_cps.append(cp_id)

print("=" * 80)
print("PART 1: 'Different behavior, same score' checkpoints")
print("=" * 80)
print(f"\nTotal checkpoints: {len(rollouts)}")
print(f"Same-score CPs (all arms within ε={EPS}): {len(same_score_cps)}")
print(f"Different-score CPs: {len(diff_score_cps)}")

# 1a. List task types
print(f"\n--- 1a. Same-score checkpoint list ---")
for cp_id in same_score_cps:
    cp = cp_lookup[cp_id]
    avg = arm_avg_progress(cp_id)
    print(f"  {cp_id}: task_type={cp['task_type']}, env={cp['env_idx']}, "
          f"step={cp['step_idx']}, avg_progress={avg['none']:.4f}")

# 1b. Breakdown by task_type
print(f"\n--- 1b. Same-score CPs per task_type ---")
tt_counts = Counter(cp_lookup[c]["task_type"] for c in same_score_cps)
for tt, cnt in tt_counts.most_common():
    print(f"  {tt}: {cnt}")

# 1c. Per-arm score details
print(f"\n--- 1c. Per-arm scores for same-score CPs ---")
print(f"{'CP ID':<12} {'task_type':<25} {'score_before':>12} | "
      f"{'none':>8} {'cue':>8} {'question':>8} {'repair':>8} | {'all_zero?':>8}")
print("-" * 110)
for cp_id in same_score_cps:
    cp = cp_lookup[cp_id]
    sb = cp["score_at_checkpoint"]
    avg_sa = arm_avg_score_after(cp_id)
    avg_prog = arm_avg_progress(cp_id)
    all_zero = all(abs(v) < EPS for v in avg_prog.values())
    print(f"{cp_id:<12} {cp['task_type']:<25} {sb:>12} | "
          f"{avg_sa['none']:>8.1f} {avg_sa['cue']:>8.1f} "
          f"{avg_sa['question']:>8.1f} {avg_sa['repair']:>8.1f} | "
          f"{'YES' if all_zero else 'NO':>8}")


# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("PART 2: Qualitative trajectory analysis (3-4 representative same-score CPs)")
print("=" * 80)

# Pick representatives from different task types
# Get unique task types in same-score
same_score_by_tt = defaultdict(list)
for cp_id in same_score_cps:
    tt = cp_lookup[cp_id]["task_type"]
    same_score_by_tt[tt].append(cp_id)

# Pick one from each task type (up to 4)
representatives = []
for tt in same_score_by_tt:
    representatives.append(same_score_by_tt[tt][0])
    if len(representatives) >= 4:
        break

print(f"\nSelected representatives: {representatives}")

for cp_id in representatives:
    cp = cp_lookup[cp_id]
    env_idx = str(cp["env_idx"])
    mem_id = cp.get("retrieved_memory_id")

    print(f"\n{'─' * 80}")
    print(f"CP: {cp_id} | task_type: {cp['task_type']} | env: {cp['env_idx']} | "
          f"step: {cp['step_idx']}/{cp['max_steps']}")
    print(f"Failure: action='{cp['failure_action']}' → obs='{cp['failure_observation'][:80]}'")
    print(f"Score at checkpoint: {cp['score_at_checkpoint']}")

    # Show injection text
    mem_entries = mem_data["envs"].get(env_idx, [])
    mem_entry = None
    if mem_id is not None:
        for e in mem_entries:
            if e["memory_id"] == mem_id:
                mem_entry = e
                break

    if mem_entry:
        print(f"\n  INJECTION (question arm): \"{mem_entry.get('question_text', 'N/A')}\"")
        print(f"  INJECTION (repair arm):   \"{mem_entry.get('repair_text', 'N/A')}\"")
        print(f"  Memory context: failure='{mem_entry['failure_action']}' → solution='{mem_entry['solution_action']}'")
    else:
        print(f"  No memory entry found (mem_id={mem_id})")

    # Show first 5 actions per arm (replay_idx=0)
    for arm in ARMS:
        r = rollouts[cp_id][arm][0]  # replay 0
        print(f"\n  [{arm.upper()}] score_before={r['score_before']} → score_after={r['score_after']} "
              f"(progress={r['progress']:.3f})")
        n_show = min(5, len(r["actions"]))
        for i in range(n_show):
            act = r["actions"][i][:100]
            obs = r["observations"][i][:100]
            print(f"    step {i}: ACT: {act}")
            print(f"            OBS: {obs}")

    # Analysis: did behavior change?
    print(f"\n  ANALYSIS:")
    # Compare none vs repair actions
    none_acts = [a[:60] for a in rollouts[cp_id]["none"][0]["actions"][:5]]
    repair_acts = [a[:60] for a in rollouts[cp_id]["repair"][0]["actions"][:5]]

    # Count valid actions (not "No known action")
    def count_valid(replays):
        valid = 0
        total = 0
        for r in replays:
            for o in r["observations"]:
                total += 1
                if "No known action" not in o:
                    valid += 1
        return valid, total

    none_v, none_t = count_valid(rollouts[cp_id]["none"])
    repair_v, repair_t = count_valid(rollouts[cp_id]["repair"])
    question_v, question_t = count_valid(rollouts[cp_id]["question"])

    print(f"    Valid actions: none={none_v}/{none_t}, question={question_v}/{question_t}, "
          f"repair={repair_v}/{repair_t}")

    # Check if actions diverge
    acts_differ = none_acts != repair_acts
    print(f"    Actions differ (none vs repair)? {acts_differ}")
    if acts_differ:
        print(f"      none:   {none_acts[:3]}")
        print(f"      repair: {repair_acts[:3]}")


# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("PART 3: Score granularity analysis (ALL 40 checkpoints)")
print("=" * 80)

# 3a. Unique score_before values
all_score_before = []
all_score_after = defaultdict(list)
all_progress = defaultdict(list)

for cp_id in rollouts:
    cp = cp_lookup[cp_id]
    all_score_before.append(cp["score_at_checkpoint"])
    for arm in ARMS:
        for r in rollouts[cp_id][arm]:
            all_score_after[arm].append(r["score_after"])
            all_progress[arm].append(r["progress"])

print(f"\n--- 3a. score_before distribution ---")
sb_counter = Counter(all_score_before)
for score in sorted(sb_counter.keys()):
    bar = "█" * sb_counter[score]
    print(f"  score={score:>3}: {sb_counter[score]:>3} CPs  {bar}")

print(f"\n  Unique score_before values: {sorted(set(all_score_before))}")
print(f"  Range: [{min(all_score_before)}, {max(all_score_before)}]")

# 3b. Unique score_after values per arm
print(f"\n--- 3b. score_after unique values per arm ---")
for arm in ARMS:
    unique = sorted(set(all_score_after[arm]))
    print(f"  {arm:>10}: {len(unique)} unique values: {unique}")

# 3c. Minimum score increment
print(f"\n--- 3c. Minimum non-zero score increment ---")
all_increments = []
for cp_id in rollouts:
    for arm in ARMS:
        for r in rollouts[cp_id][arm]:
            delta = r["score_after"] - r["score_before"]
            if delta > 0:
                all_increments.append(delta)

if all_increments:
    print(f"  All positive increments: {sorted(set(all_increments))}")
    print(f"  Minimum increment: {min(all_increments)}")
    print(f"  Count of replays with positive increment: {len(all_increments)} / "
          f"{sum(len(rollouts[cp_id][arm]) for cp_id in rollouts for arm in ARMS)}")
else:
    print("  No positive increments found!")

# Also check progress values
all_prog_vals = []
for cp_id in rollouts:
    for arm in ARMS:
        for r in rollouts[cp_id][arm]:
            all_prog_vals.append(r["progress"])

nonzero_prog = [p for p in all_prog_vals if p > EPS]
print(f"\n  Progress values (non-zero): {sorted(set(round(p, 4) for p in nonzero_prog))}")
print(f"  Minimum non-zero progress: {min(nonzero_prog):.4f}" if nonzero_prog else "  No non-zero progress!")
print(f"  Zero-progress replays: {len(all_prog_vals) - len(nonzero_prog)} / {len(all_prog_vals)} "
      f"({100*(len(all_prog_vals) - len(nonzero_prog))/len(all_prog_vals):.1f}%)")

# 3d. Steps to achieve one score increment
print(f"\n--- 3d. Steps context ---")
print(f"  Rollout window: 10 steps per replay")
for cp_id in rollouts:
    cp = cp_lookup[cp_id]
    for arm in ARMS:
        for r in rollouts[cp_id][arm]:
            if r["progress"] > EPS:
                # How many steps were taken?
                # Check when score changed (we only have final, not per-step)
                pass  # We only have final scores, not per-step

# Instead, analyze: what fraction of 10-step rollouts achieve any progress?
print(f"\n  Fraction of 10-step rollouts achieving any progress:")
for arm in ARMS:
    total = 0
    has_prog = 0
    for cp_id in rollouts:
        for r in rollouts[cp_id][arm]:
            total += 1
            if r["progress"] > EPS:
                has_prog += 1
    print(f"    {arm:>10}: {has_prog}/{total} ({100*has_prog/total:.1f}%)")

# Score dynamics: how often does score_after > score_before?
print(f"\n  Score improvement rate (score_after > score_before):")
for arm in ARMS:
    total = 0
    improved = 0
    for cp_id in rollouts:
        for r in rollouts[cp_id][arm]:
            total += 1
            if r["score_after"] > r["score_before"]:
                improved += 1
    print(f"    {arm:>10}: {improved}/{total} ({100*improved/total:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("PART 4: 'Different behavior, different score' checkpoints")
print("=" * 80)

# 4a. Task types
print(f"\n--- 4a. Different-score CPs per task_type ---")
diff_tt_counts = Counter(cp_lookup[c]["task_type"] for c in diff_score_cps)
for tt, cnt in diff_tt_counts.most_common():
    print(f"  {tt}: {cnt}")

# 4b. Typical progress magnitude
print(f"\n--- 4b. Progress magnitude for different-score CPs ---")
print(f"{'CP ID':<12} {'task_type':<25} {'score_before':>12} | "
      f"{'none':>8} {'cue':>8} {'question':>8} {'repair':>8} | {'max_diff':>8}")
print("-" * 115)

progress_by_arm_diff = defaultdict(list)
for cp_id in diff_score_cps:
    cp = cp_lookup[cp_id]
    avg_prog = arm_avg_progress(cp_id)
    max_diff = max(avg_prog.values()) - min(avg_prog.values())
    for arm in ARMS:
        progress_by_arm_diff[arm].append(avg_prog[arm])
    print(f"{cp_id:<12} {cp['task_type']:<25} {cp['score_at_checkpoint']:>12} | "
          f"{avg_prog['none']:>8.4f} {avg_prog['cue']:>8.4f} "
          f"{avg_prog['question']:>8.4f} {avg_prog['repair']:>8.4f} | "
          f"{max_diff:>8.4f}")

print(f"\n  Average progress per arm (diff-score CPs only):")
for arm in ARMS:
    vals = progress_by_arm_diff[arm]
    print(f"    {arm:>10}: mean={sum(vals)/len(vals):.4f}, "
          f"max={max(vals):.4f}, min={min(vals):.4f}")

# 4c. Pattern: which task types have scoreable intermediate steps?
print(f"\n--- 4c. Task types: scoreable vs non-scoreable intermediate steps ---")
tt_all = defaultdict(lambda: {"total": 0, "has_progress": 0, "diff_score": 0})
for cp_id in rollouts:
    tt = cp_lookup[cp_id]["task_type"]
    tt_all[tt]["total"] += 1
    avg_prog = arm_avg_progress(cp_id)
    # Has any arm with non-zero average progress?
    if any(v > EPS for v in avg_prog.values()):
        tt_all[tt]["has_progress"] += 1
    if cp_id in diff_score_cps:
        tt_all[tt]["diff_score"] += 1

print(f"{'task_type':<25} {'total_CPs':>10} {'any_progress':>13} {'diff_between_arms':>18}")
print("-" * 70)
for tt in sorted(tt_all.keys()):
    d = tt_all[tt]
    print(f"{tt:<25} {d['total']:>10} {d['has_progress']:>13} {d['diff_score']:>18}")


# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("PART 5: Summary table")
print("=" * 80)

# Build summary per task_type
summary = {}
for cp_id in rollouts:
    tt = cp_lookup[cp_id]["task_type"]
    if tt not in summary:
        summary[tt] = {
            "total_CPs": 0,
            "same_score_CPs": 0,
            "diff_score_CPs": 0,
            "progress_by_arm": defaultdict(list),
            "increments": [],
        }
    summary[tt]["total_CPs"] += 1
    if cp_id in same_score_cps:
        summary[tt]["same_score_CPs"] += 1
    else:
        summary[tt]["diff_score_CPs"] += 1

    for arm in ARMS:
        for r in rollouts[cp_id][arm]:
            summary[tt]["progress_by_arm"][arm].append(r["progress"])
            delta = r["score_after"] - r["score_before"]
            if delta > 0:
                summary[tt]["increments"].append(delta)

print(f"\n{'task_type':<25} {'total':>5} {'same':>5} {'diff':>5} | "
      f"{'avg_p(none)':>11} {'avg_p(cue)':>11} {'avg_p(ques)':>11} {'avg_p(repair)':>12} | "
      f"{'min_incr':>8} {'typical_incr':>12}")
print("-" * 130)

for tt in sorted(summary.keys()):
    s = summary[tt]
    avg_p = {}
    for arm in ARMS:
        vals = s["progress_by_arm"][arm]
        avg_p[arm] = sum(vals) / len(vals) if vals else 0

    incrs = s["increments"]
    min_incr = min(incrs) if incrs else 0
    # Typical = median
    if incrs:
        sorted_incrs = sorted(incrs)
        median_incr = sorted_incrs[len(sorted_incrs) // 2]
    else:
        median_incr = 0

    print(f"{tt:<25} {s['total_CPs']:>5} {s['same_score_CPs']:>5} {s['diff_score_CPs']:>5} | "
          f"{avg_p['none']:>11.4f} {avg_p['cue']:>11.4f} "
          f"{avg_p['question']:>11.4f} {avg_p['repair']:>12.4f} | "
          f"{min_incr:>8} {median_incr:>12}")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("PART 6: Diagnostic conclusions")
print("=" * 80)

# Compute overall stats
total_replays = sum(len(rollouts[cp_id][arm]) for cp_id in rollouts for arm in ARMS)
zero_replays = sum(1 for cp_id in rollouts for arm in ARMS
                   for r in rollouts[cp_id][arm] if abs(r["progress"]) < EPS)

print(f"\n1. SCALE OF THE PROBLEM:")
print(f"   - {len(same_score_cps)}/{len(rollouts)} checkpoints ({100*len(same_score_cps)/len(rollouts):.0f}%) "
      f"show NO difference between arms")
print(f"   - {zero_replays}/{total_replays} individual replays ({100*zero_replays/total_replays:.0f}%) "
      f"show ZERO progress")

# Task types that are entirely same-score
all_same_tt = [tt for tt in summary if summary[tt]["diff_score_CPs"] == 0]
print(f"\n2. TASK TYPES WITH ZERO SCOREABLE DIFFERENTIATION:")
for tt in all_same_tt:
    s = summary[tt]
    print(f"   - {tt}: {s['total_CPs']} CPs, all same-score → evaluation cannot detect ANY intervention effect")

# Task types with some differentiation
some_diff_tt = [tt for tt in summary if summary[tt]["diff_score_CPs"] > 0]
print(f"\n3. TASK TYPES WITH SOME DIFFERENTIATION:")
for tt in some_diff_tt:
    s = summary[tt]
    pct = 100 * s["diff_score_CPs"] / s["total_CPs"]
    print(f"   - {tt}: {s['diff_score_CPs']}/{s['total_CPs']} CPs ({pct:.0f}%) show arm differences")

# Key question: is it granularity or genuine no-effect?
print(f"\n4. GRANULARITY vs GENUINE NO-EFFECT:")
# Count CPs where ALL arms have zero progress (truly stuck)
all_zero_cps = []
mixed_zero_cps = []  # same score but some replays have progress
for cp_id in same_score_cps:
    any_progress = False
    for arm in ARMS:
        for r in rollouts[cp_id][arm]:
            if r["progress"] > EPS:
                any_progress = True
    if any_progress:
        mixed_zero_cps.append(cp_id)
    else:
        all_zero_cps.append(cp_id)

print(f"   Among {len(same_score_cps)} same-score CPs:")
print(f"   - {len(all_zero_cps)} CPs: ALL replays across ALL arms have zero progress → truly stuck")
print(f"   - {len(mixed_zero_cps)} CPs: some replays have progress but arms are indistinguishable → granularity issue")

# For each truly-stuck CP, check how many steps are left
print(f"\n5. TRULY STUCK CPs - remaining steps budget:")
for cp_id in all_zero_cps:
    cp = cp_lookup[cp_id]
    remaining = cp["max_steps"] - cp["step_idx"]
    print(f"   {cp_id}: task={cp['task_type']}, step {cp['step_idx']}/{cp['max_steps']}, "
          f"remaining={remaining}, score={cp['score_at_checkpoint']}")

# For mixed-zero CPs
if mixed_zero_cps:
    print(f"\n6. MIXED-ZERO CPs (same across arms, but some replays have progress):")
    for cp_id in mixed_zero_cps:
        cp = cp_lookup[cp_id]
        avg_prog = arm_avg_progress(cp_id)
        print(f"   {cp_id}: task={cp['task_type']}, score={cp['score_at_checkpoint']}, "
              f"progress={avg_prog}")

# Action validity analysis
print(f"\n7. ACTION VALIDITY: Do interventions help the agent take more valid actions?")
arm_validity = defaultdict(lambda: {"valid": 0, "total": 0})
for cp_id in rollouts:
    for arm in ARMS:
        for r in rollouts[cp_id][arm]:
            for obs in r["observations"]:
                arm_validity[arm]["total"] += 1
                if "No known action" not in obs:
                    arm_validity[arm]["valid"] += 1

for arm in ARMS:
    v = arm_validity[arm]
    pct = 100 * v["valid"] / v["total"] if v["total"] > 0 else 0
    print(f"   {arm:>10}: {v['valid']}/{v['total']} valid ({pct:.1f}%)")

# Same analysis but only for same-score CPs
print(f"\n   (Same analysis for same-score CPs only:)")
arm_validity_ss = defaultdict(lambda: {"valid": 0, "total": 0})
for cp_id in same_score_cps:
    for arm in ARMS:
        for r in rollouts[cp_id][arm]:
            for obs in r["observations"]:
                arm_validity_ss[arm]["total"] += 1
                if "No known action" not in obs:
                    arm_validity_ss[arm]["valid"] += 1

for arm in ARMS:
    v = arm_validity_ss[arm]
    pct = 100 * v["valid"] / v["total"] if v["total"] > 0 else 0
    print(f"   {arm:>10}: {v['valid']}/{v['total']} valid ({pct:.1f}%)")

print(f"\n{'=' * 80}")
print("END OF ANALYSIS")
print(f"{'=' * 80}")

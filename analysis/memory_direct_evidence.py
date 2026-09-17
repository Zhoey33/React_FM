"""Recompute historical and new paired evidence; never issue model calls."""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def mean(values):
    return statistics.mean(values) if values else None


def cluster_interval(values, cluster_keys, resamples=5000):
    clusters = defaultdict(list)
    for value, key in zip(values, cluster_keys):
        clusters[key].append(value)
    if len(clusters) < 2:
        return None
    groups = list(clusters.values())
    rng = random.Random(20260916)
    boot = sorted(mean([x for group in rng.choices(groups, k=len(groups)) for x in group])
                  for _ in range(resamples))
    return [boot[int(.025 * resamples)], boot[int(.975 * resamples) - 1]]


def historical():
    source = ROOT / "rollouts/scienceworld_core_train_rollouts_expanded134_judged.json"
    cp_source = ROOT / "checkpoints/scienceworld_core_train_checkpoints_resolved_expanded134.json"
    payload = json.loads(source.read_text())
    cps = {x["checkpoint_id"]: x for x in json.loads(cp_source.read_text())["checkpoints"]}
    ids = [k for k in payload if not k.startswith("__")]
    arms = ["none", "cue", "question", "repair"]
    for cid in ids:
        assert cid in cps
        assert all(len(payload[cid][a]) == 3 for a in arms)
        assert len({x["score_before"] for a in arms for x in payload[cid][a]}) == 1
    result = {"source": str(source), "checkpoints_source": str(cp_source),
              "checkpoints": len(ids), "independent_episodes": len({
                  (cps[k]["task_type"], cps[k]["env_state"]["variation_idx"]) for k in ids}),
              "limitations": ["Training checkpoints selected by previous work, not a fresh test set",
                              "cue is a generic reminder, not an extra direct-repair LLM call",
                              "No per-branch token accounting", "Memory provenance not revalidated",
                              "Repeated checkpoints/replays are not independent tasks"], "arms": {}}
    delta = {}
    for arm in arms:
        rs = [x for cid in ids for x in payload[cid][arm]]
        delta[arm] = {cid: mean([x["score_after"] - x["score_before"] for x in payload[cid][arm]])
                      for cid in ids}
        result["arms"][arm] = {"branches": len(rs),
            "mean_final_score_delta": mean([x["score_after"] - x["score_before"] for x in rs]),
            "score_increased": sum(x["score_after"] > x["score_before"] for x in rs),
            "reached_100": sum(x["score_after"] >= 100 for x in rs),
            "mean_steps": mean([x["steps_taken"] for x in rs])}
    result["paired_differences"] = {}
    clusters = [(cps[k]["task_type"], cps[k]["env_state"]["variation_idx"]) for k in ids]
    for other in ["none", "cue", "question"]:
        differences = [delta["repair"][k] - delta[other][k] for k in ids]
        result["paired_differences"]["repair_minus_" + other] = {
            "mean_final_score_delta_difference": mean(differences),
            "checkpoint_win_tie_loss": [sum((d > 0, d == 0, d < 0)[i] for d in differences)
                                       for i in range(3)],
            "episode_cluster_bootstrap_95pct": cluster_interval(differences, clusters)}
    return result


def pilot(directory, resume_directory=None):
    branches = rows(directory / "branches.jsonl")
    cps = rows(directory / "checkpoints.jsonl")
    rejected = {r["checkpoint_id"] for r in rows(directory / "rejected.jsonl")}
    duplicate_menu_ids = set()
    for cp in cps:
        for step in cp["steps"]:
            if step["observation"].startswith("Ambiguous request:"):
                descriptions = re.findall(r"(?m)^\d+:\s*(.+)$", step["observation"])
                if len(descriptions) != len(set(descriptions)):
                    duplicate_menu_ids.add(cp["checkpoint_id"])
    calls = {r["call_id"]: r for r in rows(directory / "calls.jsonl")}
    protocol = json.loads((directory / "protocol.json").read_text()) if (directory / "protocol.json").exists() else {}
    recovery = None
    if resume_directory is not None:
        resumed_protocol = json.loads((resume_directory / "protocol.json").read_text())
        ignored = {"output", "total_cny"}
        if {k: v for k, v in protocol.items() if k not in ignored} != {
                k: v for k, v in resumed_protocol.items() if k not in ignored}:
            raise ValueError("Resumed protocol differs beyond output directory and run-level cap")
        expected = {(a, r) for a in protocol["arms"] for r in range(protocol["replays"])}
        original_complete = {cp["checkpoint_id"] for cp in cps
            if cp["checkpoint_id"] not in rejected and {
                (r["arm"], r["replay"]) for r in branches
                if r["checkpoint_id"] == cp["checkpoint_id"]} == expected}
        originals = {cp["checkpoint_id"]: cp for cp in cps}
        resumed_cps = rows(resume_directory / "checkpoints.jsonl")
        resumed_ids = {cp["checkpoint_id"] for cp in resumed_cps}
        if resumed_ids != set(originals) - original_complete or any(
                cp != originals[cp["checkpoint_id"]] for cp in resumed_cps):
            raise ValueError("Resume must replace exactly all incomplete checkpoints with identical starts")
        discarded = [r for r in branches if r["checkpoint_id"] not in original_complete]
        branches = [r for r in branches if r["checkpoint_id"] in original_complete]
        branches += rows(resume_directory / "branches.jsonl")
        rejected = {r["checkpoint_id"] for r in rows(resume_directory / "rejected.jsonl")}
        resumed_calls = {r["call_id"]: r for r in rows(resume_directory / "calls.jsonl")}
        calls.update({("resume", k): v for k, v in resumed_calls.items()})
        recovery = {"directory": str(resume_directory),
                    "original_complete_checkpoints": len(original_complete),
                    "restarted_checkpoints": len(resumed_ids),
                    "discarded_original_branches": len(discarded),
                    "all_attempt_costs_included": True}
    by_cp = defaultdict(dict)
    for row in branches:
        by_cp[row["checkpoint_id"]][(row["arm"], row["replay"])] = row
    expected = {(a, r) for a in protocol.get("arms", []) for r in range(protocol.get("replays", 0))}
    complete = {k: v for k, v in by_cp.items() if set(v) == expected and k not in rejected}
    valid_rows = [r for v in complete.values() for r in v.values()]
    result = {"directory": str(directory), "protocol": protocol,
              "collected_episodes": len(rows(directory / "episodes.jsonl")),
              "collected_checkpoints": len(cps), "complete_checkpoints": len(complete),
              "rejected_checkpoints": len(rejected), "recorded_branches": len(branches),
              "complete_calls": sum(x["status"] == "complete" for x in calls.values()),
              "uncertain_or_reserved_calls": sum(x["status"] != "complete" for x in calls.values()),
              "accounted_cny_upper_estimate": sum(x["accounted_cny"] for x in calls.values()),
              "complete_independent_episodes": len({r["episode_id"] for r in valid_rows}),
              "arms": {}, "paired_differences": {}}
    if recovery is not None:
        result["interruption_recovery"] = recovery
    for arm in protocol.get("arms", []):
        rs = [x for x in valid_rows if x["arm"] == arm]
        result["arms"][arm] = {"branches": len(rs),
            "mean_final_score_delta": mean([x["score_after"] - x["score_before"] for x in rs]),
            "mean_cny": mean([x["cost_cny"] for x in rs]),
            "reached_100": sum(x["score_after"] >= 100 for x in rs),
            "mean_actor_steps": mean([len(x["steps"]) for x in rs]),
            "mean_absolute_repeat_score_gap": mean([
                abs(data[(arm, 0)]["score_after"] - data[(arm, 1)]["score_after"])
                for data in complete.values()]) if protocol.get("replays") == 2 else None,
            "terminal_counts": dict(Counter(x["status"] for x in rs))}
    for other in ["continue", "cue", "direct"]:
        if other not in protocol.get("arms", []):
            continue
        diffs, clusters = [], []
        for cid, data in complete.items():
            memory = [x for (a, _), x in data.items() if a == "memory"]
            control = [x for (a, _), x in data.items() if a == other]
            diffs.append(mean([x["score_after"] for x in memory]) - mean([x["score_after"] for x in control]))
            clusters.append(memory[0]["episode_id"])
        result["paired_differences"]["memory_minus_" + other] = {
            "mean_final_score_difference": mean(diffs),
            "episode_cluster_bootstrap_95pct": cluster_interval(diffs, clusters) if diffs else None,
            "checkpoint_win_tie_loss": [sum((d > 0, d == 0, d < 0)[i] for d in diffs) for i in range(3)]}
    result["memory_hit_checkpoints"] = sum(next(iter(v.values()))["memory"]["memory_id"] is not None
                                            for v in complete.values())
    strata = defaultdict(list)
    for cid, data in complete.items():
        example = next(iter(data.values()))
        memory_rows = [r for (a, _), r in data.items() if a == "memory"]
        direct_rows = [r for (a, _), r in data.items() if a == "direct"]
        difference = mean([r["score_after"] for r in memory_rows]) - mean([r["score_after"] for r in direct_rows])
        keys = ["memory_hit" if example["memory"]["memory_id"] is not None else "memory_miss",
                "failure_type/" + example["failure_type"], "task/" + example["task"]]
        for key in keys:
            strata[key].append((difference, example["episode_id"]))
    result["descriptive_strata_memory_minus_direct"] = {
        key: {"checkpoints": len(values), "episodes": len({eid for _, eid in values}),
              "mean_final_score_difference": mean([d for d, _ in values])}
        for key, values in strata.items()}
    unique_diffs, unique_clusters = [], []
    for cid, data in complete.items():
        if cid in duplicate_menu_ids:
            continue
        memory = [r for (a, _), r in data.items() if a == "memory"]
        direct = [r for (a, _), r in data.items() if a == "direct"]
        unique_diffs.append(mean([r["score_after"] for r in memory]) - mean([r["score_after"] for r in direct]))
        unique_clusters.append(memory[0]["episode_id"])
    result["unique_referent_sensitivity"] = {
        "excluded_checkpoint_ids": sorted(duplicate_menu_ids),
        "remaining_complete_checkpoints": len(unique_diffs),
        "memory_minus_direct": mean(unique_diffs),
        "episode_cluster_bootstrap_95pct": cluster_interval(unique_diffs, unique_clusters)}
    cp_lookup = {cp["checkpoint_id"]: cp for cp in cps}
    door_recovery = {}
    for arm in protocol.get("arms", []):
        cases, recovered, successful_action_counts = 0, 0, []
        for row in valid_rows:
            if row["arm"] != arm or row["failure_type"] != "precondition_blocked":
                continue
            cp = cp_lookup[row["checkpoint_id"]]
            failed_action = cp["steps"][-1]["action"]
            match = re.fullmatch(r"go (?:(?:to|door to) )?(.+?)(?: door)?", failed_action)
            if not match or cp["steps"][-1]["observation"].strip() != "The door is not open.":
                continue
            cases += 1
            expected_observation = "You move to the " + match.group(1) + "."
            actions = [s for s in row["steps"] if not s["action"].lower().startswith(("think:", "think "))]
            for index, step in enumerate(actions[:5], 1):
                if step["observation"].strip() in [expected_observation, "You move through the door to the " + match.group(1) + "."]:
                    recovered += 1
                    successful_action_counts.append(index)
                    break
        door_recovery[arm] = {"branches": cases, "reached_original_blocked_room_within_5_environment_actions": recovered,
                              "mean_environment_actions_when_recovered": mean(successful_action_counts)}
    result["descriptive_closed_door_recovery"] = door_recovery
    result["interpretation"] = (
        "Same immediate repairer with/without frozen history; " if protocol.get("memory_in_repair") else ""
    ) + "Conditional local recovery on explicit " + protocol.get("split", "dev") + " failures; not full-episode or lifetime equal-budget proof"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=ROOT / "results/20260916_memory_vs_direct_v2")
    parser.add_argument("--historical", action="store_true")
    parser.add_argument("--resume", type=Path, help="Replace every interrupted checkpoint as a full paired block")
    args = parser.parse_args()
    result = {"pilot": pilot(args.pilot, args.resume)}
    if args.historical:
        result["historical"] = historical()
    print(json.dumps(result, ensure_ascii=False, indent=2))

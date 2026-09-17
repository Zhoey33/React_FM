"""Budget-capped, paired ScienceWorld pilot on declared evaluation episodes.

Credentials are read only from SILICONFLOW_API_KEY. No memory is updated.
All arms share the same actor, state, prompt history, and monetary allowance.
The additional direct-repair call counts against that allowance.

Defaults reproduce the initial four-arm pilot. The corrected conditional
comparison uses --arms direct memory --repair-interface-context
--guidance-until-action --only-memory-hits with a separate output directory.
That directory must contain the shared collected checkpoints.jsonl.
For the history-information comparison, use --memory-in-repair --interface-guide
--repair-interface-context --guidance-until-action --arms direct memory.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openai import OpenAI
from prompts.scienceworld_prompts import SYSTEM_PROMPT_BASE, build_baseline_user_prompt
from src.memory import FailureMemoryStore
from src.scienceworld_env import ScienceWorldEnv
from src.scienceworld_failure_detector import ScienceWorldFailureDetector
from src.scienceworld_retrieval_gate import select_retrieval_memory

TASKS = ["boil", "melt", "power-component", "test-conductivity", "grow-plant", "chemistry-mix"]
ARMS = ["continue", "cue", "direct", "memory"]
MEMORY = ROOT / "memory_store/20260416_163135/sw_epoch1.json"
# Official SiliconFlow pricing, checked 2026-09-16. No cache/night discount assumed.
INPUT_CNY_PER_M = 3.0
OUTPUT_CNY_PER_M = 9.0
IO_LOCK = threading.RLock()

# Verified against ScienceWorld get_possible_actions(), plus observed menu behavior.
INTERFACE_GUIDE = """Action templates (replace OBJ with an observed object referent):
look around; inventory; look at OBJ; look in OBJ; read OBJ;
open OBJ; close OBJ; go OBJ; pick up OBJ; put down OBJ;
move OBJ to OBJ; pour OBJ in OBJ; dunk OBJ in OBJ; mix OBJ;
activate OBJ; deactivate OBJ; connect OBJ to OBJ; disconnect OBJ;
use OBJ on OBJ; focus on OBJ; wait.
When an observation asks for a numbered choice, output only the chosen number
(e.g. 0), with no action words. Use numbers from the current menu only.
Use move OBJ to OBJ to place an object in/on a container. put down OBJ drops it
in the current room. Mix a container, not a list of ingredient names.
Use an object's referent, not descriptive filler: open freezer, not open freezer
door; read recipe when the observed object is a recipe. Prefer look at OBJ to examine.
Do not invent objects, observations, or additional command syntax."""


def actor_instructions(interface_guide=False):
    if not interface_guide:
        return SYSTEM_PROMPT_BASE
    return re.sub(r"(?m)^Valid actions:.*$", lambda _: INTERFACE_GUIDE, SYSTEM_PROMPT_BASE)



def read_rows(path):
    with IO_LOCK:
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def append(path, row):
    with IO_LOCK, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


class BudgetExhausted(Exception):
    pass


class Calls:
    def __init__(self, directory, model, total_cap, repair_interface_context=False, interface_guide=False):
        self.lock = threading.RLock()
        self.path = directory / "calls.jsonl"
        self.model = model
        self.total_cap = total_cap
        self.repair_interface_context = repair_interface_context
        self.interface_guide = interface_guide
        self.calls = {}
        for row in read_rows(self.path):
            self.calls[row["call_id"]] = row
        self.client = OpenAI(api_key=os.environ["SILICONFLOW_API_KEY"],
                             base_url="https://api.siliconflow.cn/v1",
                             max_retries=0, timeout=60)

    def spent(self, group=None):
        with self.lock:
            return sum(x["accounted_cny"] for x in self.calls.values()
                       if group is None or x["group"] == group)

    def generate(self, prompt, group, cap, *, repair=False, system_override=None):
        actor_system = actor_instructions(self.interface_guide)
        system = ("You advise a science experiment agent. Diagnose the failed action using only "
                  "the supplied state and history. Give a short repair rationale and one valid "
                  "next action. Do not invent observations or historical experience.") if repair else (
                      actor_system + "\nNever output two think commands consecutively. "
                      "If your last turn was a think command, now execute one environment action.")
        if repair and self.repair_interface_context:
            system += ("\nThe acting agent receives these interface instructions:\n" +
                       actor_system + "\nFor this advisory call, provide the repair rationale "
                       "and suggested next action, rather than acting yourself.")
        if system_override is not None:
            system = system_override
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        max_tokens = 192 if repair else 128
        # Byte bound plus generous chat overhead; unresolved calls retain their reservation.
        input_bound = sum(len(x["content"].encode("utf-8")) for x in messages) + 1024
        reserve = (input_bound * INPUT_CNY_PER_M + max_tokens * OUTPUT_CNY_PER_M) / 1e6
        with self.lock:
            if self.spent() + reserve > self.total_cap or self.spent(group) + reserve > cap:
                raise BudgetExhausted(group)
            call_id = len(self.calls)
            row = {"call_id": call_id, "group": group, "model": self.model,
                   "status": "reserved", "accounted_cny": reserve,
                   "messages": messages, "max_tokens": max_tokens, "repair": repair}
            self.calls[call_id] = row
            append(self.path, row)
        started = time.monotonic()
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0,
                max_tokens=max_tokens, stop=None if repair else ["\n"],
                extra_body={"enable_thinking": False},
            )
            if response.usage is None:
                raise RuntimeError("Provider returned no usage; reservation retained")
            usage = response.usage.model_dump()
            cost = (usage["prompt_tokens"] * INPUT_CNY_PER_M +
                    usage["completion_tokens"] * OUTPUT_CNY_PER_M) / 1e6
            with self.lock:
                row.update(status="complete", usage=usage, accounted_cny=cost,
                           content=response.choices[0].message.content or "",
                           finish_reason=response.choices[0].finish_reason,
                           latency_s=round(time.monotonic() - started, 3))
                append(self.path, row)
            if cost > reserve:
                raise RuntimeError("Usage exceeded reservation; stop to review budget")
            return row["content"].strip()
        except Exception as exc:
            if row["status"] != "complete":
                with self.lock:
                    row.update(status="uncertain", error_type=type(exc).__name__)
                    append(self.path, row)
            raise


def normalized(text):
    # Room descriptions enumerate unordered objects. Preserve each object and
    # its containment phrase, sorting only top-level list members.
    def sort_members(value):
        pieces, start, depth = [], 0, 0
        for i, char in enumerate(value):
            depth += (char == "(") - (char == ")")
            if char == "," and depth == 0:
                pieces.append(value[start:i].strip())
                start = i + 1
        pieces.append(value[start:].strip())
        return ", ".join(sorted(pieces))
    lines = text.splitlines()
    if text.startswith("This room is called"):
        for i, line in enumerate(lines):
            if " is: " in line and line.rstrip().endswith("."):
                head, tail = line.split(" is: ", 1)
                lines[i] = head + " is: " + sort_members(tail.rstrip()[:-1]) + "."
        start = 0
        while start < len(lines):
            if not lines[start].startswith("\t"):
                start += 1
                continue
            end = start + 1
            while end < len(lines) and lines[end].startswith("\t"):
                end += 1
            lines[start:end] = sorted(lines[start:end])
            start = end
    return " ".join("\n".join(lines).split())


def snapshot(env, score):
    return {"score": score, "look": env.env.look(), "inventory": env.env.inventory()}


def same_state(left, right):
    return left["score"] == right["score"] and all(
        normalized(left[k]) == normalized(right[k]) for k in ["look", "inventory"])


def actor_prompt(task, initial, history, state, guidance=""):
    prompt = build_baseline_user_prompt(task, initial, history[-10:])
    prompt += "\nCurrent room:\n" + state["look"] + "\nInventory:\n" + state["inventory"]
    if guidance:
        prompt += "\n\n" + guidance
    if history and history[-1][0].lower().startswith(("think:", "think ")):
        prompt += "\nYour previous turn was thinking. Now output one executable environment action, not another thought."
    return prompt + "\nOutput one action or think command.\n> "


def action_from(text):
    return text.strip().split("\n")[0].removeprefix("> ").strip()


def execute(env, action, score):
    if action.lower().startswith(("think:", "think ")):
        return "OK.", score, False
    # ScienceWorld can reorder ambiguity-menu numbers across resets. Preserve
    # the recorded referent, not the unstable numeric label, during replay.
    mapped_action = getattr(env, "_pilot_choice_map", {}).get(action, action)
    env._pilot_choice_map = {}
    obs, _, done, info = env.step(mapped_action)
    return obs, info["score"], done


def menu_choices(observation):
    if not observation.startswith("Ambiguous request:"):
        return {}
    return dict(re.findall(r"(?m)^(\d+):\s*(.+)$", observation))


def align_replay_observation(expected, actual):
    before, after = menu_choices(expected), menu_choices(actual)
    if before and after and len(before) == len(after) and len(set(before.values())) == len(before) and set(before.values()) == set(after.values()):
        reverse = {value: key for key, value in after.items()}
        return {key: reverse[value] for key, value in before.items()}
    if normalized(expected) != normalized(actual):
        raise ValueError("Observation mismatch beyond ambiguity-menu numbering")
    return {}


def collect(args, calls, env):
    finished = {x["episode_id"] for x in read_rows(args.output / "episodes.jsonl")}
    existing = {x["checkpoint_id"] for x in read_rows(args.output / "checkpoints.jsonl")}
    # Interleave task families; no selection using branch outcomes or retrieval hits.
    schedule = sorted(enumerate(env._schedule), key=lambda x: (x[1][1], TASKS.index(x[1][0])))
    skip = getattr(args, "skip_variations", 0)
    if skip:
        excluded = {pair for task in TASKS for pair in [x for x in env._schedule if x[0] == task][:skip]}
        schedule = [(i, pair) for i, pair in schedule if pair not in excluded]
    for schedule_idx, (task, variation) in schedule[:args.max_episodes]:
        eid = f"{task}__{variation}"
        if eid in finished:
            continue
        initial, _, _ = env.reset_to_episode(schedule_idx)
        history, steps, kinds = [], [], set()
        score = 0
        status = "horizon"
        for step in range(args.collect_steps):
            state = snapshot(env, score)
            try:
                action = action_from(calls.generate(actor_prompt(task, initial, history, state),
                                                   "collect/" + eid, .6))
            except BudgetExhausted:
                status = "budget"
                break
            if not action:
                raise RuntimeError("Empty actor response")
            obs, score, done = execute(env, action, score)
            history.append((action, obs))
            steps.append({"action": action, "observation": obs, "score": score})
            # Explicit environment failures only. Repeated waits are not treated as errors.
            det = ScienceWorldFailureDetector(enable_implicit_failures=False).detect(obs, action, [])
            excluded_door = getattr(args, "skip_closed_door", False) and obs.strip() == "The door is not open."
            if det.is_failure and not done and det.failure_type not in kinds and not excluded_door:
                kinds.add(det.failure_type)
                cid = f"{eid}__{step}"
                cp = {"checkpoint_id": cid, "episode_id": eid, "task": task,
                      "variation": variation, "split": args.split, "initial": initial,
                      "steps": list(steps), "state": snapshot(env, score),
                      "failure_type": det.failure_type}
                if cid not in existing:
                    append(args.output / "checkpoints.jsonl", cp)
                    existing.add(cid)
                if len(kinds) == args.checkpoints_per_episode:
                    status = "checkpoint_limit"
                    break
            if done:
                status = "terminal"
                break
        append(args.output / "episodes.jsonl", {"episode_id": eid, "task": task,
               "variation": variation, "steps": steps, "status": status,
               "cost_cny": calls.spent("collect/" + eid)})
        print(json.dumps({"collected": eid, "checkpoints": len(kinds),
                          "cost_cny": round(calls.spent(), 4)}), flush=True)


def restore(env, cp):
    env.env.load(cp["task"], cp["variation"])
    env.env.reset()
    env._pilot_choice_map = {}
    score = 0
    for i, step in enumerate(cp["steps"]):
        obs, score, done = execute(env, step["action"], score)
        if score != step["score"] or done:
            raise ValueError(f"Replay mismatch at prefix step {i}")
        if not step["action"].lower().startswith(("think:", "think ")):
            env._pilot_choice_map = align_replay_observation(step["observation"], obs)
    if not same_state(snapshot(env, score), cp["state"]):
        raise ValueError("Replay room/inventory/score mismatch")


def choose_memory(store, cp):
    last = cp["steps"][-1]
    result = store.retrieve(query_action=last["action"], query_observation=last["observation"],
                            task_type=cp["task"], candidate_k=5, return_scores=True,
                            query_failure_type=cp["failure_type"])
    # Older training entries lack failure_type: infer from their recorded environment error.
    # Annotate copies so one query cannot change later BM25 corpus text.
    entries = [replace(entry) for entry in result.entries]
    for entry in entries:
        if not entry.failure_type:
            entry.failure_type = ScienceWorldFailureDetector().detect(
                entry.failure_observation, entry.failure_action, []).failure_type
    decision = select_retrieval_memory(entries=entries, retrieval_scores=result.rrf_scores,
        current_failure_type=cp["failure_type"], failed_action=last["action"],
        failure_observation=last["observation"], recent_actions=[s["action"] for s in cp["steps"]],
        relevance_score_threshold=.45)
    entry = decision.selected_entry
    text = "" if entry is None else (
        "Past failed action: " + entry.failure_action + "\nPast observation: " +
        entry.failure_observation + "\nHistorical repair: " + entry.get_repair_display())
    return {"memory_id": entry.memory_id if entry else None, "text": text[:1400],
            "rejection": decision.rejection_reason, "candidate_ids": decision.candidate_memory_ids,
            "relevance_scores": decision.relevance_scores}


def branch(args, calls, env, partition=0):
    plan_path = getattr(args, "retrieval_plan", None)
    plan = json.loads(plan_path.read_text()) if plan_path else None
    store = None
    if plan is None:
        store = FailureMemoryStore(scope="task_type", retrieval_mode=args.retrieval)
        store.load(str(MEMORY))
    completed = {x["group"] for x in read_rows(args.output / "branches.jsonl")}
    rejected = {x["checkpoint_id"] for x in read_rows(args.output / "rejected.jsonl")}
    checkpoints = read_rows(args.output / "checkpoints.jsonl")[:args.max_checkpoints]
    for cp_index, cp in enumerate(checkpoints):
        if cp_index % args.workers != partition:
            continue
        if cp["checkpoint_id"] in rejected:
            continue
        automatic_memory = plan[cp["checkpoint_id"]]["automatic"] if plan else choose_memory(store, cp)
        memory = automatic_memory
        if args.only_memory_hits and memory["memory_id"] is None:
            continue
        for replay in range(args.replays):
            arms = list(args.arms)
            random.Random(args.seed + checkpoints.index(cp) * 101 + replay).shuffle(arms)
            for arm in arms:
                memory = plan[cp["checkpoint_id"]]["curated"] if arm == "curated" else automatic_memory
                group = f"branch/{cp['checkpoint_id']}/{replay}/{arm}"
                if group in completed:
                    continue
                try:
                    restore(env, cp)
                except ValueError as exc:
                    append(args.output / "rejected.jsonl", {"checkpoint_id": cp["checkpoint_id"],
                           "group": group, "reason": str(exc)})
                    rejected.add(cp["checkpoint_id"])
                    break
                history = [(s["action"], s["observation"]) for s in cp["steps"]]
                score = cp["state"]["score"]
                guidance = ""
                failure_cue = "The previous action failed. Reconsider your approach using the observed state."
                if arm == "cue":
                    guidance = failure_cue
                elif arm == "memory":
                    guidance = failure_cue + ("\n" + memory["text"] if memory["text"] else "")
                    guidance += "\nUse historical advice only if applicable to the current state."
                steps, advice = [], ""
                status = "horizon"
                try:
                    if arm == "direct" or (arm in ["memory", "curated"] and args.memory_in_repair):
                        repair_request = actor_prompt(cp["task"], cp["initial"], history,
                                                      cp["state"], failure_cue)
                        if arm in ["memory", "curated"] and args.memory_in_repair and memory["text"]:
                            repair_request += ("\nHistorical experience, to use only if applicable:\n" +
                                               memory["text"])
                        advice = calls.generate(repair_request + "\nDiagnose this failure and suggest a repair.",
                                                group, args.branch_cny, repair=True)
                        guidance = failure_cue + "\nCurrent-state repair advice:\n" + advice
                    advice_pending = True
                    environment_actions = 0
                    for step in range(args.branch_steps):
                        prompt = actor_prompt(cp["task"], cp["initial"], history,
                                              snapshot(env, score), guidance if (step == 0 or (args.guidance_until_action and advice_pending)) else "")
                        action = action_from(calls.generate(prompt, group, args.branch_cny))
                        if not action:
                            raise RuntimeError("Empty actor response")
                        obs, score, done = execute(env, action, score)
                        if not action.lower().startswith(("think:", "think ")):
                            advice_pending = False
                            environment_actions += 1
                        history.append((action, obs))
                        steps.append({"action": action, "observation": obs, "score": score,
                                      "cumulative_cost_cny": calls.spent(group)})
                        if done:
                            status = "terminal"
                            break
                        if getattr(args, "environment_actions", 0) and environment_actions >= args.environment_actions:
                            status = "action_horizon"
                            break
                except BudgetExhausted:
                    status = "budget"
                row = {"group": group, "checkpoint_id": cp["checkpoint_id"],
                       "episode_id": cp["episode_id"], "task": cp["task"], "arm": arm,
                       "replay": replay, "failure_type": cp["failure_type"],
                       "score_before": cp["state"]["score"], "score_after": score,
                       "score_max": max([cp["state"]["score"]] + [s["score"] for s in steps]),
                       "status": status, "steps": steps, "memory": memory, "advice": advice,
                       "cost_cny": calls.spent(group)}
                append(args.output / "branches.jsonl", row)
                completed.add(group)
                print(json.dumps({"finished": group, "score": score,
                      "cost_cny": round(calls.spent(), 4), "status": status}), flush=True)
            if cp["checkpoint_id"] in rejected:
                break


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--skip-variations", type=int, default=0)
    p.add_argument("--skip-closed-door", action="store_true")
    p.add_argument("--interface-guide", action="store_true")
    p.add_argument("--memory-in-repair", action="store_true")
    p.add_argument("--arms", nargs="+", choices=ARMS + ["curated"], default=ARMS)
    p.add_argument("--retrieval-plan", type=Path)
    p.add_argument("--environment-actions", type=int, default=0)
    p.add_argument("--repair-interface-context", action="store_true")
    p.add_argument("--guidance-until-action", action="store_true")
    p.add_argument("--only-memory-hits", action="store_true")
    p.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    p.add_argument("--phase", choices=["collect", "branch"], required=True)
    p.add_argument("--output", type=Path, default=ROOT / "results/20260916_memory_vs_direct_v2")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    p.add_argument("--total-cny", type=float, default=90)
    p.add_argument("--branch-cny", type=float, default=.12)
    p.add_argument("--collect-steps", type=int, default=40)
    p.add_argument("--branch-steps", type=int, default=60)
    p.add_argument("--variations", type=int, default=2)
    p.add_argument("--max-episodes", type=int, default=12)
    p.add_argument("--max-checkpoints", type=int, default=24)
    p.add_argument("--checkpoints-per-episode", type=int, default=2)
    p.add_argument("--replays", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260916)
    p.add_argument("--retrieval", choices=["hybrid", "bm25_only"], default="hybrid")
    args = p.parse_args()
    if args.memory_in_repair and (not args.repair_interface_context or not args.guidance_until_action):
        p.error("memory-in-repair requires symmetric interface context and guidance-until-action")
    if "curated" in args.arms and (not args.retrieval_plan or not args.memory_in_repair):
        p.error("curated requires a fixed retrieval plan and the shared repairer")
    if not 0 < args.total_cny <= 90:
        p.error("Keep total pilot ceiling <= 90 CNY; user budget is 100 including overhead")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ["phase", "max_episodes", "max_checkpoints", "workers"]
                and (k not in ["retrieval_plan", "environment_actions", "skip_variations", "skip_closed_door"] or v)
                and (k not in ["repair_interface_context", "guidance_until_action", "only_memory_hits", "interface_guide", "memory_in_repair"] or v)}
    protocol.update(protocol_version=2, no_consecutive_think=True,
                    memory_source=str(MEMORY), split=args.split, tasks=TASKS, arms=args.arms,
                    input_cny_per_m=INPUT_CNY_PER_M, output_cny_per_m=OUTPUT_CNY_PER_M,
                    memory_read_only=True, offline_memory_cost_included_in_branch=False)
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            p.error("Protocol differs from saved run; use a different output directory")
    else:
        protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    calls = Calls(args.output, args.model, args.total_cny, args.repair_interface_context, args.interface_guide)
    env = ScienceWorldEnv(task_names=TASKS, split=args.split, max_variations_per_task=args.variations,
                          env_step_limit=100)
    try:
        env.setup()
        # Evaluation episodes must be disjoint from frozen-memory training episodes.
        for task in TASKS:
            env.env.load(task, 0)
            evaluation = env.env.get_variations_test() if args.split == "test" else env.env.get_variations_dev()
            if set(evaluation) & set(env.env.get_variations_train()):
                raise RuntimeError("Train/evaluation overlap")
            if args.phase == "branch":
                for cp in read_rows(args.output / "checkpoints.jsonl"):
                    if cp["task"] == task and (cp["split"] != args.split or cp["variation"] not in evaluation):
                        raise RuntimeError("Checkpoint outside declared evaluation split")
        if args.phase == "collect":
            collect(args, calls, env)
        elif args.workers == 1:
            branch(args, calls, env)
        else:
            def worker(partition):
                worker_env = ScienceWorldEnv(task_names=TASKS, split=args.split,
                    max_variations_per_task=args.variations, env_step_limit=100)
                try:
                    worker_env.setup()
                    branch(args, calls, worker_env, partition)
                finally:
                    worker_env.close()
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(worker, i) for i in range(args.workers)]
                for future in futures:
                    future.result()
    finally:
        env.close()
        print(json.dumps({"accounted_cny": round(calls.spent(), 6)}), flush=True)


if __name__ == "__main__":
    main()

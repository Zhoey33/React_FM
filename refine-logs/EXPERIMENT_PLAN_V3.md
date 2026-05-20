# Experiment Plan V3: Interface-Aligned Causal Memory

## Goal

Validate whether ScienceWorld performance is currently bottlenecked more by **interface/actionability** than by **memory retrieval**, and whether causalized memory fixes that bottleneck better than a learned gate.

## Core Experimental Blocks

### Block A. Reproduce current anchors

Run and freeze these as comparison anchors:

- ReAct baseline
- current best online React_FM ExpA
- Reflexion baseline

Use the same 12-task ScienceWorld protocol already used in `doc/scienceworld_step4_summary_cn.md`.

### Block B. Interface Adapter Only

Add task-family-specific interface guidance to the current ScienceWorld prompt, but keep memory off.

**Purpose**:
- test the ALIGN-style hypothesis directly
- measure how much of the current failure comes from action grammar / precondition mismatch

**Primary readout**:
- overall success
- per-task success
- invalid-action rate
- repeated-failure rate

### Block C. Causal Memory Only

Keep the existing retrieval system, but replace current repair text with structured causal memory.

Compare:

- current memory text
- causal `question`
- causal `repair_template`

**Purpose**:
- isolate whether better memory abstraction helps without changing interface

### Block D. Interface + Causal Memory

Combine Block B and Block C.

This is the main proposed system.

### Block E. Optional Subgoal Buffer

Only run if Block D is already positive.

Add:

- current subgoal
- blocker
- next intended action

**Purpose**:
- test whether a small SwiftSage-lite planning state adds extra value beyond alignment + causal memory

## Must-Run Ablations

1. Baseline vs interface-only
2. Baseline vs causal-memory-only
3. Current React_FM memory vs causal memory
4. Interface-only vs interface + causal memory
5. Main system vs current learned gate setup

## Task-Level Focus

Prioritize detailed diagnosis on:

- `power-component`
- `test-conductivity`
- `test-conductivity-of-unknown-substances`
- `chemistry-mix`
- `mendelian-genetics-known-plant`

These are the tasks most likely to reveal whether the failure is about interface mismatch, missing causal preconditions, or both.

## Metrics

- success rate
- average score
- average tokens per episode
- invalid-action rate
- repeated-failure rate
- memory retrieval hit rate
- memory usage rate
- per-task deltas vs baseline

## First 3 Runs To Launch

1. `interface-only` on ScienceWorld current 12-task protocol
2. `causal-memory-only` on the same protocol
3. `interface + causal-memory` on the same protocol

## Implementation Order

1. Prompt-layer interface adapter in `prompts/scienceworld_prompts.py`
2. Structured causal fields in `src/scienceworld_memory_extractor.py`
3. Minimal memory match-and-inject logic in `experiments/run_scienceworld.py`
4. Analysis script to compare per-task deltas against current Step 4 summary

## Kill Criteria

Stop this line if either condition holds:

- interface-only gives near-zero effect and causal-memory-only also gives near-zero effect
- gains appear only on one easy task family and disappear in aggregate

If killed, the next pivot should be plan-state memory or rollout-based policy distillation, not another gate revision.

## Expected Paper Positioning

- Main message:
  raw failure memory underperforms in interactive science tasks because it is both under-abstracted and misaligned to the environment interface
- Strongest ablation story:
  retrieval coverage was already high, but performance moved only after changing the *form* of memory and the *interface* seen by the agent

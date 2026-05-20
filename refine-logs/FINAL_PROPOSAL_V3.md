# Research Proposal V3: Interface-Aligned Causal Memory for ScienceWorld

## Problem Anchor

- **Bottom-line problem**: In ScienceWorld, your current React_FM variants already show that memory can help, but the benefit is small and unstable because retrieved memories are often injected in a form that is not locally actionable. The system usually knows *something relevant*, but that knowledge is expressed in the wrong interface, the wrong level of abstraction, or the wrong phase of the task.
- **Must-solve bottleneck**: The main error is no longer “missing memory”. It is a combination of:
  - action-interface mismatch
  - hidden precondition mismatch
  - over-literal memory phrasing
  - missing short-horizon plan state
- **Non-goal**:
  - not a learned gate paper
  - not a new retrieval paper
  - not an RL/world-model training project
  - not a benchmark-wide claim of SOTA over all modern RL agents

## Method Thesis

**Thesis**: Failure memories become much more useful in ScienceWorld when they are converted from verbatim fixes into **causal precondition schemas**, then surfaced through an **interface-aligned prompt layer** that exposes the correct action grammar, object roles, and local constraints of the environment.

## Dominant Contribution

Turn failure memory from:

- `"failure: X, fix: Y"`

into:

- `trigger`: what kind of local failure pattern is happening
- `missing_precondition`: what must be true before the blocked action can work
- `causal_rule`: reusable relation between state and fix
- `repair_template`: environment-valid action form
- `negative_constraint`: what not to do again

Then inject it through an interface adapter that normalizes action templates and object references for the current ScienceWorld task family.

## Proposed System

### 1. Interface Adapter

For each ScienceWorld task family, provide a compact context block with:

- valid action templates that matter for the family
- common object-role mappings
- common precondition reminders
- syntax guardrails extracted from environment conventions

This is inspired by ALIGN, but should use only information already available from the environment wrapper, prompt templates, and valid action API.

### 2. Causal Memory Rewriter

Extend the current extractor output with:

- `failure_trigger`
- `missing_precondition`
- `causal_rule`
- `repair_template`
- `negative_constraint`

Reuse the existing extractor path in `src/scienceworld_memory_extractor.py`; do not build a new memory subsystem.

### 3. Match-and-Inject Controller

Initial version should be simple:

- match current failure against memory triggers/preconditions
- inject only top-1 matched causal memory
- choose one of:
  - `none`
  - `question`
  - `repair_template`

This controller can be heuristic in V1. Do not center the contribution on training the selector.

### 4. Optional Plan Buffer

Maintain a tiny explicit `subgoal buffer`:

- current phase
- next 1-2 intended actions
- blocking condition

Only add this after the interface + causal memory fusion is validated.

## Expected Mechanism of Improvement

- On `power-component` and conductivity tasks:
  interface alignment should reduce action-format and affordance confusion.
- On `chemistry-mix` and genetics tasks:
  causal memory should help the model recover missing preconditions and action order.
- On tasks where baseline already wins:
  the system should abstain from verbose memory injection and preserve baseline behavior.

## Main Claims

### Claim 1

Interface alignment alone improves ScienceWorld success/score over the current no-gate React baseline on interface-sensitive tasks.

### Claim 2

Causalized memory beats the current verbatim failure-recovery memory at the same retrieval budget.

### Claim 3

Interface alignment and causal memory are complementary; the combination outperforms either alone.

### Claim 4

Most of the gain is concentrated on the task clusters where your current React_FM is unstable, which explains why raw memory currently has limited net benefit.

## Why this is better than the gate headline

- It aligns with both your local evidence and recent literature.
- It gives a more generalizable contribution.
- It avoids overcommitting to a fragile routing mechanism.
- It preserves your existing code and experimental assets.

## Key Risks

- The method could collapse into prompt engineering if the structured memory fields are not genuinely different from current repairs.
- Interface hints could look unfair unless carefully constrained.
- The gains may be narrow unless the task-family templates are well designed.

## Success Condition

- On the current 12-task ScienceWorld protocol:
  - clear improvement over baseline and current React_FM on at least 3 of the current hard task families
  - positive aggregate gain without relying on a learned gate
- On ALFWorld:
  - small but consistent transfer gain from causalized memory, even if interface alignment is weaker there

## Recommended Scope

- Primary benchmark: ScienceWorld
- Secondary transfer check: ALFWorld
- Gate: keep only as an appendix/ablation

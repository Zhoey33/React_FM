# Round 1 Refinement

## Problem Anchor
[Verbatim from Round 0]
- **Bottom-line problem**: Memory-augmented ReAct agents suffer from the *intervention paradox*: injecting retrieved failure memories can help failing trajectories but also disrupts trajectories that would have self-recovered. Current systems inject unconditionally, with no mechanism to decide *when* and *how much* to intervene.
- **Must-solve bottleneck**: No learned policy selects the minimum effective intervention from retrieved agent memory at failure checkpoints.
- **Non-goals**: Not proposing new storage/retrieval/finetuning. Not claiming "counterfactual" in the causal sense.
- **Constraints**: Mac M4, ~$200 API, 23 days, lightweight tabular gate.
- **Success condition**: Gate beats always-repair on 2/3 benchmarks; arm heterogeneity exists.

## Anchor Check
- **Original bottleneck**: Unconditional memory injection causes disruption-recovery tradeoff
- **Why revised method still addresses it**: We now formalize the gate as a utility-based policy with cleaner learning target, and address the replay fidelity concern head-on
- **Reviewer suggestions rejected as drift**: None — all suggestions sharpen the method

## Simplicity Check
- **Dominant contribution**: Utility-based intervention policy trained from matched branched rollouts
- **Components removed**: Manual escalation rule (subsumed into features), SHAP as main evidence (replaced with oracle frequencies), cognitive-load budgeting as separate component (subsumed into features)
- **Reviewer suggestions rejected as unnecessary complexity**: None
- **Why the remaining mechanism is still the smallest adequate route**: One tabular regressor, no RL, no finetuning, reuses all React_FM infrastructure

## Changes Made

### 1. Training Target: Best-Arm Classification → Utility Regression
- **Reviewer said**: Mixes multiclass classification and value-based selection; make learner predict arm utility explicitly
- **Action**: Changed to train `f(x, arm) → expected utility`, where utility = `episode_recovery - α·token_cost - β·disruption`. At inference: `argmax_arm f(x, arm)`. This is cleaner because it produces a calibrated value estimate, not just a label.
- **Impact**: More principled learning target; enables cost-sensitivity tuning via α, β without retraining

### 2. Question Arm: Handwritten → LLM-Canonicalized
- **Reviewer said**: Define cue/question/repair via deterministic transforms of existing memory; if question can't be reliable, drop it
- **Action**: At memory extraction time, use the base LLM (one offline call per memory entry) to produce both a `question_text` and `repair_text` variant from the stored failure-recovery triple. This is a natural FM-era use and eliminates brittle prompt templates.
- **Impact**: `question` arm becomes a stored field in the memory entry, not a runtime derivation

### 3. Escalation Rule → Removed
- **Reviewer said**: Recurrence is already in the features; let the gate learn escalation implicitly
- **Action**: Removed explicit escalation rule. Added `same_failure_recurrence_count` as feature #11. The gate can learn to escalate from question→repair when recurrence is high.
- **Impact**: One fewer heuristic; cleaner system

### 4. Cue Arm → Control Baseline Only
- **Reviewer said**: Move `cue` out of learned action space if memory format can't support clean 4-arm story
- **Action**: Keep `cue` in branched rollout data collection (for scientific analysis), but the gate selects from {none, question, repair} only. `cue` becomes a control condition in the checkpoint benchmark. This simplifies the gate while preserving the experimental control.
- **Impact**: Gate is now 3-arm (simpler), `cue` serves as experimental control (stronger analysis)

### 5. Replay Validation Protocol
- **Reviewer said**: Build replay validation first; if checkpoint restoration is noisy, remove that benchmark
- **Action**: Added explicit replay validation step: before collecting branched data, run 10 checkpoints × 2 replays with `none` arm and measure outcome agreement rate. If agreement < 80%, that benchmark is excluded from gate training (can still be used for online evaluation with a simpler threshold gate).
- **Impact**: Addresses the critical feasibility concern head-on

### 6. SHAP → Oracle Arm Frequencies
- **Reviewer said**: Drop SHAP as main evidence; use oracle-arm frequencies and conditional win rates
- **Action**: Main analysis uses: (a) oracle arm distribution per benchmark, (b) conditional win rates (when oracle=none, what does always-repair score?), (c) success-cost Pareto frontier. SHAP moved to appendix if space permits.
- **Impact**: Cleaner, more interpretable analysis

---

## Revised Proposal

# Learning Minimal Memory Interventions for LLM Agents

## Problem Anchor
[Same as above — verbatim]

## Technical Gap
[Same as Round 0 — unchanged]

## Method Thesis

- **One-sentence thesis**: We learn a lightweight utility-based intervention policy that selects the minimum effective memory intervention at each failure checkpoint, resolving the disruption-recovery tradeoff in memory-augmented ReAct agents.

- **Why this is the smallest adequate intervention**: One tabular regressor over 11 observable features. No RL, no finetuning. Reuses all React_FM infrastructure.

## Contribution Focus

- **Dominant contribution**: Formulation of memory injection as a utility-based contextual decision problem with graded arms {none, question, repair}, trained from matched branched rollouts at failure checkpoints.

- **Supporting contribution**: Empirical evidence that the optimal intervention arm varies by context, with `cue` as an experimental control showing that memory content (not just attention-focusing) drives recovery.

- **Explicit non-contributions**: Detection, storage, retrieval — all infrastructure.

## Proposed Method

### Complexity Budget
- **Frozen / reused**: All React_FM components, all LLM backbones
- **New trainable component**: One tabular regressor (XGBoost) for the intervention gate
- **New offline LLM call**: One canonicalization call per memory entry to produce question_text and repair_text variants (~negligible cost)
- **Removed**: Manual escalation rule, surprise gating, cognitive-load budgeting as separate module, SHAP as main analysis

### System Overview

```
Episode execution loop:
  Agent → action → env.step() → observation
                      ↓
           failure_detector.detect()
                      ↓ (if failure detected)
           memory.retrieve(query) → candidate memories (each with question_text + repair_text)
                      ↓
           gate(features) → arm ∈ {none, question, repair}
                      ↓
           build_prompt(arm, top_candidate) → inject into next LLM call
                      ↓ (post-episode)
           extractor → extract triple → LLM canonicalize → store (triple, question_text, repair_text)

Gate training (offline, one-time):
  Step 0: Replay validation — 10 checkpoints × 2 replays, check agreement ≥ 80%
  Step 1: Collect failure checkpoints from React_FM E1 runs
  Step 2: Deduplicate by failure signature → keep diverse pool
  Step 3: Branch each checkpoint into 4 conditions (none, cue, question, repair)
          × matched 5-step continuation budget
  Step 4: Record (features, arm, recovery, disruption, tokens)
  Step 5: Compute utility = recovery - α·tokens - β·disruption
  Step 6: Train XGBoost: f(features, arm) → utility
  Step 7: Deployment policy: π(x) = argmax_{k ∈ {none, question, repair}} f(x, k)
          (cue is collected for analysis but not in the deployment action space)
```

### Core Mechanism: The Intervention Gate

**Input**: 11-dimensional feature vector x at each failure checkpoint:
1. `failure_type` (categorical: explicit/implicit)
2. `task_type` (categorical: benchmark-specific)
3. `step_index` (normalized)
4. `action_repetition_count` (integer)
5. `retrieval_rrf_score` (float: top-1)
6. `retrieval_margin` (float: top-1 minus top-2)
7. `memory_entry_count` (integer)
8. `history_token_count` (integer)
9. `progress_since_last_failure` (binary)
10. `failures_in_last_5_steps` (integer)
11. `same_failure_recurrence_count` (integer: how many times this exact failure signature has appeared in this episode)

**Output**: arm ∈ {none, question, repair}

**Arm definitions**:
- `none`: No injection. Agent continues unassisted.
- `question`: LLM-canonicalized diagnostic question from top memory: e.g., "What precondition might you be missing before [action]?" Stored as `question_text` field in the memory entry.
- `repair`: Full corrective sequence from top memory: e.g., "Previously, [failure] was fixed by [solution]." Stored as `repair_text` field.

**Cue (control only)**: Generic "Reconsider your approach." Collected in branched rollouts for analysis but NOT in the gate's action space.

**Training signal**: Utility regression
```
utility(x, k) = recovery(x, k) - α · normalized_tokens(x, k) - β · disruption(x, k)
```
- `recovery(x, k)` ∈ {0, 1}: did the agent recover within 5 steps after arm k?
- `normalized_tokens(x, k)`: tokens used in continuation, normalized to [0, 1]
- `disruption(x, k)` ∈ {0, 1}: did intervention cause the agent to take a WORSE action than with arm=none?
- α, β tuned on dev checkpoints (grid search)

**Learner**: XGBoost with (x, arm_onehot) → utility. At inference: predict utility for all 3 arms, select argmax.

**Memory canonicalization** (offline, one call per entry):
```
Prompt to base LLM:
  "Given this failure-recovery pair:
   Failed action: {failure_action}
   Observation: {failure_observation}
   Solution: {solution_action}

   Generate:
   1. A diagnostic question (don't reveal the answer): ...
   2. A repair instruction (give the full fix): ..."
```
This produces `question_text` and `repair_text` stored alongside the original triple. Cost: ~100 tokens per entry, ~11K tokens total for 110 ALFWorld entries.

### Replay Validation Protocol

Before branched data collection on each benchmark:
1. Select 10 diverse failure checkpoints
2. For each, replay with arm=none TWICE
3. Compute outcome agreement rate (same recovery outcome both times)
4. If agreement ≥ 80%: proceed with branched rollouts
5. If agreement < 80%: exclude from gate training, use threshold gate for that benchmark

This addresses the critical concern about checkpoint replay fidelity.

### Failure Modes and Diagnostics

| Failure Mode | Detection | Action |
|---|---|---|
| Insufficient checkpoints | < 30 diverse checkpoints per benchmark | Use binary gate (none/repair) for that benchmark |
| Replay infidelity | Agreement < 80% in validation | Exclude from gate training |
| No arm heterogeneity | Oracle arm is always `repair` | Report as finding; paper becomes "always-repair is optimal" |
| question ≈ cue in recovery rate | p > 0.3 in checkpoint benchmark | Reframe: "selective abstention" paper, merge question into lightweight arm |
| Gate ≈ random | Dev accuracy < majority-class baseline | Use threshold gate fallback |

### Novelty and Elegance Argument

[Same as Round 0, plus:]
- The utility formulation makes the contribution more than "add a classifier" — it's a principled decision-theoretic treatment of memory injection
- The matched branched rollout protocol is a reusable methodology for studying any intervention decision in agent systems
- The 3-arm action space (none/question/repair) with `cue` as control is cleaner than the original 4-arm design

## Claim-Driven Validation Sketch

### Claim 1: The learned gate outperforms unconditional injection
- **Experiment**: Online evaluation on ScienceWorld + WebShop test split
- **Systems**: ReAct, always-repair, always-question, binary gate (none/repair), learned gate, oracle gate
- **Metrics**: Success rate, average score, total tokens, disruption rate
- **Control**: `cue` as a baseline in checkpoint analysis (not in gate)
- **Expected**: Learned gate ≥ always-repair in success, with lower disruption/tokens

### Claim 2: Arm heterogeneity exists
- **Experiment**: Checkpoint benchmark analysis
- **Analysis**: Oracle arm distribution per benchmark, conditional win rates, success-cost Pareto
- **Expected**: No single arm > 70% of checkpoints; different benchmarks have different arm profiles

### Claim 3: Graded > binary
- **Experiment**: 3-arm gate vs binary gate (none/repair), same data + classifier
- **Expected**: 3-arm wins when question arm is frequently optimal

### Control: question vs cue
- **Experiment**: In checkpoint benchmark, compare recovery rates of question vs cue arms
- **Expected**: question > cue (diagnostic content matters). If not: fallback framing.

## Compute & Timeline
- Checkpoint collection: ~2,000 API calls (~$8)
- Memory canonicalization: ~500 calls (~$2)
- Online evaluation: ~6 systems × 2 benchmarks × ~80 envs × ~20 steps = ~$40
- Replay validation: ~120 calls (~$1)
- **Total**: ~$50-80
- **Timeline**: 23 days (Week 1: infra + replay validation; Week 2: branched data + gate; Week 3: online + writing)

# Research Proposal: Learning Minimal Memory Interventions for LLM Agents

## Problem Anchor

- **Bottom-line problem**: Memory-augmented ReAct agents suffer from the *intervention paradox*: injecting retrieved failure memories can help failing trajectories but also disrupts trajectories that would have self-recovered, leading to net harm on some benchmarks. Current systems (Reflexion, ExpeL, base React_FM) inject unconditionally whenever memory is available, with no mechanism to decide *when* and *how much* to intervene.

- **Must-solve bottleneck**: There is no learned policy that selects the *minimum effective intervention* from retrieved agent memory at failure checkpoints. Binary intervention (inject or not) is too crude; graded intervention (none / diagnostic question / full repair) is needed, and the optimal arm varies by failure type, task complexity, and agent state.

- **Non-goals**:
  - NOT proposing a new memory storage format (failure-recovery triples from React_FM reused as infrastructure)
  - NOT proposing a new retrieval algorithm (BM25+embedding+RRF reused)
  - NOT claiming universal superiority over episode-level methods
  - NOT doing RL or model finetuning
  - NOT claiming "counterfactual" in the causal inference sense

- **Constraints**: Mac M4, ~$200 API budget, 23 days to NeurIPS 2026 (May 6), lightweight tabular gate only.

- **Success condition**: Gate beats always-repair on at least 2 of {ScienceWorld, WebShop, ALFWorld} in online evaluation (success rate or efficiency), AND arm heterogeneity exists (no single arm > 70% oracle-optimal).

---

## Contribution Focus

- **Dominant contribution**: Formulation of memory injection as a utility-based minimal-intervention decision at failure checkpoints, with graded arms {none, question, repair}, trained from matched branched rollouts.

- **Supporting contribution**: Empirical evidence that the optimal intervention arm varies by failure type and agent state, with `cue` as an experimental control showing memory content (not just attention-focusing) drives recovery.

- **Explicit non-contributions**: Two-tier failure detection, failure-recovery triple format, hybrid retrieval, per-environment isolation — all infrastructure from React_FM.

---

## Proposed Method

### Complexity Budget

- **Frozen / reused**: All React_FM components (detection, memory, retrieval, prompts), all LLM backbones
- **New trainable component**: One XGBoost regressor for the intervention gate (~100 LOC, trains in seconds on CPU)
- **New offline LLM call**: One canonicalization call per memory entry to produce question_text + repair_text variants
- **Removed**: Manual escalation rule, surprise gating, cognitive-load budgeting, token cost in utility

### System Overview

```
Episode execution:
  Agent → action → env.step() → observation
                      ↓
           failure_detector.detect()
                      ↓ (if failure detected)
           memory.retrieve(query) → top-1 candidate
                      ↓
           [if RRF_score < τ_min: bypass → arm=none]
           [else: gate(features) → arm ∈ {none, question, repair}]
                      ↓
           build_prompt(arm, candidate) → inject
                      ↓ (post-episode)
           extractor → triple → LLM canonicalize → store

Gate training (offline, one-time):
  0. Replay validation: 10 checkpoints × 3 replays → agreement ≥ 80%
     Stability pilot: 5 checkpoints × 3 replays × 3 arms → Kendall τ ≥ 0.6
  1. Collect failure checkpoints from React_FM E1
  2. Deduplicate by failure signature
  3. Branch each into 4 conditions (none, cue, question, repair) × 5 steps × 3 replays
  4. Compute replay-averaged progress + disruption per (checkpoint, arm)
  5. utility = progress_avg - β · disruption_avg
  6. Train XGBoost: f(features, arm_onehot) → utility
  7. Deploy: π(x) = argmax_{k ∈ {none,question,repair}} f(x, k)
```

### Core Mechanism: The Intervention Gate

**Input**: 11-dimensional feature vector (NO privileged environment state):
1. failure_type (explicit/implicit)
2. task_type (benchmark-specific categorical)
3. step_index (normalized)
4. action_repetition_count
5. retrieval_rrf_score (top-1)
6. retrieval_margin (top-1 minus top-2)
7. memory_entry_count
8. history_token_count
9. progress_since_last_failure (binary)
10. failures_in_last_5_steps
11. same_failure_recurrence_count

**Output**: arm ∈ {none, question, repair}

**Arm definitions**:
- `none`: No injection. Agent continues unassisted.
- `question`: LLM-canonicalized diagnostic question from top memory. Stored as `question_text` field.
- `repair`: Full corrective sequence from top memory. Stored as `repair_text` field.
- `cue` (control only): "Reconsider your approach." Collected in branched rollouts for scientific analysis, NOT in gate action space.

**Bypass rule**: If top-1 RRF score < τ_min (default 0.1, tuned on dev), bypass gate, select arm=none.

**Training signal**: Utility regression on replay-averaged progress
```
utility(x, k) = progress_avg(x, k) - β · disruption_avg(x, k)
progress_avg(x, k) = mean over N=3 replays of [progress_after_5_steps(arm=k) - progress_at_checkpoint]
disruption_avg(x, k) = mean over N=3 replays of [max(0, progress(x, none) - progress(x, k))]
β tuned on dev checkpoints via grid search. Default β=0.3.
```

**Progress signals** (offline labeling only, NOT inference-time inputs):
- ALFWorld: subgoal completion delta (PDDL state)
- WebShop: reward delta (attribute matching)
- ScienceWorld: normalized score delta (0-100)

**Learner**: XGBoost with (features, arm_onehot) → utility. Argmax over 3 arms at inference.

### Memory Canonicalization (offline, one call per entry)
```
Prompt: "Given this failure-recovery pair:
  Failed action: {failure_action}
  Observation: {failure_observation}
  Solution: {solution_action}
  Generate:
  1. A diagnostic question (don't reveal the answer): ...
  2. A repair instruction (give the full fix): ..."
```
Cost: ~100 tokens per entry, negligible total.

### Replay Validation Protocol
- Phase A: 10 checkpoints × 3 replays with arm=none → binary outcome agreement ≥ 80%
- Phase B: 5 checkpoints × 3 replays for each of {none, question, repair} → Kendall τ of utility ranking ≥ 0.6
- If Phase A fails: exclude benchmark from gate training
- If Phase B fails: use replay-averaged utilities (already default)

---

## Claim-Driven Validation

### Claim 1a (local): Gate-selected arms have higher local recovery + lower disruption than always-repair
- Checkpoint benchmark: per-arm recovery rates + disruption rates
- Control: cue as attention-focusing control

### Claim 1b (global): Local optimization translates to ≥ matched episode-level performance
- Online evaluation: ReAct, always-repair, binary gate, learned gate, oracle gate
- Benchmarks: ScienceWorld, WebShop, ALFWorld (test split)
- Metrics: success_rate, online_disruption_rate, total_tokens, abstention_rate, bypass_rate

### Claim 2: Arm heterogeneity exists
- Oracle arm distribution per benchmark
- Conditional win rates: when oracle=none, what does always-repair score?
- Success-cost Pareto frontier

### Claim 3: Graded > binary (if question is frequently optimal)
- 3-arm vs 2-arm gate, same data + classifier

### Control: question vs cue
- Checkpoint recovery rates for question vs cue
- If question ≈ cue: fallback to selective abstention framing

---

## Online Metrics

For each system × benchmark:
- Success rate (primary)
- Online disruption rate: fraction of episodes where gate activated AND outcome worse than always-repair
- Total tokens (true end-to-end)
- Gate abstention rate: fraction of gate decisions → arm=none
- Bypass rate: fraction of failures where retrieval score < τ_min

---

## Failure Modes

| Mode | Detection | Action |
|---|---|---|
| Too few checkpoints | < 30 per benchmark | Binary gate for that benchmark |
| Replay infidelity | Agreement < 80% | Exclude from gate training |
| No arm heterogeneity | Oracle always repair | Report as finding |
| question ≈ cue | p > 0.3 | Collapse to none/repair; "selective abstention" paper |
| Gate ≈ random | Dev < majority baseline | Threshold gate fallback |

---

## Fallback Framing

If question ≈ cue on dev checkpoints:
- Title: "Learning When to Intervene with Memory in LLM Agents"
- Framing: Selective abstention + minimal repair
- 3-arm collapses to 2-arm (none/repair)
- question reported as negative control result
- Contribution still holds: utility-based intervention learning from matched rollouts

---

## Novelty Statement

We formulate memory injection for LLM agents as a utility-based minimal-intervention decision at failure checkpoints. The methodological contribution is the matched branched rollout protocol for collecting intervention labels, and the formulation of the intervention paradox as a cost-sensitive contextual decision. The contribution is NOT the XGBoost classifier itself, which is standard.

**Closest work**: Self-Regulation [2502.04576] learns WHEN to call a stronger model (binary model-switching). We learn WHAT memory content to inject at WHAT intensity (graded memory dosing). Different action space, objective, and application domain.

---

## Compute & Timeline

| Item | API Calls | Cost |
|---|---|---|
| Replay validation | ~180 | ~$1 |
| Branched rollouts (80 checkpoints × 4 arms × 3 replays × 5 steps) | ~4,800 | ~$15 |
| Memory canonicalization | ~300 | ~$2 |
| Online evaluation (5 systems × 3 benchmarks) | ~15,000 | ~$40 |
| **Total** | **~20,000** | **~$60** |

Timeline: 23 days (see RESEARCH_REVIEW_CDFM.md for week-by-week plan)
Gate training: Seconds on CPU

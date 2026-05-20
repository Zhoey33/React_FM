# Round 0: Initial Proposal (Post-Review Reboot)

## Problem Anchor

- **Bottom-line problem**: Memory-augmented ReAct agents suffer from the *intervention paradox*: injecting retrieved failure memories can help failing trajectories but also disrupts trajectories that would have self-recovered, leading to net harm on some benchmarks. Current systems (Reflexion, ExpeL, base React_FM) inject unconditionally whenever memory is available, with no mechanism to decide *when* and *how much* to intervene.

- **Must-solve bottleneck**: There is no learned policy that selects the *minimum effective intervention* from retrieved agent memory at failure checkpoints. Binary intervention (inject or not) is too crude; graded intervention (none / lightweight hint / diagnostic question / full repair) is needed, and the optimal arm varies by failure type, task complexity, and agent state.

- **Non-goals**:
  - We are NOT proposing a new memory storage format (failure-recovery triples from React_FM are reused as infrastructure)
  - We are NOT proposing a new retrieval algorithm (BM25+embedding+RRF from React_FM is reused)
  - We are NOT claiming universal superiority over episode-level methods (ExpeL is better on some benchmarks)
  - We are NOT doing RL or model finetuning
  - We are NOT claiming "counterfactual" in the causal inference sense

- **Constraints**:
  - Compute: 1 Mac M4 (MPS), API calls to SiliconFlow (~$200 budget)
  - Agent models: Qwen3.5-9B (ALFWorld/WebShop), DeepSeek-V3.2 (ScienceWorld)
  - Timeline: 23 days to NeurIPS 2026 (May 6 deadline)
  - No GPU cluster, no finetuning
  - Gate must be lightweight (tabular classifier, not neural)

- **Success condition**: The learned gate (a) beats unconditional always-repair injection on at least 2/3 benchmarks in online evaluation, AND (b) either achieves higher success rate or matches success rate with meaningfully lower disruption/token cost, AND (c) shows that the optimal intervention arm varies by context (not one arm dominates everywhere).

## Technical Gap

### Why current methods fail

1. **Reflexion/ExpeL**: Inject at episode boundaries only. Cannot correct mid-episode failures. On ScienceWorld, episode-level injection drops to 4.0% (below the 10.0% baseline), because dumping all memories at episode start overwhelms context without targeting the specific failure.

2. **Base React_FM (our prior work)**: Injects unconditionally on every detected failure. This works when the memory is relevant (ScienceWorld +12%, ALFWorld +4.5%), but causes problems when:
   - Retrieved memory is irrelevant (20.8% hit rate on ALFWorld)
   - The agent would have self-recovered without intervention
   - Full repair instructions disrupt the agent's own reasoning chain

3. **Intervention Paradox [2602.03338]**: Shows that a binary LLM critic with strong offline accuracy (AUROC 0.94) can still cause 26pp performance collapse. Their fix is a pre-deployment pilot test — but this is still binary (intervene everywhere or nowhere), not adaptive per-state.

4. **Self-Regulation [2502.04576]**: Learns WHEN to request help from a stronger model, but the intervention is binary (ask for help or don't) and the "help" is calling a more powerful LLM, not injecting graded memory content.

5. **D-MEM [2603.14597]**: Does surprise+utility gating for memory WRITING, but not for memory READING/injection. Routes inputs into memory restructuring, doesn't gate what gets injected at decision time.

### What is missing

No prior system learns a **graded memory intervention policy** over {none, cue, question, repair} at failure checkpoints, trained from **matched branched rollouts** at the same failure state. The field has identified the problem (intervention paradox) but not produced the solution (adaptive graded memory dosing).

### Why naive fixes are insufficient

- "Just inject if retrieval score is high" — fails because relevance ≠ helpfulness; a highly relevant memory can still disrupt a self-recovering trajectory
- "Just use uncertainty" — AUQ uses verbalized uncertainty to trigger reflection, but this doesn't distinguish between different memory intervention intensities
- "Just use a bigger model" — Self-Regulation/ReDAct show this helps, but it's a different problem (model routing, not memory dosing)

## Method Thesis

- **One-sentence thesis**: We learn a lightweight intervention policy that selects the minimum effective memory intervention at each failure checkpoint, resolving the disruption-recovery tradeoff in memory-augmented ReAct agents.

- **Why this is the smallest adequate intervention**: The gate is a single tabular classifier (XGBoost/logistic regression) over 10 observable features. No neural training, no RL, no model finetuning. The gate reuses React_FM's existing detection, memory, and retrieval infrastructure — the only new component is the decision of what to inject.

- **Why this route is timely**: The Intervention Paradox paper (Feb 2026) identified the problem; we provide the solution. The shift from "always inject" to "learn when and how much" is the current frontier in agent memory research.

## Contribution Focus

- **Dominant contribution**: A learned graded intervention policy over {none, cue, question, repair} for failure-recovery memory, trained from branched rollouts at failure checkpoints. This is a new formulation of the agent memory injection problem as a cost-sensitive contextual decision.

- **Supporting contribution**: Empirical evidence that the optimal intervention arm varies by failure type, benchmark, and agent state — validating the need for adaptive intervention over any fixed policy.

- **Explicit non-contributions**: Two-tier failure detection, failure-recovery triple format, hybrid BM25+embedding retrieval, per-environment memory isolation — these are infrastructure from React_FM, presented as the substrate, not claimed as novel.

## Proposed Method

### Complexity Budget

- **Frozen / reused**: React_FM failure detection, memory store, retrieval pipeline, prompt templates, all LLM backbones
- **New trainable component**: One tabular classifier (XGBoost or logistic regression) for the intervention gate — ~100 lines of code, trains in seconds on CPU
- **Tempting additions intentionally not used**:
  - RL-based gate optimization (unnecessary at our data scale)
  - Surprise-gated memory formation (D-MEM occupies this space; defer to future work)
  - Cognitive-load budgeting as a separate component (subsume into gate features instead)
  - Cross-domain transfer claims (gates trained per-benchmark)
  - Neural gate (tabular features + lightweight classifier is sufficient)

### System Overview

```
Episode execution loop:
  Agent → action → env.step() → observation
                      ↓
           failure_detector.detect()
                      ↓ (if failure detected)
           memory.retrieve(query)  → candidate memories
                      ↓
           intervention_gate(state_features, candidates) → arm ∈ {none, cue, question, repair}
                      ↓
           build_prompt(arm, candidates) → inject into next LLM call
                      ↓ (post-episode)
           memory_extractor → memory.add()

Gate training (offline, one-time):
  For each logged failure checkpoint:
    → save environment state
    → branch into 4 arms (none, cue, question, repair)
    → matched continuation budget (5 steps each)
    → record outcome (recovery within 5 steps? disruption?)
    → extract 10 tabular features
  → train XGBoost/logistic regression on (features, best_arm) pairs
```

### Core Mechanism: The Intervention Gate

**Input**: At each failure checkpoint, a 10-dimensional feature vector x:
1. `failure_type` (categorical: explicit/implicit)
2. `task_type` (categorical: benchmark-specific)
3. `step_index` (normalized: how far into the episode)
4. `action_repetition_count` (integer: consecutive same-action count)
5. `retrieval_rrf_score` (float: top-1 RRF score)
6. `retrieval_margin` (float: top-1 minus top-2 score)
7. `memory_entry_count` (integer: how many memories exist for this env)
8. `history_token_count` (integer: current prompt length)
9. `progress_since_last_failure` (binary: did agent advance since last failure?)
10. `failures_in_last_5_steps` (integer: recent failure density)

**Output**: arm ∈ {none, cue, question, repair}

**Arm definitions**:
- `none`: Agent continues without any injection. Appropriate when the agent is likely to self-recover or the retrieved memory is irrelevant.
- `cue`: Generic attention-focusing prompt: "Reconsider your approach to this step." No memory content. Control for the attention-focusing effect of any intervention.
- `question`: Diagnostic question derived from the top-retrieved memory's failure pattern: "What precondition might you be missing for [action]?" Leverages the agent's latent knowledge without giving away the answer.
- `repair`: Full corrective action sequence from memory: "Previously, [failure] was fixed by [solution]." Direct repair instruction.

**Training objective**: Cost-sensitive multi-class classification.

```
V(k|x) = P̂(recovery | arm=k, x) - P̂(recovery | arm=none, x) - λ · cost(k)
π(x) = argmax_k V(k|x)
```

Where cost(none) = 0, cost(cue) = 0.1, cost(question) = 0.2, cost(repair) = 0.3.
λ is tuned on dev set.

**Training data**: From branched rollouts at failure checkpoints.
- Collect failure checkpoints from React_FM Epoch 1 runs
- At each checkpoint, resume execution with each of the 4 arms
- Matched continuation budget: exactly 5 steps per arm
- Record: (features, arm, recovered_within_5_steps, disrupted)
- Target: the arm with highest V(k|x)

**Escalation rule**: If the `question` arm was selected but the same failure signature σ recurs within 3 steps, escalate to `repair`. This captures "the agent tried to think about it but still failed."

**Inference**: At each detected failure, compute features, run gate, inject the selected arm content. ~1ms inference time (tabular classifier).

### Training Plan

1. **Data collection** (Week 1-2):
   - Run React_FM Epoch 1 on all benchmarks, logging full state at each failure checkpoint
   - Estimate: ~80 checkpoints total (40 ScienceWorld + 25 WebShop + 15 ALFWorld)
   - For each checkpoint, branch into 4 arms × 5 continuation steps
   - Total: ~1,600 API calls (manageable within budget)

2. **Gate training** (Week 2):
   - Extract features + outcomes from branched rollout data
   - Train XGBoost with cost-sensitive multi-class objective
   - Hyperparameter tuning on dev set (cross-validation)
   - Compute oracle arm distribution as upper bound

3. **Online evaluation** (Week 2-3):
   - Deploy trained gate in full React_FM pipeline
   - Compare against baselines on test environments

### Failure Modes and Diagnostics

| Failure Mode | How to Detect | Fallback |
|---|---|---|
| Too few informative checkpoints | < 30 checkpoints with arm heterogeneity | Fall back to binary gate (none/repair) |
| Gate overfits to training checkpoints | Dev set accuracy < random | Use threshold gate (inject if RRF score > τ) |
| question ≈ cue (diagnostic content doesn't matter) | No significant difference in recovery rate | Reframe as "minimal intervention / selective abstention" paper |
| Checkpoint replay not faithful | Different outcomes from same state | Report as limitation; use statistical aggregation |
| Branching budget too small (5 steps) | Many recoveries happen after step 5 | Extend to 10 steps if budget allows |

### Novelty and Elegance Argument

**Closest work**: Self-Regulation [2502.04576]
- Self-Regulation learns WHEN to intervene (binary: call stronger model or not)
- We learn WHAT to inject and at WHAT intensity (4-arm graded memory intervention)
- Self-Regulation's intervention is model-switching; ours is memory-content dosing
- Self-Regulation uses PRM scoring; we use matched branched rollouts
- Different action space, different objective, different application

**Second closest**: Intervention Paradox [2602.03338]
- They identify the disruption-recovery tradeoff (the problem)
- We learn to navigate it adaptively (the solution)
- They propose a binary pre-deployment test; we propose a per-state learned policy

**Why this is more than "add a classifier"**: The contribution is the FORMULATION of memory injection as a cost-sensitive contextual decision problem with graded arms, not just the classifier. The arm definitions, the branched training protocol, and the cost-sensitive objective are all part of the contribution.

## Claim-Driven Validation Sketch

### Claim 1: The learned gate outperforms unconditional injection
- **Minimal experiment**: Online comparison on ScienceWorld + WebShop (test split)
  - Baselines: ReAct, React_FM(always-repair), always-question, binary gate(none/repair), CDFM gate, oracle gate
  - Metric: Success rate, average score, total tokens
- **Expected evidence**: CDFM > always-repair on at least 2/3 benchmarks, either in success rate or efficiency

### Claim 2: The optimal intervention arm varies by context
- **Minimal experiment**: Checkpoint benchmark analysis
  - Show oracle arm distribution: what fraction of checkpoints have none/cue/question/repair as optimal?
  - Show that no single arm dominates (< 70% of checkpoints prefer same arm)
  - Show SHAP or feature importance: which features predict which arm?
- **Expected evidence**: Heterogeneous arm distribution; different features predict different arms

### Claim 3: Graded intervention outperforms binary intervention
- **Minimal experiment**: CDFM (4-arm) vs binary gate (none/repair only)
  - Same training data, same features, same classifier
  - Metric: Success rate + disruption rate
- **Expected evidence**: 4-arm > binary, especially on benchmarks where question arm is frequently optimal

## Experiment Handoff Inputs

- Must-prove claims: Gate > always-repair; arm heterogeneity exists; graded > binary
- Must-run ablations: question vs cue; gate vs threshold; 4-arm vs binary
- Critical datasets: ScienceWorld (50 ep, flagship), WebShop (100 sess), ALFWorld (134 envs, supportive)
- Highest-risk assumptions: (1) checkpoint branching is reproducible, (2) enough arm heterogeneity exists, (3) question ≠ cue

## Compute & Timeline Estimate

- **Checkpoint collection**: ~1,600 API calls × ~2K tokens = ~3.2M tokens (~$5)
- **Online evaluation**: 6 systems × 3 benchmarks × ~50 envs × ~20 steps × ~1K tokens = ~18M tokens (~$30)
- **Total API cost**: ~$50-100 (well within $200 budget)
- **Timeline**: 23 days (see RESEARCH_REVIEW_CDFM.md for week-by-week plan)
- **Gate training**: Seconds on CPU (tabular XGBoost)

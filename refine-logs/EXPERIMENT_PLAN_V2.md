# Experiment Plan V2 — Gate-Centered Paper

**Problem**: Memory-augmented ReAct agents suffer from the intervention paradox: unconditional memory injection can help or harm.
**Method Thesis**: A utility-based intervention gate learns when and how much to inject retrieved failure memory at checkpoints, selecting the minimum effective arm from {none, question, repair}.
**Date**: 2026-04-13
**Source**: `refine-logs/FINAL_PROPOSAL_V2.md` (post 5 rounds of `/research-refine`, score 8.7/10)

---

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|-----------------------------|---------------|
| **C1a** (local): Gate arms have higher recovery + lower disruption than always-repair | Core mechanism validity — the gate actually makes better per-checkpoint decisions | Per-arm recovery/disruption rates on held-out checkpoints; gate > always-repair on ≥60% of checkpoints | B2 |
| **C1b** (global): Local optimization → episode-level performance | Practical impact — local improvements compound into real task success | Gate beats always-repair on success_rate or efficiency on ≥2/3 benchmarks | B1 |
| **C2**: Arm heterogeneity exists | Justifies graded intervention — if one arm always wins, no gate is needed | No single arm is oracle-optimal >70% of the time; conditional analysis shows non-trivial none/question fractions | B3 |
| **C3**: Graded > binary (conditional) | Justifies 3-arm over simpler 2-arm design | 3-arm gate outperforms 2-arm gate (none/repair) on same data + classifier | B4 |
| **Control**: question ≠ cue | Memory *content* drives recovery, not just attention-focusing | question recovery rate > cue recovery rate (paired test, p<0.3 threshold for significance) | B5 |

**Anti-claims to rule out:**
- "The gate just learns to always abstain on hard tasks" → check gate arm distribution is non-degenerate
- "question ≈ cue → content doesn't matter" → if true, fall back to selective abstention framing (still publishable)
- "Gate improvement comes from fewer injections, not better ones" → compare gate vs random-subset injection at same injection rate

---

## Paper Storyline

**Main paper must prove:**
1. The intervention gate works: local decisions are better (C1a), episode outcomes improve (C1b)
2. Arm heterogeneity exists: graded intervention is necessary, not a design indulgence (C2)
3. question vs cue control establishes content-driven recovery or triggers fallback framing (Control)

**Appendix can support:**
- Feature importance analysis (XGBoost SHAP values)
- Per-benchmark gate behavior breakdown
- Replay validation protocol details and stability metrics
- β sensitivity analysis
- Per-task-type arm distributions

**Experiments intentionally cut:**
- Comparison with Self-Regulation (different action space — model switching vs memory dosing)
- Multi-memory (top-k>1) variants — top-1 only for simplicity
- Different gate learners (logistic, MLP) — XGBoost only, mention others as future work
- Transfer across benchmarks — gate is per-benchmark for now

---

## Experiment Blocks

### Block 1: Online Evaluation (Main Table)

- **Claim tested**: C1b — local gate optimization translates to episode-level success
- **Why this block exists**: The central result that proves the gate has practical impact
- **Dataset / split / task**:
  - ScienceWorld: 50 episodes (10 tasks × 5 variations), test split
  - WebShop: 100 sessions, test split (same sessions used in React_FM)
  - ALFWorld: 134 envs, eval_out_of_distribution split
- **Compared systems** (5 systems, 3 baseline families):
  1. `ReAct` — no memory, no injection (lower bound)
  2. `Always-Repair` — React_FM as-is, unconditional full repair injection (current SOTA baseline)
  3. `Binary-Gate` — learned gate with 2 arms {none, repair}, same features + training protocol
  4. `Learned-Gate` — learned gate with 3 arms {none, question, repair} (OUR METHOD)
  5. `Oracle-Gate` — selects replay-averaged best arm per checkpoint (upper bound)
- **Metrics** (in order of importance):
  1. `success_rate` (primary) — fraction of episodes with task completion
  2. `online_disruption_rate` — fraction of episodes where gate activated AND outcome worse than ReAct-no-memory
  3. `total_tokens` — end-to-end token consumption
  4. `gate_abstention_rate` — fraction of gate decisions → arm=none
  5. `bypass_rate` — fraction of failures where RRF < τ_min → automatic none
- **Setup details**:
  - Agent LLM: Qwen3.5-9B (ALFWorld/WebShop), DeepSeek-V3.2 (ScienceWorld) via SiliconFlow
  - Gate: XGBoost regressor, trained on dev checkpoint split (70/30 train/test per benchmark)
  - β = 0.3 (default), τ_min = 0.1 (default)
  - Epochs: 2 for memory systems (E1 = baseline collection, E2 = with memory), 1 for ReAct baseline
  - No seeds for online eval (deterministic given same envs + temperature=0)
- **Success criterion**: Learned-Gate success_rate ≥ Always-Repair on ≥2/3 benchmarks, AND online_disruption_rate < Always-Repair
- **Failure interpretation**: If gate ≤ always-repair everywhere → gate has no practical value; fall back to base React_FM paper
- **Table / figure target**: Table 1 (main result), Figure 2 (per-benchmark bar chart)
- **Priority**: MUST-RUN

---

### Block 2: Checkpoint-Level Recovery Analysis

- **Claim tested**: C1a — gate arms have higher local recovery + lower disruption than always-repair
- **Why this block exists**: Validates the mechanism at the decision level, not just episode outcomes
- **Dataset / split / task**: Held-out test checkpoints (30% of all collected checkpoints per benchmark)
- **Compared systems**:
  1. Per-arm recovery rates from branched rollouts: none, question, repair, cue (control)
  2. Gate-selected arm recovery vs always-repair recovery on the SAME checkpoints
  3. Gate-selected arm disruption vs always-repair disruption
- **Metrics**:
  1. `recovery_rate(arm)` = fraction of checkpoints where arm leads to progress improvement
  2. `disruption_rate(arm)` = fraction of checkpoints where arm leads to worse outcome than none
  3. `mean_utility(arm)` = average utility across checkpoints
  4. `gate_accuracy` = fraction of checkpoints where gate selects the oracle-best arm
- **Setup details**: Uses branched rollout data from M2. No additional API calls needed.
- **Success criterion**: Gate-selected recovery > always-repair recovery; gate-selected disruption < always-repair disruption
- **Failure interpretation**: If gate cannot improve local decisions → formulation is wrong, mechanism doesn't work
- **Table / figure target**: Table 2 (checkpoint-level results), Figure 3 (recovery/disruption scatter)
- **Priority**: MUST-RUN

---

### Block 3: Arm Heterogeneity & Oracle Analysis

- **Claim tested**: C2 — optimal arm varies by failure context
- **Why this block exists**: If one arm always wins, no gate is needed. Must prove heterogeneity to justify the paper.
- **Dataset / split / task**: All collected checkpoints (train + test), per benchmark
- **Compared systems**: Oracle arm distributions — what fraction of checkpoints has each arm as optimal?
- **Metrics**:
  1. `oracle_arm_distribution` — bar chart of {none, question, repair} as oracle-best per benchmark
  2. `conditional_penalty` — when oracle=none, what is always-repair's mean utility? (should be negative)
  3. `conditional_win_rate` — for each oracle arm, what does always-repair score?
  4. `arm_heterogeneity_index` = 1 - max(oracle_arm_fraction) — higher = more heterogeneous
- **Setup details**: Pure analysis on branched rollout data, no additional API calls.
- **Success criterion**: No single arm oracle-optimal >70%; arm_heterogeneity_index > 0.3; conditional_penalty for oracle=none is meaningfully negative
- **Failure interpretation**: If repair is always best → report as finding ("unconditional repair is actually optimal"), paper becomes negative result about intervention paradox being less severe than claimed
- **Table / figure target**: Figure 4 (stacked bar: oracle arm distribution), Table 3 (conditional analysis)
- **Priority**: MUST-RUN

---

### Block 4: Graded vs Binary Gate (Simplicity Check)

- **Claim tested**: C3 — 3-arm gate outperforms 2-arm gate
- **Why this block exists**: Defends the graded design against the simpler binary (none/repair) alternative. If binary is just as good, 3-arm adds unnecessary complexity.
- **Dataset / split / task**: Same test checkpoints + same online evaluation benchmarks as B1/B2
- **Compared systems**:
  1. `Binary-Gate` — {none, repair} only, same XGBoost, same features, same training protocol
  2. `Learned-Gate` — {none, question, repair} (3-arm)
- **Metrics**: Same as B1 (online: success_rate, disruption_rate) + B2 (checkpoint: recovery, disruption, gate_accuracy)
- **Setup details**: Binary-Gate trains on same data but with only 2 arm options. One additional XGBoost train (seconds).
- **Success criterion**: 3-arm > 2-arm on ≥2 metrics across ≥2 benchmarks
- **Failure interpretation**: If 3-arm ≈ 2-arm → question arm is redundant; paper becomes "selective abstention" (binary intervention learning). Still publishable with pre-committed fallback framing.
- **Table / figure target**: Table 4 (graded vs binary ablation), combined with B1 main table as additional row
- **Priority**: MUST-RUN (determines paper framing)

---

### Block 5: Question vs Cue Control

- **Claim tested**: Control — memory content (not just attention-focusing) drives recovery
- **Why this block exists**: If question ≈ cue, the "graded" story weakens. Must establish that question's value comes from content, not from the prompt interrupt itself.
- **Dataset / split / task**: Branched rollout data — checkpoints where both question and cue arms were evaluated
- **Compared systems**:
  1. `question` arm — LLM-canonicalized diagnostic question from memory
  2. `cue` arm — generic "Reconsider your approach." (no memory content)
- **Metrics**:
  1. `recovery_rate(question)` vs `recovery_rate(cue)` — paired comparison
  2. `mean_utility(question)` vs `mean_utility(cue)` — paired comparison
  3. `p_value` from paired sign test or Wilcoxon signed-rank
- **Setup details**: Already collected in branched rollouts (cue is one of the 4 arms in rollout collection). No additional API calls.
- **Success criterion**: question > cue with p < 0.3 (lenient threshold given small sample)
- **Failure interpretation**: If question ≈ cue → collapse arms to {none, repair}; fallback to "selective abstention" paper per FINAL_PROPOSAL_V2.md fallback framing. Cue reported as negative control result.
- **Table / figure target**: Table 5 (question vs cue), discussed in Section 4.4
- **Priority**: MUST-RUN (determines paper framing)

---

## Run Order and Milestones

### M0: Infrastructure & Sanity (Days 1–4, April 14–17)

| Run ID | Goal | Details | Cost | Decision Gate |
|--------|------|---------|------|---------------|
| R001 | Checkpoint save/restore infrastructure | Implement `CheckpointState` class: save agent state (history, memory, step_idx, env state) at each failure detection point. Implement `restore_and_branch()` to resume from checkpoint with injected arm content. | $0 | Code compiles, unit test passes |
| R002 | Modify `memory.retrieve()` to return RRF scores | Add `retrieval_rrf_score` and `retrieval_margin` to retrieve return value | $0 | Unit test passes |
| R003 | Add `question_text` / `repair_text` to `FailureMemoryEntry` | Extend dataclass, update serialization | $0 | Existing memory loads still work |
| R004 | Memory canonicalization pipeline | Offline LLM call per entry to produce question_text + repair_text | ~$2 | Spot-check 10 entries for quality |
| R005 | Feature extractor | Implement 11-feature extraction from agent state at checkpoint | $0 | Features extract correctly on 3 test checkpoints |
| R006 | Branched rollout pipeline | End-to-end: checkpoint → branch into 4 arms → run 5 steps → record progress | ~$1 | 3 sanity checkpoints × 4 arms × 1 replay complete |
| R007 | Progress signal extractors | Per-benchmark: ScienceWorld score delta, ALFWorld subgoal delta (PDDL state access), WebShop reward delta | $0 | Correct on 3 manual examples per benchmark |

**Decision Gate M0**: All infrastructure works end-to-end on 3 sanity checkpoints. If ALFWorld PDDL state is inaccessible, fall back to binary success for ALFWorld progress signal (weaker but functional).

**Risk**: ALFWorld intermediate progress may be hard to extract — the env wrapper doesn't expose PDDL state. Mitigation: check if TextWorld's underlying game state is accessible; if not, use action-novelty heuristic or binary-only.

---

### M1: Replay Validation (Days 5–6, April 18–19)

| Run ID | Goal | Details | Cost | Decision Gate |
|--------|------|---------|------|---------------|
| R008 | Phase A: outcome agreement | 10 checkpoints × 3 replays × arm=none per benchmark (up to 30 per benchmark). Check binary outcome agreement ≥ 80%. | ~$1 | Agreement ≥ 80% per benchmark |
| R009 | Phase B: utility ranking stability | 5 checkpoints × 3 replays × 3 arms per benchmark. Compute Kendall τ of utility ranking across replays. | ~$0.50 | Kendall τ ≥ 0.6 |

**Decision Gate M1**:
- If Phase A fails for a benchmark → exclude that benchmark from gate training (use binary gate or no gate)
- If Phase B fails → replay-averaged utilities (already default, this is confirmation)
- If ALL benchmarks fail Phase A → abort gate paper, fall back to base React_FM

**Risk**: LLM stochasticity at temperature=0 may still cause replay variance. Mitigation: if borderline, increase replays to 5.

---

### M2: Branched Rollouts & Gate Training (Days 7–12, April 20–25)

| Run ID | Goal | Details | Cost | Decision Gate |
|--------|------|---------|------|---------------|
| R010 | Collect failure checkpoints from React_FM E1 | Run React_FM E1 on all 3 benchmarks with checkpoint logging enabled. Record all failure detection events with full state. | ~$5 (reuse existing E1 if available) | ≥30 checkpoints per benchmark after dedup |
| R011 | Deduplicate by failure signature | Group checkpoints by (failure_type, task_type, failure_action_pattern). Keep representative per group. | $0 | ~40-80 unique checkpoints total |
| R012 | Memory canonicalization (full) | Run canonicalization on all memory entries collected in E1 | ~$2 | All entries have question_text + repair_text |
| R013 | Branched rollouts (all checkpoints) | Each checkpoint × {none, cue, question, repair} × 5 steps × 3 replays | ~$12 | All rollouts complete, utilities computed |
| R014 | Compute replay-averaged utilities | progress_avg - β·disruption_avg per (checkpoint, arm) | $0 | Utility matrix complete |
| R015 | Train XGBoost gate | 70/30 train/test split. Train f(features, arm_onehot) → utility. | $0 | Test MAE < baseline (predict mean) |
| R016 | Oracle analysis | Oracle arm distributions, heterogeneity index, conditional penalties | $0 | Arm heterogeneity confirmed or denied |

**Decision Gate M2 (CRITICAL — April 25, hard decision)**:
- If arm heterogeneity exists (no arm >70% oracle) AND gate > majority-baseline on test checkpoints → **GO** to online evaluation
- If no heterogeneity OR gate ≈ random → **ABORT** gate paper by April 27 deadline, fall back to base React_FM
- If question ≈ cue on dev → **ADJUST** to selective abstention framing, continue with 2-arm gate

**Risk**: Too few checkpoints (<30 per benchmark). Mitigation: lower dedup aggressiveness; if still insufficient, pool cross-benchmark for initial signal, per-benchmark for final.

---

### M3: Online Evaluation (Days 13–18, April 26–May 1)

| Run ID | Goal | Details | Cost | Decision Gate |
|--------|------|---------|------|---------------|
| R017 | ReAct baseline (no memory) | 1 epoch × 3 benchmarks | ~$5 | Results match prior runs |
| R018 | Always-Repair (React_FM) | 2 epochs × 3 benchmarks | ~$10 | Results match prior runs |
| R019 | Binary-Gate online | 2 epochs × 3 benchmarks, binary gate deployed | ~$10 | Completes without errors |
| R020 | Learned-Gate online (3-arm) | 2 epochs × 3 benchmarks, full gate deployed | ~$10 | Completes without errors |
| R021 | Oracle-Gate online | 2 epochs × 3 benchmarks, oracle arm selected per checkpoint | ~$10 | Upper bound is meaningfully above always-repair |

**Decision Gate M3**: Gate beats always-repair on ≥2/3 benchmarks. If not, check if gate matches always-repair with lower disruption (efficiency story). If neither, the paper becomes a negative result / analysis paper.

**Risk**: Online evaluation is expensive (~$40). Mitigation: run ScienceWorld first (best progress signal, most likely to show gate value), then decide whether to continue to WebShop/ALFWorld.

---

### M4: Analysis & Paper Writing (Days 19–23, May 2–6)

| Run ID | Goal | Details | Cost | Decision Gate |
|--------|------|---------|------|---------------|
| R022 | Statistical tests | Paired tests for question vs cue, gate vs always-repair per benchmark | $0 | — |
| R023 | Feature importance | XGBoost SHAP values, top features driving gate decisions | $0 | — |
| R024 | Figures | Per-benchmark bars, oracle arm distributions, recovery/disruption scatter, Pareto frontier | $0 | — |
| R025 | β sensitivity (appendix) | Retrain gate with β ∈ {0.0, 0.1, 0.3, 0.5, 1.0}, report test utility | $0 | — |
| R026 | Paper draft | Introduction, method, experiments, results, analysis, conclusion | $0 | — |

---

## Compute and Data Budget

| Item | API Calls | Cost | Days |
|------|-----------|------|------|
| Infrastructure sanity (R001-R007) | ~200 | ~$3 | 4 |
| Replay validation (R008-R009) | ~300 | ~$1.50 | 2 |
| React_FM E1 + checkpoint collection (R010) | ~5,000 | ~$5 | 2 |
| Memory canonicalization (R012) | ~300 | ~$2 | 0.5 |
| Branched rollouts (R013) | ~4,800 | ~$12 | 3 |
| Online evaluation 5×3 (R017-R021) | ~15,000 | ~$40 | 6 |
| **Total** | **~25,600** | **~$63.50** | **23** |

- **Total GPU-hours**: 0 (XGBoost trains on CPU in seconds; all LLM calls via API)
- **Data preparation**: Checkpoint collection is automated from E1 runs
- **Human evaluation**: None required
- **Biggest bottleneck**: Branched rollout collection (R013) — 4,800 API calls, ~2-3 days wall clock

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Too few checkpoints (<30/benchmark) | Medium | High — gate can't train | Lower dedup threshold; pool ScienceWorld+WebShop; binary fallback for smallest benchmark |
| ALFWorld PDDL state inaccessible | Medium | Medium — weakest progress signal | Fall back to binary success (reward=0/1 after 5 steps); ALFWorld becomes binary-only benchmark |
| Replay infidelity (agreement <80%) | Low | High — branched rollouts unreliable | Increase replays to 5; exclude worst benchmark; report replay variance |
| No arm heterogeneity | Medium | Fatal — no paper | Hard abort April 27, fall back to base React_FM paper |
| question ≈ cue | Medium | Medium — weaker story | Pre-committed fallback: selective abstention framing with 2-arm gate |
| Gate ≈ random on test | Low | High — mechanism doesn't work | Threshold gate fallback; report as negative finding |
| Budget overrun | Low | Medium — can't complete M3 | Run ScienceWorld first (cheapest, best signal); cut ALFWorld if over budget |
| Timeline slip | Medium | High — miss NeurIPS | Parallelize M3 runs across benchmarks; pre-write method section during M2 |

---

## Final Checklist

- [x] Main paper tables are covered (Table 1: online eval, Table 2: checkpoint-level, Table 3: conditional analysis, Table 4: graded vs binary, Table 5: question vs cue)
- [x] Novelty is isolated (Block 3: arm heterogeneity shows the *formulation* is needed, not just the classifier)
- [x] Simplicity is defended (Block 4: 3-arm vs 2-arm; XGBoost is intentionally simple)
- [x] Frontier contribution is explicitly NOT claimed (LLMs are infrastructure, not the contribution; XGBoost is the trainable part)
- [x] Nice-to-have runs are separated from must-run runs (R023-R025 are appendix-only)
- [x] Hard abort deadline set (April 27 = end of M2)
- [x] Fallback framing pre-committed (question≈cue → selective abstention)
- [x] Budget within constraint (~$63.50 ≤ ~$200 hard limit, ≈$60 soft target)

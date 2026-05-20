# Experiment Tracker V2 — Gate-Centered Paper

**Last updated**: 2026-04-14 08:45
**Hard abort**: April 27 (end of M2)
**Submission**: May 6 (NeurIPS 2026)

## Infrastructure Runs (M0: Days 1–4, April 14–17)

| Run ID | Purpose | System / Variant | Benchmark | Metrics | Priority | Status | Notes |
|--------|---------|------------------|-----------|---------|----------|--------|-------|
| R001 | Checkpoint save/restore | Infrastructure | All | Unit test pass | MUST | ✅ DONE | `src/gate/checkpoint.py` — CheckpointState + CheckpointStore |
| R002 | Return RRF scores from retrieve() | Infrastructure | All | Unit test pass | MUST | ✅ DONE | `src/memory.py` — RetrievalResult + return_scores param |
| R003 | Add question_text/repair_text fields | Infrastructure | All | Serialization test | MUST | ✅ DONE | `src/memory.py` — backward compatible |
| R004 | Memory canonicalization pipeline | Canonicalization | All | Spot-check 10 entries | MUST | ✅ DONE | `src/gate/canonicalize.py` — ~$2 offline LLM calls |
| R005 | 11-feature extractor | Infrastructure | All | Correct on 3 checkpoints | MUST | ✅ DONE | `src/gate/features.py` — verified with unit tests |
| R006 | Branched rollout pipeline | Infrastructure | ScienceWorld | 3×4×1 sanity run | MUST | ✅ DONE | `src/gate/branched_rollout.py` + `experiments/gate/run_branched_rollouts.py` |
| R007 | Progress signal extractors | Infrastructure | All | Manual verification | MUST | ✅ DONE | `src/gate/progress.py` — 3 benchmark signals |
| R007b | Sanity: collect_checkpoints | Infrastructure | ScienceWorld | 3 envs, 6 cp | MUST | ✅ DONE | 6 checkpoints from 3 envs, 3 memory entries. All 3 benchmarks implemented. |
| R007c | hint_text prompt builder fix | Infrastructure | All | Syntax pass | MUST | ✅ DONE | All 3 prompt builders accept hint_text, online eval uses it |
| R007d | ALFWorld/WebShop stubs filled | Infrastructure | All | Syntax pass | MUST | ✅ DONE | collect_checkpoints + run_online_eval + run_branched_rollouts all 3 benchmarks |

## Replay Validation (M1: Days 5–6, April 18–19)

| Run ID | Purpose | System / Variant | Benchmark | Metrics | Priority | Status | Notes |
|--------|---------|------------------|-----------|---------|----------|--------|-------|
| R008 | Phase A: outcome agreement | arm=none, 10cp×3rep | All | agreement ≥ 80% | MUST | TODO | ~$1 |
| R009 | Phase B: utility ranking stability | 3 arms, 5cp×3rep | All | Kendall τ ≥ 0.6 | MUST | TODO | ~$0.50 |

## Branched Rollouts & Gate Training (M2: Days 7–12, April 20–25)

| Run ID | Purpose | System / Variant | Benchmark | Metrics | Priority | Status | Notes |
|--------|---------|------------------|-----------|---------|----------|--------|-------|
| R010 | Collect failure checkpoints | React_FM E1 w/ logging | All | ≥30 cp/benchmark | MUST | ✅ DONE (2/3) | SW: 568cp/155mem (50 envs); ALF: 146cp/60mem (134 envs); WS: RUNNING |
| R011 | Deduplicate checkpoints | Analysis | All | ~40-80 unique total | MUST | ✅ DONE | ALF: 146→49; SW: 568→252 (max_per_sig=1). Use --max-checkpoints 40 for rollouts |
| R012 | Full memory canonicalization | Canonicalization | All | All entries done | MUST | ✅ DONE | 215 entries (60 ALF + 155 SW), 38K tokens, DeepSeek-V3.2 |
| R013 | Branched rollouts (BUGGED) | 4 arms × 5 steps × 3 rep | SW+ALF | — | — | ❌ INVALID | **BUG**: memory lookup returned None for all CPs; question/repair = none. Data discarded. |
| R013b | Pilot rollouts (fixed) | 4 arms × 5 steps × 3 rep | SW+ALF | Arm differentiation | MUST | ✅ DONE | 5 SW + 5 ALF. SW shows signal (cp_0016: 0.35 none/repair vs 0 question). ALF binary too coarse. |
| R013c-SW | SW main rollouts | 4 arms × 10 steps × 3 rep | SW | Utilities computed | MUST | ✅ DONE | 40 midband CPs, 480 rollouts, 3h. **Het=0.300 full, 0.429 active**. repair>none marginal (d≈0.10). |
| R013c-ALF | ALF main rollouts | 4 arms × 8 steps × 3 rep | ALF | Utilities computed | MUST | RUNNING | 49 CPs, milestone scorer. ~3h est. |
| R013d | ALF milestone scorer | Subgoal progress | ALF | Fractional progress | MUST | ✅ DONE | put(2), clean/heat/cool(3), examine(2), puttwo(4) milestones. 28/49 CPs at milestone 0, 21/49 partial. |
| R014 | Compute utilities | Analysis | SW | Utility matrix | MUST | ✅ DONE | repair 0.131 > none 0.113 > question 0.100 |
| R015 | Train XGBoost gate | Gate training | SW | Test MAE < mean | MUST | ⚠️ FAIL | XGBoost LOO=17.5% (majority=70%). Simple models=75% (barely above majority). Policy value: gate ≈ always-none. |
| R016 | Oracle analysis | Analysis | SW | Heterogeneity index | MUST | ⚠️ BORDERLINE | Het=0.300 (fails >0.3). Active het=0.462. Oracle headroom +0.034 over none. |

## Online Evaluation (M3: Days 13–18, April 26–May 1)

| Run ID | Purpose | System / Variant | Benchmark | Metrics | Priority | Status | Notes |
|--------|---------|------------------|-----------|---------|----------|--------|-------|
| R017 | ReAct baseline | No memory | All | SR, tokens | MUST | TODO | ~$5, run first as sanity |
| R018 | Always-Repair | React_FM as-is | All | SR, disruption, tokens | MUST | TODO | ~$10 |
| R019 | Binary-Gate online | 2-arm gate deployed | All | SR, disruption, tokens | MUST | TODO | ~$10 |
| R020 | Learned-Gate online | 3-arm gate deployed | All | SR, disruption, tokens | MUST | TODO | ~$10 |
| R021 | Oracle-Gate online | Oracle arm per cp | All | SR (upper bound) | MUST | TODO | ~$10 |

## Analysis & Writing (M4: Days 19–23, May 2–6)

| Run ID | Purpose | System / Variant | Benchmark | Metrics | Priority | Status | Notes |
|--------|---------|------------------|-----------|---------|----------|--------|-------|
| R022 | Statistical tests | All pairs | All | p-values | MUST | TODO | No API calls |
| R023 | Feature importance (SHAP) | XGBoost analysis | All | Top features | NICE | TODO | Appendix |
| R024 | Figures | Visualization | All | Plots ready | MUST | TODO | No API calls |
| R025 | β sensitivity | Gate retrain | All | Utility vs β | NICE | TODO | Appendix |
| R026 | Paper draft | Writing | — | Submission-ready | MUST | TODO | — |

## Decision Gates Summary

| Gate | Date | Condition | GO | ABORT / ADJUST |
|------|------|-----------|-----|----------------|
| M0 | Apr 17 | Infrastructure works on 3 sanity checkpoints | Continue to M1 | Debug; if ALFWorld PDDL fails, use binary progress |
| M1 | Apr 19 | Replay agreement ≥ 80% on ≥2 benchmarks | Continue to M2 | Exclude failing benchmarks; if all fail, abort |
| M2 | Apr 25 | Arm heterogeneity exists + gate > majority baseline | Continue to M3 | **HARD ABORT Apr 27** → base React_FM paper |
| M2b | Apr 25 | question > cue? | Keep 3-arm "graded" framing | Collapse to 2-arm "selective abstention" |
| M3 | May 1 | Gate beats always-repair ≥2/3 benchmarks | Write full paper | Efficiency-only story or negative result |

## Key Decisions (Apr 14)

1. **2-arm gate collapse**: Pilot showed question arm ≈ none. Paper uses {none, repair} as main gate, question as appendix control.
2. **ScienceWorld = quantitative core**: 10-step rollouts with max_score_in_window. Midband CPs (score 20-79) biased for signal.
3. **ALFWorld milestone scorer**: Subgoal-based progress replaces binary completion. Task-family milestones: put(2), clean/heat/cool(3), examine(2), puttwo(4).
4. **cross_env=False for all benchmarks**: Per-env memory isolation. ALFWorld envs are unique tasks. ScienceWorld per-env memory sufficient.

## R013c-SW Results & Codex Review (Apr 14, 08:30)

### ScienceWorld Main Rollout Results
- **Progress**: repair (0.135) > none (0.113) > question (0.106) > cue (0.089)
- **Utility (β=0.3)**: repair (0.131) > none (0.113) > question (0.100) > cue (0.076)
- **Oracle**: none 70%, repair 20%, question 10% → **Het=0.300 (borderline)**
- **Active subset (21/40)**: none 57%, repair 29%, question 14% → **Het=0.429 (pass)**
- **Ties**: 21/40 CPs have zero progress for all arms (10 steps insufficient)
- **Effect size**: d ≈ 0.10, CI includes 0, median progress unchanged between arms

### Codex Review Verdict
- **M2 borderline**: full-set het exactly 0.300 (fails > 0.3 rule). Active subset defensible as secondary diagnostic only.
- **ALFWorld is make-or-break**: if also borderline, reframe paper around negative result.
- **Two-stage gate**: Codex recommends (1) predict intervenable?, (2) choose arm. Zero-progress CPs are stage-1 negatives.
- **Simpler model**: 21 active CPs too small for XGBoost. Use logistic regression or shallow tree with leave-CP-out eval.
- **Paper framing**: "memory intervention is sparse and conditional; main challenge is abstention" — not "repair wins broadly".
- **Preregister**: full set primary, active subset secondary; CP-level bootstrap tests; policy-value eval vs always-none and always-repair.

## Budget Tracking

| Milestone | Planned Cost | Actual Cost | Planned Calls | Actual Calls |
|-----------|-------------|-------------|---------------|--------------|
| M0 | $3.00 | ~$2 | 200 | ~150 |
| M1 | $1.50 | — | 300 | — |
| M2 | $19.00 | ~$6 ($4 wasted R013 + $2 pilots) | 10,100 | ~3,100 (2500 wasted + 600 pilot) |
| M2 rerun | ~$15 | ~$12 (SW $10 done + ALF running) | ~7,200 | SW: 4,800 done |
| M3 | $40.00 | — | 15,000 | — |
| M4 | $0.00 | — | 0 | — |
| **Total** | **$63.50** | **~$20** | **25,600** | **~8,050** |

## R014-R016 Results & Codex Review #2 (Apr 14, 09:30)

### ScienceWorld Gate Training Results

**R014: Utility**
- repair (0.131) > none (0.113) > question (0.100)
- Oracle: none 70%, repair 20%, question 10%
- Het=0.300 (fails strict >0.3)
- Policy values: Oracle=0.147, Always-repair=0.131, Always-none=0.113

**R015: Gate Training (LOO)**
- XGBoost LOO: 17.5% — catastrophic overfit, predicts repair 83% of the time
- Decision Tree (depth=2): 75.0% — barely above 70% majority
- Logistic Regression: 75.0%
- Binary gate (none vs repair): 83.3% (majority: 77.8%)
- LOO policy value: Learned gate = 0.1129 ≈ always-none, **worse than always-repair**

**R015: Feature Importance**
- memory_entry_count (36%), task_type (34%), step_index (9%)
- retrieval_rrf_score and retrieval_margin = 0 (ScienceWorld backfilled late)

**R016: Oracle Analysis**
- Active subset (26/40): het=0.462, oracle=none 54%, repair 31%, question 15%
- 14/40 (35%) zero-progress across all arms — 10-step horizon censoring
- Oracle lift: +0.034 over none, +0.016 over repair — headroom exists but small

### Codex Review #2 Verdict
- **NO-GO for learned 3-way gate on SW alone**. Signal too weak, sample too small.
- **Always-repair beats learned gate** in policy value (0.131 vs 0.113).
- **3-way question arm unlearnable**: 4 CPs with oracle=question is an anecdote, not a learning target.
- **Active subset is diagnostic, not confirmatory** — post-hoc outcome-based selection.
- **Pivot to binary abstention** (none vs repair) for learned gate; use question as appendix.
- **ALFWorld is make-or-break**: if ALFWorld shows dense progress + stronger het, it becomes primary benchmark.
- **SW reframe**: "gate learning fails when progress signals are sparse and horizons are censored" — honest negative.
- **Increase SW horizon to 15-20 steps** to reduce zero-progress rate (35% → ~15% est).
- **Evaluate only by policy value**, not accuracy — accuracy is misleading with class imbalance.

## Bug Fixes & Code Changes (Apr 14)

### BUG: Memory lookup returned None for all rollouts (R013 INVALID)

**Root cause**: `lookup_memory_entry()` in `run_branched_rollouts.py:66` had `if mem_id < 0: return None`. All epoch-1 checkpoints have `retrieved_memory_id = -1` (no memory existed yet). Fallback retrieve was unreachable. Additionally, fallback didn't pass `env_idx` to `memory_store.retrieve()`.

**Impact**: question/repair arms injected no memory → identical to none arm. All R013 rollout data is invalid.

**Fix**: Skip early return when `mem_id < 0`, fall through to `memory_store.retrieve(env_idx=cp.env_idx)`. Verified: SW 40/40, ALF 44/49 now find entries.

### Code improvements for R013b rerun

| Change | File | Detail |
|--------|------|--------|
| Hint persistence | `src/gate/branched_rollout.py` | `hint_steps=3`: inject hint on steps 0-2 (was step 0 only). Each step rebuilds prompt from scratch, so hint disappears without this. |
| max_score_in_window | `src/gate/branched_rollout.py` | Track `score_max` during rollout. ScienceWorld progress uses max-in-window instead of final-only delta. |
| Balanced checkpoint sample | `checkpoints/scienceworld_checkpoints_balanced.json` | 40 CPs across 10 task types (was 27 boil + 13 melt). 34 score>0, 6 score=0. 88% memory coverage. |
| cross_env param | `run_branched_rollouts.py` | `lookup_memory_entry(..., cross_env)` — default False for all benchmarks (per-env memory is sufficient). |

# Refinement Report V2 — Gate-Centered Paper

**Problem**: Intervention paradox in memory-augmented ReAct agents
**Initial Approach**: CDFM — Counterfactual Diagnostic-First Memory with adaptive gate
**Date**: 2026-04-13
**Rounds**: 5 / 5
**Final Score**: 8.7 / 10
**Final Verdict**: REVISE (gap is empirical, not methodological)

## Problem Anchor
Memory-augmented ReAct agents inject retrieved failure memories unconditionally, with no mechanism to decide when and how much to intervene. This causes the intervention paradox: helping failing trajectories but disrupting self-recovering ones. No prior system learns a graded memory intervention policy at failure checkpoints.

## Output Files
- Review summary: `refine-logs/REVIEW_SUMMARY_V2.md`
- Final proposal: `refine-logs/FINAL_PROPOSAL_V2.md`
- Score history: `refine-logs/score-history-v2.md`

## Score Evolution

| Round | PF | MS | CQ | FL | Fe | VF | VR | Overall | Verdict |
|-------|----|----|----|----|----|----|-----|---------|---------|
| 1     | 8  | 6  | 8  | 7  | 6  | 8  | 6   | 7.1     | REVISE  |
| 2     | 9  | 8  | 8  | 8  | 7  | 8  | 7   | 8.0     | REVISE  |
| 3     | 9  | 8  | 8  | 8  | 7  | 8  | 7   | 8.0     | REVISE  |
| 4     | 9  | 9  | 8  | 8  | 8  | 8  | 8   | 8.4     | REVISE  |
| 5     | 9  | 9  | 9  | 8  | 8  | 9  | 8   | 8.7     | REVISE  |

## Final Proposal Snapshot
1. Learn a utility-based intervention policy over {none, question, repair} at failure checkpoints
2. Train from matched branched rollouts with replay-averaged, benchmark-native progress signals
3. XGBoost regressor over 11 observable features (no privileged state at inference)
4. Cue as experimental control (not in gate action space)
5. Bypass when retrieval confidence is low; fallback to selective abstention if question≈cue

## Method Evolution Highlights
1. **Most important simplification**: Removed escalation rule, surprise gating, load budgeting, token cost → single clean mechanism
2. **Most important mechanism upgrade**: Utility regression with benchmark-native progress signals replacing sparse binary success
3. **Most important justification**: Pre-committed fallback framing (question weak → selective abstention paper)

## Pushback / Drift Log
| Round | Reviewer Said | Author Response | Outcome |
|-------|---------------|-----------------|---------|
| 1 | Cue may not be learnable | Moved cue to control only | Accepted — cleaner |
| 1 | Remove escalation rule | Agreed, added recurrence as feature | Accepted |
| 2 | Benchmark count inconsistent | Restored ALFWorld | Accepted |
| 3 | 5-step binary success too sparse | Replaced with benchmark-native progress | Accepted |
| 4 | Progress signals = privileged state? | Explicitly offline labeling only | Accepted |
| 5 | "Graded" wording premature | Pre-committed fallback if question≈cue | Accepted |

## Remaining Weaknesses
1. Empirical separation unknown — method is sound but results may not be dramatic enough
2. question vs cue outcome uncertain — determines whether paper is "graded intervention" or "selective abstention"
3. 23-day timeline is tight for 3-benchmark branched rollout collection + online evaluation + paper writing
4. Checkpoint count (~80 total) may be too small for robust gate training

## Next Steps
- Proceed to `/experiment-plan` for detailed execution roadmap
- Then implement: checkpoint infrastructure → replay validation → branched rollouts → gate training → online evaluation
- Hard abort deadline: April 27 — if gate shows no signal, fall back to base React_FM paper

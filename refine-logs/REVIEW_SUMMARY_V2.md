# Review Summary V2 — Gate-Centered Paper

**Problem**: Memory-augmented ReAct agents suffer from the intervention paradox: unconditional memory injection can help or harm.
**Initial Approach**: Learned gate over graded intervention arms {none, cue, question, repair}
**Date**: 2026-04-13
**Rounds**: 5 / 5 (MAX_ROUNDS)
**Final Score**: 8.7 / 10
**Final Verdict**: REVISE (close to READY — gap is empirical, not methodological)

## Problem Anchor
Memory-augmented ReAct agents inject retrieved failure memories unconditionally, with no mechanism to decide when and how much to intervene. This causes the intervention paradox: helping failing trajectories but disrupting self-recovering ones.

## Round-by-Round Resolution Log

| Round | Main Reviewer Concerns | What This Round Changed | Solved? | Remaining Risk |
|-------|------------------------|-------------------------|---------|----------------|
| 1 | Training target ambiguous; checkpoint replay hand-waved; venue readiness (heuristic stack) | Utility regression; LLM-canonicalized question arm; removed escalation; cue→control | Yes | Utility label still underspecified |
| 2 | Utility label not operationally defined; benchmark count inconsistency; replay validation thin | Precise recovery/disruption definitions; ALFWorld restored; replay validation strengthened | Yes | Sparse reward problem on ALFWorld |
| 3 | Local reward too sparse on binary-success benchmarks; replay-averaged labels needed | Benchmark-native progress signals; replay-averaged utilities; token cost dropped | Yes | Progress signals = privileged state? |
| 4 | Progress = offline labeling only must be explicit; online disruption metric needed; low-retrieval behavior | Explicit offline-only statement; online_disruption_rate defined; bypass rule added | Yes | Graded story depends on question≠cue |
| 5 | Paired eval protocol; fallback framing; "graded" wording caution | All addressed — pre-committed fallback, explicit metrics | Mostly | Empirical separation is the gap |

## Overall Evolution
- **Method became more concrete**: From vague "cost-sensitive classification" to fully specified utility regression with benchmark-native progress signals, replay averaging, and bypass rules
- **Dominant contribution sharpened**: From "adaptive memory framework with 4 components" to "utility-based minimal-intervention learning from matched branched rollouts"
- **Unnecessary complexity removed**: Escalation rule, surprise gating, load budgeting, token cost term, SHAP as main evidence, cue in action space
- **Frontier leverage appropriate**: LLMs used for agent execution + offline memory canonicalization; no gratuitous RL/finetuning
- **Drift avoided**: Problem Anchor preserved across all 5 rounds

## Final Status
- Anchor status: PRESERVED
- Focus status: TIGHT — one dominant contribution, one supporting, explicit non-contributions
- Modernity status: APPROPRIATELY FRONTIER-AWARE
- Strongest parts: Utility formulation, branched rollout protocol, benchmark-native progress signals, explicit bypass/fallback
- Remaining weaknesses: Empirical punch unknown; question≠cue not yet confirmed; 23-day timeline is tight

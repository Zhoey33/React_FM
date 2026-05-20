# Round 2 Review

**Date**: 2026-04-13
**Thread ID**: `019d85c9-db06-7b70-af99-00d51cbd0bc1`

## Scores

| Dimension | R1 | R2 | Change |
|---|---:|---:|---|
| Problem Fidelity | 8 | 9 | +1 |
| Method Specificity | 6 | 8 | +2 |
| Contribution Quality | 8 | 8 | = |
| Frontier Leverage | 7 | 8 | +1 |
| Feasibility | 6 | 7 | +1 |
| Validation Focus | 8 | 8 | = |
| Venue Readiness | 6 | 7 | +1 |
| **OVERALL** | **7.1** | **8.0** | **+0.9** |

**Verdict**: REVISE (close to READY)
**Drift Warning**: NONE

## Remaining Action Items

### CRITICAL
1. **Define utility label operationally**: Precise definition of `recovery`, `disruption`, and rollout horizon. If short-horizon only, gate may optimize local repair while hurting final success. Either continue branches to episode end, or use benchmark-native progress surrogate.
2. **Align training signal with anchored claim**: If utility is based on short-horizon recovery, present it as such.

### IMPORTANT
3. **Benchmark-count inconsistency**: Success condition says "2/3 benchmarks" but only 2 named. Restore ALFWorld or rewrite claim.
4. **Strengthen replay validation**: Add same-arm stability pilot checking utility ranking stability, not just binary outcome.
5. **Tighten novelty claim**: Novelty = matched failure-state utility-learning formulation, NOT the regressor itself.

## Simplification Opportunities
- Commit to top-1 memory only
- Pick one learner (XGBoost), treat logistic as appendix robustness
- If budget tight, keep `cue` only in checkpoint control

## Modernization Opportunities
- NONE (already appropriate)

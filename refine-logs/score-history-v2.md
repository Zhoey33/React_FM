# Score Evolution (V2 — Gate-Centered Paper)

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|-------|------------------|--------------------|----------------------|-------------------|-------------|------------------|-----------------|---------|---------|
| 1     | 8                | 6                  | 8                    | 7                 | 6           | 8                | 6               | 7.1     | REVISE  |
| 2     | 9                | 8                  | 8                    | 8                 | 7           | 8                | 7               | 8.0     | REVISE  |
| 3     | 9                | 8                  | 8                    | 8                 | 7           | 8                | 7               | 8.0     | REVISE  |
| 4     | 9                | 9                  | 8                    | 8                 | 8           | 8                | 8               | 8.4     | REVISE  |
| 5     | 9                | 9                  | 9                    | 8                 | 8           | 9                | 8               | 8.7     | REVISE  |

## Key Evolution

- **R1→R2**: Utility regression replaces best-arm classification; question canonicalized offline; escalation rule removed; cue→control only (+0.9)
- **R2→R3**: Utility label defined operationally; ALFWorld restored; replay-averaged labels; novelty claim tightened (+0.0 — same score, but higher quality)
- **R3→R4**: Benchmark-specific progress signals replace sparse binary success; token cost dropped from utility; replay-averaged labels committed (+0.4)
- **R4→R5**: Progress signals = offline labeling only (no privileged state); online disruption metric defined; retrieval bypass rule; fallback framing pre-committed (+0.3)

## Remaining Gap to 9

Not missing mechanism — the gap is whether empirical results deliver enough arm heterogeneity and gate-vs-baseline separation to make the formulation feel indispensable.

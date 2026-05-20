# ScienceWorld Step 4 Summary

> Date: 2026-04-21
> Scope: Step 4 no-gate comparison for ScienceWorld core task protocol
> Summary path referenced by runbook: `doc/scienceworld_step4_summary.md`
> Updated: 2026-05-06 with Experiment D (ExpeL baseline)

## Setup

- Experiment A: online React_FM
- Experiment B: offline memory
- Experiment C: ReAct baseline
- Reflexion: online per-episode two-pass variant
- Experiment D: ExpeL baseline, Epoch 1 trajectory collection followed by Epoch 2 insight injection

Primary result files:
- `results/20260415_153753/sw_step4_expA_online_test_1.json`
- `results/20260417_092133/sw_step4_expB_offline_test_1.json`
- `results/20260418_130202/sw_step4_expC_react_baseline_test_1.json`
- `results/20260420_180544/sw_step4_reflexion_online_formal_pass0.json`
- `results/20260420_180544/sw_step4_reflexion_online_formal_pass1.json`
- `results/20260420_180544/sw_step4_reflexion_online_formal_online_checkpoint.json`
- `results/20260506_001059/sw_step4_expD_expel_test_1.json`
- `results/20260506_001059/sw_step4_expD_expel_test_2.json`
- `memory_store/20260506_001059/sw_expel_insights.json`

Related logs:
- `logs/20260415_153753/sw_step4_expA_online_test.log`
- `logs/20260417_092133/sw_step4_expB_offline_test.log`
- `logs/20260418_130202/sw_step4_expC_react_baseline_test.log`
- `logs/20260420_180544/sw_step4_reflexion_online_formal.log`
- `logs/20260506_001059/sw_step4_expD_expel_test.log`

## Score Convention

- Per-episode `score` is the raw ScienceWorld score on the native scale, typically `[-100, 100]`.
- Summary `avg_score` in result JSONs is normalized as `raw_score / 100`.
- In the tables below:
  - `Avg Score (Raw)` means the native ScienceWorld average score
  - `Avg Score (Norm)` means the normalized value used in saved summaries

## Reflexion Evaluation Convention

The raw Reflexion runner re-runs every episode in pass 2, including episodes already solved in pass 1. That raw `pass1` metric is not the metric used for Step 4 conclusions.

Step 4 Reflexion is reported with the following corrected convention:

- if `pass0` succeeds, the final Reflexion result for that episode reuses `pass0`
- if `pass0` fails, the final Reflexion result uses the real `pass1`

This corrected metric is denoted below as `Reflexion P1 Adj`.

## Overall Comparison

| Method | Episodes | Success | Success Rate | Avg Score (Raw) | Avg Score (Norm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ExpA Online React_FM | 111 | 46 | 41.44% | 64.25 | 0.6425 |
| ExpB Offline Memory | 111 | 44 | 39.64% | 48.83 | 0.4883 |
| ExpC ReAct Baseline | 111 | 42 | 37.84% | 47.21 | 0.4721 |
| Reflexion P0 | 111 | 28 | 25.23% | 40.57 | 0.4057 |
| Reflexion P1 Raw | 111 | 25 | 22.52% | 36.90 | 0.3690 |
| Reflexion P1 Adj | 111 | 41 | 36.94% | 49.39 | 0.4939 |
| ExpD ExpeL E1 | 111 | 42 | 37.84% | 51.30 | 0.5130 |
| ExpD ExpeL E2 | 111 | 55 | 49.55% | 57.78 | 0.5778 |

## Token Comparison

| Method | Total Tokens | Avg Tokens / Episode | Token Interpretation |
| --- | ---: | ---: | --- |
| ExpA Online React_FM | 21,891,951 | 197,225 | Actual runtime cost |
| ExpB Offline Memory | 20,027,604 | 180,429 | Actual runtime cost |
| ExpC ReAct Baseline | 20,591,368 | 185,508 | Actual runtime cost |
| Reflexion P0 | 12,395,129 | 111,668 | Pass-0 runtime only |
| Reflexion P1 Raw | 11,207,907 | 100,972 | Pass-1 runtime only |
| Reflexion P1 Adj | 11,965,847 | 107,800 | Post-hoc selected output cost, not full runtime |
| Reflexion Runtime Lower Bound | 23,603,036 | 212,640 | `P0 + P1 raw`; excludes reflection-generation calls |
| ExpD ExpeL E2 | 16,439,640 | 148,105 | Epoch-2 result file total; includes 279,145 extractor tokens |
| ExpD ExpeL Effective | 21,196,733 | 190,961 | Effective result cost: E1 tokens for E1 successes, E2 tokens for E1 failures retried with insights |

Token note:
- `Reflexion P1 Adj` is useful for effective-result reporting, but it is not the real runtime bill.
- The real Reflexion runtime is at least `P0 + P1 raw = 212,640` tokens per episode on average.
- This lower bound still underestimates full Reflexion cost because reflection-generation LLM calls are not included in `episodes[].total_tokens`.
- ExpeL E2 token accounting includes both agent tokens and Epoch-1 insight-extraction tokens. The effective table value used in the paper is `190.9K` tokens per episode because E1 successes keep their E1 outputs while E1 failures use E2 outputs.

## Delta Summary

| Delta | Success | Success Rate | Avg Score (Raw) | Avg Tokens / Episode |
| --- | ---: | ---: | ---: | ---: |
| A - C | +4 | +3.60pp | +17.04 | +11,717 |
| B - C | +2 | +1.80pp | +1.62 | -5,079 |
| A - B | +2 | +1.80pp | +15.42 | +16,796 |
| Reflexion P1 Adj - Reflexion P0 | +13 | +11.71pp | +8.82 | -3,868 |
| Reflexion P1 Adj - ExpC | -1 | -0.90pp | +2.18 | effective only: -77,708 |
| Reflexion P1 Adj - ExpA | -5 | -4.50pp | -14.86 | effective only: -89,424 |
| ExpeL E2 - ExpC | +13 | +11.71pp | +10.57 | effective: +5,453 |
| ExpeL E2 - ExpA | +9 | +8.11pp | -6.47 | effective: -6,264 |

Interpretation note:
- The two last token deltas use `Reflexion P1 Adj` effective-token accounting, not full runtime accounting.
- Under full runtime accounting, Reflexion is the most expensive Step 4 method.

## Per-Task Success Rate (%)

| Task Type | ExpA Online | ExpB Offline | ExpC ReAct | Reflexion P0 | Reflexion P1 Adj | ExpD ExpeL E2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `boil` | 33.3 | 33.3 | 44.4 | 11.1 | 11.1 | 44.4 |
| `chemistry-mix` | 37.5 | 37.5 | 25.0 | 12.5 | 25.0 | 37.5 |
| `grow-plant` | 30.0 | 30.0 | 0.0 | 10.0 | 30.0 | 50.0 |
| `inclined-plane-friction-unnamed-surfaces` | 40.0 | 50.0 | 60.0 | 20.0 | 50.0 | 30.0 |
| `lifespan-longest-lived-then-shortest-lived` | 50.0 | 60.0 | 50.0 | 40.0 | 40.0 | 70.0 |
| `melt` | 66.7 | 44.4 | 33.3 | 11.1 | 22.2 | 88.9 |
| `mendelian-genetics-known-plant` | 90.0 | 70.0 | 80.0 | 70.0 | 80.0 | 70.0 |
| `mendelian-genetics-unknown-plant` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `power-component` | 60.0 | 0.0 | 20.0 | 20.0 | 40.0 | 20.0 |
| `test-conductivity` | 10.0 | 10.0 | 20.0 | 30.0 | 30.0 | 40.0 |
| `test-conductivity-of-unknown-substances` | 0.0 | 20.0 | 30.0 | 10.0 | 20.0 | 30.0 |
| `use-thermometer` | 90.0 | 100.0 | 80.0 | 60.0 | 90.0 | 100.0 |
| **Overall** | **41.44** | **39.64** | **37.84** | **25.23** | **36.94** | **49.55** |

## Per-Task Avg Tokens / Episode

| Task Type | ExpA Online | ExpB Offline | ExpC ReAct | Reflexion P0 | Reflexion P1 Adj |
| --- | ---: | ---: | ---: | ---: | ---: |
| `boil` | 200,966 | 175,097 | 159,992 | 108,739 | 98,646 |
| `chemistry-mix` | 261,656 | 250,817 | 258,236 | 95,659 | 114,322 |
| `grow-plant` | 256,668 | 239,968 | 274,285 | 132,425 | 188,795 |
| `inclined-plane-friction-unnamed-surfaces` | 303,752 | 185,137 | 188,694 | 117,220 | 66,474 |
| `lifespan-longest-lived-then-shortest-lived` | 33,324 | 8,189 | 21,186 | 7,015 | 16,743 |
| `melt` | 161,424 | 197,764 | 177,501 | 107,238 | 92,299 |
| `mendelian-genetics-known-plant` | 119,182 | 115,187 | 127,504 | 118,552 | 139,359 |
| `mendelian-genetics-unknown-plant` | 265,132 | 262,135 | 260,979 | 156,250 | 167,558 |
| `power-component` | 139,671 | 241,265 | 193,826 | 177,030 | 69,348 |
| `test-conductivity` | 225,908 | 238,997 | 235,758 | 178,620 | 120,871 |
| `test-conductivity-of-unknown-substances` | 305,607 | 240,674 | 249,348 | 112,796 | 140,054 |
| `use-thermometer` | 74,312 | 55,614 | 94,138 | 57,216 | 58,749 |

## Main Findings

1. Experiment D (ExpeL) has the highest Step 4 success rate: `49.55%` with `57.78` average raw score.
2. Experiment A (online React_FM) has the highest raw average score: `64.25`, with `41.44%` success rate.
3. Experiment B remains a viable lower-cost memory setting among React_FM-style runs, but ExpeL now ranks above it in success rate.
4. Experiment C is a strong baseline and remains competitive on multiple tasks, especially `boil`, `inclined-plane-friction-unnamed-surfaces`, and `test-conductivity-of-unknown-substances`.
5. Reflexion is sensitive to evaluation convention:
   raw `pass1` underestimates its effect because it re-runs already-solved episodes without memory;
   corrected `P1 Adj` improves substantially over `P0` (`25.23% -> 36.94%`, `40.57 -> 49.39`).
6. Even under the corrected convention, Reflexion trails ExpA by `4.50` percentage points and ExpeL by `12.61` percentage points in success rate.
7. On cost, Reflexion only appears cheap if one reports selected-output tokens. Under actual runtime accounting, its lower-bound average cost is `212,640` tokens per episode, higher than ExpA, ExpB, ExpC, and ExpeL effective cost.
8. `mendelian-genetics-unknown-plant` remains unsolved in all Step 4 settings and should be treated as a persistent hard case rather than a memory-format issue.

## Step 4 Conclusion

Step 4 supports three concrete conclusions:

1. ExpeL (Experiment D) is the strongest success-rate setting on ScienceWorld core tasks after a full insight-collection epoch.
2. Online React_FM (Experiment A) remains the strongest raw-score setting and learns online in the test stream without requiring a pre-collection epoch.
3. Offline memory (Experiment B) is a viable lower-cost React_FM-style alternative, but it does not surpass Experiment A or ExpeL in success rate.
4. Reflexion-style episode-level retry can help relative to its own first pass when evaluated with the corrected convention, but it is not the strongest Step 4 method overall and is expensive in true runtime tokens.

Therefore, the main Step 4 takeaway is not that either in-loop or episode-level memory is universally best, but that:

- memory can help on ScienceWorld
- the benefit is highly task-dependent
- unconditional retry or unconditional memory use is not optimal
- ExpeL-style batch insight extraction is strong when a complete collection epoch is allowed
- the most defensible next step remains selective intervention rather than always-on intervention

## Correctness Audit

This summary was cross-checked against:

- saved JSON summaries for Experiments A/B/C
- saved JSON summaries for Reflexion `pass0` and raw `pass1`
- corrected Reflexion numbers recomputed directly from `sw_step4_reflexion_online_formal_online_checkpoint.json`
- ExpeL Epoch 2 numbers from `sw_step4_expD_expel_test_2.json` and insight artifact task types from `sw_expel_insights.json`

Audit points:

- success counts match source files exactly
- normalized `avg_score` values match source summaries exactly for A/B/C and Reflexion raw pass files
- corrected Reflexion `P1 Adj` metrics are recomputed from per-episode records, not copied from runner output
- token caveat for Reflexion is explicitly documented to avoid conflating selected-output cost with actual runtime cost
- ExpeL output has 111 episodes; per-task counts match Step 4; `insight_stats.total_entries = 200`; `total_retrievals = 69`; all insight task types are official ScienceWorld task names

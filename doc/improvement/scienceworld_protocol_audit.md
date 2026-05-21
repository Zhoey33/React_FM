

# ScienceWorld 实验协议固定版

> 状态: v1，后续 ScienceWorld 正式实验、重算和审计均以本文件为准。
> 核心原则: 主表只报告核心 test；所有分数必须从 episode JSON 逐条重算。

## 1. 数据协议

正式主实验使用 Step 4 核心 test 协议:

- split: `test`
- task types: 12 个核心任务
- total episodes: 111
- variation 选择: 每个 task 取 ScienceWorld API 在 `test` split 返回的前 `min(10, available)` 个 variations
- step limit: `100`

核心任务与 episode 数:


| Task                                         | Episodes |
| -------------------------------------------- | -------- |
| `boil`                                       | 9        |
| `melt`                                       | 9        |
| `use-thermometer`                            | 10       |
| `power-component`                            | 5        |
| `test-conductivity`                          | 10       |
| `test-conductivity-of-unknown-substances`    | 10       |
| `grow-plant`                                 | 10       |
| `chemistry-mix`                              | 8        |
| `lifespan-longest-lived-then-shortest-lived` | 10       |
| `inclined-plane-friction-unnamed-surfaces`   | 10       |
| `mendelian-genetics-known-plant`             | 10       |
| `mendelian-genetics-unknown-plant`           | 10       |


正式命令必须显式写出 `--split test --tasks ... --max-variations 10 --step-limit 100`，不要依赖脚本默认值。

## 2. 指标协议

主表固定报告:

- `Success Rate`: `success=true` 的 episode 比例。
- `Avg Raw Score`: `mean(episodes[].score)`，保留 ScienceWorld 原生负分。
- `Tokens / Episode`: `summary.total_tokens / total_envs`，并在附表拆分 `agent / judge / extractor`。
- `Steps`: `mean(episodes[].total_steps)`。

禁止事项:

- 不得把 clamped score 当 raw score 报告。
- 不得混用 `summary.avg_score` 和逐 episode 重算值；论文表格优先使用重算值。
- 正式主表不报告 normalized / clamped score；这些只用于旧结果审计时解释口径差异。

## 3. Memory 协议

ScienceWorld memory 的主作用域固定为 `task_type`:

- 同一 task type 的 variations 可共享 memory。
- 不做 topic fallback。
- 不做 global fallback。
- 没有同 task type 命中时返回空 memory。

正式 memory item 使用 failure-recovery 结构，优先保留:

- `failure_action`
- `failure_observation`
- `failure_type`
- `repair_strategy`
- `repair_tactic`
- `repair_action`

`question_text` 视为旧字段，后续不作为 ScienceWorld 主协议字段。

## 4. Online / Offline 口径

React-FM Offline:

- memory 来源只能是 train/dev 或预先冻结的 memory store。
- test 期间 read-only，不允许写入新 memory。
- 如果 runner 在 test 中写 memory，该结果不得称为严格 Offline。

React-FM Online:

- 允许在 test stream 中写入和检索 memory。
- 必须在表格或 caption 中标注 `test-time adaptation`。
- 不得与 Offline 放在同一训练/测试假设下解释。

## 5. 结果文件要求

后续正式 JSON 至少应能重算:

- split、task list、max variations、step limit
- `task_type`、`variation_idx`、`score`、`success`、`total_steps`
- `agent_tokens`、`judge_tokens`、`extractor_tokens`、`total_tokens`
- memory scope 与是否 test-time writable

Step 4 旧结果可用于审计，但部分 episode 未保存 `variation_idx`；如需最终论文级复现，应按本协议补日志或重跑受影响实验。

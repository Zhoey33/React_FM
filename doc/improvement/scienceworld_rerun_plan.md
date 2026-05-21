<!-- 本文件给出 ScienceWorld 主线从代码修正到正式重跑的执行决策，不复用不合理旧结果。 -->

# ScienceWorld 重设计与重跑计划

> 原则: 旧 Step 4 结果只作诊断，不作论文主表。代码不对齐就先改代码，再重跑。

## 1. 主表保留方法

正式主表只放能按统一协议重跑的方法:

- `ReAct`: 纯 baseline。
- `React-FM Offline`: train/dev memory，test read-only。
- `React-FM Online`: test-time writable memory，必须标注 test-time adaptation。
- `Reflexion`: paper-aligned `solved@trial`，例如 `solved@2`。
- `ExpeL`: 只有在完成 paper-aligned 重实现后才放主表；否则移到诊断实验。

## 2. 必须修的代码

先做四类工程修正:

- 所有 ScienceWorld runner 写入协议字段: `split`、`tasks`、`max_variations`、`step_limit`、`variation_idx`。
- 统一 summary: `avg_score` 与 `avg_raw_score` 都表示 raw score，不再输出 normalized / clamped。
- 统一成本: `agent_tokens`、`reflection_tokens`、`extractor_tokens`、`judge_tokens`、`total_tokens`。
- 禁止正式 eval 中跳过 episode；skip 只能用于诊断或明确的 effective-cost 附表。

## 3. Baseline 重设计

### ReAct

保留现有执行逻辑，但补日志字段后重跑 111 test episodes。

### Reflexion

重写 runner 为:

1. 对每个 episode 从 trial 0 开始。
2. 如果当前 trial 成功，停止并记录 solved trial。
3. 如果失败且未到 `max_trials`，生成 reflection，reset 同一 episode 后继续。
4. 报告 `solved@1`、`solved@2`、必要时 `solved@N`，不报告 raw pass1 作为主结果。

### ExpeL

当前实现不作为正式 baseline。若保留 ExpeL，需要按原论文重做:

1. 在 train/dev 上做 Reflexion-style experience gathering。
2. 保存 success / failure trajectories。
3. 用 failure-success pairs 和 success chunks 抽取 insights。
4. 实现 insight update: `ADD / EDIT / UPVOTE / DOWNVOTE` 或等价的 importance count。
5. test 时完整单次评估，不跳过 episode。
6. prompt 同时注入 insights 和 task-similarity retrieved successful trajectories。

## 4. 重跑顺序

1. 修公共日志与 summary。
2. 重跑 ReAct。
3. 重跑 React-FM Offline / Online。
4. 修并重跑 Reflexion `solved@2`。
5. 决定是否投入 ExpeL faithful baseline；如果不做，只在附录称为 diagnostic，不进主表。

## 5. 论文写法边界

可以写:

- `ReAct` is a pure baseline under the fixed ScienceWorld protocol.
- `React-FM Online` uses test-time adaptation.
- `Reflexion` is reported as cumulative solved rate under max trials.

不能写:

- Step 4 old results are final.
- ExpeL Step 4 is a faithful held-out ExpeL baseline.
- Reflexion raw pass1 is the final Reflexion result.
- React-FM Online raw score is `64.25`.

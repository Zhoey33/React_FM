

# React-FM 精简改进计划

> 日期: 2026-05-21  
> 当前定位: ScienceWorld 为主实验；ALFWorld 和 WebShop 为强辅助实验；HotPotQA 暂停。  
> 核心原则: 先修正和验证现有 ScienceWorld 证据链，再考虑组合实验和扩展实验。

## 1. 当前主线

React-FM 不应写成“全面超过 Reflexion / ExpeL”的方法。更稳的主张是:

> React-FM 通过失败检测和循环内记忆注入，在长程交互任务中提供局部错误恢复能力，能提升 ReAct agent 的过程推进和在线纠偏能力。

这篇论文最需要证明三件事:

1. **主实验结果可信**: ScienceWorld 指标口径必须统一，尤其是 `64.25` 和 `55.24` 的冲突。
2. **错误检测器可信**: detector 不能只是乱触发，需要有 precision / recall / failure type accuracy。
3. **记忆注入真的修错**: memory 注入后，下一步动作确实更合理，能带来 recovered@k 或过程分数提升。

## 2. 实验定位

### ScienceWorld: 主实验

ScienceWorld 是论文主战场。所有紧急工作都先围绕它展开。

主表建议包含:


| 方法               | 设置                               | Success Rate | Avg Raw Score | Avg Tokens / Episode | 备注                       |
| ---------------- | -------------------------------- | ------------ | ------------- | -------------------- | ------------------------ |
| ReAct            | frozen                           | 待核对          | 待核对           | 待核对                  | 基础 baseline              |
| Reflexion        | adjusted multi-trial             | 待核对          | 待核对           | 待核对                  | episode-level reflection |
| ExpeL            | epoch insight                    | 待核对          | 待核对           | 待核对                  | batch insight extraction |
| React-FM Offline | train/dev memory, test read-only | 待核对          | 待核对           | 待核对                  | 测 memory 迁移              |
| React-FM Online  | test stream update               | 待核对          | 待核对           | 待核对                  | 测 online adaptation      |


必须先解决:

- `scienceworld_step4_summary_cn.md` 中 React-FM Online 写的是 `64.25`。
- 从源 JSON 逐 episode 重算可能是 `55.24`。
- 正式论文只保留 raw score 口径；clamped / normalized 只用于解释旧结果冲突。

### ALFWorld: 强辅助实验

ALFWorld 用来证明动作级失败恢复最清晰。

保留证据:

- ReAct / Warning-only / React-FM 对比。
- failure-event recovered@k。
- 2-3 个明确案例，例如 closed container、`Nothing happens`、重复无效动作。

### WebShop: 强辅助实验

WebShop 用来验证局部失败恢复是否能迁移到网页交互。

保留证据:

- ReAct / Warning-only / React-FM 对比。
- Success Rate、Average Reward、Task Score。
- 搜索失败、属性不匹配、重复点击或无效选择后的修正案例。

### 暂停项

暂时不要做:

- HotPotQA。
- ToolBench。
- 大规模 WebShop 工程重构。
- Random memory 全量消融，除非 reviewer 明确质疑“多给上下文就有效”。
- Reflexion + React-FM / ExpeL + React-FM 组合实验，先放后面。

## 3. 主执行清单

### 1. 固定 ScienceWorld 实验协议

目标: 明确 ScienceWorld 数据集怎么用、实验用什么 split / task / variation、memory 的细粒度是什么，并与 Step 4 保持一致。

要做:

1. 固定数据集协议:
  - 使用 Step 4 核心 test 协议。
  - 12 个核心 task type。
  - 111 个 test episodes。
  - 保持当前 task / variation 选择与 Step 4 文档一致。
2. 固定评价指标:
  - Success Rate。
  - Avg Raw Score。
  - Avg Normalized Score。
  - Tokens / Episode。
  - Steps
  - 不再额外记录 clamped / normalized score，避免和 raw score 混用。
3. 固定 memory 粒度:
  - ScienceWorld memory 以 `task_type` 为主要 scope。
  - 记忆项保持 failure-recovery 结构。
  - 细粒度字段优先沿用 Step 4 / gate pipeline 中已有的 tiered memory:
    - `failure_action`
    - `failure_observation`
    - `failure_type`
    - `repair_strategy`
    - `repair_tactic`
    - `repair_action`
    - `question_text（废弃）`
4. 明确 online / offline 口径:
  - React-FM Offline: train/dev memory，test read-only。
  - React-FM Online: test stream 中可写 memory，必须标注 test-time adaptation。
  - 如果旧实验 test 期间写了 memory，就不能叫严格 Offline。

产物:

- `doc/scienceworld_protocol_audit.md`

验收标准:

- ScienceWorld 的 split、task list、variation、memory scope、metrics 都写清楚。
- 后续所有实验都按这一份协议执行。

### 2. 代码漏洞排查: ReAct / Reflexion / ExpeL 论文对齐审计

目标: 现阶段先不做实现修改，而是审计 ScienceWorld 代码中 ReAct、Reflexion、ExpeL 是否与各自原论文的核心机制对齐。排查结果要区分三类问题:


| 类型      | 含义                                      | 后续处理               |
| ------- | --------------------------------------- | ------------------ |
| 实现漏洞    | 与原论文机制或实验口径明显不一致，并会影响结果                 | 必须修复，受影响结果重算或重跑    |
| 适配差异    | 因 ScienceWorld 环境限制产生的合理改写              | 文档中说明，不一定重跑        |
| 统计/报告问题 | 运行轨迹可复用，但 summary、token 或 adjusted 口径不清 | 用源 JSON 重算，修正文档和表格 |


#### 2.1 ReAct 对齐审计

原论文关键机制:

- 使用少量示例提示，让模型在交互中交替输出 reasoning trace 和 environment action。
- 每一步基于当前任务、历史 action / observation 和少量示例生成下一步。
- 不使用外部 memory、不做事后反思、不做跨 episode 学习。
- 成功率和过程得分应来自环境真实反馈，而不是人工修正或选择性统计。

需要核对的代码点:

1. Prompt 对齐:
  - ReAct baseline 是否只包含 few-shot、task observation、当前 episode 历史。
  - baseline prompt 中是否意外注入了 React-FM memory、failure hint、detector 信息或其他额外规则。
  - `think:` 与 action 的格式是否符合 ReAct 思路，且没有强行泄露目标步骤。
2. 执行循环对齐:
  - 每一步是否只执行一个模型输出动作。
  - action post-processing 是否只是解析有效动作，而不是把错误动作改成更优动作。
  - 重复动作、空动作、invalid action 的处理是否会系统性偏向 ReAct 或惩罚 ReAct。
3. 评估口径:
  - `success` 是否只由 ScienceWorld score / done 判定。
  - `score` 是否保存 raw score，不能和 clamped score 混用。
  - step limit、task list、variation、split 是否和 React-FM、Reflexion、ExpeL 完全一致。
4. Token 口径:
  - baseline token 是否只包含 agent call。
  - 不能把 detector、extractor 或 memory 相关 token 算进 ReAct。

审计产出:

- ReAct 是否是纯 baseline。
- 如果不是，列出污染来源、影响范围、是否需要重跑。

#### 2.2 Reflexion 对齐审计

本项目中的实现和原论文的思路是否对齐

#### 2.3 ExpeL 对齐审计

本项目的实现和原论文是否对齐

#### 2.4 横向一致性审计

三类 baseline 必须统一核对:

1. 环境协议:
  - split、task list、variation、max envs、step limit、seed 是否一致。
  - env reset / skip / reset_to_episode 是否不会造成 episode index 错位。
2. 指标协议:
  - `success` 与 raw score 的定义固定。
  - summary 必须能从 episode JSON 逐条重算。
  - negative score 不能在不同方法间有的保留、有的 clamp。
3. Prompt 协议:
  - 三个 baseline 是否使用同一 ScienceWorld few-shot 基础，或差异有明确理由。
  - system prompt 中额外规则是否会让某个方法获得不公平优势。
4. 成本协议:
  - agent calls、reflection calls、insight extraction calls、judge/extractor calls 分开统计。
  - 主表至少报告 full cost；必要时附表报告 effective cost。

已发现需要重点核对的问题:

- React-FM Online 的 `summary.avg_score=64.25` 可能来自 negative score clamp，而逐 episode raw score 平均约为 `55.24`。
- ExpeL E2 有 skipped episodes，需要区分 all-episode result、non-skipped rerun result 和 effective-cost result。
- Reflexion 若 pass0 成功后仍执行 pass1，需要明确 full two-pass cost；若报告 adjusted result，需要单独说明 selection 规则。

产物:

- `doc/scienceworld_code_audit.md`
- 审计表: `method / paper expectation / current implementation / mismatch / severity / action / rerun needed`
- 如有漏洞，列出必须修复项和是否需要重跑。

验收标准:

- 能判断现有 ReAct、Reflexion、ExpeL 的 ScienceWorld 结果是否可作为论文 baseline。
- 对每个偏离原论文的地方，明确它是漏洞、合理适配还是报告口径问题。
- 如果发现漏洞，明确影响哪些方法、哪些结果、是否需要重跑。

### 3. 工程修正与日志补强

目标: 只做支撑主证据链所需的工程修正，不做大规模重构。

优先修正:

1. 增加 ScienceWorld 指标核对脚本。
2. 增加 failure-event JSONL 日志。
3. 增加 recovered@1 / recovered@3 统计脚本。
4. 增加 detector quality 标注/统计脚本。
5. 增加 memory generation quality 标注/统计脚本。
6. 增加 post-injection correction 标注/统计脚本。
7. 必要时优化上下文构建:
  - 缩短无关历史。
  - 明确区分 task goal、recent history、failure signal、retrieved memory。
  - memory 注入后增加 applicability decision。

failure-event JSONL 至少包含:

```json
{
  "episode_id": "sw_042",
  "task_type": "melt",
  "variation_idx": 3,
  "step": 18,
  "failure_type": "implicit_no_progress",
  "failed_action": "move to workshop",
  "failure_observation": "You move to the workshop.",
  "memory_mode": "relevant",
  "retrieval_attempted": true,
  "retrieval_hit": true,
  "retrieved_memory_ids": ["sw_m_0017"],
  "injected_memory_text": "...",
  "next_action": "go to kitchen",
  "score_before_failure": 12.0,
  "score_after_1_step": 12.0,
  "score_after_3_steps": 18.0,
  "recovered_within_1_step": false,
  "recovered_within_3_steps": true
}
```

产物:

- 指标核对脚本。
- failure-event 日志。
- 三类分析脚本: detector、memory generation、post-injection correction。

验收标准:

- 后续分析可以从日志直接生成表格，不需要人工翻大量原始轨迹。

### 4. ScienceWorld 主实验

目标: 在完成代码排查后，决定复用旧结果还是重跑。

决策规则:


| 排查结果                                   | 处理方式                  |
| -------------------------------------- | --------------------- |
| 没有影响实验结论的漏洞                            | 复用旧结果，但用新脚本重算指标       |
| 只有 summary 统计口径问题                      | 不重跑，用源 JSON 重算并修正文档   |
| prompt / memory 注入有小问题但不影响主结果          | 旧结果可作为已有结果，新版本补充小规模验证 |
| online/offline 泄漏、上下文构建严重错误、score 记录错误 | 必须重跑受影响方法             |


主实验方法:


| 方法               | 状态                           |
| ---------------- | ---------------------------- |
| ReAct            | 先排查，可复用则复用                   |
| Reflexion        | 先排查 adjusted 口径              |
| ExpeL            | 先排查 skipped / effective cost |
| React-FM Offline | 排查是否真正 test read-only        |
| React-FM Online  | 修正 score 口径后再决定是否复用          |


产物:

- `results/scienceworld_final_metrics.csv`
- `doc/scienceworld_final_metric_audit.md`

验收标准:

- ScienceWorld 主表数字可信。
- 每个数字可追溯到源 JSON 或重跑日志。

### 5. 检测质量分析实验

目标: 证明错误检测器合理高质量。

标注集:

- 从 ScienceWorld 轨迹中抽样 100-200 个 step。
- 包含 detector 触发和未触发样本。
- 覆盖成功、失败、高分未成功 episode。
- 覆盖多个 task type。

每个样本标注:

```json
{
  "episode_id": "sw_042",
  "task_type": "melt",
  "step": 18,
  "action": "move to workshop",
  "observation": "You move to the workshop.",
  "detector_prediction": true,
  "detector_failure_type": "implicit_no_progress",
  "gold_is_failure": true,
  "gold_failure_type": "wrong_location",
  "gold_needs_repair": true
}
```

汇报指标:


| 指标                  | 含义                 |
| ------------------- | ------------------ |
| Precision           | detector 触发时是否真是失败 |
| Recall              | 真实失败中有多少被抓到        |
| F1                  | 综合检测质量             |
| Type Accuracy       | 失败类型是否正确           |
| False Positive Rate | 正常步骤被误判比例          |
| False Negative Rate | 失败步骤漏检比例           |


验收标准:

- 显式失败检测应可靠。
- 隐式失败可以承认更难，但要有成功案例。
- 如果 precision 很低，先修 detector，不继续跑新实验。

### 6. 记忆生成质量分析实验

目标: 证明 memory extractor 生成的失败恢复记忆本身质量足够高。

抽样:

- 从 ScienceWorld memory store 中抽 50-100 条 memory。
- 覆盖多个 task type。
- 覆盖成功 episode 和失败 episode 后生成的 memory。

标注类别:


| 类别                     | 含义                    |
| ---------------------- | --------------------- |
| high_quality           | 失败原因清楚，修正方案具体且可执行     |
| usable_but_weak        | 基本可用，但过于笼统、过于具体或信息不完整 |
| invalid                | 修正方案错误、不可执行或与失败无关     |
| duplicate_or_redundant | 与已有 memory 重复，新增价值低   |


同时记录:

- failure_action 是否准确。
- failure_observation 是否包含关键失败证据。
- repair_strategy 是否合理。
- repair_action 是否可执行。
- 是否过度绑定具体状态。

产物:

- Memory Generation Quality 表。
- 2-3 条高质量 memory 案例。
- 1-2 条低质量 memory 失败案例。

验收标准:

- 能说明 memory 不是随意文本，而是可用的 failure-recovery 单元。
- 如果低质量比例高，优先修 extractor prompt 或 memory schema。

### 7. 证明记忆注入后确实修正错误

目标: 证明 memory 注入不是装饰，而是改变了失败后的下一步行为。

从 failure events 中抽 20-30 个 relevant memory 注入事件，标注:


| 类别                      | 含义           |
| ----------------------- | ------------ |
| error_corrected         | 注入后修正了原错误    |
| partial_progress        | 未完全修正，但方向更合理 |
| no_change               | 行为没有明显改善     |
| wrong_repair            | 采取了错误修正      |
| invalid_or_unexecutable | 修正动作不可执行     |
| memory_caused_new_error | memory 引入新错误 |


同时统计:

- recovered@1
- recovered@3
- score_delta@3
- judge_progress@3，如果 env score 太稀疏

验收标准:

- 至少能说明 React-FM 在一部分 failure event 上带来明确修正或过程推进。
- 如果多数是 `no_change` 或 `wrong_repair`，先改注入 prompt 或 memory schema。

## 4. 后置实验

### ExpeL + React-FM

放在主执行清单第 1-7 项完成之后再做。

目的:

> ExpeL 提供 episode-start 的全局 task-level insight；React-FM 提供执行过程中的 failure-triggered local repair。组合实验用于验证二者是否互补。

实验表:


| 方法               | Episode-start insight | In-loop failure memory | 目的          |
| ---------------- | --------------------- | ---------------------- | ----------- |
| ExpeL            | 是                     | 否                      | 全局经验        |
| React-FM         | 否                     | 是                      | 局部失败修复      |
| ExpeL + React-FM | 是                     | 是                      | 全局经验 + 局部修复 |


解释:

- 如果 `ExpeL + React-FM > ExpeL`: 强支持互补性。
- 如果 `ExpeL + React-FM ≈ ExpeL`: 说明整体 success rate 增益有限，但仍可看 failure-event recovery。
- 如果 `ExpeL + React-FM < ExpeL`: 检查 detector 误触发、上下文冲突、memory 过具体。

### Reflexion + React-FM

第二后置实验，可选。

风险:

- token 成本高。
- adjusted result 口径复杂。
- 成功 episode 是否重跑必须说清楚。

建议只在 ExpeL + React-FM 跑通后再考虑。

## 6. 停止条件

满足以下任意两条，可以继续推进论文:

1. ScienceWorld React-FM 在 Avg Raw Score、Judge Progress 或 recovered@k 上优于 ReAct / Warning-only。
2. detector quality 显示显式失败检测可靠，隐式失败有可解释案例。
3. 记忆注入后多数样本能修正错误或带来 partial progress。
4. ALFWorld 或 WebShop 能复现局部失败恢复现象。

需要收窄或暂停的情况:

1. ScienceWorld raw score 优势经核对后消失。
2. detector precision 很低，导致大量正常步骤被误触发。
3. detector recall 很低，导致真实失败没有触发记忆。
4. memory 注入后多数是 `no_change` 或 `wrong_repair`。
5. recovered@k 没有提升。

如果触发暂停条件，论文 claim 应降级为:

> React-FM 是一个可解释的失败分析框架，当前结果显示 failure-triggered intervention 有潜力，但 detector / memory usage 仍是主要瓶颈。

## 7. 最终证据链

```text
ScienceWorld 指标可信
→ detector 高质量触发
→ memory 注入后修正错误
→ recovered@k / process score 提升
→ 主实验性能或过程推进改善
→ 后续再验证 ExpeL / Reflexion 互补性
```

## 8. 当前 Todo List

1. 确定 ScienceWorld 的数据集怎么用，实验的时候用什么，memory 的细粒度是什么。这部分和 Step 4 那部分保持一致就好。
2. 排查 ScienceWorld 的代码是否有漏洞，修正范围包括 React-FM、Reflexion、ExpeL。需要注意看一下每次的上下文构建是否可以优化。
3. 进行工程修正，如 failure-event JSONL 等。
4. 进行 ScienceWorld 主实验。如果上述代码没有漏洞，则复用旧结果；如果有漏洞，需要重新跑受影响实验。
5. 做检测质量分析实验。
6. 做记忆生成质量分析实验。
7. 证明记忆注入后确实修正错误。

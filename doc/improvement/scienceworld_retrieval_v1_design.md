<!-- Design notes for ScienceWorld deterministic retrieval gating in React-FM. -->

# ScienceWorld Retrieval v1 设计

## 背景

2026-05-21 的 online pilot 说明主链路已经跑通: detector 会触发，episode 结束后会生成 memory，后续同 task type episode 能检索并注入 memory。

但 pilot 也暴露了一个关键问题:

- `memory.min_score=0.0` 时，只要同 task type bucket 有 memory，就会注入 top-1。
- hybrid RRF score 在小候选池里几乎都挤在 `0.0323-0.0328`，不能作为可靠相关性阈值。
- 大量 `syntax_or_parse` / `action_loop` / `redundant_repeat` failure 会反复拿到门类 memory，导致 agent 重复无效动作。
- `precondition_blocked` 的门未开场景又确实能被 memory 修正，不能简单关闭 retrieval。

所以 v1 不继续调 RRF `min_score`，而是把 retrieval 改成 deterministic gated retrieval。

## 设计目标

v1 目标不是追求最强检索模型，而是让正式 online 实验里的 memory 注入更可解释:

```text
检测到 failure
→ 从同 task_type memory 中召回候选
→ 用 failure_type 和轻量规则过滤明显不适用候选
→ 给候选打 deterministic relevance score
→ 只注入最高分且超过阈值的一条 memory
```

这版不在 retrieval 阶段额外调用 LLM。原因是:

- online failure 数量多，逐 failure judge 成本高。
- 当前要判断的大量坏例是可规则化的重复非法动作。
- detector 已经可能调用 judge LLM；retrieval 再加 judge 会让成本口径更复杂。

## 检索与过滤

第一层 hybrid retrieval 只负责候选召回，使用 failure-side 信息:

- `failure_type`
- failed action
- failure observation
- task type bucket

Memory 的 repair-side 字段不作为第一层召回核心:

- `repair_strategy`
- `repair_tactic`
- `repair_action`

这些字段只用于 relevance scoring、safety gate 和最终 prompt 注入。

v1 先按 failure type 做 compatible filter。例如:

- `precondition_blocked` 可匹配 `precondition_blocked / ambiguity / syntax_or_parse`
- `syntax_or_parse` 可匹配 `syntax_or_parse / ambiguity`
- `redundant_repeat` 可匹配 `redundant_repeat / action_loop / implicit_no_progress`

然后执行 safety gate，直接拒绝:

- repair action 为空。
- repair action 与当前 failed action 相同。
- 当前 observation 是 `No known action matches that input`，但 repair 仍重复该非法动作。
- 当前 failure 是 `action_loop / redundant_repeat`，但 repair action 出现在最近 2 个动作里。
- failure type 不兼容。

对 `precondition_blocked + The door is not open` 保留一个明确正例规则: 如果 repair action 包含 `open door`，允许进入排序。这保留了 pilot 中最有效的“先开门再移动”修复。

## Relevance Score

每条候选 memory 得到一个 `retrieval_relevance_score`，范围 `0.0-1.0`。分数由轻量特征组成:

- failure type exact/compatible match。
- failed action 与 memory failure action 的 token overlap。
- observation 与 memory failure observation 的 token overlap。
- repair action 是否覆盖当前失败需要的关键动作或目标。
- memory `confidence_score` 小权重加分。

默认阈值:

```yaml
memory.relevance_score_threshold: 0.45
```

如果所有候选低于阈值，本次 failure 不注入 memory。

## 日志口径

`memory_retrieved` 和 failure-event 的 `retrieval_hit` 只表示最终实际注入 memory，不再表示“召回到了候选”。

为了分析被拒绝的候选，failure step 和 failure-event JSONL 记录:

- `retrieval_candidate_memory_ids`
- `retrieval_candidate_scores`
- `retrieval_candidate_relevance_scores`
- `retrieval_selected_memory_id`
- `retrieval_relevance_decision`
- `retrieval_rejection_reason`
- `retrieval_filtered_by_type_count`
- `retrieval_filtered_by_safety_count`

这样 post-injection correction 分析可以区分:

- 没有候选。
- 有候选但 type 不兼容。
- 有候选但 safety gate 拒绝。
- 有候选但 relevance score 不够。
- 最终实际注入并影响下一步 prompt。

## 实验解释

正式论文中这版 retrieval 应解释为:

> React-FM injects a failure memory only when the retrieved memory is type-compatible with the current failure and passes deterministic safety/relevance checks.

不要把它写成“LLM reranker”或“learned retriever”。它是一个保守的、可解释的 gated retrieval policy。

下一步实验必须先重跑 9-episode online pilot，重点检查:

- `syntax_or_parse` 的 retrieval hit 是否显著下降。
- `precondition_blocked` 门类 memory 是否仍能注入。
- retrieval-hit recovered@3 是否不再明显低于 no-hit。
- post-injection harmful 样本是否减少。

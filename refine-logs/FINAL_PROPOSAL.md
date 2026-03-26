# React_FM v2 — 简洁版方案

## 1. Problem Anchor

ReAct agent 在工具调用失败后缺乏从过去失败中学习的能力，反复犯同样的错误。React_FM 在 ReAct 循环中加入一层失败记忆：检测失败 → 查记忆 → 有经验就用，没有就尝试其他方法 → 成功就存入记忆。

## 2. 方法

### 2.1 整体流程

```
for each step in ReAct loop:
    thought, action = LLM(prompt)
    observation = execute(action)

    if is_failure(observation):
        # 1. 从记忆中检索相似失败
        similar_cases = retrieve(memory, action, observation)

        if similar_cases:
            # 2. 带着历史经验重新推理
            thought, action = LLM(prompt + similar_cases)
        else:
            # 3. 没有经验，提示尝试其他方法
            thought, action = LLM(prompt + "上一步失败了，请尝试其他方法")

        observation = execute(action)

        if is_success(observation):
            # 4. 成功了，存入记忆
            memory.store(failure=原始失败, solution=成功的action)

    update prompt with (thought, action, observation)
```

就这么简单。

### 2.2 失败检测

不需要复杂的分类体系。直接看工具返回：

- 报错信息（exception, error message, non-zero exit code）
- 空结果 / 无效输出
- LLM 自己判断（"这个结果不对"）

先用**规则判断**跑通，之后如果发现规则不够再加 LLM judge。

### 2.3 记忆格式

每条记忆就是一个 **(问题, 解决方案)** 对：

```json
{
  "failure": "调用 search API 时参数格式错误，返回 400 Bad Request",
  "solution": "search API 的 query 参数需要 URL encode，且 date 格式必须是 YYYY-MM-DD",
  "tool": "search_api",
  "created_at": "2026-03-24"
}
```

核心字段只有 2 个：`failure` 和 `solution`。`tool` 和 `created_at` 是辅助字段，方便过滤。

不需要：confidence、recurrence_count、priority、status、lifecycle state。这些**等发现确实需要时再加**。

### 2.4 检索

保留混合检索，因为它确实有用：

1. **先按 tool 名过滤**（精确匹配，缩小范围）
2. **再按语义相似度排序**（embedding cosine similarity）
3. **取 top-k（k=3）**

检索公式简化为：
```
candidates = memory.filter(tool == current_tool)
results = top_k(candidates, by=cosine_sim(embed(query), embed(candidate.failure)), k=3)
```

如果 tool 过滤后候选为空，退化为全局语义检索。

### 2.5 Human-in-the-loop（可选扩展）

作为可选模块，不是核心贡献：
- 当记忆中没有相似案例，且 agent 自己也解决不了时，可以请求人类帮助
- 人类给出的解决方案同样存入记忆

## 3. 与已有工作的区别

| | React_FM v2 | Reflexion | ExpeL | Traj-Memory |
|---|---|---|---|---|
| 什么时候用记忆 | 工具调用失败时 | 整个任务结束后 | 任务开始时 | 任务开始时 |
| 记忆内容 | (失败, 解决方案) | 自由文本反思 | 自由文本经验 | 自由文本 tips |
| 跨任务复用 | ✅ | ❌ | ✅ | ✅ |
| 在线写入 | ✅ 成功即存 | ❌ | ❌ | ❌ |

核心差异就一句话：**在失败发生的那一刻检索和学习，而不是事后总结。**

## 4. 先跑起来，再优化

v2 方案刻意保持简单。以下是**观察到问题后才加**的机制：

| 观察到的问题 | 再加的机制 |
|---|---|
| 记忆太多，检索变慢 | 加 memory size 上限 + 去重 |
| 存了错误的解决方案 | 加验证（repair 后确认任务有进展） |
| 同一个失败存了多条 | 加 pattern_key 去重 |
| 旧记忆不再有效 | 加 TTL 或 LRU 淘汰 |
| 规则检测漏判 | 加 LLM judge |
| 检索不够精准 | 加 error_type 字段做二级过滤 |

**不要提前优化。先拿到实验结果。**

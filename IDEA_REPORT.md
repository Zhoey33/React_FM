# React_FM — Idea Discovery Report

**Direction**: LLM Agent Memory — Failure Recovery
**Idea**: React_FM (React with Failure Memory)
**Date**: 2026-03-24
**Pipeline**: research-lit → idea-creator (GPT-5.4) → novelty-check → research-review (GPT-5.4) → refine → benchmark survey

---

## Executive Summary

React_FM 在 ReAct 循环中加入一层失败记忆：当工具调用失败时，检测失败 → 从记忆库中检索相似的过去失败及其解决方案 → 带着经验重新推理 → 成功则存入记忆。方法刻意保持简洁：记忆格式只有 (failure, solution) 对，检索用语义相似度 + tool 过滤，先跑通再迭代优化。

核心差异一句话：**在失败发生的那一刻检索和学习，而不是事后总结。**

---

## 1. Literature Landscape

### 30+ 篇论文调研，关键相关工作

| Paper | Venue | Core Idea | 与 React_FM 的区别 |
|-------|-------|-----------|-------------------|
| **Reflexion** | NeurIPS 2023 | 自由文本自我反思 | 临时性反思，不跨任务，不结构化 |
| **ExpeL** | AAAI 2024 | 离线经验提取 | 非 in-loop，自由文本 |
| **Voyager** | NeurIPS 2023 | 成功技能库 | 只存成功，不存失败 |
| **LATS** | ICML 2024 | MCTS + LLM | 搜索式，非记忆检索式 |
| **Agent KB** | ICML 2025 WS | 通用知识库 + 生命周期 | 非失败专注，规划时检索 |
| **AgentDebug** | Stanford 2025 | 失败分类 + 单次调试 | 无持久化跨任务记忆 |
| **Traj-Informed Memory** | IBM 2026 | 事后从轨迹提取 tips | Post-hoc，自由文本 tips |
| **SAGE** | Amazon 2025 | RL + 技能库 | RL训练，技能是代码函数 |
| **Meta-Policy Reflexion** | 2025 | 谓词式策略规则 | 存策略规则，非 (原因,方案) 对 |
| **ReMe** | 2025 | 过程记忆蒸馏 | 通用记忆，非失败专注 |

### 工程先例

**OpenClaw self-improving-agent skill**：为 coding agent 设计的自我改进系统。核心机制：
- 结构化错误日志 `{id, category, error, context, fix, recurrence_count, pattern_key}`
- 晋升规则：重复 ≥ 3 次 + 跨 ≥ 2 任务 → 自动升级到项目记忆
- PostToolUse hook 自动检测工具调用失败
- 用最简单的方式（grep 检索、Markdown 存储）解决了同一个问题

**启示**：简洁有效 > 复杂精妙。React_FM v2 的设计哲学直接受此启发。

### 核心差异化

**没有任何现有系统同时具备：**
1. **In-loop** — 在失败发生时立即检索（非事后）
2. **结构化记忆** — (failure, solution) 对（非自由文本）
3. **在线写回** — 成功即存，记忆持续增长
4. **跨任务复用** — 上个任务的失败经验帮助下个任务

---

## 2. 方案（v2 简洁版）

### 2.1 核心流程

```
for each step in ReAct loop:
    thought, action = LLM(prompt)
    observation = execute(action)

    if is_failure(observation):
        similar_cases = retrieve(memory, action, observation)

        if similar_cases:
            thought, action = LLM(prompt + similar_cases)   # 带经验重新推理
        else:
            thought, action = LLM(prompt + "请尝试其他方法")  # 没经验，试别的

        observation = execute(action)

        if is_success(observation):
            memory.store(failure=原始失败, solution=成功的action)  # 成功即存

    update prompt with (thought, action, observation)
```

### 2.2 失败检测

先用规则判断跑通，不够再加 LLM judge：
- 报错信息（exception, error message, non-zero exit code）
- 空结果 / 无效输出
- LLM 自己判断（"这个结果不对"）

### 2.3 记忆格式

```json
{
  "failure": "调用 search API 时参数格式错误，返回 400 Bad Request",
  "solution": "search API 的 query 参数需要 URL encode，date 格式必须是 YYYY-MM-DD",
  "tool": "search_api",
  "created_at": "2026-03-24"
}
```

核心就 2 个字段：`failure` 和 `solution`。其余是辅助字段。

### 2.4 检索

```
candidates = memory.filter(tool == current_tool)
results = top_k(candidates, by=cosine_sim(embed(query), embed(candidate.failure)), k=3)
```

先按 tool 过滤 → 再按语义相似度排序 → 取 top-3。tool 过滤为空时退化为全局检索。

### 2.5 与已有工作的区别

| | React_FM v2 | Reflexion | ExpeL | Traj-Memory |
|---|---|---|---|---|
| 什么时候用记忆 | 工具调用失败时 | 任务结束后 | 任务开始时 | 任务开始时 |
| 记忆内容 | (失败, 解决方案) | 自由文本反思 | 自由文本经验 | 自由文本 tips |
| 跨任务复用 | ✅ | ❌ | ✅ | ✅ |
| 在线写入 | ✅ 成功即存 | ❌ | ❌ | ❌ |

### 2.6 先跑起来，再优化

| 观察到的问题 | 再加的机制 |
|---|---|
| 记忆太多，检索变慢 | memory size 上限 + 去重 |
| 存了错误的解决方案 | 写入前验证（确认任务有进展） |
| 同一个失败存了多条 | pattern_key 去重 |
| 旧记忆不再有效 | TTL 或 LRU 淘汰 |
| 规则检测漏判 | 加 LLM judge |
| 检索不够精准 | 加 error_type 二级过滤 |

---

## 3. Benchmarks（5个）

### 选择标准：大家都用、好部署、失败重复率高、适合记忆系统

| Benchmark | 部署 | 任务类型 | 规模 | 失败重复性 | 对比论文 | 选择理由 |
|-----------|------|---------|------|-----------|---------|---------|
| **ALFWorld** | MEDIUM (pip) | 文本环境，6类家务 | 134 环境 | 极高 | Reflexion, ExpeL, LATS, AgentDebug | 6类任务高度重复，最能展示学习曲线 |
| **HotpotQA** | EASY (pip) | 多跳QA + 搜索 | 采样 500 | 高 | Reflexion, ExpeL, LATS, AgentDebug, Agent KB | 所有记忆论文标配，直接对比 |
| **tau-bench** | EASY (pip) | 客服，API调用 | retail + airline | 极高 | 无记忆论文用过 | pass^k 天然适合记忆系统，首个用它的记忆论文 |
| **GAIA** | EASY (HF) | 通用助手，多工具 | 165 验证集 | 高 | AgentDebug, Agent KB | 主流 leaderboard，可直接对比 |
| **WebShop** | MEDIUM (16GB) | 网页购物 | 采样 500 | 高 | Reflexion（无提升）, ExpeL, LATS | Reflexion 在此失败，我们能提升就很有说服力 |

### Benchmark 详细分析

**ALFWorld** — 只有 6 类任务（pick, clean, heat, cool, examine, put），失败模式极度重复。记忆"微波炉在厨房不在卧室"可以直接迁移。Reflexion 77%, ExpeL 59%, ReAct ~50%。

**HotpotQA** — 搜索工具返回无关结果是系统性失败。记忆"搜 X 不如搜 Y"可复用。⚠️ 有数据污染担忧，用 hard subset。

**tau-bench** — pass^k 指标（k=1,2,4,8）天然测一致性。GPT-4o pass^1 < 50%, pass^8 < 25%。记忆政策违规规则应直接提升 pass^k。

**GAIA** — 466 题分 3 级。失败按难度聚类。AgentDebug +26%, Agent KB +16pp。

**WebShop** — Reflexion 在此完全无提升（负面结果）。如果 React_FM 能在 Reflexion 失败的地方成功，是很强的证据。

---

## 4. 实验计划

### 4.1 核心问题

**失败记忆能不能让 agent 少犯重复错误、提高成功率？**

### 4.2 多轮积累设计（方案 A）

React_FM 的效果取决于失败经验的积累。一轮跑完记忆可能还没积累起来，所以采用多 epoch 设计：

```
Epoch 1: 记忆为空，冷启动
Epoch 2: 继承 Epoch 1 的记忆
Epoch 3: 继承 Epoch 1+2 的记忆

对比：
- Epoch 1 vs Epoch 3 的成功率差（学习效果）
- 单 epoch 内前 N 题 vs 后 N 题（积累曲线）
- 记忆库大小 vs 成功率（饱和点在哪）
```

各 benchmark 特殊处理：
- **ALFWorld**：3 epoch × 134 环境 = 402 次
- **HotpotQA**：500 题顺序跑 1 遍，切 5 段看趋势
- **tau-bench**：用 pass^k（k=1/2/4/8），天然测多轮一致性
- **GAIA**：L1→L2→L3 分层，看低级经验能否帮高级任务
- **WebShop**：500 题顺序跑，切段看趋势

### 4.3 Baselines

| Baseline | 描述 | 目的 |
|----------|------|------|
| **ReAct** | 原始 ReAct，无记忆 | 下界 |
| **Reflexion** | 自由文本反思，任务内 | 最常比较的 baseline |
| **ExpeL** | 离线经验提取 | 另一种记忆方式 |
| **简单重试** | 失败后直接重试，无记忆 | 证明不是重试本身带来的提升 |

### 4.4 核心指标

**主要指标：**
- 任务成功率（每个 epoch 分别报告）
- 学习曲线（成功率 vs 任务序号 / epoch）
- pass^k（仅 tau-bench）

**辅助指标：**
- 重复失败抑制率（同类失败在有记忆后是否减少）
- Token 消耗（记忆检索的额外成本）
- 记忆利用率（被检索并实际使用的记忆占比）

### 4.5 消融实验（有主结果后再做）

| 消融 | 改动 | 验证什么 |
|------|------|---------|
| 无记忆 | 去掉 TFM | 记忆是否有用 |
| 随机检索 | 随机返回记忆 | 检索质量是否重要 |
| 无在线写入 | 固定记忆不新增 | 在线学习是否重要 |
| 自由文本记忆 | 自然语言而非结构化 | 格式是否重要 |

### 4.6 实验步骤

```
Step 1: Pilot（1-2天）
  - ALFWorld 上跑 React_FM vs ReAct，1 epoch
  - 验证代码跑通、失败检测有效、记忆能积累

Step 2: ALFWorld 完整实验（2-3天）
  - 3 epoch 完整跑，画学习曲线
  - 加入 Reflexion、ExpeL baseline

Step 3: 扩展到其他 4 个 benchmark（5-7天）
  - 按部署难度：tau-bench → HotpotQA → GAIA → WebShop

Step 4: 消融实验（2-3天）
  - 在 ALFWorld + HotpotQA 上跑

Step 5: 分析（1-2天）
  - 画图、统计、case study
```

### 4.7 实现优先级

```
P0（必须做）:
  - ReAct 基础框架
  - 失败检测（规则判断）
  - 记忆存储（JSON）
  - 语义检索（embedding + cosine sim）
  - 记忆写回（成功即存）
  - 多 epoch 实验框架

P1（pilot 后根据需要加）:
  - tool 名过滤
  - 记忆去重
  - LLM judge 失败检测

P2（正式实验才考虑）:
  - 记忆上限和淘汰
  - 置信度/频率统计
  - Human-in-the-loop
```

---

## 5. 审稿反馈摘要

### GPT-5.4 NeurIPS 模拟审稿

**评分**：4/10, Weak Reject（基于 v1 过度复杂方案）

**主要弱点及 v2 的应对策略：**

| 审稿意见 | v2 怎么应对 |
|----------|-----------|
| 新颖性过强 | 收窄 claim 到"in-loop 失败记忆"，不夸大 |
| 方法太复杂 | v2 砍掉所有不必要的复杂度 |
| 失败检测 oracle 风险 | 先用规则判断，后续可加 LLM judge 做对比 |
| 评估不够尖锐 | 多 epoch 设计 + 学习曲线 + pass^k |
| Baseline 太弱 | 加入简单重试 baseline |
| 记忆污染 | 先不管，观察到问题再加验证 |
| 像 systems paper | 用实验结果说话，而非堆砌机制 |

**v2 的核心策略：简洁方法 + 强实验结果 > 复杂方法 + 弱实验。**

---

## 6. 风险和应对

| 风险 | 等级 | 应对 |
|------|------|------|
| 时间压力（领域进展快） | 高 | 尽快实现 pilot，快速迭代 |
| 记忆没被用到 | 中 | ALFWorld 和 tau-bench 重复率极高 |
| 多轮没提升 | 中 | 检查记忆质量，加写入验证 |
| 检索不相关 | 中 | 加 tool 过滤，调 embedding |
| Token 成本太高 | 低 | top-k=3，压缩记忆文本 |
| Epoch 2/3 提升来自见过题目 | 中 | 打乱任务顺序做对照 |
| 新颖性被挑战 | 中 | 实验证明简洁方法就是有效 |

---

## 7. 项目文件索引

```
React_FM/
├── IDEA_REPORT.md                         ← 本文件（总报告）
├── BENCHMARK_LANDSCAPE.md                 ← Benchmark 全景调研（20个）
└── refine-logs/
    ├── FINAL_PROPOSAL.md                  ← 简洁版方案
    └── EXPERIMENT_PLAN.md                 ← 实验计划
```

---

## 8. Next Steps

- [ ] 实现 React_FM 原型（ReAct + 失败记忆模块）
- [ ] 在 ALFWorld 上跑 Pilot（1-2天）
- [ ] 拿到正向信号后扩展到全部 5 个 benchmark
- [ ] 消融实验
- [ ] 写论文

---

## References

### 核心相关工作
- [Reflexion (NeurIPS 2023)](https://arxiv.org/abs/2303.11366)
- [ExpeL (AAAI 2024)](https://arxiv.org/abs/2308.10144)
- [Voyager (NeurIPS 2023)](https://arxiv.org/abs/2305.16291)
- [LATS (ICML 2024)](https://arxiv.org/abs/2310.04406)
- [Agent KB (ICML 2025 WS)](https://arxiv.org/abs/2507.06229)
- [AgentDebug (Stanford 2025)](https://arxiv.org/abs/2509.25370)
- [Trajectory-Informed Memory (IBM 2026)](https://arxiv.org/abs/2603.10600)
- [SAGE (Amazon 2025)](https://arxiv.org/abs/2512.17102)
- [Meta-Policy Reflexion (2025)](https://arxiv.org/abs/2509.03990)
- [ReMe (2025)](https://arxiv.org/abs/2512.10696)
- [CFGM (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.659/)

### 工程参考
- [OpenClaw self-improving-agent](https://playbooks.com/skills/openclaw/skills/self-improving-agent)

### Benchmarks
- [ALFWorld (ICLR 2021)](https://github.com/alfworld/alfworld)
- [HotpotQA (EMNLP 2018)](https://hotpotqa.github.io/)
- [tau-bench (Sierra 2024)](https://github.com/sierra-research/tau-bench)
- [GAIA (ICLR 2024)](https://arxiv.org/abs/2311.12983)
- [WebShop (NeurIPS 2022)](https://webshop-pnlp.github.io/)

### Surveys
- [Agent Benchmark Survey (KDD 2025)](https://arxiv.org/abs/2507.21504)
- [Agent Evaluation Survey (2025)](https://arxiv.org/abs/2503.16416)

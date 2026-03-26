# React_FM v2 — 实验计划（简洁版）

## 1. 核心问题

只验证一件事：**失败记忆能不能让 agent 少犯重复错误、提高成功率？**

## 2. Benchmarks（5个）

| Benchmark | 部署难度 | 任务类型 | 规模 | 失败重复性 | 多轮策略 | 对比论文 |
|-----------|---------|---------|------|-----------|---------|---------|
| **ALFWorld** | MEDIUM (pip) | 文本环境，6类家务任务 | 134 环境 | 极高 | 3 epoch 跑完整 134 个环境，记忆跨 epoch 积累 | Reflexion, ExpeL, LATS, AgentDebug |
| **HotpotQA** | EASY (pip) | 多跳QA + 搜索工具 | 采样 500 | 高 | 顺序跑 500 题，记忆持续积累，对比前100 vs 后100 | Reflexion, ExpeL, LATS, AgentDebug, Agent KB |
| **tau-bench** | EASY (pip) | 客服场景，API 工具调用 | retail + airline 两个域 | 极高 | 用 pass^k 指标（k=1,2,4,8），天然测多轮一致性 | 无（首个记忆论文用它） |
| **GAIA** | EASY (HF) | 通用助手，多工具 | 165 验证集 | 高 | 按难度分层跑，L1→L2→L3 顺序积累 | AgentDebug, Agent KB |
| **WebShop** | MEDIUM (16GB) | 网页购物 | 采样 500 | 高 | 顺序跑，记忆持续积累 | Reflexion（此处无提升）, ExpeL, LATS |

### 为什么选这5个

1. **ALFWorld** — 6类任务高度重复，最能展示"越用越好"的学习曲线
2. **HotpotQA** — 所有记忆论文的标配，直接对比
3. **tau-bench** — pass^k 指标天然适合记忆系统，且无人用过（新贡献）
4. **GAIA** — 主流 leaderboard，Agent KB 用过可直接对比
5. **WebShop** — Reflexion 在这上面没提升，如果我们能提升就很有说服力

## 3. 多轮积累实验设计（方案 A）

React_FM 的核心优势是记忆随使用积累。实验需要体现这一点：

### 所有 benchmark 通用设计

```
Epoch 1: 记忆为空，从零开始积累（冷启动）
Epoch 2: 继承 Epoch 1 的记忆继续跑
Epoch 3: 继承 Epoch 1+2 的记忆继续跑

对比：
- Epoch 1 vs Epoch 3 的成功率差（学习效果）
- 前 N 个任务 vs 后 N 个任务的成功率（单 epoch 内的积累）
- 记忆库大小 vs 成功率的曲线（越多越好？还是饱和？）
```

### 各 benchmark 特殊处理

- **ALFWorld**：每个 epoch 跑完整 134 环境，3 epoch = 402 次
- **HotpotQA**：500 题顺序跑 1 遍，切分为 5 段（每段 100 题）看趋势
- **tau-bench**：用自带的 pass^k 指标，k=1/2/4/8，分别报告
- **GAIA**：按 Level 1→2→3 跑，看低级经验能否帮助高级任务
- **WebShop**：500 个任务顺序跑，切分看趋势

## 4. Baselines

| Baseline | 描述 | 目的 |
|----------|------|------|
| **ReAct** | 原始 ReAct，无任何记忆 | 下界 |
| **Reflexion** | 自由文本反思，任务内 | 最常比较的 baseline |
| **ExpeL** | 离线经验提取 | 另一种记忆方式 |
| **简单重试** | 失败后直接重试同一动作，无记忆 | 证明不是重试带来的提升 |

## 5. 核心指标

**主要指标：**
- 任务成功率（每个 epoch 分别报告）
- 学习曲线（成功率 vs 任务序号/epoch）
- pass^k（仅 tau-bench）

**辅助指标：**
- 重复失败抑制率：同类失败在有记忆后是否减少
- Token 消耗：记忆检索的额外成本
- 记忆利用率：被检索到并实际使用的记忆占比

## 6. 消融实验（有了主结果后再做）

| 消融 | 改动 | 验证什么 |
|------|------|---------|
| 无记忆 | 去掉 TFM | 记忆到底有没有用 |
| 随机检索 | 随机返回记忆而非语义检索 | 检索质量重要吗 |
| 无在线写入 | 固定记忆，不再新增 | 在线学习重要吗 |
| 自由文本记忆 | 用自然语言存储而非结构化 | 格式重要吗 |

## 7. 实验步骤

```
Step 1: Pilot（1-2天）
  - 在 ALFWorld 上跑 React_FM vs ReAct，1 epoch
  - 验证代码能跑通、失败检测有效、记忆能积累
  - 判断：有正向信号 → 继续

Step 2: ALFWorld 完整实验（2-3天）
  - 3 epoch 完整跑，画学习曲线
  - 加入 Reflexion、ExpeL baseline

Step 3: 扩展到其他 4 个 benchmark（5-7天）
  - 按部署难度：tau-bench → HotpotQA → GAIA → WebShop
  - 每个 benchmark 跑完整多轮实验

Step 4: 消融实验（2-3天）
  - 在 ALFWorld + HotpotQA 上跑消融

Step 5: 分析（1-2天）
  - 画图、统计、case study
```

## 8. 实现优先级

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

## 9. 风险和应对

| 风险 | 应对 |
|---|---|
| 记忆没被用到（失败模式不重复） | ALFWorld 和 tau-bench 的重复率极高，至少这两个不会有问题 |
| 多轮没提升（记忆质量差） | 检查存入的记忆是否正确，加写入前验证 |
| 检索到不相关记忆 | 加 tool 过滤，调整 embedding 模型 |
| Token 成本太高 | 限制 top-k=3，压缩记忆文本 |
| WebShop 部署困难 | 放在最后做，16GB RAM 不够就放弃 |
| Epoch 2/3 提升来自"见过题目"而非记忆 | 打乱任务顺序做对照实验 |

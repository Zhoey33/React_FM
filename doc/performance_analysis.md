# React_FM 性能分析：为什么在 WebShop/HotPotQA/ToolBench 上表现不如 ExpeL/Reflexion

> 日期：2026-04-14
> 基于三个 Explore Agent 的代码/数据深度审计 + Codex 讨论

## 主实验结果

| Method | ALFWorld | WebShop | ScienceWorld | HotPotQA (EM) | ToolBench |
|--------|----------|---------|--------------|---------------|-----------|
| Baseline | 91.0% | 38.0% | 10.0% | 0.396 | 62.8% |
| Reflexion | 94.0% | 45.0% | 18.0% | 0.441 | 81.4% |
| ExpeL | 94.8% | 49.0% | 12.0% | 0.440 | 90.6% |
| **React_FM E2** | **95.5%** | 44.0% | **22.0%** | 0.430 | 75.3% |

React_FM 在 ALFWorld 和 ScienceWorld 上排名第一，但在 WebShop、HotPotQA、ToolBench 上排名第三。

---

## 核心框架：战术修正 vs 战略适应

Codex 提出的核心概括：

> **React_FM 是一个战术修正 (tactical correction) 方法，Reflexion/ExpeL 是战略适应 (strategic adaptation) 方法。**

React_FM 的 failure-recovery pair 存储的是 "在步骤 T 做了 X 失败了，应该改做 Y" 这种具体的动作级修正。这在失败模式是动作级、可恢复、且跨 episode 重现的任务上效果最好。但当失败是策略级的（需要改变整体方法而非单个动作）或实例特异性的（修复方案不可迁移），就不如 Reflexion/ExpeL 的抽象反思/洞察。

---

## 三个 Benchmark 的具体病因

### 1. WebShop (React_FM 44% vs ExpeL 49%)

#### 失败模式统计
- **96.8% 的检测到的失败是搜索相关的**（592/611）
- 主要类型：`action_loop`（重复相同搜索词 ≥3 次）、`no_results`、`unproductive`
- 48-50/100 个 episode 耗尽了全部 15 步预算

#### 根因分析

**A. 搜索失败不可迁移**

存储的 failure-recovery pair 高度产品特异：
```
failure_action: "search[double sided machine washable decorative pillows printing technology 28x28]"
solution_action: "search[28x28 double sided machine washable decorative pillow printing]"
```
下一个 episode 可能要找跑鞋或厨房用品——这条记忆与新任务几乎零重叠。BM25+embedding 检索基本是随机匹配。

相比之下，Reflexion/ExpeL 生成的是**任务策略泛化**：*"搜索词太具体会失败；先去掉修饰词，搜索核心名词"*——这种策略跨产品类别迁移。

**B. Per-env 隔离导致 E1 零命中**

`FailureMemoryStore.retrieve()` 默认 `cross_env=False`，按 `env_idx` 隔离。100 个 env 顺序跑，每个 env_idx 只访问一次：
- Env 0 存储 2 条记忆 → 存入 slot `env_idx=0`
- Env 1 检索 → 查询 slot `env_idx=1` → **0 候选 → 0 命中**

**E1 全程 611 次检索尝试，0 次命中。** 记忆被写入但从未被跨 episode 读取。

**C. 短时域（15 步）**

15 步上限非常紧。失败检测平均每 episode 触发 6.1 次，agent 在搜索循环中不断收到来自不同产品的无关记忆注入，可能**损害**上下文连贯性。Reflexion/ExpeL 在 episode 边界总结完整轨迹，生成高层洞察后全新开始——更匹配搜索策略级失败。

**D. 格式消融佐证**

WebShop 格式消融实验结果反转了 ALFWorld 的模式：
| Format | WebShop E2 SR |
|--------|--------------|
| Success trajectory | **48.0%** |
| Reflexion reflection | **46.5%** |
| Failure-recovery (ours) | 44.0% |

高层指导（成功做了什么/应该避免什么）比具体的失败修复序列更有效。

---

### 2. HotPotQA (React_FM 0.430 vs ExpeL 0.440)

#### 失败模式统计
| Failure Type | Count | % |
|---|---|---|
| `search_not_found` | 901 | **76.8%** |
| `other` (unproductive) | 175 | 14.9% |
| `lookup_exhausted` | 94 | 8.0% |
| `invalid_action` | 3 | 0.3% |

#### 根因分析

**A. 超短时域（8 步）**

HotPotQA 是极短时域任务。成功的问题平均 4.65 步解决，失败的问题平均 5.60 步（大多触及上限）。失败检测触发后，仅剩 1-2 步可用——几乎没有恢复空间。

**B. 实体特异性失败**

76.8% 的失败是 Wikipedia 实体消歧：
```
FAIL: Search[Danny DeVito animated movie]
FIX:  Search[Kool Kojak] → Search[Danny DeVito voice actor animated movie]
```
修复方案 "试试 Search[Tunak Tunak Tun]" 对任何其他问题都没用。这是**问题唯一的实体导航错误**，failure-recovery pair 从结构上就不可迁移。

**C. 记忆注入反而有害**

关键数据：
- 注入记忆的 episode 成功率：**31.3%**
- 未注入记忆的 episode 成功率：**55.0%**

检索到的其他问题的记忆是**噪声**，干扰了 agent 的推理。E2 中 207/285 失败 episode 触发了检测（73%），但 episode 仍然失败——记忆检索被反复触发（最多每 episode 18 次）却无法挽救。

**D. 需要元策略而非实体替换**

Reflexion 生成 *"应该搜索具体的文章标题，而非描述性短语"* 这种元策略，跨问题迁移。ExpeL 提取 *"搜索 Wikipedia 时用精确文章标题"* 这种跨 episode 规则。React_FM 存储的 `Search[X] → Search[Y]` 是原始实体替换，不泛化。

---

### 3. ToolBench (React_FM 75.3% vs ExpeL 90.6%)

#### 失败模式统计
- **2,670 次失败检测**，765 个 episode（平均 3.5 次/episode）
- 143 个 episode 触及 10 步上限，全部有 ≥5 次失败检测
- 主要类型：`no_cache`、`api_error`、`unknown_function`、`action_loop`

#### 根因分析

**A. API 缓存未命中是根本性不可恢复的**

ToolBench（StableToolBench）使用**静态回放缓存**——只有预录制的 `(category, tool_name, api_name, exact_params_dict)` 组合才返回真实数据。缓存键是 `str(tool_input)` 精确字符串匹配。

如果 agent 用有效但未缓存的参数调用 API，**永远**返回 `"no cached response"`。这不是 action-level memory 能解决的——你不知道哪些参数组合被缓存了。

**B. 零检索命中**

Per-env 隔离 + 记忆在 episode 结束后才存储 + 检索在 episode 中间进行 = **2,670 次检索全部返回空**。React_FM 的记忆机制在整个 765 episode 运行中**一次都没有生效**。

花费了 901K extractor tokens 和 153K judge tokens 存储 211 条失败模式，但从未被召回。

**C. ExpeL 的元策略完美匹配**

ExpeL 的 `InsightEntry.rule` 存储任务类级别的泛化：
- *"当搜索航班时，先试最通用的城市名格式"*
- *"不要重试失败的 API 调用，用已有信息推理"*

这些规则在 episode 开始前注入（step 0），按 `task_type="tool_use"` 粗匹配——可靠的检索。ExpeL E1→E2 从 62.6% 跳到 **90.6%**（+28pp）。

**D. 短时域 + Token 浪费**

10 步上限，3-4 步浪费在 cache miss 上。React_FM 每 episode 多花 **25% token**（7.7K vs ExpeL 4.9K），但没有任何收益。

---

## React_FM 的最优工作域

| 条件 | React_FM 胜出 | ExpeL/Reflexion 胜出 |
|------|-------------|---------------------|
| 时域长度 | **长**（30-50步）：ScienceWorld, ALFWorld | **短**（8-15步）：HotPotQA, WebShop, ToolBench |
| 失败类型 | **动作级、可恢复**：开错容器、去错位置 | **策略级**：错误搜索方法、错误推理链 |
| 失败可迁移性 | **同类模式重现**：多个 episode 犯同样的错 | **实例特异性**：每个产品/问题/API 不同 |
| 失败来源 | **环境交互**：agent 可以改变动作来修复 | **外部限制**：API 缓存、知识缺失 |
| 触发率影响 | **高触发率 + 高恢复率** | **高触发率 + 低恢复率**（噪声注入） |

### 量化支撑（Operating Regime Table）

| Benchmark | Max Steps | Trigger Rate | Dominant Failure | React_FM Δ | Rank | Best |
|-----------|-----------|-------------|-----------------|------------|------|------|
| ScienceWorld | 30 | 84% | Action-level | **+12.0%** | **#1** | React_FM |
| ALFWorld | 50 | 6.7% | Action-level | +4.5% | **#1** | React_FM |
| ToolBench | 10 | — | External (API) | +12.5% | #3 | ExpeL |
| WebShop | 15 | 49% | Mixed | +6.0% | #3 | ExpeL |
| HotPotQA | 8 | — | Strategic | +3.4% | #3 | ExpeL |

---

## Codex 建议的改进方向

### 1. 双层记忆系统 (Dual-Memory)
保留两种记忆类型：
- **具体的 episodic 记忆**：用于长时域具身任务（ALFWorld, ScienceWorld）
- **抽象的策略记忆**：用于短时域/搜索密集任务（WebShop, HotPotQA）

检索时根据任务类型或失败类型选择。

### 2. 检索门控 / 有用性预测 (Retrieval Gating)
HotPotQA 的数据表明问题不仅是记忆质量，还有**何时注入**。

注入前判断：
- 失败类型是否属于已知可迁移类别
- 相似度是否足够高
- 剩余步数是否足够
- 检索到的记忆是否通过有用性过滤

### 3. 失败类型感知记忆 (Failure-Type-Specific Memory)
不同失败类别触发不同输出：
- 搜索失败 → 策略提示（"去掉修饰词"）
- 动作执行失败 → 具体修正对（"先开容器再取物品"）
- 外部/工具失败 → 退避策略（"不要重试未缓存的 API"）

### 4. 选择性跨环境共享 (Selective Cross-Env Sharing)
按任务相似度聚类，而非简单的全局共享或完全隔离：
- 相似任务集内共享记忆
- 高方差/外部脆弱域（ToolBench）阻止共享
- Per-env 隔离是保守实现（保精度损召回），不是理论限制

### 5. 外部失败的元策略 (Meta-Policies for External Failures)
对于 ToolBench 这类外部失败主导的任务，正确的记忆不是 "用 Y 参数调 X API"，而是：
- 避免重复失败的工具调用
- 优先用已有信息推理
- 退回到替代工具或总结不确定性

这更接近控制策略 (control policy) 而非 episodic memory。

---

## 论文框架建议

### 当前框架（已采用）
"Complementary methods for different task regimes" — React_FM 和 ExpeL/Reflexion 互补。

### Codex 建议的更强框架
提出一个记忆类型分类法 (Memory Taxonomy)：

| 工作域 | 最佳记忆类型 |
|--------|------------|
| 长时域 + 重复动作失败 | 具体的循环内失败修正记忆 (React_FM) |
| 短时域 + 搜索/规划失败 | 抽象反思/策略 (Reflexion/ExpeL) |
| 外部/工具不可靠 | 元策略/退避启发式 |
| 混合环境 | 混合记忆系统 |

> 这把论文从 "我们的方法赢/输" 提升到 **"什么粒度的记忆匹配什么任务域"** 这个更有价值的研究贡献。

---

## 关键引用数据来源

| 数据点 | 来源 |
|--------|------|
| WebShop E1 零命中 | `results/20260405_195045/ws_reactfm_1.json`: 0/611 hits |
| WebShop 失败类型分布 | `ws_reactfm_1.json` episodes 中 592/611 search failures |
| HotPotQA 记忆注入有害 | `results/20260407_180554/hqa_*`: 注入 31.3% vs 未注入 55% |
| ToolBench 零命中 | `results/20260408_092853/tb_reactfm_1.json`: 0/2670 hits |
| 格式消融 WebShop 反转 | `results/20260413_115001/ws_ablation_success_2.json`: SR=48.0% |

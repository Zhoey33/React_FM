# ScienceWorld Gate Model Issues & Diagnosis

**Date**: 2026-04-14
**Status**: 诊断完成 — 已定位多层根因，实施分级 repair 方案

---

## 问题总览

ScienceWorld 门控模型训练后，policy value = 0.113，与 always-none 相同。模型本质上没有学到任何有用的干预策略。

**根因总结**（按重要性排序）：
1. **BUG-1: max_steps=50** — 任务根本跑不完（已修复）
2. **MEMORY-1: 50% 记忆不匹配** — repair 注入了不相关内容（已修复：分级 repair）
3. **EVAL-1: 37.5% 评估盲区** — 不同行为但相同零分（已验证 LLM-as-judge 方案可行）
4. **BUG-2: bypass_threshold** — 门控永远返回 none（待修复）
5. **SIGNAL-1: 70% oracle=none** — 干预信号本质弱（部分归因于记忆不匹配）

---

## Bug 类已修复

### BUG-1: max_steps = 50（官方默认 100）⚠️ CRITICAL — ✅ 已修复

- **位置**: `config_scienceworld.yaml` 和 `config.yaml` 中 `agent.max_steps: 50`
- **影响**: `run_scienceworld.py` 用 `min(args.step_limit=100, config.max_steps=50) = 50`
- **后果**: boil (gold path 102-192步)、melt (126-150步)、grow-plant (65-69步) 在 50 步内根本无法完成
- **修复**: 改为 `max_steps: 100`（与官方 `envStepLimit=100` 一致）
- **影响范围**: 所有 ScienceWorld 主实验结果（baseline、React_FM、Reflexion）需要重跑

### BUG-2: bypass_threshold = 0.1，但 max RRF = 0.033 ⚠️ CRITICAL — 未修复

- **位置**: `src/gate/train_gate.py:213` 的 `predict_arm()` 方法
- **影响**: `retrieval_rrf_score < bypass_threshold` 恒成立 → 门控模型永远返回 "none"，XGBoost 模型从未被咨询
- **验证**: 禁用 bypass 后 policy value 反而更差 (0.107)，说明模型本身也有问题
- **待修复**: 需要设 `bypass_threshold=0.0` 或校准到实际 RRF 分布

### BUG-3: 检索特征全零 — ✅ 已通过 backfill 修复

- **位置**: `experiments/gate/run_branched_rollouts.py` 的 `lookup_memory_entry()`
- **原因**: 在内存对象上 backfill rrf_score/margin，但没有持久化到 checkpoint JSON
- **修复**: 手动 backfill 脚本，36/40 CP 已填充（但 RRF 范围极小 0-0.033）

### BUG-4: memory_entry_count 用了全局计数 — ✅ 已通过 backfill 修复

- **位置**: `experiments/gate/collect_checkpoints.py`
- **原因**: `memory_store.size()` 返回全局条目数，但检索是 per-env 的
- **修复**: backfill 为 per-env 计数（0-10 vs 原始 0-155）

---

## 记忆质量问题（新发现 2026-04-14）

### MEMORY-1: 50% 记忆检索到不相关内容 ⚠️ CRITICAL — ✅ 已修复

- **诊断数据**: 40 CP 中 20 个检索到的记忆与当前失败完全不匹配
  - ✅ 完全匹配（相同 failure_action）: 17/40 = 42%
  - 🟡 部分匹配（相同动词）: 3/40 = 8%
  - ❌ 完全不匹配: 20/40 = 50%
- **典型案例**:
  - cp_0012 (boil): 失败"focus on tin"→ 检索到"move metal pot to oven"（不相关）
  - cp_0477 (lifespan): 失败"examine crocodile egg"→ 检索到"dunk jug in sink"（完全无关）
- **根因**: per-env 记忆库只有 3-4 条，没有与当前失败匹配的条目

### MEMORY-2: 记忆粒度太细（战术级 vs 战略级）⚠️ HIGH — ✅ 已修复

- **现象**: 即使完全匹配的 17 个 CP，repair 也只有 18% 比 none 更好
- **原因**: 旧记忆只存储动作修复（"先开门再放锅"），不含策略指导（"去厨房用炉子加热"）
- **agent 需要的**: "你应该把 tin 带到有热源的地方"
- **记忆提供的**: "open oven → move metal pot to oven"
- **交叉分析**:

| 匹配度 | n | repair+ | none+ | tie |
|--------|---|---------|-------|-----|
| ✅ 完全匹配 | 17 | 3 (18%) | 4 (24%) | 10 (59%) |
| 🟡 部分匹配 | 3 | 0 | 0 | 3 |
| ❌ 不匹配 | 20 | 2 (10%) | 4 (20%) | 14 (70%) |

### 修复方案: 分级 Repair (Tiered Canonicalization)

将旧的单一 `repair_text` 升级为三级指导：

| 级别 | 字段 | 作用 | 示例 |
|------|------|------|------|
| **L1 Strategy** | `repair_strategy` | 高层目标/方向 | "要融化 tin，需要把它移到有热源的地方如厨房" |
| **L2 Tactic** | `repair_tactic` | 多步计划 | "先去 hallway，再去 kitchen，把锅放到 stove 上" |
| **L3 Action** | `repair_action` | 具体命令 | "go to hallway" |

**修改的文件**:

| 文件 | 改动 |
|------|------|
| `src/memory.py` | `FailureMemoryEntry` 添加 `repair_strategy/tactic/action` 字段 + `get_repair_display()` 统一获取方法 |
| `src/gate/canonicalize.py` | 新增 `canonicalize_entry_tiered()` 和 `canonicalize_store_tiered()` |
| `src/gate/branched_rollout.py` | `build_injection_text()` 调用 `get_repair_display()` |
| `prompts/alfworld_prompts.py` | 记忆注入改用 `entry.get_repair_display()` |
| `prompts/scienceworld_prompts.py` | 同上 |
| `prompts/webshop_prompts.py` | 同上 |
| `prompts/hotpotqa_prompts.py` | 同上 |
| `prompts/toolbench_prompts.py` | 同上 |

**`get_repair_display()` 优先级**:
```
repair_strategy (分级) → repair_text (平铺 canonicalized) → solution_action (原始)
```
完全向后兼容：没有分级字段的旧记忆自动 fallback。

**注入格式** (有分级字段时):
```
[Strategy] 要融化 tin，需要把它移到有热源的地方，如厨房的炉子。
[Plan] 先去 hallway，再去 kitchen，把金属锅放到 stove 上。
[Next action] go to hallway
```

**已生成**: `memory/scienceworld_memory_tiered.json`（155 条全部升级）

---

## 评估层问题

### EVAL-1: ScienceWorld 评分粒度太粗 ⚠️ HIGH — ✅ LLM-as-judge 方案已验证可行

- **现象**: 40 个 CP 中 15 个 (37.5%) 不同 arm 产生不同轨迹但进度分数相同（都是 0）
- **LLM-as-judge 验证**: 6 个 CP 测试，检测率 83% (5/6)
  - LLM 成功区分了 score 盲区中的行为差异
  - 评判理由有说服力（能识别 gold path 里程碑匹配度）
- **原型代码**: `experiments/gate/test_llm_judge.py`
- **结果数据**: `rollouts/llm_judge_full.json`
- **Per-task 评估盲区**:
  - chemistry-mix: 3/3 CP 评估盲区 (100%)
  - power-component: 4/5 CP 评估盲区 (80%)
  - melt: 3/5 CP 评估盲区 (60%)
  - boil: 3/7 CP 评估盲区 (43%)
  - test-conductivity: 2/4 CP 评估盲区 (50%)
  - use-thermometer, find-living-thing, lifespan-longest-lived, grow-plant: 0% 评估盲区 ✅

### EVAL-2: 10 步 rollout 窗口太短 ⚠️ HIGH

- **现象**: 对长任务 (boil/melt/grow-plant)，10 步内几乎不可能达到下一个计分里程碑
- **计划修复**: R013c-SW-v2 将 rollout 增加到 15 步

### EVAL-3: 缺少中间过程评估（LLM-as-judge 方案已验证）

- **方案**: 用 LLM 对比 agent 轨迹与 gold path，判断引导是否让 agent 更接近正确方向
- **验证结果**: 6 CP 测试，5/6 成功检测差异（见 EVAL-1）
- **输入**: 任务描述 + gold path + agent 轨迹 → LLM 打分（0-10）
- **待做**: 集成到正式 utility 计算流程

---

## 信号层问题

### SIGNAL-1: 70% oracle = none，干预信号本质弱 ⚠️ FUNDAMENTAL

- **数据**: 40 CP 中 28 个 oracle arm = none，heterogeneity = 0.300（刚过及格线 0.3）
- **新理解**: 部分归因于 MEMORY-1 和 MEMORY-2 —— 不匹配的/纯战术的记忆注入当然不如不注入
- **预期改善**: 分级 repair 修复后，repair arm 应该在更多 CP 上优于 none

### SIGNAL-2: 14/40 CP 所有 arm 零进度

- **分布**: chemistry-mix (3/3), power-component (4/4), boil (3/7), melt (3/5), test-conductivity (2/4)
- **根因分析**:
  - 7 个: agent 有改善但 10 步窗口太短 → 增加步数可部分解决
  - 5 个: 任务超出 LLM 能力（电路接线语法） → 无论多少步都不行
  - 3 个: 评分粒度问题 → 需要更细粒度评估

### SIGNAL-3: 模型过拟合

- **数据**: 40 CP × 3 arms = 120 训练样本
- **表现**: train MAE=0.013, test MAE=0.171（13x 过拟合）
- **XGBoost 参数**: max_depth=4, n_estimators=100 → 对 120 行数据过于复杂

---

## 行为层确认（无 bug）

### ✅ 不同 arm 确实产生不同轨迹
- 40/40 CP 有不同动作序列
- 37/40 CP 第一步就分岔
- 平均动作差异度 75-83%

### ✅ repair 确实改善了动作质量
- 有效动作比例: repair 60.2% > none 49.7% > question 40.2%
- 零分 CP 上: repair 52.4% vs none 34.0%
- 但这种改善被粗粒度评分掩盖了

---

## Gold Path 关键发现

ScienceWorld 提供 `get_gold_action_sequence()` API（需 `generateGoldPath=True`）。

- 已提取 24 条 gold path，保存在 `rollouts/scienceworld_gold_paths.json`
- Gold path 不是最优解，包含大量冗余（温度计轮询占 46-60%，穷举式房间探索）
- 14 个零分 CP 中 12 个 prefix_match = 0（agent 从第一步就偏离 gold path）

---

## 下一步

1. ✅ 已修复 max_steps: 50 → 100
2. ✅ 已实现分级 repair（strategy/tactic/action）并生成 155 条分级记忆
3. ✅ 已验证 LLM-as-judge 评估方案可行（83% 检测率）
4. ⬜ 修复 bypass_threshold（设为 0.0）
5. ⬜ 用分级记忆 + 100 步重跑 ScienceWorld rollout
6. ⬜ 将 LLM-as-judge 集成到 utility 计算
7. ⬜ 重新训练门控模型
8. ⬜ 用 100 步重跑 ScienceWorld 主实验（baseline + React_FM）

---

## 相关文件

| 文件 | 用途 |
|------|------|
| `config_scienceworld.yaml` | SW 配置（已修复 max_steps） |
| `config.yaml` | 通用配置（已修复 max_steps） |
| `src/memory.py` | FailureMemoryEntry（已添加 repair_strategy/tactic/action） |
| `src/gate/canonicalize.py` | 记忆 canonicalize（已添加 tiered 方法） |
| `src/gate/branched_rollout.py` | 分支 rollout（已更新 repair 注入逻辑） |
| `src/gate/train_gate.py` | 门控模型训练（bypass bug 待修复） |
| `memory/scienceworld_memory_tiered.json` | 155 条分级记忆（新生成） |
| `memory/scienceworld_memory_canonicalized.json` | 155 条旧格式记忆 |
| `experiments/gate/test_llm_judge.py` | LLM-as-judge 原型（新创建） |
| `rollouts/llm_judge_full.json` | LLM judge 评估结果 |
| `rollouts/scienceworld_gold_paths.json` | 24 条 gold path |
| `rollouts/scienceworld_main_rollouts.json` | 当前 rollout 数据（10步，待重跑） |
| `checkpoints/scienceworld_checkpoints_midband.json` | 40 CP（已 backfill） |
| `src/gate/progress.py` | 进度计算逻辑 |

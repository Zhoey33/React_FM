# ScienceWorld 门控模型流程文档

**最后更新**: 2026-04-15
**适用范围**: ScienceWorld 数据集的门控模型训练与评估。其他数据集 (ALFWorld, WebShop) 可参照此流程修改。

---

## 一、流程总览

```
数据准备 → 分支 Rollout → LLM Judge 评分 → Utility 计算 + Gate 训练 → 在线评估
```

完整的离线训练需要 5 个阶段，在线推理 1 个阶段。

---

## 二、阶段详解

### Stage 0: 数据准备

#### 0.1 主实验运行 & Checkpoint 采集

```bash
# 跑 ScienceWorld 主实验 (2 epoch, 50 episodes)
caffeinate -i python experiments/run_scienceworld.py \
    --epochs 2 --max-envs 50 --run-name sw_reactfm_

# 采集 failure checkpoint
python experiments/gate/collect_checkpoints.py \
    --benchmark scienceworld \
    --results results/<timestamp>/sw_reactfm_*.json \
    --output checkpoints/scienceworld_checkpoints.json
```

**ScienceWorld 规模**:
- 30 种任务类型，每种 5-30 个 test variation
- 主实验跑 50 个 episode
- 原始采集约 500-600 个 failure checkpoint

**Checkpoint 采样策略**:
- 从 568 个原始 CP 中去重 → 252 个
- **Midband 采样**: 只保留 score ∈ [1, 99] 的 CP（排除完全没进展和已完成的）
- 按 task type 平衡采样 → 40 个 CP，覆盖 9 种任务
- 40 个的原因: API 成本控制（40 CP × 4 arms × 3 replays = 480 次 LLM 调用）

**CheckpointState 关键字段**:

| 字段 | 说明 | 用途 |
|------|------|------|
| `checkpoint_id` | 唯一标识 (cp_0001) | 关联 rollout |
| `env_idx` | 环境编号 | per-env 记忆检索 |
| `task_type` | 任务类型 (boil, melt...) | 特征提取 |
| `failure_action` | 失败的动作 | 记忆检索 query |
| `failure_observation` | 失败后的观察 | 记忆检索 query |
| `history` | [(action, obs), ...] | Rollout 重放 |
| `action_history` | [action, ...] | 环境重放 |
| `score_at_checkpoint` | 当前分数 (官方 API score，完整成功为 100，也可能为负) | progress 基准 |
| `env_state` | {task_name, variation_idx} | 环境重置 |
| `retrieval_rrf_score` | top-1 检索分数 | 特征 + bypass |
| `retrieval_margin` | top-1 vs top-2 差值 | 特征 |

#### 0.2 记忆提取 & 分级升级

**当前流程**（一步直接提取）:
```bash
# 由 run_scienceworld.py 的 post-episode 阶段自动完成
# extractor prompt 直接输出 7 个字段:
#   failure_action, failure_observation, solution_action (原始 3 个)
#   + repair_strategy, repair_tactic, repair_action, question_text (分级 4 个)
# memory.add() 一次性写入所有字段
# 输出: memory/scienceworld_memory.json (已包含分级字段)
```

**历史流程**（已废弃）: 原先是两步——先原始提取 `failure_action → solution_action`，再调用 `canonicalize_store_tiered()` 二次升级。现已合并为一步，省去 canonicalize 步骤和额外的 LLM 调用。`canonicalize.py` 仍保留用于升级旧格式记忆文件。

**分级记忆格式 (FailureMemoryEntry)**:

| 字段 | 级别 | 说明 | 示例 |
|------|------|------|------|
| `repair_strategy` | L1 Strategy | 高层目标/方向 | "要融化 tin，移到有热源的地方" |
| `repair_tactic` | L2 Tactic | 多步计划 | "去 hallway → kitchen → 放到 stove" |
| `repair_action` | L3 Action | 具体命令 | "pick up metal pot" |
| `question_text` | 诊断问题 | 不透露答案的反思引导 | "Have you considered using a different heat source?" |

**`get_repair_display()` 优先级链**:
```
repair_strategy (分级) → repair_text (平铺) → solution_action (原始)
```
完全向后兼容：没有分级字段的旧记忆自动 fallback。

#### 0.3 Gold Path 提取

```python
# ScienceWorld 提供 gold path API
gold_actions = env.get_gold_action_sequence()
```
- 已提取 24 条，保存在 `rollouts/scienceworld_gold_paths.json`
- 格式: `{task_var: {task_name, variation_idx, gold_actions[]}}`
- 注意: Gold path 包含大量冗余（温度计轮询、穷举式探索），LLM judge 需要能识别关键步骤

---

### Stage 1: 分支 Rollout

```bash
python experiments/gate/run_branched_rollouts.py \
    --benchmark scienceworld \
    --checkpoints checkpoints/scienceworld_checkpoints_midband.json \
    --memory memory/scienceworld_memory_tiered.json \
    --config config_scienceworld.yaml \
    --output rollouts/scienceworld_rollouts_v2.json \
    --n-steps 15 --n-replays 3
```

**流程**:
```
对每个 CP (40个):
  ① lookup_memory_entry(memory_store, cp)
     用 failure_action + failure_observation 检索记忆
     返回最匹配的 FailureMemoryEntry

  ② 对每个 arm (4个):
     · none     — 无注入（基准线）
     · cue      — "Reconsider your approach."（最小干预）
     · question — mem.question_text（诊断问题）
     · repair   — mem.get_repair_display()（分级修复指导）

     对每个 replay (3个):
       a. 重置 ScienceWorld 环境 (task_name, variation_idx)
       b. 重放 action_history 到 checkpoint 状态
       c. build_injection_text(arm, mem_entry) → 注入文本
       d. 跑 N 步，第 1 步注入 hint（hint_steps=1）
       e. 记录 actions, observations, score_before, score_after
       f. 计算 env progress = (score_after - score_before) / 100
```

**关键设计决策**:

| 决策 | 设定 | 理由 |
|------|------|------|
| n_steps | 15 | 长任务 (boil/melt/grow) 需要更多步才能到达评分里程碑 |
| n_replays | 3 | LLM 输出有随机性，取均值减少方差 |
| hint_steps | 1 | 与主 agent 一致（注入 1 次即清除），避免 agent 过度依赖提示 |
| arms | 4 | none/cue/question/repair，其中 cue 只用于分析，不进入 gate 训练 |

**hint_steps=1 的理由**:
- 主 agent (`agent.py`) 的 `inject_mode="in_loop"` 在使用一次后立即清除记忆: `current_retrieved = None`
- Rollout 应与主 agent 行为一致
- 持续注入会让 agent 每一步都参考提示信息，不能反映真实的单次干预效果

**输出格式**:
```json
{
  "cp_0001": {
    "none": [
      {"progress": 0.0, "actions": [...], "observations": [...], "score_before": 10, "score_after": 10, ...},
      // replay 1, 2
    ],
    "repair": [...],
    "question": [...],
    "cue": [...]
  }
}
```

---

### Stage 1.5: LLM Judge 评分

**为什么需要**:
- ScienceWorld 的内置 score 有 **37.5% 评估盲区**
- 不同 arm 产生不同轨迹，但 env score delta 都是 0
- 例: repair arm 让 agent 走对了方向（去厨房找热源），但 10 步内没到达得分点 → env progress = 0
- LLM judge 能识别这种方向性进步

```bash
python experiments/gate/run_llm_judge_scoring.py \
    --rollouts rollouts/scienceworld_rollouts_v2.json \
    --checkpoints checkpoints/scienceworld_checkpoints_midband.json \
    --gold-paths rollouts/scienceworld_gold_paths.json \
    --config config_scienceworld.yaml \
    --output rollouts/scienceworld_rollouts_judged.json
```

**评分方式**: 单轨迹绝对评分（不是 A vs B 对比）
```
输入: task_desc + gold_path + agent_trajectory
输出: SCORE: 0-10, REASONING: ...
归一化: llm_judge_progress = score / 10.0 → [0, 1]
组合: combined_progress = max(env_progress, (1-λ) * env_progress + λ * llm_judge_progress)
推荐: λ = 0.6
```

**评分标准 (0-10)**:
| 分数 | 含义 |
|------|------|
| 0 | 完全无用：卡住、重复失败、输出乱码 |
| 1-2 | 有效动作但方向完全错误 |
| 3-4 | 方向合理但没接近任何 gold path 里程碑 |
| 5-6 | 方向正确（对的房间、对的物品）但没完成关键步骤 |
| 7-8 | 完成了 1 个或多个 gold path 关键步骤 |
| 9-10 | 连续完成多个 gold path 关键步骤 |

**与 env progress 的关系**:

| | Env progress | LLM judge progress |
|---|---|---|
| 范围 | 0-1.0 (61% 是 0) | 0-1.0 |
| 含义 | 分数增量 (score delta) | 轨迹质量 (方向正确性) |
| 粒度 | 粗（只在特定得分点变化） | 细（能区分方向性进步） |

**当前推荐**:
- 统一保留 3 个字段:
  - `progress` (env progress)
  - `llm_judge_progress`
  - `combined_progress`
- Stage 2 训练默认推荐使用:
  - `combined_progress`
- 理由:
  - env progress 是客观分数增量，作为保底信号
  - llm_judge_progress 能补足 score-blind 的方向性进步
  - 纯 env 太稀疏，纯 llm 又会丢掉官方分数增量这一客观信号

---

### Stage 2: Gate 训练

```bash
python experiments/gate/run_gate_training.py \
    --checkpoints checkpoints/scienceworld_checkpoints_midband.json \
    --rollouts rollouts/scienceworld_rollouts_judged.json \
    --output gate_models/sw_gate_v2 \
    --beta 0.3 \
    --bypass-threshold 0.0 \
    --progress-signal combined \
    --run-name sw_gate_v2_
```

#### 2.1 特征提取 (11 维)

```python
extract_features_from_checkpoint(cp) → np.ndarray(11,)
```

| # | 特征名 | 说明 | 类型 |
|---|--------|------|------|
| 0 | failure_type | 失败类型编码 | categorical |
| 1 | task_type | 任务类型编码 | categorical |
| 2 | step_index | 步数 / max_steps | float [0,1] |
| 3 | action_repetition_count | 当前动作连续重复次数 | int |
| 4 | retrieval_rrf_score | top-1 检索 RRF 分数 | float |
| 5 | retrieval_margin | top-1 vs top-2 RRF 差值 | float |
| 6 | memory_entry_count | 可用记忆条数 (per-env) | int |
| 7 | history_token_count | 历史 token 估算 | int |
| 8 | progress_since_last_failure | 上次失败后是否有进展 | binary |
| 9 | failures_in_last_5_steps | 最近 5 步的失败次数 | int |
| 10 | same_failure_recurrence_count | 同类失败重现次数 | int |

#### 2.2 Utility 计算

```
对每个 (cp, arm):
  progress_avg = mean(replays[combined_progress])

  disruption (仅 arm ≠ none):
    对每个 replay_idx:
      disruption_i = max(0, progress_none_i - progress_arm_i)
    disruption_avg = mean(disruption_i)

  utility = progress_avg - β × disruption_avg

oracle_arm = argmax_arm(utility)
```

**公式合理性**:
- `progress_avg`: 干预后 agent 的平均进步质量
- `disruption_avg`: 干预的最坏情况惩罚（arm 比 none 差时）
- `β = 0.3`: 保守惩罚权重，保证干预不会显著损害性能
- LLM judge progress 和 env progress 都在 [0, 1] 范围，β=0.3 通用

#### 2.3 XGBoost 模型

```
输入: [11 checkpoint features + 3 arm one-hot] = 14 维
输出: utility (float)
推理: π(x) = argmax_{k ∈ {none, question, repair}} f(x, k)
```

**参数**: max_depth=4, n_estimators=100, lr=0.1, min_child_weight=3

**bypass 规则**: 已禁用 (bypass_threshold=0.0)。原设 0.1 但 ScienceWorld 的 max RRF 只有 0.033，导致 gate 永远返回 none。

#### 2.4 评估指标

| 指标 | 说明 | 目标 |
|------|------|------|
| heterogeneity_index | 1 - max(oracle_fraction) | > 0.3 |
| gate_accuracy | 预测 arm = oracle arm 的比例 | > 0.5 |
| test_mae | 测试集 utility 预测误差 | < baseline_mae |
| oracle_value | oracle 策略的平均 utility | > always-none |

---

### Stage 3: 在线评估

```bash
python experiments/gate/run_online_eval.py \
    --benchmark scienceworld \
    --gate gate_models/sw_gate_v2 \
    --memory memory/scienceworld_memory_tiered.json \
    --config config_scienceworld.yaml \
    --max-envs 50 --run-name sw_gate_eval_
```

**流程**:
```
Agent 运行中，每次 failure_detector 触发:
  ① extract_features_from_agent_state() → 11 维特征
  ② gate.predict_arm(features, rrf_score) → arm
  ③ 如果 arm ≠ none:
     检索记忆 → 生成注入文本 (get_repair_display())
     注入到下一步 prompt → 使用一次后清除
  ④ 继续 agent 循环
```

---

## 三、关键文件索引

### 核心模块

| 文件 | 职责 |
|------|------|
| `src/gate/checkpoint.py` | CheckpointState 数据结构 + 存储/去重/分割 |
| `src/gate/features.py` | 11 维特征提取 |
| `src/gate/branched_rollout.py` | 分支 rollout + injection 构建 + utility 计算 |
| `src/gate/train_gate.py` | XGBoost gate 训练 + 推理 + bypass |
| `src/gate/progress.py` | 各 benchmark 的 progress 计算 |
| `src/gate/canonicalize.py` | 记忆分级升级 (strategy/tactic/action) |
| `src/gate/llm_judge.py` | LLM-as-judge 单轨迹评分 |
| `src/memory.py` | FailureMemoryEntry + 检索 (BM25+Embedding RRF) |

### 实验脚本

| 文件 | 职责 |
|------|------|
| `experiments/gate/collect_checkpoints.py` | Stage 0: 采集 checkpoint |
| `experiments/gate/run_branched_rollouts.py` | Stage 1: 分支 rollout |
| `experiments/gate/run_llm_judge_scoring.py` | Stage 1.5: LLM judge 批量评分 |
| `experiments/gate/run_gate_training.py` | Stage 2: 训练 + 评估 |
| `experiments/gate/run_online_eval.py` | Stage 3: 在线评估 |

### 数据文件

| 文件 | 内容 |
|------|------|
| `checkpoints/scienceworld_checkpoints_midband.json` | 40 个 midband CP |
| `memory/scienceworld_memory_tiered.json` | 155 条分级记忆 |
| `rollouts/scienceworld_gold_paths.json` | 24 条 gold path |
| `rollouts/scienceworld_rollouts_judged.json` | LLM judge 增强的 rollout |
| `gate_models/sw_gate_v2.pkl` + `.json` | 训练好的 gate 模型 |

---

## 四、适配其他数据集时的修改点

### ALFWorld 适配

| 环节 | 差异 | 说明 |
|------|------|------|
| 记忆提取 | 使用 `src/memory_extractor.py` | 提取逻辑类似，可复用 tiered prompt |
| 记忆文件 | `memory/alfworld_memory_canonicalized.json` | 目前只有 repair_text，需升级 |
| Progress | milestone-based (`_count_alfworld_milestones`) | 粒度比 SW 细，可能不需要 LLM judge |
| Gold path | 不可用 | ALFWorld 没有 gold path API |
| LLM judge | 可选 | ALFWorld milestone scorer 已覆盖主要评估盲区 |
| 环境重放 | 需创建新 env 实例 | ALFWorld env 是顺序的 (env_idx 次 reset) |

### WebShop 适配

| 环节 | 差异 | 说明 |
|------|------|------|
| 记忆提取 | 使用 `src/webshop_memory_extractor.py` | |
| Progress | reward delta | WebShop 给连续 reward，粒度较细 |
| Gold path | 不可用 | |
| LLM judge | 可能不需要 | reward 本身就是连续的 |
| 环境重放 | 支持 session_idx 重置 | 比 ALFWorld 方便 |

### 通用修改清单

1. **记忆提取**: 修改对应 benchmark 的 `*_memory_extractor.py`，添加分级输出
2. **记忆文件**: 运行 `canonicalize_store_tiered()` 或直接提取分级
3. **Gold path**: 如果 benchmark 提供，提取并保存
4. **LLM judge**: 如果内置 score 有评估盲区，实现对应的 judge prompt
5. **Checkpoint 采样**: 根据 benchmark 的 score 分布调整 midband 范围
6. **Progress 函数**: 在 `progress.py` 中添加或修改 benchmark 分支

---

## 五、已知问题与修复记录

| # | 问题 | 状态 | 修复 |
|---|------|------|------|
| BUG-1 | max_steps=50 (应为 100) | ✅ | config 改为 100 |
| BUG-2 | bypass_threshold=0.1 > max_rrf=0.033 | ✅ | 改为 0.0 (禁用) |
| BUG-3 | 检索特征全零 | ✅ | backfill 脚本 |
| BUG-4 | memory_entry_count 用了全局计数 | ✅ | backfill 为 per-env |
| MEMORY-1 | 50% 记忆检索不匹配 | ✅ | 分级 repair |
| MEMORY-2 | 记忆只有战术级 | ✅ | strategy/tactic/action 三级 |
| EVAL-1 | 37.5% 评估盲区 | ✅ | LLM-as-judge |
| EVAL-2 | 10 步 rollout 太短 | ✅ | 增加到 15 步 |
| HINT-1 | hint_steps=3 与主 agent 不一致 | ✅ | 改为 1 |
| SIGNAL-1 | 70% oracle=none | ⏳ | 分级修复后预期改善 |
| SIGNAL-3 | 13x 过拟合 (40 CP) | ⬜ | 需更多数据或正则化 |

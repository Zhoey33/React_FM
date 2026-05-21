<!-- 本文件汇总 ScienceWorld 工程修正与日志补强阶段已实现工具的用途、命令和结果解读。 -->

# ScienceWorld 工程工具使用说明

> 目标: 让 ScienceWorld 结果可重算、可审计、可标注，不再依赖旧 summary 或人工翻轨迹。

## 1. Raw 指标重算

用途:

- 从 result JSON 的 `episodes[]` 逐条重算正式指标。
- 避免旧 `summary.avg_score` 混入 normalized / clamped 口径。
- 当前正式口径: `avg_score == avg_raw_score == mean(episodes[].score)`。

命令:

```bash
python analysis/scienceworld_metrics_audit.py results/.../sw_result.json
```

多个文件:

```bash
python analysis/scienceworld_metrics_audit.py results/.../*.json
```

重点看:

- `avg_raw_score`: 正式 raw score。
- `avg_score`: 兼容字段，也表示 raw score。
- `success_rate`: 成功率。
- `score_mismatch`: 旧 summary 是否和重算 raw score 不一致。
- `by_task_type`: 分 task 的 success rate。

## 2. 统一 Summary 与协议字段

用途:

- `src/scienceworld_reporting.py` 统一所有 ScienceWorld runner 的 summary。
- 新结果会写入 protocol 元数据，便于确认 split/task/variation/step limit。

新 summary 重点字段:

```json
{
  "benchmark": "scienceworld",
  "total_envs": 111,
  "success_rate": 0.4144,
  "avg_raw_score": 55.2432,
  "avg_score": 55.2432,
  "avg_steps_per_episode": 74.1351,
  "agent_tokens": 21479157,
  "judge_tokens": 0,
  "extractor_tokens": 412794,
  "reflection_tokens": 0,
  "protocol": {
    "split": "test",
    "tasks": ["..."],
    "max_variations": 10,
    "step_limit": 100,
    "test_time_writable": true,
    "memory_scope": "task_type"
  }
}
```

注意:

- 正式表只用 raw score。
- 不再输出 `avg_normalized_score` / `avg_clamped_score`。
- 新实验每个 episode 应包含 `variation_idx`。

## 3. Failure Event JSONL

用途:

- 从 ScienceWorld episode 的 `steps[]` 抽取 detector 触发事件。
- 支撑 recovered@k、memory 注入效果、post-injection correction 等后续分析。

何时生成:

- `experiments/run_scienceworld.py` 每个 epoch 结束时自动写:

```text
results/.../{run_name}{epoch}_failure_events.jsonl
```

每行代表一次 detector 触发，核心字段:

```json
{
  "episode_id": "sw_007",
  "env_idx": 7,
  "task_type": "melt",
  "variation_idx": 3,
  "step": 18,
  "failure_type": "precondition_blocked",
  "failed_action": "move to workshop",
  "failure_observation": "The door is not open.",
  "memory_mode": "in_loop",
  "retrieval_attempted": true,
  "retrieval_hit": true,
  "retrieved_memory_ids": ["..."],
  "injected_memory_text": "...",
  "next_action": "open door",
  "score_before_failure": 10.0,
  "score_after_1_step": 10.0,
  "score_after_2_steps": 18.0,
  "score_after_3_steps": 16.0,
  "recovered_within_1_step": false,
  "recovered_within_3_steps": true
}
```

定义:

- `recovered_within_1_step`: failure 后第 1 步 raw score 是否上升。
- `recovered_within_3_steps`: failure 后 1-3 步窗口内任意一步 raw score 是否上升。

## 4. Recovered@1 / Recovered@3 统计

用途:

- 汇总 failure-event JSONL，量化 detector 触发后是否带来过程恢复。
- 支持整体统计和分组统计。

命令:

```bash
python analysis/scienceworld_recovery_stats.py \
  results/.../*_failure_events.jsonl
```

输出包含:

- `overall`
- `by_task_type`
- `by_failure_type`
- `by_memory_mode`
- `by_retrieval_hit`

每组重点字段:

```json
{
  "total_events": 100,
  "recovered_at_1": 18,
  "recovered_at_3": 34,
  "recovered_at_1_rate": 0.18,
  "recovered_at_3_rate": 0.34,
  "mean_score_delta_at_1": 0.7,
  "mean_score_delta_at_3": 2.4
}
```

解释:

- `recovered_at_3` 使用 1-3 步窗口内最大 score delta。
- `mean_score_delta_at_3` 也是窗口内最大 raw score delta 的均值。

## 5. Detector Quality 标注与统计

用途:

- 评估 detector 本身是否可靠。
- precision 需要 detector 触发样本。
- recall 需要 detector 未触发样本，所以输入必须是完整 result JSON，不是 failure-event JSONL。

### 5.1 生成标注模板

命令:

```bash
python analysis/scienceworld_detector_quality.py make-template \
  results/.../sw_result.json \
  --output analysis/detector_quality_annotations.jsonl \
  --max-samples 200 \
  --positive-ratio 0.5 \
  --seed 0
```

说明:

- 从 `episodes[].steps[]` 抽样。
- 默认跳过 `is_think=true` 的思考步。
- `--positive-ratio 0.5` 表示约一半 detector-positive、一半 detector-negative。
- 如需包含 think 步，加 `--include-think`。

模板自动字段:

```json
{
  "sample_id": "sw_detector_0001",
  "source_file": "results/.../sw_result.json",
  "episode_id": "sw_001",
  "env_idx": 1,
  "task_type": "melt",
  "variation_idx": 3,
  "step": 18,
  "action": "move to workshop",
  "observation": "The door is not open.",
  "score_before_action": 0.0,
  "score_after_action": 0.0,
  "detector_prediction": true,
  "detector_failure_type": "precondition_blocked",
  "gold_is_failure": null,
  "gold_failure_type": "",
  "gold_needs_repair": null,
  "annotation_notes": ""
}
```

人工填写字段:

- `gold_is_failure`: 这一步是否真实失败。
- `gold_failure_type`: 人工判断的失败类型。
- `gold_needs_repair`: 是否需要修复/记忆介入。
- `annotation_notes`: 简短备注。

### 5.2 统计标注结果

命令:

```bash
python analysis/scienceworld_detector_quality.py summarize \
  analysis/detector_quality_annotations.jsonl \
  --output analysis/detector_quality_summary.json
```

如果还有未标注行，默认报错。想临时跳过:

```bash
python analysis/scienceworld_detector_quality.py summarize \
  analysis/detector_quality_annotations.jsonl \
  --output analysis/detector_quality_summary.json \
  --allow-unlabeled
```

输出重点:

```json
{
  "overall": {
    "total_samples": 200,
    "tp": 80,
    "fp": 20,
    "fn": 30,
    "tn": 70,
    "precision": 0.8,
    "recall": 0.7273,
    "f1": 0.7619,
    "type_accuracy": 0.75,
    "false_positive_rate": 0.2222,
    "false_negative_rate": 0.2727
  },
  "by_task_type": {},
  "by_detector_failure_type": {},
  "by_gold_failure_type": {}
}
```

定义:

- `precision = TP / (TP + FP)`: detector 触发时有多少是真的 failure。
- `recall = TP / (TP + FN)`: 真实 failure 中有多少被 detector 抓到。
- `type_accuracy`: TP 中 detector type 与 gold type 完全一致的比例。
- `false_positive_rate = FP / (FP + TN)`。
- `false_negative_rate = FN / (FN + TP)`。

## 6. 建议使用顺序

1. 对旧结果先跑 raw 指标重算:

```bash
python analysis/scienceworld_metrics_audit.py results/.../sw_result.json
```

2. 用新 runner 重跑或 smoke run，确认会生成 result JSON 和 failure-event JSONL。

3. 跑 recovered@k:

```bash
python analysis/scienceworld_recovery_stats.py results/.../*_failure_events.jsonl
```

4. 从完整 result JSON 抽 detector 标注模板:

```bash
python analysis/scienceworld_detector_quality.py make-template results/.../sw_result.json \
  --output analysis/detector_quality_annotations.jsonl \
  --max-samples 200 \
  --positive-ratio 0.5 \
  --seed 0
```

5. 人工标注后统计 detector quality:

```bash
python analysis/scienceworld_detector_quality.py summarize \
  analysis/detector_quality_annotations.jsonl \
  --output analysis/detector_quality_summary.json
```

## 7. 当前完成状态

已完成:

- raw 指标审计脚本。
- 统一 ScienceWorld summary helper。
- failure-event JSONL 生成。
- recovered@1 / recovered@3 统计脚本。
- detector quality 标注模板与统计脚本。

下一步:

- memory generation quality 标注/统计脚本。
- post-injection correction 标注/统计脚本。
- context construction 优化。

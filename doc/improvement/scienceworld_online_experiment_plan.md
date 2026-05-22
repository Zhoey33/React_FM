<!-- ScienceWorld React-FM online experiment plan for stages 4-7. -->

# ScienceWorld React-FM Online 实验计划

> 目标: 按 `doc/improvement/improvement.md` 的第 4-7 项，先跑通 React-FM Online 主链路，再产出 detector、memory、post-injection correction 三类证据。  
> 口径: 正式 online 从空 memory 开始，在 test stream 中写入并检索 memory；旧 memory 只允许用于 smoke/debug，不进入 online 主结果。

## 0. 固定设置

正式 React-FM Online 使用:

- split: `test`
- task: 12 个核心 ScienceWorld task
- variations: 每个 task 前 `min(10, available)` 个 variation，共 111 episodes
- step limit: `100`
- inject mode: `in_loop`
- memory scope: `task_type`
- retrieval: `hybrid`, candidate top-5，gated top-1，pilot 阶段 `min_score=0.0`
- detector: 规则优先；规则未命中再用 judge LLM 隐式检测
- prompt: 最近 10 步 history；成功检索 memory 后只在下一步注入 failure signal + repair memory
- 禁止把 API key 写入 repo；只用 `SILICONFLOW_API_KEY` 环境变量

参数口径:

- `top_k=1` 固定为最终注入数量，保证每次注入只对应一条 memory，便于解释下一步动作是否被这条 memory 修正。
- `retrieval_candidate_k=5` 用于内部候选召回；hybrid 只负责召回，不直接决定注入。
- `min_score=0.0` 只作为兼容字段和观测字段；pilot 已显示 RRF score 在小候选池里区分度不足，不作为主阈值。
- 正式注入由 deterministic gated retrieval 决定: failure type compatible filter + safety gate + relevance score threshold。
- retrieval v1 设计见 `doc/improvement/scienceworld_retrieval_v1_design.md`。

核心 task list:

```bash
TASKS="boil melt use-thermometer power-component test-conductivity \
test-conductivity-of-unknown-substances grow-plant chemistry-mix \
lifespan-longest-lived-then-shortest-lived inclined-plane-friction-unnamed-surfaces \
mendelian-genetics-known-plant mendelian-genetics-unknown-plant"
```

## 1. 预检查

先确认代码和环境没有明显问题:

```bash
python -m pytest tests
python -m compileall -q src experiments analysis
git diff --check
```

验收:

- tests 全部通过。
- `config_scienceworld.yaml` 中 `judge.enable_implicit_failures: true`。
- `agent.max_memory_inject: 1`、`agent.prompt_history_window: 10`、`memory.min_score: 0.0`。

## 2. Online 小 pilot

目的: 不直接烧 111 episodes，先确认“检测 -> 生成 memory -> 后续检索 -> 注入下一步 prompt -> failure event 记录”全链路有效。

命令:

```bash
SILICONFLOW_API_KEY=$SILICONFLOW_API_KEY \
python experiments/run_scienceworld.py \
  --config config_scienceworld.yaml \
  --run-name sw_online_pilot_ \
  --tasks melt boil use-thermometer \
  --max-variations 3 \
  --max-envs 9 \
  --step-limit 60 \
  --split test \
  --inject-mode in_loop \
  --retrieval-mode hybrid \
  --seed 42
```

pilot 后立刻检查:

```bash
python analysis/scienceworld_metrics_audit.py results/<timestamp>/sw_online_pilot_1.json \
  > results/<timestamp>/metrics_audit.json

python analysis/scienceworld_recovery_stats.py results/<timestamp>/sw_online_pilot_1_failure_events.jsonl \
  > results/<timestamp>/recovery_stats.json
```

通过标准:

- result JSON 中有 `judge_tokens`、`extractor_tokens`、`failure_events` 相关字段。
- failure-event JSONL 中有 `retrieval_candidate_count`、`retrieved_memory_scores`。
- memory store 中至少生成若干条 `failure_recovery` memory。
- 如果 retrieval hit 全为 0，不进入正式 run，先看 detector 是否触发太少或 memory 是否没有写入。
- 如果 retrieval hit 有但明显乱注入，先抽样做 post-injection annotation，再考虑调 retrieval query 或阈值。

## 3. Retrieval calibration

pilot 跑完后，先不要直接进入 111 episodes 正式 run。需要抽查发生注入的事件，确认 deterministic gated retrieval 是否减少无关注入。

先生成 post-injection 标注模板:

```bash
python analysis/scienceworld_post_injection_correction.py make-template \
  results/<timestamp>/sw_online_pilot_1_failure_events.jsonl \
  --output results/<timestamp>/pilot_post_injection_annotations.jsonl \
  --max-samples 30 \
  --seed 42
```

人工标注时重点看:

- retrieved memory 是否和当前 failure 相关。
- `next_action` 是否使用 memory。
- `next_action` 是否修正 failure。
- 注入是否 harmful。
- 对应 `retrieval_candidate_relevance_scores` 和 `retrieval_rejection_reason`。

决策规则:

- 如果 `syntax_or_parse` 仍大量注入重复非法动作，继续收紧 safety gate。
- 如果 `precondition_blocked` 门类正例被过度拒绝，降低 `memory.relevance_score_threshold` 或放宽门类规则。
- 如果候选多数被 `below_relevance_threshold` 拒绝但人工看是相关的，重调 relevance score 权重。
- 如果注入事件总体相关且 harmful 很少，可以进入正式 run，并在论文中说明使用 deterministic gated retrieval。

这一步的目标不是追求最高分，而是避免把明显不相关的 memory 注入正式 online 实验。

## 4. 正式 React-FM Online run

正式 run 从空 online memory 开始，不使用 `--resume-memory`。

```bash
SILICONFLOW_API_KEY=$SILICONFLOW_API_KEY \
python experiments/run_scienceworld.py \
  --config config_scienceworld.yaml \
  --run-name sw_online_final_ \
  --tasks $TASKS \
  --max-variations 10 \
  --step-limit 100 \
  --split test \
  --inject-mode in_loop \
  --retrieval-mode hybrid \
  --seed 42
```

保留产物:

- `results/<timestamp>/sw_online_final_1.json`
- `results/<timestamp>/sw_online_final_1_failure_events.jsonl`
- `logs/<timestamp>/sw_online_final.log`
- `memory_store/<timestamp>/sw_epoch1.json`

验收:

- episode 数应为 111，除非 ScienceWorld API 返回数量变化；如变化，记录实际 task/variation 列表。
- 不允许 skipped episode。
- 主分数只使用 raw score 重算值，不使用旧 summary clamp 口径。

## 5. 第 4 项: 主实验指标

先给 React-FM Online 单独产出可追溯指标:

```bash
python analysis/scienceworld_metrics_audit.py results/<timestamp>/sw_online_final_1.json \
  > results/<timestamp>/metrics_audit.json
```

需要记录到实验表的字段:

- total episodes
- success rate
- avg raw score
- avg steps
- total tokens / tokens per episode
- agent / judge / extractor token 拆分
- memories stored / retrieved

注意:

- 这一步只完成 React-FM Online 的主结果。
- 和 ReAct / Offline / Reflexion / ExpeL 的最终主表合并，等对应方法按同一协议跑完后再做。

## 6. 第 5 项: Detector quality

从正式 result JSON 生成 100-200 条 step 标注模板:

```bash
python analysis/scienceworld_detector_quality.py make-template \
  results/<timestamp>/sw_online_final_1.json \
  --output results/<timestamp>/detector_quality_annotations.jsonl \
  --max-samples 200 \
  --positive-ratio 0.5 \
  --seed 42
```

人工填写:

- `gold_is_failure`
- `gold_failure_type`
- `gold_needs_repair`
- `annotation_notes`

标完后汇总:

```bash
python analysis/scienceworld_detector_quality.py summarize \
  results/<timestamp>/detector_quality_annotations.jsonl \
  --output results/<timestamp>/detector_quality_summary.json
```

关键判断:

- precision 太低: detector 误触发，会污染 retrieval 和 memory。
- recall 太低: 真失败没有被抓到，online 记忆链路很难生效。
- judge 隐式错误可以单独看 `detector_source=judge` 的样本质量。

## 7. 第 6 项: Memory generation quality

从正式 online memory store 抽 50-100 条 memory:

```bash
python analysis/scienceworld_memory_quality.py make-template \
  memory_store/<timestamp>/sw_epoch1.json \
  --result-json results/<timestamp>/sw_online_final_1.json \
  --output results/<timestamp>/memory_quality_annotations.jsonl \
  --max-samples 100 \
  --seed 42
```

人工填写:

- `quality_label`
- `failure_action_accurate`
- `failure_observation_has_evidence`
- `repair_strategy_reasonable`
- `repair_action_executable`
- `overly_state_bound`
- `annotation_notes`

标完后汇总:

```bash
python analysis/scienceworld_memory_quality.py summarize \
  results/<timestamp>/memory_quality_annotations.jsonl \
  --output results/<timestamp>/memory_quality_summary.json
```

关键判断:

- `usable_rate` 是否足够高。
- `repair_action_executable_rate` 是否足够高。
- `overly_state_bound_rate` 是否过高；如果过高，说明 memory 太依赖单个 variation。

## 8. 第 7 项: Post-injection correction

从正式 failure-event JSONL 中抽“真的发生注入”的事件:

```bash
python analysis/scienceworld_post_injection_correction.py make-template \
  results/<timestamp>/sw_online_final_1_failure_events.jsonl \
  --output results/<timestamp>/post_injection_correction_annotations.jsonl \
  --max-samples 100 \
  --seed 42
```

人工填写:

- `gold_corrected_next_action`
- `gold_used_memory`
- `gold_injection_harmful`
- `annotation_notes`

标完后汇总:

```bash
python analysis/scienceworld_post_injection_correction.py summarize \
  results/<timestamp>/post_injection_correction_annotations.jsonl \
  --output results/<timestamp>/post_injection_correction_summary.json
```

同时自动统计 recovered@k:

```bash
python analysis/scienceworld_recovery_stats.py \
  results/<timestamp>/sw_online_final_1_failure_events.jsonl \
  > results/<timestamp>/recovery_stats.json
```

关键判断:

- `corrected_next_action_rate` 高: memory 注入确实修正下一步。
- `used_memory_rate` 高但 correction 低: memory 可能相关但 prompt 或 repair action 不够好。
- `harmful_rate` 高: 需要调 retrieval 阈值或 prompt applicability 约束。
- recovered@1/@3 与人工 correction 不一致时，优先相信人工 correction 解释过程，raw score 作为辅助信号。

## 9. 进入下一轮前的决策

可以继续扩大/写论文证据的条件:

- online run 完整跑完 111 episodes。
- detector precision 不低，且 judge 隐式错误有可解释样本。
- memory `usable_rate` 不低，`repair_action_executable_rate` 可接受。
- post-injection 中有一批明确 `gold_corrected_next_action=true` 或 `gold_used_memory=true` 的案例。

需要先修再重跑的条件:

- retrieval hit 主要是低相关 memory。
- post-injection 大量 `gold_injection_harmful=true`。
- memory 多数 `invalid` 或 `duplicate_or_redundant`。
- detector false positive 明显污染后续 memory extraction。

本轮建议顺序:

1. 先跑第 2 节 pilot。
2. 按第 3 节做 retrieval calibration，决定是否设置 `memory.min_score`。
3. calibration 合格后跑第 4 节正式 online。
4. 立刻生成第 5-8 节所有自动 summary/template。
5. 先人工标 post-injection 20-30 条，再标 detector 和 memory 质量，补齐论文证据链。

<!-- 本文件审查 ScienceWorld React-FM 在正式重跑前的四个关键实现面: detector、memory extraction、retrieval、prompt construction。 -->

# ScienceWorld React-FM 实现审计

> 结论: 默认 `in_loop + failure_recovery + task_type memory` 主路径基本连通，但正式实验前仍建议先补少量修正和 sanity tests。当前最大风险不是 runner 崩溃，而是 detector 覆盖不足、检索无相关性阈值、prompt 上下文无限增长会影响结果解释。

## 1. 错误检测器

代码位置:

- `src/scienceworld_failure_detector.py`
- `experiments/run_scienceworld.py`

当前实现:

- 规则检测 unknown action、empty observation、action loop、若干显式失败短语。
- LLM judge 隐性错误检测已实现并默认启用；可通过 `--no-enable-implicit-failures` 或 `judge.enable_implicit_failures: false` 显式关闭。
- judge prompt 使用 task type、task goal、最近 history、当前 action/observation、score before/after/delta，并要求输出 strict JSON。
- judge failure type 限定为 `implicit_no_progress / redundant_repeat / irrelevant_action / premature_action`。
- detector 只在 `enable_memory and inject_mode in ("in_loop", "none")` 时运行；baseline 和 `episode` injection 不运行 detector。
- result step 会记录 `detector_source`、`failure_reason`、`failure_confidence`。

主要风险:

- 若关闭 judge 回到 rule-only，仍无法捕捉大量“合法但无进展”的 ScienceWorld 失败，例如动作执行了但没有推进 score 或目标。
- 默认启用 judge 后会增加 judge token 成本，且 detector quality 必须单独标注评估。
- `episode` injection 模式不检测 failure，后续 extractor 也拿不到 detected failures，因此不会产生新的 failure-recovery memory。

建议:

- 正式主实验默认应标记为 `rule + LLM judge detector`。
- 若用 `--no-enable-implicit-failures` 跑纯规则 detector，文档明确写 `rule-based detector`。
- detector quality 标注需要分别看 `detector_source=rule` 和 `detector_source=judge`。

## 2. 记忆提取

代码位置:

- `src/scienceworld_memory_extractor.py`
- `experiments/run_scienceworld.py`

当前实现:

- episode 后基于 `steps[]` 中 detector 触发的失败构造 `detected_failures`。
- extractor prompt 要求 LLM 只从这些失败步骤中抽取后续真实发生过的修复动作，并输出 `confidence_score`。
- `_validate_recovery()` 会验证 `solution_action` 和 `repair_action` 必须出现在 failure 之后。
- 存储字段包含 `failure_step / failure_type / detector_source / score_before_action / score_after_action / score_delta / source_episode_success / source_episode_score / confidence_score`，以及 `failure_action / failure_observation / solution_action / repair_strategy / repair_tactic / repair_action`。
- ScienceWorld 只保留 `failure_recovery` memory format；`success_trajectory / reflexion_reflection` 不再作为 ScienceWorld runner 分支。

主要风险:

- extractor 依赖 detector: detector 漏掉的失败永远不会进入 memory。
- judge 检出的隐性错误会进入 retrieval 和 memory extraction；旧的 `unproductive` 类型仍会被过滤。
- `confidence_score` 是 extractor 自评，只能作为分析字段，不能直接当作真实质量标签。

建议:

- 默认主实验继续使用 `failure_recovery`。
- 用 memory quality 标注检查 `confidence_score` 和人工质量标签是否一致。
- 如果要评估 implicit failure，单独按 `detector_source=judge` 和 `failure_type` 分组看 memory 质量。

## 3. 记忆检索

代码位置:

- `src/memory.py`
- `experiments/run_scienceworld.py`

当前实现:

- ScienceWorld runner 固定 `FailureMemoryStore(scope="task_type")`。
- 每条 memory 存入同 task type bucket。
- 检索 query 是 `failed_action | failure_observation`。
- 默认 `retrieval_mode="hybrid"`，BM25 与 embedding 排名通过 RRF 融合。
- 主实验固定 `in_loop`，默认只注入 top-1 memory。
- `min_score=0.0` 为默认安全值，表示记录 retrieval score 但不强筛。
- failure event 会记录 `retrieved_memory_scores / retrieval_candidate_count / retrieval_top_k / retrieval_min_score / retrieval_mode`。

主要风险:

- `min_score=0.0` 时同 task type bucket 非空仍会返回 top-1；弱相关注入需要靠 pilot 后设置阈值控制。
- task type 粒度仍然较粗，同一 task type 的不同 variation 可能共享过多局部经验。
- 检索统计里的 `retrieval_hit` 当前更接近“返回了 memory”，不等于“相关 memory 命中”。

建议:

- 正式实验报告中区分 `retrieval_hit` 和人工标注的 memory relevance / post-injection correction。
- 在小规模 pilot 上扫 `retrieval_mode` 与 `min_score`，再决定主实验是否保留 `min_score=0.0`。
- `episode` injection 只作为 legacy/ablation，不作为 React-FM 主实验协议。

## 4. Prompt 拼接

代码位置:

- `prompts/scienceworld_prompts.py`
- `experiments/run_scienceworld.py`

当前实现:

- baseline prompt: few-shot + task observation + full history。
- React-FM prompt: few-shot + memory section + task observation + full history。
- in-loop memory 在 detector 触发后只注入下一步 prompt，随后 `current_retrieved=None` 清空。
- memory style 支持 `original / factual / reflexion / hint`。

主要风险:

- history 不截断，100 step ScienceWorld episode 会把完整 observation 全塞进 prompt，成本和上下文噪声都可能很高。
- memory section 位于 task observation 之前，且没有显式说明“这是针对上一条失败 action 的建议”。
- 没有 applicability decision，模型可能盲用或忽略 memory。
- `hint` style 对 tiered memory 取的是第一行 `[Strategy]`，不是最直接的 `[Next action]`。

建议:

- 正式实验前至少做 prompt snapshot test，固定 baseline / in-loop / episode 三种 prompt 结构。
- 对正式主实验使用一个固定 prompt policy: `task goal + recent history + failure signal + retrieved memory + action cue`。
- 若不马上重构，至少在论文和日志中说明当前 memory 是 one-step injection，而不是持续 episode-level guidance。

## 5. 实验前必须确认

建议在正式重跑前完成:

1. 明确 detector 口径: 默认 rule + LLM judge；如需 rule-only 对照，使用关闭开关单独跑。
2. 保持 ScienceWorld memory format 为 `failure_recovery`，不混入其它 benchmark 的 memory ablation 分支。
3. 增加 prompt snapshot tests，避免后续改 prompt 时结果不可比。
4. 用已实现的三类质量脚本做 pilot:
   - detector quality
   - memory generation quality
   - post-injection correction
5. 对 retrieval 做小样本人工核查，确认 top-k memory 不是大量弱相关注入。

## 6. 当前可用性判断

可以作为正式重跑候选:

- 默认 `rule + LLM judge detector`
- `--inject-mode in_loop`
- `--memory-format failure_recovery`
- `--retrieval-mode hybrid`
- `scope=task_type`

暂不建议直接进正式主表:

- `--inject-mode episode` 且希望在线继续学习。
- 未经过 detector quality 标注就声称 LLM judge detector 可靠。

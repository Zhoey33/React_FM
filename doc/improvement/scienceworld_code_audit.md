<!-- 本文件审计 ScienceWorld ReAct / Reflexion / ExpeL 实现与论文机制、项目正式协议的一致性。 -->

# ScienceWorld 代码对齐审计

> 范围: `experiments/run_scienceworld.py`、`run_reflexion_scienceworld.py`、`run_expel_scienceworld.py`。
> 结论先行: 不把旧 Step 4 结果硬塞进论文主表。ReAct 可在补日志后重跑；Reflexion 需要改成 paper-aligned solved@trial runner；ExpeL 当前实现偏离原论文，必须重设计后重跑，旧结果只作诊断。

## 0. 执行决策

旧结果用途降级:

- 可用于定位 score、skip、token、prompt 等问题。
- 不作为最终论文主表数字。
- 不把不合理实现包装成“适配差异”；凡影响 baseline 定义的，按漏洞处理并重跑。

重跑原则:

- 先修代码和日志，再跑正式实验。
- 所有方法统一使用 `scienceworld_protocol_audit.md` 中的 12 task / 111 test episode / raw score 口径。
- 每个正式结果必须能从 JSON 重算 split、task、variation、score、success、steps、tokens、是否 test-time writable。

## 1. 总表

| Method | Paper expectation | Current implementation | Type | Severity | Action |
|---|---|---|---|---|---|
| ReAct | few-shot + 当前轨迹；无 memory、无 reflection、无跨 episode 学习 | `--baseline` 时关闭 detector、memory、extractor，只用 baseline prompt | 适配差异 | Low | 补协议日志后重跑正式 baseline |
| ReAct / all | score 使用环境原始值，负分保留 | 当前代码重算逻辑正确；旧 ExpA summary 曾用 clamped score | 统计/报告问题 | High | 所有表格从 `episodes[].score` 重算 |
| ReAct / all | 每个 episode 可复现到 task/variation | 结果保存 `task_type`，但旧 JSON 未保存 `variation_idx` | 统计/报告问题 | Medium | 新实验必须补 `variation_idx` |
| Reflexion | 失败后生成 self-reflection，下次尝试注入 memory | pass0 失败后生成 reflection；pass1 同 episode 注入最近 memory | 适配差异 | Low | 重写 runner 为 solved@trial |
| Reflexion | 报告真实试验成本 | reflection 生成使用同一 LLM tracker，但未计入 pass summary token | 实现漏洞 | High | 修 token accounting 后重跑 |
| Reflexion | 同一 task 在失败后反思、reset、重试，直到成功或达到 max trials；成功后不再重跑 | 当前 runner 为所有 episode 都保存 raw pass1；Step 4 又用 adjusted 选择 | 实现漏洞 | High | 删除 raw-pass 主口径，重跑 solved@trial |
| ExpeL | 从训练经验中抽取 insights，再用于 unseen evaluation tasks | Epoch1 在同一 test 上收集，Epoch2 在同一 test 上注入 | 实验口径漏洞 | Critical | 重设计 train/dev -> test |
| ExpeL | evaluation 完整单次评估目标集 | Epoch2 跳过 Epoch1 已成功样本，记为 success 且 token=0 | 实现漏洞 | Critical | 禁止 skip，完整重跑 |
| ExpeL | insight + successful trajectory retrieval | 当前只有 rule injection，无 successful trajectory pool | 实现漏洞 | Critical | 按原论文重实现或改名为 diagnostic |

## 2. ReAct 审计

可用结论:

- `baseline_mode=True` 时，`self.detector=None`，`enable_memory=False`，不会触发 memory retrieval / failure detection / extractor。
- baseline prompt 只包含 few-shot、task observation 和当前 history，没有 React-FM memory section。
- action post-processing 只抽取模型文本中的可执行动作；没有把错误动作改成正确动作。
- success 判定为 `score >= 100`，与 ScienceWorld 满分完成口径一致。

需要修正或记录:

- 正式协议不能使用默认 `DEFAULT_EVAL_TASKS`、`max_variations=5`；必须显式传 12 task、`--max-variations 10`、`--step-limit 100`。
- 旧结果没有 `variation_idx`，只能按 task count 和日志间接复现。
- ScienceWorld runner 已收敛为 `failure_recovery` memory 主链路；其它 benchmark 的 memory-format ablation 不进入 ScienceWorld 正式协议。

判断: ReAct ScienceWorld baseline 机制基本可信，但旧结果缺少正式复现字段；正式主表仍建议补日志后重跑。

## 3. Reflexion 审计

可用结论:

- 实现是 ScienceWorld 在线两遍适配: pass0 跑一次；失败则生成 reflection；pass1 reset 到同一 episode 并注入 reflection。
- repeated action 直接触发 exhaustion，这是 Reflexion 风格适配，不是 ReAct 污染。
- Step 4 日志确认正式运行使用了 111 episodes、12 tasks、`step_limit=100`。

主要问题:

- 代码默认值仍是 `max_envs=50`、`step_limit=30`、`max_variations=5`；复现实验必须显式传协议参数。
- 83 次 reflection 生成未进入 `pass0/pass1` token summary；当前 token 是真实成本下界。
- raw pass1 重跑所有 episode，会让 pass0 已成功样本退化；这不符合 Reflexion 原论文“任务失败才进入下一 trial、成功则保留 solved 状态”的评估逻辑。
- Step 4 的 `P1 Adj` 更接近 `solved within <=2 trials`，但它是后处理 cumulative 口径，不是 runner 原生 pass1 输出。
- 异常路径给 `score=0`，而 ReAct/ExpeL 异常路径给 `-100`；虽未见 Step 4 触发，但口径应统一。

Step 4 可报告口径:

- `Reflexion P0`: 原生 pass0。
- `Reflexion P1 Raw`: 原生 pass1，仅用于诊断第二遍是否退化，不应作为 paper-aligned 主结果。
- `Reflexion solved@2` 或 `P1 Adj`: pass0 成功沿用 pass0，否则用 pass1；这是接近原论文的 cumulative success after 2 trials 口径，必须标注 max trials=2。
- 成本: 至少报告 `pass0 + pass1 raw`，另注明 reflection generation tokens 未计入。

判断: 当前 Reflexion 旧结果只作诊断。正式 baseline 应重写为 `solved@trial`，成功 episode 不再进入后续 trial，并把 reflection token 计入成本后重跑。

## 4. ExpeL 审计

可用结论:

- Epoch1 先跑 ReAct 收集轨迹；epoch 后用 extractor 生成 insights。
- Epoch2 在 episode start 注入 insights，形式上符合 ExpeL “经验归纳后再使用”的机制。
- Step 4 产物有 200 条 insights，69 次 retrieval，核心任务计数匹配 111 episodes。

原论文/官方实现要点:

- ExpeL 分三段: experience gathering、insight extraction、evaluation。
- experience gathering 阶段用 Reflexion 式多 trial 训练任务收集 success / failure trajectories；失败会 self-reflect 后重试，成功才换下一个训练任务。
- insight extraction 阶段不仅比较 failure/success pairs，还会用 successful trajectory chunks 总结 good practices；insight set 通过 `ADD / EDIT / UPVOTE / DOWNVOTE` 和 importance count 迭代维护。
- evaluation 阶段面对 unseen tasks，单次尝试；prompt 同时使用 extracted insights 和从 training experience pool 里按 task similarity 检索出的 successful trajectories 作为 few-shot examples。

主要问题:

- Epoch1 和 Epoch2 都在 `test` split 上；这不是严格 train/eval 分离的 ExpeL baseline，而是 test-time adaptation。
- Epoch2 跳过 Epoch1 已成功的 42 个 episode，写入 `success=True, score=100, total_tokens=0`；因此 `sw_step4_expD_expel_test_2.json` 是 effective result，不是完整第二轮运行结果。
- Epoch1 不是 Reflexion 式 experience gathering: 每个 episode 只跑一遍 ReAct，没有失败后 reflection / retry，也没有把同一训练任务的多 trial 经验放入 pool。
- 当前只保存 extracted rules；没有保存可检索的 successful trajectory pool，因此 evaluation 没有原论文的 task-similarity few-shot retrieval。
- insight extraction 是“每个 failure 随机配一个 same-task success，或 failure-only 抽取 JSON rules”；没有 success chunks，也没有 `ADD / EDIT / UPVOTE / DOWNVOTE` 和 importance count。
- `InsightMemoryStore.retrieve()` 在 task-specific insights 不足时会加入其他 task insights，但实际排序只是 insertion order 的前 `k` 条，不是基于 evaluation task embedding 的 kNN 检索。
- 代码默认值仍是 50 episodes / 30 steps / 5 variations；正式命令必须显式覆盖。

判断: 当前实现与 ExpeL 原论文差距较大，更准确名称是 `ExpeL-inspired insight injection`。Step 4 ExpeL 结果只说明“同一 test stream 上收集规则后有提升”，不能进入论文主表的 ExpeL baseline。若主表要放 ExpeL，应重跑或重实现:

1. 在 train/dev 上用 Reflexion-style multi-trial 收集 experience pool。
2. 从 experience pool 中同时保留 success / failure trajectories。
3. 按原论文机制做 insight extraction: failure-success pairs + success chunks + operator/count 更新。
4. 在 test 上单次完整评估，不跳过成功 episode。
5. prompt 同时注入 insights 和 task-similarity retrieved successful trajectories。
6. 同时报 agent、reflection、extractor、retrieval/full evaluation cost。

可接受的弱写法:

- “We include an ExpeL-inspired same-test insight adaptation diagnostic.”
- “This variant uses batch insight extraction but omits Reflexion-style experience gathering and successful-trajectory retrieval.”

不可接受的写法:

- “We faithfully reproduce ExpeL.”
- “ExpeL held-out ScienceWorld baseline achieves 49.55%.”

## 5. 横向一致性结论

旧结果可复用范围:

- 12 task / 111 episode 的核心 test 选择。
- ReAct / Reflexion / ExpeL 旧轨迹可用于排查代码和 prompt 问题。
- Reflexion raw pass 只能诊断“强行第二遍会退化”。
- ExpeL Step 4 只能作为 test-time adaptation diagnostic。

必须先修后重跑:

- 所有正式表格统一从 `episodes[].score` 重算 raw score；normalized / clamped 只用于旧结果审计说明。
- 新结果 JSON 写入 `variation_idx`、split、task list、max variations、step limit。
- ReAct 补齐协议字段后重跑正式 baseline。
- Reflexion 改为 solved@trial runner，并计入 reflection token。
- ExpeL 若作为严格 baseline，需要按原论文重实现 train/dev experience -> insight -> test full eval。
- React-FM Online / Offline 也要按同一协议重跑，旧 Online `64.25` 不进主表。

不建议写法:

- “ExpeL 在标准 held-out ScienceWorld test baseline 上达到 49.55%。”
- “Reflexion P1 Raw 是 paper-aligned 最终结果。”
- “Reflexion P1 Adj 是一次真实完整 pass 的成本。”
- “React-FM Online raw score 是 64.25。”

建议写法:

- “ReAct 是纯 baseline；React-FM Online 是 test-time adaptation。”
- “Reflexion 报告 pass0、raw pass1 诊断值、solved@2 cumulative result，并单独说明成本下界。”
- “Step 4 ExpeL 是 same-test insight adaptation；严格 ExpeL baseline 需另跑 train/dev -> test。”

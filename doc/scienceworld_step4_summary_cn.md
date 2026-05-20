# ScienceWorld Step 4 中文总结

> 日期: 2026-04-21
> 范围: ScienceWorld 核心任务协议下，Step 4 无 gate 对比实验总结
> 对应 runbook 提及的总结路径: `doc/scienceworld_step4_summary.md`
> 更新: 2026-05-06，加入实验 D（ExpeL baseline）

## 实验设置

- 实验 A: 在线 React_FM
- 实验 B: 离线 memory
- 实验 C: ReAct baseline
- Reflexion: 按 episode 在线两遍的变体
- 实验 D: ExpeL baseline，Epoch 1 收集轨迹，Epoch 2 注入抽取出的 insights

主要结果文件:

- `results/20260415_153753/sw_step4_expA_online_test_1.json`
- `results/20260417_092133/sw_step4_expB_offline_test_1.json`
- `results/20260418_130202/sw_step4_expC_react_baseline_test_1.json`
- `results/20260420_180544/sw_step4_reflexion_online_formal_pass0.json`
- `results/20260420_180544/sw_step4_reflexion_online_formal_pass1.json`
- `results/20260420_180544/sw_step4_reflexion_online_formal_online_checkpoint.json`
- `results/20260506_001059/sw_step4_expD_expel_test_1.json`
- `results/20260506_001059/sw_step4_expD_expel_test_2.json`
- `memory_store/20260506_001059/sw_expel_insights.json`

相关日志:

- `logs/20260415_153753/sw_step4_expA_online_test.log`
- `logs/20260417_092133/sw_step4_expB_offline_test.log`
- `logs/20260418_130202/sw_step4_expC_react_baseline_test.log`
- `logs/20260420_180544/sw_step4_reflexion_online_formal.log`
- `logs/20260506_001059/sw_step4_expD_expel_test.log`

## 分数口径

- 单个 episode 的 `score` 是 ScienceWorld 原始分数，通常落在 `[-100, 100]`。
- 结果 JSON 里的 summary `avg_score` 是归一化后的值，即 `raw_score / 100`。
- 下方表格里:
  - `平均分（原始）` 表示 ScienceWorld 原始平均分
  - `平均分（归一化）` 表示结果文件里保存的归一化平均分

## Reflexion 统计口径

当前 Reflexion runner 的原始 `pass1` 会把所有 episode 都重新跑一遍，包括 `pass0` 已经成功的 episode。因此，原始 `pass1` 不能直接作为 Step 4 的正式结论。

Step 4 中，Reflexion 采用如下修正统计口径:

- 如果 `pass0` 成功，则该 episode 的最终 Reflexion 结果直接沿用 `pass0`
- 如果 `pass0` 失败，则该 episode 的最终 Reflexion 结果使用真实 `pass1`

下文把这套修正后结果记为 `Reflexion P1 Adj`。

## 总体对比


| 方法                  | Episodes | 成功数 | 成功率    | 平均分（原始） | 平均分（归一化） |
| ------------------- | -------- | --- | ------ | ------- | -------- |
| ExpA 在线 React_FM    | 111      | 46  | 41.44% | 64.25   | 0.6425   |
| ExpB 离线 Memory      | 111      | 44  | 39.64% | 48.83   | 0.4883   |
| ExpC ReAct Baseline | 111      | 42  | 37.84% | 47.21   | 0.4721   |
| Reflexion P0        | 111      | 28  | 25.23% | 40.57   | 0.4057   |
| Reflexion P1 Raw    | 111      | 25  | 22.52% | 36.90   | 0.3690   |
| Reflexion P1 Adj    | 111      | 41  | 36.94% | 49.39   | 0.4939   |
| ExpD ExpeL E1       | 111      | 42  | 37.84% | 51.30   | 0.5130   |
| ExpD ExpeL E2       | 111      | 55  | 49.55% | 57.78   | 0.5778   |


## Token 对比


| 方法                  | 总 Tokens   | 平均 Tokens / Episode | 说明                                |
| ------------------- | ---------- | ------------------- | --------------------------------- |
| ExpA 在线 React_FM    | 21,891,951 | 197,225             | 实际运行成本                            |
| ExpB 离线 Memory      | 20,027,604 | 180,429             | 实际运行成本                            |
| ExpC ReAct Baseline | 20,591,368 | 185,508             | 实际运行成本                            |
| Reflexion P0        | 12,395,129 | 111,668             | 仅第一遍运行成本                          |
| Reflexion P1 Raw    | 11,207,907 | 100,972             | 仅第二遍运行成本                          |
| Reflexion P1 Adj    | 11,965,847 | 107,800             | 后处理选中结果的成本，不是真实总运行成本              |
| Reflexion 运行成本下界    | 23,603,036 | 212,640             | `P0 + P1 raw`，且不含 reflection 生成调用 |
| ExpD ExpeL E2       | 16,439,640 | 148,105             | Epoch 2 结果文件总成本；包含 279,145 extractor tokens |
| ExpD ExpeL 有效成本  | 21,196,733 | 190,961             | 有效结果成本：E1 成功沿用 E1 token，E1 失败使用 E2 insight 后结果 |


Token 说明:

- `Reflexion P1 Adj` 适合汇报“最终效果”，但不代表真实账单成本。
- Reflexion 的真实运行成本至少是 `P0 + P1 raw = 212,640` tokens / episode。
- 这个值仍然是下界，因为 reflection 生成调用没有记入 `episodes[].total_tokens`。
- ExpeL E2 的 token 统计包含 agent tokens 和 Epoch 1 后的 insight-extraction tokens。论文主表使用的有效成本是 `190.9K` tokens / episode，因为 E1 成功 episode 沿用 E1 输出，E1 失败 episode 使用 E2 输出。

## 差值总结


| 对比项                             | 成功数变化 | 成功率变化    | 平均分（原始）变化 | 平均 Tokens / Episode 变化 |
| ------------------------------- | ----- | -------- | --------- | ---------------------- |
| A - C                           | +4    | +3.60pp  | +17.04    | +11,717                |
| B - C                           | +2    | +1.80pp  | +1.62     | -5,079                 |
| A - B                           | +2    | +1.80pp  | +15.42    | +16,796                |
| Reflexion P1 Adj - Reflexion P0 | +13   | +11.71pp | +8.82     | -3,868                 |
| Reflexion P1 Adj - ExpC         | -1    | -0.90pp  | +2.18     | 仅按有效结果计: -77,708       |
| Reflexion P1 Adj - ExpA         | -5    | -4.50pp  | -14.86    | 仅按有效结果计: -89,424       |
| ExpeL E2 - ExpC                 | +13   | +11.71pp | +10.57    | 有效成本: +5,453             |
| ExpeL E2 - ExpA                 | +9    | +8.11pp  | -6.47     | 有效成本: -6,264             |


说明:

- 后两行 token 变化使用的是 `Reflexion P1 Adj` 的“有效结果成本”，不是完整运行成本。
- 如果按真实运行成本计，Reflexion 是 Step 4 里成本最高的方法。

## 各任务成功率（%）


| Task Type                                    | ExpA Online | ExpB Offline | ExpC ReAct | Reflexion P0 | Reflexion P1 Adj | ExpD ExpeL E2 |
| -------------------------------------------- | ----------- | ------------ | ---------- | ------------ | ---------------- | -------------- |
| `boil`                                       | 33.3        | 33.3         | 44.4       | 11.1         | 11.1             | 44.4           |
| `chemistry-mix`                              | 37.5        | 37.5         | 25.0       | 12.5         | 25.0             | 37.5           |
| `grow-plant`                                 | 30.0        | 30.0         | 0.0        | 10.0         | 30.0             | 50.0           |
| `inclined-plane-friction-unnamed-surfaces`   | 40.0        | 50.0         | 60.0       | 20.0         | 50.0             | 30.0           |
| `lifespan-longest-lived-then-shortest-lived` | 50.0        | 60.0         | 50.0       | 40.0         | 40.0             | 70.0           |
| `melt`                                       | 66.7        | 44.4         | 33.3       | 11.1         | 22.2             | 88.9           |
| `mendelian-genetics-known-plant`             | 90.0        | 70.0         | 80.0       | 70.0         | 80.0             | 70.0           |
| `mendelian-genetics-unknown-plant`           | 0.0         | 0.0          | 0.0        | 0.0          | 0.0              | 0.0            |
| `power-component`                            | 60.0        | 0.0          | 20.0       | 20.0         | 40.0             | 20.0           |
| `test-conductivity`                          | 10.0        | 10.0         | 20.0       | 30.0         | 30.0             | 40.0           |
| `test-conductivity-of-unknown-substances`    | 0.0         | 20.0         | 30.0       | 10.0         | 20.0             | 30.0           |
| `use-thermometer`                            | 90.0        | 100.0        | 80.0       | 60.0         | 90.0             | 100.0          |
| **Overall**                                  | **41.44**   | **39.64**    | **37.84**  | **25.23**    | **36.94**        | **49.55**      |


## 各任务平均 Tokens / Episode


| Task Type                                    | ExpA Online | ExpB Offline | ExpC ReAct | Reflexion P0 | Reflexion P1 Adj |
| -------------------------------------------- | ----------- | ------------ | ---------- | ------------ | ---------------- |
| `boil`                                       | 200,966     | 175,097      | 159,992    | 108,739      | 98,646           |
| `chemistry-mix`                              | 261,656     | 250,817      | 258,236    | 95,659       | 114,322          |
| `grow-plant`                                 | 256,668     | 239,968      | 274,285    | 132,425      | 188,795          |
| `inclined-plane-friction-unnamed-surfaces`   | 303,752     | 185,137      | 188,694    | 117,220      | 66,474           |
| `lifespan-longest-lived-then-shortest-lived` | 33,324      | 8,189        | 21,186     | 7,015        | 16,743           |
| `melt`                                       | 161,424     | 197,764      | 177,501    | 107,238      | 92,299           |
| `mendelian-genetics-known-plant`             | 119,182     | 115,187      | 127,504    | 118,552      | 139,359          |
| `mendelian-genetics-unknown-plant`           | 265,132     | 262,135      | 260,979    | 156,250      | 167,558          |
| `power-component`                            | 139,671     | 241,265      | 193,826    | 177,030      | 69,348           |
| `test-conductivity`                          | 225,908     | 238,997      | 235,758    | 178,620      | 120,871          |
| `test-conductivity-of-unknown-substances`    | 305,607     | 240,674      | 249,348    | 112,796      | 140,054          |
| `use-thermometer`                            | 74,312      | 55,614       | 94,138     | 57,216       | 58,749           |


## 主要结论

1. 实验 D（ExpeL）取得 Step 4 最高成功率：`49.55%`，平均原始分 `57.78`。
2. 实验 A（online React_FM）取得最高原始平均分：`64.25`，成功率为 `41.44%`。
3. 实验 B 仍然是 React_FM 风格运行里成本较低的可行方案，但成功率不及实验 A 和 ExpeL。
4. 实验 C 仍然是一个强基线，在多个任务上依然有竞争力，尤其是 `boil`、`inclined-plane-friction-unnamed-surfaces`、`test-conductivity-of-unknown-substances`。
5. Reflexion 对统计口径非常敏感:
  原始 `pass1` 会低估它的真实效果，因为它把已成功 episode 在无记忆条件下又重跑了一次；
   修正后的 `P1 Adj` 相比 `P0` 有明显提升（`25.23% -> 36.94%`，`40.57 -> 49.39`）。
6. 即使按修正口径，Reflexion 仍然比 ExpA 低 `4.50` 个百分点，比 ExpeL 低 `12.61` 个百分点。
7. 从成本看，只有在“选中结果 token”口径下 Reflexion 才显得便宜；如果按真实运行成本计，它的平均下界为 `212,640` tokens / episode，高于 ExpA、ExpB、ExpC 和 ExpeL 有效成本。
8. `mendelian-genetics-unknown-plant` 在所有 Step 4 方案下都未解，说明它是稳定的难例，而不是单纯的 memory 形式问题。

## Step 4 最终结论

Step 4 支持以下三点明确结论:

1. ExpeL（实验 D）在完成完整 insight-collection epoch 后，是 ScienceWorld 核心任务上成功率最高的设置。
2. 在线 React_FM（实验 A）仍然是原始平均分最高的设置，并且能在 test stream 中在线学习，不需要预先完整收集一个 epoch。
3. 离线 memory（实验 B）是 React_FM 风格运行中成本较低的可行替代，但没有超过实验 A，也没有超过 ExpeL 的成功率。
4. Reflexion 风格的 episode 级重试，在修正统计口径下相对自身第一遍确实有帮助，但它不是 Step 4 的最优整体方法，而且真实运行成本较高。

因此，Step 4 的主结论不是“in-loop memory 或 episode-level memory 某一方一定最好”，而是:

- memory 在 ScienceWorld 上是有帮助的
- 帮助程度强烈依赖 task 类型
- 无条件重试或无条件注入 memory 都不是最优做法
- 当允许完整收集一个 epoch 时，ExpeL 式 batch insight extraction 在成功率上很强
- 更合理的后续方向仍然是 selective intervention，而不是 always-on intervention

## 正确性审核

本中文总结与英文版以及源结果做过交叉核对，检查范围包括:

- A / B / C 三组正式 JSON summary
- Reflexion 的 `pass0`、原始 `pass1`
- 从 `sw_step4_reflexion_online_formal_online_checkpoint.json` 逐 episode 重算得到的修正后 `P1 Adj`
- ExpeL Epoch 2 的 `sw_step4_expD_expel_test_2.json`，以及 `sw_expel_insights.json` 中的 insight task type

审核点:

- 成功数与成功率和源文件一致
- A / B / C 及 Reflexion 原始 pass 文件的归一化 `avg_score` 与源 summary 一致
- `Reflexion P1 Adj` 不是 runner 原始输出，而是基于逐 episode 记录重算得到
- Reflexion 的 token 说明明确区分了“后处理有效成本”和“真实运行下界成本”，避免误读
- ExpeL 输出包含 111 个 episodes；各任务数量匹配 Step 4；`insight_stats.total_entries = 200`；`total_retrievals = 69`；所有 insight task type 均为官方 ScienceWorld task name

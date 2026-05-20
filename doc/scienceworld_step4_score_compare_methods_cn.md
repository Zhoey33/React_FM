# ScienceWorld Step 4 三方法分数对比（中文版）

> 日期: 2026-04-21
> 对比方法:
>
> - `React_FM`: Step 4 实验 A（在线 React_FM）
> - `ReAct`: Step 4 实验 C（ReAct baseline）
> - `Reflexion`: 修正口径后的最终分数
>
> Reflexion 口径:
>
> - `pass0` 成功: 最终结果直接沿用 `pass0`
> - `pass0` 失败: 最终结果使用真实 `pass1`

数据来源:

- `results/20260415_153753/sw_step4_expA_online_test_1.json`
- `results/20260418_130202/sw_step4_expC_react_baseline_test_1.json`
- `results/20260420_180544/sw_step4_reflexion_online_formal_online_checkpoint.json`
- `logs/20260420_180544/sw_step4_reflexion_online_formal.log`

变体级明细 CSV:

- `doc/scienceworld_step4_score_compare_methods_by_variant.csv`

## 各任务类型平均分对比


| Task Type                                    | N   | React_FM Avg Score | ReAct Avg Score | Reflexion Avg Score | RFM-ReAct | RFM-Reflexion | Reflexion-ReAct |
| -------------------------------------------- | --- | ------------------ | --------------- | ------------------- | --------- | ------------- | --------------- |
| `boil`                                       | 9   | 51.89              | 63.44           | 31.78               | -11.56    | +20.11        | -31.67          |
| `chemistry-mix`                              | 8   | 42.25              | 50.50           | 49.50               | -8.25     | -7.25         | -1.00           |
| `grow-plant`                                 | 10  | 67.40              | 36.40           | 65.20               | +31.00    | +2.20         | +28.80          |
| `inclined-plane-friction-unnamed-surfaces`   | 10  | 59.50              | 57.50           | 54.50               | +2.00     | +5.00         | -3.00           |
| `lifespan-longest-lived-then-shortest-lived` | 10  | 10.00              | 0.00            | -10.00              | +10.00    | +20.00        | -10.00          |
| `melt`                                       | 9   | 65.11              | 24.22           | 40.44               | +40.89    | +24.67        | +16.22          |
| `mendelian-genetics-known-plant`             | 10  | 91.10              | 71.10           | 83.00               | +20.00    | +8.10         | +11.90          |
| `mendelian-genetics-unknown-plant`           | 10  | 14.60              | 14.80           | 14.40               | -0.20     | +0.20         | -0.40           |
| `power-component`                            | 5   | 92.00              | 82.60           | 81.80               | +9.40     | +10.20        | -0.80           |
| `test-conductivity`                          | 10  | 24.30              | 36.80           | 34.50               | -12.50    | -10.20        | -2.30           |
| `test-conductivity-of-unknown-substances`    | 10  | 70.60              | 57.30           | 70.80               | +13.30    | -0.20         | +13.50          |
| `use-thermometer`                            | 10  | 90.60              | 89.50           | 90.30               | +1.10     | +0.30         | +0.80           |


## 总体结论

- 按平均分看，`React_FM` 明显强于 `ReAct` 的任务类型有:
  - `melt` (`+40.89`)
  - `grow-plant` (`+31.00`)
  - `mendelian-genetics-known-plant` (`+20.00`)
  - `test-conductivity-of-unknown-substances` (`+13.30`)
  - `lifespan-longest-lived-then-shortest-lived` (`+10.00`)
  - `power-component` (`+9.40`)
- `ReAct` 明显强于 `React_FM` 的任务类型有:
  - `test-conductivity` (`React_FM - ReAct = -12.50`)
  - `boil` (`-11.56`)
  - `chemistry-mix` (`-8.25`)
- `Reflexion` 相比 `ReAct` 改善较大的任务类型有:
  - `grow-plant` (`+28.80`)
  - `melt` (`+16.22`)
  - `test-conductivity-of-unknown-substances` (`+13.50`)
  - `mendelian-genetics-known-plant` (`+11.90`)
- `Reflexion` 相比 `ReAct` 退化较明显的任务类型有:
  - `boil` (`-31.67`)
  - `lifespan-longest-lived-then-shortest-lived` (`-10.00`)
  - `inclined-plane-friction-unnamed-surfaces` (`-3.00`)
  - `test-conductivity` (`-2.30`)
- `React_FM` 相比 `Reflexion` 明显更强的任务类型有:
  - `melt` (`+24.67`)
  - `boil` (`+20.11`)
  - `lifespan-longest-lived-then-shortest-lived` (`+20.00`)
  - `power-component` (`+10.20`)
  - `mendelian-genetics-known-plant` (`+8.10`)
- `Reflexion` 相比 `React_FM` 更强的任务类型主要是:
  - `chemistry-mix` (`React_FM - Reflexion = -7.25`)
  - `test-conductivity-of-unknown-substances` (`-0.20`)

## 典型变体对比

### React_FM 相比 ReAct 提升最大的变体


| Env | Task                                         | Var | React_FM | ReAct | Reflexion | RFM-ReAct |
| --- | -------------------------------------------- | --- | -------- | ----- | --------- | --------- |
| 80  | `lifespan-longest-lived-then-shortest-lived` | 101 | 100      | -100  | 0         | +200      |
| 95  | `mendelian-genetics-known-plant`             | 93  | 100      | -100  | 100       | +200      |
| 52  | `test-conductivity-of-unknown-substances`    | 458 | 60       | -100  | 74        | +160      |
| 13  | `melt`                                       | 24  | 43       | -100  | 42        | +143      |
| 89  | `inclined-plane-friction-unnamed-surfaces`   | 127 | 35       | -100  | 10        | +135      |


### Reflexion 相比 ReAct 提升最大的变体


| Env | Task                                       | Var | React_FM | ReAct | Reflexion | Reflexion-ReAct |
| --- | ------------------------------------------ | --- | -------- | ----- | --------- | --------------- |
| 95  | `mendelian-genetics-known-plant`           | 93  | 100      | -100  | 100       | +200            |
| 52  | `test-conductivity-of-unknown-substances`  | 458 | 60       | -100  | 74        | +174            |
| 36  | `test-conductivity`                        | 677 | -100     | -100  | 60        | +160            |
| 13  | `melt`                                     | 24  | 43       | -100  | 42        | +142            |
| 89  | `inclined-plane-friction-unnamed-surfaces` | 127 | 35       | -100  | 10        | +110            |


### React_FM 相比 Reflexion 提升最大的变体


| Env | Task                                         | Var | React_FM | ReAct | Reflexion | RFM-Reflexion |
| --- | -------------------------------------------- | --- | -------- | ----- | --------- | ------------- |
| 77  | `lifespan-longest-lived-then-shortest-lived` | 98  | 100      | 100   | -100      | +200          |
| 8   | `boil`                                       | 28  | 75       | 73    | -100      | +175          |
| 15  | `melt`                                       | 26  | 100      | 100   | 0         | +100          |
| 17  | `melt`                                       | 28  | 100      | 0     | 0         | +100          |
| 80  | `lifespan-longest-lived-then-shortest-lived` | 101 | 100      | -100  | 0         | +100          |


## 读法建议

- 如果你关心“哪类任务更适合 React_FM”，优先看 `RFM-ReAct`
- 如果你关心“Reflexion 是否有独特优势”，优先看 `Reflexion-ReAct`
- 如果你关心“React_FM 和 Reflexion 谁更强”，优先看 `RFM-Reflexion`

从当前结果看：

- `React_FM` 更像在 `melt / grow-plant / genetics-known / power-component` 这类任务上更稳定
- `Reflexion` 在 `test-conductivity-of-unknown-substances` 和部分 `grow-plant` / `melt` 变体上也能打出很高分
- `boil` 和 `test-conductivity` 是区分方法行为差异最明显、同时也是最不稳定的任务类型

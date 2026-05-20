# ScienceWorld Step 4 分数变化分析（中文版）

> 日期: 2026-04-21
> 统计口径: Reflexion 采用修正口径
>
> - `pass0` 成功: 最终结果直接沿用 `pass0`
> - `pass0` 失败: 最终结果使用真实 `pass1`

数据来源:

- `results/20260420_180544/sw_step4_reflexion_online_formal_online_checkpoint.json`
- `logs/20260420_180544/sw_step4_reflexion_online_formal.log`

变体级明细 CSV:

- `doc/scienceworld_step4_score_delta_by_variant.csv`

## Task Type 级别分数变化


| Task Type                                    | 样本数 | Pass0 平均分 | 最终平均分  | 平均变化   | 改善  | 下降  | 不变  | Pass0 成功 | 最终成功 |
| -------------------------------------------- | --- | --------- | ------ | ------ | --- | --- | --- | -------- | ---- |
| `boil`                                       | 9   | 57.22     | 31.78  | -25.44 | 4   | 4   | 1   | 1        | 1    |
| `chemistry-mix`                              | 8   | 39.12     | 49.50  | +10.38 | 3   | 2   | 3   | 1        | 2    |
| `grow-plant`                                 | 10  | 36.30     | 65.20  | +28.90 | 7   | 2   | 1   | 1        | 3    |
| `inclined-plane-friction-unnamed-surfaces`   | 10  | 14.50     | 54.50  | +40.00 | 4   | 3   | 3   | 2        | 5    |
| `lifespan-longest-lived-then-shortest-lived` | 10  | 0.00      | -10.00 | -10.00 | 0   | 1   | 9   | 4        | 4    |
| `melt`                                       | 9   | 17.22     | 40.44  | +23.22 | 4   | 0   | 5   | 1        | 2    |
| `mendelian-genetics-known-plant`             | 10  | 72.90     | 83.00  | +10.10 | 3   | 0   | 7   | 7        | 8    |
| `mendelian-genetics-unknown-plant`           | 10  | 13.10     | 14.40  | +1.30  | 6   | 3   | 1   | 0        | 0    |
| `power-component`                            | 5   | 82.40     | 81.80  | -0.60  | 1   | 1   | 3   | 1        | 2    |
| `test-conductivity`                          | 10  | 57.70     | 34.50  | -23.20 | 0   | 4   | 6   | 3        | 3    |
| `test-conductivity-of-unknown-substances`    | 10  | 38.00     | 70.80  | +32.80 | 8   | 1   | 1   | 1        | 2    |
| `use-thermometer`                            | 10  | 78.30     | 90.30  | +12.00 | 3   | 1   | 6   | 6        | 9    |


## 按分数变化看，提升最大的任务类型

1. `inclined-plane-friction-unnamed-surfaces`: `+40.00`
2. `test-conductivity-of-unknown-substances`: `+32.80`
3. `grow-plant`: `+28.90`
4. `melt`: `+23.22`
5. `use-thermometer`: `+12.00`

## 按分数变化看，下降最明显的任务类型

1. `boil`: `-25.44`
2. `test-conductivity`: `-23.20`
3. `lifespan-longest-lived-then-shortest-lived`: `-10.00`
4. `power-component`: `-0.60`

## 提升最大的变体实例


| Env | Task                                       | Var | Pass0 | Final | Delta | 最终成功 | 来源    |
| --- | ------------------------------------------ | --- | ----- | ----- | ----- | ---- | ----- |
| 86  | `inclined-plane-friction-unnamed-surfaces` | 124 | -100  | 100   | +200  | 1    | pass1 |
| 45  | `test-conductivity-of-unknown-substances`  | 451 | -100  | 74    | +174  | 0    | pass1 |
| 88  | `inclined-plane-friction-unnamed-surfaces` | 126 | -100  | 5     | +105  | 0    | pass1 |
| 15  | `melt`                                     | 26  | -100  | 0     | +100  | 0    | pass1 |
| 58  | `grow-plant`                               | 97  | 1     | 100   | +99   | 1    | pass1 |
| 16  | `melt`                                     | 27  | 3     | 100   | +97   | 1    | pass1 |
| 26  | `use-thermometer`                          | 412 | 3     | 100   | +97   | 1    | pass1 |
| 28  | `use-thermometer`                          | 414 | 3     | 100   | +97   | 1    | pass1 |
| 61  | `grow-plant`                               | 100 | 4     | 100   | +96   | 1    | pass1 |
| 84  | `inclined-plane-friction-unnamed-surfaces` | 122 | 5     | 100   | +95   | 1    | pass1 |
| 98  | `mendelian-genetics-known-plant`           | 96  | 11    | 100   | +89   | 1    | pass1 |
| 6   | `boil`                                     | 26  | 0     | 75    | +75   | 0    | pass1 |
| 68  | `chemistry-mix`                            | 28  | 33    | 100   | +67   | 1    | pass1 |
| 51  | `test-conductivity-of-unknown-substances`  | 457 | 10    | 69    | +59   | 0    | pass1 |
| 87  | `inclined-plane-friction-unnamed-surfaces` | 125 | 50    | 100   | +50   | 1    | pass1 |


## 下降最大的变体实例


| Env | Task                                         | Var | Pass0 | Final | Delta | 最终成功 | 来源    |
| --- | -------------------------------------------- | --- | ----- | ----- | ----- | ---- | ----- |
| 8   | `boil`                                       | 28  | 70    | -100  | -170  | 0    | pass1 |
| 35  | `test-conductivity`                          | 676 | 60    | -100  | -160  | 0    | pass1 |
| 3   | `boil`                                       | 23  | 42    | -100  | -142  | 0    | pass1 |
| 77  | `lifespan-longest-lived-then-shortest-lived` | 98  | 0     | -100  | -100  | 0    | pass1 |
| 24  | `use-thermometer`                            | 410 | 85    | 3     | -82   | 0    | pass1 |
| 34  | `test-conductivity`                          | 675 | 64    | 10    | -54   | 0    | pass1 |
| 90  | `inclined-plane-friction-unnamed-surfaces`   | 128 | 40    | 5     | -35   | 0    | pass1 |
| 46  | `test-conductivity-of-unknown-substances`    | 452 | 64    | 33    | -31   | 0    | pass1 |
| 31  | `power-component`                            | 17  | 83    | 63    | -20   | 0    | pass1 |
| 36  | `test-conductivity`                          | 677 | 74    | 60    | -14   | 0    | pass1 |
| 110 | `mendelian-genetics-unknown-plant`           | 368 | 13    | 1     | -12   | 0    | pass1 |
| 85  | `inclined-plane-friction-unnamed-surfaces`   | 123 | 30    | 20    | -10   | 0    | pass1 |
| 70  | `chemistry-mix`                              | 30  | 42    | 33    | -9    | 0    | pass1 |
| 67  | `chemistry-mix`                              | 27  | 33    | 25    | -8    | 0    | pass1 |
| 83  | `inclined-plane-friction-unnamed-surfaces`   | 121 | 10    | 5     | -5    | 0    | pass1 |


## 结论

- 从分数角度看，Reflexion 并不是“整体统一增益”，而是明显表现出任务依赖性。
- 最受益的任务类型是:
  - `inclined-plane-friction-unnamed-surfaces`
  - `test-conductivity-of-unknown-substances`
  - `grow-plant`
  - `melt`
- 明显受损的任务类型是:
  - `boil`
  - `test-conductivity`
  - `lifespan-longest-lived-then-shortest-lived`
- `mendelian-genetics-unknown-plant` 虽然平均分有轻微上升，但最终成功数仍然为 `0`，说明当前 Reflexion 只带来了局部修正，没有真正解决任务。

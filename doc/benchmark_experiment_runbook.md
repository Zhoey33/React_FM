# Benchmark Experiment Runbook

最后更新: 2026-04-21

## 目的

这个文档用于记录按步骤执行 benchmark 实验的实际过程，先服务于 ScienceWorld，后续复用于 ALFWorld、WebShop 等 benchmark。

记录原则:
- 只记录已经实际验证过的命令、结果和阻塞点
- 每一步都补充“为什么做”“结果是什么”“对其他 benchmark 有什么迁移价值”
- 用户逐步下指令，实验按指令推进，不跳步

---

## 当前实验总步骤

### Step 1. 充分调查数据集

目标:
- 搞清楚 benchmark 到底是什么
- 搞清楚官方定义与仓库实现是否一致
- 搞清楚任务结构、split、variation 数量、默认步长、默认任务子集

本步已经完成的内容:
- 确认 ScienceWorld 是文本交互式科学实验 benchmark
- 确认官方共有 `30` 个 task types
- 确认官方 split 为 `train/dev/test`
- 统计了每个 task type 在 `train/dev/test` 下的 variation 数量
- 确认当前仓库默认最大 ReAct 步长为 `100`
- 核对了官方 ScienceWorld API 对 `score` / `reward` / `done` 的定义
- 识别出“官方 7 个 topic”与“仓库内部 10 个分析标签”的口径差异
- 识别出原仓库部分脚本把 `split=\"test\"` 写死的问题

本步产出:
- 对 benchmark 的定性理解
- 对 benchmark 的定量统计
- 对“官方定义 vs 仓库实现”的差异清单
- 一条额外经验: 不能只看论文或脚本表面字段，必须结合官方源码判断 score 和完成条件

### Step 2. 根据项目特点选取实验数据

目标:
- 不机械沿用仓库旧设置
- 按项目研究问题选择更合适的数据
- 明确主实验、gate 训练、调参和最终评测分别使用哪些数据

本步已经完成的内容:
- 明确本项目的重点是:
  - failure-recovery memory
  - in-loop intervention
  - gate 决策
- 因此不建议直接用:
  - 仓库当前 test-only 方案
  - 或官方 30 任务全集直接起跑
- 已确定采用:
  - 官方 `train/dev/test`
  - 加“核心任务集 + 对照任务集”的协议

本步产出:
- ScienceWorld 正式实验协议 v1
- 核心任务集 12 个
- 对照任务集 3 个
- `train/dev/test` 各自的 variation cap 和总规模

### Step 3. 根据数据集优化错误检测器与记忆提取

目标:
- 不脱离 ScienceWorld 数据结构空谈 detector / extractor
- 让 failure 定义、memory scope、post-episode extraction 与真实轨迹对齐
- 先修“能不能正确抓到 failure 和 recovery”，再谈大规模正式实验

本步已经完成的内容:
- 基于 ScienceWorld 真实失败语句，重审了显式失败抓取规则
- 关闭了对当前主线不利的隐式失败 `unproductive`
- 将 ScienceWorld memory scope 明确改为严格 `task_type` 隔离
- 修复了 failure step 与 extractor trajectory 的错位问题
- 让 extractor 显式接收 `failure_step` 和运行时检测到的 failures
- 约束 extractor 只提取 failure 之后真实发生的修复动作
- 在多个 task 上做了小规模交叉验证，而不是只盯单一任务

本步当前结论:
- detector 的显式失败规则整体合理，但仍可继续细化
- extractor 的“抓不到 memory”主因之一已经确认是 step 对齐 bug，且已修复
- 修复后已经能在真实运行中看到 memory store 和 retrieval
- 但 memory 质量还没有完全干净，当前主要噪声是局部修补型 memory 和相关性排序不足

本步产出:
- ScienceWorld failure taxonomy v1
- 严格 `task_type` 记忆隔离策略
- 对齐修复后的 extractor 流程
- 一组跨任务的小实验验证结果

### Step 4. 按优化后协议执行实验

目标:
- 在第二步选定的数据协议上，先测无 gate 的 React_FM 能达到什么水平
- 区分在线 memory 与离线 memory 两种实验口径
- 为后续 gate 实验提供清晰、可比较的无 gate 基线

当前已确定的第四步实验设计:

#### 实验 A: 在线 React_FM

用途:
- 作为当前主线的无 gate 正式基线
- 测量“空 memory 起步，边跑边积累、边检索边注入”时的真实在线水平

数据:
- 核心任务集 `test`
- 每个 task 最多 `10` 个 variations
- 如果某个 task 在 `test` 中不足 `10` 个，就取全部可用 variations

运行方式:
- 初始 memory 为空
- 开启 failure detection
- 开启 post-episode memory extraction
- 开启 in-loop memory retrieval / injection
- 不使用 gate

#### 实验 B: 离线 memory 实验

用途:
- 测量“先准备好 memory，再运行”时 React_FM 能获得多少收益
- 作为 memory 机制上限与可利用性的正式分析版本

分两段:

1. memory collection
- 数据: 核心任务集 `train`
- 每个 task 最多 `20` 个 variations
- 如果某个 task 在 `train` 中不足 `20` 个，就取全部可用 variations
- 运行方式: `inject_mode=none`
- 只做 failure detection 和 post-episode extraction
- 不做 memory retrieval / injection

2. offline evaluation
- 数据: 核心任务集 `test`
- 每个 task 最多 `10` 个 variations
- 加载第一段收集到的 memory
- 开启 in-loop retrieval / injection
- 不使用 gate

#### 第四步结果口径

本阶段所有无 gate 实验统一汇报:

- `success rate`
- `avg score`
- 分 task `success rate`
- 分 task `avg score`
- `avg steps`
- `total tokens`
- `action tokens`
- `memory extraction tokens`
- `avg total tokens / episode`
- `avg action tokens / episode`
- `avg extraction tokens / episode`
- `failures_detected`
- `memories_stored`
- `memories_retrieved`
- `retrieval hits`
- 实验 A / B / C 的总体与分 task delta

说明:
- 当前 ScienceWorld 已关闭隐式失败检测
- failure detection 以规则为主，不再依赖 judge 作为主实验成本项
- 因此第四步主结果不单独汇报 `judge tokens`
- 第四步汇总结论另存为: `doc/scienceworld_step4_summary.md`

后续将按以下顺序推进:

1. 先跑实验 A，建立在线无 gate 基线
2. 再跑实验 B 第一段，离线收集 train memory
3. 再跑实验 B 第二段，加载 memory 在 test 上评测
4. 再跑实验 C，建立无 memory 的 ReAct baseline
5. 最后汇总 A / B / C 对比结果，作为 gate 前的正式基线

#### 实验 A 最终结果: 在线 React_FM 已完成

运行信息:
- 运行时间: `2026-04-15 15:37:53` 到 `2026-04-16 00:33:21`
- 结果文件: `results/20260415_153753/sw_step4_expA_online_test_1.json`
- memory 文件: `memory_store/20260415_153753/sw_epoch1.json`

最终结果:
- 总 episode 数: `111`
- success rate: `46/111 = 41.44%`
- avg score: `0.5524`
- avg steps: `74.14`
- total tokens: `21,891,951`
- avg total tokens / episode: `197,224.8`
- action tokens: `21,479,157`
- memory extraction tokens: `412,794`
- avg action tokens / episode: `193,506.0`
- avg extraction tokens / episode: `3,718.9`
- memories stored: `297`
- memories retrieved: `1293`
- retrieval hits: `1049`

分 task 结果:
- `boil`: `3/9`, `avg_score=0.5189`
- `chemistry-mix`: `3/8`, `avg_score=0.4225`
- `grow-plant`: `3/10`, `avg_score=0.6740`
- `inclined-plane-friction-unnamed-surfaces`: `4/10`, `avg_score=0.5950`
- `lifespan-longest-lived-then-shortest-lived`: `5/10`, `avg_score=0.1000`
- `melt`: `6/9`, `avg_score=0.6511`
- `mendelian-genetics-known-plant`: `9/10`, `avg_score=0.9110`
- `mendelian-genetics-unknown-plant`: `0/10`, `avg_score=0.1460`
- `power-component`: `3/5`, `avg_score=0.9200`
- `test-conductivity`: `1/10`, `avg_score=0.2430`
- `test-conductivity-of-unknown-substances`: `0/10`, `avg_score=0.7060`
- `use-thermometer`: `9/10`, `avg_score=0.9060`

结果解读:
- 在线 memory 在部分 task 上表现出明显收益，尤其是 `use-thermometer`、`mendelian-genetics-known-plant`、`melt`
- 但收益并不均匀，`test-conductivity`、`test-conductivity-of-unknown-substances`、`mendelian-genetics-unknown-plant` 仍然很差
- 说明当前无 gate 在线 React_FM 已经具备一定利用 memory 的能力，但 task 间稳定性差异仍然很大

口径注意:
- 结果文件与日志中的 `summary.avg_score=0.6425` 不是最终正式口径
- 原因是运行进程启动时仍使用旧版 summary 逻辑，负分没有按官方定义正确计入平均分
- 本文档记录的正式 `avg score=0.5524` 是按 episode 原始 `score` 重新计算后的结果
- ScienceWorld 官方允许负分，且 `score < 0` 会直接终止 episode，因此后续所有汇报都必须使用重算后的正式口径

#### 实验 B 执行方案: 离线 memory

目标:
- 先在核心任务集 `train` 上离线收集 memory
- 再把这份 memory 加载到核心任务集 `test`
- 作为“memory 先验已准备好”条件下的正式离线基线

执行顺序:

1. 先跑 memory collection
- split: `train`
- tasks: 核心任务集 12 个
- max variations: `20`
- step limit: `100`
- inject mode: `none`
- 说明: 允许 failure detection 和 extractor 工作，但不做 retrieval / injection

2. 再跑 offline evaluation
- split: `test`
- tasks: 核心任务集 12 个
- max variations: `10`
- step limit: `100`
- inject mode: `in_loop`
- 说明: 加载第一段收集到的 memory，在正式 test 上评测

推荐命令:

1. memory collection

```bash
PATH=/opt/homebrew/opt/openjdk/bin:$PATH python experiments/run_scienceworld.py \
  --config config_scienceworld.yaml \
  --split train \
  --tasks \
    boil melt use-thermometer power-component \
    test-conductivity test-conductivity-of-unknown-substances \
    grow-plant chemistry-mix \
    lifespan-longest-lived-then-shortest-lived \
    inclined-plane-friction-unnamed-surfaces \
    mendelian-genetics-known-plant mendelian-genetics-unknown-plant \
  --max-variations 20 \
  --step-limit 100 \
  --inject-mode none \
  --run-name sw_step4_expB_collect_train_
```

2. offline evaluation

说明:
- 下面命令中的 `--resume-memory` 需要替换成第一段实际生成的 memory 文件路径
- 正式应使用第一段最终文件，例如 `memory_store/<timestamp>/sw_epoch1.json`

```bash
PATH=/opt/homebrew/opt/openjdk/bin:$PATH python experiments/run_scienceworld.py \
  --config config_scienceworld.yaml \
  --split test \
  --tasks \
    boil melt use-thermometer power-component \
    test-conductivity test-conductivity-of-unknown-substances \
    grow-plant chemistry-mix \
    lifespan-longest-lived-then-shortest-lived \
    inclined-plane-friction-unnamed-surfaces \
    mendelian-genetics-known-plant mendelian-genetics-unknown-plant \
  --max-variations 10 \
  --step-limit 100 \
  --inject-mode in_loop \
  --resume-memory memory_store/<timestamp>/sw_epoch1.json \
  --run-name sw_step4_expB_offline_test_
```

预期产物:
- 第一段结果文件: `results/<timestamp>/sw_step4_expB_collect_train_1.json`
- 第一段 memory 文件: `memory_store/<timestamp>/sw_epoch1.json`
- 第二段结果文件: `results/<timestamp>/sw_step4_expB_offline_test_1.json`

执行提醒:
- 第二段评测前，不要误用实验 A 的 online memory 文件
- 第二段只能加载实验 B 第一段在 `train` 上收集到的 memory
- 第二段结果同样需要检查 `avg_score` 是否按官方负分定义重算
- 实验 A 与实验 B 对比时，必须保持:
  - 相同任务集合
  - 相同 `test` split
  - 相同 variation cap
  - 相同 step limit

#### 实验 B 第一段结果: train memory collection 已完成

运行信息:
- 运行时间: `2026-04-16 16:31:35`
- 结果文件: `results/20260416_163135/sw_step4_expB_collect_train_1.json`
- memory 文件: `memory_store/20260416_163135/sw_epoch1.json`
- 运行方式: `inject_mode=none`

最终结果:
- 总 episode 数: `214`
- success rate: `75/214 = 35.05%`
- avg score: `0.4072`
- total tokens: `41,928,483`
- avg total tokens / episode: `195,927.5`
- memories stored: `533`
- memories retrieved: `0`
- retrieval hits: `0`

结果解读:
- 第一段是干净的离线 memory collection，因为 `retrievals / hits = 0 / 0`
- 这说明训练集阶段没有在线注入 memory，产出的 `533` 条 memory 可以作为实验 B 第二段的独立先验
- 这份 memory 使用严格 `task_type` 隔离，与当前 ScienceWorld 主线协议一致

#### 实验 B 第二段最终结果: offline test 已完成

运行信息:
- 运行时间: `2026-04-17 09:21:33` 到 `2026-04-17 17:25:11`
- 结果文件: `results/20260417_092133/sw_step4_expB_offline_test_1.json`
- 初始 memory 文件: `memory_store/20260416_163135/sw_epoch1.json`
- 最终 memory 文件: `memory_store/20260417_092133/sw_epoch1.json`
- 运行方式: `inject_mode=in_loop`

最终结果:
- 总 episode 数: `111`
- success rate: `44/111 = 39.64%`
- avg score: `0.4883`
- avg steps: `70.32`
- total tokens: `20,027,604`
- avg total tokens / episode: `180,428.9`
- action tokens: `19,618,889`
- memory extraction tokens: `408,715`
- avg action tokens / episode: `176,746.7`
- avg extraction tokens / episode: `3,682.1`
- judge tokens: `0`
- memories stored: `856`
- memories retrieved: `878`
- retrieval hits: `878`

分 task 结果:
- `boil`: `3/9`, `avg_score=0.6056`
- `chemistry-mix`: `3/8`, `avg_score=0.5962`
- `grow-plant`: `3/10`, `avg_score=0.7310`
- `inclined-plane-friction-unnamed-surfaces`: `5/10`, `avg_score=0.3950`
- `lifespan-longest-lived-then-shortest-lived`: `6/10`, `avg_score=0.2000`
- `melt`: `4/9`, `avg_score=0.5344`
- `mendelian-genetics-known-plant`: `7/10`, `avg_score=0.6300`
- `mendelian-genetics-unknown-plant`: `0/10`, `avg_score=0.0260`
- `power-component`: `0/5`, `avg_score=0.7520`
- `test-conductivity`: `1/10`, `avg_score=0.1930`
- `test-conductivity-of-unknown-substances`: `2/10`, `avg_score=0.3660`
- `use-thermometer`: `10/10`, `avg_score=1.0000`

结果解读:
- 离线 memory 的检索链路是通的，最终 `878` 次 retrieval 全部命中 task_type bucket
- 离线 memory 明显帮助了 `use-thermometer`，达到 `10/10`
- `mendelian-genetics-unknown-plant` 仍然是最困难任务，实验 B 中 `0/10`，且 `avg_score=0.0260`
- 尾段遗传未知植物任务连续失败，是实验 B 总体结果低于实验 A 的主要来源之一
- 退出阶段出现过 `py4j` shutdown traceback，但发生在结果和 memory 保存之后，不影响实验有效性

#### 实验 C 最终结果: ReAct baseline 已完成

运行信息:
- 运行时间: `2026-04-18 13:02:03` 到 `2026-04-18 20:59:17`
- 结果文件: `results/20260418_130202/sw_step4_expC_react_baseline_test_1.json`
- 日志文件: `logs/20260418_130202/sw_step4_expC_react_baseline_test.log`
- 运行方式: `--baseline`

最终结果:
- 总 episode 数: `111`
- success rate: `42/111 = 37.84%`
- avg score: `0.4721`
- avg steps: `72.93`
- total tokens: `20,591,368`
- avg total tokens / episode: `185,507.8`
- action tokens: `20,591,368`
- memory extraction tokens: `0`
- memories stored / retrieved / retrieval hits: `0 / 0 / 0`

分 task 结果:
- `boil`: `4/9`, `avg_score=0.6344`
- `chemistry-mix`: `2/8`, `avg_score=0.5050`
- `grow-plant`: `0/10`, `avg_score=0.3640`
- `inclined-plane-friction-unnamed-surfaces`: `6/10`, `avg_score=0.5750`
- `lifespan-longest-lived-then-shortest-lived`: `5/10`, `avg_score=0.0000`
- `melt`: `3/9`, `avg_score=0.2422`
- `mendelian-genetics-known-plant`: `8/10`, `avg_score=0.7110`
- `mendelian-genetics-unknown-plant`: `0/10`, `avg_score=0.1480`
- `power-component`: `1/5`, `avg_score=0.8260`
- `test-conductivity`: `2/10`, `avg_score=0.3680`
- `test-conductivity-of-unknown-substances`: `3/10`, `avg_score=0.5730`
- `use-thermometer`: `8/10`, `avg_score=0.8950`

结果解读:
- baseline 总体略低于两个 memory 方案，但差距不大，不是可以忽略的弱基线
- baseline 在 `boil`、`inclined-plane-friction-unnamed-surfaces`、`test-conductivity`、`test-conductivity-of-unknown-substances` 上仍然更强
- 这说明 Step 4 的关键问题不是“memory 有没有用”，而是“哪些 task 该注入 memory，哪些 task 应该保持 baseline 行为”
- 日志尾部也有 `py4j` shutdown traceback，但发生在 `Results saved` 之后，不影响本轮结果有效性

#### 实验 A / B / C 正式对比

总体对比:
- 实验 A 在线 React_FM: `46/111 = 41.44%`, `avg_score=0.5524`, `avg_steps=74.14`, `avg_tokens/episode=197,224.8`
- 实验 B 离线 memory: `44/111 = 39.64%`, `avg_score=0.4883`, `avg_steps=70.32`, `avg_tokens/episode=180,428.9`
- 实验 C ReAct baseline: `42/111 = 37.84%`, `avg_score=0.4721`, `avg_steps=72.93`, `avg_tokens/episode=185,507.8`
- A - C success delta: `+4 episodes`, `+3.60 percentage points`
- A - C avg score delta: `+0.0803`
- A - C avg tokens / episode delta: `+11,717.0`
- B - C success delta: `+2 episodes`, `+1.80 percentage points`
- B - C avg score delta: `+0.0162`
- B - C avg tokens / episode delta: `-5,078.9`
- A - B success delta: `+2 episodes`, `+1.80 percentage points`
- A - B avg score delta: `+0.0641`
- A - B avg tokens / episode delta: `+16,795.9`
- 在线 A 检索命中率: `1049/1293 = 81.1%`
- 离线 B 检索命中率: `878/878 = 100.0%`

分 task 观察:
- 明确受益于 memory 的任务:
- `grow-plant`: baseline `0/10`, A/B 都提升到 `3/10`
- `chemistry-mix`: baseline `2/8`, A/B 都提升到 `3/8`
- `melt`: baseline `3/9`, B `4/9`, A `6/9`
- `use-thermometer`: baseline `8/10`, A `9/10`, B `10/10`
- 更适合在线 memory 的任务:
- `power-component`: A `3/5`，明显高于 baseline `1/5` 和 B `0/5`
- `mendelian-genetics-known-plant`: A `9/10`，高于 baseline `8/10` 和 B `7/10`
- baseline 仍然更强的任务:
- `boil`: baseline `4/9`，A/B 都是 `3/9`
- `inclined-plane-friction-unnamed-surfaces`: baseline `6/10`，B `5/10`，A `4/10`
- `test-conductivity`: baseline `2/10`，A/B 都是 `1/10`
- `test-conductivity-of-unknown-substances`: baseline `3/10`，B `2/10`，A `0/10`
- 持续困难任务:
- `mendelian-genetics-unknown-plant`: A / B / C 都是 `0/10`
- `lifespan-longest-lived-then-shortest-lived`: 只有 B 从 `5/10` 提到 `6/10`

当前正式结论:
- 总体最优仍是在线 React_FM A，但相对 baseline 的提升只有 `+3.60` 个百分点，且它也是三者里最贵的方案
- 离线 memory B 是三者里最省 token 的 memory 方案，甚至比 baseline 还低 `5,078.9` tokens / episode，但总体提升有限
- B 的 `100%` retrieval hit 没有转化为最优结果，说明当前瓶颈不是“能不能取到 memory”，而是“取到的 memory 是否真的有用”
- baseline 在 12 个 task 中仍有 4 个 success 更优，说明后续 gate 结论必须始终对照 baseline，而不能只比较 A 和 B
- `test-conductivity-of-unknown-substances` 是最典型的分裂案例: A 虽然 `0/10` 成功，但 `avg_score=0.7060` 高于 baseline 的 `0.5730`，说明 memory 有时能提高局部进展，却没能把轨迹推进到最终完成
- 下一阶段 gate 的目标不应定义为“默认触发 memory”，而应定义为“只在真正有益时触发 memory，在无益时保持 baseline 行为”

---

## 当前实验对象

- Benchmark: ScienceWorld
- 当前模式: 用户逐步指挥，按步执行
- 目标: 先把 ScienceWorld 实验稳定跑通，再沉淀成可复用流程

---

## 当前主线回顾

到目前为止，ScienceWorld 主线已经形成了比较清晰的三步结构:

### 第一阶段: 充分调查数据集

这一阶段解决的是“这个 benchmark 到底是什么、结构长什么样、仓库有没有跑偏”。

已经确认的核心事实:
- ScienceWorld 是多步文本交互式科学实验 benchmark
- 官方共有 `30` 个 task types，split 为 `train/dev/test`
- 全量 variation 总数为 `7207`
- 当前仓库默认最大 ReAct 步长为 `100`
- 仓库原有默认跑法和官方数据组织方式之间存在偏差，需要显式对齐

这一阶段的价值是:
- 把后续所有实验建立在真实 benchmark 定义之上
- 避免把“仓库默认脚本习惯”误当成“官方实验协议”

### 第二阶段: 根据项目特点选取数据

这一阶段解决的是“我们的项目到底该怎么用 ScienceWorld，而不是机械照搬全量数据”。

已经确定的原则:
- 数据选择要服务于 `failure-recovery memory`、`in-loop intervention` 和 `gate`
- 主实验、调参、gate 训练、最终测试要明确区分数据用途
- 不能直接沿用旧的 test-only 跑法
- 采用官方 `train/dev/test`，并结合核心任务集与对照任务集组织实验

这一阶段的价值是:
- 让实验目标和数据协议对齐
- 为后续 detector、extractor、gate 提供干净的数据边界

### 第三阶段: 根据数据集优化错误检测器与记忆提取

这一阶段解决的是“既然数据协议已经清楚，那么 failure 怎么定义、memory 怎么提取、怎么复用才合理”。

目前已经完成的关键工作:
- 记忆作用域最终定为严格 `task_type` 隔离，不做 fallback
- 显式失败抓取规则基于 ScienceWorld 真实观测语句重新检查并拆分
- 关闭隐式失败 `unproductive`，避免把低信息噪声混入 memory 流
- extractor 现在被约束为只提取 failure 之后真实发生的修复动作
- 修复了 failure step 和 trajectory step 错位导致的伪 memory 问题
- 在 `boil`、`use-thermometer`、`chemistry-mix`、`grow-plant` 等不同任务上进行了小规模验证

这一阶段目前的结论是:
- detector 和 extractor 已经比最初版本更贴近 ScienceWorld 的真实交互结构
- memory 已经可以被生成，也已经观察到真实 retrieval
- 但“memory 是否在第二遍运行中稳定带来行为改善”还需要专门的 two-pass 实验继续验证

因此，主线下一步不再是重新讨论数据，而是:
- 在现有优化后的 detector / extractor 基础上
- 继续验证 memory 的真实可用性和收益

---

## ScienceWorld 基本信息总结

### 它是干什么的

ScienceWorld 是一个文本交互式科学实验 benchmark。Agent 不是在聊天，而是在一个可操作的模拟实验环境里完成任务，例如:
- 把物质煮沸或融化
- 使用温度计测量
- 组装简单电路
- 测试导电性
- 找到生物或非生物
- 种植植物
- 混合化学物质
- 判断寿命或遗传规律

它考察的不是单轮问答，而是:
- 多步决策
- 环境状态跟踪
- 实验常识与程序性知识
- 失败后的修正能力

对我们这个项目来说，它特别适合测试:
- 长时域任务中的失败恢复
- action-level 的错误修复
- in-loop memory 是否能帮助 agent 中途纠偏

### Benchmark 结构

ScienceWorld 在当前安装包里共有 30 个 task type。

这里有两个统计口径:

1. 官方安装包 `tasks.json` 的 `topic` 字段
   - 一共 7 个大类
2. 本仓库 `src/scienceworld_env.py` 里的 `TASK_TOPIC_MAP`
   - 为了实验分析把 Biology 进一步拆细，所以内部使用 10 个 topic label

每个 task type 下有多个 variation。也就是说:
- `task` 决定任务类型
- `variation` 决定同类任务下的具体实例

在本仓库里，ScienceWorld 的主评估不是全量 30 个 task 全跑，而是选了 10 个代表性任务，每个任务最多取 5 个 test variations，组成约 50 个 episodes 的评测集。

当前默认评测任务是:
- `boil`
- `melt`
- `use-thermometer`
- `power-component`
- `test-conductivity`
- `find-living-thing`
- `grow-plant`
- `chemistry-mix`
- `lifespan-longest-lived`
- `mendelian-genetics-known-plant`

### 定量信息

#### 1. 官方 task type 总数

- task type 总数: `30`

#### 2. 官方 topic 分布

按安装包 `tasks.json` 统计:

| Topic | Task type 数量 |
|------|---:|
| Biology | 9 |
| Chemistry | 3 |
| Classification | 4 |
| Electricity | 4 |
| Forces | 3 |
| Matter | 4 |
| Measurement | 3 |

合计: `30`

#### 3. 每个 task type 的 variation 数量

按 ScienceWorld API 实际读取:

| Task type | Train | Dev | Test |
|------|---:|---:|---:|
| boil | 14 | 7 | 9 |
| melt | 14 | 7 | 9 |
| freeze | 14 | 7 | 9 |
| change-the-state-of-matter-of | 14 | 7 | 9 |
| use-thermometer | 270 | 135 | 135 |
| measure-melting-point-known-substance | 218 | 109 | 109 |
| measure-melting-point-unknown-substance | 150 | 75 | 75 |
| power-component | 10 | 5 | 5 |
| power-component-renewable-vs-nonrenewable-energy | 10 | 5 | 5 |
| test-conductivity | 450 | 225 | 225 |
| test-conductivity-of-unknown-substances | 300 | 150 | 150 |
| find-living-thing | 150 | 75 | 75 |
| find-non-living-thing | 150 | 75 | 75 |
| find-plant | 150 | 75 | 75 |
| find-animal | 150 | 75 | 75 |
| grow-plant | 62 | 31 | 33 |
| grow-fruit | 62 | 31 | 33 |
| chemistry-mix | 16 | 8 | 8 |
| chemistry-mix-paint-secondary-color | 18 | 9 | 9 |
| chemistry-mix-paint-tertiary-color | 18 | 9 | 9 |
| lifespan-longest-lived | 62 | 31 | 32 |
| lifespan-shortest-lived | 62 | 31 | 32 |
| lifespan-longest-lived-then-shortest-lived | 62 | 31 | 32 |
| identify-life-stages-1 | 6 | 3 | 5 |
| identify-life-stages-2 | 4 | 2 | 4 |
| inclined-plane-determine-angle | 84 | 42 | 42 |
| inclined-plane-friction-named-surfaces | 692 | 346 | 348 |
| inclined-plane-friction-unnamed-surfaces | 80 | 40 | 42 |
| mendelian-genetics-known-plant | 60 | 30 | 30 |
| mendelian-genetics-unknown-plant | 240 | 120 | 120 |

#### 4. 全量 variation 总数

- Train 总数: `3592`
- Dev 总数: `1796`
- Test 总数: `1819`
- 全部 variation 总数: `7207`

#### 5. 本仓库默认 ScienceWorld 主实验规模

默认配置来自 `src/scienceworld_env.py` 和 `experiments/run_scienceworld.py`:

- 默认任务数: `10`
- 每个任务默认最多取 `5` 个 variation
- 默认 split: `test`
- 因此默认评测规模约为 `50` 个 episodes

这里说“约”其实在当前默认任务集上就是 `50`，因为这 10 个任务的 test variations 都不少于 5 个。

#### 6. 默认最大 ReAct 步长

当前仓库里 ScienceWorld 的默认最大步长是 `100`。

依据:
- `experiments/run_scienceworld.py` 参数 `--step-limit` 默认值为 `100`
- `config_scienceworld.yaml` 里的 `agent.max_steps` 也是 `100`
- 实际 agent 使用的是:
  - `min(args.step_limit, config["agent"]["max_steps"])`

所以默认情况下:
- agent 最大步数 = `100`
- 环境 step limit = `100`

这也是当前项目修复后的设定。文档里有修复记录:
- 旧问题: `max_steps=50`
- 当前已改为: `100`

### 交互形式

ScienceWorld 是典型的“文本环境 + 动作接口”结构。

每个 episode 基本流程是:
1. 环境 `reset`
2. 返回任务描述和初始观察
3. Agent 输出一个动作
4. 环境执行动作并返回新观察、reward、done、score 等信息
5. 直到成功、失败或达到步数上限

在本仓库里，封装后的返回结构是:
- `observation`
- `reward`
- `done`
- `info`

其中 `info` 里最重要的是:
- `score`: 0 到 100
- `moves`
- `look`
- `inventory`
- `valid_actions`

### 动作空间长什么样

它不是自由生成长文本，而是输出单个动作或 `think`。

当前仓库里允许的动作前缀包括:
- `go to`
- `pick up`
- `put down`
- `open`
- `close`
- `activate`
- `deactivate`
- `use`
- `pour`
- `mix`
- `focus on`
- `wait`
- `look around`
- `inventory`
- `examine`
- `read`
- `connect`
- `move`
- `teleport to`
- `dunk`

所以它的本质是:
- LLM 负责决策
- 环境负责执行和状态转移

### 成功判定与评分

根据官方 ScienceWorld API 源码定义:
- `score = int(round(100 * getScore()))`
- `reward = 当前 score - 上一步 score`
- 如果 `score < 0`，环境会把 `done/isCompleted` 置为 `True`

这意味着在当前项目实现里:
- `score` 不是严格的 `0-100`
- `100` 表示完整成功
- 实际运行中也可能出现负分，例如严重错误或失败终止时出现 `-100`
- `score == 100` 仍可作为任务成功判定

这意味着:
- 它不是简单的“答对/答错”
- episode 中途可能有部分进展
- 也可能因为明显错误而进入负分终止状态

### 为什么 Step 1 必须看源码

这次 ScienceWorld 调研补充出了一个很重要的经验:
- benchmark 的字段名看起来简单，但真实语义不一定能从文档标题直接看出来
- 例如 `score` 表面上容易被理解成 `0-100`，但官方 API 实际允许负分
- `done` 也不只是“任务完成”，还可能由负分失败触发

因此，在 Step 1 的“充分调查数据集”阶段，除了看论文、README 和任务列表，还需要:
- 直接看官方安装包源码或 GitHub 仓库源码
- 明确 `step()` 返回的 `score / reward / done / info` 到底怎么定义
- 明确任务成功条件是否等于 `done`
- 明确是否存在失败终止、负分终止、超步数终止等额外完成语义

对 ScienceWorld 来说，这一步已经确认:
- 官方 API 用 `score >= 100` 表示完整成功
- `reward` 是分数增量
- `score < 0` 时 episode 会被环境直接终止
- 很适合分析“有没有向正确方向前进”

不过这个 benchmark 在当前项目里还有一个关键特征:
- 内置 score 粒度不够细
- 有些轨迹明明更接近正确解，但分数暂时不变

因此 gate 流程里又额外引入了:
- gold path
- LLM-as-judge trajectory scoring

### 在本仓库里的封装结构

ScienceWorld 相关代码大致分 5 层:

1. 环境封装
   - `src/scienceworld_env.py`
2. prompt 构造
   - `prompts/scienceworld_prompts.py`
3. 失败检测
   - `src/scienceworld_failure_detector.py`
4. 记忆提取
   - `src/scienceworld_memory_extractor.py`
5. 实验入口
   - `experiments/run_scienceworld.py`

对应关系是:
- 环境负责 reset / step / task schedule
- prompt 负责把任务、历史、记忆拼成给 LLM 的输入
- failure detector 负责判断某一步是不是失败或无效交互
- memory extractor 负责从 episode 中抽取 failure-recovery 经验
- runner 负责把这些组件接起来执行完整实验

### 当前项目里如何使用这个 benchmark

这个仓库不是单纯跑一个 ScienceWorld baseline，而是把它作为 failure-recovery benchmark 来用。

核心假设是:
- ScienceWorld 任务长
- 中间失败很多
- 很多失败是局部可修复的

因此它特别适合验证:
- failure-triggered retrieval
- in-loop 注入记忆
- action-level repair 是否优于 episode-level 反思

### 为什么这个 benchmark 值得先搞清楚

后面很多实验现象都跟它的结构直接相关:
- 为什么 step limit 要比较长
- 为什么 failure checkpoint 多
- 为什么 rollout 需要 replay 多次
- 为什么 env score 不够，要加 LLM judge
- 为什么 ScienceWorld 上 in-loop memory 效果更明显

如果不先理解这些结构，后面看日志时很容易误判：
- 以为 agent 没进步，其实只是没触发 score 点
- 以为一次失败就说明方法不行，其实任务本身就允许中途试错

---

## 已确认环境信息

### Python 与包

- 工作目录: `/Users/zhoey/React_FM`
- Python: `3.13.9`
- `scienceworld` Python 包: 可成功 import

### Java

- 系统直接执行 `java -version` 失败，原因是 shell 的 `PATH` 里没有 Java
- 本机实际存在 Java:
  - 路径: `/opt/homebrew/opt/openjdk/bin/java`
  - 版本: `openjdk 25.0.2`
- 仓库代码中，ScienceWorld 相关脚本会主动把 `/opt/homebrew/opt/openjdk/bin` 加入 `PATH`

### ScienceWorld 本地环境

已验证:
- `scienceworld.jar` 可以启动
- ScienceWorld 环境在无沙箱限制下可以成功 `setup()` 和 `reset()`
- smoke test 成功加载任务:
  - task: `boil`
  - variation: `21`

已发现的执行约束:
- ScienceWorld 依赖 Java + Py4J 本地端口绑定
- 在受限沙箱中运行时，会因为本地端口绑定被拒绝而失败
- 典型报错:
  - `java.net.SocketException: Operation not permitted`

结论:
- 只要涉及 ScienceWorld 真正启动环境，通常都需要无沙箱限制执行
- 只做静态读文件、查文档、检查脚本参数时，不需要放开权限

---

## 已读文档结论

核心文档:
- `doc/scienceworld_gate_pipeline.md`
- `CLAUDE.md`

当前理解:
- ScienceWorld 有两条实验线
- 第一条是主实验线: baseline / React_FM
- 第二条是 gate 实验线: checkpoint -> branched rollout -> LLM judge -> gate training -> online eval

建议执行顺序:
1. 先跑通 ScienceWorld 最小主实验
2. 再决定是否进入 gate 流程

原因:
- gate 流程依赖主实验产物和更多离线数据
- 先确认主实验链路稳定，后续排错成本更低

---

## 与官方定义对齐后的结论

### 已识别的不对齐点

在这次核对之前，仓库里有两个容易混淆的地方:

1. topic 口径混用
   - 官方安装包 `tasks.json` 是 7 个 coarse topics
   - 仓库 `TASK_TOPIC_MAP` 为分析方便拆成了 10 个标签
2. ScienceWorld split 使用不充分
   - 多个实验脚本把 `split="test"` 写死
   - gate 训练是在 test 产生的 checkpoint 上再做内部切分
   - 这不符合 ScienceWorld 已经提供 `train/dev/test` 的官方数据组织方式

### 已完成的代码对齐

已修改的文件:
- `src/scienceworld_env.py`
- `experiments/run_scienceworld.py`
- `experiments/run_reflexion_scienceworld.py`
- `experiments/run_expel_scienceworld.py`
- `experiments/gate/collect_checkpoints.py`
- `experiments/gate/run_online_eval.py`

已完成的对齐内容:
- 在 `src/scienceworld_env.py` 明确区分:
  - 官方 30 个 task types
  - 官方 7 个 coarse topics
  - 仓库内部分析用的 10 个细粒度 labels
- 增加 `OFFICIAL_TASKS` 常量，表示官方任务全集
- 所有核心 ScienceWorld 入口脚本都支持:
  - `--split {train,dev,test}`
  - `--tasks ...`
  - `--max-variations N`

保持不变的部分:
- 默认行为仍然是仓库原有设置
- 默认仍使用 repo 的 10 个代表性任务子集
- 默认 split 仍是 `test`

这样做的原因是:
- 不破坏已有实验复现口径
- 但从现在开始，所有正式实验都可以按官方 split 明确指定数据来源

---

## 推荐数据协议

### 先区分两类实验

#### A. 主实验

这里指:
- ReAct baseline
- React_FM
- Reflexion
- ExpeL

这些实验的目标是:
- 报告最终任务完成效果
- 比较不同 agent 方法在同一 benchmark 上的表现

#### B. 模型训练实验

这里主要指:
- checkpoint 收集
- branched rollout
- LLM judge 打分
- gate 训练
- gate 的选择策略评估

这些实验的目标是:
- 学出一个 intervention policy
- 决定 failure 时是 `none / question / repair`

---

## ScienceWorld 数据如何使用

### 1. Debug / smoke test

建议使用:
- `dev` 或很小规模 `test`
- 1 到 3 个 task
- 每个 task 1 个 variation

用途:
- 验证 Java / API / prompt / 日志 / 落盘链路

原因:
- 不应该在 test 上反复调环境问题

### 2. 主实验开发阶段

建议使用:
- `dev`

用途:
- prompt 调整
- 步长设置
- failure detector 调整
- 记忆格式选择
- gate 超参数粗调

原因:
- 一旦你在一个 split 上反复观察结果并据此改方法，这个 split 就已经被“开发使用”了
- 因此不应该再把它当作最终汇报集

### 3. 主实验最终汇报

建议使用:
- `test`

用途:
- 最终报告 ReAct / React_FM / Reflexion / ExpeL 的成功率和平均分

建议做法:
- 在 `dev` 上定好配置
- 冻结 prompt、参数和任务子集
- 再在 `test` 上跑最终结果

### 4. Gate 训练数据

建议使用:
- `train`

用途:
- 收集 checkpoint
- 跑 branched rollout
- 跑 LLM judge
- 训练 gate 模型

原因:
- gate 是真正的数据驱动模型
- 既然官方已经提供 `train/dev/test`，就不应在 `test` 上采 checkpoint 再训练

### 5. Gate 模型选择 / 验证

建议使用:
- `dev`

用途:
- 选 `beta`
- 选 bypass threshold
- 选 checkpoint 采样策略
- 比较 `question` 和 `repair`
- 比较是否使用 LLM judge

### 6. Gate 最终在线评估

建议使用:
- `test`

用途:
- 用冻结后的 gate 在真实在线 episode 中做最终评估

标准流程应该是:
1. `train` 上采 checkpoint 和 rollout
2. `train` 上训练 gate
3. `dev` 上选超参数和分析误差
4. `test` 上只做一次最终在线评估

---

## 两种实验设置

### 设置 A: 复现仓库当前论文口径

特征:
- 使用 repo 默认的 10 个代表性 task
- 每个 task 最多 5 个 variations

适用场景:
- 先把当前仓库原实验跑通
- 与已有结果直接对比

优点:
- 成本低
- 和当前代码/文档最一致

风险:
- 不是 ScienceWorld 全量官方 benchmark
- 更准确地说，这是“官方 benchmark 上的固定子集评估”

### 设置 B: 严格按官方全集做

特征:
- 使用全部 30 个 task types
- 指定 `train/dev/test`
- 每个 split 跑该 split 下的全部 variations，或预先声明一致的采样规则

适用场景:
- 做更规范的 benchmark 论文设置
- 做和官方定义更一致的实验报告

优点:
- 数据协议最干净
- 最容易解释训练/验证/测试边界

代价:
- 成本显著更高
- 对 API、时间和日志管理要求更高

---

## 当前建议

如果目标是“先稳定跑通，再逐步规范化”，建议采用两阶段策略:

第一阶段:
- 先用设置 A 跑通整个链路
- 但从现在开始显式指定 `split`

第二阶段:
- 在 gate 训练和正式比较中切到官方 split 协议
- 即:
  - `train` 训练
  - `dev` 调参
  - `test` 汇报

这比继续把所有流程都堆到 `test` 上更合理，也更容易迁移到其他 benchmark。

---

## 面向本项目的数据选择建议

### 先明确项目目标

这个项目当前最适合的 ScienceWorld 目标，不是单纯追求“把所有官方样本都跑一遍”，而是要验证两件事:

1. failure-recovery memory 是否有效
2. gate 是否能在 failure 时学会选择 `none / question / repair`

所以数据选择应该优先服务于:
- 失败信号充足
- 局部修复有意义
- 轨迹足够长，能体现 in-loop intervention 的价值

而不是优先服务于:
- 最省成本
- 最短平快的分类题

### 我的核心建议

不要直接用“官方全部 30 个 task × 全部 variation”作为第一阶段主设置。

更合适的方案是:
- 用官方 `train/dev/test`
- 但先定义一个“核心任务集”
- 这个核心任务集覆盖多种失败类型和长时域实验结构
- 等方法稳定后，再扩展到官方全集

原因:
- 你的项目研究的是失败恢复，不是单轮识别
- ScienceWorld 各任务的结构差异很大
- 有些任务天然更能产生有价值的 checkpoint
- 有些任务更像识别/查找，不足以支撑 gate 学习

---

## 推荐的任务分层

### A. 核心训练集

这部分任务应该优先进入主实验开发和 gate 数据收集。

推荐保留:
- `boil`
- `melt`
- `use-thermometer`
- `power-component`
- `test-conductivity`
- `test-conductivity-of-unknown-substances`
- `grow-plant`
- `chemistry-mix`
- `lifespan-longest-lived-then-shortest-lived`
- `inclined-plane-friction-unnamed-surfaces`
- `mendelian-genetics-known-plant`
- `mendelian-genetics-unknown-plant`

选择理由:
- 有明显的多步过程
- 动作错误会累积
- 局部修复通常是可能的
- 更容易出现“方向对了但还没得分”的轨迹
- 更适合检验 memory 和 gate 的价值

### B. 对照任务集

这部分任务可以保留少量，用来证明方法不是在所有任务上都该强行介入。

建议只保留少量代表:
- `find-living-thing`
- `find-animal`
- `find-plant`

作用:
- 这些任务通常更短、更直接
- 可以作为负对照
- 如果 gate 在这些任务上倾向 `none`，反而说明策略合理

### C. 第二阶段扩展集

当核心流程稳定后，再考虑加上:
- `freeze`
- `change-the-state-of-matter-of`
- `measure-melting-point-known-substance`
- `measure-melting-point-unknown-substance`
- `chemistry-mix-paint-secondary-color`
- `chemistry-mix-paint-tertiary-color`
- `lifespan-longest-lived`
- `lifespan-shortest-lived`
- `identify-life-stages-1`
- `identify-life-stages-2`
- `inclined-plane-determine-angle`
- `inclined-plane-friction-named-surfaces`
- `power-component-renewable-vs-nonrenewable-energy`
- `grow-fruit`
- `find-non-living-thing`

这部分不是不重要，而是:
- 有些 variation 数量特别大，成本高
- 有些任务和核心任务高度相似
- 有些更偏识别/知识，不一定最能提供 gate 学习信号

---

## 推荐的数据协议

### 方案 1: 我最推荐的正式方案

用于你的项目当前阶段。

- `train`
  - 使用“核心训练集”任务
  - 每个 task 先设一个统一上限，例如 `20-40` 个 variations
  - 用于:
    - React_FM 训练期记忆积累
    - checkpoint 收集
    - branched rollout
    - gate 训练

- `dev`
  - 使用同一批核心任务
  - 每个 task 取较小但稳定的一批 variations，例如 `10-15`
  - 用于:
    - prompt 调整
    - failure detector 调整
    - gate 超参数选择
    - 分析哪类任务适合 `question`，哪类适合 `repair`

- `test`
  - 分两部分报告
  - 第一部分: 核心训练集对应的 test split
  - 第二部分: 加上少量对照任务

优点:
- 研究问题和数据高度对齐
- 成本可控
- 比仓库当前“固定 10 任务 test-only”更规范
- 比“30 任务全量全跑”更现实

### 方案 2: 追求 benchmark 完整性的方案

- `train/dev/test` 都使用官方全部 30 个 task
- 每个 split 全量或统一采样

优点:
- 最容易对外解释

缺点:
- 成本高很多
- 对当前项目未必最优
- 容易把大量弱信号任务混进来，稀释 failure-recovery 研究结论

我不建议你现在就上这个方案，除非你的目标是做非常严格的完整 benchmark 复现。

---

## 为什么不建议直接全量 30 任务起跑

主要有 4 个原因:

1. 任务异质性太强
   - ScienceWorld 里既有长时域实验任务，也有较短的查找/分类任务
   - 如果混在一起看平均数，很容易掩盖你方法真正擅长的 regime

2. Gate 学习需要高质量 failure signal
   - gate 不是只要数据多就行
   - 它更需要“失败后确实存在不同干预价值”的 checkpoint

3. 成本和时间会迅速膨胀
   - 特别是 branched rollout + LLM judge 阶段
   - variation 特别多的任务会把预算迅速拉高

4. 研究结论应该先稳，再扩
   - 先在最适合的问题上证明方法成立
   - 再讨论泛化到更广任务集

---

## 我给你的最终建议

如果按“最适合你当前项目”的标准，我建议:

1. 不用仓库当前 test-only 方案继续推进
2. 也不要立刻跑官方全量 30 任务全集
3. 采用“官方 split + 核心任务集”的折中方案

具体建议是:
- 主方法开发: `dev`
- gate 训练数据: `train`
- 最终结果: `test`
- 任务选择: 以长时域、实验型、动作型任务为主
- 短平快分类任务: 只保留少量作为负对照，不作为主要训练信号来源

如果你要我替你直接定一个第一版可执行清单，我建议第一版核心任务集就是这 12 个:
- `boil`
- `melt`
- `use-thermometer`
- `power-component`
- `test-conductivity`
- `test-conductivity-of-unknown-substances`
- `grow-plant`
- `chemistry-mix`
- `lifespan-longest-lived-then-shortest-lived`
- `inclined-plane-friction-unnamed-surfaces`
- `mendelian-genetics-known-plant`
- `mendelian-genetics-unknown-plant`

---

## ScienceWorld 正式实验协议 v1

这是当前建议采用的第一版正式协议，服务于本项目的主实验与 gate 实验。

### 目标

- 用官方 `train/dev/test` split
- 但不直接跑官方全量 30 任务
- 优先覆盖 failure-recovery 信号最强的任务
- 保持成本可控，同时让最终结果具备清晰解释性

### 任务集合

#### 核心任务集

共 12 个:

- `boil`
- `melt`
- `use-thermometer`
- `power-component`
- `test-conductivity`
- `test-conductivity-of-unknown-substances`
- `grow-plant`
- `chemistry-mix`
- `lifespan-longest-lived-then-shortest-lived`
- `inclined-plane-friction-unnamed-surfaces`
- `mendelian-genetics-known-plant`
- `mendelian-genetics-unknown-plant`

#### 对照任务集

共 3 个:

- `find-living-thing`
- `find-animal`
- `find-plant`

### variation 上限

统一采用:

- `train`: 每个核心任务最多 `20` 个 variations
- `dev`: 每个核心任务最多 `10` 个 variations
- `test`: 每个核心任务最多 `10` 个 variations
- `test` 对照任务: 每个任务最多 `10` 个 variations

如果某个 task 在对应 split 中不足这个数量，就取该 split 的全部可用 variations。

### 实际规模

#### 核心任务集实际规模

按当前 ScienceWorld variation 数量统计:

| Task | Train | Dev | Test |
|------|---:|---:|---:|
| boil | 14 | 7 | 9 |
| melt | 14 | 7 | 9 |
| use-thermometer | 20 | 10 | 10 |
| power-component | 10 | 5 | 5 |
| test-conductivity | 20 | 10 | 10 |
| test-conductivity-of-unknown-substances | 20 | 10 | 10 |
| grow-plant | 20 | 10 | 10 |
| chemistry-mix | 16 | 8 | 8 |
| lifespan-longest-lived-then-shortest-lived | 20 | 10 | 10 |
| inclined-plane-friction-unnamed-surfaces | 20 | 10 | 10 |
| mendelian-genetics-known-plant | 20 | 10 | 10 |
| mendelian-genetics-unknown-plant | 20 | 10 | 10 |

汇总:

- 核心 train: `214`
- 核心 dev: `107`
- 核心 test: `111`

#### 对照任务集实际规模

按每个任务 `test <= 10`:

- `find-living-thing`: 10
- `find-animal`: 10
- `find-plant`: 10

汇总:

- 对照 test: `30`

#### 最终测试总规模

- 核心 test: `111`
- 对照 test: `30`
- final reported test total: `141`

### 数据用途分配

#### 1. 主实验开发

使用:
- 核心任务集 `dev`

用途:
- prompt 调整
- failure detector 调整
- memory 格式确定
- agent 步长与注入策略选择

#### 2. 主实验最终汇报

使用:
- 核心任务集 `test`
- 对照任务集 `test`

汇报方式建议:
- 主表报告核心 test
- 附表或补充表报告对照 test

原因:
- 核心 test 体现方法优势区间
- 对照 test 用于说明该方法不是在所有任务上都应强干预

#### 3. Gate 数据收集与训练

使用:
- 核心任务集 `train`

用途:
- collect checkpoints
- branched rollout
- LLM judge scoring
- gate training

#### 4. Gate 选择与超参数验证

使用:
- 核心任务集 `dev`

用途:
- `beta`
- bypass threshold
- `question` vs `repair`
- checkpoint 采样策略

#### 5. Gate 最终在线评估

使用:
- 核心任务集 `test`
- 可选加对照任务 `test`

### 命令层面的执行原则

从现在开始，所有正式 ScienceWorld 实验都应显式写出:

- `--split`
- `--tasks`
- `--max-variations`

不要再依赖:
- 默认 `test`
- 默认 10 个任务子集

原因:
- 默认值适合快速复现
- 但不适合作为正式实验协议

### 这版协议的优点

- 比仓库当前 test-only 口径更规范
- 比官方 30 任务全量协议成本更可控
- 与 failure-recovery / gate 研究目标更匹配
- 后续可以平滑扩展成完整 benchmark 协议

### 后续扩展条件

当以下两点成立时，再考虑扩到官方 30 任务:

1. 核心任务集上的方法优势已经稳定
2. gate 训练和在线评估链路已经完全跑通

---

## 记忆隔离形式判断

这是当前 ScienceWorld 设计里非常关键的一步。

### 先看当前实现是什么

当前仓库的 `FailureMemoryStore` 采用的是:
- 按 `env_idx` 存储
- 按 `env_idx` 检索
- `task_type` 只是候选集里的二次过滤条件

也就是说，当前主索引不是:
- task type
- topic
- task family

而是:
- 当前 episode 的顺序编号

这在代码里表现为:
- `_env_entries: dict[int, list[FailureMemoryEntry]]`
- `retrieve(..., env_idx=..., cross_env=False)`

### 为什么这种方式不适合 ScienceWorld

对于 ScienceWorld，`env_idx` 本质上只是“第几个 episode”，而不是有语义的任务簇。

问题在于:
- Episode 1 的 `boil` 经验会被存到 `env_idx=1`
- Episode 2 再跑另一个 `boil` variation 时，会去查 `env_idx=2`
- 如果 `cross_env=False`，它根本读不到 Episode 1 的记忆

这意味着:
- 记忆写进去了
- 但跨 variation 的复用几乎不会发生

对于 ScienceWorld 这种“同一 task type 下有大量 variations”的 benchmark，这显然不是我们想要的行为。

### 三种候选隔离方式

#### 方案 A. 按 episode / env_idx 隔离

不推荐。

优点:
- 最保守
- 几乎不会引入错误迁移

缺点:
- 记忆几乎无法跨 episode 复用
- 在 ScienceWorld 上等于放弃同 task type variation 间的迁移
- 不符合当前项目的研究目标

结论:
- 不应作为 ScienceWorld 的正式方案

#### 方案 B. 按 task type 精确隔离

这是当前最推荐的默认方案。

例如:
- `boil` 的记忆只给 `boil`
- `melt` 的记忆只给 `melt`
- `test-conductivity` 的记忆只给 `test-conductivity`

优点:
- variation 间可以稳定复用
- 语义边界清楚
- 风险低
- 最容易解释实验结果

缺点:
- 会错过一些跨 task 的相似修复模式
- 例如 `known` / `unknown` 成对任务之间的共享

结论:
- 应作为 ScienceWorld 的主记忆存储与检索单位

#### 方案 C. 按 topic / 任务类别隔离

不建议直接作为唯一主方案。

原因:
- 官方 topic 太粗
- 同一 topic 内也可能存在相反或不兼容的修复动作

典型风险:
- `Matter` 里同时有 `boil`、`melt`、`freeze`
- 这些任务都和物态变化有关
- 但“该加热”与“该降温”并不通用

结论:
- topic 级别可以做辅助回退层
- 但不适合直接当主索引

#### 方案 D. 全局共享，不做任务隔离

不推荐用于当前 failure-recovery memory。

原因:
- 当前 memory 里保存的是很具体的失败动作和修复动作
- 这些内容强依赖对象、地点和实验步骤
- 全局共享很容易引入噪声注入

结论:
- 不适合作为 ScienceWorld 主方案

### 我的最终建议

对于 ScienceWorld，本项目应该采用:

1. 主存储单位: `task_type`
2. 主检索范围: 同 `task_type`
3. 不做 fallback
4. 不做全局 cross-task 共享

换句话说:
- 不是按 episode 隔离
- 不是按粗 topic 隔离
- 也不是全局随便共享
- 而是严格按“精确任务类型”隔离

### 为什么这是最合理的

因为 ScienceWorld 的 variation 结构天然支持这种设计:
- 同一个 `task_type` 下有很多 variations
- 这些 variations 往往共享目标结构和操作逻辑

例如:
- `boil` 的不同 variation 虽然物质和房间不同
- 但“寻找热源、放到容器、加热、等待”的模式高度稳定

这正是 failure-recovery memory 最适合复用的地方。

相反，跨太远的 task type 会有很大风险:
- `boil` 的修复经验不该直接注入给 `freeze`
- `find-animal` 的经验也不该主导 `chemistry-mix`

### Gold 轨迹带来的修正

在查看现有 gold trajectories 之后，这个结论应进一步收紧。

gold 轨迹说明:
- 同一 `task_type` 内部通常确实共享稳定的核心子程序
- 但跨 task 的“相似性”很多只是来自导航动作，例如:
  - `open`
  - `go to`
  - `look around`
  - `wait`
- 一旦去掉这些通用动作，很多 task 的真正操作逻辑差别很大

这意味着:
- `task_type` 级共享是合理的
- 但跨 task fallback 很容易因为“表面相似”而误共享

因此当前正式协议改为:
- 只做 exact `task_type` 检索
- 没有命中就返回 `none`
- 不做 task family fallback
- 不做 topic fallback
- 不做全局 cross-task fallback

### 对当前项目的直接结论

如果你问“ScienceWorld 该按什么粒度隔离记忆”，我的结论是:

- 正式实验默认:
  - 按 `task_type` 隔离
- 不推荐:
  - 按 `env_idx`
  - 按官方粗 topic 直接共享
  - 全局跨任务自由复用

### 这对后续实验意味着什么

后续应该增加一个新的设计步骤:

Step 2.5:
- 在完成数据选择后，定义 benchmark 的记忆作用域

对于 ScienceWorld，这一步的结论已经确定为:
- 数据选择单位: 核心任务集 / 对照任务集
- 记忆作用域单位: `task_type`
- 回退共享单位: 无

---

## 记忆作用域实现状态

### 当前决定

ScienceWorld 正式采用:
- 记忆按 `task_type` 隔离
- 不做 fallback
- 没命中直接返回 `none`

### 已完成的代码改造

已修改:
- `src/memory.py`
- `experiments/run_scienceworld.py`
- `experiments/gate/collect_checkpoints.py`
- `experiments/gate/run_online_eval.py`
- `experiments/gate/run_branched_rollouts.py`

改造内容:
- `FailureMemoryStore` 新增 `scope` 参数
  - 默认仍是 `env_idx`
  - ScienceWorld 显式使用 `task_type`
- ScienceWorld 主实验 memory store 改为 `scope="task_type"`
- ScienceWorld gate 的 checkpoint 收集、online eval、branched rollouts 也改为同样语义
- memory 文件保存时会记录 `scope`
- 重新加载后会保留原来的 scope

### 已完成验证

1. 静态验证
- `py_compile` 已通过

2. 作用域行为验证
- 在本地桩测试中:
  - 两条来自不同 `env_idx` 的 `boil` 记忆会进入同一个 `boil` bucket
  - `melt` 进入独立 bucket

3. 序列化验证
- `scope="task_type"` 保存到 JSON 后
- 重新加载仍保持:
  - `loaded_scope = task_type`
  - bucket keys 正确
  - `boil` bucket 计数正确

4. 端到端运行验证
- 重新运行了一个最小 ScienceWorld React_FM smoke test
- 主实验正常完成
- memory 文件正常写出
- 写出的 JSON 明确包含:
  - `"scope": "task_type"`

示例产物:
- `results/20260415_110936/sw_tasktype_scope_smoke_1.json`
- `memory_store/20260415_110936/sw_epoch1.json`

### 注意事项

- 旧的 ScienceWorld memory 文件如果是按 `env_idx` 存的，就仍然会按旧语义加载
- 因此后续正式实验应重新生成 ScienceWorld memory，不建议直接沿用旧文件

---

## 固定任务参数

为了避免后续命令重复手写，先固定两组任务参数。

### 核心任务集参数

```text
boil
melt
use-thermometer
power-component
test-conductivity
test-conductivity-of-unknown-substances
grow-plant
chemistry-mix
lifespan-longest-lived-then-shortest-lived
inclined-plane-friction-unnamed-surfaces
mendelian-genetics-known-plant
mendelian-genetics-unknown-plant
```

### 对照任务集参数

```text
find-living-thing
find-animal
find-plant
```

---

## 命令模板

以下模板是后续正式实验的基础命令。所有模板都显式指定:
- `--split`
- `--tasks`
- `--max-variations`

### 1. 主实验开发模板

用于开发阶段，建议跑 `dev`。

```bash
PATH=/opt/homebrew/opt/openjdk/bin:$PATH python experiments/run_scienceworld.py \
  --config config_scienceworld.yaml \
  --split dev \
  --tasks \
    boil melt use-thermometer power-component \
    test-conductivity test-conductivity-of-unknown-substances \
    grow-plant chemistry-mix \
    lifespan-longest-lived-then-shortest-lived \
    inclined-plane-friction-unnamed-surfaces \
    mendelian-genetics-known-plant mendelian-genetics-unknown-plant \
  --max-variations 10 \
  --run-name sw_dev_reactfm_
```

### 2. 主实验最终汇报模板

用于核心任务集 `test`。

```bash
PATH=/opt/homebrew/opt/openjdk/bin:$PATH python experiments/run_scienceworld.py \
  --config config_scienceworld.yaml \
  --split test \
  --tasks \
    boil melt use-thermometer power-component \
    test-conductivity test-conductivity-of-unknown-substances \
    grow-plant chemistry-mix \
    lifespan-longest-lived-then-shortest-lived \
    inclined-plane-friction-unnamed-surfaces \
    mendelian-genetics-known-plant mendelian-genetics-unknown-plant \
  --max-variations 10 \
  --run-name sw_test_reactfm_
```

### 3. 对照任务测试模板

用于短任务负对照。

```bash
PATH=/opt/homebrew/opt/openjdk/bin:$PATH python experiments/run_scienceworld.py \
  --config config_scienceworld.yaml \
  --split test \
  --tasks find-living-thing find-animal find-plant \
  --max-variations 10 \
  --run-name sw_test_control_
```

### 4. Gate checkpoint 收集模板

用于 `train` 上收集 checkpoint。

```bash
PATH=/opt/homebrew/opt/openjdk/bin:$PATH python experiments/gate/collect_checkpoints.py \
  --benchmark scienceworld \
  --config config_scienceworld.yaml \
  --split train \
  --tasks \
    boil melt use-thermometer power-component \
    test-conductivity test-conductivity-of-unknown-substances \
    grow-plant chemistry-mix \
    lifespan-longest-lived-then-shortest-lived \
    inclined-plane-friction-unnamed-surfaces \
    mendelian-genetics-known-plant mendelian-genetics-unknown-plant \
  --max-variations 20 \
  --output checkpoints/scienceworld_core_train_checkpoints.json \
  --run-name sw_cp_train_
```

### 5. Gate online eval 模板

用于冻结 gate 后在 `test` 上做在线评估。

```bash
PATH=/opt/homebrew/opt/openjdk/bin:$PATH python experiments/gate/run_online_eval.py \
  --benchmark scienceworld \
  --system learned-gate \
  --gate-model gate_models/scienceworld_gate \
  --config config_scienceworld.yaml \
  --split test \
  --tasks \
    boil melt use-thermometer power-component \
    test-conductivity test-conductivity-of-unknown-substances \
    grow-plant chemistry-mix \
    lifespan-longest-lived-then-shortest-lived \
    inclined-plane-friction-unnamed-surfaces \
    mendelian-genetics-known-plant mendelian-genetics-unknown-plant \
  --max-variations 10 \
  --output results/gate_eval_scienceworld_core \
  --run-name sw_gate_test_
```

---

## 当前建议的最小验证顺序

### Step 1

目标:
- 验证 `LLM API + ScienceWorld + 结果落盘` 这条最小闭环

建议命令:

```bash
PATH=/opt/homebrew/opt/openjdk/bin:$PATH python experiments/run_scienceworld.py \
  --config config_scienceworld.yaml \
  --baseline \
  --tasks boil \
  --max-variations 1 \
  --max-envs 1 \
  --step-limit 10 \
  --run-name sw_smoke_baseline_
```

说明:
- 只跑 1 个 task 的 1 个 variation
- 先用 baseline，减少记忆和提取链路带来的变量
- 这是最适合排通路的第一步

状态:
- 尚未完成
- 上一轮尝试在启动后被用户主动中断，未形成可复用结论

---

## 执行记录

### Step 3.1 最小 smoke test

目标:
- 验证“新协议下的 ScienceWorld 主实验链路”是否可运行
- 验证 `--split dev` 是否真正生效
- 验证环境、LLM、日志和结果落盘是否正常

执行命令:

```bash
PATH=/opt/homebrew/opt/openjdk/bin:$PATH python experiments/run_scienceworld.py \
  --config config_scienceworld.yaml \
  --baseline \
  --split dev \
  --tasks boil \
  --max-variations 1 \
  --max-envs 1 \
  --step-limit 10 \
  --run-name sw_smoke_dev_baseline_
```

结果:
- 成功启动 ScienceWorld JVM
- `split=dev` 生效
- 调度规模为 `1` episode
- 实际加载任务:
  - `task=boil`
  - `variation=14`
- episode 正常执行完 10 步
- 结果正常落盘

关键输出:
- 成功率: `0/1`
- 平均分: `0.00`
- token: `7988`
- 结果文件:
  - `results/20260415_103217/sw_smoke_dev_baseline_1.json`
- 日志文件:
  - `logs/20260415_103217/sw_smoke_dev_baseline.log`

观察:
- baseline 在这个 10-step smoke test 中未完成任务，这本身不是问题
- 这一步的目的不是看方法效果，而是验证实验链路

发现的问题:
- Py4J 在 setup / shutdown 阶段有 callback server 关闭异常
- 典型异常:
  - `Py4JNetworkError: Gateway is not connected.`
  - `AttributeError: 'NoneType' object has no attribute 'join'`

当前判断:
- 这些异常没有阻塞实验执行
- episode 仍然正常运行并保存结果
- 暂时将其记为“非阻塞稳定性噪声”
- 如果后续长实验中频繁造成资源泄漏或进程残留，再单独处理

结论:
- 新的 split-aware 协议已经被实际跑通
- 可以继续进入下一步:
  - `dev` 上的小规模 React_FM / baseline 对照
  - 或继续做更稳的 smoke test 扩展

对其他 benchmark 的启发:
- 当 benchmark 依赖 JVM / 子进程桥接时，先做最小 split-aware smoke test 是必要的
- 先验证 “新数据协议真的被脚本执行到了”，再谈方法效果

---

### Step 3.2 React_FM 对照 smoke test

目标:
- 在与 Step 3.1 相同设置下验证 React_FM 主链路
- 确认以下组件是否实际工作:
  - failure detection
  - judge token 统计
  - memory retrieval
  - post-episode memory save

执行命令:

```bash
PATH=/opt/homebrew/opt/openjdk/bin:$PATH python experiments/run_scienceworld.py \
  --config config_scienceworld.yaml \
  --split dev \
  --tasks boil \
  --max-variations 1 \
  --max-envs 1 \
  --step-limit 10 \
  --run-name sw_smoke_dev_reactfm_
```

结果:
- 成功启动 ScienceWorld JVM
- `split=dev` 生效
- 实际加载任务:
  - `task=boil`
  - `variation=14`
- episode 正常执行完 10 步
- 结果正常落盘
- memory store 文件正常保存

关键输出:
- 成功率: `0/1`
- 平均分: `0.00`
- token: `9022`
  - agent: `7893`
  - judge: `1129`
  - extractor: `0`
- failure detected: `1`
- retrievals: `1`
- retrieval hits: `0`
- memory entries saved: `0`
- 结果文件:
  - `results/20260415_103427/sw_smoke_dev_reactfm_1.json`
- memory 文件:
  - `memory_store/20260415_103427/sw_epoch1.json`
- 日志文件:
  - `logs/20260415_103427/sw_smoke_dev_reactfm.log`

关键观察:
- 第 7 步 `examine drawer` 被识别为:
  - `FAILURE detected: unproductive`
- 说明 failure detector + judge 已经实际参与运行
- 随后发生了 1 次 memory retrieval
- 但由于这是首个 episode，memory store 初始为空，因此 retrieval hit 为 0

关于 extractor:
- 本次 extractor token 为 `0`
- memory entries 最终也为 `0`
- 当前解释是:
  - 该 episode 没有形成可抽取的 failure-recovery pair
  - 因此 post-episode extractor 没有产出记忆

这在 smoke test 中是可接受的，不能据此判断 extractor 有 bug。

与 baseline 的对比:
- baseline:
  - failures = `0`
  - tokens = `7988`
- React_FM:
  - failures = `1`
  - judge tokens = `1129`
  - retrievals = `1`
  - memory hits = `0`

结论:
- React_FM 的主链路已经被最小规模实际验证:
  - agent
  - failure detector
  - judge
  - retrieval
  - memory save
- 下一步不需要再停留在单 episode smoke test
- 可以进入:
  - 小规模多任务 `dev` 实验
  - 或专门验证 extractor 在更长轨迹上是否能稳定产出记忆

对其他 benchmark 的启发:
- 对 memory-based agent，baseline smoke test 只能验证环境主链路
- 还必须补一个“方法本身”的 smoke test，确认失败检测、检索、记忆落盘都真的被触发

---

### Step 3.3 `task_type` 复用探针实验

目标:
- 验证把 ScienceWorld 记忆作用域改成 `task_type` 后
- 同一个 `task_type` 的不同 variation 之间，记忆是否开始真正复用

执行命令:

```bash
PATH=/opt/homebrew/opt/openjdk/bin:$PATH python experiments/run_scienceworld.py \
  --config config_scienceworld.yaml \
  --split dev \
  --tasks boil \
  --max-variations 3 \
  --max-envs 3 \
  --step-limit 15 \
  --run-name sw_tasktype_reuse_probe_
```

实际运行:
- `boil` dev split
- variations: `14`, `15`, `16`
- 共 3 个 episodes

结果:
- 总成功率: `0/3`
- 平均分: `1.67%`
- 总 token: `53046`
- failures detected: `7`
- retrieval attempts: `7`
- retrieval hits: `0`
- memory entries saved: `0`

结果文件:
- `results/20260415_111206/sw_tasktype_reuse_probe_1.json`

memory 文件:
- `memory_store/20260415_111206/sw_epoch1.json`

关键观察:
- 第 1、2 个 variation 都没有产出 memory
- 第 3 个 variation 出现了多次明确 failure:
  - `unproductive`
  - `unknown_action`
- 但即便如此，最终 memory 仍然是 `0`

明确阻塞点:
- extractor 在第 3 个 episode 中报错:
  - `Extractor JSON parse error`
  - 原因是返回 JSON 被 observation 中的多行内容打断，导致无法正确解析

结论:
- `task_type` 作用域已经生效，但这次实验还没有观察到“跨 variation 复用”
- 当前原因不是 memory scope 设计错误
- 而是 extractor 没有稳定地产出可存储记忆

这意味着下一步的优先级不是继续扩大 `task_type` 复用实验，而是:
- 优先修 ScienceWorld memory extractor 的稳健性

对后续实验设计的影响:
- 现在可以把“记忆作用域”视为已确定
- 当前真正的系统瓶颈转移到了:
  - extractor 输出格式稳定性
  - 尤其是带换行、枚举、歧义提示的 observation 解析

---

## Extractor 支线分析

### 现象

在 `boil × 3 variations` 的开发探针里:
- 第 3 个 episode 明确出现了多个 failure
- 但 episode 结束后没有产出任何 memory

日志里的关键错误是:
- `Extractor JSON parse error`
- `Unterminated string ...`

### 根因分析

不是单一问题，而是两个问题叠加。

#### 根因 1. detector 和 extractor 的 failure 定义不一致

ScienceWorld 运行时的 failure detector 可以把以下情况标成失败:
- `unproductive`
- `unknown_action`
- 其他显式失败

但原 extractor 只靠 observation 文本里的少数关键词判断“这条轨迹里有没有 failure”，例如:
- `no known action`
- `you can't`
- `nothing happens`

这会漏掉两类很重要的失败:
- judge 判出来的 `unproductive`
- 歧义交互，例如 `Ambiguous request ...`

结果:
- episode 里明明发生了失败
- extractor 却可能直接认为“没有值得提取的 failure”

#### 根因 2. extractor 输出 JSON 过于脆弱

原 prompt 要求模型输出:
- 完整 JSON
- 并把 `failure_observation` 原样写入

但 ScienceWorld 的 observation 经常包含:
- 多行文本
- tab
- 编号菜单
- 歧义提示

例如:
- `Ambiguous request: Please enter the number ...`
- 带 `0:`、`1:` 的菜单

模型很容易把这些内容原样塞回 JSON 字符串里，导致:
- 未转义换行
- 未转义引号
- 或输出过长后被截断

同时原 extractor 调用的 `max_tokens=512`，对于:
- 多个 failure
- 每个 failure 7 个字段

是偏紧的，进一步增加了截断风险。

### 已完成修复

已修改文件:
- `src/llm.py`
- `src/scienceworld_memory_extractor.py`
- `experiments/run_scienceworld.py`
- `experiments/gate/collect_checkpoints.py`
- `experiments/gate/run_online_eval.py`

修复内容:

1. 对齐 detector 和 extractor
- ScienceWorld post-episode 提取时，显式把运行中已检测到的 failures 传给 extractor
- extractor 不再只靠 observation 关键词猜测 failure

2. 收紧 extractor 输出格式
- 明确要求所有字段都是单行字符串
- 禁止原样复制菜单、tab、多行 observation
- 要求只输出简短 summary

3. 压缩 prompt 中的 observation
- 进入 extractor prompt 前，先把 multiline observation 归一化成单行

4. 提高 extractor 输出上限
- ScienceWorld extractor 调用从默认配置上再显式提高到 `max_tokens=1024`

5. 增强 parser 鲁棒性
- 支持去掉 code fence
- 尝试抽取最外层 JSON array 再解析
- 对疑似截断情况单独打 warning

### 已完成验证

本地桩验证已通过:

1. `detected_failures` 会被正确放进 prompt
2. extractor 调用已使用 `max_tokens=1024`
3. multiline / ambiguous observation 在 prompt 中会被压成单行
4. code fence 包裹的 JSON 可以被成功解析

### 当前状态

结论已经很明确:
- 为什么之前“明明有 failure 却没有 memory”已经找到了
- 修复也已经落地

但还缺一层最终确认:
- 再跑一次真实 ScienceWorld 小规模实验
- 观察 memory entries 是否从 `0` 变成 `>0`

前一次重跑在 JVM 启动阶段异常卡住，因此没有形成新的有效端到端验证结论。

### 2026-04-15 小规模端到端验证

目标:
- 用最小真实 ScienceWorld 运行验证 extractor 修复是否生效
- 核实两件事:
  1. 明确 failure 的 episode 是否会成功提取 memory
  2. 后续同 `task_type=boil` 的 episode 是否会实际召回这些 memory

执行命令:

```bash
PATH=/opt/homebrew/opt/openjdk/bin:$PATH python experiments/run_scienceworld.py \
  --config config_scienceworld.yaml \
  --split dev \
  --tasks boil \
  --max-variations 3 \
  --max-envs 3 \
  --step-limit 15 \
  --run-name sw_tasktype_reuse_probe_fix_
```

产物:
- 结果文件: `results/20260415_113117/sw_tasktype_reuse_probe_fix_1.json`
- 日志文件: `logs/20260415_113117/sw_tasktype_reuse_probe_fix.log`
- memory 文件: `memory_store/20260415_113117/sw_epoch1.json`

结果概览:
- 共跑 `3` 个 `boil` dev episodes
- 总成功数 `0/3`
- 平均分 `12.67%`
- memory 总计 `4` 条
- retrieval 总计 `4` 次，其中 hit `3` 次

逐 episode 结果:
- Env #1: `failures_detected=1`, `memories_stored=1`, `memories_retrieved=0`
- Env #2: `failures_detected=0`, `memories_stored=0`, `memories_retrieved=0`
- Env #3: `failures_detected=3`, `memories_stored=3`, `memories_retrieved=3`

关键日志证据:
- `Extractor found 1 failure-recovery patterns (tiered)`
- `Memory stored #0 [task_type=boil]`
- `Retrieved 1 memories`
- `Extractor found 3 failure-recovery patterns (tiered)`
- `Memory stored #1 [task_type=boil]`
- `Memory stored #2 [task_type=boil]`
- `Memory stored #3 [task_type=boil]`

memory 落盘核对:
- `scope = task_type`
- 仅有一个 bucket: `boil`
- bucket 内共有 `4` 条记忆

落盘内容样例:
- `examine drawer` 失败后，提取出 `open drawer → look around → examine drawer`
- `put down soap into metal pot` 失败后，提取出 `put down soap`
- `put down soap` 触发歧义菜单后，提取出需要回复编号 `1`

结论:
- extractor 修复已被真实 ScienceWorld 小实验验证为有效
- 之前“有明确 failure 但 memory=0”的问题已被修复
- 修复不仅让 post-episode extraction 成功，而且后续同 task_type episode 已经发生真实 retrieval hit
- 因此主线可以继续，不需要再保留 ScienceWorld extractor 为当前 blocker

注意:
- 这次运行末尾仍出现 Py4J callback shutdown 噪声
- 但不影响结果文件、memory 文件和提取行为，本次应视为有效实验

### 提取质量复核

这次小实验共落盘 `4` 条 `boil` memory，质量判断如下:

1. `examine drawer` -> `open drawer → look around → examine drawer`
- 质量: 低
- 问题: 这是明显的时间顺序错误。`open drawer` 和 `look around` 本来就在 failure 之前已经做过。
- 进一步说，这条轨迹里并不存在一个真正“修复了 examine drawer 失败”的 recovery。
- 判断: 属于 extractor 为了响应 `detected_failures` 强行拼出了一条伪 recovery。

2. `focus on soap on table` -> `pick up soap on table`
- 质量: 中低
- 问题: `You focus on the soap.` 本身不是环境报错，而是 detector 的 `unproductive` 判定。
- 提取结果虽然有一定启发性，但它不是严格意义上的 failure-recovery pair，更像“下一步更有效的动作建议”。
- 判断: 可作为弱启发，但不适合和明确语法失败的 memory 等价看待。

3. `put down soap into metal pot` -> `put down soap`
- 质量: 中
- 优点: 识别到了明确的 `unknown_action`，并给出更接近合法语法的修复。
- 问题: 这不是完整修复，因为下一步仍然进入了歧义菜单，原目标并未真正完成。
- 判断: 是局部修复，不应被表述成“问题已完全解决”。

4. `put down soap` -> `1`
- 质量: 中高
- 优点: 基本忠实于轨迹，确实抓住了“出现歧义菜单后需要选择编号”的修复方式。
- 问题: `repair_action = 1` 强依赖当前菜单编号，跨 variation 复用时有脆弱性。
- 判断: 是当前四条里最接近高质量的一条，但更适合作为“歧义时需选正确菜单项”的模式记忆，而不是把 `1` 当作通用动作。

总体判断:
- 提取覆盖率: 好，已经从 `0` 提升到能稳定产出 memory
- 提取精度: 一般，`4` 条里大约只有 `1` 条较扎实，`1` 条局部有效，另外 `2` 条偏弱或失真
- 当前主要问题已从“提取不出来”转为“提取语义不够干净”

对正式实验的含义:
- 当前版本足以继续主线实验，不必再把 extractor 当成 blocker
- 但如果后续要做严肃对比，建议把“memory 质量控制”列为下一个优化点

下一步最值得改的地方:
1. 对 `unproductive` 单独降级，不要与 `unknown_action`、`ambiguous request` 等显式失败同权存储
2. extractor 只允许把 failure 之后真实发生的动作写成 `solution_action`
3. 遇到歧义菜单时，优先存“被选中的完整动作文本”，不要把裸编号 `1` 直接当 `repair_action`

### 2026-04-15 第二轮小实验: 关闭隐式失败 + 严格 post-failure 修复约束

本轮修改:
- 关闭 ScienceWorld `unproductive` 隐式失败
- 只保留显式失败:
  - `unknown_action`
  - `action_failed`
  - `ambiguous request`
  - `the door is not open`
- extractor prompt 增加 `failure_step`
- extractor 只允许提取 failure 之后、且真实出现在轨迹里的修复动作

执行命令:

```bash
PATH=/opt/homebrew/opt/openjdk/bin:$PATH python experiments/run_scienceworld.py \
  --config config_scienceworld.yaml \
  --split dev \
  --tasks boil \
  --max-variations 3 \
  --max-envs 3 \
  --step-limit 15 \
  --run-name sw_tasktype_reuse_probe_strict4_
```

产物:
- 结果文件: `results/20260415_131836/sw_tasktype_reuse_probe_strict4_1.json`
- 日志文件: `logs/20260415_131836/sw_tasktype_reuse_probe_strict4.log`
- memory 文件: `memory_store/20260415_131836/sw_epoch1.json`

对照结果:

旧版本:
- `total_entries = 4`
- `total_retrievals = 4`
- `total_hits = 3`

新版本:
- `total_entries = 0`
- `total_retrievals = 1`
- `total_hits = 0`

逐 episode:
- Env #1: 旧版 `failures_detected=1, memories_stored=1` -> 新版 `failures_detected=0, memories_stored=0`
- Env #2: 旧版 `failures_detected=0, memories_stored=0` -> 新版 `failures_detected=0, memories_stored=0`
- Env #3: 旧版 `failures_detected=3, memories_stored=3` -> 新版 `failures_detected=1, memories_stored=0`

关键观察:
- `examine drawer` 这类只是“没什么进展”的动作，不再被当作 failure
- `focus on soap on table` 也不再被当作 failure
- Env #3 中真正保留下来的显式失败是:
  - `go to kitchen` -> `The door is not open.`
- 这条 failure 后面轨迹里确实有真实修复:
  - `open door to kitchen`
  - `go to kitchen`
- 但 extractor 最终仍返回:
  - `Extractor found 0 failure-recovery patterns (tiered)`

结论:
- 这次改动显著提升了 precision
- 之前那些明显失真的伪 memory 被成功清掉了
- 但 recall 掉得过头，连 `go to kitchen -> The door is not open -> open door to kitchen -> go to kitchen` 这种明确 recovery 都没有提取出来
- 因此当前状态不是“坏了”，而是“约束正确，但 extractor 现在过严或 prompt 没有把显式 recovery 抓出来”

下一步判断:
- 方向是对的，应该保留“关闭隐式失败”
- 也应该保留“修复动作必须真实出现在 failure 之后”
- 但需要继续修 extractor 本身，让它能稳定抓到显式且短链条的 repair
- 最值得优先处理的 case 就是:
  - `The door is not open.` -> `open door to <room> → go to <room>`

### 显式失败规则的整体合理性判断

这一步不再只看单个小实验，而是结合:
- ScienceWorld gold 轨迹结构样本: `rollouts/scienceworld_gold_paths.json`
- 真实 ScienceWorld 运行日志: `logs/20*/sw*.log`
- 当前 detector / extractor 实现

说明:
- 当前 gold 轨迹文件只覆盖了本项目已抽取的 `24` 条 task-variation 记录，不是官方全量 `30` 个 task type
- 但它足够反映 ScienceWorld 的典型交互结构

结构证据:
- gold 样本中高频动作依次是:
  - `open: 183`
  - `go: 147`
  - `use: 130`
  - `examine: 129`
  - `look: 108`
  - `pick up: 54`
  - `focus: 27`
- 其中明确出现了 `147` 次 `open door to X -> go to X`

这说明:
- ScienceWorld 的任务普遍依赖前置条件
- “门必须先打开再移动”
- “必须先拿到物品再 use / move / pour”
- “歧义对象需要 disambiguate”

真实日志证据:
- 已明确观察到的 ScienceWorld 失败语句包括:
  - `No known action matches that input.`
  - `Ambiguous request: Please enter the number for the action you intended`
  - `The door is not open.`
  - `You can't pick up a liquid directly. Try pouring it from one container to another...`

结论:
- 当前显式失败规则作为 runtime failure detection，整体是合理的
- 它已经覆盖了 ScienceWorld 中最重要的一批“可操作失败”:
  1. 非法动作语法
  2. 对象歧义
  3. 门未打开等前置条件失败
  4. 物理约束类失败，例如液体不能直接 pick up

但它还不完全适合作为 memory 候选筛选器，原因有两点:

1. `action_failed` 这个桶太粗
- 当前下面这些 observation 都会被压成同一个 `failure_type=action_failed`:
  - `The door is not open.`
  - `You can't pick up a liquid directly...`
  - `Ambiguous request...`
  - `You don't have ...`
- 对 detector 来说这没问题
- 对 memory/extractor 来说语义差异太大，最好拆开

2. 不是所有显式失败都适合进 memory
- `empty_obs` 更像环境桥接异常
- `action_loop` 更像 agent 卡住信号
- 这两类更适合 runtime 监控，不适合直接抽 failure-recovery memory

因此更合理的分层应该是:
- memory-eligible:
  - `unknown_action`
  - `ambiguous_request`
  - `door_not_open`
  - `missing_inventory`
  - `invalid_physical_interaction`
- runtime-only:
  - `empty_obs`
  - `action_loop`

### 为什么小实验里“抓到了 failure，但没有 memory”

这次结论已经比较明确:
- 核心问题不在 detector
- 核心问题在 extractor

证据 1:
- 在严格版小实验 `sw_tasktype_reuse_probe_strict4_1.json` 中
- Env #3 明确检测到了:
  - `go to kitchen`
  - `The door is not open.`
  - `failure_type = action_failed`
- 但 extractor 返回 `0` 条 memory

证据 2:
- 在后续 prompt/few-shot 修订后的重跑中
- Env #1 也出现了非常标准的 repair 链:
  - `go to hallway`
  - `The door is not open.`
  - `open door to the hallway`
  - `go to hallway`
- 但 extractor 仍返回:
  - `Extractor found 0 failure-recovery patterns (tiered)`

这说明:
- failure 已经被 detector 正确抓到
- failure 之后也确实存在真实修复动作
- 但 extractor 仍没有把它抽出来

进一步判断:
- 而且日志里没有出现 `Dropped invalid ScienceWorld recovery...`
- 所以更像是 LLM 在 extractor 阶段就直接输出了 `[]`
- 而不是“先提取出来，再被后处理过滤掉”

根因:
- 当前 extractor 虽然已经开始支持“precondition fix”
- 但它的整体 few-shot 仍主要在教模型处理:
  - 非法语法改写
  - 对象/容器表达改写
- 对于这种最短链条的 precondition repair:
  - `door not open -> open door -> retry move`
- 召回仍然不稳定

当前判断:
- 显式失败规则本身，大方向合理
- “failure detected but memory=0” 的主因是 extractor recall 不足
- 下一步应优先把 extractor 对短链条 precondition repair 的召回做稳

### 2026-04-15 中粒度 failure taxonomy 决策

担心:
- 如果直接把 `action_failed` 拆成很多非常具体的小类，容易围绕单个小实验过拟合

本次决策:
- 不按单条字符串细拆
- 改为按“修复机制”做中粒度拆分

新的 ScienceWorld failure types:
- `syntax_or_parse`
  - 典型 observation:
    - `No known action matches that input.`
    - `I'm not sure what you mean.`
- `ambiguity`
  - 典型 observation:
    - `Ambiguous request: ...`
    - `Please enter the number for the action you intended`
- `precondition_blocked`
  - 典型 observation:
    - `The door is not open.`
    - `You don't have ...`
- `physics_or_affordance`
  - 典型 observation:
    - `You can't pick up a liquid directly...`
- `no_effect_or_other`
  - 典型 observation:
    - `Nothing happens.`
    - `That doesn't seem to work.`

保留的非 memory 样式 failure:
- `empty_obs`
- `action_loop`
- `unproductive` 目前对 ScienceWorld 仍保持关闭

这样拆的原因:
- 比 `action_failed` 更有语义
- 但又不会像 `door_not_open` / `missing_inventory` / `liquid_pickup_blocked` 那样过细
- 更接近跨任务可复用的 repair 模式，而不是单个 phrase patch

快速校验结果:
- `The door is not open.` -> `precondition_blocked`
- `You can't pick up a liquid directly...` -> `physics_or_affordance`
- `Ambiguous request...` -> `ambiguity`
- `No known action matches that input.` -> `syntax_or_parse`
- `Nothing happens.` -> `no_effect_or_other`

### 2026-04-15 跨任务验证与关键实现问题

为了避免围绕 `boil` 单任务做局部优化，额外做了一个跨任务小实验:
- 任务: `boil`, `use-thermometer`, `chemistry-mix`, `power-component`, `grow-plant`
- 设置: `dev`, 每任务最多 `2` 个 variation, `step_limit=18`
- 结果文件: `results/20260415_135031/sw_multitask_failurecheck_1.json`
- memory 文件: `memory_store/20260415_135031/sw_epoch1.json`

这轮验证暴露出一个比 prompt 更关键的实现问题:

根因:
- ScienceWorld 在 post-episode 提取时，传给 extractor 的 `trajectory` 是删掉 `think` 后重新编号的
- 但 `detected_failures` 仍保留原始 episode step 编号
- 导致 failure step 与 trajectory step 错位

直接后果:
- extractor / validator 会把真正的 failure 错配到后续修复动作上
- 典型错误 memory:
  - `failure_action = open door to kitchen`
  - `failure_observation = The door is now open.`
  - `solution_action = go to kitchen`
- 这其实本应对应:
  - `failure_action = go to kitchen`
  - `failure_observation = The door is not open.`
  - `solution_action = open door to kitchen -> go to kitchen`

结论:
- 之前一部分 “memory 质量差” 不是纯 prompt 问题
- 而是 failure / trajectory 对齐 bug

### 2026-04-15 对齐修复

已修复:
- `experiments/run_scienceworld.py`
- `experiments/gate/collect_checkpoints.py`
- `experiments/gate/run_online_eval.py`

修复方式:
- ScienceWorld extractor 不再使用删掉 `think` 后重新编号的 `(action, observation)` tuple 轨迹
- 改为传入带原始 `step` 编号的 `env_history_records`
- 从而让 `failure_step`、`failure_action` 和真实轨迹动作保持一致

### 2026-04-15 对齐修复后的跨任务复验

复验设置:
- 任务: `boil`, `use-thermometer`, `chemistry-mix`, `grow-plant`
- 设置: `dev`, 每任务最多 `2` 个 variation, `step_limit=18`
- 结果文件: `results/20260415_140554/sw_multitask_alignfix_1.json`
- memory 文件: `memory_store/20260415_140554/sw_epoch1.json`

复验结果:
- 共 `8` 个 episodes
- `11` 条 memory
- `19` 次 retrieval
- `12` 次 hit

失败类型分布:
- `syntax_or_parse`: `10`
- `precondition_blocked`: `6`
- `ambiguity`: `3`

说明:
- 中粒度 taxonomy 在跨任务运行中确实分化出了有意义的 failure 类型
- 没有再次退化成单一 `action_failed`

修复后的正向证据:
- `boil`:
  - `failure_action = go to hallway`
  - `failure_observation = The door is not open.`
  - `solution_action = open door to the hallway -> go to hallway`
- `use-thermometer`:
  - `failure_action = go to kitchen`
  - `failure_observation = The door is not open.`
  - `solution_action = open door -> go to kitchen`
- `chemistry-mix`:
  - `failure_action = read recipe titled instructions to make smores`
  - `solution_action = read instructions to make smores`
  - `failure_action = use sink`
  - `solution_action = activate sink`

这说明:
- 错位 bug 已经基本修掉
- failure / observation / solution 的对应关系明显比上一轮正确

当前剩余问题:

1. extractor 仍会生成“局部修补 memory”
- 例如:
  - `put down marshmallow in glass cup` -> `put down marshmallow`
- 这只是把非法命令变成合法命令
- 但未必真的完成原目标

2. ambiguity / syntax 情况下会出现过度记忆化
- 例如 `use-thermometer` 中:
  - `open door` -> `open door to the kitchen`
  - `open door between kitchen and hallway` -> `open door to the kitchen`
- 这些条目不算完全错误
- 但上下文依赖强，泛化价值一般

3. retrieval 已经开始命中，但相关性仍不够稳
- 例如 `grow-plant` 中 `syntax_or_parse` 失败也可能召回“开门”类 memory
- 说明 taxonomy 改善了 failure labeling
- 但 retrieval query 与 memory ranking 还没有完全对齐 failure mechanism

当前判断:
- failure taxonomy 改动是合理的，不是局部最优 patch
- extractor 之前的最大问题之一其实是 step 对齐 bug，而不是纯 prompt 不够好
- 对齐修复后，memory 的基础正确性明显提升
- 但现在还没有达到“高质量可放心大规模实验”的状态
- 当前 blocker 已从“错位 memory”转为“局部修补 memory 和相关性噪声”

### 2026-04-15 当前已知小问题: 部分提取记忆质量不高

现阶段需要记录一个明确但不阻塞主线的小问题:
- 某些 ScienceWorld memory 已经能够被成功提取、存储和检索
- 但其中一部分提取结果仍然是“局部修补型”经验，质量不够高

目前观察到的典型现象:
- 某些 memory 没有提取到真正可复用的解决方法
- extractor 有时会把 failure 之后的局部交互片段，当成完整 repair
- 这类 memory 在运行时可能会被正常检索出来，但对后续决策帮助有限

当前判断:
- 这不是主检索链路失效
- 更像是 extractor 对“什么才算真正解决问题的方法”仍然不够稳定

当前处理策略:
- 先把它记录为已知小问题
- 不把它作为当前主线 blocker
- 后续如果需要，再单独补 extractor 的后置质量过滤

---

### Step 5.1 Gate checkpoint 收集 smoke test

目标:
- 不复用旧 gate 产物
- 从 gate 流程起点重新验证 ScienceWorld checkpoint 收集链路
- 确认两件事:
  - checkpoint 收集阶段实际使用什么数据
  - 该阶段累计的 memory 在后续 branched rollout 中是否会被使用

执行命令:

```bash
PATH=/opt/homebrew/opt/openjdk/bin:$PATH python experiments/gate/collect_checkpoints.py \
  --benchmark scienceworld \
  --config config_scienceworld.yaml \
  --split train \
  --tasks boil \
  --max-variations 1 \
  --max-envs 1 \
  --output checkpoints/scienceworld_cp_smoke_train.json \
  --run-name sw_cp_smoke_train_
```

结果:
- 需要在沙箱外运行，原因是 ScienceWorld 的 JVM / Py4J 桥接在沙箱内会卡住启动阶段
- `split=train` 生效
- 实际加载任务:
  - `task=boil`
  - `variation=0`
- 共运行 `1` 个 episode
- 成功保存 `1` 个 checkpoint
- post-episode extractor 成功提取并保存 `1` 条 memory

产物:
- checkpoint 文件:
  - `checkpoints/scienceworld_cp_smoke_train.json`
- memory 文件:
  - `memory/scienceworld_cp_smoke_train.json`
- 日志文件:
  - `logs/20260415_154758/sw_cp_smoke_train.log`

对第一个问题的结论: checkpoint 收集阶段用什么数据
- 正式协议应使用官方 `train` split
- 对 ScienceWorld gate 来说，checkpoint 收集、branched rollout、LLM judge 和 gate training 都属于训练数据准备阶段，不应使用 `test`
- 本次 smoke test 只是最小验证:
  - `train`
  - `boil`
  - `1` 个 variation
  - `1` 个 episode
- 正式运行时应切回 runbook 里已经定义的核心任务集协议:
  - 核心任务集
  - `split=train`
  - 每 task 最多 `20` 个 variations

对第二个问题的结论: 这里累计的 memory 后面 rollout 会不会用
- 会，用法是“作为 branched rollout 的 memory store 输入”
- `collect_checkpoints.py` 结束时除了保存 checkpoint，还会自动把同轮运行中累计出的 memory 另存为一个 JSON 文件
- 后续 `run_branched_rollouts.py` 会显式加载这份 memory 文件
- 对每个 checkpoint，rollout 阶段会先根据:
  - `failure_action`
  - `failure_observation`
  - `task_type`
  在 memory store 中做检索
- 检索到的 memory entry 会被用于构造:
  - `question` arm 的诊断问题
  - `repair` arm 的修复提示

需要特别澄清:
- 后续 rollout 不是直接“读取 checkpoint 里缓存的一段提示文本”
- checkpoint 里保存的是:
  - failure 上下文
  - 当时检索分数
  - top-1 memory id
- branched rollout 真正注入时，会优先按 `retrieved_memory_id` 找 entry
- 如果该 id 不可用，仍会根据 checkpoint 的 failure 上下文重新检索
- 因此 rollout 确实依赖 checkpoint 收集阶段产出的 memory 文件，而不是只依赖 checkpoint JSON 本身

结论:
- gate 流程已经可以从“checkpoint 收集”这一步重新开始推进
- 数据口径已经明确:
  - gate 训练数据使用 `train`
- memory 依赖关系也已经明确:
  - checkpoint 收集阶段累计的 memory 会进入后续 branched rollout
  - 它是 `question / repair` 两个干预臂的直接输入来源

关于“新版 failure detector 会不会影响 checkpoint / 去重”的结论:
- 会，而且影响是结构性的，不只是标签名字变化
- 当前 ScienceWorld detector 已改为:
  - 保留显式失败
  - 使用更细的中粒度 failure taxonomy
  - 关闭隐式失败 `unproductive`
- 这会直接改变 checkpoint 收集阶段哪些 step 会被记为 failure，因此:
  - checkpoint 总数会变化
  - checkpoint 的 failure type 分布会变化
  - checkpoint 出现的 episode 位置也可能变化
- 这也会直接影响去重，因为当前 dedup key 是:
  - `failure_type | task_type | failure_action前3词`
- 因此 failure type 从粗粒度改成细粒度后:
  - 原来会被压进同一个桶的 checkpoint，现在可能会分散到多个桶
  - 原来由 `unproductive` 形成的一批桶，现在会直接消失
- 进一步地，这还会影响 gate 训练分布，因为 `failure_type` 本身就是 gate 的输入特征之一
- 所以一旦确认继续使用这版新版 detector，就不应再混用旧口径下已经生成的:
  - checkpoints
  - rollout
  - judged rollout
  - gate analysis / gate model
- 更合理的做法是:
  - 从 checkpoint 收集开始整条 gate 数据链重新生成
  - 保持 detector 口径、dedup 口径和 gate 特征口径一致

对其他 benchmark 的启发:
- gate 流程的 smoke test 不能只看 checkpoint 有没有落盘
- 还必须同时确认:
  - checkpoint 用的是不是正确 split
  - 同轮运行累计出的 memory 有没有被明确设计为后续 rollout 输入
- 否则很容易出现“checkpoint 有了，但 question / repair 实际没有可用记忆”的伪启动状态

---

### Step 5.2 ScienceWorld core-train checkpoint 收集完成

目标:
- 按新版 ScienceWorld detector 口径，重新生成 gate 流程使用的 fresh checkpoint 数据
- 明确使用官方 `train` split，而不是复用旧的 `test` 口径或旧 gate 产物
- 同时生成后续 branched rollout 会使用的 fresh memory 文件

执行命令:

```bash
PATH=/opt/homebrew/opt/openjdk/bin:$PATH python experiments/gate/collect_checkpoints.py \
  --benchmark scienceworld \
  --config config_scienceworld.yaml \
  --split train \
  --tasks \
    boil melt use-thermometer power-component \
    test-conductivity test-conductivity-of-unknown-substances \
    grow-plant chemistry-mix \
    lifespan-longest-lived-then-shortest-lived \
    inclined-plane-friction-unnamed-surfaces \
    mendelian-genetics-known-plant mendelian-genetics-unknown-plant \
  --max-variations 20 \
  --output checkpoints/scienceworld_core_train_checkpoints.json \
  --run-name sw_cp_train_
```

结果:
- 运行完成时间:
  - `2026-04-16 07:32:34`
- 本轮 fresh Stage 0 数据准备已经完成
- 最终产物:
  - `checkpoints/scienceworld_core_train_checkpoints.json`
  - `memory/scienceworld_core_train_memory.json`
- 汇总结果:
  - `Total checkpoints = 1868`
  - `Unique signatures = 398`
  - `Memory entries = 635`
  - memory bucket 数 = `12 task_type buckets`

阶段判断:
- 对照 `doc/scienceworld_gate_pipeline.md`
- 当前这条新实验链已经完成:
  - `Stage 0: 数据准备`
- 还未开始:
  - `Stage 1: 分支 Rollout`
  - `Stage 1.5: LLM Judge 评分`
  - `Stage 2: Gate 训练`
  - `Stage 3: 在线评估`

为什么这一步可以视为 Stage 0 完成:
- checkpoint 收集已经按新版 detector 口径重跑
- fresh memory 也已经同步生成
- 后续 branched rollout 所需的两类输入都已具备:
  - checkpoint 文件
  - memory 文件

需要特别说明:
- 本地虽然存在旧的 rollout / judged rollout / gate model 文件
- 但它们不是基于这次新版 detector + fresh checkpoint 链条生成的
- 因此不应把旧产物视为当前这条新实验链已经完成 Stage 1 或 Stage 2

当前遗留小问题:
- ScienceWorld checkpoint 中还没有原生写入 `env_state.score_at_last_failure`
- 这会导致 gate 第 8 维特征 `progress_since_last_failure` 在 offline 训练时存在轻微上下文缺失
- 该问题可以在当前已生成的 checkpoint 文件上做 backfill 修复，不需要重跑本轮 checkpoint 实验

下一步:
1. 对 `scienceworld_core_train_checkpoints.json` 做 `score_at_last_failure` backfill
2. 再做 dedup / 采样，准备 Stage 1 branched rollout 输入
3. 然后正式进入 `doc/scienceworld_gate_pipeline.md` 的 Stage 1

对其他 benchmark 的启发:
- 只要 detector 口径发生明显变化，checkpoint、dedup、gate 特征分布都会跟着变化
- 这种情况下更稳妥的做法是从 Stage 0 重新生成 fresh checkpoint 和 memory
- 不要把旧 detector 口径下的 rollout / judged rollout / gate model 混进新链条

---

### Step 5.3 Backfill `score_at_last_failure`

目标:
- 修复刚生成的 ScienceWorld checkpoint 文件中缺失的 `env_state.score_at_last_failure`
- 让 offline gate 特征与 online evaluation 的第 8 维特征语义对齐
- 不重跑刚完成的 checkpoint 收集实验

背景:
- `src/gate/features.py` 的第 8 维特征 `progress_since_last_failure`
  依赖 `env_state.score_at_last_failure`
- 刚完成的 `scienceworld_core_train_checkpoints.json` 中，这个字段原本是缺失的
- 该信息可以由现有 checkpoint 序列恢复:
  - 同一 `env_idx` 内按 `step_idx` 排序
  - 当前 checkpoint 的 `score_at_last_failure`
    等于前一个 checkpoint 的 `score_at_checkpoint`
  - 每个 env 的第一个 checkpoint 取 `0.0`

已做修改:
- `experiments/gate/collect_checkpoints.py`
  - ScienceWorld 后续新收集的 checkpoint 现在会原生写入 `score_at_last_failure`
- 新增:
  - `experiments/gate/backfill_score_at_last_failure.py`
  - 用于对旧 checkpoint JSON 做一次性回填

执行命令:

```bash
python experiments/gate/backfill_score_at_last_failure.py \
  --input checkpoints/scienceworld_core_train_checkpoints.json
```

结果:
- 已完成对:
  - `checkpoints/scienceworld_core_train_checkpoints.json`
  的回填
- 回填结果:
  - `1868 / 1868` 个 checkpoints 均已写入 `env_state.score_at_last_failure`
  - 其中非零值 checkpoint 数为 `1529`
- 抽样核对:
  - `cp_0000`: `score_at_checkpoint=0`, `score_at_last_failure=0`
  - `cp_0001`: `score_at_checkpoint=70`, `score_at_last_failure=0`
  - `cp_0002`: `score_at_checkpoint=70`, `score_at_last_failure=70`

产物:
- 回填后的 checkpoint 文件:
  - `checkpoints/scienceworld_core_train_checkpoints.json`
- 备份文件:
  - `checkpoints/scienceworld_core_train_checkpoints.json.bak`

注意:
- 第一次运行 backfill 脚本时，`.bak` 备份生成顺序有一个小问题:
  - 备份文件也写成了回填后的版本
- 这个问题已经在脚本中修复
- 对当前主线没有实质影响，因为主文件已经正确回填
- 但如果后续需要保留“回填前原始版本”，应重新从外部副本恢复，而不能依赖当前这份 `.bak`

结论:
- Stage 0 产物现在已经满足进入 Stage 1 的最小要求:
  - checkpoint 文件已齐备
  - memory 文件已齐备
  - `score_at_last_failure` 已补齐
- 因此下一步可以继续:
  - dedup / checkpoint 采样
  - Stage 1 branched rollout

对其他 benchmark 的启发:
- 如果某个 offline 特征所需的上下文可以由 checkpoint 序列恢复，就优先做 backfill，而不是直接重跑整轮实验
- 这样可以把“修特征一致性”与“重做数据采集”分离，节省时间和成本

---

### Step 5.4 Checkpoint dedup 完成

目标:
- 先对 fresh ScienceWorld checkpoint 池做去重
- 在不丢失 failure signature 覆盖面的前提下，压掉高频重复 checkpoint
- 为后续 Stage 1 branched rollout 降低成本

当前采用的 dedup 规则:
- checkpoint 的 `failure_signature` 定义为:
  - `failure_type | task_type | failure_action前3词`
- 每个 signature 最多保留 `3` 条
- 这与当前 `run_gate_training.py` 的默认 `--max-per-sig 3` 保持一致

新增脚本:
- `experiments/gate/dedup_checkpoints.py`

执行命令:

```bash
python experiments/gate/dedup_checkpoints.py \
  --input checkpoints/scienceworld_core_train_checkpoints.json \
  --output checkpoints/scienceworld_core_train_checkpoints_deduped.json \
  --max-per-signature 3
```

结果:
- raw checkpoint:
  - `1868`
- dedup 后:
  - `790`
- unique signatures:
  - `398 -> 398`
- 说明:
  - dedup 没有丢失 signature 覆盖面
  - 只是压掉了高频重复样本

产物:
- dedup 后 checkpoint 文件:
  - `checkpoints/scienceworld_core_train_checkpoints_deduped.json`

分布观察:
- raw failure types:
  - `syntax_or_parse = 1513`
  - `ambiguity = 171`
  - `precondition_blocked = 160`
  - `action_loop = 21`
  - `physics_or_affordance = 3`
- dedup 后 failure types:
  - `syntax_or_parse = 592`
  - `precondition_blocked = 89`
  - `ambiguity = 89`
  - `action_loop = 17`
  - `physics_or_affordance = 3`

结论:
- dedup 显著降低了数据规模:
  - `1868 -> 790`
- 但仍然保留了全部 `398` 个 unique signatures
- 当前 checkpoint 池仍然明显偏向 `syntax_or_parse`
- 这说明 dedup 解决的是“重复过多”的问题
- 但还没有解决“哪些 checkpoint 最值得进入 branched rollout”的问题
- 因此下一步仍需要:
  - checkpoint 采样

当前阶段判断:
- 现在已经完成:
  - Stage 0 完成
  - Stage 0 后置修复（score_at_last_failure backfill）
  - dedup
- 还未开始:
  - checkpoint 采样
  - Stage 1 branched rollout

对其他 benchmark 的启发:
- dedup 的作用是限制重复样本，不是替代采样
- 如果 raw 数据强烈偏向某一类 failure，dedup 之后仍可能保留这种偏向
- 因此大多数情况下仍需要在 dedup 之后单独做采样

---

### Step 5.5 Checkpoint 采样完成

目标:
- 从 dedup 后的 checkpoint 池中选出一个可控规模的 Stage 1 输入子集
- 保持 task 覆盖尽量均衡
- 优先保留更适合作 branched rollout 的 midband checkpoint

当前采用的采样规则:
- 输入:
  - `checkpoints/scienceworld_core_train_checkpoints_deduped.json`
- 只保留 midband checkpoint:
  - `score_at_checkpoint ∈ [1, 99]`
- 按 `task_type` 平衡采样
- 每个 task 最多取 `8` 条
- task 内优先级:
  1. `20-79`
  2. `1-19`
  3. `80-99`

新增脚本:
- `experiments/gate/sample_checkpoints.py`

执行命令:

```bash
python experiments/gate/sample_checkpoints.py \
  --input checkpoints/scienceworld_core_train_checkpoints_deduped.json \
  --output checkpoints/scienceworld_core_train_checkpoints_sampled.json \
  --max-per-task 8
```

结果:
- dedup 后 checkpoint 池:
  - `790`
- 其中 midband (`1-99`) checkpoint:
  - `708`
- 最终 sampled checkpoint:
  - `90`

产物:
- sampled checkpoint 文件:
  - `checkpoints/scienceworld_core_train_checkpoints_sampled.json`

task 分布:
- `boil = 8`
- `melt = 8`
- `use-thermometer = 8`
- `power-component = 8`
- `test-conductivity = 8`
- `test-conductivity-of-unknown-substances = 8`
- `grow-plant = 8`
- `chemistry-mix = 8`
- `inclined-plane-friction-unnamed-surfaces = 8`
- `mendelian-genetics-known-plant = 8`
- `mendelian-genetics-unknown-plant = 8`
- `lifespan-longest-lived-then-shortest-lived = 2`

failure type 分布:
- `syntax_or_parse = 80`
- `precondition_blocked = 5`
- `ambiguity = 4`
- `action_loop = 1`

score band 分布:
- `20-79 = 73`
- `1-19 = 17`
- `80-99 = 0`

结论:
- 采样已经成功把 Stage 1 输入规模控制到可运行范围:
  - `90` 个 checkpoint
- task 覆盖基本均衡
- 但 failure type 仍然明显偏 `syntax_or_parse`
- 这说明:
  - 当前 raw checkpoint 池的主导问题不是“任务不平衡”
  - 而是“failure mechanism 不平衡”
- 因此这份 sampled 文件可以作为第一版 task-balanced 基线
- 但它不应直接作为最终 Stage 1 输入定稿，因为 rollout 很可能被 `syntax_or_parse` 主导
- 后续需要继续检查是否应做第二版“按 failure type 也平衡”的采样

当前阶段判断:
- 现在已经完成:
  - Stage 0 完成
  - `score_at_last_failure` backfill
  - dedup
  - checkpoint 采样
- 下一步可以正式进入:
  - `Stage 1: branched rollout`

对其他 benchmark 的启发:
- 按 task_type 平衡采样，不能自动解决 failure type 偏斜
- 如果 benchmark 的错误分布天然很偏，task-balanced sampling 之后仍可能得到 mechanism-skewed 的 rollout 输入
- 因此采样应至少同时检查:
  - task 覆盖
  - failure type 覆盖
  - score band 覆盖

### Step 5.6 Failure-aware checkpoint 采样定稿

目标:
- 在保持 task 覆盖基本不变的前提下
- 降低 Stage 1 输入被 `syntax_or_parse` 单一 failure 主导的风险
- 让 branched rollout 更能覆盖 `ambiguity`、`precondition_blocked`、`action_loop` 等更有训练价值的失败机制

为什么要补这一步:
- Step 5.5 的 task-balanced v1 虽然已经把规模控制到了 `90`
- 但 failure type 分布仍然是:
  - `syntax_or_parse = 80`
  - `precondition_blocked = 5`
  - `ambiguity = 4`
  - `action_loop = 1`
- 这种分布意味着 Stage 1 大部分 rollout 都会在评估“语法修补是否有效”
- 对后续 gate 学习“何时该介入、介入哪种 memory 更有用”不够充分
- 因此需要在 task-balanced 的基础上，再补一层 failure-aware 约束

本轮采用的 failure-aware 规则:
- 输入:
  - `checkpoints/scienceworld_core_train_checkpoints_deduped.json`
- 仍然只保留 midband checkpoint:
  - `score_at_checkpoint ∈ [1, 99]`
- 每个 task 最多取 `8` 条
- 若某个 task 下存在非 `syntax_or_parse` 的 midband checkpoint:
  - 先保留最多 `4` 条非 syntax failure
  - 再用 `syntax_or_parse` 填满剩余名额
- task 内排序优先级保持不变:
  1. `20-79`
  2. `1-19`
  3. `80-99`

执行命令:

```bash
python experiments/gate/sample_checkpoints.py \
  --input checkpoints/scienceworld_core_train_checkpoints_deduped.json \
  --output checkpoints/scienceworld_core_train_checkpoints_sampled_ftaware.json \
  --max-per-task 8 \
  --failure-aware \
  --max-non-syntax-per-task 4
```

结果:
- 最终 sampled checkpoint:
  - `90`
- task 覆盖与 v1 保持一致:
  - 11 个主任务各 `8`
  - `lifespan-longest-lived-then-shortest-lived = 2`

产物:
- failure-aware sampled checkpoint 文件:
  - `checkpoints/scienceworld_core_train_checkpoints_sampled_ftaware.json`

failure type 分布对比:
- v1 task-balanced:
  - `syntax_or_parse = 80`
  - `precondition_blocked = 5`
  - `ambiguity = 4`
  - `action_loop = 1`
- v2 failure-aware:
  - `syntax_or_parse = 53`
  - `precondition_blocked = 11`
  - `ambiguity = 20`
  - `action_loop = 5`
  - `physics_or_affordance = 1`

结论:
- v2 没有改变 task 覆盖规模
- 但显著降低了 `syntax_or_parse` 的占比:
  - `80/90 -> 53/90`
- 同时明显补强了更有代表性的 failure 机制:
  - `ambiguity: 4 -> 20`
  - `precondition_blocked: 5 -> 11`
  - `action_loop: 1 -> 5`
- 因此从 Stage 1 开始，正式使用:
  - `checkpoints/scienceworld_core_train_checkpoints_sampled_ftaware.json`
- Step 5.5 保留为第一版 task-balanced 采样记录
- Step 5.6 作为当前正式定稿的 Stage 1 输入口径

当前阶段判断:
- 现在已经完成:
  - Stage 0 完成
  - `score_at_last_failure` backfill
  - dedup
  - task-balanced v1 采样
  - failure-aware v2 采样定稿
- 下一步可以正式进入:
  - `Stage 1: branched rollout`

对其他 benchmark 的启发:
- 只按 task 做平衡，容易忽略 failure mechanism 的结构性偏斜
- 对 gate 训练而言，failure-aware 采样通常比纯 task-balanced 更稳妥
- 更合理的顺序通常是:
  1. 先做 dedup
  2. 再做 task coverage 检查
  3. 再做 failure type coverage 检查
  4. 最后定稿 Stage 1 输入

### Step 5.7 Stage 1 branched rollout smoke 验证通过

目标:
- 在正式启动 Stage 1 大规模 branched rollout 之前
- 先用极小规模样本验证更新后的 rollout 代码链路确实可运行
- 特别检查三件事:
  - rollout 是否复用了主 agent 的 system prompt 和 action postprocess
  - rollout 时刻重新解析的 retrieval 特征是否成功落盘
  - partial rollout 是否不再被静默吞掉

执行前定位到的问题:
- 在当前沙箱内，ScienceWorld 的 Py4J / JVM 无法正常启动
- 最小 `ScienceWorldEnv` smoke 会卡住或报 gateway 初始化失败
- 提权后最小 JVM smoke 通过，说明问题来自运行环境限制，而不是 gate / rollout 代码本身

最小 JVM smoke:

```bash
PATH=/opt/homebrew/opt/openjdk/bin:$PATH python - <<'PY'
import time
from scienceworld import ScienceWorldEnv
print('before')
t0 = time.time()
env = ScienceWorldEnv('', envStepLimit=100)
print('after_init', round(time.time() - t0, 2))
env.close()
print('closed')
PY
```

结果:
- `before`
- `after_init 0.31`
- `closed`

说明:
- 提权环境下 ScienceWorld JVM 能正常启动
- 因此后续 Stage 1 / Stage 3 相关 ScienceWorld 运行都应沿用这一权限前提

执行命令:

```bash
PATH=/opt/homebrew/opt/openjdk/bin:$PATH python experiments/gate/run_branched_rollouts.py \
  --benchmark scienceworld \
  --checkpoints checkpoints/scienceworld_core_train_checkpoints_sampled_ftaware.json \
  --memory memory/scienceworld_core_train_memory.json \
  --config config_scienceworld.yaml \
  --output rollouts/scienceworld_core_train_rollouts_smoke.json \
  --resolved-checkpoints-output checkpoints/scienceworld_core_train_checkpoints_resolved_smoke.json \
  --n-steps 5 \
  --n-replays 1 \
  --max-checkpoints 2 \
  --run-name sw_core_rollout_smoke_
```

结果:
- smoke checkpoint 数:
  - `2`
- 总 rollout 数:
  - `8`
- 每个 checkpoint 均完整包含:
  - `none`
  - `cue`
  - `question`
  - `repair`
- 每个 arm replay 数均为:
  - `1`

产物:
- smoke rollout:
  - `rollouts/scienceworld_core_train_rollouts_smoke.json`
- smoke resolved checkpoints:
  - `checkpoints/scienceworld_core_train_checkpoints_resolved_smoke.json`
- 日志:
  - `logs/20260417_095704/sw_core_rollout_smoke.log`

关键验证点:
- rollout JSON 已包含:
  - `__meta__`
  - `__resolved_checkpoints__`
- 训练端后续可以直接优先读取 resolved checkpoint 特征
- 至少有一个原本 retrieval 特征为空的 checkpoint 被成功回填:
  - `cp_0001`
  - `retrieved_memory_id: -1 -> 1`
  - `retrieval_rrf_score: 0.0 -> 0.03278688524590164`
  - `memory_entry_count: 0 -> 20`
- 另一个 checkpoint 也发生了 rollout-time 重解析:
  - `cp_0052`
  - `retrieved_memory_id: 2 -> 12`
  - `memory_entry_count: 12 -> 20`

运行口径说明:
- 本轮 smoke 末尾出现 Py4J callback server shutdown 警告
- 但发生在 rollout 已完成并写盘之后，不影响产物有效性
- 本轮没有出现 arm 缺失、partial rollout 被静默保存、或 resolved checkpoint 丢失的问题

当前阶段判断:
- Stage 1 的代码链路已经通过最小规模 smoke 验证
- 现在可以进入正式的 `Stage 1: branched rollout`
- 正式运行时仍需注意:
  - ScienceWorld 相关命令要在能正常启动 JVM 的权限环境下执行
  - Step 4 与 Step 5 继续保持产物隔离

对其他 benchmark 的启发:
- 在开始大规模 gate rollout 前，先做一个 end-to-end smoke 非常有价值
- smoke 不仅要看“命令能不能跑完”，还要检查:
  - 输出结构是否完整
  - resolved 特征是否真实更新
  - partial failure 是否会污染训练数据

### Step 5.8 Stage 1.5 LLM judge smoke 验证通过

目标:
- 在正式批量执行 Stage 1.5 之前
- 先验证 judge 链路本身能正常读取 smoke rollout、resolved checkpoint 和 gold path
- 同时验证新加入的:
  - failure-step context
  - `combined_progress`
  - judge 配置透传

执行命令:

```bash
python experiments/gate/run_llm_judge_scoring.py \
  --rollouts rollouts/scienceworld_core_train_rollouts_smoke.json \
  --checkpoints checkpoints/scienceworld_core_train_checkpoints_resolved_smoke.json \
  --gold-paths rollouts/scienceworld_gold_paths.json \
  --config config_scienceworld.yaml \
  --output rollouts/scienceworld_core_train_rollouts_smoke_judged.json \
  --llm-weight 0.6
```

结果:
- 处理 checkpoint:
  - `2`
- 评分轨迹:
  - `8`
- 平均 LLM judge score:
  - `1.50 / 10`
- 平均耗时:
  - `8.0s / trajectory`

产物:
- judged smoke rollout:
  - `rollouts/scienceworld_core_train_rollouts_smoke_judged.json`

关键验证点:
- judged rollout 已写出:
  - `llm_judge_score`
  - `llm_judge_progress`
  - `combined_progress`
- `__meta__.judge_config` 已保留:
  - `llm_weight = 0.6`
  - `combined_progress` 公式说明
- replay 级别已显式标记:
  - `judge_context_includes_failure_step = true`
- 这说明:
  - judge 端现在使用了 checkpoint history + failure step 的完整决策点上下文
  - Stage 2 后续可以直接选择 `combined_progress` 作为训练信号

本轮 smoke 的一个重要发现:
- 这次 judge smoke 使用的是旧版 Stage 1 smoke rollout:
  - `rollouts/scienceworld_core_train_rollouts_smoke.json`
- 该文件生成时还没有 `observations_full`
- 因此本轮 judge smoke 实际仍回退使用了截断版 `observations`
- 这不影响 judge 主链路可运行的结论
- 但意味着:
  - `observations_full` 这个新增字段本身
  - 需要在重新生成的 Stage 1 rollout 上再验证一次

信号解读:
- 在 `cp_0052` 上，env progress 对 `none` 和 `repair` 都是 `0.25`
- 但 LLM judge 给出:
  - `none = 0.00`
  - `repair = 0.30`
- 说明 judge 确实能识别 env-blind 场景下的轨迹差异
- 本轮 `combined_progress` 采用 `λ = 0.6` 后:
  - `none = 0.25`
  - `repair = 0.28`
- 差异被明显压缩
- 这说明组合信号更保守，后续正式 Stage 1.5 完成后需要再评估:
  - 当前 `λ = 0.6` 是否过弱
  - 是否需要提高 LLM 权重

当前阶段判断:
- Stage 1.5 judge 代码链路已经通过 smoke 验证
- 可以继续进入正式 Stage 1 rollout 与后续正式 Stage 1.5
- 但正式 judge 前应明确:
  - 最好使用重新生成且包含 `observations_full` 的 Stage 1 rollout
  - `combined_progress` 的权重仍需要在正式样本上再看一次分布

---

### Step 5.9 Stage 1.5 正式重跑完成（修复 gold path 覆盖）

目标:
- 在正式 `Stage 1.5` 上对全部 `90` 个 checkpoint 做完整 LLM judge 评分
- 不再复用旧版只覆盖 `24` 条 task-variation 的历史 gold path 文件
- 让 `combined_progress` 对当前 Stage 1 正式 rollout 的 `1080` 条轨迹全部可用

问题定位:
- 第一次正式 judge 时，脚本报告:
  - `No gold path: 34`
- 根因不是 judge 代码本身，而是旧的:
  - `rollouts/scienceworld_gold_paths.json`
  只包含历史上提取的 `24` 条 gold path
- 这份旧文件无法覆盖当前 Stage 1 输入里的全部 `90` 个 checkpoint
- 当前这轮 Stage 1 的真实唯一 `(task_name, variation_idx)` 数量是:
  - `63`
- 因此必须按当前:
  - `checkpoints/scienceworld_core_train_checkpoints_resolved_v1.json`
  重新提取 gold path，而不是继续复用旧文件

已做修改:
- `experiments/extract_gold_paths.py`
  - 支持通过 `--checkpoints` 指定当前实验链的 checkpoint 文件
  - 支持通过 `--output` 指定输出文件
  - 不再依赖旧的硬编码输入路径
  - 将零进展分析改为可选，不再阻塞 gold path 主提取流程

重新提取 gold path:

```bash
python experiments/extract_gold_paths.py \
  --checkpoints checkpoints/scienceworld_core_train_checkpoints_resolved_v1.json \
  --output rollouts/scienceworld_gold_paths.reextract_tmp.json
```

提取结果:
- 新 gold path 条数:
  - `63`
- 对当前 Stage 1 checkpoint 的覆盖:
  - `90 / 90` 个 checkpoint 均被覆盖
- 说明:
  - `63` 条 gold path 已完整覆盖当前 `90` 个 checkpoint
  - 原因是多个 checkpoint 共享同一个 `(task_name, variation_idx)`
  - 当前不存在:
    - missing checkpoint
    - extra checkpoint
    - 重复挂载到多个 gold path 的 checkpoint

按当前实验要求做的替换与清理:
- 删除旧文件:
  - `rollouts/scienceworld_gold_paths.json`
  - `rollouts/scienceworld_core_train_rollouts_v1_judged.json`
  - `rollouts/scienceworld_core_train_rollouts_v1_judged_intermediate.json`
- 将新提取文件替换为正式 gold path:
  - `rollouts/scienceworld_gold_paths.json`

正式重跑命令:

```bash
python experiments/gate/run_llm_judge_scoring.py \
  --rollouts rollouts/scienceworld_core_train_rollouts_v1.json \
  --checkpoints checkpoints/scienceworld_core_train_checkpoints_resolved_v1.json \
  --gold-paths rollouts/scienceworld_gold_paths.json \
  --config config_scienceworld.yaml \
  --output rollouts/scienceworld_core_train_rollouts_v1_judged.json \
  --llm-weight 0.6
```

结果:
- 完成时间:
  - `2026-04-19 16:27:27`
- 处理 checkpoint:
  - `90`
- 评分轨迹:
  - `1080`
- `No gold path`:
  - `0`
- 平均 LLM judge score:
  - `4.54 / 10`
- 平均耗时:
  - `6.2s / trajectory`

产物:
- 正式 gold path:
  - `rollouts/scienceworld_gold_paths.json`
- 正式 judged rollout:
  - `rollouts/scienceworld_core_train_rollouts_v1_judged.json`

关键判断:
- 这次 `Stage 1.5` 才是当前实验链的正式有效版本
- 原先那版 `No gold path: 34` 的 judged rollout 不应再用于任何后续分析或训练
- 修复后，`combined_progress` 已经对全部:
  - `90` 个 checkpoints
  - `1080` 条 rollout trajectories
  可用

信号解读:
- 全量 judged rollout 上:
  - `none avg_combined = 0.3098`
  - `question avg_combined = 0.3188`
  - `repair avg_combined = 0.3222`
  - `cue avg_combined = 0.3242`
- 相对 `none` 的 checkpoint 级平均增益:
  - `cue = +0.0143`
  - `question = +0.0089`
  - `repair = +0.0123`
- oracle 在 `cue / question / repair` 中选最优时:
  - `avg best - none = +0.0558`
  - `62 / 90` 个 checkpoints 为正
  - `42 / 90` 个 checkpoints 超过 `0.05`
  - `22 / 90` 个 checkpoints 超过 `0.10`

结论:
- Stage 1.5 结果现在已经可用于进入:
  - `Stage 2: Gate 训练`
- 但这批信号属于:
  - “可训练，但增益不算强”
- 因此更合理的使用方式是:
  - 用 `combined_progress`
  - 不直接依赖裸 `llm_judge_progress`

对其他 benchmark 的启发:
- gold path 一旦参与 judge，就不能只做“按 task_type 任取一条”的宽松兜底
- 更稳妥的要求是:
  - 当前训练链里的每个 `(task, variation)` 都有对应 gold path
- 否则 judge 结果会产生结构性缺口，且这种缺口会直接污染后续 gate 训练

### Step 5.10 Stage 2 Gate 训练完成（但暂不进入在线评估）

目标:
- 使用修复后的全量 judged rollout
- 以 `combined_progress` 作为 utility 信号，训练当前 ScienceWorld gate
- 判断这条数据链是否已经足以进入 `Stage 3: 在线评估`

执行命令:

```bash
python experiments/gate/run_gate_training.py \
  --checkpoints checkpoints/scienceworld_core_train_checkpoints_resolved_v1.json \
  --rollouts rollouts/scienceworld_core_train_rollouts_v1_judged.json \
  --output gate_models/scienceworld_core_train_gate_v1 \
  --beta 0.3 \
  --progress-signal combined \
  --run-name sw_gate_train_v1_
```

结果:
- 完成时间:
  - `2026-04-19 22:37:19`
- 原始 checkpoint:
  - `90`
- dedup 后:
  - `90`
- unique signatures:
  - `74`
- 训练样本:
  - `90` 个 checkpoints
- train / test split:
  - `61 / 29` checkpoints

Stage 2 前置 oracle 分析:
- `oracle_counts`:
  - `none = 38`
  - `question = 29`
  - `repair = 23`
- `oracle_fractions`:
  - `none = 0.4222`
  - `question = 0.3222`
  - `repair = 0.2556`
- `heterogeneity_index`:
  - `0.5778`
- `question_vs_cue`:
  - `question_better = 36`
  - `cue_better = 39`
  - `question_win_rate = 0.4`

训练结果:
- train:
  - `n_train_rows = 183`
  - `train_mae = 0.0314`
  - `baseline_mae = 0.1276`
  - `train_better = true`
- test:
  - `n_test_rows = 87`
  - `test_mae = 0.1765`
  - `test_baseline_mae = 0.1440`
  - `test_better = false`
  - `gate_accuracy = 0.2414`
  - `gate_arm_distribution = {'none': 7, 'question': 5, 'repair': 17}`

产物:
- gate 模型:
  - `gate_models/scienceworld_core_train_gate_v1.pkl`
  - `gate_models/scienceworld_core_train_gate_v1.json`
- 分析文件:
  - `gate_models/scienceworld_core_train_gate_v1_analysis.json`
- 日志:
  - `logs/20260419_223719/sw_gate_train_v1.log`

关键判断:
- 这批数据已经满足“可以训练 gate”的最低要求:
  - judged rollout 完整
  - `combined_progress` 全量可读
  - oracle 异质性足够:
    - `heterogeneity_index = 0.5778 > 0.3`
- 但当前模型泛化结果不理想:
  - `test_mae = 0.1765 > baseline 0.1440`
  - `gate_accuracy = 0.2414`
- 因此当前版本 gate 虽然已经训练完成
  - 但还不应直接进入:
    - `Stage 3: 在线评估`

为什么暂不进入在线评估:
- 训练集拟合明显成立，但测试集反而劣于 baseline
- 这说明当前问题不在“能不能把模型跑起来”
- 而在于:
  - utility 信号仍偏弱
  - checkpoint 数量仍然偏少
  - `question` 臂质量没有稳定优于 `cue`
  - 当前特征或采样分布对测试泛化还不够稳

更合理的下一步:
1. 先做 `Stage 2` 后置分析，而不是直接在线 eval
2. 重点检查:
   - `question` 臂质量
   - failure type / task type 上的 train-test 分布偏差
   - 是否需要扩大 Stage 1 输入规模
   - 是否需要重新设计 utility 或特征
3. 待离线测试指标至少不差于 baseline 后，再进入 `Stage 3`

结论:
- 当前实验链已经成功推进到:
  - `Stage 2 gate 训练`
- 但本轮训练结果显示:
  - “数据链是通的，模型也能训练”
  - “当前 gate 还不具备直接上线在线评估的证据”
- 因此此时最稳妥的决策是:
  - 继续做 Stage 2 分析 / 修正
  - 暂缓 Stage 3

对其他 benchmark 的启发:
- “oracle 异质性足够”不等于“gate 已经可以上线”
- 更可靠的推进顺序是:
  1. judge 信号全覆盖
  2. oracle 异质性达标
  3. 离线测试指标至少不劣于 baseline
  4. 再做在线评估

### Step 5.11 修复 Stage 2 评估口径错误并重跑 gate 训练

目标:
- 先确认当前 `Stage 2` 结果是不是模型本身的问题，还是训练/评估代码有问题
- 修复会直接影响泛化结论的代码级错误
- 在修复后重新训练 gate，再判断是否可以进入 `Stage 3`

问题复盘:
- 第一次 `Stage 2` 训练后，出现:
  - `train_mae` 很好
  - `test_mae` 反而差于 baseline
- 继续审查代码后，确认这里至少有两个会影响结论的实现问题:

1. train/test split 有泄漏
- `src/gate/checkpoint.py`
  旧版 `split()` 是按 `failure_signature` 切分
- 这会把同一 episode 中的多个 checkpoint 拆到 train 和 test 两边
- 实际核对时，确实存在同一 `(task_type, variation_idx, env_idx)` 横跨 train / test 的情况
- 这种 split 会污染“泛化”评估

2. test baseline 口径错误
- `src/gate/train_gate.py`
  旧版 `test_baseline_mae` 是用 `y_test` 自身均值计算
- 这等于在 baseline 里直接偷看了测试集标签分布
- 因此 `test_better` 不能作为严格可解释的 holdout 对照

已做修改:
- `src/gate/checkpoint.py`
  - `split()` 新增:
    - `group_by`
    - `stratify_by_task`
  - 现在可按 `episode` 分组切分，保证同一 episode 不会被拆开
- `experiments/gate/run_gate_training.py`
  - 训练入口改为:
    - `group_by="episode"`
    - `stratify_by_task=True`
  - 让 train/test 两边都保留全部 task，同时避免 episode 泄漏
- `src/gate/train_gate.py`
  - `test_baseline_mae` 改为只使用 train target mean
  - 不再使用 test 标签自身均值

修复后快速核对:
- 同一 episode 横跨 train/test:
  - `5 -> 0`
- train/test 两边 task 覆盖:
  - 现在两边都保留全部 task types

重跑命令:

```bash
python experiments/gate/run_gate_training.py \
  --checkpoints checkpoints/scienceworld_core_train_checkpoints_resolved_v1.json \
  --rollouts rollouts/scienceworld_core_train_rollouts_v1_judged.json \
  --output gate_models/scienceworld_core_train_gate_v2_grouped \
  --beta 0.3 \
  --progress-signal combined \
  --run-name sw_gate_train_v2_grouped_
```

结果:
- 完成时间:
  - `2026-04-19 22:59:53`
- split:
  - `65` train checkpoints
  - `25` test checkpoints
- train:
  - `train_mae = 0.0317`
  - `baseline_mae = 0.1271`
- test:
  - `test_mae = 0.1690`
  - `test_baseline_mae = 0.1574`
  - `test_better = false`
  - `gate_accuracy = 0.36`
  - `gate_arm_distribution = {'none': 12, 'question': 9, 'repair': 4}`

产物:
- gate 模型:
  - `gate_models/scienceworld_core_train_gate_v2_grouped.pkl`
  - `gate_models/scienceworld_core_train_gate_v2_grouped.json`
- 分析:
  - `gate_models/scienceworld_core_train_gate_v2_grouped_analysis.json`
- 日志:
  - `logs/20260419_225953/sw_gate_train_v2_grouped.log`

结果解读:
- 修复 split 泄漏和 test baseline 口径后:
  - `gate_accuracy` 从 `0.2414` 提升到 `0.36`
  - 说明原先的训练链确实存在评估设计问题
- 但修复后 test 指标仍没有超过 baseline:
  - `0.1690 > 0.1574`
- 这说明问题不只是评估代码
- 更大的可能是:
  - 当前 utility 信号仍偏弱
  - 样本规模偏小
  - `question` 臂信息质量仍不稳定

当前判断:
- 现在可以更有把握地说:
  - 训练代码里的关键评估问题已经修掉
  - 当前 gate 仍未在 holdout 上证明优于 baseline
- 因此结论更新为:
  - 不是“代码口径错导致误判”
  - 而是“修正代码后，模型仍然没有充分证据进入在线评估”

结论:
- 当前最合理的下一步仍然不是 `Stage 3`
- 而是继续做 `Stage 2` 后置分析:
  - 按 task / failure type 细看错误
  - 检查 `question` 臂质量
  - 评估是否需要扩大 Stage 1 checkpoint 规模或重新设计 utility

对其他 benchmark 的启发:
- 只按 failure signature 切 train/test，容易把同一 episode 的邻近 checkpoint 拆开，造成隐性泄漏
- 更可靠的 holdout 单位通常应至少是:
  - episode
  - 或 task-variation
- baseline 也必须只依赖 train 信息，不能直接用 test 标签本身

### Step 5.12: 修复 split 非确定性并重跑稳定版 Stage 2

目标:
- 让 `CheckpointStore.split(seed=42)` 在跨进程时也可复现
- 基于稳定 split 重跑一版 Stage 2，避免后续误差分析依赖漂移的 test 集

发现的问题:
- `src/gate/checkpoint.py` 里的 `split()` 先把 group 放进 `set`，再做 `shuffle`
- 这会导致:
  - 同一个 `seed=42`
  - 跨不同 Python 进程
  - 仍可能切出不同的 train/test
- 实测修复前连续重放会出现:
  - `66/24`
  - `67/23`
  - `64/26`
  等不同 split

已做修改:
- `src/gate/checkpoint.py`
  - 在 `_pick_train_groups()` 中先按 `repr` 排序，再 `shuffle`
  - `stratify_by_task=True` 时按排序后的 `task_type` 迭代
- 这样可以保证:
  - 同数据
  - 同 seed
  - 同 split

修复后核对:
- 连续跨进程重放 5 次:
  - 都稳定为 `69 train / 21 test`

重跑命令:

```bash
python experiments/gate/run_gate_training.py \
  --checkpoints checkpoints/scienceworld_core_train_checkpoints_resolved_v1.json \
  --rollouts rollouts/scienceworld_core_train_rollouts_v1_judged.json \
  --output gate_models/scienceworld_core_train_gate_v3_grouped_stable \
  --beta 0.3 \
  --progress-signal combined \
  --run-name sw_gate_train_v3_grouped_stable_
```

结果:
- 完成时间:
  - `2026-04-19 23:32:44`
- split:
  - `69` train checkpoints
  - `21` test checkpoints
- train:
  - `train_mae = 0.0363`
  - `baseline_mae = 0.1258`
- test:
  - `test_mae = 0.1648`
  - `test_baseline_mae = 0.1711`
  - `test_better = true`
  - `gate_accuracy = 0.4762`
  - `gate_arm_distribution = {'none': 14, 'question': 1, 'repair': 6}`

产物:
- gate 模型:
  - `gate_models/scienceworld_core_train_gate_v3_grouped_stable.pkl`
  - `gate_models/scienceworld_core_train_gate_v3_grouped_stable.json`
- 分析:
  - `gate_models/scienceworld_core_train_gate_v3_grouped_stable_analysis.json`
- 日志:
  - `logs/20260419_233244/sw_gate_train_v3_grouped_stable.log`

结论:
- `v2_grouped` 的方向性结论不能再按“固定 split 的严格 holdout 结果”来引用
- 修复 split 可复现性后:
  - gate 在 stable holdout 上已经略好于 baseline
- 但提升幅度仍然很小:
  - `0.1648 vs 0.1711`
- 所以当前状态更准确的表述是:
  - `gate 有一点可学信号`
  - 但还没有强到足以直接进入在线实验

### Step 5.13: Stage 2 postmortem 分析

目标:
- 按 `task / failure / oracle_arm` 分析稳定版 Stage 2 的误差来源
- 单独检查 `question` 臂是否值得继续保留
- 评估是否值得改成两阶段决策

新增脚本:
- `experiments/gate/analyze_gate_postmortem.py`
  - 作用:
    - 复用与训练完全一致的 offline utility 构造
    - 复放稳定 split
    - 输出 `task / failure / confusion / question-vs-cue / hardest misses`

执行命令:

```bash
python experiments/gate/analyze_gate_postmortem.py \
  --checkpoints checkpoints/scienceworld_core_train_checkpoints_resolved_v1.json \
  --rollouts rollouts/scienceworld_core_train_rollouts_v1_judged.json \
  --model gate_models/scienceworld_core_train_gate_v3_grouped_stable.pkl \
  --output gate_models/scienceworld_core_train_gate_v3_grouped_stable_postmortem.json \
  --progress-signal combined \
  --beta 0.3
```

核心结果:
- overall:
  - `test_accuracy = 0.4762`
  - `binary_intervene_accuracy = 0.6667`
  - `oracle_intervene_rate = 0.6667`
  - `predicted_intervene_rate = 0.3333`
  - `question_vs_repair_accuracy_on_oracle_intervene = 0.2143`
- confusion:
  - `oracle=none`: 全部 `7/7` 预测正确
  - `oracle=question`: `1/8` 预测成 `question`
  - `oracle=repair`: `2/6` 预测成 `repair`
- failure slice:
  - `precondition_blocked`: `3/3` 正确
  - `ambiguity`: `2/6` 正确，且 `6/6` 都被预测成 `none`
  - `syntax_or_parse`: `5/11` 正确，binary intervene 还可以，但 `question/repair` 区分差

### Step 5.14: added44 子集完成 Stage 1.5 LLM judge

目标:
- 对新增 `44` 个 checkpoint 的 branched rollouts 完成正式 LLM judge 评分
- 让这批新增数据也具备 `combined_progress`，可直接并入 Stage 2 训练

执行命令:

```bash
python experiments/gate/run_llm_judge_scoring.py \
  --rollouts rollouts/scienceworld_core_train_rollouts_added44_v2.json \
  --checkpoints checkpoints/scienceworld_core_train_checkpoints_resolved_added44_v2.json \
  --gold-paths rollouts/scienceworld_gold_paths_added44_v2.json \
  --config config_scienceworld.yaml \
  --output rollouts/scienceworld_core_train_rollouts_added44_v2_judged.json \
  --llm-weight 0.6
```

结果:
- 完成时间:
  - `2026-04-20 22:32:56`
- 处理 checkpoint:
  - `44`
- 评分轨迹:
  - `528`
- `No gold path`:
  - `0`
- 平均 LLM judge score:
  - `5.03 / 10`
- 平均耗时:
  - `6.3s / traj`
- `env-blind` checkpoint:
  - `17`
- 其中:
  - `LLM detection rate = 10/17 = 59%`
  - `combined detection rate = 5/17 = 29%`

产物:
- judged rollout:
  - `rollouts/scienceworld_core_train_rollouts_added44_v2_judged.json`
- 中间文件:
  - `rollouts/scienceworld_core_train_rollouts_added44_v2_judged_intermediate.json`

结论:
- 新增 `44` 条数据已经完成正式 judge，可直接用于扩容后的 Stage 2
- 这批数据上 `LLM judge` 仍能识别一部分 `env-blind` 差异
- 但 `combined_progress` 只保留了其中一部分区分度，后续仍需观察它对 gate 是否真的有帮助

### Step 5.15: 合并 `90 + 44 = 134` 条 judged 数据并重训 Stage 2 gate

目标:
- 将原 `90` 条正式 judged 数据与新增 `44` 条 judged 数据合并
- 用同一套 Stage 2 配置验证“扩 checkpoint 规模”是否改善 gate 泛化

合并产物:
- rollout:
  - `rollouts/scienceworld_core_train_rollouts_expanded134_judged.json`
- resolved checkpoints:
  - `checkpoints/scienceworld_core_train_checkpoints_resolved_expanded134.json`

重训命令:

```bash
python experiments/gate/run_gate_training.py \
  --checkpoints checkpoints/scienceworld_core_train_checkpoints_resolved_expanded134.json \
  --rollouts rollouts/scienceworld_core_train_rollouts_expanded134_judged.json \
  --output gate_models/scienceworld_core_train_gate_v4_expanded134 \
  --beta 0.3 \
  --progress-signal combined \
  --run-name sw_gate_train_v4_expanded134_
```

结果:
- 完成时间:
  - `2026-04-20 22:44:31`
- split:
  - `90` train checkpoints
  - `44` test checkpoints
- dedup:
  - `134 -> 134` checkpoints
  - `105` unique signatures
- train:
  - `train_mae = 0.0343`
  - `baseline_mae = 0.1359`
- test:
  - `test_mae = 0.1344`
  - `test_baseline_mae = 0.1259`
  - `test_better = false`
  - `gate_accuracy = 0.25`
  - `gate_arm_distribution = {'none': 13, 'question': 5, 'repair': 26}`
- oracle:
  - `{'none': 54, 'question': 37, 'repair': 43}`
  - heterogeneity index `= 0.597`

postmortem 命令:

```bash
python experiments/gate/analyze_gate_postmortem.py \
  --checkpoints checkpoints/scienceworld_core_train_checkpoints_resolved_expanded134.json \
  --rollouts rollouts/scienceworld_core_train_rollouts_expanded134_judged.json \
  --model gate_models/scienceworld_core_train_gate_v4_expanded134.pkl \
  --output gate_models/scienceworld_core_train_gate_v4_expanded134_postmortem.json \
  --progress-signal combined \
  --beta 0.3 \
  --train-ratio 0.7
```

postmortem 核心结果:
- overall:
  - `test_accuracy = 0.25`
  - `binary_intervene_accuracy = 0.4318`
  - `oracle_intervene_rate = 0.5455`
  - `predicted_intervene_rate = 0.7045`
  - `question_vs_repair_accuracy_on_oracle_intervene = 0.2917`
- confusion:
  - `oracle=none`: `4/20` 预测正确，`13/20` 被误判成 `repair`
  - `oracle=question`: `1/11` 预测正确，`7/11` 被误判成 `repair`
  - `oracle=repair`: `6/13` 预测正确，`6/13` 被误判成 `none`
- 典型坏 slice:
  - `grow-plant`: `0/5`
  - `mendelian-genetics-known-plant`: `0/3`
  - `use-thermometer`: `0/2`
  - `test-conductivity`: `1/5`
  - `test-conductivity-of-unknown-substances`: `2/6`
- 典型 hardest miss:
  - `cp_1716`: oracle=`none`，pred=`repair`，margin=`0.4021`
  - `cp_0150`: oracle=`none`，pred=`repair`，margin=`0.2600`
  - `cp_1447`: oracle=`repair`，pred=`none`，margin=`0.2267`

对比旧版 `90` 样本 stable holdout:
- 旧版:
  - `test_mae = 0.1648`
  - `test_baseline_mae = 0.1711`
  - `gate_accuracy = 0.4762`
- 扩容版:
  - `test_mae = 0.1344`
  - `test_baseline_mae = 0.1259`
  - `gate_accuracy = 0.25`

结论:
- 扩容到 `134` 后，模型仍然学到了强 train signal
- 但在当前 holdout 上，gate 仍然没有稳定优于 baseline
- 更具体地说:
  - 模型明显过度偏向 `intervene`
  - 尤其偏向 `repair`
  - `oracle=none` 的误报显著变多
- 因此当前状态仍不建议直接进入在线 gate 实验
- 更合理的下一步是:
  - 先做两阶段 gate:
    - `none vs intervene`
    - 再判 `question vs repair`
  - 或者重新审视 `combined_progress` 对 `none` 样本的压缩是否过强

### Step 5.16: 改成 task-balanced grouped split 后重跑 Stage 2

动机:
- 之前虽然使用了 `stratify_by_task=True`
- 但旧实现是在每个 task 桶内按“group 数量比例”切分
- 这会导致某些 task 出现明显不均衡:
  - 如 `10/2`、`7/5`
- 用户要求进一步验证:
  - train/test 的 task 类别计数应尽量均衡

实现修改:
- 更新 `src/gate/checkpoint.py`
- 当 `stratify_by_task=True` 时:
  - 保持 `group_by=episode` 不泄漏
  - 但改为在每个 task 桶内按“checkpoint 数量最接近目标值”选 train groups
- 在当前 `134` 条数据上，新的切分结果为:
  - `11` 个主要 task 都是 `8 train / 4 test`
  - `lifespan-longest-lived-then-shortest-lived` 是 `1 train / 1 test`

重跑命令:

```bash
python experiments/gate/run_gate_training.py \
  --checkpoints checkpoints/scienceworld_core_train_checkpoints_resolved_expanded134.json \
  --rollouts rollouts/scienceworld_core_train_rollouts_expanded134_judged.json \
  --output gate_models/scienceworld_core_train_gate_v4_expanded134_taskbalanced \
  --beta 0.3 \
  --progress-signal combined \
  --run-name sw_gate_train_v4_expanded134_taskbalanced_
```

结果:
- 完成时间:
  - `2026-04-20 22:52:21`
- split:
  - `89` train checkpoints
  - `45` test checkpoints
- train:
  - `train_mae = 0.0362`
  - `baseline_mae = 0.1362`
- test:
  - `test_mae = 0.1223`
  - `test_baseline_mae = 0.1269`
  - `test_better = true`
  - `gate_accuracy = 0.2222`
  - `gate_arm_distribution = {'none': 13, 'question': 6, 'repair': 26}`

postmortem 命令:

```bash
python experiments/gate/analyze_gate_postmortem.py \
  --checkpoints checkpoints/scienceworld_core_train_checkpoints_resolved_expanded134.json \
  --rollouts rollouts/scienceworld_core_train_rollouts_expanded134_judged.json \
  --model gate_models/scienceworld_core_train_gate_v4_expanded134_taskbalanced.pkl \
  --output gate_models/scienceworld_core_train_gate_v4_expanded134_taskbalanced_postmortem.json \
  --progress-signal combined \
  --beta 0.3 \
  --train-ratio 0.7
```

postmortem 核心结果:
- `test_accuracy = 0.2222`
- `binary_intervene_accuracy = 0.4222`
- `oracle_intervene_rate = 0.5778`
- `predicted_intervene_rate = 0.7111`
- confusion:
  - `oracle=none`: `3/19` 正确，`12/19` 被判成 `repair`
  - `oracle=question`: `1/13` 正确，`8/13` 被判成 `repair`
  - `oracle=repair`: `6/13` 正确，`6/13` 被判成 `none`

结论:
- task-balanced split 后，离线回归指标变得更合理:
  - `test_mae` 重新略好于 baseline
- 这说明之前那版 `44 test` 的负结果，确实受 task 分布不均衡影响
- 但新的 postmortem 也说明:
  - gate 仍然严重偏向 `repair`
  - `none/question/repair` 的离散分类仍不稳定
- 因此当前最准确的判断是:
  - `回归意义上的 utility 预测有一点泛化`
  - 但 `直接三分类选臂` 仍然不够稳

### Step 5.17: 两阶段 gate 最小实现

目标:
- 将单阶段 `none / question / repair` 三分类拆成两步:
  1. `none vs intervene`
  2. `question vs repair`
- 用当前 `expanded134` + task-balanced split 做首轮 sanity check

实现:
- 新增:
  - `src/gate/train_gate_two_stage.py`
  - `experiments/gate/run_gate_training_two_stage.py`
  - `experiments/gate/analyze_gate_postmortem_two_stage.py`
- 数据定义:
  - Stage A:
    - 标签由 `max(question, repair)` 与 `none` 的 utility 比较得到
    - 过滤 `|margin| <= 0.01` 的模糊样本
  - Stage B:
    - 只使用 `intervene` 更优的 checkpoint
    - 标签由 `repair - question` 的 utility 差决定
    - 同样过滤 `|margin| <= 0.01`

训练命令:

```bash
python experiments/gate/run_gate_training_two_stage.py \
  --checkpoints checkpoints/scienceworld_core_train_checkpoints_resolved_expanded134.json \
  --rollouts rollouts/scienceworld_core_train_rollouts_expanded134_judged.json \
  --output gate_models/scienceworld_core_train_gate_two_stage_expanded134_taskbalanced \
  --beta 0.3 \
  --progress-signal combined \
  --margin-eps 0.01 \
  --intervene-threshold 0.5 \
  --run-name sw_gate_train_two_stage_expanded134_taskbalanced_
```

结果:
- 完成时间:
  - `2026-04-20 23:05:51`
- split:
  - `89` train checkpoints
  - `45` test checkpoints
- Stage A 训练样本:
  - `73`
- Stage B 训练样本:
  - `44`
- train:
  - `stage_a_train_accuracy = 0.7945`
  - `stage_b_train_accuracy = 0.7727`
- test:
  - `two_stage_accuracy = 0.3111`
  - `binary_intervene_accuracy = 0.5333`
  - `intervene_precision = 0.5641`
  - `intervene_recall = 0.8462`
  - `question_vs_repair_accuracy_on_oracle_intervene = 0.4615`
  - `predicted_intervene_rate = 0.8667`
  - `avg_selected_utility = 0.2914`
  - `avg_none_utility = 0.3125`
  - `avg_gain_over_none = -0.0211`

postmortem 命令:

```bash
python experiments/gate/analyze_gate_postmortem_two_stage.py \
  --checkpoints checkpoints/scienceworld_core_train_checkpoints_resolved_expanded134.json \
  --rollouts rollouts/scienceworld_core_train_rollouts_expanded134_judged.json \
  --model gate_models/scienceworld_core_train_gate_two_stage_expanded134_taskbalanced.pkl \
  --output gate_models/scienceworld_core_train_gate_two_stage_expanded134_taskbalanced_postmortem.json \
  --progress-signal combined \
  --beta 0.3 \
  --train-ratio 0.7
```

postmortem 核心结果:
- `test_accuracy = 0.3111`
- `binary_intervene_accuracy = 0.5333`
- `intervene_precision = 0.5641`
- `intervene_recall = 0.8462`
- `question_vs_repair_accuracy_when_predicted_intervene = 0.5455`
- confusion:
  - `oracle=none`: 仅 `2/19` 正确，`12/19` 仍被判成 `repair`
  - `oracle=question`: `5/13` 正确
  - `oracle=repair`: `7/13` 正确

结论:
- 两阶段实现本身已经跑通
- 与单阶段相比:
  - `question vs repair` 子问题确实更好学了
  - 但 Stage A 仍明显过度干预
- 在当前 `threshold=0.5` 下:
  - recall 很高
  - precision 偏低
  - 实际 policy utility 甚至低于 always-`none`
- 因此下一步不应急着上线两阶段
- 更合理的是:
  - 先对 Stage A 扫阈值
  - 或加强 `none vs intervene` 的损失/采样策略
- question vs cue:
  - 全量 `90` 个 checkpoint 上:
    - `question_better = 38`
    - `cue_better = 45`
    - `avg(question - cue) = -0.0062`
  - 说明 `question` 并不是稳定优于 `cue`
  - 但也不是完全没价值:
    - 全量 oracle 里仍有 `29/90` 是 `question`
- 数据覆盖:
  - stable train 共 `69` 个 checkpoint，分布为:
    - `none = 31`
    - `question = 21`
    - `repair = 17`
  - 按 `task x oracle_arm` 看:
    - 共 `36` 个格子
    - `4` 个格子完全为 `0`
    - `15` 个格子只有 `0` 或 `1` 个样本
  - 典型缺口:
    - `grow-plant` 训练集中没有 `repair` oracle
    - `test-conductivity` 训练集中没有 `question` oracle
    - 多个 task 的 `question` 或 `repair` 只有 `1` 个样本

额外小实验:
- 在同一 stable split 上试了一个独立两阶段原型:
  - Stage 1: `none vs intervene`
  - Stage 2: `question vs repair`
- 结果:
  - Stage 1 accuracy = `0.6190`，低于多数类 baseline `0.6667`
  - Stage 2 accuracy = `0.5000`，低于多数类 baseline `0.5714`
  - 合成后的 overall accuracy = `0.4286`
- 说明:
  - 以当前数据和特征，直接改成两阶段并没有现成收益

结论:
- 当前最主要的问题不是“完全学不到”
- 而是:
  - gate 明显偏保守，`predicted_intervene_rate` 只有 `0.3333`
  - 一旦决定 intervene，又几乎学不会 `question`
  - `question` 相对 `cue` 的收益不稳定，标签边界偏噪
  - `task x arm` 覆盖极稀疏，样本规模确实偏小
- 因此最实际的下一步是:
  - 先扩大 Stage 1 checkpoint 规模
  - 同时重点补 `question/repair` 稀缺 task 的样本
  - 暂时不要把“两阶段”当成已经被验证可行的解法

### Step 5.14: 扩容 Stage 1 checkpoint 池（expanded v2）

目标:
- 在不放宽 dedup 规则的前提下，先扩一版 Stage 1 输入
- 优先补充每个 task 的覆盖，而不是一次性把大量剩余 midband checkpoint 全部拉回

现状核对:
- 原始 intermediate:
  - `1834`
- deduped:
  - `790`
- dedup 后 midband (`score in [1,99]`):
  - `708`
- 当前正式 sampled_ftaware:
  - `90`
- 说明仍有大量 midband checkpoint 尚未用于 rollout

扩容前的判断:
- 剩余 midband 虽然很多，但大头仍是 `syntax_or_parse`
- 因此不建议:
  - 直接把所有剩余 checkpoint 全部拉回
  - 或直接放宽 `max_per_signature`
- 更合理的是:
  - 继续维持 failure-aware 采样
  - 先把每 task 上限从 `8` 提到 `12`
  - 同时把 `max_non_syntax_per_task` 从 `4` 提到 `6`

执行命令:

```bash
python experiments/gate/sample_checkpoints.py \
  --input checkpoints/scienceworld_core_train_checkpoints_deduped.json \
  --output checkpoints/scienceworld_core_train_checkpoints_sampled_ftaware_expanded_v2.json \
  --max-per-task 12 \
  --failure-aware \
  --max-non-syntax-per-task 6
```

结果:
- 新 checkpoint 集:
  - `checkpoints/scienceworld_core_train_checkpoints_sampled_ftaware_expanded_v2.json`
- 规模:
  - `134` checkpoints
- 相比当前正式版:
  - `90 -> 134`
  - 新增 `44`
- task 分布:
  - 除 `lifespan-longest-lived-then-shortest-lived` 仍只有 `2`
  - 其余 task 都扩到 `12`
- failure 分布:
  - `syntax_or_parse = 85`
  - `ambiguity = 28`
  - `precondition_blocked = 13`
  - `action_loop = 6`
  - `physics_or_affordance = 2`

相对旧版新增的 44 个 checkpoint:
- task 分布:
  - 11 个主要 task 各新增 `4`
- failure 分布:
  - `syntax_or_parse = 32`
  - `ambiguity = 8`
  - `precondition_blocked = 2`
  - `action_loop = 1`
  - `physics_or_affordance = 1`

结论:
- 这版扩容是可行且相对稳妥的
- 它没有大改 dedup 逻辑，只是把每 task 覆盖做大一圈
- 下一步可以直接基于这版 checkpoint 跑:
  - Stage 1 branched rollout
  - Stage 1.5 LLM judge
  - Stage 2 retrain

### Step 5.15: 只对新增 44 个 checkpoint 跑增量 rollout

目标:
- 避免把原有 90 个 checkpoint 的 rollout 全部重跑一遍
- 只对 expanded v2 相比正式版新增的 `44` 个 checkpoint 做增量 rollout

执行:
- 从以下两个集合做差:
  - `checkpoints/scienceworld_core_train_checkpoints_sampled_ftaware.json`
  - `checkpoints/scienceworld_core_train_checkpoints_sampled_ftaware_expanded_v2.json`
- 生成新增子集:
  - `checkpoints/scienceworld_core_train_checkpoints_added44_v2.json`

子集规模:
- `44` checkpoints
- task 分布:
  - 11 个主要 task 各 `4`
- failure 分布:
  - `syntax_or_parse = 32`
  - `ambiguity = 8`
  - `precondition_blocked = 2`
  - `action_loop = 1`
  - `physics_or_affordance = 1`

增量 rollout 命令:

```bash
PATH=/opt/homebrew/opt/openjdk/bin:$PATH python experiments/gate/run_branched_rollouts.py \
  --benchmark scienceworld \
  --checkpoints checkpoints/scienceworld_core_train_checkpoints_added44_v2.json \
  --memory memory/scienceworld_core_train_memory.json \
  --config config_scienceworld.yaml \
  --output rollouts/scienceworld_core_train_rollouts_added44_v2.json \
  --resolved-checkpoints-output checkpoints/scienceworld_core_train_checkpoints_resolved_added44_v2.json \
  --n-steps 15 \
  --n-replays 3 \
  --run-name sw_core_rollout_added44_v2_
```

注意:
- rollout 可以只跑新增 44 个
- 但当前 `rollouts/scienceworld_gold_paths.json` 不能直接覆盖这 44 个新增 checkpoint
- 核对结果:
  - 仅覆盖了 `22/44`
- 因此在做 Stage 1.5 judge 前，需要针对新增子集或合并后的新集合，重新提取对应 gold path

---

## 跨 Benchmark 迁移经验

目前已沉淀:
- 任何依赖本地 JVM、子进程桥接、端口绑定的 benchmark，都要优先检查沙箱限制，而不是先怀疑代码
- 先做“环境 smoke test”，再做“最小 LLM 闭环”，最后才做完整实验
- 文档应区分:
  - 静态前置条件
  - 运行时权限要求
  - 最小复现命令
  - 完整实验命令

---

## 2026-04-20 两阶段 Gate 小改动优化

目标:
- 在不扩 checkpoint 数据的前提下，先验证两阶段 gate 是否能通过更保守的 `Stage A` 训练目标得到可用提升
- 重点尝试:
  - `Stage A` 类别权重
  - `validation` 选阈值
  - 基于 `margin` 的样本加权 / 边界样本过滤

代码改动:
- `src/gate/train_gate_two_stage.py`
  - 新增 `Stage A` 样本权重:
    - `stage_a_positive_weight`
    - `stage_a_negative_weight`
    - `stage_a_margin_weight_alpha`
  - 支持覆写 `intervene_threshold` 做评估 / 扫阈值
- `experiments/gate/run_gate_training_two_stage.py`
  - 新增 `val_ratio`
  - 新增 validation threshold sweep
  - 新增 `always-none` baseline 输出

实验设置:
- 数据:
  - `checkpoints/scienceworld_core_train_checkpoints_resolved_expanded134.json`
  - `rollouts/scienceworld_core_train_rollouts_expanded134_judged.json`
- split:
  - task-balanced grouped split
  - `89 train / 45 test`
  - 其中优化版再从 train 内切 `65 fit / 24 val`
- 阈值选择目标:
  - `avg_gain_over_none`

对照结果:
- 原始 two-stage:
  - `avg_gain_over_none = -0.0211`
  - `avg_regret = 0.0616`
  - `binary_intervene_accuracy = 0.5333`
  - `predicted_intervene_rate = 0.8667`
- `A` 仅加 val 选阈值:
  - `threshold = 0.70`
  - `avg_gain_over_none = -0.0139`
  - `avg_regret = 0.0544`
- `B` `negative_weight=2.0`:
  - `threshold = 0.50`
  - `avg_gain_over_none = -0.0191`
  - `avg_regret = 0.0597`
- `C` `negative_weight=2.0 + margin_weight_alpha=1.0`:
  - `threshold = 0.80`
  - `avg_gain_over_none = -0.0107`
  - `avg_regret = 0.0512`
- `E` `negative_weight=4.0 + margin_weight_alpha=1.0`:
  - `threshold = 0.50`
  - `avg_gain_over_none = -0.0106`
  - `avg_regret = 0.0511`
- `F` `negative_weight=2.0 + margin_weight_alpha=1.0 + margin_eps=0.03`:
  - `threshold = 0.60`
  - `avg_gain_over_none = -0.0131`
  - `avg_regret = 0.0536`
  - `binary_intervene_accuracy = 0.5778`
- `G` `negative_weight=2.0 + margin_weight_alpha=1.0 + margin_eps=0.05`:
  - `threshold = 0.85`
  - `avg_gain_over_none = -0.0120`
  - `avg_regret = 0.0525`

baseline:
- `always-none`
  - `avg_gain_over_none = 0.0000`
  - `avg_regret = 0.0405`

结论:
- 这些优化把 two-stage 从“明显过度干预”拉回了一些，但测试集上仍然全部输给 `always-none`
- 当前最好两组是:
  - `E`: utility 最接近 baseline
  - `F`: `binary_intervene_accuracy` 最高，但 utility 仍然为负
- 说明当前主要问题依然是:
  - `Stage A` 泛化信号弱
  - validation 上选出的阈值不能稳定迁移到 test
  - `Stage B` 在更保守设置下仍有明显误判成本

当前判断:
- 这些“小改动优化”值得保留，但还不足以宣布 two-stage gate 可用
- 下一步不应继续只调阈值 / 小权重
- 更值得做的是:
  - 做 `Stage A` 误差切片分析
  - 检查 `question` 臂是否本身就不稳定
  - 如继续建模，优先尝试更贴近 utility 的 `Stage A` 目标，而不是继续硬二分类微调

---

## 2026-04-20 judged 数据质量与模型选择对照

目标:
- 回到最原始的 two-stage 设定
- 先判断 134 条 judged 数据里哪种评分信号更适合作为监督
- 再比较更合适的模型类型

评分信号质量观察:
- replay 级别相关性:
  - `corr(env, llm) = 0.4838`
  - `corr(env, combined) = 0.7898`
  - `corr(llm, combined) = 0.9144`
- 当前 judged 数据的 `combined` 配比:
  - `llm_weight = 0.6`
- 结论:
  - `combined` 当前与 `llm` 非常接近
  - `env` 与 `llm` 差异显著

oracle 分布:
- `env`:
  - `none=82 question=22 repair=30`
- `llm`:
  - `none=55 question=40 repair=39`
- `combined(w=0.2)`:
  - `none=53 question=34 repair=47`
- `combined(w=0.4)`:
  - `none=54 question=38 repair=42`
- `combined(w=0.6)`:
  - `none=54 question=37 repair=43`

Stage A 可学习性:
- `env`:
  - `abs(stage_a_margin) <= 0.01` 的 checkpoint 有 `75/134`
- `llm`:
  - `23/134`
- `combined(w=0.6)`:
  - `25/134`
- 结论:
  - `env` 的边界样本太多，监督更弱
  - `llm / combined` 更有区分度

模型对照:
- `xgb_cls` = 原始 two-stage 二分类
- `xgb_reg` = 两阶段 margin regression
- `logistic` = `logistic regression + one-hot(task_type, failure_type)` baseline

结果摘要:
- `xgb_cls + env`
  - `avg_gain_over_none = +0.0009`
  - `avg_regret = 0.0211`
  - 但 `oracle none` 占比过高，`Stage A` 训练样本只有 `40`
- `xgb_cls + llm`
  - `avg_gain_over_none = -0.0235`
  - 过度干预明显
- `xgb_cls + combined(w=0.2)`
  - `avg_gain_over_none = -0.0117`
  - `avg_regret = 0.0386`
  - 是 `llm/combined` 组里原始分类模型最稳的一档
- `xgb_cls + combined(w=0.4)`
  - `avg_gain_over_none = -0.0236`
- `xgb_cls + combined(w=0.6)`
  - `avg_gain_over_none = -0.0211`

- `xgb_reg + llm`
  - `avg_gain_over_none = -0.0129`
- `xgb_reg + combined(w=0.2)`
  - `avg_gain_over_none = -0.0064`
  - `avg_regret = 0.0332`
  - 是本轮 `llm/combined` 主线里最好的模型组合
- `xgb_reg + combined(w=0.4)`
  - `avg_gain_over_none = -0.0036`
  - `avg_regret = 0.0368`
- `xgb_reg + combined(w=0.6)`
  - `avg_gain_over_none = -0.0087`

- `logistic + llm`
  - `avg_gain_over_none = -0.0207`
- `logistic + combined(w=0.2)`
  - `avg_gain_over_none = -0.0118`
- `logistic + combined(w=0.4)`
  - `avg_gain_over_none = -0.0165`
- `logistic + combined(w=0.6)`
  - `avg_gain_over_none = -0.0242`

baseline 对照:
- `always-none` 在不同评分信号下仍然是重要基线:
  - `combined(w=0.2)` baseline `avg_regret = 0.0268`
  - `combined(w=0.4)` baseline `avg_regret = 0.0332`
  - `combined(w=0.6)` baseline `avg_regret = 0.0405`

结论:
- 如果只看数据质量:
  - `env` 不适合作为主监督信号
  - `llm / combined` 更有学习价值
- 如果只看 `llm/combined` 主线:
  - `combined(w=0.2)` 比当前默认 `w=0.6` 更稳
  - `margin regression` 明显优于原始二分类 two-stage 和 logistic baseline
- 但到目前为止:
  - 除了 `env` 那条近乎退化的结果外
  - 其余主线组合仍没有稳定超过 `always-none`

当前建议:
- 后续主线默认切到:
  - 评分信号: `combined(w=0.2)` 或 `combined(w=0.4)`
  - 模型: `xgb_reg` 两阶段 margin regression
- 不建议再把主要精力放在:
  - 纯 `env` 监督
  - `logistic` baseline
  - 当前默认 `combined(w=0.6)` 的原始二分类 two-stage

固定主线:
- 评分信号:
  - `combined(w=0.4)`
- 模型:
  - `xgb_reg` 两阶段 margin regression
- 选择理由:
  - 在 `llm/combined` 主线上，`avg_gain_over_none = -0.0036`，最接近打平 `always-none`
  - 同时仍保留比纯 `env` 更合理的 oracle 分布与更强的监督区分度
  - 比原始 `xgb_cls` 和 `logistic` baseline 更贴近 utility 目标

主线标准产物:
- 模型:
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_mainline_v1.pkl`
- 配置:
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_mainline_v1.json`
- 分析:
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_mainline_v1_analysis.json`

主线当前指标:
- `n_train_checkpoints = 89`
- `n_test_checkpoints = 45`
- `n_stage_a_train = 73`
- `n_stage_b_train = 45`
- `avg_gain_over_none = -0.0036`
- `avg_regret = 0.0368`
- `binary_intervene_accuracy = 0.4889`
- `predicted_intervene_rate = 0.7111`

主线 postmortem:
- 输出文件:
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_mainline_v1_postmortem.json`
- overall:
  - `test_accuracy = 0.2444`
  - `binary_intervene_accuracy = 0.4889`
  - `question_vs_repair_accuracy_on_oracle_intervene = 0.24`
  - `avg_stage_a_abs_error = 0.0900`
  - `avg_stage_b_abs_error_on_oracle_intervene = 0.0843`
- 责任划分:
  - `34` 个 test miss 中
  - `20` 个主要是 `Stage A` 符号判反
  - `10` 个主要是 `Stage B` 在已决定干预后把 `question/repair` 选反
  - 其余 `4` 个为边界型或混合错误
- 直接表现:
  - false-positive intervene: `15`
  - false-negative intervene: `8`
  - 已决定 intervene 但 arm 选错: `10`

最差 task 切片:
- `boil`
  - `accuracy = 0.0`
  - `binary_intervene_accuracy = 0.0`
- `mendelian-genetics-known-plant`
  - `accuracy = 0.0`
  - `binary_intervene_accuracy = 0.0`
- `use-thermometer`
  - `accuracy = 0.0`
  - `binary_intervene_accuracy = 0.5`
  - `avg_stage_b_abs_error = 0.1608`
- `test-conductivity-of-unknown-substances`
  - `accuracy = 0.0`
  - `binary_intervene_accuracy = 0.5`

最差 failure 切片:
- `ambiguity`
  - `accuracy = 0.2222`
  - `binary_intervene_accuracy = 0.4444`
- `syntax_or_parse`
  - `accuracy = 0.25`
  - `binary_intervene_accuracy = 0.4688`
- hardest misses 中:
  - `syntax_or_parse = 8`
  - `ambiguity = 4`

当前理解:
- 主线的第一瓶颈仍然是 `Stage A`
  - 特别是把本应 `none` 的点误判成 intervene
- 但 `Stage B` 也明显不稳
  - 尤其在 `use-thermometer / chemistry-mix / grow-plant` 上，`question vs repair` 容易翻车

---

## 2026-04-21 单门控回查

目标:
- 回到最初的单门控模型
- 使用当前固定主线数据:
  - `combined(w=0.4)`
- 观察单门控在同一数据上的效果

模型产物:
- `gate_models/scienceworld_core_train_gate_single_stage_expanded134_combined_w04_mainline_probe.pkl`
- `gate_models/scienceworld_core_train_gate_single_stage_expanded134_combined_w04_mainline_probe.json`
- `gate_models/scienceworld_core_train_gate_single_stage_expanded134_combined_w04_mainline_probe_analysis.json`

结果:
- 训练指标:
  - `train_mae = 0.0331`
  - `baseline_mae = 0.1299`
- 测试指标:
  - `test_mae = 0.1100`
  - `test_baseline_mae = 0.1119`
  - `gate_accuracy = 0.2444`
- 转成和两阶段一致的 utility 口径后:
  - `binary_intervene_accuracy = 0.5111`
  - `predicted_intervene_rate = 0.7778`
  - `avg_regret = 0.0512`
  - `avg_gain_over_none = -0.0180`

对比主线 two-stage regression:
- 主线 two-stage regression:
  - `avg_regret = 0.0368`
  - `avg_gain_over_none = -0.0036`
- 单门控:
  - `avg_regret = 0.0512`
  - `avg_gain_over_none = -0.0180`

结论:
- 在当前主线数据 `combined(w=0.4)` 上
- 单门控明显差于当前两阶段主线
- 因此不建议把主线切回最初的单门控模型

---

## 2026-04-21 当前项目现状

核心方法定位:
- 当前项目研究的是一种 `in-loop failure-triggered retrieval + selective memory intervention` 框架
- 即:
  - agent 在执行过程中进入 failure checkpoint
  - failure 触发对键值形式 memory 的在线检索
  - 检索到 memory 后，不默认强制注入
  - 而是通过 gate 决定是否以及如何干预

已经成立的部分:
- `failure checkpoint` 作为决策点是合理的
- `failure-triggered retrieval` 作为在线 memory 调用机制是合理的
- 不同 intervention arm 在 checkpoint 上确实存在 utility 异质性
  - 某些点更适合 `none`
  - 某些点更适合 `question`
  - 某些点更适合 `repair`
- 因此:
  - 固定 memory policy 不是最优
  - selective intervention 是必要问题

已经跑通但尚未完全验证的部分:
- 通过 branched rollout + judged scoring，已经可以为 gate 构造监督信号
- 已经可以训练:
  - 单门控模型
  - 两阶段 gate
  - margin regression 版本
- 但 learned gate 目前还没有稳定超过强 baseline，尤其是 `always-none`

当前最关键的问题:
- 问题不在整体框架
- 问题主要集中在 `gate learning`
- 具体表现为:
  - 监督噪声较大
  - 边界样本较多
  - 特征对“该不该干预”的刻画还不够强
  - 训练目标与最终 utility 目标尚未完全对齐

当前最稳妥的判断:
- 方法框架本身是成立的
- selective memory intervention 的需求也是成立的
- 但 learned gate 这一层，目前还不能说已经被稳定验证为有效

当前主线:
- 数据:
  - `combined(w=0.4)`
- 模型:
  - `xgb_reg` 两阶段 margin regression
- 标准产物:
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_mainline_v1.pkl`
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_mainline_v1_analysis.json`

当前主线指标:
- `avg_gain_over_none = -0.0036`
- `avg_regret = 0.0368`
- `binary_intervene_accuracy = 0.4889`
- `predicted_intervene_rate = 0.7111`

对 gate 的当前判断:
- gate 仍然是重要模块
- 但现阶段它更像“部分跑通的策略学习层”
- 还不是整个项目里已经被完全站稳的 strongest claim

后续优化方向的原则:
- 不再频繁切换框架
- 先固定主线
- 优先优化 gate 中“是否干预”这一层
- 目标不是追求分类准确率，而是让 utility 指标稳定超过 baseline

---

## 2026-04-21 主线 gate 最小优化

目标:
- 不换框架、不换数据
- 沿当前主线:
  - `combined(w=0.4)`
  - `xgb_reg` two-stage margin regression
- 先做最小的决策校准和 `Stage A` 训练优化

### Step 1: 阈值校准

做法:
- 固定已训练好的 `mainline_v1`
- 扫描:
  - `intervene_margin_threshold`
  - `repair_margin_threshold`

结果:
- 最优点:
  - `intervene_margin_threshold = 0.00`
  - `repair_margin_threshold = 0.04`
- 指标:
  - `avg_gain_over_none = -0.0019`
  - `avg_regret = 0.0351`
  - `two_stage_accuracy = 0.2889`
- 相比原始 `mainline_v1`:
  - `avg_gain_over_none`: `-0.0036 -> -0.0019`
  - `avg_regret`: `0.0368 -> 0.0351`

判断:
- 当前 regression 主线的主要可调收益来自 `Stage B`
- `repair` 选择过于激进，适当提高 `Stage B` 阈值是有效的
- `Stage A` 单靠抬阈值没有明显改进

主线候选升级版:
- `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_mainline_v1_1.pkl`
- `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_mainline_v1_1_analysis.json`

### Step 2: 只动 Stage A 的训练

实验设置:
- 固定:
  - `repair_margin_threshold = 0.04`
- 只改 `Stage A`

对照组:
- `neg2`
  - `stage_a_negative_weight = 2.0`
- `eps003`
  - `stage_a_min_abs_margin = 0.03`
- `neg2 + eps003`
  - 两者叠加

结果:
- `mainline_v1.1`
  - `avg_gain_over_none = -0.0019`
  - `avg_regret = 0.0351`
- `neg2`
  - `avg_gain_over_none = -0.0163`
  - `avg_regret = 0.0495`
- `eps003`
  - `avg_gain_over_none = -0.0016`
  - `avg_regret = 0.0348`
- `neg2 + eps003`
  - `avg_gain_over_none = -0.0108`
  - `avg_regret = 0.0440`

结论:
- `Stage A` 负样本加权没有帮助，反而明显恶化
- `Stage A` 去噪有轻微提升
  - 是当前最好的主线版本
- 当前最优候选可更新为:
  - `combined(w=0.4)`
  - `xgb_reg` two-stage
  - `repair_margin_threshold = 0.04`
  - `stage_a_min_abs_margin = 0.03`

当前最优指标:
- `avg_gain_over_none = -0.0016`
- `avg_regret = 0.0348`

当前判断:
- 主线已经非常接近 `always-none`
- 但仍未稳定超过 baseline
- 继续优化时，应优先保留:
  - `Stage B` 较保守阈值
  - `Stage A` margin 去噪
- 不建议继续沿着 `Stage A` 负样本加权这条线投入

---

## 2026-04-21 规范化验证

目标:
- 停止继续在旧 test 上调参
- 对当前 regression 主线做一次干净的 `train / val / test` 验证
- 只在 val 上选配置，在 untouched outer test 上最终评估一次

协议:
- 数据:
  - `combined(w=0.4)`
- 模型:
  - `xgb_reg` two-stage margin regression
- split:
  - outer split:
    - seed=`123`
    - `70% trainval / 30% test`
  - inner split:
    - seed=`456`
    - trainval 内再切 `fit / val`
- 候选空间:
  - `stage_a_min_abs_margin ∈ {0.01, 0.03, 0.05}`
  - `intervene_margin_threshold ∈ {0.00, 0.02}`
  - `repair_margin_threshold ∈ {0.00, 0.02, 0.04, 0.06}`

验证产物:
- 模型:
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_cleanval_v1.pkl`
- 配置:
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_cleanval_v1.json`
- 分析:
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_cleanval_v1_analysis.json`

val 选中的配置:
- `stage_a_min_abs_margin = 0.01`
- `intervene_margin_threshold = 0.00`
- `repair_margin_threshold = 0.00`

val 指标:
- `avg_gain_over_none = +0.0159`
- `avg_regret = 0.0219`

untouched test 指标:
- `avg_gain_over_none = -0.0181`
- `avg_regret = 0.0519`
- `binary_intervene_accuracy = 0.5111`
- `predicted_intervene_rate = 0.6000`

untouched test baseline:
- `always-none`
  - `avg_gain_over_none = 0.0000`
  - `avg_regret = 0.0338`

结论:
- 在规范化验证下，当前 gate 仍然明显输给 `always-none`
- 之前在旧 test 上得到的接近 baseline 的结果，不能视为正式成立
- 当前 learned gate 还没有被干净地验证为有效

当前项目判断更新:
- 主线框架仍成立
- 但 gate 模块暂时还不能进入正式主结果
- 后续若继续优化 gate，应明确视为“继续探索”，而不是“已可进入正式实验”

继续调参:
- 在同一 clean protocol 上，进一步做了 `val-only` 参数搜索
- 搜索范围:
  - `stage_a_min_abs_margin ∈ {0.00, 0.01, 0.02, 0.03, 0.04, 0.05}`
  - `intervene_margin_threshold ∈ {0.00, 0.01, 0.02, 0.03}`
  - `repair_margin_threshold ∈ {0.00, 0.02, 0.04, 0.06, 0.08}`
  - `stage_a_negative_weight ∈ {1.0, 1.25, 1.5, 2.0}`

val-only 搜索最优点:
- `stage_a_min_abs_margin = 0.01`
- `intervene_margin_threshold = 0.02`
- `repair_margin_threshold = 0.00`
- `stage_a_negative_weight = 1.25`
- val 指标:
  - `avg_gain_over_none = +0.0193`
  - `avg_regret = 0.0184`

以该配置做新的 untouched test 评估:
- 产物:
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_cleanval_v2.pkl`
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_cleanval_v2_analysis.json`
- test 指标:
  - `avg_gain_over_none = -0.0127`
  - `avg_regret = 0.0465`
- baseline:
  - `avg_gain_over_none = 0.0000`
  - `avg_regret = 0.0338`

补充结论:
- 继续调参之后，clean test 上仍然输给 `always-none`
- 相比 `cleanval_v1`
  - 有小幅改善
  - 但仍不足以支持“gate 已可进入正式实验”
- 当前阶段再继续做参数微调，收益已经明显递减

---

## 2026-04-21 80/20 clean protocol 复核

目标:
- 在不污染评估的前提下，给 gate 更多训练数据
- 把 outer split 从 `70/30` 改成 `80/20`
- 仍保持:
  - `group_by=episode`
  - `stratify_by_task=True`
  - 只在 inner val 上选配置
  - outer test 只评一次

实现:
- 新增脚本:
  - `experiments/gate/run_gate_training_two_stage_regression_cleanval.py`
- 协议:
  - outer split: `111 trainval / 23 test`
  - inner split: `89 fit / 22 val`
  - 数据信号: `combined(w=0.4)`
  - 模型: `xgb_reg` two-stage regression
  - 搜索空间:
    - `stage_a_min_abs_margin ∈ {0.00,0.01,0.02,0.03,0.04,0.05}`
    - `intervene_margin_threshold ∈ {0.00,0.01,0.02,0.03}`
    - `repair_margin_threshold ∈ {0.00,0.02,0.04,0.06,0.08}`
    - `stage_a_negative_weight ∈ {1.0,1.25,1.5,2.0}`

split 检查:
- outer test 覆盖全部 task type
- 11 个 12-checkpoint task:
  - `10 trainval / 2 test`
- 特殊 2-checkpoint task:
  - `1 trainval / 1 test`
- inner split:
  - 对 11 个主任务是 `8 fit / 2 val`

产物:
- 模型:
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_cleanval_80_20_v1.pkl`
- 配置:
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_cleanval_80_20_v1.json`
- 分析:
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_cleanval_80_20_v1_analysis.json`

val 选中的配置:
- `stage_a_min_abs_margin = 0.03`
- `intervene_margin_threshold = 0.03`
- `repair_margin_threshold = 0.04`
- `stage_a_negative_weight = 1.25`

val 最优点:
- `avg_gain_over_none = +0.0040`
- `avg_regret = 0.0252`
- `binary_intervene_accuracy = 0.5000`

untouched outer test:
- `n_train_checkpoints = 111`
- `n_test_checkpoints = 23`
- `two_stage_accuracy = 0.3913`
- `binary_intervene_accuracy = 0.5652`
- `intervene_precision = 0.6667`
- `intervene_recall = 0.5714`
- `avg_gain_over_none = +0.0126`
- `avg_regret = 0.0301`

outer test baseline:
- `always-none`
  - `avg_gain_over_none = 0.0000`
  - `avg_regret = 0.0428`
  - `two_stage_accuracy = 0.3913`

当前判断更新:
- 相比 `70/30 cleanval_v2`，`80/20` 明显更有希望
- 这说明“多给 train、少给 test”对当前 gate 是有帮助的
- 但这仍然只是:
  - 单个 outer seed
  - 仅 `23` 个 test checkpoint
- 因此:
  - 可以作为“继续推进 gate”的正向信号
  - 但还不足以当作正式稳定结论
- 更合理的下一步是:
  - 固定这套 `80/20` protocol
  - 再做少量多 seed repeated holdout
  - 看 `avg_gain_over_none` 是否稳定为正

---

## 2026-04-21 80/20 multi-seed repeated holdout

目标:
- 验证 `80/20` 单次 seed 的正收益是否稳定
- 避免把 `seed=123` 的偶然正结果误判成“gate 已成立”

实现:
- 新增脚本:
  - `experiments/gate/run_gate_training_two_stage_regression_cleanval_multiseed.py`
- 协议:
  - 固定 `80/20 outer split + inner val search`
  - outer seeds:
    - `123, 231, 341, 451, 561`
  - inner seeds:
    - `456, 457, 458, 459, 460`
  - 其余搜索空间与 `80/20 cleanval v1` 相同

产物:
- 汇总:
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_cleanval_80_20_multiseed_v1_summary.json`
- 各 seed 产物:
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_cleanval_80_20_multiseed_v1_seed123_*`
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_cleanval_80_20_multiseed_v1_seed231_*`
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_cleanval_80_20_multiseed_v1_seed341_*`
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_cleanval_80_20_multiseed_v1_seed451_*`
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_cleanval_80_20_multiseed_v1_seed561_*`

每个 seed 的 outer test 指标:
- `seed=123`
  - `avg_gain_over_none = +0.0126`
  - `avg_regret = 0.0301`
- `seed=231`
  - `avg_gain_over_none = -0.0100`
  - `avg_regret = 0.0385`
- `seed=341`
  - `avg_gain_over_none = -0.0001`
  - `avg_regret = 0.0308`
- `seed=451`
  - `avg_gain_over_none = -0.0158`
  - `avg_regret = 0.0358`
- `seed=561`
  - `avg_gain_over_none = -0.0176`
  - `avg_regret = 0.0480`

汇总:
- `positive_gain_seed_count = 1 / 5`
- `avg_gain_over_none_mean = -0.0062`
- `avg_gain_over_none_std = 0.0125`
- `avg_regret_mean = 0.0366`
- `avg_regret_std = 0.0072`
- `binary_intervene_accuracy_mean = 0.5043`
- `intervene_precision_mean = 0.5447`
- `intervene_recall_mean = 0.6661`

最终判断更新:
- `80/20` 单次 seed 的正收益并不稳定
- 多给训练数据以后，gate 确实更有机会接近 baseline
- 但在 repeated holdout 下:
  - 平均仍然输给 `always-none`
  - 只有 `1/5` 个 seed 为正收益
- 因此当前最严谨的判断仍是:
  - gate 还不能作为正式稳定模块进入 headline result
  - 但可以继续作为探索模块推进

对当前主线的含义:
- 不能再把“调阈值/调少量参数”当主要优化方向
- 当前瓶颈更像是:
  - 训练信号不够稳
  - 特征对“是否该干预”刻画不足
  - 小样本下 val 选出的配置方差较大

### 同日补充: 特征表示小型对照

目标:
- 不改训练协议，只改特征表示
- 判断当前 gate 的主要瓶颈是不是来自类别特征编码方式，或少数疑似噪声特征

对照 1: `one-hot` 替代原版 `int` 类别编码

产物:
- 汇总:
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_cleanval_80_20_multiseed_onehot_v1_summary.json`

汇总:
- `positive_gain_seed_count = 0 / 5`
- `avg_gain_over_none_mean = -0.0122`
- `avg_gain_over_none_std = 0.0043`
- `avg_regret_mean = 0.0426`
- `two_stage_accuracy_mean = 0.3304`
- `binary_intervene_accuracy_mean = 0.5131`
- `intervene_precision_mean = 0.5488`
- `intervene_recall_mean = 0.6910`

判断:
- 单独把类别特征从 `int` 改成 `one-hot` 没有带来提升
- 这版比原版 `int` 更差，因此不作为后续主线

对照 2: 保留原版 `int` 编码，仅删除 3 个字段

配置:
- `categorical_encoding = int`
- `drop_features = history_token_count, step_index, retrieval_rrf_score`

产物:
- 汇总:
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_cleanval_80_20_multiseed_int_drop3_v2_summary.json`
- 各 seed 产物:
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_cleanval_80_20_multiseed_int_drop3_v2_seed123_*`
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_cleanval_80_20_multiseed_int_drop3_v2_seed231_*`
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_cleanval_80_20_multiseed_int_drop3_v2_seed341_*`
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_cleanval_80_20_multiseed_int_drop3_v2_seed451_*`
  - `gate_models/scienceworld_core_train_gate_two_stage_regression_expanded134_cleanval_80_20_multiseed_int_drop3_v2_seed561_*`

每个 seed 的 outer test 指标:
- `seed=123`
  - `avg_gain_over_none = -0.0115`
  - `avg_regret = 0.0430`
- `seed=231`
  - `avg_gain_over_none = -0.0268`
  - `avg_regret = 0.0653`
- `seed=341`
  - `avg_gain_over_none = -0.0051`
  - `avg_regret = 0.0403`
- `seed=451`
  - `avg_gain_over_none = -0.0285`
  - `avg_regret = 0.0526`
- `seed=561`
  - `avg_gain_over_none = -0.0116`
  - `avg_regret = 0.0349`

汇总:
- `positive_gain_seed_count = 0 / 5`
- `avg_gain_over_none_mean = -0.0167`
- `avg_gain_over_none_std = 0.0104`
- `avg_regret_mean = 0.0472`
- `avg_regret_std = 0.0083`
- `two_stage_accuracy_mean = 0.3217`
- `binary_intervene_accuracy_mean = 0.5652`
- `intervene_precision_mean = 0.5766`
- `intervene_recall_mean = 0.8210`

判断:
- 删除 `history_token_count`、`step_index`、`retrieval_rrf_score` 后，结果进一步变差
- 虽然 `binary_intervene_accuracy` 略升，但 `intervene_recall` 明显升高，说明模型更偏向“多干预”
- 在这种更激进的干预策略下，`avg_gain_over_none_mean` 反而降到 `-0.0167`，说明这不是有用改进

这一轮特征对照后的结论:
- 当前三版里仍然是原版 `int` 特征最好
- `one-hot` 不是改进方向
- 删除这 3 个字段也不是改进方向
- 因此后续主线继续保留原版特征集，不采用 `one-hot` 或 `int + drop3`

---

## 2026-04-21 gate arm 语义对齐修复

问题性质:
- 这是一个实现语义错配问题，不是普通调参问题
- 用户最终确认的目标语义是:
  - `none`: 不注入记忆
  - `question`: 只注入一次 `question`
  - `repair`: 只注入一次，且注入形式必须与实验 A 完全一致
- rollout 与 online eval 必须严格同构

修复前发现的问题:
- `branched rollout`
  - `question` / `repair` 都走 `hint_text`
  - `repair` 不是实验 A 的标准 memory block 注入
  - 但 rollout 默认只注入 `1` 步
- `online eval`
  - `question` / `repair` 也都走 `hint_text`
  - `repair` 没走实验 A 的 `retrieved_memories` 注入路径
  - 还会把同一条记忆连续注入 `3` 步
- 因此:
  - rollout 的 arm utility
  - online eval 的真实执行 policy
  - 两者不一致

这意味着什么:
- 旧 rollout / judge / gate training 结果，对“目标 policy”不再严格有效
- 原因不是数据脏，而是 arm 本身的执行语义变了
- 尤其是 `repair`:
  - 旧语义更接近“一次 hint”
  - 新目标语义是“实验 A 标准 memory block 注入”

本次代码修复:
- `src/gate/branched_rollout.py`
  - 改成单步注入 payload:
    - `none`: 无注入
    - `question`: 单步 `hint_text`
    - `repair`: 单步 `retrieved_memories=top3_memories`
  - `repair` 现在走标准 `build_user_prompt(..., retrieved_memories=top3_memories, memory_style=\"original\")`
- `experiments/gate/run_online_eval.py`
  - 改成与 rollout 同构的单步注入
  - 移除连续 `3` 步 `pending_hint` 逻辑
  - `repair` 改为标准 memory block 注入
  - `question` 改为单步 question hint
  - 额外落盘 `gate_decisions`，用于检查 gate 是否真的发生注入

修复后的统一语义:
- `none`
  - 不注入
- `question`
  - 只在下一步 prompt 注入一次 `question_text`
- `repair`
  - 只在下一步 prompt 注入一次标准 memory block
  - 其格式与实验 A 保持一致
  - 注入条数也与实验 A 保持一致，即 `top3`

对当前实验链的影响:
- 这次修复之后，旧 rollout 产物不应再直接拿来支撑正式 gate 结论
- 如果后续正式采用这套对齐后的 arm 语义，则必须重新执行:
  - branched rollout
  - judge scoring
  - gate training
- 在此之前，已有 gate 结果只能视为“旧语义下的探索结果”

强调:
- 这是本轮实验中一个关键实现问题
- 后续所有正式 gate 实验，必须默认检查:
  - `repair` 是否与实验 A 同构
  - 注入是否只发生一次
  - rollout 与 online eval 是否完全一致

### 同步 smoke 验证

目的:
- 在不跑正式 rollout 的前提下，先确认 prompt 注入语义已经按目标对齐

验证方式:
- 直接构造真实 memory entry，检查生成的 prompt 片段
- 用 dummy rollout 跑 `2` 步，检查第 `1` 步和第 `2` 步注入 payload
- 用 online eval helper 模拟两步 prompt，检查注入是否在下一步后清空

关键结果:
- `none`
  - 不包含 memory block
  - 不包含 question hint
  - 第 `1/2` 步都无注入
- `question`
  - 第 `1` 步只注入 `question_text`
  - 不出现实验 A 的 memory block
  - 第 `2` 步不再保留 question 注入
- `repair`
  - 第 `1` 步出现实验 A 标准 memory block:
    - `You can refer to these past failure-recovery experiences ...`
    - `- failure: ..., fix: {entry.get_repair_display()}`
  - 第 `2` 步不再保留 repair 注入

本次 smoke 的直接结论:
- 单次注入已经成立
- `repair` 的 prompt 形式已经与实验 A 对齐
- `question` 的 prompt 形式已经收敛到“只注入一次 question”
- 现在可以进入“基于新语义重跑 rollout”的下一步

---

## 2026-04-21 ScienceWorld gate 主线当前状态与重跑决定

当前状态:
- 当前只讨论 `ScienceWorld`
- gate 主线里最关键的执行语义已经完成代码对齐:
  - `none`: 不注入
  - `question`: 只在下一步 prompt 注入一次 `question_text`
  - `repair`: 只在下一步 prompt 注入一次实验 A 同构的标准 memory block
  - `repair` 注入条数与实验 A 对齐，为 `top3`
- `rollout` 与 `online eval` 的注入步数、prompt 形式、检索条数现在已按目标语义统一
- `LLM judge` 的 `gold path` 匹配规则也已收紧为严格 `(task_type, variation_idx)` 精确匹配，不再允许“同 task 任取一条”的宽松兜底

已经完成的代码对齐点:
- `src/gate/branched_rollout.py`
  - `question` / `repair` arm 语义与目标协议对齐
- `experiments/gate/run_branched_rollouts.py`
  - 检索 `top_k` 改为读配置，不再写死
- `experiments/gate/run_online_eval.py`
  - 单步注入
  - `repair=top3` 标准 memory block
  - `question=top1 question`
  - `memory_entry_count` 改成按 `(task_type, env_idx)` bucket 统计
- `src/gate/llm_judge.py`
  - `gold path` 必须按 `task_type + variation_idx` 精确命中
- `experiments/gate/collect_checkpoints.py`
  - ScienceWorld 路径下的检索条数与配置保持一致

由此带来的正式结论:
- 旧的 `rollout`
- 旧的 `judge`
- 旧的 `gate training`
- 旧的 `online eval`

都不能再作为当前目标 policy 的正式结果引用。

原因不是简单“数据质量不好”，而是训练链条中的 arm 执行语义已经发生了关键变化:
- 旧版 `repair` 不是实验 A 的标准 memory 注入语义
- 旧版 `online eval` 还是多步重复注入
- 旧版 judge 对 `gold path` 的匹配也过于宽松

关于 `checkpoint` 是否需要重跑:
- 当前判断是: 原有 `ScienceWorld checkpoint collection` 可以继续复用
- 原因是 `collect_checkpoints.py` 记录的是:
  - failure detector 命中的决策点
  - 当时 episode 的历史、动作、观测和检索统计特征
- 它本身并不执行后续 `none/question/repair` 三臂分叉，因此不直接受到这次 arm 语义修复的影响
- 对 ScienceWorld 而言，checkpoint 收集路径本身已经是:
  - 规则型 failure detection
  - 单步 memory 注入
  - `task_type` 作用域检索
  这一段和当前主线语义没有本质冲突

因此，本轮决定是:
- 不再沿用旧的 gate 训练产物作为正式主线
- checkpoint 可以作为本轮重跑的输入继续使用
- 从 `branched rollout` 开始，按当前已对齐的新语义，重新走一遍后续训练数据链
- 并且这次显式把“checkpoint -> rollout -> gold path -> judge -> training”视为一个不可拆开的同语义闭环

### 本轮重跑主线

目标:
- 构造一版与当前 ScienceWorld 正式 policy 完全一致的 gate 训练数据
- 避免继续混用旧语义产物和新语义代码

执行顺序:

1. 复用当前有效 checkpoint，并按需补充
- 当前已经收集到的 ScienceWorld checkpoint 可以继续作为正式输入
- 如果后续判断 task 覆盖或 failure 类型覆盖不足，再单独补采 checkpoint
- 但这不是本轮语义修复后的必做重跑项

2. 重新进行 branched rollout
- 对当前选定的 checkpoint 跑三臂 `none/question/repair`
- 使用当前已经对齐的新 arm 语义
- `repair` 必须保持实验 A 同构的单步 `top3` memory block 注入

3. 根据本轮实际使用的 checkpoint 重新构造 gold path
- gold path 必须覆盖本轮实际参与 judge 的 `(task_type, variation_idx)`
- 不允许再复用只覆盖历史子集的旧 gold path 文件
- 不允许用“同 task 任取一条 path”的宽松替代

4. 对 rollout 重新做 LLM judge 打分
- judge 输入必须来自第 2 步的新 rollout
- gold path 必须来自第 3 步的新覆盖集合
- 这一步的输出才是当前正式可训练的监督数据

5. 整理数据并重新做 gate 模型训练
- 只使用第 4 步新生成的 judged rollout
- 后续 train/val/test 切分、特征选择、模型对照，都以这批新数据为唯一正式来源
- 在这之前，旧 gate 指标只能作为历史探索参考，不能作为正式结论

### 当前可复用与不可复用

可以直接复用:
- Step 4 实验 A 作为 `repair` 注入语义的基准定义
- 当前已经修复过的 ScienceWorld gate 代码路径
- 任务集合、split 协议、`task_type` 记忆隔离策略
- 已有且语义未失效的 ScienceWorld checkpoint 集

不能直接复用为正式产物:
- 旧 rollout 文件
- 旧 judged rollout 文件
- 旧 gate model
- 基于旧语义产物做出的正式性能判断

### 本轮执行要求

后续每次推进时，都必须额外核对以下四点:
- `repair` 是否仍然是实验 A 同构的标准 memory block
- 注入是否只发生一次
- `gold path` 是否精确覆盖到实际 checkpoint 的 `(task_type, variation_idx)`
- 训练数据是否全部来自同一轮重跑产物

这四点中任意一点不满足，该轮结果都不应进入正式汇报。

### Step 5.17: 确认正式 rollout 输入为 `expanded156 + varcap3`

目标:
- 在正式重跑 `Stage 1 branched rollout` 之前，先确认 checkpoint 分布、reference memory 和 rollout 参数都正确
- 避免继续沿用旧的 `expanded134` 或高重复 variation 的 checkpoint 子集

输入候选审查:
- 原始扩容版:
  - `checkpoints/scienceworld_core_train_checkpoints_sampled_ftaware_expanded_v3.json`
  - 总数 `156`
  - 但个别 `(task, variation)` 重复偏高，最坏达到 `5`
- 严格 `varcap=2` 版:
  - 重复控制最干净
  - 但 `boil` 从 `14` 掉到 `12`
  - 总量从 `156` 降到 `154`
- 严格 `varcap=3` 版:
  - `checkpoints/scienceworld_core_train_checkpoints_sampled_ftaware_expanded_v3_varcap3_strict.json`
  - 总数仍为 `156`
  - task 分布保持:
    - `11 x 14 + 1 x 2`
  - failure 分布保持:
    - `syntax_or_parse = 95`
    - `ambiguity = 37`
    - `precondition_blocked = 13`
    - `action_loop = 9`
    - `physics_or_affordance = 2`
  - 同时把最坏 variation 重复从 `5` 降到 `3`
  - 唯一 `(task, variation)` 对数量从 `85` 提升到 `87`

正式决定:
- 本轮 `Stage 1 branched rollout` 正式采用:
  - `checkpoints/scienceworld_core_train_checkpoints_sampled_ftaware_expanded_v3_varcap3_strict.json`
- 不再使用:
  - 原始 `expanded_v3`
  - 更激进的 `varcap=2`

reference memory 核对:
- 本轮正式 gate rollout 不再使用早期导出的:
  - `memory/scienceworld_core_train_memory.json`
- 原因:
  - 该文件是旧版 memory 导出，只含:
    - `failure_action`
    - `failure_observation`
    - `solution_action`
  - 不含当前 ScienceWorld memory extractor 已产出的 richer 字段:
    - `question_text`
    - `repair_strategy`
    - `repair_tactic`
    - `repair_action`
- 本轮正式 gate rollout 改用 Step 4 实验 B 第一段的正式 train memory:
  - `memory_store/20260416_163135/sw_epoch1.json`
- 该文件已核对:
  - `scope = task_type`
  - entry 数 = `533`
  - entry schema 包含:
    - `question_text`
    - `repair_strategy`
    - `repair_tactic`
    - `repair_action`
- 这才与当前真实系统中的 ScienceWorld memory 形态一致

原则强调:
- gate 数据应使用真实 memory、真实检索、真实 prompt 路径构造
- 因此这里不能为了让 rollout 更“好看”而改造 memory 内容
- 正式 gate rollout 应直接面对真实系统里的 retrieval 噪声
- 但必须使用语义正确、字段完整的正式 memory 文件

rollout 参数确认:
- benchmark:
  - `scienceworld`
- checkpoint:
  - `checkpoints/scienceworld_core_train_checkpoints_sampled_ftaware_expanded_v3_varcap3_strict.json`
- memory:
  - `memory_store/20260416_163135/sw_epoch1.json`
- config:
  - `config_scienceworld.yaml`
- arms:
  - 显式指定 `none cue question repair`
  - 不依赖默认值，避免后续 arm 列表变动造成歧义
- n_steps:
  - `15`
- n_replays:
  - `3`
- resolved checkpoint 输出:
  - `checkpoints/scienceworld_core_train_checkpoints_resolved_expanded156_varcap3_expbmem_v1.json`
- rollout 输出:
  - `rollouts/scienceworld_core_train_rollouts_expanded156_varcap3_expbmem_v1.json`

正式运行命令:

```bash
PATH=/opt/homebrew/opt/openjdk/bin:$PATH python experiments/gate/run_branched_rollouts.py \
  --benchmark scienceworld \
  --checkpoints checkpoints/scienceworld_core_train_checkpoints_sampled_ftaware_expanded_v3_varcap3_strict.json \
  --memory memory_store/20260416_163135/sw_epoch1.json \
  --config config_scienceworld.yaml \
  --output rollouts/scienceworld_core_train_rollouts_expanded156_varcap3_expbmem_v1.json \
  --resolved-checkpoints-output checkpoints/scienceworld_core_train_checkpoints_resolved_expanded156_varcap3_expbmem_v1.json \
  --n-steps 15 \
  --n-replays 3 \
  --arms none cue question repair \
  --run-name sw_core_rollout_expanded156_varcap3_expbmem_v1_
```

阶段判断:
- 当前 `Stage 1` 的正式输入已经确认
- 可以开始按这版配置重跑 branched rollout

### Step 5.18: `cp_0829` replay 失配定位与预检修复

目标:
- 解释正式 rollout 在 `35/156` 处停止的根因
- 给正式长程 rollout 补上“开跑前体检”机制，避免半路因坏 checkpoint 直接中止

问题复盘:
- 正式 rollout 在 `cp_0829` 处报错:
  - `Replay score mismatch before rollout: cp_0829, replay=63.0, saved=38.0, diff=25.0`
- `cp_0829` 的关键信息:
  - task = `grow-plant`
  - variation = `6`
  - `failure_action = wait`
  - `score_at_checkpoint = 38`
- 进一步审查发现:
  - `cp_0829.action_history` 中包含多次纯数字动作:
    - `0`
    - `0`
    - `0`
  - 这类动作本质上是 ScienceWorld 歧义菜单的临时索引选择
  - 如果前面状态有轻微漂移，后续 `0/1/...` 很可能指向不同对象，从而导致 replay 不再回到原始 checkpoint

当前结论:
- `cp_0829` 暴露的不是 prompt 注入问题
- 也不是 memory 检索范围问题
- 而是 checkpoint replay-stability 问题:
  - 部分 checkpoint 的 `action_history` 含原始菜单数字动作
  - 这类轨迹在 ScienceWorld 中不保证可稳定重放
  - 因而不应直接进入正式 branched rollout / gate 训练数据链
- 在当前 `expanded156_varcap3` 正式样本中，已确认有 `23/156` 个 checkpoint 含纯数字动作历史
  - 其中 `grow-plant` 占 `14`

代码修复:
- `experiments/gate/run_branched_rollouts.py`
  - 新增 `--reject-numeric-action-history`
    - 在 rollout 前剔除 `action_history` 中含纯数字动作的 checkpoint
  - 新增 `--replay-precheck`
    - 在正式 rollout 前，对剩余 ScienceWorld checkpoint 先做一次单次 replay 验证
  - 新增 `--replay-mismatch-tolerance`
    - 控制 replay score 与保存 score 的允许误差
  - 新增 `--rejected-checkpoints-output`
    - 将被预检拒绝的 checkpoint 单独落盘，并写明拒绝原因

阶段判断:
- 正式 ScienceWorld gate rollout 不应再直接用“未预检”的 checkpoint 集硬跑
- 后续正式命令应带上:
  - `--reject-numeric-action-history`
  - `--replay-precheck`
- 只有通过 replay 预检的 checkpoint，才进入后续:
  - branched rollout
  - gold path
  - LLM judge
  - gate training

---

## 后续记录模板

### [待填写] Step N

目标:
-

执行命令:

```bash
# command here
```

结果:
-

问题:
-

结论:
-

对其他 benchmark 的启发:
-

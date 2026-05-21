<!--
本文档用于讨论 React-FM 论文实验设计，记录当前判断、实验口径、消融方案和待定问题。
-->

# React-FM 实验设计讨论稿

> 创建日期: 2026-05-20  
> 状态: 讨论中  
> 用途: 先把实验怎么做、怎么讲、哪里有风险写清楚，后续边讨论边改。

## 1. 先说结论

这个实验包可以尝试投 CCF-B，但论文不能写成“React-FM 全面超过 Reflexion / ExpeL”。

更稳的说法是:

- React-FM 是一种在执行过程中发现失败、再临时调用失败恢复记忆的方法。
- 它适合长程任务，尤其是动作失败可以复用的任务。
- 它和 Reflexion / ExpeL 不是谁替代谁，而是适合不同错误类型。
- online 结果展示“边跑边学”的能力。
- offline 结果展示“不用 test 数据写记忆”的迁移能力。

当前最要注意的风险:

1. WebShop 正式结果必须等 full data / full index / official split 对齐。
2. ScienceWorld 上 ExpeL 的成功率更高，所以不能说 React-FM 在 ScienceWorld 最强。
3. online memory 是 test-time adaptation，不能和普通 frozen test 混在同一张公平主表里。
4. offline 如果在 test 期间继续写 memory，就不是严格 offline。
5. 消融实验必须一次只改一个因素，否则 reviewer 会觉得不公平。

## 2. 论文主张怎么写

主张一:

React-FM 能让 ReAct agent 在失败刚发生时利用历史经验纠错，而不是等 episode 结束后再总结。

主张二:

React-FM 在 ALFWorld、ScienceWorld 这类长程交互任务上更有价值；在 WebShop、HotPotQA 这类策略型或知识型任务上，Reflexion / ExpeL 可能更强。

不要写:

- React-FM 是所有 benchmark 最强。
- 检索命中率高就说明 memory 有效。
- online 和 offline 是同一种实验。
- test stream 里边跑边写 memory 还能叫普通 i.i.d. test。

## 3. 主实验怎么放

建议主实验先放在一张表里，不单独做 online 表。

原因:

- Reflexion 和 ExpeL 在当前实现里不是自然的 online 方法。
- 硬写 Reflexion Online / ExpeL Online 容易变成不真实的实验设置。
- React-FM Online 可以放进主表，但必须标注它是 online / test-time adaptation。

### 3.1 主实验表

这张表用于比较 ReAct、Reflexion、ExpeL、React-FM Offline 和 React-FM Online。

规则:

- ReAct、Reflexion、ExpeL、React-FM Offline 按 frozen / paper-aligned 口径跑。
- React-FM Online 单独标注为 online / transductive，不伪装成 frozen test。
- React-FM Offline 的 memory 只能来自 train 或 dev，test 期间不写新 memory。
- 同一个 benchmark 里使用相同 LLM、split、step limit、采样列表和评价指标。
- 多轮方法要说明真实成本或 effective cost。

方法:

| 方法 | 记忆来源 | Test 期间写 memory | 口径 | 说明 |
|---|---|---:|---|---|
| ReAct | 无 | 否 | frozen | 基础 baseline |
| Reflexion | train/dev 或规定 trial | 否 | frozen / multi-trial | episode 级反思 |
| ExpeL | train/dev 轨迹 | 否 | frozen | 批量抽 insight |
| React-FM Offline | train/dev 失败恢复记忆 | 否 | frozen | 测 memory 迁移 |
| React-FM Online | 空 memory 起步 | 是 | online / transductive | 测边跑边积累 |

## 4. 各数据集当前口径

### ALFWorld

定位: 主优势数据集。

建议:

- 所有方法用同一批测试任务。
- 如果成本允许，用完整 `valid_unseen`，共 134 个任务；这个规模其实不算大，也是 Reflexion 原论文常用口径。
- 如果成本不够，先固定一个小测试子集，比如每类任务 10 个，共 60 个；必须保存 env id 列表，后面所有方法都用同一个列表。
- 指标主要看 success rate，再补 steps / tokens。
- 如果某个方法对同一批 test 任务反复重试，要标注为 retry / online，不要说成普通一次性 frozen test。

ALFWorld 上各方法建议这样做:

| 方法 | 怎么跑 | 注意点 |
|---|---|---|
| ReAct | 直接在固定 test 集上跑一轮 | 这是最干净的 baseline |
| Reflexion | 按任务级别多轮 retry；失败后生成 reflection，下轮同一个任务带 reflection 重试 | 原论文 ALFWorld 是 134 个任务、最多 12 consecutive trials，memory 保留最近 3 条 reflection；我们可以先小规模跑 3-5 轮，正式对比最好跑到 12 轮或说明轮数 |
| ExpeL | 先做 experience gathering，再抽 insight，再在 test 集上评估 | 不建议直接用 test 上 Reflexion 的结果抽 insight；更稳的是在 train/开发子集上收集 Reflexion 风格轨迹和成功经验，再抽 insight，test 只读使用 |
| React-FM Online | 空 memory 起步，在固定 test stream 上边跑边写 memory | 标注 online / transductive |
| React-FM Offline | 先在 train/开发子集收集 failure-recovery memory，再在固定 test 集上只读评估 | test 期间不要写新 memory；如果代码还会写，需要加 read-only 开关 |

关于 ExpeL:

- 原始流程不是“直接拿 Reflexion test 结果再跑一遍”。
- 更准确是三步: experience gathering -> insight extraction -> evaluation。
- ALFWorld insight extraction 可参考 ExpeL 官方设置: `max_num_rules=10`，`success_critique_num=8`。
- evaluation 时把抽出的 insights 和成功经验放在 episode 开头使用。
- 如果我们复用 Reflexion 结果，最好复用 train/开发子集上的 Reflexion 轨迹，而不是 test 轨迹。

### WebShop

定位: 对齐和边界数据集。

正式协议:

- full product data。
- full search index。
- `human_goals=1`。
- 官方 test split。
- success = `reward == 1.0`。
- task score = `100 * avg_reward`。
- `max_steps=100`。
- 使用官方 wrapper / valid-action interface。

当前问题:

- full WebShop 数据和完整索引还没准备好。
- 1000-product preview 只能做 smoke test，不能做正式论文主结果。

论文里可以这样讲:

- WebShop 不一定要求 React-FM 最强。
- 如果 Reflexion / ExpeL 更强，正好说明策略型购物任务更适合抽象经验。

WebShop 上各方法建议这样做:

| 方法 | 怎么跑 | 注意点 |
|---|---|---|
| ReAct | 在固定 test 子集上跑一轮 | 建议先用官方 test split 固定 100 条；保存 sample ids |
| Reflexion | 对同一批 test 指令做多 trial；失败后生成 reflection，下个 trial 对同一 shopping instruction 重试 | 当前脚本默认 4 trials；要报告最终 trial 结果和真实 token 成本 |
| ExpeL | 用 train split 固定 100 条做 experience gathering，抽 shopping insights，再在固定 test 100 上评估 | 不要用 test 轨迹抽 insight；如果先用 preview 数据，只能算 smoke |
| React-FM Online | 空 memory 在固定 test stream 上跑，前面 test episode 的 memory 可帮助后面 episode | 标注 online / transductive；WebShop 使用共享 `shopping` memory |
| React-FM Offline | 先用 train split 固定 100 条收集 frozen memory，再在固定 test 100 上只读评估 | test 期间不写 memory；memory scope 用共享 `shopping` |

WebShop 测试集建议:

- 开发版: 官方 test split 固定 100 条，seed 固定，比如 42。
- 正式版: 如果 full data 和时间允许，扩到官方 500 条 test。
- 所有方法都必须使用同一份 sample ids。
- 1000-product preview 不进正式主表。

### ScienceWorld

定位: 有收益但不能吹太满的数据集。

当前 Step 4 核心 test:

- 12 个核心 task type。
- 111 episodes。
- 指标: success rate、原始 avg score、tokens / episode。

已有结果:

| 方法 | Success Rate | Avg Raw Score | Tokens / Episode |
|---|---:|---:|---:|
| ReAct | 37.84% | 47.21 | 185.5K |
| Reflexion P1 Adj | 36.94% | 49.39 | 真实成本下界约 212.6K |
| ExpeL E2 | 49.55% | 57.78 | effective 190.9K |
| React-FM Offline | 39.64% | 48.83 | 180.4K |
| React-FM Online | 41.44% | 55.24 | 197.2K |

注意:

- `doc/scienceworld_step4_summary_cn.md` 里的 React-FM online `64.25` 是旧 summary 口径。
- 按源 JSON 逐 episode 重算是 `55.24`。
- 正式论文和图表都应该用 `55.24`。
- ScienceWorld 上可以说 React-FM 比 ReAct 好，但不能说超过 ExpeL。

ScienceWorld 上各方法建议这样做:

| 方法 | 怎么跑 | 注意点 |
|---|---|---|
| ReAct | 在 Step 4 核心 test 上跑一轮 | 当前核心 test 是 111 episodes，规模可以接受 |
| Reflexion | 每个 episode 先跑 pass 0；失败后生成 reflection，再跑 pass 1 | 当前 ScienceWorld 方案就是两遍；报告时要用 adjusted 口径，避免把 pass0 已成功的任务又重跑坏 |
| ExpeL | Epoch 1 收集轨迹，抽 task-level insights；Epoch 2 注入 insight 再跑 | 当前已有 Step 4 ExpeL；它在成功率上强于 React-FM |
| React-FM Online | 空 memory 起步，在核心 test stream 上边跑边写 memory | 标注 online / transductive；适合展示在线积累 |
| React-FM Offline | 在核心 train 上先收集 failure-recovery memory，再在核心 test 上只读评估 | 旧实验如果 test 期间继续写 memory，要改名为 offline-initialized online，或补真正 read-only 版 |

ScienceWorld 测试集建议:

- 先沿用 Step 4 核心 test: 12 个 task type，共 111 episodes。
- 不建议再缩太小，因为分 task 后每类本来就只有 5-10 个 variation。
- 如果加对照任务，放附录，不影响主表。

### HotPotQA

定位: 边界数据集。

建议:

- 固定 validation/dev 子集。
- 开发版可以先用 100 或 200 samples。
- 正式版如果成本允许，再扩到 500 samples。
- 指标用 EM / F1。
- ReAct、Reflexion、ExpeL、React-FM 使用同一 Search / Lookup / Finish 接口。
- offline memory 不能来自 evaluation subset。

论文里可以这样讲:

- HotPotQA 的知识内容差异大，动作级失败记忆不一定好迁移。
- 如果 ExpeL / Reflexion 更强，也符合预期。

HotPotQA 上各方法建议这样做:

| 方法 | 怎么跑 | 注意点 |
|---|---|---|
| ReAct | 在固定 validation/dev 子集上跑一轮 | 保存 question ids 或固定 seed，所有方法一致 |
| Reflexion | 对同一批问题做 2 trials；失败后生成 reflection，下个 trial 对同一问题重试 | 当前脚本默认 2 trials；这是合理的成本上限 |
| ExpeL | 先用非评测子集收集 QA 轨迹并抽 search/lookup insights，再在固定评测子集上评估 | 不要用 eval subset 抽 insight；否则变成 test leakage |
| React-FM Online | 空 memory 在固定 evaluation stream 上跑，边跑边写 memory | 标注 online / transductive；预计收益有限 |
| React-FM Offline | 用 train/开发子集收集 failure-recovery memory，再在固定 eval 子集只读评估 | 检查 memory 是否只是实体级替换，避免误导 |

HotPotQA 测试集建议:

- 先用固定 100 或 200 个问题做开发版。
- 如果结果稳定，再扩到 500 个问题做正式版。
- 所有方法必须用同一批问题。
- 主要看 EM / F1，不要只看 success rate。

## 5. 消融和分析重新设计

这部分先推翻旧版“注入时机 / 记忆内容 / 检索方式”三件套。

原因:

- `Episode-start memory` 和 `Always-on memory` 很难定义得公平。
- 不同 memory 形式天然适合不同用法，硬比较会混淆变量。
- Reflexion / ExpeL 本身已经在主实验里代表 reflection 和 insight，不需要再做一遍“记忆内容消融”。
- offline / online 本身已经在主实验里出现，也不需要再重复做成消融。

### 5.1 核心消融: 相关记忆是否真的有用

这个是最重要、也最值得做的消融。

固定:

- 使用同一批 offline failure-recovery memory。
- 使用同一批 test episodes。
- test 期间不写新 memory。
- 失败触发点保持一致。
- LLM、step limit、top-k、prompt 模板保持一致。

比较:

| 变体 | 说明 | 回答的问题 |
|---|---|---|
| ReAct | 无 detector，无 memory | 基础表现 |
| Warning only | 检测到失败后，只提示“上一步可能失败”，不给 memory | 是不是只要提醒失败就够了 |
| Random memory | 检测到失败后，随机给一条 failure-recovery memory | 是不是随便多给上下文就有用 |
| React-FM | 检测到失败后，给相关 failure-recovery memory | 相关失败经验是否真正有用 |

这个消融的好处是清楚:

- `Warning only` 控制“失败提醒”的作用。
- `Random memory` 控制“多给一段上下文”的作用。
- `React-FM` 才是真正测试“相关记忆”的作用。

### 5.2 注入时机暂不作为主消融

这块先不放主实验。

原因:

- `Episode-start memory` 必须在 episode 开头就检索 memory，但这时还没有失败动作和失败反馈，只能用任务描述检索。这和 React-FM 的 failure-state retrieval 不是同一个问题。
- `Always-on memory` 如果每一步都检索，会大幅增加检索次数和 token；如果只复用开头 top-k，又可能和当前状态无关。
- 所以它们更像 stress test，不适合作为主消融证明。

如果后面想简单做，可以放附录:

| 变体 | 合理定义 | 解释方式 |
|---|---|---|
| Episode-start failure memory | episode 开头用任务描述检索 top-k failure-recovery memory | 不是严格公平消融，只看提前给 memory 是否会造成帮助或噪声 |
| Always-on failure memory | 每一步用当前 observation/action history 检索并注入 top-k | 主要看过度注入的成本和噪声，不作为核心 claim |

### 5.3 记忆形式不单独做消融

这部分和主实验高度重合。

- Reflexion 已经代表 `reflection`。
- ExpeL 已经代表 `insight`。
- React-FM 已经代表 `failure-recovery memory`。

如果再做一张 memory form 表，很容易变成重复主实验，而且很难保证每种记忆形式使用同一种自然注入方式。

所以这里不做主消融。论文中可以在主结果分析里讨论:

- 为什么 failure-recovery 更适合 ALFWorld / ScienceWorld。
- 为什么 reflection / insight 可能更适合 WebShop / HotPotQA。

### 5.4 Offline / Online 不单独做消融

这也已经在主实验里体现。

主表里保留:

- `React-FM Offline`: train/dev 收集 memory，test 只读。
- `React-FM Online`: 空 memory 起步，test stream 边跑边积累。

如果时间够，可以补一个附录设置:

- `React-FM Offline+Online`: 先加载 train/dev memory，test 中继续积累。

但它不是必须项。

### 5.5 必做分析: 失败恢复

这个很有必要，应该作为 React-FM 的核心分析。

统计:

- 检测到失败后，下一步动作是否改变。
- 检测到失败后，后续是否恢复成功。
- 检索 memory 后平均几步恢复。
- 重复无效动作是否减少。
- 哪些显式失败最容易被修复，比如 `Nothing happens`、invalid action、repeated action。

这部分回答:

> React-FM 不是只提高最终成功率，而是在失败刚发生时改变了 agent 的局部行为。

### 5.6 必做分析: 记忆生成、检索和使用质量

这里只看检索质量不够，还要看两件事：第一，从轨迹里抽出来的 memory 本身是否高质量；第二，模型拿到 memory 后是否真的生成了合理动作。

建议把一次 memory pipeline 拆成三层:

| 层级 | 要看什么 |
|---|---|
| Memory generation quality | 从轨迹抽出的 memory 是否准确、可执行、可迁移 |
| Retrieval quality | 检索到的 memory 是否和当前失败相关 |
| Generation / usage quality | Agent 是否根据 memory 生成了合理下一步动作 |

可以人工抽样标注两类对象:

- 每个主要数据集抽 30-50 条 memory entry，标注 memory 生成质量。
- 每个主要数据集抽 30-50 次 memory injection，标注检索和使用质量。

memory 生成质量标注类别:

- `Valid and actionable`: 记忆准确，并给出可执行修正。
- `Correct but too specific`: 记忆正确，但太绑定具体物体或场景。
- `Vague reflection`: 记忆太泛，不够可执行。
- `Wrong repair`: 修正动作错误或不可执行。
- `Duplicate / low-value`: 重复或信息量低。

检索和使用质量标注类别:

- `Relevant + used correctly`: 记忆相关，下一步动作也合理。
- `Relevant but not used`: 记忆相关，但模型没听。
- `Relevant but wrong generation`: 记忆相关，但模型生成了错误动作。
- `Irrelevant memory`: 检索本身不相关。
- `Harmful memory`: 记忆误导了 agent。

这部分比单纯汇报 retrieval hit 更有价值，因为它能区分三种问题：memory 抽错、memory 检索错、memory 用错。

### 5.7 Task-level 分析暂不作为重点

这部分不强做。

原因:

- 不是所有数据集都有清楚的 task type。
- WebShop 基本都是 shopping，HotPotQA 基本都是 QA。
- ScienceWorld / ALFWorld 可以做分 task 观察，但不要把它包装成通用 task-level 实验。

可以保留在结果分析里，用来解释正负迁移。

### 5.8 简单做 memory size / top-k 分析

这部分可以简单做，不要太大。

建议只在 ALFWorld 或 ScienceWorld 上做:

| 设置 | 取值 |
|---|---|
| top-k | 1 / 3 / 5 |
| memory size | 小 / 中 / 全量，例如 25% / 50% / 100% |

目的:

- 看 memory 太少是否覆盖不够。
- 看 top-k 太大是否引入噪声。
- 给 reviewer 一个“我们检查过超参数敏感性”的信号。

## 6. 最小实验包

必须做:

| 实验 | 数据集 | 目的 |
|---|---|---|
| 主实验表 | ALFWorld, WebShop, ScienceWorld, HotPotQA | 比较 baselines、React-FM offline、React-FM online |
| 相关记忆消融 | ALFWorld, ScienceWorld | 比较 warning only / random memory / relevant memory |
| 失败恢复分析 | ALFWorld, ScienceWorld | 看检测失败后是否真的恢复 |
| 记忆生成、检索和使用质量分析 | ALFWorld, ScienceWorld，可选 WebShop | 区分 memory 抽错、检索错、记忆没用上、生成错 |
| memory size / top-k 分析 | ALFWorld 或 ScienceWorld | 简单检查记忆规模和 top-k 敏感性 |
| case study | ALFWorld, ScienceWorld | 展示 memory 如何改变动作 |
| cost table | 全部主实验 | 说明成本 |

可以放附录:

- ScienceWorld 对照任务。
- 注入时机 stress test: episode-start / always-on。
- React-FM Offline+Online。
- WebShop 500-test 完整结果。
- oracle retrieval / human audit。

## 7. 建议执行顺序

1. 先冻结每个 benchmark 的 split、采样列表、step limit、指标。
2. 跑 ReAct、Reflexion、ExpeL。
3. 跑 React-FM frozen offline。
4. 跑 React-FM online。
5. 做 ALFWorld + ScienceWorld 的相关记忆消融。
6. 做失败恢复、记忆生成质量、检索质量、生成质量分析。
7. 简单补 top-k / memory size 分析。
8. 最后整理 case study 和 token cost。

## 8. 现在最该讨论的三个问题

1. ScienceWorld 旧 offline 是否改名为 `offline-initialized online`，再补一个真正 test-time read-only 的 offline？
2. WebShop 正式主表是先用固定 100-test，还是必须等完整 500-test？
3. 相关记忆消融先只做 ALFWorld + ScienceWorld，WebShop / HotPotQA 是否只做边界分析？

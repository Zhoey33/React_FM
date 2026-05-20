# ScienceWorld 上 Agent Memory / Experience 策略调研汇总

> 更新日期：2026-04-27
> 说明：ScienceWorld 没有一个统一的、官方的 “agent memory strategy” leaderboard。不同论文使用的任务子集、variant 数、模型 backbone、训练/测试方式、是否允许 test-time adaptation 都不一致。因此下表只能作为**文献中报告结果的横向参考**，不能作为严格公平排名。

## 1. 关键结论

1. **不能只看分数排名**：ScienceWorld 很大，包含 30 个 task types、约 11k+ variations；很多论文只跑了子集。
2. **任务覆盖差异很大**：
  - 全 30 tasks：SwiftSage、DAVIS、GPT-J history model、ICRL、GraSP 等。
  - 18-task 子集：SSO、CLIN、EMPO²。
  - 小任务子集：AutoRefine 只评 Boil 和 Temperature Measurement 两类任务。
3. **memory 定义不统一**：有些方法是 textual memory / reflection，有些是 skill library，有些是 retrieval / KG / offline RL critic / in-context RL。
4. **最适合按组比较**：
  - 全 30 tasks 高分方法：ICRL、SwiftSage、GraSP、DAVIS、GPT-J history model。
  - 18-task 子集高分方法：SSO、EMPO²、CLIN。
  - 小任务子集但表现强：AutoRefine。
  - 指标不同不能混排：SkillGen 等。

## 2. 可粗略按 ScienceWorld score / reward 排序的方法


| 粗排  | 方法                            | 论文日期                   | ScienceWorld 报告分数                             | 任务覆盖 / 评测设置                                                                                                                                                                                                                                                                                                                  | Memory / Experience 策略                                                                                              |
| --- | ----------------------------- | ---------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| 1   | **ICRL Preset**               | 2025-05-21 / ICLR 2026 | **88 ± 0.7** running max return               | **全部 30 tasks**；GPT-4.1-mini；多 trial test-time self-improvement                                                                                                                                                                                                                                                              | 将历史 trajectory、action、observation、reward 放入上下文，让模型进行 in-context reinforcement learning                              |
| 2   | **SwiftSage**                 | 2023-05-27             | **84.7**                                      | **30 task types**                                                                                                                                                                                                                                                                                                            | Swift 模块用 behavior cloning / oracle trajectories；Sage 模块用 GPT-4 做 subgoal planning 与 replanning；不算纯 memory，但强依赖轨迹学习 |
| 3   | **GraSP**                     | 2026-04-20             | **84.9 seen / 81.3 unseen**，DeepSeek V3.2 主结果 | **30 task types**；论文报告 ScienceWorld seen / unseen                                                                                                                                                                                                                                                                            | 将 skill library 编译成 executable skill graph；skill retrieval + DAG execution + local repair                           |
| 4   | **SSO**                       | 2024-02-05             | **83.7 adaptation / 60.1 transfer**           | **18 tasks**：Temperature, Melting Temp, Find Plant, Find Living, Chemistry, Color Mixing, Lifespan Longest/Shortest, Life Stages Plant/Animal, Boil, Freeze, Grow Plant/Fruit, Gravity, Friction, Genetics Known/Unknown。Adaptation：单 variant 跑 5 trials；Transfer：10 train variants 学 30 iterations，再测 heldout test variants | 从高 reward 子轨迹抽取 transferable skills，并持续 pruning / refining                                                          |
| 5   | **KnowMap**                   | 2025-06-24             | **76.25**                                     | 论文描述 ScienceWorld 有 **30 task types**、每类 100+ variants；表中与 SayCan / ReAct / Reflexion / SwiftSage / ReasonPlanner 对比                                                                                                                                                                                                         | 从 environmental + experiential data 动态构建 knowledge base，并训练小 embedding model 做检索                                    |
| 6   | **EMPO²**                     | 2026-02-26             | **75.9**                                      | **18 task entries**，覆盖 Chemistry / Classification / Biology / Electricity / Measurement；每任务前 5 variants 训练，20 unseen test variants 测试                                                                                                                                                                                        | Memory-augmented on/off-policy RL；训练后测试时报告 without memory performance                                               |
| 7   | **AutoRefine**                | 2026-01-30             | **70.4 ± 1.9**                                | **只评 2 类 ScienceWorld 任务**：Boil / Task 1-1，30 variations；Temperature Measurement / Task 2-1，540 variations                                                                                                                                                                                                                   | 从 trajectories 抽取 reusable expertise：procedural subagents + skill patterns，并维护 / 剪枝 repository                      |
| 8   | **STEP**                      | 2024-11-13             | **67.4**                                      | **18 tasks**；论文称完成 12 / 18 tasks                                                                                                                                                                                                                                                                                           | Planner / Executor / Evaluator / Memory 四模块；Memory 存储 previous experiences，Evaluator 用 learned rules 约束 action        |
| 9   | **DAVIS / ReasonPlanner**     | 2024-10-11             | **65.06**                                     | **全部 30 task types**；RAG agents 用每任务 5 个 golden trajectory variations 构建 memory，共 150 train variations；测试每任务随机 3 variants，共 90 test variations                                                                                                                                                                               | Temporal knowledge graph memory + inner monologue retrieval + actor-critic planning                                 |
| 10  | **GPT-J history model**       | 2023-10-30             | **62.57 / 63.35**                             | **全部 30 classes**；ScienceWorld test set **1819 games**；训练用 7359 games 或去 variation 设置                                                                                                                                                                                                                                        | 将尽可能多的历史 steps 填入 LLM context；不是复杂 memory manager，但证明长历史上下文非常重要                                                     |
| 11  | **CLIN**                      | 2023-10-16             | **62.2 adaptation**                           | **18 tasks**，9 类 × 2 task instances；总计 164 task-environment combinations；GEN-ENV 用每任务 10 train environment settings 再测 unseen                                                                                                                                                                                                | Persistent dynamic textual memory；核心是 causal abstractions，而不是普通 reflection hints                                    |
| 12  | **Retrospex**                 | 2025-05-17             | **55.98**                                     | 主文按复杂度聚合：Short <20 steps、Medium 20–50、Long >50；论文说完整 30 sub-tasks 在 appendix                                                                                                                                                                                                                                                 | 不把经验直接塞进上下文，而是用 past experiences 训练 offline RL critic，再做 dynamic action rescoring                                   |
| 13  | **MCMA**                      | 2026-01-12             | **51.95 Dev / 48.60 Test**，Qwen3-32B          | 使用 ScienceWorld 标准 Dev/Test split；论文片段未列具体 task list                                                                                                                                                                                                                                                                         | 学习 memory abstraction：memory copilot 决定如何结构化、抽象、复用 memory                                                           |
| 14  | **Knowledge-enhanced Agents** | 2023-05-08             | 未整理成统一排名分数；表按模型/任务给 reward                    | **10 tasks** from ScienceWorld                                                                                                                                                                                                                                                                                               | 注入 previous correct actions memory + object affordances，可接 RL / LM agents                                           |


## 3. 指标不同，建议单独列的方法

这些方法也与 memory / experience / skill reuse 相关，但使用的指标不是统一的 ScienceWorld 0–100 score / reward，或者当前检索到的信息不足以放入同一排名。


| 方法                                   | 日期         | ScienceWorld 指标 / 结果                                                         | 任务覆盖                            | 备注                                                                                 |
| ------------------------------------ | ---------- | ---------------------------------------------------------------------------- | ------------------------------- | ---------------------------------------------------------------------------------- |
| **SkillGen**                         | 2025-11-18 | GPT-4o-mini：PR **57.6%**, SR **41.1%**；Qwen-Turbo：PR **59.4%**, SR **45.5%** | ScienceWorld 表，但当前片段未明确列 task 数 | 报 Grounding Rate / Progress Rate / Success Rate / AUPC，不是 0–100 ScienceWorld score |
| **Embodied Planner-R1**              | 2025-06-29 | completion rate **79.92%**                                                     | ScienceWorld；unseen environments drop -3.66% | Outcome-driven RL / Interactive Policy Optimization；偏 RL 训练，不是纯 memory 策略 |
| **RLVMR**                            | 2025-07-30 | 摘要称 7B model 在最难 unseen split 达 **83.6% success rate**                     | ALFWorld + ScienceWorld；具体 ScienceWorld split 需核表 | 用 verifiable meta-reasoning rewards 奖励 planning / exploration / reflection 等过程标签 |
| **BPO**                              | 2025-08-05 | 摘要称在 ALFWorld / ScienceWorld / WebShop 达到 SOTA                              | ScienceWorld；具体任务覆盖与分数需核表 | Data curation flywheel；reward-gated rejection sampling 选择经验，训练 long-horizon planner |
| **TDP: Task-Decoupled Planning**     | 2026-01-12 | 摘要称优于强 baseline，并减少 token up to 82%                                      | TravelPlanner / ScienceWorld / HotpotQA；具体 ScienceWorld 表需核查 | 将任务分解为 sub-goal DAG，局部上下文规划与 replanning；更偏 planning/context isolation |
| **AEC**                              | 2026-02-03 | 摘要称 competitive success with fewer replanning rounds                           | ALFWorld + ScienceWorld；具体分数需核查 | Active Epistemic Control；维护 grounded fact store / belief store，降低错误 belief 对规划的污染 |
| **LGE: Language Guided Exploration** | 2024-03-05 | 摘要称优于 vanilla RL / BC / TDT                                                  | ScienceWorld                    | 更偏 RL exploration，不是 memory 策略                                                     |
| **Neoplanner**                       | 2023-12-12 | 摘要称比 current best 平均 reward 提升 124%                                          | multiple tasks                  | LLM + state-space search + textual entity-relation learnings；需要进一步核对表格             |
| **SkillNet**                         | 2026-02-26 | 使用 ScienceWorld，但需进一步核对具体表                                                   | 多 benchmark                     | skill creation / evaluation / connection，相关但尚未确认 ScienceWorld 单项分数                 |
| **RPMS**                             | 2026-03-18 | ScienceWorld avg. score **54.0** vs ReAct baseline **44.9**                      | Adapted to ScienceWorld；具体任务覆盖需核查 | Rule-augmented memory synergy；rule retrieval + belief-state memory gating + rules-first arbitration |


## 4. 按任务覆盖重新分组

### 4.1 全 30 tasks / 近似全覆盖


| 方法                  | 分数                      | 说明                                                         |
| ------------------- | ----------------------- | ---------------------------------------------------------- |
| ICRL Preset         | 88 ± 0.7                | 全 30 tasks；多 trial test-time adaptation；running max return |
| SwiftSage           | 84.7                    | 全 30 task types；强规划 + BC 轨迹学习                              |
| GraSP               | 84.9 seen / 81.3 unseen | 全 30 task types；skill graph orchestration                  |
| DAVIS               | 65.06                   | 全 30 task types；每任务 3 test variants，总 90 test variations   |
| GPT-J history model | 62.57 / 63.35           | 全 30 classes；1819 test games                               |


### 4.2 18-task 子集


| 方法    | 分数                              | 说明                                                             |
| ----- | ------------------------------- | -------------------------------------------------------------- |
| SSO   | 83.7 adaptation / 60.1 transfer | 18 tasks；skill extraction + skill pruning                      |
| EMPO² | 75.9                            | 18 task entries；每任务 5 train variants + 20 unseen test variants |
| STEP  | 67.4                            | 18 tasks；stepwise planning + experience memory；完成 12 / 18 tasks     |
| CLIN  | 62.2 adaptation                 | 18 tasks；persistent causal memory                              |


### 4.3 小任务子集


| 方法         | 分数         | 说明                                                         |
| ---------- | ---------- | ---------------------------------------------------------- |
| AutoRefine | 70.4 ± 1.9 | 只跑 Boil 和 Temperature Measurement 两类任务；不能直接与全 30-task 方法比较 |


### 4.4 标准 split 但任务明细不足


| 方法   | 分数                     | 说明                                           |
| ---- | ---------------------- | -------------------------------------------- |
| MCMA | 51.95 Dev / 48.60 Test | 使用标准 Dev/Test split，但当前论文片段未明确列出具体 task list |


## 5. 方法简述

### ICRL

- **核心思想**：把过往尝试的 actions、observations、rewards 和 final outcome 放入 context，让 LLM 在推理时根据奖励进行自我改进。
- **优点**：全 30 tasks，分数高。
- **注意**：多 trial / running max 设置与单次 episode 评测不可直接比较。

### SwiftSage

- **核心思想**：双系统架构，Swift 负责快速 routine action，Sage 负责慢速 deliberative planning。
- **优点**：全 30 task types，经典强 baseline。
- **注意**：依赖行为克隆和大模型规划，不是纯 memory manager。

### GraSP

- **核心思想**：将 flat skill library 编译为带前置条件和效果边的 DAG，执行时可局部 repair。
- **优点**：强调 skill orchestration，而不只是更多 skill。
- **注意**：结果跨多个 LLM backbone；主表中 ScienceWorld 有 seen / unseen。

### SSO

- **核心思想**：从高 reward 子轨迹中抽取 transferable skills，作为 in-context policy improvement 的载体。
- **优点**：18-task adaptation 分数非常高。
- **注意**：adaptation 与 transfer 是两个不同设置，不能混看。

### KnowMap

- **核心思想**：从环境和经验数据中构建知识库，并微调小 embedding model，让大 LLM 获取任务相关知识。
- **优点**：报告 76.25，优于 ReasonPlanner / DAVIS 的 65.06。
- **注意**：需要进一步确认它在每个 task type 上的详细覆盖和 variant 采样。

### EMPO²

- **核心思想**：将 memory 引入 on-policy / off-policy RL，提升探索和泛化。
- **优点**：18-task entries 上平均 75.9。
- **注意**：论文强调训练时 memory-augmented，但表中报告的是 trained model without memory at test time。

### AutoRefine

- **核心思想**：从执行轨迹自动抽取两类 Experience Patterns：procedural subagents 与 static skill patterns。
- **优点**：ScienceWorld 分数 70.4，步数显著减少。
- **注意**：只评了 Boil 和 Temperature Measurement 两类任务，覆盖很窄。

### STEP

- **核心思想**：把语言 agent 拆成 Planner、Executor、Evaluator、Memory；Memory 保存 previous experiences，Evaluator 用 learned rules 检查行动是否符合经验。
- **优点**：ScienceWorld 18 tasks 上报告 67.4，完成 12 / 18 tasks。
- **注意**：属于 planning + experience memory 框架，任务覆盖与 SSO / CLIN 更接近，不能直接与全 30 tasks 方法比较。

### DAVIS / ReasonPlanner

- **核心思想**：构建 temporal knowledge graph 作为 world model，通过 inner monologue 多轮检索支持规划。
- **优点**：覆盖全 30 task types，任务表和 subject 表完整。
- **注意**：测试每任务 3 variants，总 90 variations；RAG agents 使用 golden trajectories 构建 memory。

### GPT-J history model

- **核心思想**：单个 GPT-J 模型覆盖 30 classes，并将尽可能多的历史 actions / observations 放入 context。
- **优点**：证明长历史上下文显著优于 Markov assumption。
- **注意**：更像 sequence modeling / offline imitation，不是显式 memory 架构。

### CLIN

- **核心思想**：trial 后更新 persistent textual memory，存储 causal abstractions。
- **优点**：比 Reflexion 更强调因果抽象，而不是简单错误反思。
- **注意**：主要报告 18-task adaptation / generalization 设置。

### Retrospex

- **核心思想**：不直接把经验塞入上下文，而是用经验训练 RL critic；推理时结合 LLM likelihood 和 critic value 做 action rescoring。
- **优点**：对中长任务明显提升。
- **注意**：主文按 short / medium / long 聚合，完整 30 sub-tasks 需看 appendix。

### MCMA

- **核心思想**：学习如何抽象和管理 memory，由 memory copilot 决定 memory 的结构化、抽象层级和复用方式。
- **优点**：关注 transferable memory abstraction。
- **注意**：ScienceWorld 使用 Dev/Test split，但当前检索到的片段未列具体 task 清单。

### 5.1 其他近期相关工作

- **Embodied Planner-R1**：通过 outcome-driven RL 和 Interactive Policy Optimization 训练 embodied task planner；摘要报告 ScienceWorld completion rate 79.92%，但指标是 completion rate，不宜直接混入 0–100 reward 排名。
- **RLVMR**：用 verifiable meta-reasoning rewards 奖励 planning / exploration / reflection 等过程行为；摘要报告 7B model 在最难 unseen split 达 83.6% success rate，具体 ScienceWorld split 和任务覆盖需核表。
- **BPO**：提出 bootstrapping / extrapolation / refinement 的 data curation flywheel，用 reward-gated rejection sampling 选择经验训练 long-horizon planner；摘要称在 ScienceWorld 达 SOTA，但具体分数需查表。
- **TDP**：Task-Decoupled Planning 将任务分解为 sub-goal DAG，用 scoped contexts 降低长历史干扰；更偏 planning/context isolation，不是显式 memory benchmark。
- **AEC**：Active Epistemic Control 区分 grounded fact store 与 belief store，用主动查询和 feasibility check 降低不确定 belief 对规划的污染；具体 ScienceWorld 数值需进一步核查。
- **RPMS**：rule retrieval + belief-state memory gating + rules-first arbitration；摘要报告 ScienceWorld average score 54.0，相比 ReAct baseline 44.9 有提升。

## 6. 推荐引用口径

如果要在论文或报告中引用，建议写成：

> Existing ScienceWorld agent-memory methods are difficult to compare directly because they evaluate on different task subsets and adaptation protocols. Among methods evaluated on or near the full 30 task types, ICRL, SwiftSage, GraSP, DAVIS, and GPT-J history-based models report strong results. On the commonly used 18-task subset, SSO, EMPO², and CLIN are representative memory/experience-based methods. AutoRefine reports high performance but only on two ScienceWorld task categories, so its score should not be compared directly with full-benchmark methods.

中文版本：

> 现有 ScienceWorld 上的 agent memory 方法不能直接按分数做官方排名，因为各论文使用的 task subset、variant 数、模型 backbone 和 adaptation 协议差异很大。若只看接近全 30 task types 的方法，ICRL、SwiftSage、GraSP、DAVIS 和 GPT-J history-based model 是主要代表；若看常见的 18-task 子集，SSO、EMPO² 和 CLIN 更可比；AutoRefine 虽然分数高，但只评了两个任务类别，不能直接与全 benchmark 方法比较。

## 7. 实验模型 / Backbone 汇总

模型差异也是这些结果不能直接混排的重要原因：有些方法使用 GPT-4 / Claude / DeepSeek 等闭源 frontier model，有些使用较小的开源模型或 imitation/RL agent，还有些把 LLM 与 KG、embedding model、offline RL critic 组合使用。

| 方法 | ScienceWorld 实验使用的模型 / Backbone | 备注 |
| --- | --- | --- |
| **ICRL Preset** | **GPT-4.1-mini** | ScienceWorld policy model；Game of 24 等其他实验可能使用 GPT-4.1，但 ScienceWorld 对比主要是 GPT-4.1-mini。 |
| **SwiftSage** | Swift：T5 / Flan-T5-large 风格的 encoder-decoder imitation model；Sage：**GPT-4** | Swift 负责 routine action，Sage 负责 subgoal planning / replanning；不是单一 LLM memory 方法。 |
| **GraSP** | **DeepSeek V3.2** 主结果；另测 GPT-4.1、Claude-4-Sonnet、GLM-5、Gemini 2.5 Pro、o4 Mini、Qwen3-235B、Kimi-K2.5 | 多 backbone 对比；ScienceWorld 表中报告 seen / unseen。 |
| **SSO** | **GPT-4-based LLM actor / baselines** | ScienceWorld 中与 ReAct、Reflexion、CLIN 等 GPT-4 系方法比较；具体 API variant 需按论文实现细节进一步核查。 |
| **KnowMap** | **gpt-4-turbo** 主设置；另测 **gpt-4o-mini**、**DeepSeek-V3-241226** | 论文称对比框架用 gpt-4-turbo 保持公平；KnowMap 额外训练小 embedding model 做知识检索。 |
| **EMPO²** | **Qwen2.5-7B-Instruct** | ScienceWorld / WebShop 主实验统一使用该 backbone；论文中 Retrospex baseline 也被统一到同一 backbone。 |
| **AutoRefine** | **Claude-sonnet-4**，temperature 0.7 | 除非特别说明，所有方法默认用 Claude-sonnet-4；ALFWorld 子任务对比使用 GPT-4-turbo。 |
| **DAVIS / ReasonPlanner** | Reasoning：**GPT-4-Turbo**；QA：**GPT-4o**；KG pipeline：**LLaMA3-70B-Instruct** | Temporal KG construction 与 actor-critic planning 使用不同模型组件。 |
| **GPT-J history model** | **GPT-J 6B** | 单个 GPT-J model 覆盖全部 30 classes，并尽量利用长历史上下文。 |
| **CLIN** | **gpt-4** | Controller、executor、memory generator 均基于 gpt-4；强调 persistent causal textual memory。 |
| **Retrospex** | ScienceWorld warm-up / IL agent：**Flan-T5-large / IL-T5**；critic：**GRU-based RL critic**，约 2.7M 参数 | ScienceWorld 设置不同于其 ALFWorld / WebShop 的 LLaMA3-8B-Instruct + LoRA 设置。 |
| **MCMA** | Task model：**Qwen3-8B / Qwen3-32B**；memory copilot：**Qwen3-4B** | 主表分数对应 Qwen3-32B；task model frozen，memory learning 放在 copilot。 |
| **Knowledge-enhanced Agents** | **DRRN、KG-A2C、RoBERTa、Swift** | 更像在 RL / LM agents 上注入 previous correct actions memory 与 object affordances。 |
| **STEP** | 需进一步核查 | 框架包含 Planner / Executor / Evaluator / Memory；当前已确认 ScienceWorld 67.4，但具体 backbone 需查表。 |
| **SkillGen** | **Qwen2.5-7B-Instruct、Qwen-Turbo、GPT-4o-mini** | 报 GR / PR / SR / AUPC，不是统一 ScienceWorld 0–100 score。 |
| **LGE** | 需进一步核查 | 当前只确认其 ScienceWorld exploration 设置，未整理出可引用的具体 backbone。 |
| **Embodied Planner-R1** | 需进一步核查 | 摘要报告 ScienceWorld completion rate 79.92%；具体 base model / RL setup 需查表。 |
| **RLVMR** | 7B model | 摘要称 7B model 在最难 unseen split 达 83.6% success rate；具体模型名与 ScienceWorld split 需核查。 |
| **BPO** | 需进一步核查 | Data curation flywheel / reasoning model 训练框架；具体 backbone 需查表。 |
| **TDP** | 需进一步核查 | Training-free planning framework；具体 LLM backbone 需查表。 |
| **AEC** | 需进一步核查 | Epistemic-categorical planning layer；具体 world model / agent backbone 需查表。 |
| **Neoplanner** | 需进一步核查 | 当前只确认 LLM + state-space search + textual entity-relation learnings，模型细节仍需查表。 |
| **SkillNet** | 需进一步核查 | 当前只确认使用 LLM evaluator / 多 benchmark；ScienceWorld agent backbone 需进一步核查。 |
| **RPMS** | **GPT-4** for ScienceWorld adaptation | 摘要报告 ScienceWorld avg. score 54.0 vs ReAct 44.9；ALFWorld 主结果另用 Llama 3.1 8B / Claude Sonnet 4.5。 |

## 8. 待核查 / 不确定内容清单

下面这些条目目前主要来自摘要、主表片段或横向引用，建议在正式论文 / 报告中引用前回到原文表格、appendix 或代码仓库核查。

| 优先级 | 方法 | 需要核查的内容 | 当前文档中的临时处理 |
| --- | --- | --- | --- |
| 高 | **KnowMap** | 是否真的覆盖完整 30 task types；76.25 的 variant 采样、train/test split、是否与 DAVIS / SwiftSage 同协议。 | 暂写“论文描述 ScienceWorld 有 30 task types、每类 100+ variants”，并注明需确认 task type 明细。 |
| 高 | **MCMA** | ScienceWorld 标准 Dev/Test split 的具体 task list、variant 数、是否覆盖全部 30 task types；Qwen3-32B 分数对应哪张表。 | 暂列为“标准 split 但任务明细不足”。 |
| 高 | **STEP** | 18 tasks 的具体任务清单、variant 采样、67.4 是平均 score 还是 reward；实验 backbone / prompt model。 | 暂按摘要列 18 tasks、67.4、12/18 tasks；backbone 标为需进一步核查。 |
| 高 | **Embodied Planner-R1** | 79.92% 是 ScienceWorld completion rate、success rate 还是另一种 completion metric；任务覆盖和 unseen split 定义；base model / RL 训练设置。 | 放在“指标不同”表，不混入 0–100 score 排名。 |
| 高 | **RLVMR** | 83.6% 是否是 ScienceWorld 单项、ALFWorld/ScienceWorld 综合，还是最难 unseen split；具体 7B backbone 名称、任务覆盖。 | 标为摘要级结果，要求核查 split 和任务覆盖。 |
| 高 | **BPO** | ScienceWorld 具体分数、任务覆盖、backbone、是否真是 SOTA 以及比较对象。 | 仅列为摘要称 SOTA，不给具体排名。 |
| 高 | **TDP** | ScienceWorld 表中的具体 score、任务覆盖、使用的 LLM backbone、是否包含 memory/experience 或只是 context isolation。 | 放在“指标不同”表，注明更偏 planning/context isolation。 |
| 高 | **AEC** | ScienceWorld 具体分数、replanning 指标、backbone / world model、任务覆盖。 | 仅列摘要描述，不参与排序。 |
| 中 | **SSO** | ScienceWorld 实验具体 GPT-4 API variant；是否所有 actor / baselines 都统一使用同一 GPT-4 设置。 | 暂写 GPT-4-based，并提示具体 API variant 需核查。 |
| 中 | **AutoRefine** | ScienceWorld 是否完全使用 Claude-sonnet-4；GPT-4-turbo 是否只用于 ALFWorld 子任务对比；ScienceWorld 两类任务的 exact variant split。 | 暂写默认 Claude-sonnet-4，ALFWorld 对比用 GPT-4-turbo。 |
| 中 | **Retrospex** | 55.98 对应的是完整 30 sub-tasks 平均、按复杂度聚合后的平均，还是 appendix 表格中的结果；ScienceWorld IL-T5 / Flan-T5-large 设置细节。 | 暂写主文按 Short/Medium/Long 聚合，完整 30 sub-tasks 需看 appendix。 |
| 中 | **RPMS** | ScienceWorld adaptation 的具体任务覆盖、score 54.0 的 metric 定义、GPT-4 具体版本。 | 暂写 adapted to ScienceWorld，avg. score 54.0 vs ReAct 44.9。 |
| 中 | **SkillGen** | ScienceWorld task 数、variant split、GR/PR/SR/AUPC 与 reward score 的关系。 | 放入“指标不同”表，不混入 0–100 score 排名。 |
| 中 | **SkillNet** | ScienceWorld 单项分数、任务覆盖、agent backbone，而不是 evaluator backbone。 | 暂列为需进一步核查。 |
| 中 | **Neoplanner** | 124% improvement 的 baseline、具体平均 reward、任务数量、LLM backbone。 | 暂列摘要级描述，不参与排序。 |
| 中 | **LGE** | 具体 score、任务覆盖、agent/RL backbone、是否属于 memory/experience 策略。 | 暂列为 RL exploration 相关，不作为 memory 排名方法。 |
| 低 | **GraSP** | DeepSeek V3.2 是否为最终主表默认 backbone；seen/unseen split 的 variant 具体定义。 | 当前已列多 backbone，但正式引用时可核主表与 appendix。 |
| 低 | **DAVIS / ReasonPlanner** | 65.06 是否应标为 DAVIS 还是 ReasonPlanner/RAG agent 的对应名字；subject-level 与 task-type-level平均方式。 | 当前合并写作 DAVIS / ReasonPlanner。 |
| 低 | **Knowledge-enhanced Agents** | 10 tasks 的具体 task names、不同模型表格中的 reward 如何汇总。 | 暂不纳入统一排名。 |

## 9. Sources

- [ICRL / Reward Is Enough](https://arxiv.org/abs/2506.06303)
- [SwiftSage](https://arxiv.org/abs/2305.17390)
- [GraSP](https://arxiv.org/abs/2604.17870)
- [Skill Set Optimization / SSO](https://arxiv.org/abs/2402.03244)
- [KnowMap](https://arxiv.org/abs/2506.19527)
- [EMPO²](https://arxiv.org/abs/2602.23008)
- [AutoRefine](https://arxiv.org/abs/2601.22758)
- [DAVIS / ReasonPlanner](https://arxiv.org/abs/2410.09252)
- [Remember what you did so you know what to do next](https://arxiv.org/abs/2311.01468)
- [CLIN](https://arxiv.org/abs/2310.10134)
- [Retrospex](https://arxiv.org/abs/2505.11807)
- [MCMA](https://arxiv.org/abs/2601.07470)
- [Knowledge-enhanced Agents for Interactive Text Games](https://arxiv.org/abs/2305.05091)
- [SkillGen](https://arxiv.org/abs/2511.14670)
- [Embodied Planner-R1](https://arxiv.org/abs/2506.23127)
- [RLVMR](https://arxiv.org/abs/2507.22844)
- [BPO / Data Curation Flywheel](https://arxiv.org/abs/2508.03018)
- [TDP / Task-Decoupled Planning](https://arxiv.org/abs/2601.07577)
- [AEC / Active Epistemic Control](https://arxiv.org/abs/2602.03974)
- [STEP](https://arxiv.org/abs/2411.08432)
- [Neoplanner](https://arxiv.org/abs/2312.07368)
- [Language Guided Exploration / LGE](https://arxiv.org/abs/2403.03141)
- [SkillNet](https://arxiv.org/abs/2603.04448)
- [RPMS](https://arxiv.org/abs/2603.17831)

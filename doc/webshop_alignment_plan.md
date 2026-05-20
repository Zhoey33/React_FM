<!--
本文档记录本项目 WebShop 数据集与评估协议相对原论文的未对齐问题，
以及后续 WebShop 实验需要采用的论文对齐方案。
-->

# WebShop 数据集与评估协议对齐方案

## 背景

本文档总结当前仓库的 WebShop 设置与原 WebShop 论文协议之间的差异，并记录已经确定的对齐方案。

原论文目标协议：

- Benchmark：WebShop
- 数据规模：完整 WebShop，约 1.18M 商品和 12,087 条众包购物指令
- 数据划分：10,587 train / 1,000 development / 500 test instructions
- 评估集：500 条 test instructions
- 指标：
  - Task Score = `100 * average_reward`
  - Success Rate = `reward == 1.0` 的 episode 占比

## 当前未对齐点

### 1. 数据规模

当前仓库行为：

- 本地 WebShop 安装只有 1000-product preview 文件：
  - `/Users/zhoey/WebShop/data/items_shuffle_1000.json`
  - `/Users/zhoey/WebShop/data/items_ins_v2_1000.json`
  - `/Users/zhoey/WebShop/search_engine/indexes_1k`
- 当前 bridge 默认使用 `num_products=1000`。

为什么不对齐：

- 原论文评估使用完整 WebShop 商品库和搜索空间，而不是 1000-product preview。
- 更小的商品库会改变搜索难度、候选商品分布和任务难度。

已确定方案：

- 使用完整 WebShop 数据和完整搜索索引。
- 论文对齐结果不再使用 1000-product preview。

### 2. Instruction 来源

当前仓库行为：

- `experiments/webshop_bridge.py` 创建 `WebAgentTextEnv` 时没有传入 `human_goals=1`。
- 在 WebShop 中，这会默认走 synthetic goals。
- 本地检查结果：
  - `human_goals=0`, `num_products=1000`：6910 条 synthetic goals
  - `human_goals=1`, `num_products=1000`：本地 preview 数据中只有 13 条 human goals

为什么不对齐：

- 原论文评估的是 crowd-sourced / human instructions。
- Synthetic goals 更模板化，任务分布与人类众包指令不同。

已确定方案：

- 使用 `human_goals=1`。
- 只评估 crowd-sourced / human instructions。

### 3. Train / Development / Test Split

当前仓库行为：

- 当前 WebShop runner 直接对前 `N` 个 goals 调用 `reset(session_idx)`。
- 没有显式使用官方 test split wrapper。

为什么不对齐：

- 原论文使用固定 i.i.d. 划分：
  - train：10,587
  - development：1,000
  - test：500
- 论文结果在 500 条 test instructions 上报告。

已确定方案：

- 使用官方 test split。
- 当前真实实验从官方 500 条 test instructions 中固定抽取 100 条运行。
- 完整论文对齐结果后续再扩展到全部 500 条 test instructions。

### 4. 评估 Episode 数量

当前仓库行为：

- `experiments/run_webshop.py` 默认评估 200 条。
- `experiments/run_reflexion_webshop.py` 默认评估 100 条。
- `experiments/run_expel_webshop.py` 默认评估 100 条。
- `experiments/gate/run_online_eval.py` 的 WebShop 默认评估 100 条。

为什么不对齐：

- 原论文对所有 500 条 test instructions 取平均并报告结果。

已确定方案：

- 完整论文对齐评估仍以 500 条 test instructions 为准。
- 当前真实实验先采用快速评估设置：从官方 500 条 test instructions 中固定抽取 100 条运行。
- 100 条 test 子集需要固定随机种子并保存采样列表，保证可复现。

### 5. Success Rate 定义

当前仓库行为：

- 大多数 WebShop 脚本把 `reward >= 0.5` 算作 success。
- `experiments/gate/run_online_eval.py` 使用 `WebShopFailureDetector.is_task_complete`，当前返回 `done, done`，因此只要完成购买就可能被算作 success，不管 reward 是否满分。

为什么不对齐：

- 原论文定义 Success Rate 为 `reward == 1.0` 的指令比例。

已确定方案：

- WebShop 论文对齐评估使用 `success = (reward == 1.0)`。
- 不使用 `reward >= 0.5`。
- 不使用 `done=True` 作为 success。

### 6. Task Score 定义

当前仓库行为：

- WebShop summary 报告的是 0-1 范围内的 `avg_reward`。

为什么不对齐：

- 原论文报告的 Task Score 是 `100 * average_reward`。

已确定方案：

- 同时报告两个字段：
  - `avg_reward`：0-1 原始平均 reward，便于内部调试
  - `task_score`：`100 * avg_reward`，用于论文结果对比
- 与原 WebShop 论文数字比较时使用 `task_score`。

### 7. Step Limit

当前仓库行为：

- WebShop 脚本默认 `max_steps=15`。

为什么不对齐：

- 官方 WebShop baseline 使用 100-step episode limit。

已确定方案：

- WebShop 论文对齐评估使用 `max_steps=100`。
- 更短 step limit 只用于快速 smoke test。

### 8. 环境 Wrapper 与 Agent Interface

当前仓库行为：

- 项目直接桥接到 `WebAgentTextEnv`。
- 使用 `observation_mode="text"`。
- Agent 自由生成 `search[...]` 和 `click[...]` 动作。
- Runner 没有完整复用官方 `baseline_models/env.py::WebEnv` wrapper。

官方 wrapper 提供什么：

- 官方 split 处理，包括 `test = range(500)`。
- 默认 `state_format="text_rich"`。
- 通过 `info["valid"]` 构造当前合法动作集合。
- 商品点击动作规范化，例如 `click[item - product name]`。
- 与官方 baseline 代码一致的 step limit 和 score bookkeeping。

为什么这很重要：

- 即使用同一个底层环境，不同的 observation / action interface 也会改变任务难度。
- Free-form LLM 动作生成和官方 valid-action candidate 选择不是同一种评估接口。

已确定方案：

- 尽可能复用官方 WebShop wrapper，尤其是 `baseline_models/env.py::WebEnv`。
- 使用官方 valid actions，以及官方 split / score / step-limit 处理逻辑。
- 如果未来实验刻意保留 free-form LLM interface，需要单独标注，不能称为完全 paper-aligned。

## 最终对齐协议

本仓库后续 WebShop 评估应采用：

1. 使用完整 WebShop 商品数据和完整搜索索引。
2. 使用 `human_goals=1`。
3. 使用官方 test split。
4. 当前真实实验从官方 500 条 test instructions 中固定抽取 100 条运行；完整论文对齐结果再扩展到全部 500 条。
5. 使用 `success = (reward == 1.0)`。
6. 报告 `task_score = 100 * avg_reward`。
7. 使用 `max_steps=100`。
8. 复用官方 WebShop wrapper 和 valid-action interface。

## Memory 复用粒度与实验顺序

### 第一阶段：Online Memory

先实现并运行 online memory 设置：

- 从官方 500 条 test instructions 中固定抽取 100 条，并按固定顺序运行。
- Test 过程中允许从前序 test episodes 中提取并写入 memory。
- 后续 test episodes 可以复用前面 episode 产生的 memory。
- WebShop 内共享一套 `shopping` memory。
- 不按 `env_idx` 隔离 memory。
- 该设置用于评估 React_FM 的在线经验积累能力。
- 结果需要标注为 online / transductive memory setting，不直接作为与原论文 static baseline 的公平对比。

### 第二阶段：Frozen Test-Time Memory

Online memory 完成后，再实现 frozen test-time memory 设置：

- Test 期间不写入新 memory。
- 第一版 offline memory 从官方 train split 中固定抽取 100 条 instructions 生成。
- Memory 也可以来自后续扩展的 train/dev 预先固定 memory 文件。
- Frozen 设置下 memory 仍然是 WebShop 内共享的一套 `shopping` memory，不按 `env_idx` 隔离。
- Offline memory 文件必须使用共享粒度保存，例如 `scope="task_type"` 且 `task_type="shopping"`。
- Test 期间只读 memory，不更新 memory。
- 该设置避免 test leakage，更适合与原论文 baseline 对比。

## 实践备注

- 1000-product 数据只用于本地 smoke test。
- Synthetic goals 只用于调试环境机制。
- 当前 100 条 test 子集用于快速真实实验，不应直接等同于原论文 500-test 完整结果。
- 使用 `reward >= 0.5` 的运行只能视为 relaxed internal metric，不是 WebShop 原论文 SR。

## 当前环境阻塞记录

- 2026-05-20 本地 `/Users/zhoey/WebShop` 仍缺少 full WebShop 数据与完整搜索索引：
  - `/Users/zhoey/WebShop/data/items_shuffle.json`
  - `/Users/zhoey/WebShop/data/items_ins_v2.json`
  - `/Users/zhoey/WebShop/search_engine/indexes`
- 当前网络环境下载不稳定：
  - Google Drive 官方 `gdown` 下载 `items_shuffle.json` 时在约 398 MB 处断线。
  - Hugging Face 镜像在当前网络下连接超时或卡在大文件握手阶段。
- 等网络较快时，优先直接使用 WebShop 官方 Google Drive 链接下载 full 数据：
  - `cd /Users/zhoey/WebShop/data`
  - `/opt/miniconda3/envs/webshop/bin/python -m gdown --continue 'https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB' -O items_shuffle.json`
  - `/opt/miniconda3/envs/webshop/bin/python -m gdown --continue 'https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi' -O items_ins_v2.json`
  - `cd /Users/zhoey/WebShop/search_engine && /opt/miniconda3/envs/webshop/bin/python convert_product_file_format.py && ./run_indexing.sh`
- full 数据和 `indexes` 建好前，React-FM 的 WebShop paper-aligned 100-test 结果不要启动；只能使用 1000-product preview 做环境 smoke test。

## TODO List

### 实现与环境 TODO

- [ ] 安装并验证 full WebShop 数据文件与完整搜索索引。  
  状态：未完成；当前受网络环境阻塞，缺少 `items_shuffle.json`、`items_ins_v2.json` 和完整 `search_engine/indexes`。
- [x] 修改 WebShop bridge / runner，使其使用 `human_goals=1`。  
  状态：已完成；`experiments/webshop_bridge.py` 和 `src/webshop_env.py` 已支持并默认使用 human goals。
- [x] 复用官方 `baseline_models/env.py::WebEnv` wrapper。  
  状态：已完成；WebShop bridge 默认使用 `--wrapper official`，并保留 `--webshop-wrapper direct` 作为 legacy/smoke fallback。
- [x] 实现官方 test split 中固定抽取 100 条的采样逻辑，并保存采样 ID 列表。  
  状态：已完成；`experiments/run_webshop.py` 支持 `--eval-split test --eval-sample-size 100 --eval-sample-seed 42`，并保存 `*_sample_ids.json`。
- [ ] 实现官方 train split 中固定抽取 100 条的 offline memory 收集逻辑，并保存采样 ID 列表。  
  状态：部分完成；runner 已支持 train split offset 采样，但还没有单独实现“从 train split 生成 frozen offline memory”的收集流程。生成的 offline memory 必须按共享 `shopping` memory 保存，不按 `env_idx` 隔离。
- [x] 将 WebShop success 统一改为 `reward == 1.0`。  
  状态：已完成；summary、episode result、resume skip 逻辑均使用 exact success。
- [x] 在 summary 中同时保存 `avg_reward` 和 `task_score = 100 * avg_reward`。  
  状态：已完成；summary 已新增 `task_score`。
- [x] 将 WebShop 主实验 step limit 统一为 100。  
  状态：已完成；runner 默认 `--max-steps 100`，bridge official wrapper 使用同一 `step_limit`。
- [x] 将 online memory 设置为共享 `shopping` memory，不按 `env_idx` 隔离。  
  状态：已完成；online memory 使用 `FailureMemoryStore(scope="task_type")`，WebShop task type 固定为 `shopping`。
- [ ] 实现 frozen test-time memory：test 期间只读 memory，不写入新 memory。  
  状态：部分完成；runner 已支持 `--memory-setting frozen` 并禁止 test-time 写入，但 frozen 所需 offline memory 生成流程尚未完成。Frozen test-time memory 应加载共享 `shopping` memory。
- [x] 在结果文件中明确记录 memory setting：`online` 或 `frozen`。  
  状态：已完成；summary 已记录 `memory_setting`。

### 实验运行 TODO

- [x] WebShop official wrapper smoke test：1000-product preview，`human_goals=1`，`max_steps=3`，验证 reset/valid actions/step reward。  
  状态：已完成过一次本地 smoke；输出显示 official wrapper 能返回 human instruction、valid actions，reward 已按 0-1 归一。若后续改 bridge，需重跑。
- [ ] Paper-aligned WebShop 环境 smoke test：full products + full index，`human_goals=1`，official test split，运行 1 条 episode。  
  状态：未完成；依赖 full 数据和完整索引。
- [ ] React-FM online memory 主实验：official test split 固定抽取 100 条，`memory_setting=online`，`max_steps=100`，使用共享 `shopping` memory。  
  状态：未完成；依赖 full 数据和完整索引。建议命令：
  `python experiments/run_webshop.py --config config.yaml --max-envs 100 --eval-split test --eval-sample-size 100 --eval-sample-seed 42 --memory-setting online --num-products full --max-steps 100 --run-name ws_react_fm_online_aligned_`
- [ ] 保存并检查 online 100-test 采样列表。  
  状态：未完成；主实验完成后检查 `*_sample_ids.json` 是否为 100 个唯一 test IDs，且 summary 记录 `eval_sample_seed=42`。
- [ ] 分析 React-FM online 100-test 结果。  
  状态：未完成；需要检查 `avg_reward`、`task_score`、`success_rate`、`memory_stats`、失败 episode 分布，并明确标注 online/transductive setting。
- [ ] 生成 frozen offline memory：从 official train split 固定抽取 100 条，运行 memory 收集流程并保存 memory 文件与 train sample IDs。  
  状态：未完成；需要先实现/确认 offline memory 收集脚本或 runner 模式。offline memory 必须是一套共享 `shopping` memory，例如保存为 `scope="task_type"` / `task_type="shopping"`，不能按 train `env_idx` 隔离。
- [ ] React-FM frozen test-time memory 实验：official test split 固定抽取 100 条，加载 offline memory，`memory_setting=frozen`，test 期间不写 memory。  
  状态：未完成；依赖 offline memory 文件。建议命令形态：
  `python experiments/run_webshop.py --config config.yaml --max-envs 100 --eval-split test --eval-sample-size 100 --eval-sample-seed 42 --memory-setting frozen --resume-memory <offline_memory.json> --num-products full --max-steps 100 --run-name ws_react_fm_frozen_aligned_`
- [ ] 对比 online vs frozen 100-test 结果。  
  状态：未完成；比较 `task_score`、exact `success_rate`、token 开销和 memory 检索命中。
- [ ] 扩展完整 paper-aligned 500-test React-FM 评估。  
  状态：未完成；100-test 结果稳定后运行 official test split 全 500 条，并保存完整结果。
- [ ] 可选 baseline 对照：运行 vanilla ReAct official 100-test。  
  状态：未完成；用于判断 React-FM 增益，建议与 React-FM 使用同一 `*_sample_ids.json`。

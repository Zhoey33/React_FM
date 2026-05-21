<!-- This document records the official StableToolBench alignment requirements for the local ToolBench implementation and evaluation. -->

# StableToolBench Official Alignment

## 目标

本项目的 ToolBench / StableToolBench 实验统一改为走官方 StableToolBench 协议。当前自定义的“调用 `Finish` 且 final answer 非空即成功”的统计只能作为调试指标，不能作为论文或主实验的 ToolBench 结果。

官方对齐后的主结果应报告 StableToolEval 指标，至少包括 Solvable Pass Rate (SoPR)。如使用 MirrorAPI Final-Answer-Correctness 评估，则单独报告 FAC，并明确和 SoPR 的区别。

官方参考：

- StableToolBench 官方仓库：<https://github.com/THUNLP-MT/StableToolBench>
- 本地快照说明：[data/StableToolBench/README.md](/Users/zhoey/React_FM/data/StableToolBench/README.md)
- 官方 pass-rate evaluator：[data/StableToolBench/toolbench/tooleval/eval_pass_rate.py](/Users/zhoey/React_FM/data/StableToolBench/toolbench/tooleval/eval_pass_rate.py)

## 对齐要求

### 1. 成功判定走官方

主指标不得再使用本项目内部的非空 `Finish` 判定。

当前内部判定：

- [src/toolbench_env.py](/Users/zhoey/React_FM/src/toolbench_env.py) 中 `Finish` 返回非空答案即给 reward / success。
- [experiments/run_toolbench.py](/Users/zhoey/React_FM/experiments/run_toolbench.py) 中 episode success 直接继承该判定。

官方判定：

- 先检查最终轨迹是否包含 `Finish`。
- 再由 StableToolEval evaluator 调用 `check_is_solved(...)` 判断 answer 是否解决 query。
- `AnswerStatus.Solved` 计 1，`AnswerStatus.Unsure` 计 0.5，其他计 0。

验收标准：

- 论文和正式结果中的 ToolBench `Pass` 必须来自 `eval_pass_rate.py` 或 FAC evaluator。
- 内部非空 `Finish` 只能命名为 `finish_rate` 或 `submission_rate`。

### 2. 评估流程走官方

正式评估必须按官方三段式执行：

1. 使用官方格式保存每个 query 的原始预测。
2. 运行 `toolbench/tooleval/convert_to_answer_format.py` 转成 ToolEval answer format。
3. 运行 `toolbench/tooleval/eval_pass_rate.py` 计算 SoPR。

推荐目录约定：

```text
data/StableToolBench/data/answer/<model_name>/<subset>/<query_id>_<method>.json
data/StableToolBench/data/model_predictions_converted/<model_name>/<subset>.json
data/StableToolBench/data/pass_rate_results/<model_name>/<subset>_<model_name>.json
```

验收标准：

- 每个正式 run 都能生成 `converted_answer_path/<model_name>/<subset>.json`。
- 每个 subset 都能由官方 `eval_pass_rate.py` 产出 JSON / CSV。
- 汇总脚本只读取官方 evaluator 输出，不读取内部 `success_rate`。

### 3. 指标命名走官方

正式结果表中使用官方指标名：

- `SoPR`: Solvable Pass Rate，主 pass-rate 指标。
- `SoWR`: Solvable Win Rate，仅在候选方法和 reference method 做偏好比较时报告。
- `FAC`: MirrorAPI Final-Answer-Correctness，仅在使用 FAC evaluator 时报告。

需要避免的命名：

- 不把非空 `Finish` 率写成 `Pass`。
- 不把内部 `success_rate` 写成 StableToolBench pass rate。

验收标准：

- 论文和文档中的 ToolBench 表头改为 `SoPR`，或明确标注 `Finish rate (debug only)`。
- 对历史结果保留时，必须加注“not official StableToolEval”。

### 4. 环境交互走官方

正式 run 应通过 StableToolBench 官方 inference / virtual server 路径执行 API 调用，而不是直接在本项目 wrapper 中手写 cache lookup 作为最终协议。

官方路径：

- 启动 StableToolBench server。
- 使用 `SERVICE_URL=http://localhost:8080/virtual`。
- 通过 `toolbench/inference/qa_pipeline_multithread.py` 或兼容该协议的 adapter 调用工具。

对齐要求：

- cache 命中、cache miss、real API fallback、GPT simulator / MirrorAPI fallback 的行为交给官方 server。
- 本项目如需保留 `ToolBenchEnv`，只能作为 adapter；adapter 必须调用官方 server endpoint，而不是复制 server 逻辑。
- 交互步数对齐官方 `qa_pipeline_multithread.py` 默认设置：`--single_chain_max_step 50`。

验收标准：

- 一次 smoke run 的 API request 能在官方 server log 中看到。
- cache miss 时的返回行为与官方 server 一致。
- 结果目录结构能被官方 convert / eval 脚本直接消费。

### 5. 工具 schema 走官方

工具定义、函数名、参数名和类型映射必须复用官方逻辑。

官方关键规则：

- `standardize(...)` / `change_name(...)` 处理 API 名和参数名。
- 函数名形如 `<api_name>_for_<standard_tool_name>`。
- 函数名按官方逻辑截断到 64 字符。
- 参数类型按官方 `NUMBER -> integer`、`STRING -> string`、`BOOLEAN -> boolean` 映射。
- required / optional 字段和 example/default value 采用官方 schema。

验收标准：

- 本项目不再维护一套不同的 schema builder 作为正式协议。
- 任一 query 的 `available_tools` 和官方 inference 生成结果保持一致，允许仅有方法注入 prompt 的差异。

### 6. 输出轨迹格式走官方

正式输出必须兼容 ToolEval execution graph / answer format。

官方 convert 脚本需要的信息包括：

- `answer_generation.valid_data`
- `answer_generation.train_messages`
- `answer_generation.function`
- `answer_generation.query`
- `answer_generation.final_answer`
- invalid data fallback 所需的 `trys` 或 DFS tree 信息

验收标准：

- 本项目每个 episode 的输出能被 `convert_to_answer_format.py` 直接转换。
- 转换后包含 `query`、`available_tools`、`answer.method`、`answer.final_answer`、`answer.answer_details`。
- 不再只保存本项目自定义的 `episodes` summary JSON 作为正式评估输入。

### 7. 多次评估和聚合走官方

正式 SoPR 评估使用官方 evaluator 的重复评估设置。

默认要求：

- `--evaluate_times 3`，与 StableToolBench README 示例一致。
- evaluator 使用官方推荐模型配置；若因环境限制替换 evaluator，必须在结果表和 run log 中说明。
- 每个 subset 单独评估，再按官方方式汇总平均值和标准差。

验收标准：

- 每个 query 有多次 `is_solved` 记录。
- 汇总结果包含 mean / std。
- 论文表格不使用单次内部判断替代官方多次评估。

## 推荐执行命令

以下命令应作为正式 ToolBench 评估的基础模板，路径可按 run name 调整。

```bash
cd data/StableToolBench

export TOOLBENCH_KEY=""
export OPENAI_KEY=""
export OPENAI_API_BASE=""
export PYTHONPATH=./
export GPT_MODEL="gpt-3.5-turbo-16k"
export SERVICE_URL="http://localhost:8080/virtual"
export MODEL_NAME="<model_name>"
export METHOD="CoT@1"

for group in G1_instruction G1_category G1_tool G2_instruction G2_category G3_instruction; do
  mkdir -p data/answer/${MODEL_NAME}/${group}
  python toolbench/inference/qa_pipeline_multithread.py \
    --tool_root_dir toolenv/tools \
    --backbone_model chatgpt_function \
    --openai_key ${OPENAI_KEY} \
    --max_observation_length 1024 \
    --single_chain_max_step 50 \
    --method ${METHOD} \
    --input_query_file solvable_queries/test_instruction/${group}.json \
    --output_answer_file data/answer/${MODEL_NAME}/${group} \
    --toolbench_key ${TOOLBENCH_KEY} \
    --num_thread 1
done
```

```bash
cd data/StableToolBench/toolbench/tooleval

export RAW_ANSWER_PATH=../../data/answer
export CONVERTED_ANSWER_PATH=../../data/model_predictions_converted
export MODEL_NAME="<model_name>"
export METHOD="CoT@1"

mkdir -p ${CONVERTED_ANSWER_PATH}/${MODEL_NAME}

for group in G1_instruction G1_category G1_tool G2_instruction G2_category G3_instruction; do
  python convert_to_answer_format.py \
    --answer_dir ${RAW_ANSWER_PATH}/${MODEL_NAME}/${group} \
    --method ${METHOD} \
    --output ${CONVERTED_ANSWER_PATH}/${MODEL_NAME}/${group}.json
done
```

```bash
cd data/StableToolBench/toolbench/tooleval

export API_POOL_FILE=../../openai_key.json
export CONVERTED_ANSWER_PATH=../../data/model_predictions_converted
export SAVE_PATH=../../data/pass_rate_results
export CANDIDATE_MODEL="<model_name>"

mkdir -p ${SAVE_PATH}/${CANDIDATE_MODEL}

python eval_pass_rate.py \
  --converted_answer_path ${CONVERTED_ANSWER_PATH} \
  --save_path ${SAVE_PATH}/${CANDIDATE_MODEL} \
  --reference_model ${CANDIDATE_MODEL} \
  --test_ids ../../solvable_queries/test_query_ids \
  --max_eval_threads 35 \
  --evaluate_times 3 \
  --test_set G1_instruction G1_category G1_tool G2_instruction G2_category G3_instruction
```

## 实施备注

- 如果继续保留 React_FM / Reflexion / ExpeL 的自定义 agent loop，需要新增 official-output adapter，把每一步写成官方 `train_messages` / function-call 轨迹。
- 如果直接复用官方 `qa_pipeline_multithread.py`，则需要把 React_FM 的 memory injection 接到官方 LLM / algorithm 层，避免破坏输出格式。
- 历史 ToolBench 数字可以留作 ablation 或 debug，但不能和官方 StableToolBench 结果同表直接比较。
- 环境问题优先修复；安装依赖遇到网络问题时，按项目约定使用清华源。

## React_FM Memory 适配建议

StableToolBench 的每个 query 都有自己的 `api_list`。多数 query 的工具/API 集合不同，但也存在一批共享同类工具的任务。因此最合适的 React_FM 记忆模式不是 per-env 完全隔离，也不是无约束全局共享，而是：

```text
全局共享 memory 池
+
tool/API-aware retrieval
+
failure-triggered in-loop injection
```

每条 ToolBench memory 建议至少记录：

```text
query_id
subset
category
tool_name
api_name
function_name
failure_type
error_signature
failure_action
failure_observation
repair_action
repair_strategy
```

检索时按层级降级：

```text
1. 同 category + tool_name + api_name
2. 同 category + tool_name
3. 同 toolset cluster
4. 全局抽象策略
```

不要只按 query 文本相似度检索。ToolBench 中两个 query 文本可能都像“帮我查信息”，但 API schema 完全不同。检索 query 应包含当前工具调用：

```text
task + current_action + category/tool/api + observation/error + failure_type
```

注入策略：

- 具体 API 参数修复：只在同 API / 同 tool 下使用。
- cache miss、重复调用、过早/过晚 Finish 等策略：可以全局共享。
- episode 开头少注入或不注入具体 memory，优先在失败发生后 in-loop 注入。

## 主实验设计

StableToolBench 实验放在论文主实验中，只做 online memory，用来测试 React_FM 在该数据集上的整体性能。主评估必须走官方 StableToolEval。

步数预算对齐官方默认配置：每个 query 的 single-chain 最大步数为 50，即 `--single_chain_max_step 50`。本项目旧脚本中的 10/12 步设置不能作为正式 ToolBench 主实验配置。

### A. Baseline

无 memory，走官方 StableToolBench inference / output / eval 流程。

目的：

- 得到官方 SoPR baseline。
- 记录每个 query 的完整轨迹，用于分析 online memory 的失败类型和可迁移性。

### B. Online Memory

从空 memory 开始，边运行边写入，后续步骤或后续任务遇到失败时检索已有 memory。

机制：

- 当前任务内或当前 benchmark 流中遇到失败后，写入共享 memory。
- 后续失败触发检索，按 tool/API-aware retrieval 取 memory。
- 注入方式以 in-loop 为主。

适合验证：

- React_FM 是否能在同一批任务的在线流中积累经验。
- 共享 memory 是否能帮助后续相同 tool/API cluster 的任务。

注意：

- 若只跑一个 epoch，online memory 主要帮助后面出现的相似工具任务。
- 若跑多 epoch，必须明确这是 repeated-query online adaptation，不要和标准 one-pass SoPR 混淆。

### 主实验设置

```text
1. Baseline official SoPR
2. Online memory, one-pass
```

核心对比只保留：

```text
Baseline ReAct
vs
Online React_FM
```

论文主表报告：

```text
official SoPR
token cost
subset-level SoPR
```

主结论只回答一个问题：

```text
从空 memory 开始，React_FM 能否通过在线共享记忆提升 StableToolBench 官方 SoPR？
```

不在主实验中展开 offline memory、two-epoch repeated-query adaptation 或 retrieval ablation。若后续需要，这些内容只作为附加分析或消融实验单独处理。

## TODO List

- [ ] 确认 StableToolBench 官方环境可运行：依赖安装、server 启动、`/virtual` endpoint 可访问。
- [ ] 确认 6 个 solvable test subsets 均可读取，共 765 条 query。
- [ ] 将 ToolBench agent 输出改成官方 raw answer 格式，保证可被 `convert_to_answer_format.py` 转换。
- [ ] 让 Baseline ReAct 走官方 tool schema、official server、official output format。
- [ ] 将 ToolBench 最大交互步数对齐官方默认：`single_chain_max_step=50`。
- [ ] 跑 Baseline ReAct one-pass，保存 6 个 subset 的完整轨迹。
- [ ] 用官方 `convert_to_answer_format.py` 转换 Baseline 输出。
- [ ] 用官方 `eval_pass_rate.py` 计算 Baseline official SoPR。
- [ ] 扩展 ToolBench memory entry，记录 `query_id/subset/category/tool_name/api_name/function_name/failure_type/error_signature`。
- [ ] 实现 online shared memory：从空 memory 开始，运行中写入，后续失败触发检索。
- [ ] 实现 tool/API-aware retrieval：优先同 `category + tool_name + api_name`，再同 `category + tool_name`，最后全局抽象策略。
- [ ] 将检索结果只在 failure-triggered in-loop 时注入 prompt。
- [ ] 让 Online React_FM 走官方 tool schema、official server、official output format。
- [ ] 跑 Online React_FM one-pass，保存 6 个 subset 的完整轨迹和 memory 日志。
- [ ] 用官方 `convert_to_answer_format.py` 转换 Online React_FM 输出。
- [ ] 用官方 `eval_pass_rate.py` 计算 Online React_FM official SoPR。
- [ ] 汇总主实验结果：Baseline ReAct vs Online React_FM。
- [ ] 报告整体 SoPR、subset-level SoPR、token cost。
- [ ] 检查论文表述，确保 ToolBench 指标写为 official SoPR，不再使用非空 `Finish` 率作为主结果。
- [ ] 保存所有运行命令、结果路径、配置和 evaluator 设置，保证实验可复现。

# 安全清理清单（2026-04-21）

## 1. 清理边界

本清单遵循两个硬约束：

- 不删除论文中已使用的数据、图表和正式结果
- 不删除 `doc/benchmark_experiment_runbook.md`、`doc/scienceworld_step4_summary.md`、`doc/scienceworld_gate_pipeline.md` 已明确引用的落盘文件

因此，这一轮只做“保留集”和“候选清理集”的划分，不建议直接整目录清空。

## 2. 明确保留

### 论文相关

论文源码实际使用了以下图：
- `paper/figures/architecture.pdf`
- `paper/figures/alfworld_per_task.pdf`
- `paper/figures/learning_curve.pdf`

对应的 `paper/figures/` 整体建议暂时保留，不做删除。

### 文档已引用的正式产物

按文档扫描结果，至少保留以下集合：

- `results/`：已被正式引用 `16` 项
- `memory_store/`：已被正式引用 `10` 项
- `checkpoints/`：已被正式引用 `17` 项
- `rollouts/`：已被正式引用 `15` 项
- `gate_models/`：已被正式引用 `55` 项
- `memory/`：已被正式引用 `6` 项
- `logs/`：已被正式引用 `13` 个具体日志，另有一个文档中的通配模式 `logs/20*/sw*.log`

最核心、最不能动的一批是：
- `results/20260415_153753/sw_step4_expA_online_test_1.json`
- `results/20260416_163135/sw_step4_expB_collect_train_1.json`
- `results/20260417_092133/sw_step4_expB_offline_test_1.json`
- `results/20260418_130202/sw_step4_expC_react_baseline_test_1.json`
- `results/20260420_180544/sw_step4_reflexion_online_formal_pass0.json`
- `results/20260420_180544/sw_step4_reflexion_online_formal_pass1.json`
- `memory_store/20260415_153753/sw_epoch1.json`
- `memory_store/20260416_163135/sw_epoch1.json`
- `memory_store/20260417_092133/sw_epoch1.json`
- `logs/20260415_153753/sw_step4_expA_online_test.log`
- `logs/20260417_092133/sw_step4_expB_offline_test.log`
- `logs/20260418_130202/sw_step4_expC_react_baseline_test.log`
- `logs/20260420_180544/sw_step4_reflexion_online_formal.log`

## 3. 候选清理

### A 类：高优先级，可先处理

这些内容不属于论文正式结果，也不是 runbook 已点名资产：

- 顶层异常目录 `~/`
- `data/StableToolBench/.git`
- `data/StableToolBench/openai_key.json`
- `config.yaml`、`config_scienceworld.yaml` 中的明文 `api_key`

这部分优先级最高，因为分别涉及仓库污染、子仓库残留和安全风险。

### B 类：优先归档而不是立刻删

这批大多是“未被文档明确引用”的实验日志和中间结果，适合先打包归档，再决定是否删除：

- `results/`：未被引用约 `237` 项
- `memory_store/`：未被引用约 `121` 项
- `checkpoints/`：未被引用约 `11` 项
- `rollouts/`：未被引用约 `17` 项
- `gate_models/`：未被引用约 `165` 项
- `memory/`：未被引用约 `3` 项
- `logs/`：未被引用约 `213` 个文件

其中最值得优先归档的大日志目录是：
- `logs/20260417_102644` 约 `6.3G`
- `logs/20260416_163135` 约 `4.2G`
- `logs/20260415_161618` 约 `4.1G`
- `logs/20260414_005940` 约 `3.4G`
- `logs/20260414_051341` 约 `2.6G`
- `logs/20260407_135442` 约 `1.5G`

注意：`logs/20260416_163135` 虽然目录很大，但其中对应的正式 step4 日志和结果有关，不建议整目录直接删。

## 4. 建议执行顺序

1. 先改密钥：改为环境变量，不再在配置文件中存明文 key
2. 再处理仓库污染：隔离 `~/`、`data/StableToolBench/.git`、`openai_key.json`
3. 建立保留白名单：把文档已引用文件单独记录
4. 对未引用日志先归档，再评估删除
5. 最后再处理未引用的 `results/`、`memory_store/`、`gate_models/` 等中间产物

## 5. 当前建议

现阶段最稳妥的做法不是“删目录”，而是“先白名单保护，再归档大头”。如果继续清理，我建议下一步直接生成一份机器可执行的白名单文件。

# 仓库排查总结（2026-04-21）

## 1. 当前状态

- 当前分支：`main`
- Git 已跟踪文件仅 `25` 个，主体仍是最早期 ALFWorld 主线代码
- 工作区存在大量未提交内容：新 benchmark、新实验脚本、论文、分析结果、模型产物、日志、数据目录全部混在同一个仓库里
- 当前 `git status` 同时包含：
  - 已跟踪文件的修改：`config.yaml`、`src/*`、`experiments/run_alfworld.py`、`CLAUDE.md` 等
  - 已跟踪文件的删除：`BENCHMARK_LANDSCAPE.md`、`IDEA_REPORT.md`、`refine-logs/EXPERIMENT_PLAN.md`、`refine-logs/FINAL_PROPOSAL.md`
  - 大量未跟踪目录：`checkpoints/`、`rollouts/`、`memory/`、`gate_models/`、`doc/`、`paper/`、`data/`、`experiments/gate/` 等

## 2. 主要问题

### A. 代码、数据、实验产物未分层

- `src/`、`experiments/` 中是源码
- `logs/`、`results/`、`memory_store/` 是运行产物
- `checkpoints/`、`rollouts/`、`gate_models/`、`memory/` 更像中间结果或实验资产
- `paper/` 同时包含源文件和编译产物
- `data/StableToolBench/` 像外部项目快照，且内嵌了自己的 `.git`

目前这些内容没有清晰边界，导致仓库职责不明确。

### B. `.gitignore` 覆盖不足

当前只忽略了：
- `logs/`
- `results/`
- `memory_store/`

未忽略但明显会持续膨胀的目录包括：
- `checkpoints/`
- `rollouts/`
- `gate_models/`
- `memory/`
- `paper/` 中的编译产物
- `data/StableToolBench/` 的子仓库与缓存

### C. 安全风险

- `config.yaml`
- `config_scienceworld.yaml`

这两个文件都直接写入了 `api_key`

另外还发现：
- `data/StableToolBench/openai_key.json`
- `data/StableToolBench/.git`

这说明仓库里混入了外部项目密钥文件和子仓库元数据，风险较高。

### D. 体积失控

当前目录体积中，最突出的几项是：
- `logs/`：约 `33G`
- `data/`：约 `2.7G`
- `~/`：约 `2.2G`，这是明显不应出现在仓库内的杂项目录
- `memory_store/`：约 `275M`
- `results/`：约 `132M`
- `checkpoints/`：约 `106M`
- `rollouts/`：约 `49M`

另外顶层已有：
- `181` 个日志批次目录
- `84` 个结果目录
- `53` 个 memory store 目录

## 3. 建议的清理优先级

### 清理硬约束

以下内容在当前阶段视为“保留资产”，不进入删除范围：

1. 论文直接使用或可能继续引用的数据与图表资产
2. `doc/benchmark_experiment_runbook.md` 中已经明确记录路径的落盘结果
3. `doc/scienceworld_step4_summary.md`、`doc/scienceworld_gate_pipeline.md` 中已作为正式口径引用的结果文件

按当前文档引用情况，至少应保留以下类别：
- `results/` 中被 runbook 和 step4 summary 明确点名的结果文件
- `memory_store/` 中被 runbook 明确作为正式 memory 输入/输出的文件
- `checkpoints/`、`rollouts/`、`memory/`、`gate_models/` 中被 gate pipeline 或 runbook 明确引用的正式文件
- `paper/figures/` 下论文使用的图

因此，现阶段可清理的重点应优先放在“未被文档引用的冗余日志、缓存、临时目录、子仓库残留、密钥文件”，而不是直接清空整个产物目录。

### P0：先止血

1. 立即移除明文 key，统一改为环境变量
2. 把 `data/StableToolBench/.git` 和 `openai_key.json` 从主仓库边界中隔离出去
3. 处理顶层 `~/` 杂项目录

### P1：理清版本控制边界

1. 保留源码：`src/`、`experiments/`、`prompts/`、必要的 `doc/`
2. 明确实验资产是否要入库：
   - 对“已被论文或 runbook 引用”的资产，先保留原位或建立清单后再迁移
   - 对“未被引用的中间产物”，再考虑忽略 `checkpoints/`、`rollouts/`、`gate_models/`、`memory/`
   - 若需要长期保存，建议迁到独立 `artifacts/` 或外部存储
3. `paper/` 只保留源码，忽略 `aux/log/pdf/bbl/blg/fls/fdb_latexmk/out`

### P2：整理提交历史

1. 先把“源码改动”和“实验产物”拆开
2. 再把“文档迁移”单独整理
3. 最后决定哪些新增 benchmark 功能要正式提交，哪些只是本地研究材料

## 4. 结论

现在的问题不是单点 bug，而是仓库边界失控：主仓库同时承担了代码仓、数据仓、实验缓存仓、论文仓、外部子项目仓。下一步最合理的动作不是继续堆功能，而是先做一次“边界重构”。

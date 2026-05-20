<!-- This document records how the local WebShop Reflexion runner should align with the Reflexion paper and the authors' official WebShop implementation. -->

# WebShop Reflexion 论文对齐说明

## 目标

本文档记录本仓库 `experiments/run_reflexion_webshop.py` 与 Reflexion 论文中 WebShop 附录实验的对齐要求。

这里的“对齐”特指 Reflexion 方法本身：trial 组织方式、memory 更新方式、self-reflection prompt、ReAct actor interface、成功判定和结果曲线。WebShop 数据集、test split、Task Score 等底层 benchmark 协议另见 [doc/webshop_alignment_plan.md](/Users/zhoey/React_FM/doc/webshop_alignment_plan.md)。

参考来源：

- Reflexion 论文：<https://arxiv.org/abs/2303.11366>
- WebShop 论文：<https://arxiv.org/abs/2207.01206>
- Reflexion 官方 WebShop 实现：<https://github.com/noahshinn/reflexion/tree/main/webshop_runs>
- 当前本地 runner：[experiments/run_reflexion_webshop.py](/Users/zhoey/React_FM/experiments/run_reflexion_webshop.py)

## 论文中的 WebShop Reflexion 设置

Reflexion 论文附录 B.1 把 WebShop 作为 limitation case：作者在 100 个 customer shopping requests 上比较 ReAct 与 ReAct + Reflexion，并说明 ReAct + Reflexion 在 WebShop 上没有明显优于 ReAct。

论文和官方代码共同体现的关键设置：

- 任务数量：100 个 WebShop environments / customer shopping requests。
- Trial 曲线：论文图展示到第 4 个 trial 左右，并说明因为没有改善而终止。
- 官方脚本默认运行更多 trial：`webshop_runs/run_reflexion.sh` 使用 `--num_trials 10 --num_envs 100`，论文分析时在约 4 trials 后停止。
- 每个 trial 对未成功的同一 session 重新尝试；已经成功的 session 后续 trial 跳过。
- 失败 session 在 trial 结束后生成自然语言 reflection，并追加到该 session 的 memory。
- 下一轮 actor 在 episode 开头读入 memory。
- actor 使用 ReAct 风格轨迹，包括 `think[...]`、`search[...]`、`click[...]`。
- actor 和 self-reflection 阶段都只使用最近 3 条 reflection memory。
- WebShop success 使用 `reward == 1.0`，不是 `reward >= 0.5`。

## 当前实现状态

当前 [experiments/run_reflexion_webshop.py](/Users/zhoey/React_FM/experiments/run_reflexion_webshop.py) 已具备 Reflexion 的基本外壳：

- 每个 WebShop session 有独立 `memory` 和 `is_success` 状态。
- 每个 trial 只重跑未成功 session，已成功 session 会 skip。
- 失败后调用 `generate_reflection(..., domain="webshop")` 生成反思。
- 下一 trial 会把 memory 注入到 episode 开头。

但当前实现仍有若干和论文 / 官方代码不一致的地方，不能直接称为 paper-aligned Reflexion WebShop 复现。

## 未对齐点与修正要求

### 1. Trial 数默认不足以复现论文 WebShop 曲线

当前行为：

- `run_reflexion_webshop.py` 默认 `--num-trials 2`。

为什么不对齐：

- Reflexion 论文 WebShop 附录关注的是多轮 trial 后是否持续改善。
- 只跑 2 个 trial 只能观察“初次尝试 + 一次反思重试”，不能复现论文中“到第 4 轮仍没有明显改善”的 limitation 结论。

对齐要求：

- 如果目标是复现 Reflexion paper WebShop limitation figure，至少运行 4 个 trial。
- 建议正式命令显式写出 trial 数，避免误用默认值：

```bash
python experiments/run_reflexion_webshop.py --num-trials 4 --max-envs 100 --max-steps 15
```

验收标准：

- 正式 Reflexion WebShop 对齐结果至少保存 trial 0、1、2、3 四个结果文件。
- 结果报告应展示按 trial 变化的 success rate / average reward，而不是只报告最后一轮。

### 2. Success 判定应使用 `reward == 1.0`

当前行为：

- episode result 中使用 `final_reward >= 0.5` 作为 success。
- summary 中也使用 `reward >= 0.5` 统计 success。

为什么不对齐：

- Reflexion 官方 WebShop runner 在 done 后返回 `reward == 1.0`。
- WebShop 论文定义 Success Rate 为 `r = 1` 的 episode 比例。
- 使用 `reward >= 0.5` 会把部分匹配商品当作成功，并导致这些 session 在后续 trial 被错误 skip。

对齐要求：

- 所有 WebShop Reflexion success 统一为 `reward == 1.0`。
- summary 增加 `success_definition: "reward == 1.0"`。
- 同时报告 `avg_reward` 和 `task_score = 100 * avg_reward`。

验收标准：

- `reward = 0.5` 的 episode 在 result 和 summary 中都计为失败。
- 后续 trial 只 skip `reward == 1.0` 的 session。

### 3. WebShop self-reflection prompt 缺少官方 few-shot

当前行为：

- [src/reflexion_agent.py](/Users/zhoey/React_FM/src/reflexion_agent.py) 只为 `domain == "alfworld"` 加载 `prompts/reflexion_few_shot_examples.txt`。
- `domain == "webshop"` 时没有 WebShop-specific reflection examples。

为什么不对齐：

- Reflexion 官方 WebShop 实现有 `webshop_runs/reflection_few_shot_examples.txt`。
- 论文 WebShop failure case 的核心观察之一是 self-reflection 不够有帮助；如果本地 prompt 与官方不同，reflection 质量差异会混入结果。

对齐要求：

- 新增 WebShop reflection few-shot 文件，内容风格对齐官方 Reflexion WebShop examples。
- `generate_reflection(..., domain="webshop")` 应加载 WebShop-specific examples。
- reflection query 应从 `Instruction:` 或当前 WebShop task block 中抽取 scenario，并包含失败轨迹。

验收标准：

- WebShop reflection prompt 中能看到两个 WebShop 失败反思例子。
- 生成的 reflection 以具体搜索 / 点击策略为核心，而不是泛泛总结环境。

### 4. Actor 应保留 ReAct 的 `think[...]`

当前行为：

- WebShop system prompt 要求只输出 `search[...]` 或 `click[...]`。
- action 解析逻辑在检查 `think` 前，会把非 `search/click` 输出 fallback 成 `search[product]`。
- 因此即使模型输出 `think[...]`，也很可能无法进入 `think` 分支。

为什么不对齐：

- Reflexion WebShop 官方 base prompt 使用 ReAct 轨迹，包含 `think[...]`。
- 论文描述的是 ReAct + Reflexion，而不是 action-only Reflexion。

对齐要求：

- Reflexion WebShop actor prompt 允许 `think[...]`、`search[...]`、`click[...]`。
- `think[...]` 应作为内在思考动作处理，环境 observation 返回 `OK.`，不发送给 WebShop environment。
- action 解析顺序应先识别 `think[...]`，再处理 `search/click` fallback。

验收标准：

- 单元测试覆盖 `think[...]` 不会被改写成 `search[product]`。
- 运行日志中允许出现 `think[...]` 和 `OK.` 观察。

### 5. Memory 注入应限制最近 3 条 reflection

当前行为：

- actor prompt 注入全部 `memory`。
- reflection 生成时也传入全部历史 memory。

为什么不对齐：

- Reflexion 论文说明实践中 memory 通常限制为 1-3 条。
- 官方 WebShop implementation 对 actor 和 reflection query 都使用最近 3 条 memory。

对齐要求：

- actor prompt 注入 `memory[-3:]`。
- self-reflection query 中的 `Plans from past attempts` 也只包含 `memory[-3:]`。

验收标准：

- 当某个 session 已有 4 条以上 reflection 时，prompt 只出现最近 3 条。

### 6. Prompt 格式应和 WebShop 官方 Reflexion 更接近

当前行为：

- 本地 `FEWSHOT_EXAMPLE` 是一条简化 shopping 示例。
- 官方 Reflexion WebShop base prompt 使用 `Webshop`、`Instruction:`、`Action:`、`Observation:` 格式，并包含 `think[...]`。

为什么不对齐：

- Reflexion 的效果高度依赖 prompt 格式。
- 如果 actor prompt 改成 action-only 简化版，结果更像一个本项目自定义 baseline，而不是论文中的 ReAct + Reflexion。

对齐要求：

- `run_reflexion_webshop.py` 使用独立的 Reflexion WebShop prompt，不复用 action-only WebShop prompt。
- prompt 格式至少包含：
  - WebShop task instruction。
  - few-shot ReAct shopping trajectory。
  - 可选 memory block：`Your memory for the task below:`。
  - 历史 `Action:` / `Observation:` 或 `> action` / observation，但同一 runner 内要保持一致。

验收标准：

- prompt 明确允许 `think[...]`。
- prompt 示例中至少出现一次 search、think、click option、Buy Now。

## 推荐对齐命令

用于复现 Reflexion 论文 WebShop limitation figure 的最小命令：

```bash
python experiments/run_reflexion_webshop.py \
  --num-trials 4 \
  --max-envs 100 \
  --max-steps 15 \
  --run-name ws_reflexion_paper_aligned_
```

如果本地目标是使用完整 WebShop 官方数据和 split，还需要同步满足 [doc/webshop_alignment_plan.md](/Users/zhoey/React_FM/doc/webshop_alignment_plan.md) 中的数据与评估协议要求。

## 结果报告要求

正式报告中应包含：

- 每个 trial 的 `success_rate`，使用 `reward == 1.0`。
- 每个 trial 的 `avg_reward`。
- 每个 trial 的 `task_score = 100 * avg_reward`。
- 已成功 session 的累计 skip 数。
- 每个 trial 新增 reflection 数。
- 运行参数：`num_trials`、`max_envs`、`max_steps`、`num_products`、WebShop split / session ids。

不应写法：

- 不把 `reward >= 0.5` 称为 WebShop Success Rate。
- 不把 2-trial 快速结果称为 Reflexion paper WebShop figure reproduction。
- 不把 action-only prompt 的结果直接称为 ReAct + Reflexion。

## 实现 TODO

- [x] 将 `run_reflexion_webshop.py` 的 success 判定改为 `reward == 1.0`。
  状态：已完成；episode result、summary、skip 逻辑均使用 exact success。
- [x] 在 summary 中增加 `success_definition` 和 `task_score`。
  状态：已完成；summary 现在同时保存 `success_definition: "reward == 1.0"`、`avg_reward` 和 `task_score = 100 * avg_reward`。
- [x] 默认或 paper-aligned preset 使用至少 4 trials。
  状态：已完成；`run_reflexion_webshop.py` 默认 `--num-trials 4`。
- [x] 新增 WebShop-specific reflection few-shot prompt。
  状态：已完成；新增 [prompts/webshop_reflexion_few_shot_examples.txt](/Users/zhoey/React_FM/prompts/webshop_reflexion_few_shot_examples.txt)。
- [x] 让 `generate_reflection(..., domain="webshop")` 加载 WebShop reflection examples。
  状态：已完成；WebShop reflection query 现在注入 WebShop-specific examples。
- [x] 允许 actor 输出并处理 `think[...]`。
  状态：已完成；Reflexion WebShop runner 使用独立 system prompt，并在发送环境动作前保留 `think[...]` 为 `OK.` 内部动作。
- [x] actor 和 reflection prompt 均只注入最近 3 条 memory。
  状态：已完成；actor prompt 和 self-reflection prompt 都使用最近 3 条 reflection memory。
- [x] 为 success 判定、memory 截断、`think[...]` 解析增加单元测试。
  状态：已完成；新增 [tests/test_webshop_reflexion_alignment.py](/Users/zhoey/React_FM/tests/test_webshop_reflexion_alignment.py)。

<!-- This document audits and plans alignment between the local WebShop ExpeL runner and the ExpeL paper/official implementation. -->

# WebShop ExpeL Paper Alignment

## Scope

This note audits `experiments/run_expel_webshop.py` against ExpeL, "LLM Agents Are Experiential Learners" (arXiv:2308.10144), and the official `LeapLabTHU/ExpeL` implementation. The goal is to separate three things:

1. What is already aligned enough to keep.
2. What is currently ExpeL-inspired but not paper-aligned.
3. What we should modify first without hiding larger remaining gaps.

## Paper Reference Protocol

The ExpeL paper uses three stages:

1. Experience gathering on training tasks.
   - Use Reflexion-style retries, up to 3 reflection retries.
   - Store all trajectories in an experience pool.
   - A task can contribute both failed and successful trajectories.
2. Insight extraction from the experience pool.
   - Compare success/failure trajectories for the same task.
   - Also analyze chunks of successful trajectories.
   - Maintain a rule set through ADD, EDIT, REMOVE/DOWNVOTE, and AGREE/UPVOTE operations with importance counts.
3. Evaluation on held-out tasks.
   - Each evaluation task gets a single attempt.
   - The prompt is augmented with the learned insight list.
   - The prompt also receives top-k successful trajectories retrieved by task similarity from the experience pool.

For WebShop, the paper reports:

- 100 fixed WebShop tasks shared with ReAct/Reflexion.
- 2 manual WebShop few-shot examples.
- Maximum environment steps `H = 15`.
- Top-k retrieved successful examples `k = 2`.
- Insight success chunk size `L = 4`.
- Success rate uses exact full reward, equivalent to `reward == 1.0`.
- WebShop task score is the mean reward, often reported as `100 * avg_reward` in WebShop-oriented summaries.

## Current Local Implementation

The local runner currently does this:

- Runs epoch 1 on `max_envs` sessions.
- Extracts JSON insight rules after epoch 1.
- Runs epoch 2 on the same session ids, injecting only insight rules.
- Skips sessions already considered successful.
- Uses one simplified WebShop few-shot example.
- Defaults to `num_products=1000`.
- Uses `reward >= 0.5` as success.
- Does not expose the official WebShop wrapper, split, observation mode, sample ids, or valid-action interface.

## Alignment Gaps

### 1. Train/Test Separation

The runner reuses the same sessions across epochs. This is useful for a quick ablation, but it is not the paper's train/eval protocol. Paper-aligned ExpeL should gather experience on training tasks and evaluate on held-out tasks.

Impact: current epoch 2 can leak test-task experience into evaluation.

### 2. Reflexion-Based Experience Gathering

The paper gathers experience with Reflexion retries. The local runner only runs one ReAct-style attempt in epoch 1.

Impact: fewer successful trajectories are collected, and there are no same-task failed/successful pairs produced by retrying the same task.

### 3. Insight Extraction

The local extractor directly asks for a JSON list of rules. The paper iteratively updates a rule set with operations and counts. The local runner also pairs each failure with a random success, while the paper compares successes and failures for the same task when available and also processes chunks of successful trajectories.

Impact: insight quality, stability, and interpretability differ from ExpeL.

### 4. Missing Successful-Trajectory Retrieval

The paper uses both insights and retrieved successful trajectories. The local runner only injects insight rules.

Impact: the local result is closer to an insights-only ablation than full ExpeL.

### 5. WebShop Evaluation Interface

The local runner lacks the paper-aligned WebShop knobs that already exist in `experiments/run_webshop.py`: full product data, official wrapper, split selection, observation mode, human goals, sample ids, and valid actions.

Impact: WebShop difficulty and action validity can differ across baselines.

### 6. Success Metric

The local runner treats `reward >= 0.5` as success. WebShop success for paper comparison should be `reward == 1.0`.

Impact: partial purchases are counted as full success, inflating success rate.

## Modification Plan

### Phase 1: Evaluation-Protocol Alignment

Implement now:

- Change WebShop ExpeL success to `reward == 1.0`.
- Add `success_definition` and `task_score` to summaries.
- Add fixed sample-id loading/sampling helpers.
- Change `--num-products` to support `full`, matching the existing WebShop runner.
- Add CLI flags for `--eval-split`, `--eval-sample-size`, `--eval-sample-seed`, `--sample-ids`, `--observation-mode`, `--human-goals`, and `--webshop-wrapper`.
- Pass those options into `WebShopEnv`.
- Use official valid actions in the prompt and normalize model output against them.

This phase does not make the implementation full ExpeL. It makes the current simplified runner much less misleading and easier to compare with other local WebShop runners.

### Phase 2: Experience Pool

Implement later:

- Store successful trajectories separately from insights.
- Build a task-similarity retrieval pool over successful trajectories.
- Inject top-k retrieved successful trajectories into evaluation prompts.
- Keep manual few-shot examples distinct from retrieved examples.

### Phase 3: Paper-Style Training and Insight Extraction

Implement later:

- Split experience gathering from evaluation.
- Use Reflexion retries for training tasks.
- Extract insights from same-task success/failure comparisons.
- Add successful-trajectory chunk extraction.
- Replace direct JSON insight extraction with operation-based rule updates and importance counts.

## Naming Guidance

Until all three phases are complete, reports should call this runner:

- `webshop_expel_simplified`
- or `webshop_expel_insights_only`

It should not be described as fully paper-aligned ExpeL.

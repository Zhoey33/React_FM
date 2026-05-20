# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

React_FM: Failure-triggered in-loop memory for ReAct LLM agents. Evaluated on three benchmarks (ALFWorld, WebShop, ScienceWorld) against Baseline ReAct and Reflexion.

- **Agent LLM**: Qwen3.5-9B (ALFWorld/WebShop), DeepSeek-V3.2 (ScienceWorld) via SiliconFlow API
- **Judge LLM**: Same as agent model (failure detection)
- **Extractor LLM**: DeepSeek-V3.2 (post-episode memory extraction for all benchmarks)
- **Embedding**: all-MiniLM-L6-v2 (local, MPS)
- **Runtime**: Python 3.13.9, macOS M4

## Commands

```bash
# ALFWorld (134 envs)
caffeinate -i python experiments/run_alfworld.py --baseline --max-envs 134 --run-name baseline_
caffeinate -i python experiments/run_alfworld.py --epochs 2 --max-envs 134 --run-name reactfm_
caffeinate -i python experiments/run_reflexion.py --num-trials 2 --max-envs 134 --run-name reflexion_

# WebShop (100 sessions) — requires conda env `webshop` with Java
caffeinate -i python experiments/run_webshop.py --baseline --max-envs 100 --run-name ws_baseline_
caffeinate -i python experiments/run_webshop.py --epochs 2 --max-envs 100 --run-name ws_reactfm_

# ScienceWorld (50 episodes) — requires scienceworld pip package + Java 1.8+
caffeinate -i python experiments/run_scienceworld.py --baseline --max-envs 50 --run-name sw_baseline_
caffeinate -i python experiments/run_scienceworld.py --epochs 2 --max-envs 50 --run-name sw_reactfm_

# Ablation experiments (ALFWorld, 134 envs)
python experiments/run_alfworld.py --epochs 2 --max-envs 134 --inject-mode episode --run-name ablation_episode_
python experiments/run_alfworld.py --epochs 2 --max-envs 134 --memory-format success_trajectory --run-name ablation_success_
python experiments/run_alfworld.py --epochs 2 --max-envs 134 --memory-format reflexion_reflection --run-name ablation_reflexion_

# Resume interrupted experiment
python experiments/run_alfworld.py --resume --epochs 2 --max-envs 134 --run-name reactfm_

# Sanity test (3 envs)
python experiments/run_alfworld.py --baseline --max-envs 3 --run-name sanity_
```

## Architecture

### React_FM Pipeline (per step)

```
LLM → action → env.step() → observation
                    ↓
         failure_detector.detect()  (explicit: pattern rules, implicit: LLM judge)
                    ↓ (if failure)
         memory.retrieve(env_idx)   (BM25 + embedding RRF, per-env isolation)
                    ↓
         build_user_prompt(hints) → LLM (next step, hints cleared after use)
                    ↓ (post-episode)
         memory_extractor → memory.add(env_idx)
```

### Per-Benchmark Component Pattern

Each benchmark implements three components following the same interface:

| Component | ALFWorld | WebShop | ScienceWorld |
|-----------|----------|---------|--------------|
| Env wrapper | `src/alfworld_env.py` | `src/webshop_env.py` | `src/scienceworld_env.py` |
| Failure detector | `src/failure_detector.py` | `src/webshop_failure_detector.py` | `src/scienceworld_failure_detector.py` |
| Memory extractor | `src/memory_extractor.py` | `src/webshop_memory_extractor.py` | `src/scienceworld_memory_extractor.py` |
| Prompts | `prompts/alfworld_prompts.py` | `prompts/webshop_prompts.py` | `prompts/scienceworld_prompts.py` |
| Runner | `experiments/run_alfworld.py` | `experiments/run_webshop.py` | `experiments/run_scienceworld.py` |

The agent (`src/agent.py` `ReactFMAgent`) is shared across all benchmarks. Each runner wires benchmark-specific components into it.

### Key Shared Files

| File | What |
|------|------|
| `src/agent.py` | `ReactFMAgent` — main loop, failure detection, memory inject |
| `src/memory.py` | `FailureMemoryStore` — per-env BM25+embedding RRF retrieval |
| `src/llm.py` | OpenAI-compatible client with retry (3 attempts) + token tracking |
| `src/log_utils.py` | Dual logging: console INFO + file DEBUG |
| `src/memory_extractor_ablation.py` | Alternative extractors for ablation (success_trajectory, reflexion_reflection) |
| `config.yaml` | LLM/agent/memory settings (ALFWorld default; other benchmarks use custom configs) |

### Agent Parameters

- `inject_mode`: `"in_loop"` (on failure, default) | `"episode"` (at start) | `"none"` (disabled)
- `memory_format`: `"failure_recovery"` (default) | `"success_trajectory"` | `"reflexion_reflection"`
- `memory_style`: `"original"` | `"factual"` | `"reflexion"` | `"hint"` — prompt formatting for injected memories

### Memory Storage

Per-env isolation: `_env_entries: dict[int, list[FailureMemoryEntry]]`. Each env has independent memory. Serialized as single JSON with env_idx keys. Retrieval uses BM25 + cosine embedding with RRF fusion (k=60), top-3 entries.

## Benchmark-Specific Notes

### ALFWorld
- `put X in/on Y` must be translated to `move X to Y` for the programmatic API (`alfworld_env.py:translate_action()`)
- Max 134 envs in `eval_out_of_distribution` — ALFWorld cycles after exhaustion
- Python 3.13 textworld bug: `eval()` in `textgen/__init__.py:97` — pass vars dict directly
- Set `HF_HUB_OFFLINE=1` at script top to prevent sentence-transformers from hitting HuggingFace

### WebShop
- Runs via subprocess bridge (`experiments/webshop_bridge.py`) in conda env `webshop` (Python 3.10)
- Direct Python binary: `/opt/miniconda3/envs/webshop/bin/python` (not `conda run` — stdin doesn't work)
- Requires `JAVA_HOME` pointing to OpenJDK for Lucene index
- Bridge uses stdin/stdout JSON protocol; startup output redirected to stderr until "READY" signal

### ScienceWorld
- Requires `scienceworld` pip package + Java 1.8+ (py4j backend)
- Uses DeepSeek-V3.2 as agent LLM (science reasoning needs larger model)
- `get_variations_test()` takes no args (task set in constructor)
- Valid actions via `"\n".join(env.get_valid_action_object_combinations())`

## Output Structure

```
results/{timestamp}/{run_name}{epoch}.json              # full results + summary
results/{timestamp}/{run_name}{epoch}_intermediate.json  # checkpoint every 10 envs
memory_store/{timestamp}/epoch{N}.json                   # per-env memory entries
logs/{timestamp}/{run_name}.log                          # DEBUG-level log
```

## Config

API key: set `SILICONFLOW_API_KEY` env var or in `config.yaml`. Three separate LLM client configs in config.yaml: `llm` (agent), `judge` (failure detection), `extractor` (memory extraction). ScienceWorld uses a custom config that overrides the agent model to DeepSeek-V3.2.

## Paper

LaTeX paper in `paper/` using ICLR 2025 style. Compile with:
```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
Figures generated via `python paper/figures/gen_figures.py` and `python paper/figures/gen_architecture.py`.

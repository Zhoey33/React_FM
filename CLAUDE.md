# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

React_FM: In-loop failure memory for ReAct agents on ALFWorld. Three-way comparison: Baseline ReAct vs Reflexion vs React_FM.

- **LLM**: Qwen3.5-9B (agent + judge) via SiliconFlow, DeepSeek-V3.2 (extractor)
- **Embedding**: all-MiniLM-L6-v2 (local, MPS)
- **Environment**: ALFWorld valid_unseen (134 envs, 6 task types)
- **Runtime**: Python 3.13.9, macOS M4

## Commands

```bash
# Sanity test (3 envs)
python experiments/run_alfworld.py --baseline --max-envs 3 --run-name sanity_baseline_
python experiments/run_alfworld.py --epochs 2 --max-envs 3 --run-name sanity_reactfm_
python experiments/run_reflexion.py --num-trials 2 --max-envs 3 --run-name sanity_reflexion_

# Full experiments (134 envs, use caffeinate -i to prevent sleep)
caffeinate -i python experiments/run_alfworld.py --baseline --max-envs 134 --run-name baseline_
caffeinate -i python experiments/run_alfworld.py --epochs 2 --max-envs 134 --run-name reactfm_
caffeinate -i python experiments/run_reflexion.py --num-trials 2 --max-envs 134 --run-name reflexion_

# Resume interrupted experiment
python experiments/run_alfworld.py --resume --epochs 2 --max-envs 134 --run-name reactfm_
```

## Architecture

### Three Systems (same LLM, different memory)

| System | Memory | When Used | Runner |
|--------|--------|-----------|--------|
| Baseline ReAct | None | — | `run_alfworld.py --baseline` |
| React_FM | Per-env (failure, solution) pairs | In-loop on failure | `run_alfworld.py` |
| Reflexion | Per-env free-text reflection | Injected at episode start | `run_reflexion.py` |

### React_FM Pipeline (per step)

```
LLM → action → env.step() → observation
                    ↓
         failure_detector.detect()
                    ↓ (if failure)
         memory.retrieve(env_idx) → hints
                    ↓
         build_user_prompt(hints) → LLM (next step)
                    ↓ (post-episode)
         memory_extractor → memory.add(env_idx)
```

### Key Files

| File | What |
|------|------|
| `src/agent.py` | `ReactFMAgent` — main loop, failure detection, memory inject |
| `src/memory.py` | `FailureMemoryStore` — per-env BM25+embedding RRF retrieval |
| `src/failure_detector.py` | Rule-based ("Nothing happens", loops) + LLM judge |
| `src/memory_extractor.py` | Post-episode LLM extraction of failure-recovery pairs |
| `src/alfworld_env.py` | ALFWorld wrapper, `translate_action()` for put→move |
| `src/llm.py` | OpenAI-compatible client with retry + token tracking |
| `src/log_utils.py` | Dual logging: console INFO + file DEBUG (`logs/{timestamp}/`) |
| `prompts/alfworld_prompts.py` | System prompts + user prompt builder with memory injection |
| `config.yaml` | All LLM/agent/memory/experiment settings |

### Memory Storage

Per-env isolation: `_env_entries: dict[int, list[FailureMemoryEntry]]`. Each env has independent memory. Serialized as single JSON with env_idx keys. Retrieval uses BM25 + cosine embedding with RRF fusion (k=60).

## ALFWorld Quirks

- **`put X in/on Y` fails**: ALFWorld programmatic API uses `move X to Y`. Fixed via `translate_action()` in alfworld_env.py.
- **Max 134 envs**: valid_unseen has exactly 134 games. ALFWorld cycles after exhaustion.
- **Python 3.13 textworld bug**: `eval()` in `textgen/__init__.py:97` — `locals().update()` broken. Fix: pass vars dict directly.
- **Rewards/dones**: ALFWorld returns tuples not lists, use `isinstance(x, (list, tuple))`.
- **HF_HUB_OFFLINE=1**: Set at top of experiment scripts to prevent sentence-transformers from hitting HuggingFace.

## Output Structure

```
results/{run_name}{epoch}.json          # full results + summary
results/{run_name}{epoch}_intermediate.json  # checkpoint every 10 envs
memory_store/epoch{N}.json              # per-env memory entries + embeddings
logs/{YYYYMMDD_HHMMSS}/{run_name}.log   # DEBUG-level log
```

## Config

API key: set `SILICONFLOW_API_KEY` env var or `llm.api_key` in config.yaml. The `judge` and `extractor` sections in config.yaml configure separate LLM clients for React_FM's failure detection and memory extraction.

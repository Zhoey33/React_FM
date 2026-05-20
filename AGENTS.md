# Repository Guidelines

## Project Structure & Module Organization
Core runtime code lives in `src/`: shared agent, memory store, LLM client, benchmark env wrappers, failure detectors, and gate utilities under `src/gate/`. Entry points live in `experiments/`, including baseline React_FM runs, Reflexion/ExpeL variants, and gate-training scripts. Prompt builders are in `prompts/`. Generated artifacts are written to `results/`, `logs/`, and `memory_store/`. Analysis notes and runbooks live in `doc/` and `analysis/`. Paper sources and figure scripts are in `paper/` and `paper/figures/`.

## Build, Test, and Development Commands
Install the Python environment with `pip install -r requirements.txt`.

Common runs:
- `python experiments/run_alfworld.py --baseline --max-envs 3 --run-name sanity_` runs a quick ALFWorld smoke test.
- `python experiments/run_scienceworld.py --config config_scienceworld.yaml --baseline --max-envs 5 --run-name sw_smoke_` checks ScienceWorld wiring.
- `python experiments/run_reflexion.py --num-trials 2 --max-envs 134 --run-name reflexion_` runs the ALFWorld Reflexion baseline.
- `cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex` builds the paper.

Some benchmarks require extra system setup: ScienceWorld needs Java plus the `scienceworld` package; WebShop uses a dedicated conda environment and Java-backed search.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, type hints on new code, dataclasses for structured episode results, and `snake_case` for functions/files/CLI flags. Use `PascalCase` for classes such as `ReactFMAgent` and `FailureMemoryStore`. Keep benchmark-specific logic in the matching file pair, for example `src/scienceworld_env.py` and `src/scienceworld_failure_detector.py`. Prefer small helper functions over deeply nested control flow.

## Testing Guidelines
There is no dedicated `tests/` directory today. Validate changes with targeted smoke runs against the affected benchmark and record the exact command used. For non-runtime edits, run `python -m py_compile src experiments` to catch syntax issues. If you change result schemas or analysis code, verify one output JSON under `results/` and update any dependent docs.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects such as `Add ablation support` or `Add memory_style parameter`. Keep commits focused and mention the benchmark or subsystem touched. PRs should include: purpose, commands run, key output paths, and before/after metrics when behavior changes. Add plots or screenshots only when updating `paper/figures` or analysis docs.

## Security & Configuration Tips
Do not commit live API keys or regenerated benchmark secrets. Prefer environment variables such as `SILICONFLOW_API_KEY`, and treat `config.yaml` as local-only when adding credentials.

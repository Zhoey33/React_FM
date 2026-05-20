# LLM Agent Benchmark Landscape (2023-2026)

## Comprehensive Survey for React_FM Project

---

## MASTER TABLE: Sorted by Deployment Ease

### EASY Deployment (pip install + API key, pure text/API)

| # | Benchmark | Year | Venue | Task Type | # Tasks | Tools/APIs | Deployment | Popularity | Failure Recurrence | Memory Papers Using It |
|---|-----------|------|-------|-----------|---------|------------|------------|------------|-------------------|----------------------|
| 1 | **HotpotQA** | 2018 | EMNLP | Multi-hop QA + search tool | 112K (typically 100-500 sampled) | Wikipedia search API (2-3 tools) | **EASY** - pip install, just needs LLM API key | Very High (used by 50+ agent papers) | HIGH - search returns irrelevant docs, reasoning chain breaks, hallucination on multi-hop | Reflexion, ExpeL, LATS, AgentDebug, Agent KB |
| 2 | **FEVER** | 2018 | NAACL | Fact verification + search | 185K (sampled) | Wikipedia search API | **EASY** - same as HotpotQA | High | HIGH - similar search failures as HotpotQA | ExpeL (transfer target) |
| 3 | **ToolQA** | 2023 | NeurIPS | QA with external tools | 1,530 (easy+hard across 8 domains) | 13 tool types (calculator, DB query, graph tools, etc.) | **EASY** - pip install, text-based | Medium | HIGH - compositional tool failures, wrong tool selection, parameter errors | - |
| 4 | **API-Bank** | 2023 | EMNLP | Tool-augmented dialogue | 314 dialogues, 753 API calls (eval); 73 APIs | 53+ APIs (search, calendar, smart home, hotel, AI models) | **EASY** - pip install, runnable eval system | Medium | HIGH - plan/retrieve/call failures at each level | - |
| 5 | **BFCL** | 2024-25 | ICML 2025 | Function calling | 2,000+ test cases across 17 tasks | Synthetic function schemas (Python, Java, JS, REST) | **EASY** - `pip install bfcl-eval` | Very High (de facto standard for function calling) | MEDIUM - wrong arg types, missing params, irrelevance detection failures | - |
| 6 | **MINT** | 2024 | ICLR | Multi-turn interaction with tools + feedback | Curated subset from HumanEval, MBPP, GSM8K, HotpotQA, MATH, MMLU, TheoremQA, ALFWorld | Python interpreter + natural language feedback | **EASY** - pip install, text-based | Medium-High | HIGH - multi-turn degradation, feedback not helping, SIFT/RLHF hurting multi-turn | - |
| 7 | **GAIA** | 2023 | ICLR 2024 | General assistant (web + tools + reasoning) | 466 questions (3 levels) | Web browsing, file handling, calculator, code exec | **EASY** - HuggingFace dataset + Inspect AI eval | Very High (major leaderboard) | HIGH - multi-step planning failures, tool chaining errors, wrong info retrieval | AgentDebug, Agent KB |
| 8 | **GTA** | 2024 | NeurIPS | General tool use (multimodal) | 229 instances | 14 tools (perception, operation, logic, creativity) | **EASY-MEDIUM** - pip install, needs image files | Medium | HIGH - tool selection errors, argument prediction failures | - |
| 9 | **tau-bench** | 2024 | arXiv (Sierra) | Customer service (tool + policy + user) | 2 domains (retail + airline) | Domain-specific APIs (order mgmt, booking, etc.) | **EASY** - pip install from GitHub | High (industry standard) | VERY HIGH - policy violation, inconsistency across trials (pass^8 < 25%), API misuse | - |
| 10 | **StableToolBench** | 2024 | ACL Findings | Large-scale tool learning | 16,464 APIs, ~800 test queries | Real REST APIs via virtual API server (160K cached) | **EASY-MEDIUM** - pip install, virtual API server included | High | HIGH - wrong API selection from thousands, parameter hallucination | - |

### MEDIUM Deployment (some Docker/env setup)

| # | Benchmark | Year | Venue | Task Type | # Tasks | Tools/APIs | Deployment | Popularity | Failure Recurrence | Memory Papers Using It |
|---|-----------|------|-------|-----------|---------|------------|------------|------------|-------------------|----------------------|
| 11 | **ALFWorld** | 2021 | ICLR | Embodied household tasks (text mode) | 134 environments, 6 task types | Text actions (go to, pick up, put, open, close, heat, clean) | **MEDIUM** - pip install but leaks memory, needs ~4GB RAM | Very High | VERY HIGH - same 6 task types repeat, hallucinated actions, wrong exploration order, stuck in loops | Reflexion, ExpeL, LATS, MINT, AgentDebug |
| 12 | **WebShop** | 2022 | NeurIPS | Web shopping | 12,087 instructions, 1.18M products | Web navigation (search, click, select options) | **MEDIUM** - needs ~16GB RAM for product DB | Very High | HIGH - query reformulation failures, wrong product selection, attribute mismatch | Reflexion, ExpeL, LATS, AgentDebug |
| 13 | **InterCode** | 2023 | NeurIPS | Interactive coding (Bash, SQL, Python, CTF, SWE) | 5 environments, ~100-300 tasks each | Code execution with feedback (Docker) | **MEDIUM** - Docker required, `pip install intercode-bench` | Medium | HIGH - wrong commands, syntax errors, not using feedback | - |
| 14 | **AppWorld** | 2024 | ACL (Best Resource Paper) | Interactive coding with app APIs | 750 tasks, 9 apps, 457 APIs | 457 APIs across 9 simulated apps (Amazon, Spotify, etc.) | **MEDIUM** - pip install appworld, local backend | Medium-High | HIGH - complex control flow errors, wrong API sequences, collateral damage | - |
| 15 | **AgentBench** | 2024 | ICLR | Multi-environment agent eval | 8 environments (OS, DB, KG, games, web, household) | Environment-specific tools | **MEDIUM** - Docker, multiple containers, WebShop needs 16GB | High | HIGH - long-term reasoning failures, instruction following errors | - |
| 16 | **AgentBoard** | 2024 | NeurIPS | Multi-turn agent eval | 9 tasks, 1,013 environments | Embodied, game, web, tool interfaces | **MEDIUM** - Docker-based setup | Medium-High | HIGH - progress rate metric reveals partial completion patterns | - |
| 17 | **SWE-bench** | 2024 | ICLR (Oral) | Software engineering | 2,294 tasks (Lite: 300, Verified: 500) | Bash, code editing | **MEDIUM** - Docker-based eval harness, needs 120GB storage, 16GB RAM | Very High | HIGH - wrong file identification, incomplete patches, test failures | Agent KB |

### HARD Deployment (full environment: websites, VMs, complex infra)

| # | Benchmark | Year | Venue | Task Type | # Tasks | Tools/APIs | Deployment | Popularity | Failure Recurrence | Memory Papers Using It |
|---|-----------|------|-------|-----------|---------|------------|------------|------------|-------------------|----------------------|
| 18 | **WebArena** | 2024 | ICLR | Realistic web tasks | 812 tasks across 4 websites | Browser actions (click, type, scroll, navigate) | **HARD** - multiple Docker containers, ~13h per full eval run | Very High | HIGH - navigation errors, wrong element selection, state management | Agent KB |
| 19 | **Voyager** (Minecraft) | 2023 | TMLR | Open-ended embodied | Custom tech tree + exploration | Minecraft API via MineDojo, code generation | **HARD** - Minecraft server, MineDojo, GPU | High | MEDIUM - code generation errors, skill library retrieval | Voyager (itself, skill library = memory) |
| 20 | **TheAgentCompany** | 2025 | NeurIPS | Real-world professional tasks | Diverse professional tasks | Multiple real-world tools | **HARD** - complex deployment | Medium | HIGH - multi-step professional task failures | - |

---

## KEY PAPERS AND THEIR BENCHMARK CHOICES

| Paper | Venue | Benchmarks Used | Why These Benchmarks |
|-------|-------|----------------|---------------------|
| **Reflexion** | NeurIPS 2023 | HotpotQA (100), ALFWorld (134), WebShop (100), HumanEval, MBPP, LeetcodeHard | Core ReAct-style tasks with clear success/failure signals for self-reflection |
| **ExpeL** | AAAI 2024 | HotpotQA, ALFWorld, WebShop + FEVER (transfer) | Same as Reflexion for direct comparison; added FEVER for transfer learning |
| **LATS** | ICML 2024 | HotpotQA, WebShop, HumanEval, GSM8K | Multi-domain validation of tree search over reasoning + acting |
| **AgentDebug** | arXiv 2025 | ALFWorld (100 trajectories), GAIA (50), WebShop (50) | Diverse failure trajectories for error taxonomy; 200 annotated failures total |
| **Agent KB** | arXiv 2025 | GAIA, SWE-bench | Cross-domain knowledge transfer; challenging multi-step tasks |
| **Voyager** | TMLR 2023 | Custom Minecraft metrics (tech tree, items, distance) | No standard benchmark; open-ended exploration requires custom eval |
| **Trajectory-Informed Memory (IBM)** | ~2026 | Likely WebArena, ITBench (IBM's own) | Could not confirm exact benchmark; IBM focuses on enterprise agent evals |

---

## FAILURE PATTERN ANALYSIS (Critical for React_FM)

### Benchmarks with Highest Failure Recurrence

| Benchmark | Recurring Failure Types | Why Good for React_FM |
|-----------|------------------------|----------------------|
| **ALFWorld** | (1) Wrong exploration order (2) Hallucinated actions (3) Stuck in loops (4) Same 6 task types = same errors repeat | IDEAL: Only 6 task types, failures are highly patterned, memory of "don't explore fridge for cleaning tasks" directly transfers |
| **HotpotQA** | (1) Irrelevant search results derail reasoning (2) Multi-hop chain breaks at same points (3) Hallucination on bridge entities | GOOD: Search tool failures are systematic; memory of "search for X instead of Y" is reusable |
| **tau-bench** | (1) Policy violations repeat (2) API call parameter errors (3) Extreme inconsistency (pass^8 < 25%) | EXCELLENT: Same policy rules violated repeatedly; memory of policies should directly improve consistency |
| **WebShop** | (1) Query reformulation failures (2) Wrong attribute matching (3) Option selection errors | GOOD: Shopping patterns repeat; memory of "check size before color" type insights |
| **GAIA** | (1) Multi-step planning failures (2) Tool chaining errors (3) Wrong information retrieval | GOOD: Diverse but failure patterns cluster by difficulty level |
| **ToolQA** | (1) Wrong tool selection (2) Compositional tool use failures (3) Parameter errors | GOOD: Systematic tool selection errors that memory could fix |

### Failure Pattern Categories (from AgentDebug taxonomy)

1. **Memory Errors**: Forgetting prior observations, losing track of state
2. **Reflection Errors**: Incorrect self-assessment, missing root cause
3. **Planning Errors**: Wrong decomposition, infeasible plans
4. **Action Errors**: Wrong tool/API selection, wrong parameters, hallucinated actions
5. **System Errors**: Environment issues, timeout, format errors

---

## TOP 5 RECOMMENDATIONS FOR React_FM

### Selection Criteria Applied:
1. Easy to deploy
2. Widely used (community acceptance)
3. High failure recurrence (same error types repeat)
4. Involves tool calls that can fail
5. Used by related memory papers (fair comparison)

---

### **#1: HotpotQA (with search tools)** -- STRONGLY RECOMMENDED

- **Deployment**: EASY (pip install, sample 100-500 questions, just need LLM API)
- **Community**: Used by Reflexion, ExpeL, LATS, AgentDebug, Agent KB, and dozens more
- **Failure recurrence**: HIGH -- search retrieval failures and multi-hop reasoning breaks are systematic
- **Tool calls**: Wikipedia search API that frequently returns irrelevant results
- **Comparison baseline**: Direct comparison with Reflexion (39%), ExpeL (40%), LATS
- **CAVEAT**: Data contamination concerns (published 2018); use hard subset or distractor setting
- **Sample size**: 100 questions is standard (same as Reflexion/ExpeL)

### **#2: ALFWorld** -- STRONGLY RECOMMENDED

- **Deployment**: MEDIUM (pip install, some memory leak issues but manageable)
- **Community**: Used by Reflexion, ExpeL, LATS, MINT, AgentDebug
- **Failure recurrence**: VERY HIGH -- only 6 task types, same errors repeat across episodes
- **Tool calls**: Text-based actions that frequently fail (wrong object, wrong location)
- **Comparison baseline**: Reflexion (77%), ExpeL (59%), ReAct (baseline ~50%)
- **Why ideal for React_FM**: Repetitive task structure means memory of past failures is maximally useful; if your FM correctly recalls "microwave is in kitchen, not bedroom" it directly prevents the same failure
- **Sample size**: 134 environments is the standard full eval

### **#3: tau-bench** -- HIGHLY RECOMMENDED

- **Deployment**: EASY (pip install from GitHub, just needs LLM API)
- **Community**: Industry standard (used by Anthropic, Sierra); rapidly growing
- **Failure recurrence**: VERY HIGH -- policy violations and API errors repeat; pass^8 metric explicitly measures consistency
- **Tool calls**: Domain-specific APIs (retail/airline) that fail on policy violations
- **Comparison baseline**: GPT-4o < 50%, Claude 3.5 Sonnet SOTA; no memory papers yet = opportunity to be first
- **Why ideal for React_FM**: Consistency metric (pass^k) is a natural fit for memory -- if your system remembers "refund policy requires order within 30 days" it should improve pass^k dramatically

### **#4: GAIA** -- RECOMMENDED

- **Deployment**: EASY (HuggingFace dataset, Inspect AI eval framework)
- **Community**: Major leaderboard (HAL Princeton); used by AgentDebug, Agent KB
- **Failure recurrence**: HIGH -- failures cluster by difficulty level; Level 1-2 have systematic patterns
- **Tool calls**: Web browsing, file handling, code execution -- diverse tool failures
- **Comparison baseline**: AgentDebug (+26% with debugging), Agent KB (+16.28pp)
- **Why good for React_FM**: Multi-step tasks where memory of "how to approach Level 2 questions" could help; but tasks are more diverse so memory transfer is harder
- **Sample size**: 165 validation questions

### **#5: WebShop** -- RECOMMENDED

- **Deployment**: MEDIUM (needs ~16GB RAM for product database)
- **Community**: Used by Reflexion, ExpeL, LATS, AgentDebug
- **Failure recurrence**: HIGH -- shopping patterns repeat, attribute matching fails systematically
- **Tool calls**: Web search + navigation actions
- **Comparison baseline**: Reflexion (ReAct+Reflexion did NOT improve on WebShop = interesting negative result), ExpeL (37-38%), LATS (75.9 score)
- **Why good for React_FM**: Interesting because Reflexion FAILED here -- if React_FM can show improvement where Reflexion couldn't, that's a strong result
- **CAVEAT**: Heavier deployment than #1-4; consider only if you need 5 benchmarks

---

## ALTERNATIVE/HONORABLE MENTIONS

| Benchmark | Why Consider | Why Not Top 5 |
|-----------|-------------|---------------|
| **ToolQA** | Pure tool-use QA, easy deploy, systematic failures | Less community adoption; no memory paper baselines |
| **API-Bank** | Clean plan/retrieve/call decomposition | Dated (2023); smaller community |
| **MINT** | Multi-turn focus matches memory use case | Composite of other benchmarks |
| **AppWorld** | Rich, ACL Best Paper, 457 APIs | Newer, fewer baseline comparisons |
| **BFCL** | De facto function calling standard | Tests function calling, not full agent loops |
| **SWE-bench** | Very popular, Agent KB uses it | Hard deployment, not tool-use focused |
| **InterCode** | Docker-based, structured RL setup | Niche; code-focused |

---

## SUGGESTED EVALUATION PLAN FOR React_FM

### Minimum Viable Evaluation (3 benchmarks):
1. **HotpotQA** (EASY, QA+search, all memory papers use it)
2. **ALFWorld** (MEDIUM, embodied, highest failure recurrence)
3. **tau-bench** (EASY, tool+policy, novel contribution opportunity)

### Full Evaluation (5 benchmarks):
1-3 above, plus:
4. **GAIA** (EASY, general assistant, strong baselines)
5. **WebShop** (MEDIUM, web shopping, Reflexion failed here)

### Metrics to Report:
- **Success rate** (standard across all)
- **Pass^k** (tau-bench consistency metric -- novel for memory papers)
- **Progress rate** (AgentBoard-style partial completion)
- **Improvement over ReAct baseline** (standard)
- **Comparison with Reflexion and ExpeL** (direct baselines)
- **Token/cost efficiency** (increasingly expected)

---

## SOURCES

### Surveys
- [Evaluation and Benchmarking of LLM Agents: A Survey (KDD 2025)](https://arxiv.org/abs/2507.21504)
- [Survey on Evaluation of LLM-based Agents (2025)](https://arxiv.org/abs/2503.16416)

### Benchmarks
- [AgentBench (ICLR 2024)](https://github.com/THUDM/AgentBench)
- [AgentBoard (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/877b40688e330a0e2a3fc24084208dfa-Paper-Datasets_and_Benchmarks_Track.pdf)
- [GAIA (ICLR 2024)](https://arxiv.org/abs/2311.12983)
- [tau-bench (2024)](https://github.com/sierra-research/tau-bench)
- [AppWorld (ACL 2024 Best Resource)](https://github.com/StonyBrookNLP/appworld)
- [API-Bank (EMNLP 2023)](https://arxiv.org/abs/2304.08244)
- [ToolQA (NeurIPS 2023)](https://github.com/night-chen/ToolQA)
- [InterCode (NeurIPS 2023)](https://github.com/princeton-nlp/intercode)
- [MINT (ICLR 2024)](https://github.com/xingyaoww/mint-bench)
- [HotpotQA](https://hotpotqa.github.io/)
- [ALFWorld (ICLR 2021)](https://github.com/alfworld/alfworld)
- [WebShop (NeurIPS 2022)](https://webshop-pnlp.github.io/)
- [ToolBench (ICLR 2024)](https://github.com/OpenBMB/ToolBench)
- [StableToolBench (ACL Findings 2024)](https://github.com/THUNLP-MT/StableToolBench)
- [BFCL (ICML 2025)](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [GTA (NeurIPS 2024)](https://github.com/open-compass/GTA)
- [SWE-bench (ICLR 2024)](https://github.com/SWE-bench/SWE-bench)
- [WebArena (ICLR 2024)](https://webarena.dev/)
- [MemoryAgentBench (ICLR 2026)](https://github.com/HUST-AI-HYZ/MemoryAgentBench)

### Key Agent/Memory Papers
- [Reflexion (NeurIPS 2023)](https://github.com/noahshinn/reflexion)
- [ExpeL (AAAI 2024)](https://github.com/LeapLabTHU/ExpeL)
- [LATS (ICML 2024)](https://github.com/lapisrocks/LanguageAgentTreeSearch)
- [AgentDebug (2025)](https://github.com/ulab-uiuc/AgentDebug)
- [Agent KB (2025)](https://arxiv.org/abs/2507.06229)
- [Voyager (TMLR 2023)](https://github.com/MineDojo/Voyager)
- [AgentArch (2025)](https://arxiv.org/abs/2509.10769)
- [HotpotQA Deprecation Warning](https://qipeng.me/blog/stop-using-hotpotqa/)

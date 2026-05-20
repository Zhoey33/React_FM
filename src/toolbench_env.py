"""ToolBench (StableToolBench) environment wrapper.

Loads solvable queries, provides tool definitions as function schemas,
executes API calls via local cache lookup with LLM fallback.
"""

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_STB_ROOT = Path(__file__).parent.parent / "data" / "StableToolBench"
_TOOLS_DIR = _STB_ROOT / "server" / "tools"
_CACHE_DIR = _STB_ROOT / "server" / "tool_response_cache"
_QUERIES_DIR = _STB_ROOT / "solvable_queries" / "test_instruction"


def _standardize(name: str) -> str:
    """Standardize tool/api name (same logic as StableToolBench)."""
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_').lower()


def _load_tool_definitions(api_list: list[dict]) -> list[dict]:
    """Load full tool definitions for the APIs in a query."""
    tools = []
    seen = set()
    for api_info in api_list:
        cat = api_info["category_name"]
        tool_name = api_info["tool_name"]
        api_name = api_info["api_name"]
        key = f"{cat}/{tool_name}/{api_name}"
        if key in seen:
            continue
        seen.add(key)

        # Load tool JSON
        std_tool = _standardize(tool_name)
        tool_path = _TOOLS_DIR / cat / f"{std_tool}.json"
        if not tool_path.exists():
            # Try with category suffix
            for f in (_TOOLS_DIR / cat).glob("*.json"):
                if _standardize(f.stem) == std_tool:
                    tool_path = f
                    break

        if not tool_path.exists():
            logger.warning(f"Tool definition not found: {tool_path}")
            continue

        with open(tool_path) as f:
            tool_data = json.load(f)

        # Find the specific API
        for api in tool_data.get("api_list", []):
            if api["name"] == api_name or _standardize(api["name"]) == _standardize(api_name):
                func_name = f"{_standardize(api_name)}_for_{_standardize(tool_name)}"
                params = {}
                required = []
                for p in api.get("required_parameters", []):
                    params[p["name"]] = {
                        "type": p.get("type", "string").lower(),
                        "description": p.get("description", ""),
                    }
                    required.append(p["name"])
                for p in api.get("optional_parameters", []):
                    params[p["name"]] = {
                        "type": p.get("type", "string").lower(),
                        "description": p.get("description", ""),
                    }
                tools.append({
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "description": f"{api.get('description', '')} (from {tool_name})",
                        "parameters": {
                            "type": "object",
                            "properties": params,
                            "required": required,
                        },
                    },
                    "_meta": {
                        "category": cat,
                        "tool_name": tool_name,
                        "api_name": api_name,
                        "std_tool": std_tool,
                        "std_api": _standardize(api_name),
                    },
                })
                break
    # Add Finish tool
    tools.append({
        "type": "function",
        "function": {
            "name": "Finish",
            "description": "Submit your final answer to the user's question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "return_type": {
                        "type": "string",
                        "description": "give_answer or give_up_and_restart",
                    },
                    "final_answer": {
                        "type": "string",
                        "description": "The final answer to the user's question.",
                    },
                },
                "required": ["return_type", "final_answer"],
            },
        },
        "_meta": None,
    })
    return tools


def _lookup_cache(category: str, tool_name: str, api_name: str, tool_input: dict) -> dict | None:
    """Look up cached API response."""
    std_cat = category.replace(" ", "_").replace(",", "_").replace("/", "_")
    std_cat = re.sub(r'_+', '_', std_cat)
    std_tool = _standardize(tool_name) + f"_for_{std_cat}"
    std_api = _standardize(api_name)

    cache_path = _CACHE_DIR / std_cat / std_tool / f"{std_api}.json"
    if not cache_path.exists():
        # Try flat cache
        flat_path = _CACHE_DIR / f"{std_api}.json"
        if flat_path.exists():
            cache_path = flat_path
        else:
            return None

    try:
        with open(cache_path) as f:
            cache = json.load(f)
        key = str(tool_input)
        if key in cache:
            return cache[key]
    except Exception as e:
        logger.warning(f"Cache lookup error: {e}")
    return None


class ToolBenchEnv:
    """Environment wrapper for StableToolBench solvable queries."""

    def __init__(
        self,
        subsets: list[str] | None = None,
        max_queries: int | None = None,
        fallback_llm=None,
    ):
        self.subsets = subsets or [
            "G1_instruction", "G1_category", "G1_tool",
            "G2_instruction", "G2_category", "G3_instruction",
        ]
        self.max_queries = max_queries
        self.fallback_llm = fallback_llm  # LLMClient for cache miss simulation
        self._queries = []
        self._idx = 0
        self._current = None
        self._tools = []
        self._tool_meta = {}  # func_name -> _meta dict
        self._history = []

    def setup(self):
        """Load all solvable queries."""
        all_queries = []
        for subset in self.subsets:
            path = _QUERIES_DIR / f"{subset}.json"
            if not path.exists():
                logger.warning(f"Query file not found: {path}")
                continue
            with open(path) as f:
                queries = json.load(f)
            for q in queries:
                q["_subset"] = subset
            all_queries.extend(queries)
            logger.info(f"Loaded {len(queries)} queries from {subset}")

        if self.max_queries and len(all_queries) > self.max_queries:
            all_queries = all_queries[:self.max_queries]

        self._queries = all_queries
        self._idx = 0
        logger.info(f"ToolBench: {len(self._queries)} total solvable queries")

    def reset(self) -> tuple[str, str, dict]:
        """Reset to next query. Returns (observation, task_type, info)."""
        if self._idx >= len(self._queries):
            raise StopIteration("No more queries")
        self._current = self._queries[self._idx]
        self._idx += 1
        self._history = []

        # Load tool definitions
        self._tools = _load_tool_definitions(self._current["api_list"])
        self._tool_meta = {}
        for t in self._tools:
            name = t["function"]["name"]
            self._tool_meta[name] = t.get("_meta")

        query = self._current["query"]
        subset = self._current.get("_subset", "unknown")
        query_id = self._current.get("query_id", self._idx)

        # Build tool descriptions for the prompt
        tool_descs = []
        for t in self._tools:
            func = t["function"]
            if func["name"] == "Finish":
                continue
            params = func["parameters"].get("properties", {})
            param_str = ", ".join(f"{k}: {v.get('type', 'str')}" for k, v in params.items())
            tool_descs.append(f"- {func['name']}({param_str}): {func['description']}")

        obs = f"Task: {query}\n\nAvailable APIs:\n" + "\n".join(tool_descs)
        obs += "\n- Finish(return_type, final_answer): Submit your final answer."

        info = {
            "query": query,
            "query_id": query_id,
            "subset": subset,
            "relevant_apis": self._current.get("relevant APIs", []),
            "tools": self._tools,
        }
        return obs, subset, info

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Execute a function call action.

        action format: "func_name({json_params})" or "Finish(give_answer, answer text)"
        Returns (observation, reward, done, info).
        """
        action = action.strip()

        # Parse function call
        func_name, params = self._parse_action(action)

        if not func_name:
            return "Invalid action format. Use: function_name({\"param\": \"value\"})", 0.0, False, {}

        # Handle Finish
        if func_name == "Finish":
            answer = params.get("final_answer", "")
            return_type = params.get("return_type", "give_answer")
            done = True
            success = return_type == "give_answer" and len(answer) > 0
            return answer, float(success), done, {
                "final_answer": answer,
                "return_type": return_type,
            }

        # Look up tool meta
        meta = self._tool_meta.get(func_name)
        if meta is None:
            return f"Unknown function: {func_name}. Check available APIs.", 0.0, False, {}

        # Try cache lookup
        cached = _lookup_cache(
            meta["category"], meta["tool_name"], meta["api_name"], params,
        )
        if cached is not None:
            obs = self._format_response(cached)
            self._history.append((func_name, params, obs))
            return obs, 0.0, False, {"cache_hit": True}

        # Fallback: use LLM to simulate response
        if self.fallback_llm is not None:
            obs = self._simulate_response(func_name, params, meta)
            self._history.append((func_name, params, obs))
            return obs, 0.0, False, {"cache_hit": False}

        obs = f"API call to {func_name} failed: no cached response available."
        self._history.append((func_name, params, obs))
        return obs, 0.0, False, {"cache_hit": False, "error": "no_cache"}

    def _parse_action(self, action: str) -> tuple[str | None, dict]:
        """Parse action string into (func_name, params)."""
        # Extract function name
        m = re.match(r'(\w+)\(', action)
        if m:
            func_name = m.group(1)
            # Find matching closing paren (outermost)
            start = m.end() - 1  # position of '('
            depth = 0
            end = -1
            for i in range(start, len(action)):
                if action[i] == '(':
                    depth += 1
                elif action[i] == ')':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end > start:
                inner = action[start + 1:end]
                try:
                    params = json.loads(inner)
                    return func_name, params
                except json.JSONDecodeError:
                    # Try as Finish(give_answer, text)
                    if func_name == "Finish":
                        parts = inner.split(",", 1)
                        return "Finish", {
                            "return_type": parts[0].strip().strip('"\''),
                            "final_answer": parts[1].strip().strip('"\'') if len(parts) > 1 else "",
                        }
                    return func_name, {}

        # Format 2: Action: func_name\nAction Input: {"key": "value"}
        lines = action.split("\n")
        func_name = None
        params_str = ""
        for line in lines:
            line = line.strip()
            if line.lower().startswith("action:"):
                func_name = line[len("action:"):].strip()
            elif line.lower().startswith("action input:"):
                params_str = line[len("action input:"):].strip()
        if func_name:
            try:
                params = json.loads(params_str) if params_str else {}
            except json.JSONDecodeError:
                params = {}
            return func_name, params

        return None, {}

    def _format_response(self, cached: dict) -> str:
        """Format cached API response."""
        error = cached.get("error", "")
        response = cached.get("response", "")
        if error:
            return f"API Error: {error}"
        resp_str = str(response)
        if len(resp_str) > 1024:
            resp_str = resp_str[:1024] + "... [truncated]"
        return resp_str

    def _simulate_response(self, func_name: str, params: dict, meta: dict) -> str:
        """Use fallback LLM to simulate API response."""
        prompt = (
            f"Simulate the API response for:\n"
            f"API: {meta['api_name']} (from {meta['tool_name']})\n"
            f"Input: {json.dumps(params)}\n\n"
            f"Return a realistic JSON response with 'error' and 'response' fields."
        )
        try:
            resp = self.fallback_llm.complete_text(prompt, label="api_sim")
            return resp[:1024]
        except Exception as e:
            return f"API simulation failed: {e}"

    def skip(self):
        self._idx += 1

    def close(self):
        pass

    @property
    def total_episodes(self) -> int:
        return len(self._queries)

    @property
    def current_query(self) -> dict | None:
        return self._current

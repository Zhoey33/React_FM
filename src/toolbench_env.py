"""ToolBench (StableToolBench) official-server environment adapter.

Loads solvable queries, provides tool definitions as function schemas,
and executes API calls through the StableToolBench virtual server.
"""

import json
import logging
import os
import re
from pathlib import Path

import requests

from src.stabletoolbench_official import (
    official_function_schemas,
    official_tool_metadata,
    parse_action_call,
)

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
    """Build official StableToolBench function schemas for a query."""
    return official_function_schemas(api_list, include_finish=True)


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
        api_mode: str = "server",
        service_url: str | None = None,
        toolbench_key: str | None = None,
        max_observation_length: int = 1024,
        observ_compress_method: str = "truncate",
    ):
        if api_mode not in {"server", "cache"}:
            raise ValueError(f"Unsupported ToolBench api_mode: {api_mode}")
        self.subsets = subsets or [
            "G1_instruction", "G1_category", "G1_tool",
            "G2_instruction", "G2_category", "G3_instruction",
        ]
        self.max_queries = max_queries
        self.fallback_llm = fallback_llm  # LLMClient for cache miss simulation
        self.api_mode = api_mode
        self.service_url = service_url or os.environ.get("SERVICE_URL", "http://localhost:8080/virtual")
        self.toolbench_key = toolbench_key if toolbench_key is not None else os.environ.get("TOOLBENCH_KEY", "")
        self.max_observation_length = max_observation_length
        self.observ_compress_method = observ_compress_method
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
        self._tool_meta = official_tool_metadata(self._current["api_list"])
        self._tool_meta["Finish"] = None

        query = self._current["query"]
        subset = self._current.get("_subset", "unknown")
        query_id = self._current.get("query_id", self._idx)

        # Build tool descriptions for the prompt
        tool_descs = []
        for t in self._tools:
            if t["name"] == "Finish":
                continue
            params = t["parameters"].get("properties", {})
            param_str = ", ".join(f"{k}: {v.get('type', 'str')}" for k, v in params.items())
            tool_descs.append(f"- {t['name']}({param_str}): {t['description']}")

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

        if self.api_mode == "server":
            obs, status_code = self._call_official_server(meta, params)
            self._history.append((func_name, params, obs))
            return obs, 0.0, False, {
                "official_server": True,
                "status_code": status_code,
                "tool_meta": meta,
            }

        # Debug-only cache lookup.
        cached = _lookup_cache(
            meta["category"], meta["original_tool_name"], meta["original_api_name"], params,
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
        func_name, params, _ = parse_action_call(action)
        return func_name, params

    def _call_official_server(self, meta: dict, params: dict) -> tuple[str, int]:
        """Call the StableToolBench official virtual API endpoint."""
        payload = {
            "category": meta["category"],
            "tool_name": meta["tool_name"],
            "api_name": meta["api_name"],
            "tool_input": json.dumps(params, ensure_ascii=False),
            "strip": self.observ_compress_method,
            "toolbench_key": self.toolbench_key,
        }
        headers = {"toolbench_key": self.toolbench_key}
        timeout = None if self.service_url.endswith("virtual") else 15
        try:
            response = requests.post(self.service_url, json=payload, headers=headers, timeout=timeout)
        except requests.exceptions.Timeout:
            return json.dumps({"error": "Timeout error...", "response": ""}), 5
        except requests.RequestException as exc:
            return json.dumps({"error": f"request failed: {exc}", "response": ""}), 12

        if response.status_code != 200:
            return json.dumps(
                {"error": f"request invalid, data error. status_code={response.status_code}", "response": ""},
                ensure_ascii=False,
            ), 12
        try:
            data = response.json()
        except ValueError:
            return json.dumps({"error": "request invalid, data error", "response": ""}), 12

        obs = json.dumps(data, ensure_ascii=False)
        if len(obs) > self.max_observation_length:
            obs = obs[:self.max_observation_length] + "..."
        return obs, self._official_status_code(data)

    @staticmethod
    def _official_status_code(response: dict) -> int:
        """Map StableToolBench virtual-server errors to official status codes."""
        error = response.get("error", "")
        if error == "API not working error...":
            return 6
        if error == "Unauthorized error...":
            return 7
        if error == "Unsubscribed error...":
            return 8
        if error == "Too many requests error...":
            return 9
        if error == "Rate limit per minute error...":
            return 10
        if error == "Message error...":
            return 11
        return 0

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

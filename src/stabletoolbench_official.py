"""StableToolBench official-format helpers for schema, raw answers, and eval."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

STABLETOOLBENCH_ROOT = Path(__file__).parent.parent / "data" / "StableToolBench"
TOOLEVAL_DIR = STABLETOOLBENCH_ROOT / "toolbench" / "tooleval"


def standardize_category(category: str) -> str:
    """Match StableToolBench category normalization."""
    save_category = category.replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in save_category or "," in save_category:
        save_category = save_category.replace(" ", "_").replace(",", "_")
    save_category = save_category.replace("__", "_")
    return save_category


def standardize(name: str) -> str:
    """Match StableToolBench tool/API/parameter normalization."""
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    name = res.sub("_", name)
    name = re.sub(r"(_)\1+", "_", name).lower()
    while name.startswith("_"):
        name = name[1:]
    while name.endswith("_"):
        name = name[:-1]
    if name and name[0].isdigit():
        name = "get_" + name
    return name


def change_name(name: str) -> str:
    """Match StableToolBench reserved-word renaming."""
    if name in {"from", "class", "return", "false", "true", "id", "and"}:
        return f"is_{name}"
    return name


def official_function_schemas(api_list: list[dict[str, Any]], include_finish: bool = True) -> list[dict[str, Any]]:
    """Build StableToolBench-compatible unwrapped function schemas."""
    functions = []
    seen = set()
    for api_json in api_list:
        category = api_json.get("category_name", "")
        tool_name = api_json.get("tool_name", "")
        api_name = api_json.get("api_name", "")
        standard_tool_name = standardize(tool_name)
        pure_api_name = change_name(standardize(api_name))
        function_name = f"{pure_api_name}_for_{standard_tool_name}"[-64:]
        key = (category, standard_tool_name, pure_api_name, function_name)
        if key in seen:
            continue
        seen.add(key)

        schema = {
            "name": function_name,
            "description": f'This is the subfunction for tool "{standard_tool_name}", you can use this tool.',
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "optional": [],
            },
        }
        description = (api_json.get("api_description") or "").strip()
        if description:
            truncated = description.replace(api_name, function_name)[:256]
            schema["description"] += f'The description of this function is: "{truncated}"'

        for field, target in (("required_parameters", "required"), ("optional_parameters", "optional")):
            for para in api_json.get(field, []) or []:
                param_name = change_name(standardize(str(para.get("name", ""))))
                if not param_name:
                    continue
                param_type = {
                    "NUMBER": "integer",
                    "STRING": "string",
                    "BOOLEAN": "boolean",
                }.get(str(para.get("type", "")).upper(), "string")
                prompt = {
                    "type": param_type,
                    "description": str(para.get("description", ""))[:256],
                }
                default_value = para.get("default", "")
                if len(str(default_value)) != 0:
                    prompt["example_value"] = default_value
                schema["parameters"]["properties"][param_name] = prompt
                schema["parameters"][target].append(param_name)

        functions.append(schema)

    if include_finish:
        functions.append(finish_function_schema())
    return functions


def official_tool_metadata(api_list: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Return metadata needed to call the official virtual API server."""
    metadata = {}
    for api_json in api_list:
        standard_tool_name = standardize(api_json.get("tool_name", ""))
        pure_api_name = change_name(standardize(api_json.get("api_name", "")))
        function_name = f"{pure_api_name}_for_{standard_tool_name}"[-64:]
        metadata[function_name] = {
            "category": api_json.get("category_name", ""),
            "tool_name": standard_tool_name,
            "api_name": pure_api_name,
            "original_tool_name": api_json.get("tool_name", ""),
            "original_api_name": api_json.get("api_name", ""),
        }
    return metadata


def finish_function_schema() -> dict[str, Any]:
    """Return the official StableToolBench Finish function schema."""
    return {
        "name": "Finish",
        "description": (
            "If you believe that you have obtained a result that can answer the task, "
            "please call this function to provide the final answer. Alternatively, if you "
            "recognize that you are unable to proceed with the task in the current state, "
            "call this function to restart. Remember: you must ALWAYS call this function "
            "at the end of your attempt, and the only part that will be shown to the user "
            "is the final answer, so it should contain sufficient information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "return_type": {
                    "type": "string",
                    "enum": ["give_answer", "give_up_and_restart"],
                },
                "final_answer": {
                    "type": "string",
                    "description": (
                        'The final answer you want to give the user. You should have this field '
                        'if "return_type"=="give_answer"'
                    ),
                },
            },
            "required": ["return_type"],
        },
    }


def parse_action_call(action: str) -> tuple[str | None, dict[str, Any], str]:
    """Parse a ReAct-style function call into name, params, and JSON argument text."""
    action = action.strip()
    match = re.match(r"(\w+)\(", action)
    if match:
        func_name = match.group(1)
        start = match.end() - 1
        depth = 0
        end = -1
        for idx in range(start, len(action)):
            if action[idx] == "(":
                depth += 1
            elif action[idx] == ")":
                depth -= 1
                if depth == 0:
                    end = idx
                    break
        if end > start:
            inner = action[start + 1:end]
            try:
                params = json.loads(inner)
                return func_name, params, json.dumps(params, ensure_ascii=False)
            except json.JSONDecodeError:
                if func_name == "Finish":
                    parts = inner.split(",", 1)
                    params = {
                        "return_type": parts[0].strip().strip('"\''),
                        "final_answer": parts[1].strip().strip('"\'') if len(parts) > 1 else "",
                    }
                    return func_name, params, json.dumps(params, ensure_ascii=False)
                return func_name, {}, "{}"

    lines = action.splitlines()
    func_name = None
    params_text = ""
    for line in lines:
        line = line.strip()
        if line.lower().startswith("action:"):
            func_name = line[len("action:"):].strip()
        elif line.lower().startswith("action input:"):
            params_text = line[len("action input:"):].strip()
    if func_name:
        try:
            params = json.loads(params_text) if params_text else {}
        except json.JSONDecodeError:
            params = {}
        return func_name, params, json.dumps(params, ensure_ascii=False)
    return None, {}, "{}"


def build_raw_answer_record(
    query: str,
    functions: list[dict[str, Any]],
    method: str,
    steps: list[dict[str, Any]],
    final_answer: str,
    total_tokens: int = 0,
) -> dict[str, Any]:
    """Build a StableToolBench raw answer JSON record from a local episode."""
    valid_data = bool(final_answer)
    train_messages = [
        {"role": "system", "content": _official_system_message(functions)},
        {"role": "user", "content": f"\n{query}\nBegin!\n"},
    ]
    chain = []
    depth = 1

    for step in steps:
        thought = step.get("thought", "")
        action = step.get("action", "")
        observation = step.get("observation", "")
        if thought:
            train_messages.append({"role": "assistant", "content": thought})
            chain.append(_chain_node("Thought", thought, depth))
            depth += 1

        func_name, _, arguments = parse_action_call(action)
        if not func_name:
            continue
        train_messages.append({"role": "assistant", "function_call": {"name": func_name, "arguments": arguments}})
        chain.append(_chain_node("Action", func_name, depth))
        depth += 1

        observation_code = 3 if func_name == "Finish" and final_answer else 0
        action_input_node = _chain_node("Action Input", arguments, depth)
        action_input_node["observation"] = "" if func_name == "Finish" else observation
        action_input_node["observation_code"] = observation_code
        action_input_node["io_state"] = {}
        chain.append(action_input_node)
        depth += 1

        if func_name != "Finish":
            train_messages.append({"role": "function", "name": func_name, "content": observation})

    return {
        "win": valid_data,
        "try_count": 1,
        "trys": [{"chain": chain, "win": valid_data}],
        "compare_candidates": [chain] if valid_data else [],
        "forward_args": {"method": method, "answer": 1},
        "answer_generation": {
            "valid_data": valid_data,
            "final_answer": final_answer,
            "function": functions,
            "query_count": len(steps),
            "total_tokens": total_tokens,
            "train_messages": [train_messages] if valid_data else [],
            "chain": chain,
            "query": query,
        },
    }


def write_raw_answer(
    record: dict[str, Any],
    answer_dir: Path | str,
    query_id: str | int,
    method: str,
) -> Path:
    """Write a raw answer record with the official query-method filename."""
    path = Path(answer_dir)
    path.mkdir(parents=True, exist_ok=True)
    output_path = path / f"{query_id}_{method}.json"
    output_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def run_official_converter(answer_dir: Path | str, method: str, output: Path | str) -> None:
    """Run StableToolBench's official convert_to_answer_format.py."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(STABLETOOLBENCH_ROOT)
    subprocess.run(
        [
            sys.executable,
            "convert_to_answer_format.py",
            "--answer_dir",
            str(Path(answer_dir).resolve()),
            "--method",
            method,
            "--output",
            str(output_path.resolve()),
        ],
        cwd=TOOLEVAL_DIR,
        env=env,
        check=True,
    )


def run_official_pass_rate_eval(
    converted_answer_path: Path | str,
    save_path: Path | str,
    candidate_model: str,
    test_sets: list[str],
    evaluate_times: int = 3,
    max_eval_threads: int = 35,
    reference_model: str | None = None,
) -> None:
    """Run StableToolBench's official eval_pass_rate.py for SoPR."""
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(STABLETOOLBENCH_ROOT)
    subprocess.run(
        [
            sys.executable,
            "eval_pass_rate.py",
            "--converted_answer_path",
            str(Path(converted_answer_path).resolve()),
            "--save_path",
            str(save_dir.resolve()),
            "--reference_model",
            reference_model or candidate_model,
            "--test_ids",
            str((STABLETOOLBENCH_ROOT / "solvable_queries" / "test_query_ids").resolve()),
            "--max_eval_threads",
            str(max_eval_threads),
            "--evaluate_times",
            str(evaluate_times),
            "--test_set",
            *test_sets,
        ],
        cwd=TOOLEVAL_DIR,
        env=env,
        check=True,
    )


def _official_system_message(functions: list[dict[str, Any]]) -> str:
    return (
        "You are AutoGPT, you can use many tools(functions) to do the following task.\n"
        "First I will give you the task description, and your task start.\n"
        "At each step, you need to give your thought to analyze the status now and what to do next, "
        "with a function call to actually excute your step.\n"
        "After the call, you will get the call result, and you are now in a new state.\n"
        "Then you will analyze your status now, then decide what to do next...\n"
        "After many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.\n"
        "Remember: \n"
        "1.the state change is irreversible, you can't go back to one of the former state, "
        'if you want to restart the task, say "I give up and restart".\n'
        "2.All the thought is short, at most in 5 sentence.\n"
        "3.You can do more then one trys, so if your plan is to continusly try some conditions, "
        "you can do one of the conditions per try.\n"
        "Let's Begin!\n"
        "Task description: You should use functions to help handle the real time user querys. "
        'Remember to ALWAYS call "Finish" function at the end of the task. And the final answer '
        "should contain enough information to show to the user.\n"
        f"Specifically, you have access to the following functions: {functions}"
    )


def _chain_node(node_type: str, description: str, depth: int) -> dict[str, Any]:
    return {
        "is_terminal": False,
        "pruned": False,
        "finished": node_type == "Action Input",
        "depth": depth,
        "node_type": node_type,
        "description": description,
        "Elo": 1000.0,
        "child_count": 1,
        "expand_num": 0,
    }

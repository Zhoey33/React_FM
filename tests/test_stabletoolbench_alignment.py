"""Tests for StableToolBench official schema, server calls, and raw answers."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.stabletoolbench_official import (
    build_raw_answer_record,
    official_function_schemas,
    run_official_converter,
)
from src.toolbench_env import ToolBenchEnv


class StableToolBenchAlignmentTests(unittest.TestCase):
    def test_schema_uses_official_names_and_unwrapped_functions(self):
        api_list = [
            {
                "category_name": "Data",
                "tool_name": "Example Tool",
                "api_name": "class",
                "api_description": "Return an item by id.",
                "required_parameters": [
                    {"name": "id", "type": "NUMBER", "description": "identifier", "default": 7}
                ],
                "optional_parameters": [
                    {"name": "include details", "type": "BOOLEAN", "description": "details", "default": ""}
                ],
            }
        ]

        functions = official_function_schemas(api_list, include_finish=True)

        self.assertEqual(functions[0]["name"], "is_class_for_example_tool")
        self.assertNotIn("type", functions[0])
        self.assertEqual(functions[0]["parameters"]["properties"]["is_id"]["type"], "integer")
        self.assertEqual(functions[0]["parameters"]["properties"]["include_details"]["type"], "boolean")
        self.assertEqual(functions[0]["parameters"]["required"], ["is_id"])
        self.assertEqual(functions[-1]["name"], "Finish")

    @patch("src.toolbench_env.requests.post")
    def test_env_calls_official_virtual_server_payload(self, post):
        post.return_value = Mock(status_code=200)
        post.return_value.json.return_value = {"error": "", "response": {"ok": True}}
        env = ToolBenchEnv(
            subsets=["G1_instruction"],
            max_queries=1,
            api_mode="server",
            service_url="http://localhost:8080/virtual",
            toolbench_key="tb-key",
        )
        env.setup()
        _, _, info = env.reset()
        fn = next(f for f in info["tools"] if f["name"] != "Finish")

        obs, reward, done, step_info = env.step(f'{fn["name"]}({{"example": "value"}})')

        self.assertFalse(done)
        self.assertEqual(reward, 0.0)
        self.assertEqual(json.loads(obs), {"error": "", "response": {"ok": True}})
        _, kwargs = post.call_args
        self.assertEqual(kwargs["json"]["toolbench_key"], "tb-key")
        self.assertEqual(kwargs["json"]["tool_input"], '{"example": "value"}')
        self.assertEqual(kwargs["headers"], {"toolbench_key": "tb-key"})
        self.assertTrue(step_info["official_server"])

    def test_raw_answer_record_converts_with_official_converter(self):
        functions = [
            {
                "name": "lookup_for_demo",
                "description": "Demo lookup",
                "parameters": {"type": "object", "properties": {}, "required": [], "optional": []},
            },
            {
                "name": "Finish",
                "description": "Finish",
                "parameters": {"type": "object", "properties": {}, "required": ["return_type"]},
            },
        ]
        record = build_raw_answer_record(
            query="Find the demo answer.",
            functions=functions,
            method="ReactFM_CoT@1",
            steps=[
                {
                    "thought": "I should look it up.",
                    "action": "lookup_for_demo({})",
                    "observation": '{"error": "", "response": "demo"}',
                },
                {
                    "thought": "I can answer now.",
                    "action": 'Finish({"return_type": "give_answer", "final_answer": "demo"})',
                    "observation": '{"response":"successfully giving the final answer."}',
                },
            ],
            final_answer="demo",
            total_tokens=12,
        )

        with tempfile.TemporaryDirectory() as tmp:
            answer_dir = Path(tmp) / "answer"
            answer_dir.mkdir()
            raw_path = answer_dir / "1_ReactFM_CoT@1.json"
            raw_path.write_text(json.dumps(record), encoding="utf-8")
            converted_path = Path(tmp) / "converted.json"

            run_official_converter(answer_dir, "ReactFM_CoT@1", converted_path)

            converted = json.loads(converted_path.read_text(encoding="utf-8"))
            self.assertIn("1", converted)
            self.assertEqual(converted["1"]["answer"]["method"], "ReactFM_CoT@1")
            self.assertEqual(converted["1"]["answer"]["final_answer"], "demo")


if __name__ == "__main__":
    unittest.main()

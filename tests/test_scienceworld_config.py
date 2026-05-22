"""Tests for ScienceWorld-compatible default configuration values."""

from pathlib import Path

import yaml


def test_default_config_gives_judge_enough_tokens_for_structured_output():
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))

    judge_cfg = config["judge"]
    assert judge_cfg["enable_implicit_failures"] is True
    assert judge_cfg["detector_max_tokens"] == 128
    assert judge_cfg["repair_max_tokens"] == 192
    assert judge_cfg["enable_cache"] is True
    assert judge_cfg["enable_repair_intent_filter"] is True
    assert judge_cfg["implicit_failure_confidence_threshold"] == 0.8
    assert judge_cfg["repair_confidence_threshold"] == 0.7
    assert config["memory"]["enable_intent_gate"] is True


def test_scienceworld_config_uses_small_judge_model_without_changing_agent_or_extractor():
    config = yaml.safe_load(Path("config_scienceworld.yaml").read_text(encoding="utf-8"))

    assert config["llm"]["model"] == "deepseek-ai/DeepSeek-V3.2"
    assert config["extractor"]["model"] == "deepseek-ai/DeepSeek-V3.2"
    assert config["judge"]["model"] == "Qwen/Qwen3.5-9B"
    assert config["judge"]["detector_max_tokens"] == 128
    assert config["judge"]["repair_max_tokens"] == 192
    assert config["judge"]["enable_cache"] is True
    assert config["judge"]["enable_repair_intent_filter"] is True
    assert config["memory"]["enable_intent_gate"] is True

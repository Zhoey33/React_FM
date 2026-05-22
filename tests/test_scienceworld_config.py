"""Tests for ScienceWorld-compatible default configuration values."""

from pathlib import Path

import yaml


def test_default_config_gives_judge_enough_tokens_for_structured_output():
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))

    judge_cfg = config["judge"]
    assert judge_cfg["enable_implicit_failures"] is True
    assert judge_cfg["max_tokens"] >= 512
    assert judge_cfg["implicit_failure_confidence_threshold"] == 0.8
    assert judge_cfg["repair_confidence_threshold"] == 0.7

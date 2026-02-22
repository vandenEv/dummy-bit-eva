"""test_config.py — Validate YAML config for scenario_fail_big_gpu_like."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCENARIO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SCENARIO_ROOT))

from src.utils import load_config

CONFIG_PATH = SCENARIO_ROOT / "configs" / "config.yaml"

REQUIRED_KEYS = [
    "seed",
    "dataset_size",
    "input_dim",
    "num_classes",
    "hidden_dim",
    "num_layers",
    "num_heads",
    "epochs",
    "batch_size",
    "learning_rate",
    "eval_every_n_steps",
    "use_cache",
    "augmentation_repeats",
]


def test_config_loads():
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg, dict)


def test_config_required_keys():
    cfg = load_config(CONFIG_PATH)
    for key in REQUIRED_KEYS:
        assert key in cfg, f"Missing required key: '{key}'"


def test_config_is_intentionally_heavy():
    """Assert the config is heavy — if someone accidentally lightens it,
    this test will catch it and preserve the demo intent."""
    cfg = load_config(CONFIG_PATH)
    assert cfg["hidden_dim"] >= 256, "hidden_dim should be large for this scenario."
    assert cfg["num_layers"] >= 4, "num_layers should be large for this scenario."
    assert (
        cfg["dataset_size"] >= 5000
    ), "dataset_size should be large for this scenario."
    assert cfg["eval_every_n_steps"] == 1, "eval_every_n_steps should be 1 (wasteful)."
    assert cfg["use_cache"] is False, "use_cache should be False (wasteful)."


def test_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config(SCENARIO_ROOT / "configs" / "does_not_exist.yaml")

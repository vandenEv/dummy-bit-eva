"""test_config.py — Validate config for scenario_fail_override_permission_only."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCENARIO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SCENARIO_ROOT))

from src.utils import load_config

CONFIG_PATH = SCENARIO_ROOT / "configs" / "config.yaml"

REQUIRED_KEYS = [
    "seed", "dataset_size", "input_dim", "num_classes",
    "hidden_dim", "num_layers", "num_heads",
    "epochs", "batch_size", "learning_rate",
    "eval_every_n_steps", "use_cache", "allow_override_demo",
]


def test_config_loads():
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg, dict)


def test_config_required_keys():
    cfg = load_config(CONFIG_PATH)
    for key in REQUIRED_KEYS:
        assert key in cfg, f"Missing required key: '{key}'"


def test_override_flag_present():
    """allow_override_demo must be present and set to true for demo narration."""
    cfg = load_config(CONFIG_PATH)
    assert "allow_override_demo" in cfg
    assert cfg["allow_override_demo"] is True, (
        "allow_override_demo should be True in this scenario's config."
    )


def test_config_is_intentionally_heavy():
    """Assert compute parameters remain heavy (preserve demo intent)."""
    cfg = load_config(CONFIG_PATH)
    assert cfg["hidden_dim"] >= 256
    assert cfg["num_layers"] >= 4
    assert cfg["dataset_size"] >= 5000
    assert cfg["eval_every_n_steps"] == 1
    assert cfg["use_cache"] is False


def test_config_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_config(SCENARIO_ROOT / "configs" / "nope.yaml")

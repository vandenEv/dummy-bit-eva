"""test_config.py — Validate that the YAML config loads correctly."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow imports from the scenario root.
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
    "epochs",
    "batch_size",
    "learning_rate",
    "eval_every_n_steps",
]


def test_config_loads():
    """Config file should load without errors."""
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg, dict), "Config should be a dict."


def test_config_required_keys():
    """All required keys should be present in the config."""
    cfg = load_config(CONFIG_PATH)
    for key in REQUIRED_KEYS:
        assert key in cfg, f"Missing required key: '{key}'"


def test_config_value_types():
    """Numeric config values should have sensible types and ranges."""
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg["epochs"], int) and cfg["epochs"] > 0
    assert isinstance(cfg["batch_size"], int) and cfg["batch_size"] > 0
    assert isinstance(cfg["hidden_dim"], int) and cfg["hidden_dim"] > 0
    assert isinstance(cfg["num_layers"], int) and cfg["num_layers"] > 0
    assert isinstance(cfg["dataset_size"], int) and cfg["dataset_size"] > 0
    assert isinstance(cfg["learning_rate"], float) and cfg["learning_rate"] > 0


def test_config_file_not_found():
    """Loading a missing config should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config(SCENARIO_ROOT / "configs" / "does_not_exist.yaml")

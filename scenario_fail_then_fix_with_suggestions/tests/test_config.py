"""test_config.py — Validate BEFORE and AFTER configs for scenario 3."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCENARIO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SCENARIO_ROOT))

from src.utils import load_config

BEFORE_CFG = SCENARIO_ROOT / "configs" / "config_before.yaml"
AFTER_CFG = SCENARIO_ROOT / "configs" / "config_after.yaml"

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
    "use_cache",
    "early_stopping",
    "fast_dev_run",
]


def test_before_config_loads():
    cfg = load_config(BEFORE_CFG)
    assert isinstance(cfg, dict)


def test_after_config_loads():
    cfg = load_config(AFTER_CFG)
    assert isinstance(cfg, dict)


def test_required_keys_before():
    cfg = load_config(BEFORE_CFG)
    for key in REQUIRED_KEYS:
        assert key in cfg, f"Missing key in before config: '{key}'"


def test_required_keys_after():
    cfg = load_config(AFTER_CFG)
    for key in REQUIRED_KEYS:
        assert key in cfg, f"Missing key in after config: '{key}'"


def test_before_is_heavier_than_after():
    """The before config should consume more compute than the after config."""
    before = load_config(BEFORE_CFG)
    after = load_config(AFTER_CFG)

    assert before["epochs"] > after["epochs"], "Before should have more epochs."
    assert (
        before["hidden_dim"] > after["hidden_dim"]
    ), "Before should have larger hidden_dim."
    assert before["num_layers"] > after["num_layers"], "Before should have more layers."
    assert (
        before["dataset_size"] > after["dataset_size"]
    ), "Before should have larger dataset."
    assert (
        before["eval_every_n_steps"] < after["eval_every_n_steps"]
    ), "Before should evaluate more frequently (lower step interval)."


def test_after_has_remediations_enabled():
    after = load_config(AFTER_CFG)
    assert after["early_stopping"] is True, "After config should enable early stopping."
    assert after["use_cache"] is True, "After config should enable caching."
    assert after["fast_dev_run"] is True, "After config should enable fast_dev_run."


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config(SCENARIO_ROOT / "configs" / "nonexistent.yaml")

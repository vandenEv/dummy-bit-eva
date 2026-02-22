"""test_train_step.py — Smoke tests for scenario_fail_override_permission_only."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

SCENARIO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SCENARIO_ROOT))

from src.data import make_loaders
from src.eval import evaluate
from src.model import build_model
from src.utils import load_config, set_seed

CONFIG_PATH = SCENARIO_ROOT / "configs" / "config.yaml"


def _fast_cfg() -> dict:
    cfg = load_config(CONFIG_PATH)
    cfg["epochs"] = 1
    cfg["dataset_size"] = 128
    cfg["batch_size"] = 32
    cfg["hidden_dim"] = 64
    cfg["num_layers"] = 2
    cfg["num_heads"] = 4
    return cfg


def test_model_builds():
    cfg = _fast_cfg()
    model = build_model(cfg)
    assert model is not None


def test_single_forward_pass():
    cfg = _fast_cfg()
    set_seed(cfg["seed"])
    model = build_model(cfg)
    x = torch.randn(cfg["batch_size"], cfg["input_dim"])
    logits = model(x)
    assert logits.shape == (cfg["batch_size"], cfg["num_classes"])


def test_single_training_step():
    cfg = _fast_cfg()
    set_seed(cfg["seed"])
    device = torch.device("cpu")
    train_loader, _ = make_loaders(cfg, seed=cfg["seed"])
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    x, y = next(iter(train_loader))
    optimizer.zero_grad()
    loss = criterion(model(x.to(device)), y.to(device))
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss)
    assert loss.item() < 1e6


def test_evaluate_runs():
    cfg = _fast_cfg()
    set_seed(cfg["seed"])
    device = torch.device("cpu")
    _, val_loader = make_loaders(cfg, seed=cfg["seed"])
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    assert 0.0 <= val_acc <= 1.0
    assert val_loss >= 0.0


def test_allow_override_demo_flag_is_env_driven():
    """ALLOW_OVERRIDE_DEMO should be read from env, not hardcoded."""
    os.environ["ALLOW_OVERRIDE_DEMO"] = "true"
    # Re-import to pick up env change
    import importlib
    import src.train as train_module
    importlib.reload(train_module)
    assert train_module.ALLOW_OVERRIDE_DEMO is True

    os.environ["ALLOW_OVERRIDE_DEMO"] = "false"
    importlib.reload(train_module)
    assert train_module.ALLOW_OVERRIDE_DEMO is False

"""test_train_step.py — Smoke tests for both BEFORE and AFTER states."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

SCENARIO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SCENARIO_ROOT))

# BEFORE imports
from src.data import make_loaders as before_loaders
from src.model import build_model as before_build
from src.utils import load_config, set_seed

# AFTER imports
from src_fixed.data import make_loaders as after_loaders
from src_fixed.eval import evaluate
from src_fixed.model import build_model as after_build

BEFORE_CFG = SCENARIO_ROOT / "configs" / "config_before.yaml"
AFTER_CFG = SCENARIO_ROOT / "configs" / "config_after.yaml"


def _fast_cfg(path: Path) -> dict:
    cfg = load_config(path)
    cfg["epochs"] = 1
    cfg["dataset_size"] = 64
    cfg["batch_size"] = 32
    cfg["hidden_dim"] = 64
    cfg["num_layers"] = 2
    return cfg


# ---------------------------------------------------------------------------
# BEFORE state
# ---------------------------------------------------------------------------


def test_before_model_builds():
    cfg = _fast_cfg(BEFORE_CFG)
    model = before_build(cfg)
    assert model is not None


def test_before_single_training_step():
    cfg = _fast_cfg(BEFORE_CFG)
    set_seed(cfg["seed"])
    device = torch.device("cpu")
    train_loader, _ = before_loaders(cfg, seed=cfg["seed"])
    model = before_build(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    x, y = next(iter(train_loader))
    optimizer.zero_grad()
    logits = model(x.to(device))
    loss = criterion(logits, y.to(device))
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss)
    assert loss.item() < 1e6


# ---------------------------------------------------------------------------
# AFTER (fixed) state
# ---------------------------------------------------------------------------


def test_after_model_builds():
    cfg = _fast_cfg(AFTER_CFG)
    model = after_build(cfg)
    assert model is not None


def test_after_single_training_step():
    cfg = _fast_cfg(AFTER_CFG)
    set_seed(cfg["seed"])
    device = torch.device("cpu")
    train_loader, _ = after_loaders(cfg, seed=cfg["seed"])
    model = after_build(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    x, y = next(iter(train_loader))
    optimizer.zero_grad()
    logits = model(x.to(device))
    loss = criterion(logits, y.to(device))
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss)
    assert loss.item() < 1e6


def test_after_evaluate_runs():
    cfg = _fast_cfg(AFTER_CFG)
    set_seed(cfg["seed"])
    device = torch.device("cpu")
    _, val_loader = after_loaders(cfg, seed=cfg["seed"])
    model = after_build(cfg).to(device)
    criterion = nn.CrossEntropyLoss()

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    assert 0.0 <= val_acc <= 1.0
    assert val_loss >= 0.0


def test_after_model_is_smaller_than_before():
    """AFTER model should have fewer parameters than BEFORE."""
    before_cfg = _fast_cfg(BEFORE_CFG)
    # Restore the realistic before values for this comparison
    before_cfg["hidden_dim"] = 256
    before_cfg["num_layers"] = 6

    after_cfg = _fast_cfg(AFTER_CFG)
    after_cfg["hidden_dim"] = 128
    after_cfg["num_layers"] = 3

    before_params = sum(p.numel() for p in before_build(before_cfg).parameters())
    after_params = sum(p.numel() for p in after_build(after_cfg).parameters())
    assert after_params < before_params, (
        f"After model ({after_params} params) should be smaller "
        f"than before model ({before_params} params)."
    )

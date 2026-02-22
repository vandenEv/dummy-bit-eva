"""eval.py — Evaluation helpers for scenario_fail_big_gpu_like."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Compute average loss and accuracy over a DataLoader.

    Args:
        model: The model to evaluate.
        loader: DataLoader yielding (features, labels) batches.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        (avg_loss, accuracy) as floats.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy

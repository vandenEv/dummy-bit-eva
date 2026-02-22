"""model.py — Smaller MLP for the AFTER (fixed) state.

FIX 2 applied (via config):
- hidden_dim: 256 → 128
- num_layers: 6 → 3

The architecture is identical to the BEFORE version; the compute reduction
comes entirely from the smaller config values. No code change required here
beyond the docstring — demonstrating that config-level fixes are often enough.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepMLP(nn.Module):
    """A configurable MLP classifier.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Width of each hidden layer.
        num_layers: Number of hidden layers.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = input_dim
        for _ in range(num_layers):
            layers += [
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            ]
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(cfg: dict) -> DeepMLP:
    """Instantiate DeepMLP from config."""
    return DeepMLP(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_classes=cfg["num_classes"],
    )

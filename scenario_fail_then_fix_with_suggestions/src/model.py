"""model.py — Deep MLP for the BEFORE state.

WASTEFUL: 6 hidden layers at 256 dims is excessive for a synthetic
classification task. FIX 2 reduces this to 3 layers at 128 dims.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepMLP(nn.Module):
    """A configurable deep MLP classifier.

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
        """Forward pass.

        Args:
            x: (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        return self.net(x)


def build_model(cfg: dict) -> DeepMLP:
    """Instantiate DeepMLP from a config dict."""
    return DeepMLP(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_classes=cfg["num_classes"],
    )

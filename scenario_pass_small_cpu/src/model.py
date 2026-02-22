"""model.py — Tiny MLP for scenario_pass_small_cpu.

A straightforward multi-layer perceptron. Each hidden layer is
`hidden_dim` units wide with ReLU activation. The number of hidden
layers is controlled by `num_layers`.

Deliberately minimal so it produces a low compute footprint.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SmallMLP(nn.Module):
    """A lightweight MLP classifier.

    Args:
        input_dim: Dimension of the input features.
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
            layers += [nn.Linear(in_features, hidden_dim), nn.ReLU()]
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        return self.net(x)


def build_model(cfg: dict) -> SmallMLP:
    """Instantiate a SmallMLP from a config dict.

    Args:
        cfg: Parsed YAML config.

    Returns:
        An un-trained SmallMLP.
    """
    return SmallMLP(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_classes=cfg["num_classes"],
    )

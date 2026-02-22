"""model.py — Deep transformer-style model for scenario_fail_override_permission_only.

Intentionally heavy; no remediation applied for this scenario.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.ffn(x))
        return x


class HeavyTransformerClassifier(nn.Module):
    """Deep transformer classifier — intentionally over-engineered.

    Args:
        input_dim: Raw feature dimension.
        d_model: Hidden dimension.
        num_layers: Number of encoder blocks.
        num_heads: Attention heads.
        num_classes: Output classes.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        d_ff = d_model * 4
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x).unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        return self.classifier(x.mean(dim=1))


def build_model(cfg: dict) -> HeavyTransformerClassifier:
    """Instantiate the heavy transformer model from config."""
    return HeavyTransformerClassifier(
        input_dim=cfg["input_dim"],
        d_model=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        num_classes=cfg["num_classes"],
        dropout=cfg.get("dropout", 0.1),
    )

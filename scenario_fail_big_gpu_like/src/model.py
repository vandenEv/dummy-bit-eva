"""model.py — Deep transformer-style model for scenario_fail_big_gpu_like.

INTENTIONAL WASTE: This model is architecturally far heavier than needed
for a simple classification task on synthetic data:

- 8 transformer encoder layers (vs. 2 MLP layers in scenario_pass_small_cpu)
- 512-dim embeddings with 8 attention heads
- Position-wise FFN with 4× expansion (2048 intermediate dim)
- Dropout for all sub-layers

On a synthetic 10-class classification problem this is severe overkill.
The carbon-check analyzer should flag the depth and width.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PositionwiseFFN(nn.Module):
    """Two-layer feed-forward sublayer inside each transformer block.

    Args:
        d_model: Model dimension.
        d_ff: Inner (expanded) dimension. Typically 4 × d_model.
        dropout: Dropout probability.
    """

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
    """Single transformer encoder block with self-attention + FFN.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        d_ff: Inner FFN dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one transformer encoder block.

        Args:
            x: Tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor of the same shape.
        """
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.ffn(x))
        return x


class HeavyTransformerClassifier(nn.Module):
    """Deep transformer encoder for classification. Deliberately over-engineered.

    Architecture:
        input → linear projection → N × TransformerEncoderBlock
              → mean-pool → classifier head

    Args:
        input_dim: Raw feature dimension.
        d_model: Transformer hidden dimension (hidden_dim in config).
        num_layers: Number of stacked encoder blocks.
        num_heads: Attention heads per block.
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
        """Run forward pass.

        Args:
            x: Input of shape (batch_size, input_dim).

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        x = self.input_proj(x).unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        return self.classifier(x)


def build_model(cfg: dict) -> HeavyTransformerClassifier:
    """Instantiate the heavy transformer model from config.

    Args:
        cfg: Parsed YAML config.

    Returns:
        Un-trained HeavyTransformerClassifier.
    """
    return HeavyTransformerClassifier(
        input_dim=cfg["input_dim"],
        d_model=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        num_classes=cfg["num_classes"],
        dropout=cfg.get("dropout", 0.1),
    )

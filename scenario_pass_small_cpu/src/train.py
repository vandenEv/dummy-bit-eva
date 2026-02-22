"""train.py — Training entry point for scenario_pass_small_cpu.

Usage:
    python src/train.py --config configs/config.yaml

This script intentionally stays lightweight:
- Small model (SmallMLP)
- Short training loop
- Infrequent evaluation
- No redundant computation

These choices keep the estimated carbon footprint below the gate threshold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script from any working directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.data import make_loaders
from src.eval import evaluate
from src.model import build_model
from src.utils import get_logger, load_config, set_seed

logger = get_logger(__name__)


def train(cfg: dict) -> None:
    """Run the full training loop.

    Args:
        cfg: Parsed YAML config dict.
    """
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Data ---
    train_loader, val_loader = make_loaders(cfg, seed=cfg["seed"])
    logger.info(
        "Dataset — total: %d  |  train batches: %d  |  val batches: %d",
        cfg["dataset_size"],
        len(train_loader),
        len(val_loader),
    )

    # --- Model ---
    model = build_model(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", param_count)

    # --- Optimisation ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    eval_every = cfg.get("eval_every_n_steps", 50)
    global_step = 0

    # --- Training loop ---
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            # Evaluate only every N steps — not every step.
            if global_step % eval_every == 0:
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                logger.info(
                    "step %d | val_loss=%.4f | val_acc=%.3f",
                    global_step,
                    val_loss,
                    val_acc,
                )

        avg_loss = epoch_loss / max(len(train_loader), 1)
        logger.info("Epoch %d/%d — avg_train_loss=%.4f", epoch, cfg["epochs"], avg_loss)

    # Final evaluation
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    logger.info("Final — val_loss=%.4f | val_acc=%.3f", val_loss, val_acc)
    logger.info("Training complete.")


def main() -> None:
    """Parse CLI arguments and launch training."""
    parser = argparse.ArgumentParser(
        description="Train a small MLP (carbon-pass scenario)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()

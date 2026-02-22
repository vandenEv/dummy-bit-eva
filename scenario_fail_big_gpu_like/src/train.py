"""train.py — Training entry point for scenario_fail_big_gpu_like.

Usage:
    python src/train.py --config configs/config.yaml

INTENTIONAL WASTEFUL PATTERNS (for demo):
1. `expensive_preprocessing` is called at the top of every epoch even though
   the result is always identical (deterministic transform, fixed seed).
2. The validation loop runs after EVERY training step when
   eval_every_n_steps == 1 (the default in this config).
3. No early stopping — the loop always runs all `epochs` regardless of
   val loss behaviour.
4. Large model with heavy attention blocks (see model.py).
5. No gradient accumulation, no mixed precision.

→ These patterns collectively push the carbon-check score above the gate
  threshold and cause the PR to be blocked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn

from src.data import expensive_preprocessing, get_raw_numpy, make_loaders
from src.eval import evaluate
from src.model import build_model
from src.utils import get_logger, load_config, set_seed

logger = get_logger(__name__)


def train(cfg: dict) -> None:
    """Run the full training loop with intentionally wasteful patterns.

    Args:
        cfg: Parsed YAML config dict.
    """
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Data ---
    train_loader, val_loader = make_loaders(cfg, seed=cfg["seed"])
    # Keep raw arrays for the per-epoch preprocessing waste
    X_raw, _ = get_raw_numpy(cfg, seed=cfg["seed"])

    logger.info(
        "Dataset — total: %d  |  train batches: %d  |  val batches: %d",
        cfg["dataset_size"],
        len(train_loader),
        len(val_loader),
    )

    # --- Model ---
    model = build_model(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model parameters: %d  (hidden_dim=%d, num_layers=%d)",
        param_count,
        cfg["hidden_dim"],
        cfg["num_layers"],
    )

    # --- Optimisation ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    eval_every: int = cfg.get("eval_every_n_steps", 1)
    global_step = 0

    # --- Training loop ---
    for epoch in range(1, cfg["epochs"] + 1):

        # WASTEFUL PATTERN 1: Redo expensive preprocessing every epoch.
        # The result is identical each time because seed is fixed.
        # In a real project this would be done once and cached.
        _ = expensive_preprocessing(X_raw, seed=cfg["seed"])
        logger.debug("Epoch %d: preprocessing recomputed (wasteful).", epoch)

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

            # WASTEFUL PATTERN 2: Evaluate the full validation set every step.
            # eval_every_n_steps is set to 1 in config — this is extremely
            # expensive and provides essentially no additional signal.
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
        # WASTEFUL PATTERN 3: No early stopping check here — always continues.

    # Final evaluation
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    logger.info("Final — val_loss=%.4f | val_acc=%.3f", val_loss, val_acc)
    logger.info("Training complete.")


def main() -> None:
    """Parse CLI arguments and launch training."""
    parser = argparse.ArgumentParser(
        description="Train a heavy transformer model (carbon-fail scenario)."
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

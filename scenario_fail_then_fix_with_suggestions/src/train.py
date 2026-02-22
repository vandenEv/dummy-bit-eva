"""train.py — BEFORE state (wasteful). Triggers carbon gate failure.

Usage:
    python src/train.py --config configs/config_before.yaml

WASTEFUL PATTERNS IN THIS VERSION:
1. `preprocess()` called every epoch — always returns the same result (FIX 3).
2. Validation loop runs every step (eval_every_n_steps=1 in config) (FIX 3).
3. No early stopping — runs all 40 epochs unconditionally (FIX 1).
4. Model is 6 layers × 256 dims — much larger than needed (FIX 2).
5. dataset_size=8000 — excessive for a PR smoke-check (FIX 3).

Apply fixes in order (commits 1 → 2 → 3) to watch the carbon score drop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.data import get_raw_numpy, make_loaders, preprocess
from src.eval import evaluate
from src.model import build_model
from src.utils import get_logger, load_config, set_seed

logger = get_logger(__name__)


def train(cfg: dict) -> None:
    """Run the wasteful BEFORE training loop.

    Args:
        cfg: Parsed YAML config dict.
    """
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    train_loader, val_loader = make_loaders(cfg, seed=cfg["seed"])
    X_raw, _ = get_raw_numpy(cfg, seed=cfg["seed"])

    logger.info(
        "Dataset size: %d  |  train batches: %d", cfg["dataset_size"], len(train_loader)
    )

    model = build_model(cfg).to(device)
    logger.info(
        "Model: %d params  (hidden_dim=%d, num_layers=%d)",
        sum(p.numel() for p in model.parameters()),
        cfg["hidden_dim"],
        cfg["num_layers"],
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    eval_every: int = cfg.get("eval_every_n_steps", 1)
    early_stopping: bool = cfg.get("early_stopping", False)
    patience: int = cfg.get("patience", 0)
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    for epoch in range(1, cfg["epochs"] + 1):
        # WASTEFUL: recompute preprocessing every epoch (result is identical)
        _ = preprocess(X_raw, seed=cfg["seed"])

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

            # WASTEFUL: per-step validation
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

        # FIX 1 target: early stopping block (currently disabled via config)
        if early_stopping:
            val_loss, _ = evaluate(model, val_loader, criterion, device)
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d.", epoch)
                    break

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    logger.info("Final — val_loss=%.4f | val_acc=%.3f", val_loss, val_acc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train (BEFORE state — wasteful).")
    parser.add_argument("--config", type=str, default="configs/config_before.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()

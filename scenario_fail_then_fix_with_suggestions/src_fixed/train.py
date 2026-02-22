"""train.py — AFTER state. All three remediations applied.

Usage:
    python src_fixed/train.py --config configs/config_after.yaml

REMEDIATIONS APPLIED (relative to src/train.py):

FIX 1 — Early stopping + reduced epochs
    - `early_stopping: true` in config + patience-based loop exit.
    - `epochs: 10` (was 40) — acts as an upper bound only.

FIX 2 — Smaller model
    - `hidden_dim: 128`, `num_layers: 3` in config_after.yaml.
    - No code change here; entirely config-driven.

FIX 3 — Cache preprocessing + fast_dev_run + less frequent eval
    - `preprocess()` in src_fixed/data.py now uses a module-level cache.
      The function is called once per epoch, but returns cached data instantly
      after the first call.
    - `fast_dev_run: true` in config — training loop breaks after a single
      batch, acting as a rapid smoke test for PR checks.
    - `eval_every_n_steps: 50` (was 1) — 50× fewer validation passes.

BONUS — Optional mixed precision (auto-detected)
    - If a CUDA device is available, torch.autocast reduces memory and latency.
    - Falls back gracefully to full precision on CPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src_fixed.data import get_raw_numpy, make_loaders, preprocess
from src_fixed.eval import evaluate
from src_fixed.model import build_model
from src_fixed.utils import get_logger, load_config, set_seed

logger = get_logger(__name__)


def train(cfg: dict) -> None:
    """Run the optimised training loop.

    Args:
        cfg: Parsed YAML config dict.
    """
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    logger.info("Device: %s  |  AMP: %s", device, use_amp)

    # --- Data ---
    train_loader, val_loader = make_loaders(cfg, seed=cfg["seed"])
    X_raw, _ = get_raw_numpy(cfg, seed=cfg["seed"])

    # FIX 3: Warm the preprocessing cache once before the epoch loop.
    _ = preprocess(X_raw, seed=cfg["seed"])
    logger.info("Preprocessing cache warmed.")

    logger.info(
        "Dataset size: %d  |  train batches: %d", cfg["dataset_size"], len(train_loader)
    )

    # --- Model ---
    model = build_model(cfg).to(device)
    logger.info(
        "Model: %d params  (hidden_dim=%d, num_layers=%d)",
        sum(p.numel() for p in model.parameters()),
        cfg["hidden_dim"],
        cfg["num_layers"],
    )

    # --- Optimisation ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # --- Config knobs ---
    eval_every: int = cfg.get("eval_every_n_steps", 50)  # FIX 3
    early_stopping: bool = cfg.get("early_stopping", True)  # FIX 1
    patience: int = cfg.get("patience", 3)  # FIX 1
    fast_dev_run: bool = cfg.get("fast_dev_run", False)  # FIX 3
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    # --- Training loop ---
    for epoch in range(1, cfg["epochs"] + 1):
        # FIX 3: preprocess() returns the cached result instantly after the
        # first call — negligible overhead, no repeated matrix multiply.
        _ = preprocess(X_raw, seed=cfg["seed"])

        model.train()
        epoch_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # BONUS: mixed precision forward + backward
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            global_step += 1

            # FIX 3: evaluate only every eval_every steps, not every step
            if global_step % eval_every == 0:
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                logger.info(
                    "step %d | val_loss=%.4f | val_acc=%.3f",
                    global_step,
                    val_loss,
                    val_acc,
                )

            # FIX 3: fast_dev_run — one batch is enough for a smoke test
            if fast_dev_run:
                logger.info("fast_dev_run=True: stopping after 1 batch.")
                return

        avg_loss = epoch_loss / max(len(train_loader), 1)
        logger.info("Epoch %d/%d — avg_train_loss=%.4f", epoch, cfg["epochs"], avg_loss)

        # FIX 1: early stopping check after each epoch
        if early_stopping:
            val_loss, _ = evaluate(model, val_loader, criterion, device)
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(
                    "Early stopping patience: %d/%d", patience_counter, patience
                )
                if patience_counter >= patience:
                    logger.info("Early stopping triggered at epoch %d.", epoch)
                    break

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    logger.info("Final — val_loss=%.4f | val_acc=%.3f", val_loss, val_acc)
    logger.info("Training complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train (AFTER state — optimised).")
    parser.add_argument("--config", type=str, default="configs/config_after.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()

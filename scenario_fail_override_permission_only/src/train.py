"""train.py — scenario_fail_override_permission_only.

Usage:
    python src/train.py --config configs/config.yaml

    # Demo narration only (no effect on gate):
    ALLOW_OVERRIDE_DEMO=true python src/train.py --config configs/config.yaml

─────────────────────────────────────────────────────────────────────────────
ABOUT ALLOW_OVERRIDE_DEMO
─────────────────────────────────────────────────────────────────────────────
The environment variable ALLOW_OVERRIDE_DEMO is read by this script and
logged, but it has ZERO effect on the carbon-check gate.

Why is it here?
  During a live demo, the presenter can point to this line and explain:
  "Developers cannot grant themselves an override inside code.
   The gate enforcement is in the GitHub Actions workflow, which calls
   the GitHub API to verify the commenter has 'write' or 'admin' permissions
   on this repository before accepting a /carbon-override comment."

What the actual override looks like:
  1. PR is opened → carbon-check detects high compute → gate blocks merge.
  2. A maintainer comments:   /carbon-override reason: "approved for nightly batch"
  3. carbon-gate.yml re-runs, reads the comment, calls
     GET /repos/{owner}/{repo}/collaborators/{commenter}/permission,
     gets { "permission": "admin" } → marks the gate status as passing.
  4. A contributor (role: read) posting the same comment gets:
     "Override denied: your repository role (read) is insufficient."

ALLOW_OVERRIDE_DEMO in the env does not hook into step 3 in any way.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.data import make_loaders
from src.eval import evaluate
from src.model import build_model
from src.utils import get_logger, load_config, set_seed

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# DEMO NARRATION FLAG — read from environment, has no gate effect.
# ---------------------------------------------------------------------------
ALLOW_OVERRIDE_DEMO: bool = os.getenv("ALLOW_OVERRIDE_DEMO", "false").lower() == "true"


def _log_override_flag() -> None:
    """Log the demo flag state so it's visible in CI output."""
    if ALLOW_OVERRIDE_DEMO:
        logger.info(
            "ALLOW_OVERRIDE_DEMO=True — demo flag is set "
            "(enforcement is in the workflow, not here)."
        )
    else:
        logger.info(
            "ALLOW_OVERRIDE_DEMO=False — no override flag active. "
            "Gate will block this PR unless a maintainer uses /carbon-override."
        )


def train(cfg: dict) -> None:
    """Run the heavy training loop — no remediations applied.

    Args:
        cfg: Parsed YAML config dict.
    """
    _log_override_flag()
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    train_loader, val_loader = make_loaders(cfg, seed=cfg["seed"])
    logger.info(
        "Dataset: %d samples  |  train batches: %d",
        cfg["dataset_size"], len(train_loader),
    )

    model = build_model(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model: %d params  (hidden_dim=%d, num_layers=%d) — intentionally heavy.",
        param_count, cfg["hidden_dim"], cfg["num_layers"],
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    eval_every: int = cfg.get("eval_every_n_steps", 1)
    global_step = 0

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

            # Per-step evaluation — intentionally wasteful
            if global_step % eval_every == 0:
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                logger.info("step %d | val_loss=%.4f | val_acc=%.3f",
                            global_step, val_loss, val_acc)

        avg_loss = epoch_loss / max(len(train_loader), 1)
        logger.info("Epoch %d/%d — avg_train_loss=%.4f", epoch, cfg["epochs"], avg_loss)

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    logger.info("Final — val_loss=%.4f | val_acc=%.3f", val_loss, val_acc)
    logger.info("Training complete. This run would require a maintainer override in a PR.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train heavy model — override scenario demo."
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()

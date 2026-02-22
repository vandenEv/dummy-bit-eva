"""utils.py — Shared helpers for scenario_pass_small_cpu."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config file and return it as a plain dict.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Dictionary of config values.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open() as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def get_logger(name: str) -> logging.Logger:
    """Return a simple stdout logger.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured Logger instance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)

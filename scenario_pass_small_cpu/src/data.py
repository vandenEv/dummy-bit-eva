"""data.py — Synthetic dataset for scenario_pass_small_cpu.

Generates a small, fast, fully in-memory classification dataset.
No disk I/O; deterministic given a fixed seed.
"""

print("Importing data.py....more edits...hopefully lastttttt")
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def make_dataset(
    dataset_size: int,
    input_dim: int,
    num_classes: int,
    seed: int = 42,
) -> TensorDataset:
    """Return a TensorDataset with synthetic Gaussian features and integer labels.

    Args:
        dataset_size: Number of samples to generate.
        input_dim: Dimensionality of each input vector.
        num_classes: Number of output classes.
        seed: Random seed for reproducibility.

    Returns:
        A TensorDataset of (features, labels).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((dataset_size, input_dim)).astype(np.float32)
    y = rng.integers(0, num_classes, size=(dataset_size,)).astype(np.int64)
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def make_loaders(
    cfg: dict,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from config.

    Args:
        cfg: Parsed YAML config dict.
        seed: Random seed for dataset generation.

    Returns:
        (train_loader, val_loader)
    """
    dataset_size: int = cfg["dataset_size"]
    input_dim: int = cfg["input_dim"]
    num_classes: int = cfg["num_classes"]
    batch_size: int = cfg["batch_size"]

    # 80/20 train-val split
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    full_ds = make_dataset(dataset_size, input_dim, num_classes, seed=seed)
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

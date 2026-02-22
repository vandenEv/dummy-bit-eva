"""data.py — Synthetic dataset for scenario_fail_override_permission_only.

No caching; no optimisation. Kept heavy deliberately.
"""

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
    """Generate a large synthetic classification dataset."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((dataset_size, input_dim)).astype(np.float32)
    y = rng.integers(0, num_classes, size=(dataset_size,)).astype(np.int64)
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def make_loaders(cfg: dict, seed: int = 42) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders."""
    dataset_size: int = cfg["dataset_size"]
    input_dim: int = cfg["input_dim"]
    num_classes: int = cfg["num_classes"]
    batch_size: int = cfg["batch_size"]

    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    full_ds = make_dataset(dataset_size, input_dim, num_classes, seed=seed)
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )

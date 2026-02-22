"""data.py — Synthetic dataset for the BEFORE state.

WASTEFUL PATTERNS present in this version:
- `preprocess` re-normalises data on every call (no caching).
- Called once per epoch in train.py, but result is never stored.

FIX 3 removes this waste by enabling `use_cache` in the config and
adding a cache path in src_fixed/data.py.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def preprocess(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """Normalise and apply a fixed random projection.

    WASTEFUL: This is deterministic and always produces the same output,
    but it is called from the training loop every epoch.

    Args:
        X: Feature matrix of shape (N, D).
        seed: Fixed seed — the projection is always identical.

    Returns:
        Normalised, projected feature matrix.
    """
    rng = np.random.default_rng(seed)
    D = X.shape[1]
    W = rng.standard_normal((D, D)).astype(np.float32)
    X_out = X @ W
    X_out = (X_out - X_out.mean(axis=0)) / (X_out.std(axis=0) + 1e-8)
    return X_out


def make_dataset(
    dataset_size: int,
    input_dim: int,
    num_classes: int,
    seed: int = 42,
) -> TensorDataset:
    """Generate a synthetic classification dataset."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((dataset_size, input_dim)).astype(np.float32)
    y = rng.integers(0, num_classes, size=(dataset_size,)).astype(np.int64)
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def make_loaders(cfg: dict, seed: int = 42) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders.

    Args:
        cfg: Parsed YAML config.
        seed: RNG seed.

    Returns:
        (train_loader, val_loader)
    """
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
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def get_raw_numpy(cfg: dict, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return raw numpy arrays (used for the per-epoch preprocessing waste)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((cfg["dataset_size"], cfg["input_dim"])).astype(np.float32)
    y = rng.integers(0, cfg["num_classes"], size=(cfg["dataset_size"],)).astype(
        np.int64
    )
    return X, y

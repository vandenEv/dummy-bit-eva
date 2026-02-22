"""data.py — Synthetic dataset for the AFTER (fixed) state.

FIX 3 applied:
- `preprocess()` now stores its result in a module-level cache keyed by
  (dataset_size, input_dim, seed). Subsequent calls with the same arguments
  return the cached result instantly — zero recomputation.
- DataLoaders use num_workers=0 (safe cross-platform default); increase for
  production workloads.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Module-level cache: (dataset_size, input_dim, seed) → preprocessed X
_PREPROCESS_CACHE: dict[tuple, np.ndarray] = {}


def preprocess(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """Normalise and project features — result is cached after the first call.

    The cache key is derived from the array shape and seed so identical inputs
    never recompute, saving one matrix multiply + normalisation per epoch.

    Args:
        X: Feature matrix of shape (N, D).
        seed: Fixed seed for the projection matrix.

    Returns:
        Cached (or freshly computed) normalised, projected feature matrix.
    """
    cache_key = (X.shape, seed)
    if cache_key in _PREPROCESS_CACHE:
        return _PREPROCESS_CACHE[cache_key]

    rng = np.random.default_rng(seed)
    D = X.shape[1]
    W = rng.standard_normal((D, D)).astype(np.float32)
    X_out = X @ W
    X_out = (X_out - X_out.mean(axis=0)) / (X_out.std(axis=0) + 1e-8)
    _PREPROCESS_CACHE[cache_key] = X_out
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
    """Build train/val DataLoaders with optional fast_dev_run truncation.

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
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def get_raw_numpy(cfg: dict, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return raw numpy arrays for preprocessing demo."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((cfg["dataset_size"], cfg["input_dim"])).astype(np.float32)
    y = rng.integers(0, cfg["num_classes"], size=(cfg["dataset_size"],)).astype(
        np.int64
    )
    return X, y

"""data.py — Synthetic dataset for scenario_fail_big_gpu_like.

INTENTIONAL WASTE (for demo purposes):
1. `expensive_preprocessing` re-normalises and applies a redundant matrix
   multiplication on every call — it is called once per epoch in the training
   loop rather than being computed once up front.
2. `augment_sample` runs a nested loop (`augmentation_repeats` times per
   sample) that does unnecessary floating-point operations each time.

Both patterns are highlighted here so the carbon-check bot can call them out.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# WASTEFUL PATTERN 1: Expensive preprocessing called repeatedly each epoch
# ---------------------------------------------------------------------------


def expensive_preprocessing(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """Apply a redundant random matrix transform on every call.

    In a production system this would be cached after the first call or,
    better, performed in a proper Dataset.__getitem__ with caching.
    Here it is called from the training loop each epoch — pure waste.

    Args:
        X: Raw feature matrix of shape (N, D).
        seed: RNG seed (kept constant so the "transform" is always the same,
              making re-running it truly pointless).

    Returns:
        Transformed feature matrix of the same shape.
    """
    rng = np.random.default_rng(seed)
    D = X.shape[1]
    # Random orthogonal-ish projection — same every call because seed is fixed.
    W = rng.standard_normal((D, D)).astype(np.float32)
    X_transformed = X @ W
    # Redundant normalisation on top of the projection.
    X_transformed = (X_transformed - X_transformed.mean(axis=0)) / (
        X_transformed.std(axis=0) + 1e-8
    )
    return X_transformed


# ---------------------------------------------------------------------------
# WASTEFUL PATTERN 2: Nested augmentation loop
# ---------------------------------------------------------------------------


def augment_sample(x: np.ndarray, repeats: int, rng: np.random.Generator) -> np.ndarray:
    """Apply Gaussian noise to a sample N times in a nested loop.

    Each iteration is a no-op from a learning perspective (the noise averages
    out). The loop purely burns CPU cycles.

    Args:
        x: Single feature vector.
        repeats: Number of augmentation rounds (each costs ~D ops).
        rng: NumPy random generator.

    Returns:
        The sample after `repeats` rounds of meaningless augmentation.
    """
    for _ in range(repeats):
        # Add tiny noise and then immediately subtract it — literally nothing
        # changes except wasted computation.
        noise = rng.standard_normal(x.shape).astype(np.float32) * 1e-8
        x = x + noise - noise
    return x


def make_dataset(
    dataset_size: int,
    input_dim: int,
    num_classes: int,
    augmentation_repeats: int = 1,
    seed: int = 42,
) -> TensorDataset:
    """Generate a large synthetic dataset with augmentation applied.

    Args:
        dataset_size: Number of samples.
        input_dim: Feature dimension.
        num_classes: Number of output classes.
        augmentation_repeats: Times each sample passes through augment_sample
                              (WASTEFUL when > 1).
        seed: RNG seed.

    Returns:
        TensorDataset of (features, labels).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((dataset_size, input_dim)).astype(np.float32)
    y = rng.integers(0, num_classes, size=(dataset_size,)).astype(np.int64)

    # WASTEFUL: augmentation loop applied during dataset construction.
    if augmentation_repeats > 1:
        for i in range(dataset_size):
            X[i] = augment_sample(X[i], repeats=augmentation_repeats, rng=rng)

    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def make_loaders(cfg: dict, seed: int = 42) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders.

    Note: `expensive_preprocessing` is NOT called here — it is called inside
    the training loop each epoch to simulate the wasteful pattern.

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
    augmentation_repeats: int = cfg.get("augmentation_repeats", 1)

    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    full_ds = make_dataset(
        dataset_size,
        input_dim,
        num_classes,
        augmentation_repeats=augmentation_repeats,
        seed=seed,
    )
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_raw_numpy(cfg: dict, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return raw numpy arrays for the expensive_preprocessing demo.

    Args:
        cfg: Parsed YAML config.
        seed: RNG seed.

    Returns:
        (X, y) numpy arrays.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((cfg["dataset_size"], cfg["input_dim"])).astype(np.float32)
    y = rng.integers(0, cfg["num_classes"], size=(cfg["dataset_size"],)).astype(
        np.int64
    )
    return X, y

"""Lightweight geometry utilities (no heavy deps required)."""
from __future__ import annotations

import numpy as np


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel-grid downsample a (N, 3) point cloud.

    Args:
        points: (N, 3) float32.
        voxel_size: side length of each voxel.

    Returns:
        (M, 3) downsampled points (one mean per occupied voxel).
    """
    if points.size == 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    out = np.zeros((inv.max() + 1, 3), dtype=np.float64)
    counts = np.zeros(inv.max() + 1, dtype=np.int64)
    np.add.at(out, inv, points)
    np.add.at(counts, inv, 1)
    return (out / counts[:, None]).astype(np.float32)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 SE(3) transform to a (..., 3) point array.

    Args:
        points: (..., 3) float32.
        transform: (4, 4) float32.

    Returns:
        Transformed points with the same leading shape as ``points``.
    """
    if transform.shape != (4, 4):
        raise ValueError("transform must be 4x4")
    flat = points.reshape(-1, 3)
    homo = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=flat.dtype)], axis=-1)
    out = homo @ transform.T
    return out[..., :3].reshape(points.shape)

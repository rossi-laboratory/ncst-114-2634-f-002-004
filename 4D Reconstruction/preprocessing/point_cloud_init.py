"""Grid-based 3D point initialization from RGB + depth.

Each grid pixel is back-projected to camera space using the depth map and
camera intrinsics, then assigned a fixed integer ID that persists for the
entire video (a key requirement for 3D point tracking with stable identity).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def initialize_grid_points(
    frame: np.ndarray,
    depth: np.ndarray,
    camera_intrinsics: Dict[str, float],
    grid_stride: int = 8,
    max_points: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize a grid of 3D points from an RGB frame and depth map.

    Args:
        frame: (H, W, 3) uint8 RGB (currently unused but kept for future use,
            e.g. masking by saliency).
        depth: (H, W) float32 depth in metres or relative units.
        camera_intrinsics: dict with keys ``fx``, ``fy``, ``cx``, ``cy``.
        grid_stride: stride between sampled grid pixels.
        max_points: optional cap; if exceeded, points are uniformly sub-sampled.

    Returns:
        points_3d: (N, 3) float32 array of camera-space xyz coordinates.
        point_ids: (N,) int64 array of stable IDs.
    """
    if frame.shape[:2] != depth.shape:
        raise ValueError(
            f"frame {frame.shape[:2]} and depth {depth.shape} must match."
        )
    H, W = depth.shape
    us = np.arange(0, W, grid_stride)
    vs = np.arange(0, H, grid_stride)
    u_grid, v_grid = np.meshgrid(us, vs)
    u_flat = u_grid.flatten()
    v_flat = v_grid.flatten()

    z = depth[v_flat, u_flat].astype(np.float32)
    valid = z > 0
    u_v, v_v, z_v = u_flat[valid], v_flat[valid], z[valid]

    fx = float(camera_intrinsics["fx"])
    fy = float(camera_intrinsics["fy"])
    cx = float(camera_intrinsics["cx"])
    cy = float(camera_intrinsics["cy"])

    x = (u_v - cx) * z_v / fx
    y = (v_v - cy) * z_v / fy
    points_3d = np.stack([x, y, z_v], axis=-1).astype(np.float32)  # (N, 3)
    point_ids = np.where(valid)[0].astype(np.int64)                 # (N,)

    if max_points is not None and points_3d.shape[0] > max_points:
        idx = np.linspace(0, points_3d.shape[0] - 1, max_points).astype(np.int64)
        points_3d = points_3d[idx]
        point_ids = point_ids[idx]

    return points_3d, point_ids


def project_3d_to_2d(
    points_3d: np.ndarray,
    camera_intrinsics: Dict[str, float],
) -> np.ndarray:
    """Project 3D camera-space points back to 2D pixel coordinates.

    Args:
        points_3d: (N, 3) float32.
        camera_intrinsics: dict with keys ``fx``, ``fy``, ``cx``, ``cy``.

    Returns:
        pixels: (N, 2) float32 array of (u, v) pixel coordinates.
    """
    fx = float(camera_intrinsics["fx"])
    fy = float(camera_intrinsics["fy"])
    cx = float(camera_intrinsics["cx"])
    cy = float(camera_intrinsics["cy"])
    z = np.clip(points_3d[:, 2], a_min=1e-6, a_max=None)
    u = points_3d[:, 0] * fx / z + cx
    v = points_3d[:, 1] * fy / z + cy
    return np.stack([u, v], axis=-1).astype(np.float32)

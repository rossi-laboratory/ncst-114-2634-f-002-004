"""Unit tests for grid initialization and tracker fallback bookkeeping."""
from __future__ import annotations

import numpy as np

from preprocessing.point_cloud_init import initialize_grid_points, project_3d_to_2d


def test_initialize_grid_points_shapes_and_ids():
    H, W = 64, 80
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    depth = np.ones((H, W), dtype=np.float32)
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": W / 2, "cy": H / 2}

    pts, ids = initialize_grid_points(frame, depth, intrinsics, grid_stride=8)
    expected = (H // 8) * (W // 8)
    assert pts.shape == (expected, 3)
    assert ids.shape == (expected,)
    assert pts.dtype == np.float32
    assert ids.dtype == np.int64
    assert np.unique(ids).size == ids.size  # IDs unique


def test_initialize_grid_points_skips_invalid_depth():
    H, W = 32, 32
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    depth = np.zeros((H, W), dtype=np.float32)
    depth[8:24, 8:24] = 1.5
    intrinsics = {"fx": 50.0, "fy": 50.0, "cx": W / 2, "cy": H / 2}
    pts, ids = initialize_grid_points(frame, depth, intrinsics, grid_stride=4)
    assert pts.shape[0] > 0
    assert pts.shape[0] == ids.shape[0]


def test_project_round_trip():
    intrinsics = {"fx": 200.0, "fy": 200.0, "cx": 50.0, "cy": 30.0}
    pts = np.array([[0.1, -0.2, 1.0], [0.3, 0.4, 2.0]], dtype=np.float32)
    px = project_3d_to_2d(pts, intrinsics)
    assert px.shape == (2, 2)
    assert np.isfinite(px).all()

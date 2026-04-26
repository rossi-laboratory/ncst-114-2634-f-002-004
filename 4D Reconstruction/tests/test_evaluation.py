"""Tests for the four evaluation metrics."""
from __future__ import annotations

import numpy as np

from evaluation.chamfer_distance import chamfer_distance
from evaluation.temporal_consistency import temporal_consistency
from evaluation.tracking_error import tracking_error


def test_chamfer_zero_for_identical():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(5, 100, 3)).astype(np.float32)
    res = chamfer_distance(pts, pts)
    assert res["chamfer_distance"] < 1e-5
    assert res["pass"] is True


def test_temporal_consistency_zero_for_identical():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(5, 50, 3)).astype(np.float32)
    res = temporal_consistency(pts, pts)
    assert res["temporal_error"] < 1e-5
    assert res["pass"] is True


def test_tracking_error_basic():
    rng = np.random.default_rng(2)
    gt = rng.normal(size=(4, 20, 3)).astype(np.float32)
    pred = gt + 0.001
    res = tracking_error(pred, gt)
    assert res["tracking_success_rate"] >= 0.9
    assert res["id_switch_rate"] == 0.0
    assert res["pass_id_switch"] is True
    assert res["pass_tracking_success"] is True


def test_tracking_handles_nans():
    gt = np.zeros((3, 5, 3), dtype=np.float32)
    gt[1, 2] = np.nan
    pred = gt.copy()
    res = tracking_error(pred, gt)
    assert 0.0 <= res["tracking_success_rate"] <= 1.0

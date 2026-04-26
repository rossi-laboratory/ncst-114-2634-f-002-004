"""Chamfer distance over a 3D point sequence (target: <= 5%)."""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.spatial import cKDTree


def chamfer_distance(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: float = 0.05,
) -> Dict[str, object]:
    """Compute average bidirectional Chamfer distance over a clip.

    Args:
        pred: (T, N, 3) predicted point cloud sequence.
        gt:   (T, N, 3) ground-truth sequence.
        threshold: pass/fail threshold (default 5%).

    Returns:
        dict with ``chamfer_distance`` (float), ``per_frame`` (np.ndarray),
        and ``pass`` (bool).
    """
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape}, gt={gt.shape}")

    T = pred.shape[0]
    per_frame = np.zeros(T, dtype=np.float64)
    for t in range(T):
        p = pred[t]
        g = gt[t]
        valid_p = ~np.isnan(p).any(axis=-1)
        valid_g = ~np.isnan(g).any(axis=-1)
        if valid_p.sum() == 0 or valid_g.sum() == 0:
            per_frame[t] = np.nan
            continue
        pp = p[valid_p]
        gg = g[valid_g]
        d_p2g, _ = cKDTree(gg).query(pp)
        d_g2p, _ = cKDTree(pp).query(gg)
        per_frame[t] = 0.5 * (d_p2g.mean() + d_g2p.mean())

    cd = float(np.nanmean(per_frame))
    return {
        "chamfer_distance": cd,
        "per_frame": per_frame,
        "pass": bool(cd <= threshold),
    }

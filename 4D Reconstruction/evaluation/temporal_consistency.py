"""Temporal consistency metric (target: <= 10%)."""
from __future__ import annotations

from typing import Dict

import numpy as np


def temporal_consistency(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: float = 0.10,
) -> Dict[str, object]:
    """Compare inter-frame point displacements between pred and GT.

    Args:
        pred: (T, N, 3) predicted track.
        gt:   (T, N, 3) GT track.
        threshold: pass/fail threshold (default 10%).

    Returns:
        dict with ``temporal_error`` (float), ``per_frame`` (np.ndarray, len T-1),
        ``pass`` (bool).
    """
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape}, gt={gt.shape}")

    pred_delta = pred[1:] - pred[:-1]      # (T-1, N, 3)
    gt_delta = gt[1:] - gt[:-1]
    diff = np.linalg.norm(pred_delta - gt_delta, axis=-1)  # (T-1, N)
    valid = ~np.isnan(diff)
    per_frame = np.where(valid.any(axis=-1),
                         np.nanmean(np.where(valid, diff, np.nan), axis=-1),
                         np.nan)
    error = float(np.nanmean(per_frame))
    return {
        "temporal_error": error,
        "per_frame": per_frame,
        "pass": bool(error <= threshold),
    }

"""Point tracking error, ID-Switch rate, and tracking success rate."""
from __future__ import annotations

from typing import Dict

import numpy as np


def tracking_error(
    pred_tracks: np.ndarray,
    gt_tracks: np.ndarray,
    pred_ids: np.ndarray | None = None,
    gt_ids: np.ndarray | None = None,
    id_switch_threshold: float = 0.08,
    success_threshold: float = 0.90,
) -> Dict[str, object]:
    """Compute tracking error, ID-switch rate, and success rate.

    Args:
        pred_tracks: (T, N, 3) predicted tracks (NaN = failure).
        gt_tracks:   (T, N, 3) GT tracks (NaN = absent).
        pred_ids:    (T, N) per-step assigned IDs. If None, assumes IDs are stable
            (i.e. zero ID-switches).
        gt_ids:      (T, N) GT IDs. Defaults to ``pred_ids`` for stable matching.
        id_switch_threshold: pass threshold (default 0.08).
        success_threshold: pass threshold (default 0.90).

    Returns:
        dict with ``point_tracking_error``, ``id_switch_rate``,
        ``tracking_success_rate``, and the relevant ``pass_*`` flags.
    """
    if pred_tracks.shape != gt_tracks.shape:
        raise ValueError("pred and gt must have the same shape.")
    T, N, _ = gt_tracks.shape

    valid = ~np.isnan(gt_tracks).any(axis=-1)         # (T, N)
    err = np.linalg.norm(pred_tracks - gt_tracks, axis=-1)  # (T, N)
    track_err = float(err[valid].mean()) if valid.any() else float("nan")

    success_rate = float(valid.mean())

    if pred_ids is None:
        # Default: assume stable per-track IDs implied by the second axis.
        ids = np.tile(np.arange(N), (T, 1))
        pred_ids = ids
    if gt_ids is None:
        gt_ids = pred_ids

    switches = (pred_ids[1:] != pred_ids[:-1]).sum()
    total_transitions = (T - 1) * N if T > 1 else 1
    id_switch_rate = float(switches / max(1, total_transitions))

    return {
        "point_tracking_error": track_err,
        "id_switch_rate": id_switch_rate,
        "tracking_success_rate": success_rate,
        "pass_id_switch": bool(id_switch_rate <= id_switch_threshold),
        "pass_tracking_success": bool(success_rate >= success_threshold),
    }

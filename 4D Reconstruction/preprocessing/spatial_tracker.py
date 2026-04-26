"""SpatialTracker wrapper with optional 2D-flow + depth fallback.

Installation (recommended):
    git clone https://github.com/henry123-boy/SpatialTracker.git
    cd SpatialTracker && pip install -e .

If SpatialTracker is unavailable, this module falls back to dense 2D optical
flow (Farneback) plus a depth lookup, which is less accurate but enables a
runnable end-to-end pipeline for development.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from omegaconf import DictConfig

from preprocessing.point_cloud_init import project_3d_to_2d
from utils.logger import get_logger

logger = get_logger(__name__)


class SpatialTrackerWrapper:
    """Track stable-ID 3D points across an entire video.

    Args:
        cfg: Hydra config node containing camera intrinsics and tracker options.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.tracker: Optional[object] = None
        self._use_fallback: bool = True
        self._load_tracker()

    def _load_tracker(self) -> None:
        try:
            from spatialtracker import SpatialTracker  # type: ignore

            self.tracker = SpatialTracker()
            self._use_fallback = False
            logger.info("Loaded SpatialTracker.")
        except ImportError:
            logger.warning(
                "SpatialTracker not installed; falling back to 2D optical flow + "
                "depth lookup. See preprocessing/spatial_tracker.py docstring "
                "for installation instructions."
            )
            self._use_fallback = True

    def track(
        self,
        frames: np.ndarray,
        init_points_3d: np.ndarray,
        init_point_ids: np.ndarray,
        depths: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Track 3D points throughout the video.

        Args:
            frames: (T, H, W, 3) uint8.
            init_points_3d: (N, 3) initial 3D positions at frame 0.
            init_point_ids: (N,) stable point IDs.
            depths: optional (T, H, W) float32 depth maps used by the fallback
                tracker. Required when SpatialTracker is unavailable.

        Returns:
            tracks: (T, N, 3) float32 with NaN for points that failed to track.
        """
        if not self._use_fallback:
            return self._spatial_tracker_track(frames, init_points_3d, init_point_ids)
        if depths is None:
            raise ValueError(
                "Depths must be provided when SpatialTracker is unavailable."
            )
        return self._fallback_track(frames, init_points_3d, depths)

    def _spatial_tracker_track(
        self,
        frames: np.ndarray,
        init_points_3d: np.ndarray,
        init_point_ids: np.ndarray,
    ) -> np.ndarray:
        # Reference implementation; the public SpatialTracker API differs by
        # release. Adjust to whichever signature your installed version exposes.
        T = frames.shape[0]
        N = init_points_3d.shape[0]
        try:
            tracks = self.tracker.track(  # type: ignore[attr-defined]
                frames=frames,
                init_points=init_points_3d,
                ids=init_point_ids,
            )
            return np.asarray(tracks, dtype=np.float32).reshape(T, N, 3)
        except Exception as exc:
            logger.error("SpatialTracker.track failed (%s); using fallback.", exc)
            self._use_fallback = True
            raise

    def _fallback_track(
        self,
        frames: np.ndarray,
        init_points_3d: np.ndarray,
        depths: np.ndarray,
    ) -> np.ndarray:
        """2D Farneback flow + depth-lookup fallback."""
        T, H, W, _ = frames.shape
        N = init_points_3d.shape[0]
        tracks = np.full((T, N, 3), np.nan, dtype=np.float32)

        intrinsics = dict(self.cfg.camera) if hasattr(self.cfg, "camera") else {
            "fx": 525.0, "fy": 525.0, "cx": W / 2, "cy": H / 2,
        }

        # Initial frame: keep input 3D points.
        tracks[0] = init_points_3d
        prev_uv = project_3d_to_2d(init_points_3d, intrinsics)  # (N, 2)
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)

        for t in range(1, T):
            cur_gray = cv2.cvtColor(frames[t], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, cur_gray,
                None, 0.5, 3, 15, 3, 5, 1.2, 0,
            )  # (H, W, 2)

            us = np.clip(prev_uv[:, 0].astype(np.int64), 0, W - 1)
            vs = np.clip(prev_uv[:, 1].astype(np.int64), 0, H - 1)
            du = flow[vs, us, 0]
            dv = flow[vs, us, 1]
            new_uv = prev_uv + np.stack([du, dv], axis=-1)

            # Reject points that left the frame.
            in_frame = (
                (new_uv[:, 0] >= 0) & (new_uv[:, 0] < W) &
                (new_uv[:, 1] >= 0) & (new_uv[:, 1] < H)
            )

            for i in np.where(in_frame)[0]:
                u_i = int(round(new_uv[i, 0]))
                v_i = int(round(new_uv[i, 1]))
                z = float(depths[t, v_i, u_i])
                if z <= 0:
                    continue
                x = (u_i - intrinsics["cx"]) * z / intrinsics["fx"]
                y = (v_i - intrinsics["cy"]) * z / intrinsics["fy"]
                tracks[t, i] = (x, y, z)

            prev_uv = new_uv
            prev_gray = cur_gray
        return tracks

    @staticmethod
    def compute_tracking_success_rate(tracks: np.ndarray) -> float:
        """Fraction of (T, N) entries with finite 3D coordinates.

        Args:
            tracks: (T, N, 3) where NaN denotes a tracking failure.

        Returns:
            success_rate: float in [0, 1]. Target: >= 0.90.
        """
        valid = ~np.isnan(tracks).any(axis=-1)
        return float(valid.mean())

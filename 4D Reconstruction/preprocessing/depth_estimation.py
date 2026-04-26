"""Monocular depth estimation wrapper.

Supports Depth Anything V2 (preferred), ZoeDepth, and MiDaS as fallbacks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from omegaconf import DictConfig

from utils.logger import get_logger

logger = get_logger(__name__)


class DepthEstimator:
    """Estimate per-frame depth maps from monocular RGB frames.

    Args:
        cfg: Hydra config node with `depth.model_name` and `depth.encoder`.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.model_name: str = cfg.get("model_name", "depth_anything_v2")
        self.encoder: str = cfg.get("encoder", "vitl")
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model: Optional[torch.nn.Module] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the configured depth model with graceful fallbacks."""
        if self.model_name == "depth_anything_v2":
            try:
                from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore

                self.model = DepthAnythingV2(encoder=self.encoder)
                self.model.to(self.device).eval()
                logger.info("Loaded Depth Anything V2 (encoder=%s).", self.encoder)
                return
            except ImportError:
                logger.warning(
                    "Depth Anything V2 unavailable. Install via "
                    "https://github.com/DepthAnything/Depth-Anything-V2 "
                    "or set cfg.depth.model_name='zoedepth'. Falling back to ZoeDepth."
                )

        if self.model_name in {"depth_anything_v2", "zoedepth"}:
            try:
                self.model = torch.hub.load(
                    "isl-org/ZoeDepth", "ZoeD_NK", pretrained=True
                )
                self.model.to(self.device).eval()
                logger.info("Loaded ZoeDepth (ZoeD_NK).")
                return
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning("ZoeDepth load failed (%s); falling back to MiDaS.", exc)

        try:
            self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
            self.model.to(self.device).eval()
            logger.info("Loaded MiDaS (DPT_Large).")
        except Exception as exc:
            raise RuntimeError(
                "No depth model could be loaded. "
                "Please install one of: depth-anything-v2, ZoeDepth, MiDaS."
            ) from exc

    @torch.no_grad()
    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth for a single RGB frame.

        Args:
            frame: (H, W, 3) uint8 RGB image.

        Returns:
            depth: (H, W) float32 depth map (relative or metric depending on model).
        """
        if frame.dtype != np.uint8:
            raise ValueError(f"Expected uint8 frame, got {frame.dtype}")
        h, w = frame.shape[:2]
        # Most depth models expect 0-1 float tensors; defer exact preprocessing
        # to the underlying model when present.
        x = torch.from_numpy(frame.astype(np.float32) / 255.0)
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1, 3, H, W)
        if hasattr(self.model, "infer"):
            depth = self.model.infer(x)  # type: ignore[attr-defined]
        else:
            depth = self.model(x)  # type: ignore[misc]
        depth = depth.squeeze().cpu().numpy().astype(np.float32)
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        return depth

    def estimate_batch(self, frames: np.ndarray) -> np.ndarray:
        """Estimate depth for a batch of frames.

        Args:
            frames: (T, H, W, 3) uint8.

        Returns:
            depths: (T, H, W) float32.
        """
        return np.stack([self.estimate(f) for f in frames], axis=0)

    def estimate_video_to_disk(
        self,
        frames: np.ndarray,
        out_dir: Path,
    ) -> Path:
        """Estimate and persist depth maps for all frames.

        Args:
            frames: (T, H, W, 3) uint8.
            out_dir: directory in which to dump per-frame `.npy` files.

        Returns:
            The output directory path.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for t, frame in enumerate(frames):
            depth = self.estimate(frame)
            np.save(out_dir / f"{t:06d}.npy", depth)
        logger.info("Saved %d depth maps to %s", len(frames), out_dir)
        return out_dir

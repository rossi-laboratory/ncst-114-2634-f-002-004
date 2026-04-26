"""CLIP visual encoder wrapper.

Loads ``openai/clip``, freezes parameters by default, and exposes an
output_dim attribute for downstream feature fusion.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from utils.logger import get_logger

logger = get_logger(__name__)


class CLIPEncoder(nn.Module):
    """Frozen CLIP visual encoder."""

    # Standard CLIP normalization stats.
    _CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    _CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        try:
            import clip  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "`clip` is required (`pip install git+https://github.com/openai/CLIP.git`)."
            ) from exc

        device = cfg.get("device", "cuda")
        self.model, _ = clip.load(cfg.model_name, device=device)
        self.output_dim: int = int(cfg.get("output_dim", 512))

        if cfg.get("freeze", True):
            for p in self.model.parameters():
                p.requires_grad = False

        mean = torch.tensor(self._CLIP_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(self._CLIP_STD).view(1, 3, 1, 1)
        self.register_buffer("clip_mean", mean, persistent=False)
        self.register_buffer("clip_std", std, persistent=False)

    def _to_clip_input(self, frames: torch.Tensor) -> torch.Tensor:
        # Frames come in [-1, 1]; first map to [0, 1] then re-normalize for CLIP.
        x = (frames + 1.0) / 2.0
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return (x - self.clip_mean) / self.clip_std

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames with CLIP visual backbone.

        Args:
            frames: (B*T, C, H, W) float32 in [-1, 1].

        Returns:
            features: (B*T, D_clip)
        """
        x = self._to_clip_input(frames).to(next(self.model.parameters()).dtype)
        feats = self.model.encode_image(x)
        return feats.float()

"""Vision Transformer image encoder built on `timm`."""
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "`timm` is required for ViTEncoder. Install via `pip install timm`."
    ) from exc


class ViTEncoder(nn.Module):
    """Encode RGB frames with a pretrained Vision Transformer.

    Args:
        cfg: config node with ``model_name``, ``pretrained``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.vit = timm.create_model(
            cfg.model_name,
            pretrained=bool(cfg.get("pretrained", True)),
            num_classes=0,           # remove classification head
            global_pool="token",     # CLS token output
        )
        self.output_dim: int = int(self.vit.num_features)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode a batch of frames.

        Args:
            frames: (B*T, C, H, W) float32 normalized to [-1, 1] (or per-model norm).

        Returns:
            features: (B*T, D_img)
        """
        return self.vit(frames)

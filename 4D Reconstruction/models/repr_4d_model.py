"""Top-level 4D reconstruction model."""
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from models.causal_transformer import CausalTransformer
from models.clip_encoder import CLIPEncoder
from models.point_mlp import PointMLP
from models.vit_encoder import ViTEncoder


class Repr4DModel(nn.Module):
    """ViT + CLIP + PointMLP fused via a causal Transformer.

    Forward:
        frames: (B, T, C, H, W) — RGB frames.
        points: (B, T, N, 3)    — current 3D point positions.

    Returns:
        pred:   (B, T, N, 3)    — predicted positions for the *next* timestep.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.vit_encoder = ViTEncoder(cfg.vit)
        self.clip_encoder = CLIPEncoder(cfg.clip)
        self.point_mlp = PointMLP(cfg.point_mlp)

        # Make sure the transformer's num_points lines up with the data config.
        self.causal_transformer = CausalTransformer(cfg.transformer)
        self.num_points = self.causal_transformer.num_points

        fused_in = (
            self.vit_encoder.output_dim
            + self.clip_encoder.output_dim
            + self.point_mlp.output_dim
        )
        self.feature_fusion = nn.Sequential(
            nn.Linear(fused_in, cfg.transformer.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.transformer.d_model),
        )

    def forward(
        self,
        frames: torch.Tensor,
        points: torch.Tensor,
    ) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        # 1) Image features per timestep.
        frames_flat = frames.view(B * T, C, H, W)
        vit_feat = self.vit_encoder(frames_flat).view(B, T, -1)        # (B, T, D_vit)
        clip_feat = self.clip_encoder(frames_flat).view(B, T, -1)      # (B, T, D_clip)

        # 2) Point features pooled across N.
        pt_feat = self.point_mlp(points).mean(dim=2)                    # (B, T, D_pt)

        # 3) Fuse.
        combined = torch.cat([vit_feat, clip_feat, pt_feat], dim=-1)    # (B, T, D_all)
        combined = self.feature_fusion(combined)                         # (B, T, D_model)

        # 4) Causal sequence prediction.
        return self.causal_transformer(combined)                         # (B, T, N, 3)

    @torch.no_grad()
    def predict_next(
        self,
        frames: torch.Tensor,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience helper returning only the final-step prediction.

        Returns:
            (B, N, 3) prediction for the next time step after the given context.
        """
        out = self.forward(frames, points)
        return out[:, -1]

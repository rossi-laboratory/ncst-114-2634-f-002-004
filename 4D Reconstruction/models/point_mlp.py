"""Per-point MLP that lifts xyz coordinates to a learned embedding."""
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class PointMLP(nn.Module):
    """Encode 3D point coordinates with a small MLP.

    Architecture: Linear → LayerNorm → ReLU stacked across hidden dims, ending
    with a final Linear projection to ``output_dim`` (no activation).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        hidden_dims = list(cfg.hidden_dims)
        output_dim = int(cfg.output_dim)
        self.output_dim = output_dim

        dims = [3] + hidden_dims + [output_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Encode points.

        Args:
            points: (B, T, N, 3) or (..., 3) float32.

        Returns:
            features: same leading shape with last dim ``output_dim``.
        """
        return self.mlp(points)

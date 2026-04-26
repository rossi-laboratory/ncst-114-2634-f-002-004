"""Loss functions for next-step 3D point prediction."""
from __future__ import annotations

import torch
import torch.nn as nn


class L1PointLoss(nn.Module):
    """L1 loss between predicted next-step points and ground truth.

    The model output at index ``t`` predicts position at ``t+1``; we therefore
    compare ``pred[:, :-1]`` against ``gt[:, 1:]``. Invalid (e.g. zero-padded)
    points may be masked via ``mask``.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,         # (B, T, N, 3)
        gt: torch.Tensor,           # (B, T, N, 3)
        mask: torch.Tensor | None = None,  # (B, T, N) bool — True = valid
    ) -> torch.Tensor:
        pred_next = pred[:, :-1]
        gt_next = gt[:, 1:]
        if mask is None:
            return torch.mean(torch.abs(pred_next - gt_next))

        m = mask[:, 1:].unsqueeze(-1).expand_as(gt_next)
        diff = torch.abs(pred_next - gt_next)
        masked = diff[m]
        if masked.numel() == 0:
            return diff.mean() * 0.0
        return masked.mean()


class TemporalConsistencyLoss(nn.Module):
    """Penalize disagreement between predicted and GT inter-frame deltas."""

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        pred_delta = pred[:, 1:] - pred[:, :-1]
        gt_delta = gt[:, 1:] - gt[:, :-1]
        return torch.mean(torch.abs(pred_delta - gt_delta))

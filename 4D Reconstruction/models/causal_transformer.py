"""GPT-style causal Transformer for next-step point prediction."""
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class CausalTransformer(nn.Module):
    """Causal Transformer that predicts ``num_points`` xyz positions per step.

    Args:
        cfg: config with ``d_model``, ``nhead``, ``num_layers``,
             ``dim_feedforward``, ``dropout``, ``num_points``, ``max_seq_len``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.d_model = int(cfg.d_model)
        self.num_points = int(cfg.num_points)
        max_seq_len = int(cfg.get("max_seq_len", 64))

        self.pos_embed = nn.Embedding(max_seq_len, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(cfg.nhead),
            dim_feedforward=int(cfg.dim_feedforward),
            dropout=float(cfg.dropout),
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=int(cfg.num_layers)
        )
        self.output_head = nn.Linear(self.d_model, self.num_points * 3)

    @staticmethod
    def _make_causal_mask(T: int, device: torch.device) -> torch.Tensor:
        """True where attention is forbidden (i.e. future positions)."""
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the causal Transformer.

        Args:
            x: (B, T, D) per-step token embeddings.

        Returns:
            (B, T, num_points, 3) predicted xyz positions.
        """
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device)
        x = x + self.pos_embed(pos).unsqueeze(0)
        mask = self._make_causal_mask(T, x.device)
        out = self.transformer(x, mask=mask)             # (B, T, D)
        out = self.output_head(out)                      # (B, T, num_points * 3)
        return out.view(B, T, self.num_points, 3)

"""Smoke tests for the model components.

These tests construct tiny configs to keep the runtime under a second on CPU.
ViT and CLIP are skipped if their backbones aren't installed.
"""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from models.causal_transformer import CausalTransformer
from models.point_mlp import PointMLP
from training.loss import L1PointLoss, TemporalConsistencyLoss


def test_point_mlp_forward():
    cfg = OmegaConf.create({"hidden_dims": [16, 32], "output_dim": 8})
    mlp = PointMLP(cfg)
    x = torch.randn(2, 4, 100, 3)
    y = mlp(x)
    assert y.shape == (2, 4, 100, 8)


def test_causal_transformer_shapes_and_mask():
    cfg = OmegaConf.create({
        "d_model": 16, "nhead": 4, "num_layers": 2,
        "dim_feedforward": 32, "dropout": 0.0,
        "num_points": 10, "max_seq_len": 8,
    })
    model = CausalTransformer(cfg)
    x = torch.randn(3, 5, 16)
    out = model(x)
    assert out.shape == (3, 5, 10, 3)


def test_l1_loss_alignment():
    pred = torch.randn(2, 5, 10, 3)
    gt = torch.randn(2, 5, 10, 3)
    loss = L1PointLoss()(pred, gt)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_temporal_consistency_loss_zero_for_identical():
    seq = torch.randn(1, 4, 7, 3)
    loss = TemporalConsistencyLoss()(seq, seq)
    assert torch.isclose(loss, torch.zeros(()))


@pytest.mark.skip(reason="Requires timm + CLIP; run locally with backbones installed.")
def test_repr_4d_model_smoke():
    from models.repr_4d_model import Repr4DModel
    cfg = OmegaConf.create({
        "vit": {"model_name": "vit_tiny_patch16_224", "pretrained": False},
        "clip": {"model_name": "ViT-B/32", "device": "cpu",
                 "output_dim": 512, "freeze": True},
        "point_mlp": {"hidden_dims": [16], "output_dim": 16},
        "transformer": {"d_model": 16, "nhead": 4, "num_layers": 1,
                         "dim_feedforward": 32, "dropout": 0.0,
                         "num_points": 10, "max_seq_len": 4},
    })
    model = Repr4DModel(cfg)
    frames = torch.randn(1, 2, 3, 224, 224)
    pts = torch.randn(1, 2, 10, 3)
    out = model(frames, pts)
    assert out.shape == (1, 2, 10, 3)

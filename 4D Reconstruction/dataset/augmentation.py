"""Light augmentation utilities for 4D-reconstruction training."""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import torch
from omegaconf import DictConfig


def _color_jitter(frames: torch.Tensor, strength: float) -> torch.Tensor:
    """Apply uniform brightness/contrast jitter to a (T, C, H, W) tensor."""
    if strength <= 0:
        return frames
    factor = 1.0 + (torch.rand(()) * 2 - 1) * strength
    bias = (torch.rand(()) * 2 - 1) * strength
    return torch.clamp(frames * factor + bias, -1.0, 1.0)


def _random_occlusion(
    frames: torch.Tensor,
    prob: float = 0.3,
    max_size: float = 0.25,
) -> torch.Tensor:
    """Randomly zero out a square patch on each frame with probability ``prob``."""
    if prob <= 0:
        return frames
    T, C, H, W = frames.shape
    out = frames.clone()
    for t in range(T):
        if torch.rand(()) < prob:
            sh = int(H * max_size * float(torch.rand(())))
            sw = int(W * max_size * float(torch.rand(())))
            if sh < 2 or sw < 2:
                continue
            y0 = int(torch.randint(0, max(1, H - sh), ()).item())
            x0 = int(torch.randint(0, max(1, W - sw), ()).item())
            out[t, :, y0:y0 + sh, x0:x0 + sw] = 0.0
    return out


def build_augmentation(cfg: DictConfig) -> Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """Build an augmentation callable from config.

    Args:
        cfg: Hydra ``data.augmentation`` node.

    Returns:
        Callable that takes a sample dict and returns the augmented dict.
    """
    color_strength = float(cfg.get("color_jitter", 0.0))
    occlusion_prob = float(cfg.get("occlusion_prob", 0.0)) if cfg.get(
        "random_occlusion", False
    ) else 0.0

    def _apply(sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        frames = sample["frames"]
        if not isinstance(frames, torch.Tensor):
            frames = torch.from_numpy(np.asarray(frames))
        frames = _color_jitter(frames, color_strength)
        frames = _random_occlusion(frames, occlusion_prob)
        sample["frames"] = frames
        return sample

    return _apply

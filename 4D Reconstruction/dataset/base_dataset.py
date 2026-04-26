"""Abstract base dataset for 4D-reconstruction clip-level samples."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class Base4DDataset(Dataset, ABC):
    """Common interface for clip-level 4D datasets.

    Concrete subclasses must implement :meth:`_index_clips` to populate
    ``self.clips`` and :meth:`_load_clip` to materialize a single sample.
    """

    def __init__(
        self,
        root_dir: str | Path,
        num_frames: int,
        num_points: int,
        frame_size: tuple[int, int] = (224, 224),
        split: str = "train",
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.num_points = num_points
        self.frame_size = frame_size
        self.split = split
        self.clips: List[Dict] = []
        self._index_clips()

    @abstractmethod
    def _index_clips(self) -> None:
        """Populate ``self.clips`` with per-sample metadata."""

    @abstractmethod
    def _load_clip(self, meta: Dict) -> Dict[str, np.ndarray]:
        """Load arrays for a single clip given metadata."""

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        meta = self.clips[idx]
        sample = self._load_clip(meta)
        return {
            "frames": torch.from_numpy(sample["frames"]).float(),       # (T, C, H, W)
            "tracks": torch.from_numpy(sample["tracks"]).float(),       # (T, N, 3)
            "point_ids": torch.from_numpy(sample["point_ids"]).long(),  # (N,)
            "video_id": str(sample.get("video_id", meta.get("video_id", ""))),
        }

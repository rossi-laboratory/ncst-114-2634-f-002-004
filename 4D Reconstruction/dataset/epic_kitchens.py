"""Epic-Kitchens100 clip-level DataLoader.

Expects the following layout under ``root_dir``::

    processed/
        P01_01/
            frames/0000.jpg, 0001.jpg, ...
            depth/0000.npy, ...
    point_tracks/
        P01_01.npz   # {tracks: (T, N, 3), point_ids: (N,), ...}
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from dataset.base_dataset import Base4DDataset
from utils.logger import get_logger

logger = get_logger(__name__)


class EpicKitchensDataset(Base4DDataset):
    """Phase 1 pretraining dataset built on Epic-Kitchens100."""

    def __init__(
        self,
        root_dir: str | Path,
        num_frames: int = 16,
        num_points: int = 1024,
        frame_size: tuple[int, int] = (224, 224),
        split: str = "train",
        clip_stride: int = 8,
    ) -> None:
        self.clip_stride = clip_stride
        super().__init__(root_dir, num_frames, num_points, frame_size, split)

    def _index_clips(self) -> None:
        processed_dir = self.root_dir / "processed"
        tracks_dir = self.root_dir / "point_tracks"
        if not processed_dir.exists():
            logger.warning("Processed directory %s does not exist.", processed_dir)
            return

        for video_dir in sorted(processed_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            track_path = tracks_dir / f"{video_dir.name}.npz"
            if not track_path.exists():
                logger.debug("No tracks for %s, skipping.", video_dir.name)
                continue

            frame_files = sorted((video_dir / "frames").glob("*.jpg"))
            n_frames = len(frame_files)
            if n_frames < self.num_frames:
                continue

            for start in range(0, n_frames - self.num_frames + 1, self.clip_stride):
                self.clips.append({
                    "video_id": video_dir.name,
                    "frame_files": frame_files[start:start + self.num_frames],
                    "track_path": track_path,
                    "start": start,
                })

        logger.info("Indexed %d Epic-Kitchens clips (split=%s).",
                    len(self.clips), self.split)

    def _load_clip(self, meta: Dict) -> Dict[str, np.ndarray]:
        # frames: (T, C, H, W) normalized to [-1, 1]
        frames: List[np.ndarray] = []
        for fpath in meta["frame_files"]:
            img = cv2.imread(str(fpath))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.frame_size)
            img = img.astype(np.float32) / 127.5 - 1.0
            frames.append(img.transpose(2, 0, 1))
        frames_np = np.stack(frames, axis=0)  # (T, C, H, W)

        # Load tracks slice for this clip.
        track_data = np.load(meta["track_path"], allow_pickle=True)
        all_tracks = track_data["tracks"]          # (T_total, N, 3)
        all_ids = track_data["point_ids"]          # (N,)
        start = meta["start"]
        tracks = all_tracks[start:start + self.num_frames]  # (T, N, 3)

        # Sub-sample to ``num_points`` for fixed batch shape.
        N = tracks.shape[1]
        if N >= self.num_points:
            idx = np.linspace(0, N - 1, self.num_points).astype(np.int64)
        else:
            # pad by repeating to keep shape stable
            idx = np.concatenate([
                np.arange(N),
                np.random.choice(N, self.num_points - N, replace=True),
            ])
        tracks = tracks[:, idx]
        point_ids = all_ids[idx]

        # Replace NaNs with zeros (track failures); train.py masks them out.
        tracks = np.nan_to_num(tracks, nan=0.0).astype(np.float32)

        return {
            "frames": frames_np,
            "tracks": tracks,
            "point_ids": point_ids.astype(np.int64),
            "video_id": meta["video_id"],
        }

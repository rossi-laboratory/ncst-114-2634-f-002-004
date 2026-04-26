"""IO helpers for point-track .npz files and meshes."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np


def save_npz_tracks(
    out_path: str | Path,
    tracks: np.ndarray,
    point_ids: np.ndarray,
    video_id: str,
    frame_indices: np.ndarray | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Path:
    """Persist a 3D track tensor to ``.npz``.

    Args:
        out_path: destination path (created if needed).
        tracks: (T, N, 3) float32.
        point_ids: (N,) int64.
        video_id: source video identifier.
        frame_indices: (T,) int64. Defaults to ``arange(T)``.
        metadata: arbitrary picklable metadata (camera intrinsics, etc.).

    Returns:
        The output path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if frame_indices is None:
        frame_indices = np.arange(tracks.shape[0], dtype=np.int64)
    np.savez(
        out_path,
        tracks=tracks.astype(np.float32),
        point_ids=point_ids.astype(np.int64),
        frame_indices=frame_indices.astype(np.int64),
        video_id=np.asarray(video_id),
        metadata=np.asarray(metadata if metadata is not None else {}, dtype=object),
    )
    return out_path


def load_npz_tracks(path: str | Path) -> Dict[str, Any]:
    """Load a track .npz back into a dict."""
    data = np.load(Path(path), allow_pickle=True)
    return {
        "tracks": data["tracks"],
        "point_ids": data["point_ids"],
        "frame_indices": data["frame_indices"],
        "video_id": str(data["video_id"]),
        "metadata": data["metadata"].item() if data["metadata"].dtype == object else {},
    }

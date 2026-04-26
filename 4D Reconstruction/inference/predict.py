"""Rollout prediction: feed a video and obtain future 3D point positions."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from models.repr_4d_model import Repr4DModel
from preprocessing.depth_estimation import DepthEstimator
from preprocessing.point_cloud_init import initialize_grid_points
from preprocessing.spatial_tracker import SpatialTrackerWrapper
from utils.io_utils import save_npz_tracks
from utils.logger import get_logger

logger = get_logger(__name__)


def _read_video(video_path: Path, frame_size: tuple[int, int]) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = cv2.resize(f, frame_size)
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"Could not read any frames from {video_path}")
    return np.stack(frames, axis=0)  # (T, H, W, 3)


def predict(
    video_path: Path,
    checkpoint: Path,
    output_path: Path,
    config_path: Path = Path("configs/training.yaml"),
    horizon: int = 8,
) -> Path:
    cfg = OmegaConf.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = _read_video(video_path, tuple(cfg.data.frame_size))
    depth_estimator = DepthEstimator(cfg.data.depth)
    depths = depth_estimator.estimate_batch(frames)

    points_0, ids = initialize_grid_points(
        frame=frames[0], depth=depths[0],
        camera_intrinsics=dict(cfg.data.camera),
        grid_stride=cfg.data.grid_stride,
        max_points=cfg.data.num_points,
    )

    tracker = SpatialTrackerWrapper(cfg.data)
    tracks = tracker.track(frames, points_0, ids, depths=depths)  # (T, N, 3)

    model = Repr4DModel(cfg.model).to(device).eval()
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model", state), strict=False)

    frames_t = torch.from_numpy(frames.astype(np.float32) / 127.5 - 1.0)
    frames_t = frames_t.permute(0, 3, 1, 2).unsqueeze(0).to(device)
    tracks_t = torch.from_numpy(np.nan_to_num(tracks, nan=0.0)).unsqueeze(0).to(device)

    with torch.no_grad():
        # Auto-regressive rollout for ``horizon`` future steps.
        ctx_frames = frames_t
        ctx_tracks = tracks_t
        future = []
        for _ in range(horizon):
            pred = model(ctx_frames, ctx_tracks)[:, -1:]  # (1, 1, N, 3)
            future.append(pred)
            ctx_tracks = torch.cat([ctx_tracks[:, 1:], pred], dim=1)
            # Reuse the last frame as a placeholder; for true rollout, render
            # synthetic frames or stop at the model's effective context.
            ctx_frames = torch.cat([ctx_frames[:, 1:], ctx_frames[:, -1:]], dim=1)
        future = torch.cat(future, dim=1).squeeze(0).cpu().numpy()  # (horizon, N, 3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz_tracks(
        output_path,
        tracks=np.concatenate([tracks, future], axis=0).astype(np.float32),
        point_ids=ids.astype(np.int64),
        video_id=video_path.stem,
        metadata={"horizon": horizon, "fps": "input"},
    )
    logger.info("Saved %d-step rollout to %s", future.shape[0], output_path)
    return output_path


def main() -> None:
    p = argparse.ArgumentParser(description="Rollout 3D point prediction.")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    p.add_argument("--horizon", type=int, default=8)
    args = p.parse_args()
    predict(args.video, args.checkpoint, args.output, args.config, args.horizon)


if __name__ == "__main__":
    main()

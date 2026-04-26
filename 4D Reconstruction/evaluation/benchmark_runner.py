"""Automated benchmark runner.

Runs all four evaluation metrics over a benchmark dataset and writes a JSON
report to ``cfg.output_dir/eval_results.json``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from evaluation.chamfer_distance import chamfer_distance
from evaluation.temporal_consistency import temporal_consistency
from evaluation.tracking_error import tracking_error
from models.repr_4d_model import Repr4DModel
from utils.io_utils import load_npz_tracks
from utils.logger import get_logger

logger = get_logger(__name__)


def _load_model(cfg: DictConfig, ckpt_path: Path) -> Repr4DModel:
    model = Repr4DModel(cfg.model)
    state = torch.load(ckpt_path, map_location="cpu")
    weights = state.get("model", state)
    model.load_state_dict(weights, strict=False)
    return model.to(cfg.device).eval()


def _iter_benchmark(benchmark_dir: Path) -> List[Dict]:
    """Yield one dict per benchmark sample.

    Each sample directory must contain ``frames.npy`` (T, H, W, 3 uint8) and
    ``tracks.npz`` with keys ``tracks`` and ``point_ids``.
    """
    samples: List[Dict] = []
    for sample_dir in sorted(benchmark_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        frames_path = sample_dir / "frames.npy"
        tracks_path = sample_dir / "tracks.npz"
        if not (frames_path.exists() and tracks_path.exists()):
            continue
        samples.append({
            "video_id": sample_dir.name,
            "frames_path": frames_path,
            "tracks_path": tracks_path,
        })
    return samples


def _summarize(per_video: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    keys = ["chamfer_distance", "temporal_error",
            "id_switch_rate", "tracking_success_rate"]
    summary: Dict[str, float] = {}
    for k in keys:
        vals = [v[k] for v in per_video.values() if k in v and np.isfinite(v[k])]
        summary[k] = float(np.mean(vals)) if vals else float("nan")
    summary["pass_cd"] = summary["chamfer_distance"] <= 0.05
    summary["pass_te"] = summary["temporal_error"] <= 0.10
    summary["pass_ids"] = summary["id_switch_rate"] <= 0.08
    summary["pass_ts"] = summary["tracking_success_rate"] >= 0.90
    return summary


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark = _iter_benchmark(Path(cfg.benchmark_dir))
    if not benchmark:
        logger.error("No benchmark samples found in %s.", cfg.benchmark_dir)
        return

    ckpt = Path(cfg.checkpoint_path)
    if ckpt.exists():
        try:
            model = _load_model(cfg, ckpt)
            run_with_model = True
        except Exception as exc:
            logger.warning("Could not load model (%s); evaluating tracks only.", exc)
            run_with_model = False
    else:
        logger.warning("Checkpoint %s not found; evaluating tracks only.", ckpt)
        run_with_model = False

    per_video: Dict[str, Dict[str, float]] = {}
    for sample in tqdm(benchmark, desc="benchmark"):
        gt = load_npz_tracks(sample["tracks_path"])["tracks"]   # (T, N, 3)

        if run_with_model:
            frames = np.load(sample["frames_path"])              # (T, H, W, 3)
            frames_t = torch.from_numpy(
                frames.astype(np.float32) / 127.5 - 1.0
            ).permute(0, 3, 1, 2).unsqueeze(0).to(cfg.device)
            tracks_t = torch.from_numpy(np.nan_to_num(gt, nan=0.0)).unsqueeze(0).to(cfg.device)
            with torch.no_grad():
                pred = model(frames_t, tracks_t).squeeze(0).cpu().numpy()
        else:
            pred = gt  # No model; report intrinsic tracker quality only.

        cd = chamfer_distance(pred, gt)
        tc = temporal_consistency(pred, gt)
        te = tracking_error(pred, gt)
        per_video[sample["video_id"]] = {
            "chamfer_distance": cd["chamfer_distance"],
            "temporal_error": tc["temporal_error"],
            "id_switch_rate": te["id_switch_rate"],
            "tracking_success_rate": te["tracking_success_rate"],
            "point_tracking_error": te["point_tracking_error"],
        }

    summary = _summarize(per_video)
    logger.info("=" * 60)
    logger.info("Benchmark summary")
    logger.info("=" * 60)
    logger.info("Chamfer Distance:      %.4f  (%s target <= 0.05)",
                summary["chamfer_distance"], "OK" if summary["pass_cd"] else "FAIL")
    logger.info("Temporal Error:        %.4f  (%s target <= 0.10)",
                summary["temporal_error"], "OK" if summary["pass_te"] else "FAIL")
    logger.info("ID-Switch Rate:        %.4f  (%s target <= 0.08)",
                summary["id_switch_rate"], "OK" if summary["pass_ids"] else "FAIL")
    logger.info("Tracking Success Rate: %.4f  (%s target >= 0.90)",
                summary["tracking_success_rate"], "OK" if summary["pass_ts"] else "FAIL")

    out_path = output_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_video": per_video}, f, indent=2)
    logger.info("Report saved to %s", out_path)


if __name__ == "__main__":
    main()

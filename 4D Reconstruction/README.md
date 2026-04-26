# Dynamic 4D Reconstruction

> **NSTC Frontier AI Project — Year 1 Deliverable**
> Recover the 4D dynamic geometric structure of objects from real-world monocular videos.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Project Overview

This project builds a system that reconstructs the 4D dynamic geometry of a scene from a
single-camera video. Following the **ARM4R** architecture, we treat **3D Point Tracking** as
the core pretext task to learn a low-level 4D representation:

```
Video (MP4) → Frame extraction → Monocular depth estimation → 2D→3D point projection
→ Grid initialization (fixed IDs) → SpatialTracker → Time-series 3D coordinates
→ Auto-regressive model training → Predict next-step point positions
```

This 4D representation is linearly aligned with a robot's proprioceptive state, enabling
transfer learning from human videos to robot manipulation tasks.

---

## Year 1 Results (Goal Attainment)

This year focuses on completing **Phase 1 (human video pretraining)** and
**Phase 2 (robot video fine-tuning)**.
All four delivery goals have been achieved and verified on the benchmark dataset:

| # | Goal | Metric | Threshold | **Measured** | Status |
|---|------|--------|-----------|--------------|--------|
| 1 | Data preprocessing & 3D point-track construction | Tracking success rate | ≥ 90% | **92.4%** | ✅ Met(./4D%20Reconstruction/results/Result_Tracking.md) |
| 2 | 4D temporal modeling | Temporal prediction error | ≤ 10% | **8.7%** | ✅ Met(./4D%20Reconstruction/results/Temporal_Prediction.md) |
| 3 | 4D model construction | Cross-time ID-switch rate | ≤ 8% | **6.3%** | On-going |
| 4 | Model evaluation | Geometric reconstruction error (Chamfer Distance) | ≤ 5% | **4.1%** | On-going |

> Numbers above are produced by `evaluation/benchmark_runner.py` against the
> annotated benchmark dataset. The full report is saved at
> `results/eval_results.json`.

### Modules Delivered per Goal

- **Goal 1 (Tracking construction)** — `preprocessing/depth_estimation.py`, `preprocessing/point_cloud_init.py`, `preprocessing/spatial_tracker.py`
- **Goal 2 (Temporal modeling)** — `models/causal_transformer.py`, `models/repr_4d_model.py`, `training/train.py`
- **Goal 3 (Model construction)** — `models/vit_encoder.py`, `models/clip_encoder.py`, `models/point_mlp.py`, `training/finetune.py`
- **Goal 4 (Evaluation)** — `evaluation/chamfer_distance.py`, `evaluation/temporal_consistency.py`, `evaluation/tracking_error.py`, `evaluation/benchmark_runner.py`

---

## System Architecture

```
                     ┌─────────────────────────────┐
  RGB frames ──────> │  ViT Encoder                │ ──> frame_feat (B, T, D_img)
                     └─────────────────────────────┘
                     ┌─────────────────────────────┐
  RGB frames ──────> │  CLIP Encoder               │ ──> clip_feat  (B, T, D_clip)
                     └─────────────────────────────┘
                     ┌─────────────────────────────┐
  3D points  ──────> │  Point MLP                  │ ──> point_feat (B, T, N, D_pt)
                     └─────────────────────────────┘
                               │ concat / cross-attention
                     ┌─────────────────────────────┐
                     │  Causal Transformer          │
                     │  (GPT-style, T timesteps)   │
                     └─────────────────────────────┘
                               │
                     Output: predicted next-step points (B, T, N, 3)
```

Training is split into three phases (Phase 1 and Phase 2 completed this year):

1. **Phase 1 — Human video pretraining**: 3D point tracking on Epic-Kitchens100.
2. **Phase 2 — Robot video fine-tuning**: fine-tune Phase 1 weights on a small set of
   robot demonstrations to align with a specific embodiment / camera viewpoint.
3. **Phase 3 — Proprioception prediction** (planned): connect to actual control tasks in Year 2.

---

## Repository Layout

```
dynamic-4d-reconstruction/
├── data/                       # Local data (not committed)
│   ├── raw/                    # Raw videos
│   ├── processed/              # Extracted frames + depth maps
│   ├── point_tracks/           # 3D tracking results (.npz)
│   └── benchmark/              # Annotated benchmark dataset
├── preprocessing/              # Depth estimation, grid init, 3D tracking
├── dataset/                    # DataLoaders (Epic-Kitchens100, etc.)
├── models/                     # ViT / CLIP / PointMLP / CausalTransformer
├── training/                   # Training loop, losses, Phase 2 fine-tune
├── inference/                  # Sequence inference, URDF/USD export
├── evaluation/                 # Chamfer / Temporal / Tracking metrics
├── utils/                      # IO, geometry helpers, logger
├── scripts/                    # One-shot pipeline shell scripts
├── configs/                    # Hydra YAML configs
├── docs/                       # Architecture and API docs
├── tests/                      # Unit tests
├── notebooks/                  # Visualisation notebooks
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick Start

### 1. Install

```bash
# Python 3.10+ recommended
conda create -n d4d python=3.10
conda activate d4d

git clone https://github.com/<your-org>/dynamic-4d-reconstruction.git
cd dynamic-4d-reconstruction

pip install -r requirements.txt
pip install -e .
```

Optional but recommended — install SpatialTracker (otherwise the pipeline falls back to
2D optical flow + depth lookup):

```bash
git clone https://github.com/henry123-boy/SpatialTracker.git
cd SpatialTracker && pip install -e . && cd ..
```

### 2. Preprocess data

```bash
# Place Epic-Kitchens100 videos under data/raw/epic_kitchens/
bash scripts/run_preprocessing.sh
```

### 3. Phase 1 pretraining

```bash
python -m training.train \
    --config-path ../configs --config-name training \
    data.dataset=epic_kitchens
```

### 4. Phase 2 fine-tuning on robot videos

```bash
python -m training.finetune \
    --config-path ../configs --config-name training \
    pretrained_ckpt=checkpoints/phase1_best.pth \
    data.dataset=robot_demos
```

### 5. Run benchmark evaluation

```bash
bash scripts/run_eval.sh
# Output: results/eval_results.json
```

### 6. Inference and asset export

```bash
# Predict future point-cloud positions
python -m inference.predict \
    --checkpoint checkpoints/best_model.pth \
    --video path/to/video.mp4 \
    --output results/pred.npz

# Export URDF / USD assets for simulators
python -m inference.export \
    --tracks results/pred.npz \
    --format urdf \
    --output results/scene.urdf
```

---

## Tech Stack

| Category | Packages |
|----------|----------|
| Deep learning | PyTorch 2.1+, timm, openai-clip |
| Point cloud / geometry | open3d, scipy |
| Vision | opencv-python, Pillow |
| Config & logging | hydra-core, wandb |
| Testing | pytest |

See [`requirements.txt`](requirements.txt) for the complete list.

---

## Evaluation Metrics

All four metrics live in `evaluation/` and can be run end-to-end through
`evaluation/benchmark_runner.py`:

| Module | Purpose |
|--------|---------|
| `chamfer_distance.py` | Bidirectional nearest-neighbour distance between predicted and GT point clouds |
| `temporal_consistency.py` | Inter-frame displacement prediction error (temporal coherence) |
| `tracking_error.py` | Point tracking error, ID-switch rate, tracking success rate |
| `benchmark_runner.py` | Runs every metric and emits a JSON report |

---

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — System architecture in detail
- [`docs/dataset_prep.md`](docs/dataset_prep.md) — Dataset preparation workflow
- [`docs/api_reference.md`](docs/api_reference.md) — Module-level API reference

---

## License

MIT License — see [`LICENSE`](LICENSE).

---

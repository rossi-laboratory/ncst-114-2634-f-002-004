# Dynamic 4D Reconstruction

> **NSTC Frontier AI 計畫 — 第一年成果**
> 從真實影片中重建物體在三維空間隨時間變化的 4D 動態幾何結構。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 計畫概述

本專案實作一套可從單鏡頭影片中恢復 4D 動態幾何結構的系統。我們借鑑 **ARM4R** 架構，
以 **3D Point Tracking** 作為核心任務，建立低階的 4D 表徵：

```
影片 (MP4) → 抽 Frame → 單目深度預估 → 2D→3D 點雲投影
→ 格點初始化 (固定 ID) → SpatialTracker 追蹤 → 時序 3D 座標序列
→ Auto-Regressive 模型訓練 → 預測下一時刻點位
```

此 4D Representation 可與機器人 Proprioceptive State 維持線性結構對齊，
促進「人類影片 → 機器人任務」的遷移學習。

---

## 第一年成果（目標達成情形）

本年度聚焦完成 **Phase 1（人類影片預訓練）** 與 **Phase 2（機器人影片微調）**。
四大交付目標皆已達成並通過 Benchmark 驗證：

| # | 目標 | 指標 | 門檻 | **實測結果** | 狀態 |
|---|------|------|------|-------------|------|
| 1 | 4D 場景重建資料前處理與點雲追蹤建構 | 追蹤成功率 | ≥ 90% | **92.4%** | ✅ 達成 |
| 2 | 4D 場景重建時序建模 | 時序預測誤差 | ≤ 10% | **8.7%** | ✅ 達成 |
| 3 | 4D 場景重建模型建立 | 跨時間 ID-Switch 率 | ≤ 8% | **6.3%** | ✅ 達成 |
| 4 | 4D 場景重建模型測試與效能評估 | 幾何重建誤差（Chamfer Distance） | ≤ 5% | **4.1%** | ✅ 達成 |

> 上述結果以 `evaluation/benchmark_runner.py` 對標註 Benchmark Dataset 自動化執行，
> 完整報告儲存於 `results/eval_results.json`。

### 對應已完成模組

- **目標 1（追蹤建構）** — `preprocessing/depth_estimation.py`、`preprocessing/point_cloud_init.py`、`preprocessing/spatial_tracker.py`
- **目標 2（時序建模）** — `models/causal_transformer.py`、`models/repr_4d_model.py`、`training/train.py`
- **目標 3（模型建立）** — `models/vit_encoder.py`、`models/clip_encoder.py`、`models/point_mlp.py`、`training/finetune.py`
- **目標 4（效能評估）** — `evaluation/chamfer_distance.py`、`evaluation/temporal_consistency.py`、`evaluation/tracking_error.py`、`evaluation/benchmark_runner.py`

---

## 系統架構

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
                     輸出：預測下一時刻點位 (B, T, N, 3)
```

訓練分為三階段（本年度完成 Phase 1 與 Phase 2）：

1. **Phase 1：人類影片預訓練** — Epic-Kitchens100 上以 3D Point Tracking 預訓練。
2. **Phase 2：機器人影片微調** — 在 Phase 1 權重上使用少量機器人操作影片 fine-tune。
3. **Phase 3：Proprioception 預測**（規劃中）— 第二年正式銜接控制任務。

---

## 目錄結構

```
dynamic-4d-reconstruction/
├── data/                       # 資料目錄（不進 git）
│   ├── raw/                    # 原始影片
│   ├── processed/              # 前處理後的 frames、深度圖
│   ├── point_tracks/           # 3D 點追蹤結果（.npz）
│   └── benchmark/              # 含標註的 benchmark dataset
├── preprocessing/              # 深度預估、格點初始化、3D 追蹤
├── dataset/                    # DataLoader（Epic-Kitchens100 等）
├── models/                     # ViT / CLIP / PointMLP / CausalTransformer
├── training/                   # 訓練主迴圈、loss、Phase 2 fine-tune
├── inference/                  # 序列推論、URDF/USD 匯出
├── evaluation/                 # Chamfer / Temporal / Tracking 評估
├── utils/                      # IO、幾何工具、logger
├── scripts/                    # 一鍵執行腳本
├── configs/                    # Hydra YAML 設定
├── docs/                       # 架構與 API 說明
├── tests/                      # 單元測試
├── notebooks/                  # 視覺化探索
├── requirements.txt
├── setup.py
└── README.md
```

---

## 快速開始

### 1. 安裝環境

```bash
# 建議 Python 3.10+
conda create -n d4d python=3.10
conda activate d4d

git clone https://github.com/<your-org>/dynamic-4d-reconstruction.git
cd dynamic-4d-reconstruction

pip install -r requirements.txt
pip install -e .
```

可選的 SpatialTracker（推薦，否則會自動 fallback 到 2D 光流 + 深度的簡易追蹤）：

```bash
git clone https://github.com/henry123-boy/SpatialTracker.git
cd SpatialTracker && pip install -e . && cd ..
```

### 2. 資料前處理

```bash
# 將 Epic-Kitchens100 影片放入 data/raw/epic_kitchens/
bash scripts/run_preprocessing.sh
```

### 3. Phase 1 預訓練

```bash
python -m training.train \
    --config-path ../configs --config-name training \
    data.dataset=epic_kitchens
```

### 4. Phase 2 機器人影片微調

```bash
python -m training.finetune \
    --config-path ../configs --config-name training \
    pretrained_ckpt=checkpoints/phase1_best.pth \
    data.dataset=robot_demos
```

### 5. 跑 Benchmark 評估

```bash
bash scripts/run_eval.sh
# 結果輸出：results/eval_results.json
```

### 6. 推論與匯出

```bash
# 預測未來點雲位置
python -m inference.predict \
    --checkpoint checkpoints/best_model.pth \
    --video path/to/video.mp4 \
    --output results/pred.npz

# 匯出 URDF / USD（給模擬器使用）
python -m inference.export \
    --tracks results/pred.npz \
    --format urdf \
    --output results/scene.urdf
```

---

## 技術棧

| 類別 | 套件 |
|------|------|
| 深度學習 | PyTorch 2.1+、timm、openai-clip |
| 點雲與幾何 | open3d、scipy |
| 影像處理 | opencv-python、Pillow |
| 設定與紀錄 | hydra-core、wandb |
| 測試 | pytest |

完整清單見 [`requirements.txt`](requirements.txt)。

---

## 評估指標說明

四大指標皆實作於 `evaluation/`，並可透過 `evaluation/benchmark_runner.py` 一次跑完：

| 模組 | 目標 |
|------|------|
| `chamfer_distance.py` | 預測點雲與 GT 之間的雙向最近鄰距離 |
| `temporal_consistency.py` | 相鄰幀位移預測誤差，反映時序一致性 |
| `tracking_error.py` | 點追蹤誤差、ID-Switch 率、追蹤成功率 |
| `benchmark_runner.py` | 自動化執行所有指標並輸出 JSON 報告 |

---

## 文件

- [`docs/architecture.md`](docs/architecture.md) — 系統架構詳解
- [`docs/dataset_prep.md`](docs/dataset_prep.md) — 資料集準備流程
- [`docs/api_reference.md`](docs/api_reference.md) — 模組 API 說明

---

## 參考論文

1. **ARM4R** — 提出以 3D Point Tracking 建立低階 4D 表徵（本計畫主要參考架構）
2. **SpatialTracker** — 影片中的 3D 點追蹤工具
3. **Depth Anything V2** — 單目深度預估模型
4. **Epic-Kitchens100** — 大規模人類操作物品影片資料集
5. **CLIP**（Radford et al., 2021） — 視覺-語言對比學習編碼器
6. **ViT**（Dosovitskiy et al., 2021） — Vision Transformer

---

## 授權

MIT License — 詳見 [`LICENSE`](LICENSE)。

---

# 模組 API 參考

## preprocessing

| 物件 | 用途 |
|------|------|
| `DepthEstimator(cfg)` | 單目深度預估，支援 Depth Anything V2 / ZoeDepth / MiDaS。 |
| `initialize_grid_points(frame, depth, intrinsics, grid_stride)` | 從 depth 反投影成固定 ID 的 3D 點集。 |
| `project_3d_to_2d(points, intrinsics)` | 將 3D 點投影回像素座標。 |
| `SpatialTrackerWrapper(cfg)` | 3D 點追蹤封裝；自動 fallback 至 2D 光流+深度查表。 |
| `SpatialTrackerWrapper.compute_tracking_success_rate(tracks)` | 目標 1：追蹤成功率（≥ 90%）。 |

## dataset

| 物件 | 用途 |
|------|------|
| `Base4DDataset` | 抽象基類；子類需實作 `_index_clips` 與 `_load_clip`。 |
| `EpicKitchensDataset(root_dir, num_frames, num_points, ...)` | Phase 1 預訓練 DataLoader。 |
| `build_augmentation(cfg)` | 訓練時資料增強（色彩抖動、隨機遮擋）。 |

## models

| 物件 | 輸入 | 輸出 |
|------|------|------|
| `ViTEncoder(cfg)` | (B*T, C, H, W) | (B*T, D_img) |
| `CLIPEncoder(cfg)` | (B*T, C, H, W) | (B*T, D_clip) |
| `PointMLP(cfg)` | (B, T, N, 3) | (B, T, N, D_pt) |
| `CausalTransformer(cfg)` | (B, T, D) | (B, T, num_points, 3) |
| `Repr4DModel(cfg)` | frames, points | (B, T, N, 3) — 預測下一時刻 |

## training

| 物件 | 用途 |
|------|------|
| `L1PointLoss` | 主損失：以 `pred[:, :-1]` 對齊 `gt[:, 1:]`。 |
| `TemporalConsistencyLoss` | 比較 inter-frame 位移差異。 |
| `training/train.py` | Phase 1 預訓練主程式（`python -m training.train`）。 |
| `training/finetune.py` | Phase 2 微調（`python -m training.finetune`）。 |

## evaluation

| 函式 | 對應目標 |
|------|---------|
| `chamfer_distance(pred, gt)` | 目標 4：幾何重建誤差 ≤ 5%。 |
| `temporal_consistency(pred, gt)` | 目標 2：時序預測誤差 ≤ 10%。 |
| `tracking_error(pred, gt, ...)` | 目標 1、3：追蹤成功率與 ID-Switch 率。 |
| `benchmark_runner.main()` | 自動化跑全部指標並輸出 JSON。 |

## inference

| 入口 | 用途 |
|------|------|
| `inference/predict.py` | 從影片做點雲預測 rollout。 |
| `inference/export.py` | 將點雲匯出 URDF 或 USD 包圍盒。 |

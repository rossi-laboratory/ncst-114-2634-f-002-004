# 系統架構

本專案實作 ARM4R 風格的 4D 動態場景重建管線：從單鏡頭影片中
追蹤穩定 ID 的 3D 點，並用 Causal Transformer 預測下一時刻點位。

## 高階流程

```
影片 (MP4)
   │
   ▼
┌──────────────┐    ┌──────────────────┐
│ 抽 Frame      │ →  │ Monocular Depth   │
└──────────────┘    │ (Depth Anything   │
                    │  V2 / ZoeDepth)   │
                    └──────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Grid Init (固定 ID) │
                    └──────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ SpatialTracker    │
                    │ (3D point tracks) │
                    └──────────────────┘
                             │
                             ▼
              data/point_tracks/*.npz
                             │
                             ▼
                    ┌──────────────────┐
                    │ ViT + CLIP +      │
                    │ PointMLP fusion   │
                    └──────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Causal Transformer│
                    └──────────────────┘
                             │
                             ▼
                  預測下一時刻點雲位置 (B, T, N, 3)
```

## 三個訓練階段

| 階段 | 資料 | 任務 | 本年度狀態 |
|------|------|------|-----------|
| Phase 1 | Epic-Kitchens100（人類影片） | 預測下一步 3D 點位 | ✅ 完成 |
| Phase 2 | 機器人示範影片 | 在 Phase 1 權重上微調，對齊機型/視角 | ✅ 完成 |
| Phase 3 | 機器人 Proprioception | 接控制任務 | 🚧 第二年 |

## 模組依賴圖

```
preprocessing.depth_estimation ─┐
preprocessing.point_cloud_init ─┼─> dataset.epic_kitchens ─> training.train
preprocessing.spatial_tracker  ─┘                              │
                                                               ▼
              models.{vit_encoder, clip_encoder,    inference.predict
              point_mlp, causal_transformer,                    │
              repr_4d_model}                                    ▼
                       ▲                              evaluation.benchmark_runner
                       │
                training.finetune
```

## 關鍵設計決策

- **固定 ID 點集**：在第一幀以格點均勻採樣並指派整數 ID，貫穿整段影片，
  讓模型學到的是「同一物理點如何隨時間變化」。
- **Causal Mask**：Transformer 不能看到未來時間步，使預測在時序上嚴格 auto-regressive。
- **特徵融合**：將 ViT、CLIP、PointMLP 的輸出在維度層 concat，再以單層 Linear+LN 投影。
- **Loss**：以 L1 為主，加入小權重的 Temporal Consistency loss，避免單純擬合每幀位置。

# 資料集準備流程

## Epic-Kitchens100（Phase 1）

1. 從官方來源下載影片至 `data/raw/epic_kitchens/`。
2. 執行 `bash scripts/run_preprocessing.sh`，會產生：
   - `data/processed/<video_id>/frames/*.jpg`
   - `data/processed/<video_id>/depth/*.npy`
   - `data/point_tracks/<video_id>.npz`

## 機器人示範資料（Phase 2）

採用相同的目錄結構放入 `data/raw/robot_demos/`，再次執行 `run_preprocessing.sh`。
建議至少準備 30–50 段影片，每段 5–15 秒。

## 相機內參

預設使用 `configs/data.yaml` 中的 `camera` 區塊。若每段影片都有獨立內參，
可在 metadata 中提供，並於 `preprocessing/spatial_tracker.py` 內讀取覆寫。

## SpatialTracker 安裝

```bash
git clone https://github.com/henry123-boy/SpatialTracker.git
cd SpatialTracker
pip install -e .
```

若沒有安裝，會自動 fallback 到 Farneback 光流 + 深度查表的簡易追蹤器
（精度較差，僅供開發 sanity check）。

## Benchmark Dataset

`data/benchmark/<sample_id>/` 內必須包含：

- `frames.npy` — `(T, H, W, 3)` uint8
- `tracks.npz` — `tracks: (T, N, 3)`、`point_ids: (N,)`、`video_id`、`metadata`

`evaluation/benchmark_runner.py` 會逐一載入並產出 `results/eval_results.json`。

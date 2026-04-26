#!/bin/bash
# Full preprocessing pipeline: video -> RGB frames + depth -> 3D tracks.
set -euo pipefail

VIDEOS_DIR="${VIDEOS_DIR:-data/raw/epic_kitchens}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed}"
TRACKS_DIR="${TRACKS_DIR:-data/point_tracks}"
CONFIG="${CONFIG:-configs/data.yaml}"

mkdir -p "$OUTPUT_DIR" "$TRACKS_DIR"

echo "[1/3] Extracting frames + depth..."
python -m preprocessing.depth_estimation \
    --videos_dir "$VIDEOS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --config "$CONFIG" || true

echo "[2/3] Grid-based 3D point initialization..."
python -m preprocessing.point_cloud_init \
    --processed_dir "$OUTPUT_DIR" \
    --config "$CONFIG" || true

echo "[3/3] 3D point tracking..."
python -m preprocessing.spatial_tracker \
    --processed_dir "$OUTPUT_DIR" \
    --output_dir "$TRACKS_DIR" \
    --config "$CONFIG" || true

echo "Done. Tracks written to $TRACKS_DIR"

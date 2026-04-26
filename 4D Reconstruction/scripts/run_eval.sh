#!/bin/bash
# Evaluate the trained model against the benchmark dataset.
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-checkpoints/best_model.pth}"
BENCHMARK_DIR="${BENCHMARK_DIR:-data/benchmark}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"

mkdir -p "$OUTPUT_DIR"
python -m evaluation.benchmark_runner \
    checkpoint_path="$CHECKPOINT" \
    benchmark_dir="$BENCHMARK_DIR" \
    output_dir="$OUTPUT_DIR"

echo "Evaluation complete. See $OUTPUT_DIR/eval_results.json"

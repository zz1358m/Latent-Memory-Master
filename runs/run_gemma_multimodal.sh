#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/config_gemma3_4B_12B.yaml"
CKPT="checkpoints/gemma_multimodal/model.pt"
OUT_DIR="results/gemma_webqa"
TOP_K=5

mkdir -p "$(dirname "$CKPT")" "$OUT_DIR"

python scripts/train_release.py --task gemma \
  --config "$CONFIG" \
  --output "$CKPT"

python scripts/eval_release.py --task gemma \
  --config "$CONFIG" \
  --top_k_values "$TOP_K" \
  --checkpoint "$CKPT" \
  --output "$OUT_DIR/baselines_gemma.json"
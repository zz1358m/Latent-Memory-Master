#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/config_llava.yaml"
CKPT="checkpoints/llava/model.pt"
MEMORY_BANK="memory_bank/llava_webqa"
OUT_DIR="results/llava_webqa"
TOP_K=5

mkdir -p "$(dirname "$CKPT")" "$MEMORY_BANK" "$OUT_DIR"

python scripts/train_release.py --task llava \
  --config "$CONFIG" \
  --output "$CKPT"

python scripts/compile_release.py --task llava \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --data "data/webqa_train.json" \
  --output "$MEMORY_BANK"

python scripts/eval_release.py --task llava \
  --config "$CONFIG" \
  --top_k_values "$TOP_K" \
  --checkpoint "$CKPT" \
  --output "$OUT_DIR/baselines_llava.json"
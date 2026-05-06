#!/usr/bin/env bash
set -euo pipefail

RUN_YAML="runs/text_llama.yaml"
CONFIG="configs/config.yaml"
TRAIN_DATA="data/hotpotqa_train.json"
VAL_DATA="data/hotpotqa_val.json"
CKPT="checkpoints/text_llama/model.pt"
MEMORY_BANK="memory_bank/text_llama"
OUT_DIR="results/text_llama"
TOP_K=5

mkdir -p "$(dirname "$CKPT")" "$MEMORY_BANK" "$OUT_DIR"

python scripts/train_release.py --task text \
  --config "$CONFIG" \
  --data "$TRAIN_DATA" \
  --val_data "$VAL_DATA" \
  --output "$CKPT"

python scripts/compile_release.py --task text \
  --config "$CONFIG" \
  --compiler "$CKPT" \
  --data "$TRAIN_DATA" \
  --output "$MEMORY_BANK"

python scripts/eval_release.py --task text \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --output "$OUT_DIR" \
  --top_k "$TOP_K"
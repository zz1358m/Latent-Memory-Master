#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/config_llama1b_mistral7b.yaml}"
CKPT="${CKPT:-checkpoints/text_llama_mistral/model_best.pt}"
OUT_DIR="${OUT_DIR:-results/text_llama_mistral}"
DATASET="${DATASET:-hotpotqa}"
OOD_DATASETS="${OOD_DATASETS:-2wikimultihopqa,musique}"
K_LIST="${K_LIST:-1,2,5}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"

mkdir -p "$OUT_DIR"

python scripts/eval_release.py --task text \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --output "$OUT_DIR" \
  --dataset "$DATASET" \
  --ood_datasets "$OOD_DATASETS" \
  --k_list "$K_LIST" \
  --max_samples "$MAX_SAMPLES" \
  --skip_baselines

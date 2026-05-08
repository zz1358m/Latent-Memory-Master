# Latent Memory

Release-oriented code for Latent Memory: compact latent evidence representations for retrieval-augmented question answering.

This repository contains the core `src/` modules and executable `scripts/` for three experiment families:

- Text-only LLaMA experiments: LLaMA compressor with a frozen LLaMA generator.
- LLaVA multimodal experiments: LLaVA compressor/generator over unified text-image evidence.
- Gemma multimodal experiments: Gemma-family multimodal compressor/generator setup.

## Layout

```text
src/        Core model, compressor, retrieval, evaluation, and memory-bank modules.
scripts/    Public train/eval entry points plus optional memory-bank builder.
configs/    Main YAML configs for text-only, LLaVA multimodal, and Gemma multimodal runs.
runs/       Ready-to-edit YAML and shell launchers for the main experiments.
```

## Installation

```bash
pip install -r requirements.txt
```

## Released Checkpoints and Datasets Offline

https://huggingface.co/zz1358m/Latent-Memory-Master

## Main workflow

The default release workflow is intentionally simple:

```text
train -> eval
```

Memory-bank construction is optional and only needed for offline indexing over a fixed corpus. The multimodal evaluation scripts build per-sample candidate pools online and do not require a separate compile step.

## Ready-to-run entry points

```bash
bash runs/run_text_llama.sh
bash runs/run_llava_multimodal.sh
bash runs/run_gemma_multimodal.sh
```

Each run file calls the unified public entry points:

```bash
python scripts/train_release.py --task text  --config configs/config.yaml
python scripts/train_release.py --task llava --config configs/config_llava.yaml
python scripts/train_release.py --task gemma --config configs/config_gemma3_4B_12B.yaml

python scripts/eval_release.py --task text  --config configs/config.yaml --checkpoint checkpoints/text_llama/model.pt --output results/text_llama --top_k 5
python scripts/eval_release.py --task llava --config configs/config_llava.yaml --checkpoint checkpoints/llava/model.pt --top_k_values 5 --output results/llava_webqa/baselines_llava.json
python scripts/eval_release.py --task gemma --config configs/config_gemma3_4B_12B.yaml --checkpoint checkpoints/gemma_multimodal/model.pt --top_k_values 5 --output results/gemma_webqa/baselines_gemma.json
```

## Optional: offline memory-bank construction

For fixed-corpus retrieval experiments, build a memory bank explicitly:

```bash
python scripts/build_memory_bank.py --task text \
  --config configs/config.yaml \
  --compiler checkpoints/text_llama/model.pt \
  --data data/hotpotqa_train.json \
  --output memory_bank/text_llama

python scripts/build_memory_bank.py --task llava \
  --config configs/config_llava.yaml \
  --checkpoint checkpoints/llava/model.pt \
  --data data/webqa_train.json \
  --output memory_bank/llava_webqa
```

This step is not part of the default release runs.

## Public Python entry points

```text
scripts/train_release.py
scripts/eval_release.py
scripts/build_memory_bank.py    # optional offline indexing
```

Task-specific implementations live in `scripts/internal/` so the release remains easy to navigate while preserving the original training and evaluation logic.

## Notes

- Data, checkpoints, cache directories, and paper artifacts are intentionally excluded from this release scaffold.
- Multimodal baseline token accounting includes effective visual-token expansion for full prompt, no-system, and context-only token statistics.
- Add model checkpoint links or HuggingFace dataset/model cards before public release if needed.

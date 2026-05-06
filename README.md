# Latent Memory

Release-oriented code for Latent Memory: compact latent evidence representations for retrieval-augmented question answering.

This repository contains the core `src/` modules and executable `scripts/` for three experiment families:

- Text-only LLaMA experiments: LLaMA compressor with a frozen LLaMA generator.
- LLaVA multimodal experiments: LLaVA compressor/generator over unified text-image evidence.
- Gemma multimodal experiments: Gemma-family multimodal configuration files and compatible multimodal evaluation setup.

## Layout

```text
src/        Core model, compressor, retrieval, evaluation, and memory-bank modules.
scripts/    Training, corpus compilation, evaluation, and baseline entry points.
configs/    YAML configs for text-only, LLaMA text-only, LLaVA multimodal, and Gemma multimodal experiment variants.
```

## Installation

```bash
pip install -r requirements.txt
```

## Text-only LLaMA workflow

Prepare data externally, then train and evaluate with the LLaMA text-only configuration:

```bash
python scripts/train.py --config configs/config.yaml --data data/hotpotqa_train.json --val_data data/hotpotqa_val.json --output checkpoints/text_llama.pt
python scripts/compile_release.py --task text --config configs/config.yaml --compiler checkpoints/text_llama.pt --data data/hotpotqa_train.json --output memory_bank/text_llama
python scripts/run_experiment.py --config configs/config.yaml --checkpoint checkpoints/text_llama.pt --output results/text_llama --top_k 5
```

## LLaVA multimodal workflow

```bash
python scripts/train_llava.py --config configs/config_llava.yaml
python scripts/compile_release.py --task webqa --config configs/config_llava.yaml --checkpoint checkpoints/llava.pt --output memory_bank/llava_webqa
python scripts/baselines_llava.py --config configs/config_llava.yaml --top_k 5
```

## Gemma multimodal workflow

Gemma configurations are provided under `configs/` (for example `config_gemma3_4B_12B.yaml`). Use the same multimodal training/evaluation pattern as LLaVA with the Gemma config selected:

```bash
python scripts/train_gemma.py --config configs/config_gemma3_4B_12B.yaml
python scripts/baselines_gemma.py --config configs/config_gemma3_4B_12B.yaml --top_k 5
```

## Notes

- Data, checkpoints, cache directories, and paper artifacts are intentionally excluded from this release scaffold.
- Multimodal baseline token accounting includes effective visual-token expansion for full prompt, no-system, and context-only token statistics.
- Before public release, add model checkpoint links or HuggingFace dataset/model cards as appropriate.

## Main Python entry points

The release exposes two unified command-line entry points:

```bash
python scripts/train_release.py --task text  --config configs/config.yaml
python scripts/train_release.py --task llava --config configs/config_llava.yaml
python scripts/train_release.py --task gemma --config configs/config_gemma3_4B_12B.yaml

python scripts/eval_release.py --task text  --config configs/config.yaml --checkpoint checkpoints/text_llama/model.pt --output results/text_llama --top_k 5
python scripts/eval_release.py --task llava --config configs/config_llava.yaml --top_k 5 --output results/llava_webqa
python scripts/eval_release.py --task gemma --config configs/config_gemma3_4B_12B.yaml --top_k 5 --output results/gemma_webqa
```

Lower-level scripts are kept for transparency, but the run files in `runs/` call these unified entry points.## Ready-to-run entry points

The release keeps only the main experiment entry points:

```text
runs/text_llama.yaml          runs/run_text_llama.sh
runs/llava_multimodal.yaml    runs/run_llava_multimodal.sh
runs/gemma_multimodal.yaml    runs/run_gemma_multimodal.sh
```

Example:

```bash
bash runs/run_text_llama.sh
bash runs/run_llava_multimodal.sh
bash runs/run_gemma_multimodal.sh
```

Edit the YAML or shell variables if your data/checkpoint paths differ from the defaults.
## Python file layout

The public `scripts/` directory intentionally exposes only three entry points:

```text
scripts/train_release.py
scripts/compile_release.py
scripts/eval_release.py
```

Task-specific implementations live in `scripts/internal/` so the release remains easy to navigate while preserving the original training and evaluation logic.
# Latent Retrieval over Compiled Memory

Latent Memory compiles each evidence item into a small latent memory, retrieves these memories at inference time, and feeds them directly to a frozen generation model.

## Install

```bash
pip install -r requirements.txt
```

## Main Experiments

This repository has two primary experiment tracks:

1. text-only QA on HotpotQA / 2WikiMultihopQA / MuSiQue
2. multimodal QA on WebQA with LLaVA or Qwen-VL backbones

## Text-only: train and test

Train the text-only compression model:

```bash
python scripts/train.py \
    --config config.yaml \
    --data data/hotpotqa_train.json \
    --val_data data/hotpotqa_val.json \
    --output checkpoints/model.pt
```

Evaluate the trained checkpoint on the main text-only setting:

```bash
python scripts/run_experiment.py \
    --config config.yaml \
    --checkpoint checkpoints/model_best.pt \
    --dataset hotpotqa \
    --ood_datasets 2wikimultihopqa,musique \
    --k_list 1,2,5 \
    --output results/
```

## Multimodal: train and test

Train the LLaVA-based WebQA compression model:

```bash
python scripts/train_llava.py \
    --config config_llava.yaml \
    --data data/webqa_train.json \
    --val data/webqa_val.json \
    --output checkpoints/llava_model.pt
```

Evaluate the trained LLaVA checkpoint on WebQA:

```bash
python scripts/baselines_llava.py \
    --checkpoint checkpoints/llava_model_best.pt \
    --config config_llava.yaml \
    --val data/webqa_val.json \
    --methods full_context bm25 dense latent \
    --top_k_values 1 2 5 \
    --max_samples 10000
```

If you use the Qwen-VL multimodal setting instead, evaluate with:

```bash
python scripts/baselines_qwen.py \
    --checkpoint checkpoints/qwen_model_best.pt \
    --config config_qwen.yaml \
    --val data/webqa_val.json \
    --methods full_context bm25 dense latent \
    --top_k_values 1 2 5 \
    --max_samples 10000
```

## Notes

- `config.yaml` is the main text-only configuration.
- `config_llava.yaml` and `config_qwen.yaml` are the multimodal configurations.
- Text-only evaluation uses `scripts/run_experiment.py`.
- WebQA multimodal evaluation uses `scripts/baselines_llava.py` or `scripts/baselines_qwen.py`.
- Replace checkpoint paths above with your actual saved checkpoints.

"""
Train LCC Compiler
==================
Trains the LCC compiler (LoRA + buffer embeddings + retrieval head)
using the self-aligned reconstruction objective — no synthetic QA pairs needed.

Training procedure
------------------
For each document:
  1. Run [buffer | document] through LoRA-LLM → compiled memory
  2. Reconstruction loss: frozen LLM([compiled | doc[:-1]]) predicts doc[1:]
  3. (Optional) QA loss if labeled (question, answer) pairs are available

Backpropagation only updates:
  - LoRA weights
  - buffer_embeddings
  - retrieval_head

Usage
-----
python scripts/train_compiler.py \
    --config config.yaml \
    --data data/samples.json \
    --output checkpoints/compiler.pt
"""

import argparse
import logging
import os
import sys

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.compiler import LCCCompiler
from data.prepare_data import load_dataset_by_name, load_samples

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def train(args):
    # --- Config ---
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # --- Data ---
    if os.path.exists(args.data):
        samples = load_samples(args.data)
    else:
        samples = load_dataset_by_name(
            cfg["data"]["dataset"],
            split="train",
            max_samples=cfg["data"]["max_samples"],
            cache_dir=cfg["data"]["cache_dir"],
        )

    # Use only documents for self-supervised training (no labels needed)
    documents = [s["document"] for s in samples]
    # Optionally use QA pairs for supervised fine-tuning
    qa_pairs = [(s["question"], s["answers"][0]) for s in samples if s["answers"]]

    logger.info(f"Training on {len(documents)} documents")

    # --- Model ---
    comp_cfg = cfg["compiler"]
    compiler = LCCCompiler(
        model_name=cfg["model"]["name"],
        num_buffer_tokens=comp_cfg["num_buffer_tokens"],
        lora_r=comp_cfg["lora_r"],
        lora_alpha=comp_cfg["lora_alpha"],
        lora_dropout=comp_cfg["lora_dropout"],
        target_modules=comp_cfg["target_modules"],
        retrieval_dim=comp_cfg["retrieval_dim"],
        max_doc_length=comp_cfg["max_doc_length"],
    ).to(device)

    # --- Optimizer ---
    # Only optimize LoRA params + buffer_embeddings + retrieval_head
    params = [
        p for name, p in compiler.named_parameters()
        if "lora" in name or "buffer_embeddings" in name or "retrieval_head" in name
    ]
    optimizer = AdamW(params, lr=cfg["training"]["learning_rate"])
    n_steps = (
        len(documents)
        * cfg["training"]["num_epochs"]
        // cfg["training"]["gradient_accumulation_steps"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-6)

    # --- Training loop ---
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    compiler.train()

    global_step = 0
    alpha_recon = cfg["training"]["alpha_recon"]
    alpha_qa = cfg["training"]["alpha_qa"]
    grad_accum = cfg["training"]["gradient_accumulation_steps"]
    log_every = cfg["training"]["log_every"]

    for epoch in range(cfg["training"]["num_epochs"]):
        logger.info(f"Epoch {epoch + 1}/{cfg['training']['num_epochs']}")
        epoch_loss = 0.0

        for i, doc in enumerate(tqdm(documents, desc="Training")):
            # Self-supervised reconstruction loss
            loss = compiler.reconstruction_loss(doc, alpha=alpha_recon)

            # Optional QA loss
            if alpha_qa > 0 and i < len(qa_pairs):
                q, a = qa_pairs[i]
                loss = loss + compiler.qa_loss(doc, q, a, alpha=alpha_qa)

            loss = loss / grad_accum
            loss.backward()
            epoch_loss += loss.item()

            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % log_every == 0:
                    logger.info(
                        f"Step {global_step} | Loss: {epoch_loss / log_every:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e}"
                    )
                    epoch_loss = 0.0

        # Save checkpoint after each epoch
        ckpt_path = args.output.replace(".pt", f"_epoch{epoch+1}.pt")
        compiler.save(ckpt_path)

    # Save final checkpoint
    compiler.save(args.output)
    logger.info(f"Training complete. Final model saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", default="data/samples.json")
    parser.add_argument("--output", default="checkpoints/compiler.pt")
    args = parser.parse_args()
    train(args)

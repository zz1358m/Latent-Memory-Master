"""
Unified Training Script
========================
Trains the full system end-to-end with a combined loss:

  L_total = λ_recon   * L_recon
          + λ_contrast * L_contrast
          + λ_distill  * L_distill

All three losses are computed on every batch and summed before backward().
Gradients flow through:
  - L_recon   → mem_embedding + LoRA_compress + LoRA_decode
  - L_contrast → mem_embedding + LoRA_compress + retrieval_proj
  - L_distill  → mem_embedding + LoRA_compress  (latent_token is produced by compress)

Validation
----------
After every cfg["training"]["val_every"] steps (default 500) and at the end of
each epoch the full latent retrieval pipeline is evaluated on a held-out set:
  - If --val_data is provided, that JSON file is used.
  - Otherwise 10 % of training data is split off at startup.

The best checkpoint (highest validation F1) is saved to
  <output>.replace(".pt", "_best.pt")

Usage
-----
python scripts/train.py --config config.yaml \
    --data data/hotpotqa_train.json \
    --output checkpoints/model.pt \
    [--val_data data/hotpotqa_val.json]
"""

import argparse
import logging
import os
import random
import sys

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False
    SummaryWriter = None
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.prepare_data import load_dataset_by_name, load_samples
from data.sentence_utils import prepare_samples_with_sentences
from src.compressor import SentenceAutoencoder
from src.distillation import batch_distillation_loss_multi
from src.retriever import ContrastiveRetriever
from src.validator import ValidationRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batch building
# ---------------------------------------------------------------------------

def build_batch(
    samples: list,
    batch_size: int,
    max_hard_negatives: int = 0,
    rng: random.Random = None,
) -> list:
    """
    Sample a mini-batch and attach hard negatives.

    Each batch item contains:
      {
        "question":         str,
        "pos_sentences":    list[str],  # ALL ground-truth positive sentences
        "hard_neg_sents":   list[str],  # randomly sampled other sentences from same context
      }
    """
    if rng is None:
        rng = random.Random()

    batch_samples = rng.choices(samples, k=batch_size)
    batch = []
    for s in batch_samples:
        sents = s["sentences"]
        pos_idx = list(s["positive_indices"])
        pos_sents = [sents[i] for i in pos_idx]

        pos_set = set(pos_idx)
        all_neg_idx = [i for i in range(len(sents)) if i not in pos_set]

        # Random subset of hard negatives (capped at max_hard_negatives)
        rand_neg_idx = list(all_neg_idx)
        if max_hard_negatives > 0 and len(rand_neg_idx) > max_hard_negatives:
            rng.shuffle(rand_neg_idx)
            rand_neg_idx = rand_neg_idx[:max_hard_negatives]

        batch.append({
            "question":       s["question"],
            "pos_sentences":  pos_sents,
            "hard_neg_sents": [sents[i] for i in rand_neg_idx],
        })
    return batch


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------

def train_step(
    autoencoder: SentenceAutoencoder,
    retriever: ContrastiveRetriever,
    batch: list,
    lambda_recon: float,
    lambda_contrast: float,
    lambda_distill: float,
    device: str,
    generator=None,               # frozen 8B LLM for distillation (optional)
    generator_tokenizer=None,
    n_distill_tokens: int = 1,
    max_hard_negatives: int = 0,
    recon_negatives: bool = False,    # include hard negatives in sentence reconstruction
    recon_query: bool = True,         # include query reconstruction loss
    student_distill_hard_negatives_max: int = 0,
    rng: random.Random = None,
) -> dict:
    """
    Compute combined loss for one batch — fully batched for H200.

    L_total = λ_recon * L_recon + λ_contrast * L_contrast + λ_distill * L_distill

    All three losses share the same compressed embeddings, so we only
    run the compressor once per batch (one forward pass).
    """
    questions        = [b["question"]      for b in batch]
    pos_sents_list   = [b["pos_sentences"] for b in batch]   # list of lists
    all_hard_negs    = [sn for b in batch for sn in b["hard_neg_sents"]]
    hard_neg_sents_list = [b["hard_neg_sents"] for b in batch]
    if rng is None:
        rng = random.Random()

    # Flat list of all positive sentences (for batched compression)
    all_pos_flat = [s for sents in pos_sents_list for s in sents]

    # ============================================================
    # L_recon: sentence recon (compress adapter) + query recon (query adapter)
    # ============================================================
    recon_sents  = all_pos_flat + all_hard_negs if recon_negatives else all_pos_flat
    l_recon_sent = autoencoder.reconstruction_loss_batch(recon_sents, adapter="compress", decode_adapter="decode")
    l_recon_query = (
        autoencoder.reconstruction_loss_batch(questions, adapter="query", decode_adapter="query_decode")
        if recon_query else torch.tensor(0.0, device=device)
    )
    l_recon = l_recon_sent + l_recon_query

    # ============================================================
    # Compress all positive sentences (with grad) — shared for L_contrast + L_distill
    # ============================================================
    all_pos_compressed = autoencoder.compress_batch(all_pos_flat, with_grad=True)  # (total_pos, H)
    all_pos_embs = autoencoder.get_retrieval_embedding(all_pos_compressed)          # (total_pos, D)

    # Split back into per-sample lists
    pos_compressed_list = []
    pos_embs_list = []
    offset = 0
    for sents in pos_sents_list:
        K = len(sents)
        pos_compressed_list.append(all_pos_compressed[offset:offset+K])
        pos_embs_list.append(all_pos_embs[offset:offset+K])
        offset += K

    # ============================================================
    # L_contrast: multi-positive NT-Xent
    # ============================================================
    # Query embedding via dedicated "query" adapter — no recon gradient conflict
    q_compressed = autoencoder.compress_batch(questions, with_grad=True, adapter="query")  # (B, H)
    q_embs = autoencoder.get_retrieval_embedding(q_compressed)                              # (B, D)

    neg_comp = None
    hard_neg_embs = None
    neg_compressed_list = []
    if all_hard_negs:
        neg_comp = autoencoder.compress_batch(all_hard_negs, with_grad=True)
        hard_neg_embs = autoencoder.get_retrieval_embedding(neg_comp)   # (N_rand, D)

        offset = 0
        for neg_sents in hard_neg_sents_list:
            n = len(neg_sents)
            neg_compressed_list.append(neg_comp[offset:offset+n])
            offset += n
    else:
        neg_compressed_list = [
            all_pos_compressed.new_zeros((0, all_pos_compressed.shape[-1]))
            for _ in batch
        ]

    l_contrast = retriever.contrastive_loss(q_embs, pos_embs_list, hard_neg_embs)

    # Contrast diagnostics: avg positive sim and avg negative sim
    with torch.no_grad():
        pos_sim_mean = torch.stack([
            (q_embs[i:i+1] @ pos_embs_list[i].T).mean()
            for i in range(len(q_embs))
        ]).mean().item()
        if hard_neg_embs is not None and hard_neg_embs.shape[0] > 0:
            # Per-sample mean: query_i × its own hard negatives only
            neg_sizes = [len(b["hard_neg_sents"]) for b in batch]
            per_sample_sims = []
            offset = 0
            for i, n in enumerate(neg_sizes):
                if n > 0:
                    per_sample_sims.append(
                        (q_embs[i:i+1] @ hard_neg_embs[offset:offset+n].T).mean()
                    )
                offset += n
            neg_sim_mean = torch.stack(per_sample_sims).mean().item() if per_sample_sims else 0.0
        else:
            # in-batch negatives: off-diagonal of query×sentence sim matrix
            all_pos_cat = torch.cat(pos_embs_list, dim=0)   # (total_pos, D)
            sim_mat = q_embs @ all_pos_cat.T                # (B, total_pos)
            neg_sim_mean = sim_mat.mean().item()

    # ============================================================
    # L_distill: multi-sentence KL distillation
    # Teacher: 8B sees K sentence texts; Student: 8B sees K latent tokens
    # Subsample to keep compute manageable
    # ============================================================
    l_distill = torch.tensor(0.0, device=device)
    if lambda_distill > 0 and generator is not None:
        projected_list = []
        for i in range(len(questions)):
            student_latents = pos_compressed_list[i]

            if student_distill_hard_negatives_max > 0 and neg_compressed_list[i].shape[0] > 0:
                max_extra = min(student_distill_hard_negatives_max, neg_compressed_list[i].shape[0])
                n_extra = rng.randint(0, max_extra)
                if n_extra > 0:
                    selected = rng.sample(range(neg_compressed_list[i].shape[0]), k=n_extra)
                    extra_latents = neg_compressed_list[i][selected]
                    student_latents = torch.cat([student_latents, extra_latents], dim=0)

            projected_list.append(autoencoder.project_for_generator(student_latents))

        l_distill = batch_distillation_loss_multi(
            generator, generator_tokenizer,
            questions,
            pos_sents_list,
            projected_list,
            device,
            n_distill_tokens=n_distill_tokens,
        )

    # ============================================================
    # Combined loss
    # ============================================================
    total = (
        lambda_recon        * l_recon
        + lambda_contrast   * l_contrast
        + lambda_distill    * l_distill
    )

    return {
        "total":              total,
        "recon":              l_recon.item(),
        "recon_sent":         l_recon_sent.item(),
        "recon_query":        l_recon_query.item(),
        "contrast":           l_contrast.item(),
        "contrast_pos_sim":   pos_sim_mean,
        "contrast_neg_sim":   neg_sim_mean,
        "distill":            l_distill.item() if isinstance(l_distill, torch.Tensor) else 0.0,
    }


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _run_validation(
    validator: ValidationRunner,
    val_samples: list,
    val_samples_n: int,
    global_step: int,
    best_f1: float,
    best_path: str,
    autoencoder: SentenceAutoencoder,
    writer: SummaryWriter = None,
) -> float:
    """
    Run validation, log results to TensorBoard, and save best checkpoint.
    """
    logger.info(f"Running validation at step {global_step} (n={val_samples_n})...")
    val_metrics = validator.run(val_samples, n_samples=val_samples_n)

    logger.info(
        f"[Val step={global_step}] "
        f"EM={val_metrics['em']:.4f} | "
        f"F1={val_metrics['f1']:.4f} | "
        f"ROUGE-L={val_metrics['rouge_l']:.4f} | "
        f"Recall@k={val_metrics.get('recall_at_k', 0):.4f} | "
        f"Precision@k={val_metrics.get('precision_at_k', 0):.4f} | "
        f"avg_tokens={val_metrics.get('avg_tokens', 0):.1f}"
    )

    if writer is not None:
        writer.add_scalar("val/em",              val_metrics["em"],                            global_step)
        writer.add_scalar("val/f1",              val_metrics["f1"],                            global_step)
        writer.add_scalar("val/rouge_l",         val_metrics["rouge_l"],                       global_step)
        writer.add_scalar("val/recall_at_k",     val_metrics.get("recall_at_k", 0),            global_step)
        writer.add_scalar("val/precision_at_k",  val_metrics.get("precision_at_k", 0),         global_step)
        writer.add_scalar("val/avg_tokens",      val_metrics.get("avg_tokens", 0),             global_step)
        writer.add_scalar("val/avg_retrieved",   val_metrics.get("avg_retrieved_sentences", 0), global_step)
        writer.flush()

    current_f1 = val_metrics["f1"]
    if current_f1 > best_f1:
        best_f1 = current_f1
        autoencoder.save(best_path)
        logger.info(
            f"New best F1={best_f1:.4f} at step {global_step}. "
            f"Checkpoint saved to {best_path}"
        )

    return best_f1


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    """
    Main entry point for training.

    Reads configuration from args.config (YAML), loads data, builds the
    SentenceAutoencoder, optional frozen 8B generator, and runs the training
    loop.  Validation is performed periodically and at the end of each epoch.

    Parameters
    ----------
    args : argparse.Namespace
        Must have: config, data, output, val_data (optional).
    """
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # --- Data ---
    # Try local file first; fall back to HuggingFace and cache locally.
    def _try_load_local(path):
        if not os.path.exists(path):
            return None
        try:
            return load_samples(path)
        except Exception as e:
            logger.warning(f"Local file {path} is corrupted ({e}), re-downloading.")
            return None

    raw_samples = _try_load_local(args.data)
    if raw_samples is None:
        data_cfg = cfg["data"]
        raw_samples = load_dataset_by_name(
            data_cfg["dataset"],
            data_cfg["split"],
            data_cfg["max_samples"],
            data_cfg["cache_dir"],
        )
        # Cache locally for future runs
        os.makedirs(os.path.dirname(os.path.abspath(args.data)), exist_ok=True)
        from data.prepare_data import save_samples
        save_samples(raw_samples, args.data)
        logger.info(f"Cached training data to {args.data}")

    # --- Build SentenceAutoencoder (1B compressor + LoRA adapters) ---
    model_cfg  = cfg["model"]
    comp_cfg   = cfg["compressor"]
    dec_cfg    = cfg["decoder"]
    query_cfg  = cfg.get("query", {})
    train_cfg  = cfg["training"]

    autoencoder = SentenceAutoencoder(
        model_name=model_cfg["compressor_name"],
        compress_lora_r=comp_cfg["lora_r"],
        compress_lora_alpha=comp_cfg["lora_alpha"],
        decode_lora_r=dec_cfg["lora_r"],
        decode_lora_alpha=dec_cfg["lora_alpha"],
        query_lora_r=query_cfg.get("lora_r", comp_cfg["lora_r"]),
        query_lora_alpha=query_cfg.get("lora_alpha", comp_cfg["lora_alpha"]),
        query_target_modules=query_cfg.get("target_modules", comp_cfg["target_modules"]),
        lora_dropout=comp_cfg["lora_dropout"],
        compress_target_modules=comp_cfg["target_modules"],
        decode_target_modules=dec_cfg["target_modules"],
        retrieval_dim=comp_cfg["retrieval_dim"],
        generator_hidden=model_cfg.get("generator_hidden", 4096),
        num_latent_tokens=comp_cfg.get("num_latent_tokens", 1),
    ).to(device)

    # --- Load frozen 8B generator (for distillation loss + validation) ---
    generator, generator_tokenizer = None, None
    needs_generator = (
        train_cfg["lambda_distill"] > 0
        or True  # always load for validation if val data is available
    )
    if needs_generator:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        gen_name = model_cfg["generator_name"]
        logger.info(f"Loading frozen 8B generator: {gen_name}")
        generator_tokenizer = AutoTokenizer.from_pretrained(gen_name)
        if generator_tokenizer.pad_token is None:
            generator_tokenizer.pad_token = generator_tokenizer.eos_token
        generator = AutoModelForCausalLM.from_pretrained(
            gen_name, torch_dtype=torch.bfloat16, device_map=device,
        )
        generator.eval()
        for p in generator.parameters():
            p.requires_grad_(False)
        logger.info("8B generator loaded (frozen).")

    retriever = ContrastiveRetriever(
        temperature=train_cfg["temperature"]
    ).to(device)

    # --- Sentence-split and label data ---
    logger.info("Preparing sentence-split samples...")
    samples = prepare_samples_with_sentences(
        raw_samples,
        tokenizer=autoencoder.tokenizer,
        max_sentence_tokens=cfg["data"]["max_sentence_length"],
        min_words=cfg["data"]["min_sentence_length"],
    )
    logger.info(f"Training samples after filtering: {len(samples)}")
    if len(samples) == 0:
        raise RuntimeError("No valid samples after sentence splitting. Check data.")

    # --- Validation data setup ---
    val_every   = train_cfg.get("val_every",   500)
    val_samples_n = train_cfg.get("val_samples", 200)

    val_samples_raw = None
    if hasattr(args, "val_data") and args.val_data and os.path.exists(args.val_data):
        logger.info(f"Loading validation data from {args.val_data}")
        val_raw = load_samples(args.val_data)
        val_samples_raw = prepare_samples_with_sentences(
            val_raw,
            tokenizer=autoencoder.tokenizer,
            max_sentence_tokens=cfg["data"]["max_sentence_length"],
            min_words=cfg["data"]["min_sentence_length"],
        )
        logger.info(f"Validation samples: {len(val_samples_raw)}")
    else:
        # 10% split from training data
        rng_split = random.Random(0)
        all_idx = list(range(len(samples)))
        rng_split.shuffle(all_idx)
        n_val = max(1, len(samples) // 10)
        val_idx = set(all_idx[:n_val])
        val_samples_raw = [samples[i] for i in sorted(val_idx)]
        samples = [samples[i] for i in range(len(samples)) if i not in val_idx]
        logger.info(
            f"Split 10% for validation: {len(val_samples_raw)} val, "
            f"{len(samples)} train."
        )

    # --- Build ValidationRunner ---
    validator = None
    if generator is not None and val_samples_raw:
        validator = ValidationRunner(
            autoencoder=autoencoder,
            generator=generator,
            generator_tokenizer=generator_tokenizer,
            top_k=cfg.get("retrieval", {}).get("top_k", 5),
            max_new_tokens=cfg.get("generation", {}).get("max_new_tokens", 64),
            device=device,
        )
        logger.info("ValidationRunner ready.")
    else:
        logger.warning(
            "No ValidationRunner: generator not loaded or no validation data."
        )

    # Best checkpoint path
    best_path = args.output.replace(".pt", "_best.pt")
    best_f1 = -1.0

    # --- TensorBoard ---
    log_dir = getattr(args, "logdir", None) or os.path.join(
        os.path.dirname(os.path.abspath(args.output)), "tb_logs"
    )
    if _TB_AVAILABLE:
        writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard logs → {log_dir}  (tensorboard --logdir {log_dir})")
    else:
        writer = None
        logger.warning("tensorboard not installed — TB logging disabled. Run: pip install tensorboard")

    # --- Optimizer (only trainable params) ---
    params = list(autoencoder.parameters()) + list(retriever.parameters())

    # Log which params have requires_grad=True, grouped by adapter
    grad_groups = {}
    for name, p in autoencoder.model.named_parameters():
        if p.requires_grad:
            key = "compress" if "compress" in name else "query" if "query" in name else "decode" if "decode" in name else "other"
            grad_groups.setdefault(key, [0, 0])
            grad_groups[key][0] += 1
            grad_groups[key][1] += p.numel()
    for name, p in autoencoder.named_parameters():
        if p.requires_grad and not hasattr(p, '_lora_checked'):
            if any(k in name for k in ("mem_embedding", "retrieval_proj", "cross_proj")):
                grad_groups.setdefault(name.split(".")[0], [0, 0])
                grad_groups[name.split(".")[0]][0] += 1
                grad_groups[name.split(".")[0]][1] += p.numel()
    logger.info("=== Trainable parameter groups ===")
    for grp, (n_tensors, n_params) in sorted(grad_groups.items()):
        logger.info(f"  {grp:20s}: {n_tensors:4d} tensors, {n_params/1e6:.2f}M params")
    logger.info(f"  {'TOTAL optimizer':20s}: {sum(p.numel() for p in params if p.requires_grad)/1e6:.2f}M params")

    optimizer = AdamW(params, lr=train_cfg["learning_rate"])
    total_steps = (
        len(samples) * train_cfg["num_epochs"]
        // (train_cfg["batch_size"] * train_cfg["gradient_accumulation_steps"])
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=1e-6)

    # --- Training ---
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    rng = random.Random(42)

    lam_r  = train_cfg["lambda_recon"]
    lam_c  = train_cfg["lambda_contrast"]
    lam_d  = train_cfg["lambda_distill"]
    grad_accum        = train_cfg["gradient_accumulation_steps"]
    log_every         = train_cfg["log_every"]
    distill_layers    = train_cfg.get("distill_layers", [-1])
    n_distill_tokens  = train_cfg.get("n_distill_tokens", 1)

    global_step = 0
    running = {
        "total": 0.0, "recon": 0.0,
        "recon_sent": 0.0, "recon_query": 0.0,
        "contrast": 0.0, "contrast_pos_sim": 0.0, "contrast_neg_sim": 0.0,
        "distill": 0.0,
    }

    # --- Initial validation before any training ---
    if validator is not None:
        best_f1 = _run_validation(
            validator, val_samples_raw, val_samples_n,
            global_step, best_f1, best_path, autoencoder,
            writer=writer,
        )

    for epoch in range(train_cfg["num_epochs"]):
        logger.info(f"Epoch {epoch + 1}/{train_cfg['num_epochs']}")
        n_batches = len(samples) // train_cfg["batch_size"]

        for _ in tqdm(range(n_batches), desc="Training"):
            batch = build_batch(
                samples,
                batch_size=train_cfg["batch_size"],
                max_hard_negatives=train_cfg.get("max_hard_negatives", 0),
                rng=rng,
            )

            loss_dict = train_step(
                autoencoder, retriever, batch,
                lam_r, lam_c, lam_d,
                device,
                generator=generator if lam_d > 0 else None,
                generator_tokenizer=generator_tokenizer,
                n_distill_tokens=n_distill_tokens,
                max_hard_negatives=train_cfg.get("max_hard_negatives", 0),
                recon_negatives=train_cfg.get("recon_negatives", False),
                recon_query=train_cfg.get("recon_query", True),
                student_distill_hard_negatives_max=train_cfg.get("student_distill_hard_negatives_max", 0),
                rng=rng,
            )

            (loss_dict["total"] / grad_accum).backward()

            for k in running:
                running[k] += loss_dict[k]

            if (global_step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            # --- Periodic validation ---
            if validator is not None and global_step % val_every == 0:
                best_f1 = _run_validation(
                    validator, val_samples_raw, val_samples_n,
                    global_step, best_f1, best_path, autoencoder,
                    writer=writer,
                )

            # --- Periodic logging ---
            if global_step % log_every == 0:
                avg = {k: v / log_every for k, v in running.items()}
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Step {global_step} | "
                    f"total={avg['total']:.4f} "
                    f"recon_sent={avg['recon_sent']:.4f} "
                    f"recon_query={avg['recon_query']:.4f} "
                    f"contrast={avg['contrast']:.4f} "
                    f"pos_sim={avg['contrast_pos_sim']:.3f} "
                    f"neg_sim={avg['contrast_neg_sim']:.3f} "
                    f"distill={avg['distill']:.4f} | "
                    f"lr={lr:.2e}"
                )
                if writer is not None:
                    writer.add_scalar("train/loss_total",            avg["total"],              global_step)
                    writer.add_scalar("train/loss_recon",            avg["recon"],              global_step)
                    writer.add_scalar("train/loss_recon_sent",       avg["recon_sent"],         global_step)
                    writer.add_scalar("train/loss_recon_query",      avg["recon_query"],        global_step)
                    writer.add_scalar("train/loss_contrast",         avg["contrast"],           global_step)
                    writer.add_scalar("train/contrast_pos_sim",      avg["contrast_pos_sim"],   global_step)
                    writer.add_scalar("train/contrast_neg_sim",      avg["contrast_neg_sim"],   global_step)
                    writer.add_scalar("train/loss_distill",          avg["distill"],            global_step)
                    writer.add_scalar("train/lr",                    lr,                        global_step)
                running = {k: 0.0 for k in running}


        # --- End-of-epoch checkpoint ---
        ckpt_path = args.output.replace(".pt", f"_epoch{epoch + 1}.pt")
        autoencoder.save(ckpt_path)
        logger.info(f"Epoch {epoch + 1} checkpoint saved to {ckpt_path}")

        # --- End-of-epoch validation ---
        if validator is not None:
            best_f1 = _run_validation(
                validator, val_samples_raw, val_samples_n,
                global_step, best_f1, best_path, autoencoder,
                writer=writer,
            )

    autoencoder.save(args.output)
    logger.info(f"Training complete. Final checkpoint saved to {args.output}")
    if best_f1 >= 0:
        logger.info(f"Best validation F1={best_f1:.4f}. Best checkpoint: {best_path}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the latent sentence retrieval system."
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--data",   default="data/hotpotqa_train.json",
                        help="Path to training samples JSON (or dataset name).")
    parser.add_argument("--output", default="/scratch/e1374322/latent-retrieval-save/checkpoints/model.pt",
                        help="Path for the final checkpoint.")
    parser.add_argument("--val_data", default=None,
                        help="Optional path to validation samples JSON. "
                             "If omitted, 10%% of training data is used.")
    parser.add_argument("--logdir", default="/scratch/e1374322/latent-retrieval-save/tb_logs",
                        help="TensorBoard log directory.")
    args = parser.parse_args()
    train(args)

"""
Run Evaluation Experiment
==========================
Compares the proposed latent retrieval system against three baselines.
Supports in-distribution (HotpotQA) and multiple OOD datasets including
2WikiMultiHopQA, MuSiQue, NQ, TriviaQA, LongBench-Qasper, and WICE.

Proposed system (latent retrieval):
  1B compressor (LoRA) → CrossModelProjection → frozen 8B generator

Baselines (all use same frozen 8B generator):
  - FullContext:    all sentences concatenated as raw text  (run once, k-independent)
  - BM25Retrieval:  top-k sentences via BM25 → raw text
  - DenseRetrieval: top-k sentences via sentence-transformers → raw text

Single-k mode (default):
  python scripts/run_experiment.py --config config.yaml --checkpoint ckpt.pt

Multi-k sweep mode (--k_list):
  FullContext is evaluated once; BM25, Dense and Latent Retrieval are each
  evaluated at every k in the list.

  python scripts/run_experiment.py \\
      --config config.yaml \\
      --checkpoint checkpoints/model_best.pt \\
      --output results/ \\
      --dataset hotpotqa \\
      --k_list 1,2,3,5,10 \\
      [--max_samples 500] \\
      [--skip_baselines] \\
      [--ood_datasets 2wikimultihopqa,musique,nq,triviaqa,qasper_longbench,wice]
"""

import argparse
import json
import logging
import os
import sys

_RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _RELEASE_ROOT not in sys.path:
    sys.path.insert(0, _RELEASE_ROOT)

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.prepare_data import load_dataset_by_name, load_hotpotqa
from scripts.internal.baselines import (
    BM25RetrievalBaseline,
    DenseRetrievalBaseline,
    FullContextBaseline,
    run_all_baselines,
)
from src.compressor import SentenceAutoencoder
from src.evaluation import compare_results, print_results
from src.validator import ValidationRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

def load_autoencoder(cfg, checkpoint_path, device):
    comp_cfg  = cfg["compressor"]
    dec_cfg   = cfg.get("decoder", {})
    query_cfg = cfg.get("query", {})
    autoencoder = SentenceAutoencoder(
        model_name=cfg["model"]["compressor_name"],
        compress_lora_r=comp_cfg["lora_r"],
        compress_lora_alpha=comp_cfg["lora_alpha"],
        decode_lora_r=dec_cfg.get("lora_r", comp_cfg["lora_r"]),
        decode_lora_alpha=dec_cfg.get("lora_alpha", comp_cfg["lora_alpha"]),
        query_lora_r=query_cfg.get("lora_r", comp_cfg["lora_r"]),
        query_lora_alpha=query_cfg.get("lora_alpha", comp_cfg["lora_alpha"]),
        query_target_modules=query_cfg.get("target_modules", comp_cfg.get("target_modules")),
        lora_dropout=comp_cfg.get("lora_dropout", 0.05),
        compress_target_modules=comp_cfg.get("target_modules"),
        decode_target_modules=dec_cfg.get("target_modules"),
        retrieval_dim=comp_cfg["retrieval_dim"],
        generator_hidden=cfg["model"]["generator_hidden"],
        num_latent_tokens=comp_cfg.get("num_latent_tokens", 1),
    )
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading autoencoder checkpoint: {checkpoint_path}")
        autoencoder.load(checkpoint_path)
    else:
        logger.warning(
            f"Checkpoint not found at '{checkpoint_path}'. "
            "Running with untrained weights (expect poor latent retrieval metrics)."
        )
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    return autoencoder


def load_generator(cfg, device):
    gen_name = cfg["model"]["generator_name"]
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(cfg["model"].get("dtype", "bfloat16"), torch.bfloat16)

    logger.info(f"Loading frozen generator: {gen_name}")
    tokenizer = AutoTokenizer.from_pretrained(gen_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        gen_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Per-dataset evaluation helper
# ---------------------------------------------------------------------------

def _evaluate_dataset(
    dataset_name: str,
    samples: list,
    runner,
    generator,
    gen_tokenizer,
    top_k: int,
    max_new_tokens: int,
    device: str,
    output_dir: str,
    skip_baselines: bool,
    skip_latent: bool = False,
    compression_cfg: dict = None,
    baseline_batch_size: int = 16,
    dump_full_examples: int = 0,
) -> dict:
    """Run latent retrieval + optional baselines on one dataset. Returns results dict."""
    results = {}
    tag = dataset_name

    logger.info("=" * 60)
    logger.info(f"Dataset: {tag}  ({len(samples)} samples, top_k={top_k})")

    # Latent Retrieval
    if not skip_latent:
        logger.info("Running: Latent Retrieval (proposed)")
        latent_metrics = runner.run(samples)
        results["Latent Retrieval (ours)"] = latent_metrics
        print_results(latent_metrics, f"[{tag}] Latent Retrieval (ours)")

    # Baselines
    if not skip_baselines:
        logger.info("Running baselines…")
        baseline_results = run_all_baselines(
            samples=samples,
            generator=generator,
            generator_tokenizer=gen_tokenizer,
            top_k=top_k,
            device=device,
            output_dir=os.path.join(output_dir, tag),
            max_new_tokens=max_new_tokens,
            compression_cfg=compression_cfg,
            skip_standard=False,
            batch_size=baseline_batch_size,
            dump_full_examples=dump_full_examples,
        )
        results.update(baseline_results)

    logger.info("=" * 60)
    compare_results(results)
    return results


# ---------------------------------------------------------------------------
# Multi-k sweep evaluation
# ---------------------------------------------------------------------------

def _evaluate_dataset_multi_k(
    dataset_name: str,
    samples: list,
    autoencoder,
    generator,
    gen_tokenizer,
    k_list: list,
    max_new_tokens: int,
    device: str,
    output_dir: str,
    skip_baselines: bool,
    skip_latent: bool = False,
    compression_cfg: dict = None,
    baseline_batch_size: int = 16,
    latent_compress_batch_size: int = 128,
    latent_gen_batch_size: int = 16,
    dump_full_examples: int = 0,
) -> dict:
    """
    Evaluate on one dataset across multiple values of k.

    FullContext is run once (k-independent).
    Latent Retrieval, BM25, and Dense are each run for every k in k_list.

    Returns
    -------
    {
        "FullContext":  {metrics},
        "k=1":  {"Latent Retrieval (ours)": {metrics}, "BM25Retrieval": {...}, "DenseRetrieval": {...}},
        "k=3":  {...},
        ...
    }
    """
    tag = dataset_name
    results = {}

    logger.info("=" * 60)
    logger.info(f"Dataset: {tag}  ({len(samples)} samples)  k_list={k_list}")

    # --- FullContext: run once (k-independent), even in baseline-only mode ---
    if not skip_baselines:
        logger.info("Running FullContext (once)...")
        fc = FullContextBaseline(
            generator,
            gen_tokenizer,
            max_new_tokens,
            device,
            batch_size=baseline_batch_size,
            dump_examples=dump_full_examples,
            dump_dir=os.path.join(output_dir, tag),
        )
        fc_metrics = fc.run(samples)
        results["FullContext"] = fc_metrics
        print_results(fc_metrics, f"[{tag}] FullContext")

    # --- Latent Retrieval: compress once, evaluate at all k values in one pass ---
    all_latent_metrics: Dict[int, dict] = {}
    if not skip_latent:
        logger.info("Running Latent Retrieval (all k values, single compression pass)...")
        runner = ValidationRunner(
            autoencoder=autoencoder,
            generator=generator,
            generator_tokenizer=gen_tokenizer,
            top_k=k_list[0],          # overridden inside run_multi_k
            max_new_tokens=max_new_tokens,
            device=device,
            compress_batch_size=latent_compress_batch_size,
            gen_batch_size=latent_gen_batch_size,
            dump_examples=dump_full_examples,
            dump_dir=os.path.join(output_dir, tag),
        )
        all_latent_metrics = runner.run_multi_k(samples, k_list)
        for k in k_list:
            print_results(all_latent_metrics[k], f"[{tag}] Latent k={k}")

    # --- Pre-build Dense baseline once (model load is expensive) ---
    dense_baseline = None
    if not skip_baselines:
        try:
            dense_baseline = DenseRetrievalBaseline(
                generator, gen_tokenizer,
                top_k=k_list[0],
                max_new_tokens=max_new_tokens,
                device=device,
                batch_size=baseline_batch_size,
                dump_examples=dump_full_examples,
                dump_dir=os.path.join(output_dir, tag),
            )
        except ImportError as e:
            logger.warning(f"DenseRetrievalBaseline unavailable: {e}")

    # --- Per-k: BM25 + Dense (Latent results already computed above) ---
    for k in k_list:
        logger.info(f"--- k={k} ---")
        k_results = {}

        if not skip_latent:
            k_results["Latent Retrieval (ours)"] = all_latent_metrics[k]

        if not skip_baselines:
            bm25 = BM25RetrievalBaseline(
                generator, gen_tokenizer, k, max_new_tokens, device,
                batch_size=baseline_batch_size,
                dump_examples=dump_full_examples,
                dump_dir=os.path.join(output_dir, tag),
                dump_name_suffix=f"_k{k}",
            )
            bm25_metrics = bm25.run(samples)
            k_results["BM25Retrieval"] = bm25_metrics
            print_results(bm25_metrics, f"[{tag}] BM25 k={k}")

            if dense_baseline is not None:
                dense_baseline.top_k = k
                dense_baseline.dump_name_suffix = f"_k{k}"
                dense_metrics = dense_baseline.run(samples)
                k_results["DenseRetrieval"] = dense_metrics
                print_results(dense_metrics, f"[{tag}] Dense k={k}")

        results[f"k={k}"] = k_results

        # Save incrementally after each k
        ds_path = os.path.join(output_dir, f"{tag}_multik_results.json")
        with open(ds_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_samples(ds_name, max_samples, cache_dir):
    """Load and filter samples for a named dataset."""
    if ds_name == "hotpotqa":
        samples = load_hotpotqa(
            split="validation",
            max_samples=max_samples,
            cache_dir=cache_dir,
        )
    elif ds_name == "webqa_text":
        # WebQA text portion — load from data/ directory
        json_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "webqa_val.json"
        )
        from data.prepare_data import load_webqa_text
        samples = load_webqa_text(json_path, max_samples=max_samples or -1)
    else:
        samples = load_dataset_by_name(
            ds_name,
            split=None,
            max_samples=max_samples or 500,
            cache_dir=cache_dir,
        )
        samples = [s for s in samples if s.get("sentences")]
        logger.info(f"  -> {len(samples)} samples with sentence structure")
    return samples


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    os.makedirs(args.output, exist_ok=True)

    max_new_tokens = cfg["generation"]["max_new_tokens"]
    cache_dir      = cfg["data"].get("cache_dir", "data/cache/")

    # --- Build compression_cfg from CLI flags + config.yaml ---
    compression_cfg = None
    if getattr(args, "compression_baselines", None):
        requested = [b.strip().lower() for b in args.compression_baselines.split(",") if b.strip()]
        cfg_comp  = cfg.get("compression_baselines", {})
        compression_cfg = {}
        if "llmlingua" in requested:
            entry = dict(cfg_comp.get("llmlingua", {}))
            if getattr(args, "llmlingua_model", None):
                entry["model"] = args.llmlingua_model
            compression_cfg["llmlingua"] = entry
        if "lcc" in requested:
            entry = dict(cfg_comp.get("lcc", {}))
            if getattr(args, "lcc_checkpoint", None):
                entry["checkpoint"] = args.lcc_checkpoint
            compression_cfg["lcc"] = entry
        if "xrag" in requested:
            entry = dict(cfg_comp.get("xrag", {}))
            if getattr(args, "xrag_model", None):
                entry["model"] = args.xrag_model
            compression_cfg["xrag"] = entry

    # --- Parse k_list ---
    k_list = None
    if args.k_list:
        k_list = [int(k.strip()) for k in args.k_list.split(",") if k.strip()]

    # --- Build dataset list ---
    if args.dataset:
        dataset_names = [d.strip() for d in args.dataset.split(",") if d.strip()]
    elif args.ood_datasets:
        dataset_names = []   # OOD-only run; hotpotqa not added automatically
    else:
        dataset_names = ["hotpotqa"]   # default when nothing specified
    if args.ood_datasets:
        dataset_names += [d.strip() for d in args.ood_datasets.split(",") if d.strip()]

    # --- Load models once ---
    skip_latent = getattr(args, "skip_latent", False)
    autoencoder = None if skip_latent else load_autoencoder(cfg, args.checkpoint, device)
    generator, gen_tokenizer = load_generator(cfg, device)

    all_results = {}

    for ds_name in dataset_names:
        logger.info(f"Loading dataset: {ds_name}")
        samples = _load_samples(ds_name, args.max_samples, cache_dir)
        if not samples:
            logger.warning(f"No usable samples for {ds_name}, skipping.")
            continue

        if k_list:
            # ---- Multi-k sweep ----
            ds_results = _evaluate_dataset_multi_k(
                dataset_name=ds_name,
                samples=samples,
                autoencoder=autoencoder,
                generator=generator,
                gen_tokenizer=gen_tokenizer,
                k_list=k_list,
                max_new_tokens=max_new_tokens,
                device=device,
                output_dir=args.output,
                skip_baselines=args.skip_baselines,
                skip_latent=skip_latent,
                compression_cfg=compression_cfg,
                baseline_batch_size=args.baseline_batch_size,
                latent_compress_batch_size=args.latent_compress_batch_size,
                latent_gen_batch_size=args.latent_gen_batch_size,
                dump_full_examples=args.dump_full_examples,
            )
        else:
            # ---- Single-k (original mode) ----
            top_k = args.top_k or cfg["retrieval"]["top_k"]
            runner = None
            if not skip_latent:
                runner = ValidationRunner(
                    autoencoder=autoencoder,
                    generator=generator,
                    generator_tokenizer=gen_tokenizer,
                    top_k=top_k,
                    max_new_tokens=max_new_tokens,
                    device=device,
                    compress_batch_size=args.latent_compress_batch_size,
                    gen_batch_size=args.latent_gen_batch_size,
                    dump_examples=args.dump_full_examples,
                    dump_dir=os.path.join(args.output, ds_name),
                )
            ds_results = _evaluate_dataset(
                dataset_name=ds_name,
                samples=samples,
                runner=runner,
                generator=generator,
                gen_tokenizer=gen_tokenizer,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                device=device,
                output_dir=args.output,
                skip_baselines=args.skip_baselines,
                skip_latent=skip_latent,
                compression_cfg=compression_cfg,
                baseline_batch_size=args.baseline_batch_size,
                dump_full_examples=args.dump_full_examples,
            )
            ds_path = os.path.join(args.output, f"{ds_name}_results.json")
            with open(ds_path, "w") as f:
                json.dump(ds_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {ds_path}")

        all_results[ds_name] = ds_results

    summary_path = os.path.join(args.output, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"All results saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent retrieval vs baselines — multi-dataset / multi-k")
    parser.add_argument("--config",         default="config.yaml",               help="Path to config.yaml")
    parser.add_argument("--checkpoint",     default="checkpoints/model_best.pt", help="Trained autoencoder checkpoint")
    parser.add_argument("--output",         default="results/",                  help="Directory for result files")
    parser.add_argument("--dataset",        default=None,                        help="Primary dataset or comma-separated dataset list (default: hotpotqa); also supports: nq, triviaqa, qasper_longbench, wice, truthfulqa, webqa_text")
    parser.add_argument("--ood_datasets",   default=None,                        help="Comma-separated extra datasets (e.g. '2wikimultihopqa,musique,nq,triviaqa,qasper_longbench,wice')")
    parser.add_argument("--max_samples",    type=int, default=None,              help="Max samples per dataset")
    parser.add_argument("--top_k",          type=int, default=None,              help="k for single-k mode (overrides config)")
    parser.add_argument("--k_list",         default=None,                        help="Comma-separated k values for multi-k sweep (e.g. '1,2,3,5,10')")
    parser.add_argument("--skip_baselines", action="store_true",                 help="Skip all baseline evaluation (BM25, Dense, FullContext, compression)")
    parser.add_argument("--skip_latent",    action="store_true",                 help="Skip latent retrieval; run only baselines (no autoencoder checkpoint needed)")
    parser.add_argument("--baseline_batch_size", type=int, default=16,           help="Batch size for FullContext/BM25/Dense generation")
    parser.add_argument("--latent_compress_batch_size", type=int, default=128,   help="Batch size for latent validation compression/embedding passes")
    parser.add_argument("--latent_gen_batch_size", type=int, default=16,         help="Batch size for latent validation generation")
    parser.add_argument("--dump_full_examples", type=int, default=0,             help="Dump first N FullContext formatted examples to JSON")
    parser.add_argument(
        "--compression_baselines",
        default=None,
        help=(
            "Comma-separated list of compression baselines to run. "
            "Options: llmlingua,lcc,xrag"
        ),
    )
    parser.add_argument("--lcc_checkpoint", default=None, help="Override LCC compiler checkpoint path")
    parser.add_argument("--llmlingua_model", default=None, help="Override LLMLingua-2 model path")
    parser.add_argument("--xrag_model",      default=None, help="Override xRAG model path")
    args = parser.parse_args()
    main(args)

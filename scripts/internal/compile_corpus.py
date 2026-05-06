"""
Compile Corpus
==============
Offline step: compile all documents in the corpus into latent memories
and build the FAISS retrieval index.

This is a one-time preprocessing step. After running this script,
the memory bank is ready for fast query-time retrieval.

Steps
-----
1. Load trained compiler checkpoint
2. For each document: compile → get (N, H) latent memory + retrieval vector
3. Store everything in the MemoryBank
4. Build FAISS index
5. Save memory bank to disk

Usage
-----
python scripts/compile_corpus.py \
    --config config.yaml \
    --compiler checkpoints/compiler.pt \
    --data data/samples.json \
    --output memory_bank/
"""

import argparse
import logging
import os
import sys

_RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _RELEASE_ROOT not in sys.path:
    sys.path.insert(0, _RELEASE_ROOT)

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.compiler import LCCCompiler
from src.memory_bank import MemoryBank
from data.prepare_data import load_dataset_by_name, load_samples

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compile_corpus(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # --- Load data ---
    if os.path.exists(args.data):
        samples = load_samples(args.data)
    else:
        samples = load_dataset_by_name(
            cfg["data"]["dataset"],
            split=cfg["data"]["split"],
            max_samples=cfg["data"]["max_samples"],
            cache_dir=cfg["data"]["cache_dir"],
        )
    logger.info(f"Compiling {len(samples)} documents...")

    # --- Load compiler ---
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

    if args.compiler and os.path.exists(args.compiler):
        compiler.load(args.compiler)
        logger.info(f"Loaded compiler from {args.compiler}")
    else:
        logger.warning(
            "No compiler checkpoint found — using untrained compiler. "
            "Run train_compiler.py first for best results."
        )

    compiler.eval()

    # --- Compile all documents ---
    bank = MemoryBank(
        retrieval_dim=comp_cfg["retrieval_dim"],
        index_type=cfg["retrieval"]["index_type"],
    )

    # Deduplicate by document text (multiple QA pairs may share a document)
    seen_docs = {}
    doc_records = []
    for s in samples:
        doc_key = s["document"][:200]  # first 200 chars as dedup key
        if doc_key not in seen_docs:
            seen_docs[doc_key] = s["id"]
            doc_records.append({"id": s["id"], "text": s["document"]})

    logger.info(f"Unique documents to compile: {len(doc_records)}")

    for record in tqdm(doc_records, desc="Compiling"):
        doc_id = record["id"]
        doc_text = record["text"]

        try:
            # Compile document → latent memory
            with torch.no_grad():
                memory = compiler.compile(doc_text)               # (N, H)
                retrieval_vec = compiler.get_retrieval_embedding(memory)  # (D,)

            bank.add(doc_id, memory, retrieval_vec, text=doc_text)

        except Exception as e:
            logger.warning(f"Failed to compile doc {doc_id}: {e}")
            continue

    # --- Build FAISS index ---
    logger.info("Building FAISS index...")
    bank.build_index()

    # --- Save ---
    os.makedirs(args.output, exist_ok=True)
    bank.save(args.output)
    logger.info(f"Memory bank saved to {args.output}")
    logger.info(f"Total memories: {len(bank)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--compiler", default="checkpoints/compiler.pt")
    parser.add_argument("--data", default="data/samples.json")
    parser.add_argument("--output", default="memory_bank/")
    args = parser.parse_args()
    compile_corpus(args)

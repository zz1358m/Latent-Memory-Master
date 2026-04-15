"""
Build Sentence-Level FAISS Index
==================================
Offline step: split all corpus documents into sentences, compress each
sentence into 1 token embedding, and store in a FAISS index for fast retrieval.

Usage
-----
python scripts/index_corpus.py \
    --config config.yaml \
    --checkpoint checkpoints/model.pt \
    --data data/samples.json \
    --output sentence_index/
"""

import argparse
import json
import logging
import os
import sys

import faiss
import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.prepare_data import load_dataset_by_name, load_samples
from data.sentence_utils import split_sentences_by_tokens
from src.compressor import SentenceAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_index(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load data ---
    if os.path.exists(args.data):
        raw_samples = load_samples(args.data)
    else:
        raw_samples = load_dataset_by_name(
            cfg["data"]["dataset"],
            cfg["data"]["split"],
            cfg["data"]["max_samples"],
            cfg["data"]["cache_dir"],
        )

    # --- Load model ---
    comp_cfg = cfg["compressor"]
    dec_cfg  = cfg["decoder"]
    autoencoder = SentenceAutoencoder(
        model_name=cfg["model"]["compressor_name"],
        compress_lora_r=comp_cfg["lora_r"],
        compress_lora_alpha=comp_cfg["lora_alpha"],
        decode_lora_r=dec_cfg["lora_r"],
        decode_lora_alpha=dec_cfg["lora_alpha"],
        lora_dropout=comp_cfg.get("lora_dropout", 0.05),
        compress_target_modules=comp_cfg.get("target_modules"),
        decode_target_modules=dec_cfg.get("target_modules"),
        retrieval_dim=comp_cfg["retrieval_dim"],
        generator_hidden=cfg["model"].get("generator_hidden", 4096),
        num_latent_tokens=comp_cfg.get("num_latent_tokens", 1),
    ).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        autoencoder.load(args.checkpoint)
    else:
        logger.warning("No checkpoint found — using untrained model.")
    autoencoder.eval()

    # --- Deduplicate documents ---
    seen, docs = set(), []
    for s in raw_samples:
        key = s["document"][:200]
        if key not in seen:
            seen.add(key)
            docs.append({"id": s["id"], "text": s["document"]})
    logger.info(f"Unique documents: {len(docs)}")

    # --- Compress all sentences ---
    retrieval_dim = comp_cfg["retrieval_dim"]
    all_vecs, metadata = [], []

    for doc in tqdm(docs, desc="Indexing"):
        sentences = split_sentences_by_tokens(
            doc["text"],
            autoencoder.tokenizer,
            max_tokens=cfg["data"]["max_sentence_length"],
            min_words=cfg["data"]["min_sentence_length"],
        )

        for idx, sent in enumerate(sentences):
            try:
                with torch.no_grad():
                    compressed = autoencoder.compress(sent, with_grad=False)    # (T, H)
                    ret_vec = autoencoder.get_retrieval_embedding(compressed.unsqueeze(0)).squeeze(0)    # (D,)

                all_vecs.append(ret_vec.cpu().numpy())
                metadata.append({
                    "doc_id":    doc["id"],
                    "sent_idx":  idx,
                    "sentence":  sent,
                    # Store the compressed embedding for LLM injection at inference
                    "latent":    compressed.cpu().numpy().tolist(),
                })
            except Exception as e:
                logger.warning(f"Skipping sentence in {doc['id']}: {e}")

    if not all_vecs:
        raise RuntimeError("No sentences were compressed. Check the data and model.")

    vecs = np.stack(all_vecs, axis=0).astype(np.float32)  # (N, D)
    logger.info(f"Total sentences indexed: {len(vecs)}")

    # --- Build FAISS index ---
    index_type = cfg["retrieval"]["index_type"]
    if index_type == "flat":
        index = faiss.IndexFlatIP(retrieval_dim)
    elif index_type == "ivf":
        nlist = min(100, len(vecs) // 10)
        quantizer = faiss.IndexFlatIP(retrieval_dim)
        index = faiss.IndexIVFFlat(quantizer, retrieval_dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vecs)
    else:
        raise ValueError(f"Unknown index_type: {index_type}")

    index.add(vecs)

    # --- Save ---
    os.makedirs(args.output, exist_ok=True)
    faiss.write_index(index, os.path.join(args.output, "faiss.index"))
    with open(os.path.join(args.output, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    logger.info(f"Index saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--data",       default="data/samples.json")
    parser.add_argument("--output",     default="sentence_index/")
    args = parser.parse_args()
    build_index(args)

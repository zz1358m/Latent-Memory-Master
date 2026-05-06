"""
Compile WebQA Region Corpus
============================
Offline step: encode all WebQA image regions into latent memories and build
a FAISS retrieval index for region-level retrieval.

Each image is split into spatial regions (1/8-scaled, 72×72 px), and each
region is compressed by LLaVACompressor → (1, hidden_dim) latent stored in
the MemoryBank with its L2-normalised retrieval vector.

Usage
-----
python scripts/compile_webqa_corpus.py \\
    --config config_llava.yaml \\
    --checkpoint checkpoints/llava_model_best.pt \\
    --data data/webqa_train.json \\
    --output memory_bank_webqa/

# Merge with existing text bank for joint retrieval:
python scripts/compile_webqa_corpus.py \\
    --config config_llava.yaml \\
    --checkpoint checkpoints/llava_model_best.pt \\
    --data data/webqa_train.json \\
    --merge_text_bank memory_bank/ \\
    --output memory_bank_joint/
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
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.prepare_webqa import load_webqa_samples
from src.llava_compressor import LLaVACompressor
from src.memory_bank import MemoryBank
from src.region_encoder import image_doc_id, resize_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compile_corpus(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = cfg["model"]
    comp_cfg  = cfg["compressor"]
    dec_cfg   = cfg.get("decoder", {})
    img_cfg   = cfg.get("image", {})

    # ---- compressor ----
    logger.info("Loading LLaVACompressor...")
    compressor = LLaVACompressor(
        model_name       = model_cfg["compressor_name"],
        decoder_name     = model_cfg.get("decoder_name", "meta-llama/Llama-3.2-1B-Instruct"),
        lora_r           = comp_cfg["lora_r"],
        lora_alpha       = comp_cfg["lora_alpha"],
        lora_dropout     = comp_cfg.get("lora_dropout", 0.05),
        target_modules   = comp_cfg.get("target_modules"),
        decode_lora_r    = dec_cfg.get("lora_r", comp_cfg["lora_r"]),
        decode_lora_alpha= dec_cfg.get("lora_alpha", comp_cfg["lora_alpha"]),
        decode_lora_dropout= dec_cfg.get("lora_dropout", comp_cfg.get("lora_dropout", 0.05)),
        decode_target_modules= dec_cfg.get("target_modules", comp_cfg.get("target_modules")),
        retrieval_dim    = comp_cfg["retrieval_dim"],
        generator_hidden = model_cfg.get("generator_hidden", 5120),
        num_latent_tokens= comp_cfg.get("num_latent_tokens", 1),
    ).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        compressor.load(args.checkpoint)
    else:
        logger.warning("No checkpoint — using untrained weights.")
    compressor.eval()

    # ---- data ----
    samples = load_webqa_samples(args.data)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    logger.info(f"Samples: {len(samples)}")

    # ---- memory bank ----
    retrieval_dim = comp_cfg["retrieval_dim"]
    index_type    = cfg.get("retrieval", {}).get("index_type", "flat")

    if args.merge_text_bank and os.path.exists(args.merge_text_bank):
        bank = MemoryBank.load(args.merge_text_bank)
        logger.info(f"Loaded text bank: {len(bank)} entries")
    else:
        bank = MemoryBank(retrieval_dim=retrieval_dim, index_type=index_type)

    # ---- image encoding params ----
    scale = img_cfg.get("scale", 1 / 20)

    # ---- collect unique images ----
    seen_img_ids = set(e.doc_id for e in bank.entries)
    queue = []
    for s in samples:
        img_dir = s["image_dir"]
        for img_id, cap in zip(s["pos_image_ids"], s["pos_captions"]):
            if img_id not in seen_img_ids:
                queue.append({"image_id": img_id, "caption": cap, "image_dir": img_dir})
                seen_img_ids.add(img_id)

    logger.info(f"Unique new images to encode: {len(queue)}")
    n_encoded = 0
    n_skipped = 0

    from src.llava_compressor import LLaVACompressor as _C
    for item in tqdm(queue, desc="Encoding images"):
        img_id  = item["image_id"]
        caption = item["caption"]
        img_dir = item["image_dir"]

        path = os.path.join(img_dir, f"{img_id}.jpg")
        if not os.path.exists(path):
            n_skipped += 1
            continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            logger.debug(f"Skipping {img_id}: {e}")
            n_skipped += 1
            continue

        pil_small = resize_image(img, scale)
        doc_id    = image_doc_id(img_id)
        prompt    = _C.format_compress_prompt(caption, has_image=True)

        with torch.no_grad():
            latent = compressor.compress_batch(
                [prompt], images=[pil_small], with_grad=False, adapter="compress"
            )   # (1, T, hidden_size)
            rvec = compressor.get_retrieval_embedding(latent).squeeze(0)   # (retrieval_dim,)

        bank.add(doc_id, latent.squeeze(0).cpu(), rvec.cpu(), text=caption)
        n_encoded += 1

    logger.info(f"Encoded {n_encoded} images | skipped {n_skipped} images")
    logger.info(f"Total bank entries: {len(bank)}")

    if len(bank) == 0:
        logger.error("Empty bank — nothing to save.")
        return

    logger.info("Building FAISS index...")
    bank.build_index()
    os.makedirs(args.output, exist_ok=True)
    bank.save(args.output)
    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile WebQA image regions into a latent memory bank.")
    parser.add_argument("--config",          default="config_llava.yaml")
    parser.add_argument("--checkpoint",      default="checkpoints/llava_model_best.pt")
    parser.add_argument("--data",            default="data/webqa_train.json")
    parser.add_argument("--output",          default="memory_bank_webqa/")
    parser.add_argument("--merge_text_bank", default=None,
                        help="Existing text MemoryBank directory to extend")
    parser.add_argument("--max_samples",     type=int, default=-1)
    args = parser.parse_args()
    compile_corpus(args)

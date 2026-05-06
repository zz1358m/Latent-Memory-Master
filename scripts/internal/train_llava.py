"""
LLaVA-based Latent Retrieval Training
======================================
Training script for the LLaVA-v1.5-7b compressor on WebQA.

WebQA has two modalities handled jointly:
  - image samples (~21k): positive evidence is image regions (1/8-scale, 72×72px)
  - text  samples (~20k): positive evidence is text fact snippets

Losses
------
  Image samples:
    L_contrast (λ=0.5): NT-Xent between query emb and region embs
    L_distill  (λ=0.3): KL(LLaVA-13b(caption) || LLaVA-13b(proj latent))

  Text samples:
    L_recon    (λ=1.0): CE on fact text reconstruction (compress + decode)
    L_contrast (λ=0.5): NT-Xent between query emb and fact embs
    L_distill  (λ=0.3): KL(LLaVA-13b(Evidence N: text) || LLaVA-13b(Latent N: proj))

Usage
-----
python scripts/train_llava.py \\
    --config config_llava.yaml \\
    --data   data/webqa_train.json \\
    --val    data/webqa_val.json \\
    --output checkpoints/llava_model.pt
"""

import argparse
import logging
import os
import random
import sys

_RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _RELEASE_ROOT not in sys.path:
    sys.path.insert(0, _RELEASE_ROOT)
from typing import Dict, List, Optional

import torch
import yaml
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter

    _TB = True
except ImportError:
    _TB = False
    SummaryWriter = None

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from data.prepare_webqa import load_webqa_samples
from src.distillation import _split_chat_template, _tok
from src.llava_compressor import LLaVACompressor
from src.region_encoder import image_doc_id, resize_image
from src.retriever import ContrastiveRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batch builders
# ---------------------------------------------------------------------------

def build_text_batch(
        samples: List[Dict],
        batch_size: int,
        max_hard_negatives: int = 32,
        rng: Optional[random.Random] = None,
) -> List[Dict]:
    """Sample a mini-batch from WebQA text-modality samples."""
    if rng is None:
        rng = random.Random()
    chosen = rng.choices(samples, k=batch_size)
    batch = []
    for s in chosen:
        pos_facts = s.get("txt_pos_facts", [])
        if not pos_facts:
            continue
        neg_facts = s.get("txt_neg_facts", [])[:max_hard_negatives]
        batch.append({
            "question": s["question"],
            "pos_sentences": pos_facts,
            "hard_neg_sents": neg_facts,
            "answers": s["answers"],
        })
    return batch


def build_image_batch(
        samples: List[Dict],
        batch_size: int,
        max_neg_images: int = 16,
        scale: float = 1 / 20,
        rng: Optional[random.Random] = None,
) -> List[Dict]:
    """
    Sample a mini-batch from WebQA image-modality samples.
    Each image is resized to 1/20 × 1/20 and used as a single sample (no splitting).
    """
    if rng is None:
        rng = random.Random()
    chosen = rng.choices(samples, k=batch_size)
    batch = []
    for s in chosen:
        img_dir = s["image_dir"]

        def _load_image(img_id: str, title: str):
            path = os.path.join(img_dir, f"{img_id}.jpg")
            if not os.path.exists(path):
                return None
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                return None
            return (resize_image(img, scale), image_doc_id(img_id), title)

        # Positive images
        pos_images = []
        for img_id, title in zip(s["pos_image_ids"], s["pos_captions"]):
            item = _load_image(img_id, title)
            if item is not None:
                pos_images.append(item)

        if not pos_images:
            continue

        # Negative images (randomly capped)
        neg_images = []
        for img_id, title in zip(s["neg_image_ids"][:max_neg_images],
                                 s["neg_captions"][:max_neg_images]):
            item = _load_image(img_id, title)
            if item is not None:
                neg_images.append(item)

        batch.append({
            "question": s["question"],
            "answers": s["answers"],
            "pos_regions": pos_images,  # list of (PIL, doc_id, title)
            "neg_regions": neg_images,
        })
    return batch


def build_multimodal_batch(
        samples: List[Dict],
        batch_size: int,
        max_hard_negatives: int = 32,
        max_neg_images: int = 16,
        scale: float = 1 / 20,
        rng: Optional[random.Random] = None,
) -> List[Dict]:
    if rng is None:
        rng = random.Random()
    chosen = rng.choices(samples, k=batch_size)
    batch = []
    for s in chosen:
        img_dir = s["image_dir"]

        def _load_image(img_id: str, title: str):
            path = os.path.join(img_dir, f"{img_id}.jpg")
            if not os.path.exists(path):
                return None
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                return None
            return (resize_image(img, scale), image_doc_id(img_id), title)

        pos_regions = []
        for img_id, title in zip(s.get("pos_image_ids", []), s.get("pos_captions", [])):
            item = _load_image(img_id, title)
            if item is not None:
                pos_regions.append(item)

        neg_regions = []
        for img_id, title in zip(
                s.get("neg_image_ids", [])[:max_neg_images],
                s.get("neg_captions", [])[:max_neg_images],
        ):
            item = _load_image(img_id, title)
            if item is not None:
                neg_regions.append(item)

        pos_sentences = list(s.get("txt_pos_facts", []))
        neg_sentences = list(s.get("txt_neg_facts", [])[:max_hard_negatives])

        if not pos_regions and not pos_sentences:
            continue

        batch.append({
            "question": s["question"],
            "answers": s["answers"],
            "pos_sentences": pos_sentences,
            "neg_sentences": neg_sentences,
            "pos_regions": pos_regions,
            "neg_regions": neg_regions,
            "positive_modality": "image" if pos_regions else "text",
        })
    return batch


# ---------------------------------------------------------------------------
# Training steps
# ---------------------------------------------------------------------------

def text_train_step(
        compressor: LLaVACompressor,
        retriever: ContrastiveRetriever,
        batch: List[Dict],
        lam_recon: float,
        lam_cont: float,
        device: str,
        lam_distill: float = 0.0,
        generator=None,
        generator_proc=None,
        n_distill_tokens: int = 8,
        recon_negatives: bool = False,
        recon_query: bool = True,
) -> Dict:
    """Text step (WebQA text samples): L_recon + L_contrast [+ L_distill]."""
    questions = [b["question"] for b in batch]
    pos_sents = [s for b in batch for s in b["pos_sentences"]]
    neg_sents = [s for b in batch for s in b["hard_neg_sents"]]

    pos_prompts = [LLaVACompressor.format_compress_prompt(s, False) for s in pos_sents]
    neg_prompts = [LLaVACompressor.format_compress_prompt(s, False) for s in neg_sents]
    q_prompts = [LLaVACompressor.format_query_prompt(q) for q in questions]

    # L_recon: CE reconstruction of raw sentences (+ negatives if recon_negatives=true)
    recon_sents = pos_sents + neg_sents if recon_negatives else pos_sents
    l_recon_sent = compressor.reconstruction_loss_batch(recon_sents, adapter="compress", decode_adapter="decode")
    l_recon_query = (
        compressor.reconstruction_loss_batch(questions, adapter="query", decode_adapter="query_decode")
        if recon_query else torch.tensor(0.0, device=device)
    )
    l_recon = l_recon_sent + l_recon_query

    # Embeddings
    pos_latents = compressor.compress_batch(pos_prompts, with_grad=True, adapter="compress")
    pos_embs = compressor.get_retrieval_embedding(pos_latents)

    q_latents = compressor.embed_query_batch(questions, with_grad=True)
    q_embs = compressor.get_retrieval_embedding(q_latents)

    # Per-sample positive lists
    pos_counts = [len(b["pos_sentences"]) for b in batch]
    pos_embs_list = []
    offset = 0
    for c in pos_counts:
        pos_embs_list.append(pos_embs[offset:offset + c])
        offset += c

    neg_embs = None
    if neg_sents:
        neg_latents = compressor.compress_batch(neg_prompts, with_grad=True, adapter="compress")
        neg_embs = compressor.get_retrieval_embedding(neg_latents)

    l_contrast = retriever.contrastive_loss(q_embs, pos_embs_list, neg_embs)

    # L_distill: KL(teacher text evidence || student latent tokens)
    l_distill = torch.tensor(0.0, device=device)
    if lam_distill > 0 and generator is not None and pos_latents.shape[0] > 0:
        proj_latents = compressor.project_for_generator(pos_latents)
        l_distill = _text_distill_loss(
            generator, generator_proc,
            questions, pos_sents, proj_latents,
            pos_counts, device, n_distill_tokens,
        )

    total = (
            lam_recon * l_recon
            + lam_cont * l_contrast
            + lam_distill * l_distill
    )

    return {
        "total": total,
        "recon_sent": l_recon_sent.item(),
        "recon_query": l_recon_query.item(),
        "contrast": l_contrast.item(),
        "distill": l_distill.item() if isinstance(l_distill, torch.Tensor) else 0.0,
    }


def image_train_step(
        compressor: LLaVACompressor,
        retriever: ContrastiveRetriever,
        batch: List[Dict],
        lam_cont: float,
        lam_distill: float,
        device: str,
        generator=None,
        generator_proc=None,
        n_distill_tokens: int = 8,
        lambda_embed_recon: float = 0.0,
        lam_recon: float = 1.0,
        recon_query: bool = True,
        recon_title: bool = True,
        recon_negatives: bool = False,
) -> Dict:
    """Image step (WebQA image samples): L_contrast + L_recon_title + L_recon_query + L_recon_clip [+ L_distill]."""
    questions = [b["question"] for b in batch]

    # Positive regions (with grad) — tuples are (PIL, doc_id, title)
    pos_pil = [r[0] for b in batch for r in b["pos_regions"]]
    pos_titles = [r[2] for b in batch for r in b["pos_regions"]]
    pos_prompts = [LLaVACompressor.format_compress_prompt(t, True) for t in pos_titles]

    if not pos_pil:
        z = torch.tensor(0.0, device=device)
        return {"total": z, "contrast": 0.0, "distill": 0.0,
                "recon_title": 0.0, "recon_query": 0.0, "recon_clip": 0.0}

    pos_latents = compressor.compress_batch(
        pos_prompts, images=pos_pil, with_grad=True, adapter="compress"
    )
    pos_embs = compressor.get_retrieval_embedding(pos_latents)

    pos_counts = [len(b["pos_regions"]) for b in batch]
    pos_embs_list = []
    offset = 0
    for c in pos_counts:
        pos_embs_list.append(pos_embs[offset:offset + c])
        offset += c

    # Negative regions (with grad — gradients flow through neg denominator of InfoNCE)
    neg_embs = None
    neg_pil = [r[0] for b in batch for r in b["neg_regions"]]
    neg_titles = [r[2] for b in batch for r in b["neg_regions"]]
    if neg_pil:
        neg_prompts = [LLaVACompressor.format_compress_prompt(t, True) for t in neg_titles]
        neg_latents = compressor.compress_batch(
            neg_prompts, images=neg_pil, with_grad=True, adapter="compress"
        )
        neg_embs = compressor.get_retrieval_embedding(neg_latents)

    # L_recon_title: image latent → reconstruct caption text (image_decode adapter)
    # L_recon_clip:  image latent → reconstruct CLIP CLS hidden (img_embed_decode_proj MLP)
    if recon_title:
        recon_prompts = pos_prompts
        recon_pil = pos_pil
        recon_titles = pos_titles
        if recon_negatives and neg_pil:
            recon_prompts = pos_prompts + neg_prompts
            recon_pil = pos_pil + neg_pil
            recon_titles = pos_titles + neg_titles
        l_recon_title = compressor.reconstruction_loss_batch(
            recon_prompts, images=recon_pil, adapter="compress", decode_adapter="image_decode",
            target_texts=recon_titles,
        )
    else:
        l_recon_title = torch.tensor(0.0, device=device)

    clip_prompts = pos_prompts + neg_prompts if (recon_negatives and neg_pil) else pos_prompts
    clip_pil = pos_pil + neg_pil if (recon_negatives and neg_pil) else pos_pil
    l_recon_clip = compressor.embed_reconstruction_loss_batch(
        clip_prompts, images=clip_pil, adapter="compress",
    )

    l_recon_query = (
        compressor.reconstruction_loss_batch(questions, adapter="query", decode_adapter="query_decode")
        if recon_query else torch.tensor(0.0, device=device)
    )
    q_latents = compressor.embed_query_batch(questions, with_grad=True)
    q_embs = compressor.get_retrieval_embedding(q_latents)

    l_contrast = retriever.contrastive_loss(q_embs, pos_embs_list, neg_embs)

    # L_distill: KL(LLaVA-13b(image+question) || LLaVA-13b(proj_latent+question))
    l_distill = torch.tensor(0.0, device=device)
    if lam_distill > 0 and generator is not None:
        proj_latents = compressor.project_for_generator(pos_latents)
        l_distill = _llava_distill_loss(
            generator, generator_proc,
            questions, pos_titles, pos_pil, proj_latents,
            pos_counts, device, n_distill_tokens,
        )

    total = (
            lam_cont * l_contrast
            + lam_distill * l_distill
            + lam_recon * (l_recon_title + l_recon_query)
            + lambda_embed_recon * l_recon_clip
    )
    return {
        "total": total,
        "recon_title": l_recon_title.item(),
        "recon_query": l_recon_query.item(),
        "recon_clip": l_recon_clip.item(),
        "contrast": l_contrast.item(),
        "distill": l_distill.item() if isinstance(l_distill, torch.Tensor) else 0.0,
    }


def _text_distill_loss(
        generator,
        processor,
        questions: List[str],
        pos_sents: List[str],        # flat list of all positive facts
        student_proj_latents: List[torch.Tensor],
        pos_counts: List[int],
        device: str,
        n_distill_tokens: int,
) -> torch.Tensor:
    """KL divergence for text evidence distillation, aligned with HotpotQA."""
    import torch.nn.functional as F

    embed_fn = generator.get_input_embeddings()
    dtype = next(generator.parameters()).dtype
    eos_id = processor.tokenizer.eos_token_id

    def _e(text: str, max_len: int = 512):
        return embed_fn(_tok(processor.tokenizer, text, device, max_len)).to(dtype)

    # ── Per-sample triples ───────────────────────────────────────────────────
    valid = []
    offset = 0
    for idx, (q, n_pos) in enumerate(zip(questions, pos_counts)):
        if n_pos > 0:
            valid.append(
                (
                    q,
                    pos_sents[offset: offset + n_pos],
                    student_proj_latents[idx],
                )
            )
        offset += n_pos

    if not valid:
        return torch.tensor(0.0, device=device)

    total_kl = torch.tensor(0.0, device=device)
    count = 0

    for q, facts, sample_proj in valid:
        prefix_str, suffix_str = _split_chat_template(processor.tokenizer, q)
        try:
            ctx_text = "\n".join(f"Context {k+1}: {fact}" for k, fact in enumerate(facts))
            t_embeds = _e(prefix_str + ctx_text + suffix_str, max_len=1024)

            with torch.no_grad():
                gen_out = generator.generate(
                    inputs_embeds=t_embeds,
                    max_new_tokens=n_distill_tokens,
                    do_sample=False,
                    pad_token_id=eos_id,
                )
            l_teacher = t_embeds.shape[1]
            gen_ids = (gen_out[0, l_teacher:] if gen_out.shape[1] > l_teacher else gen_out[0])
            gen_ids = gen_ids[:n_distill_tokens]
            n_actual = int(gen_ids.numel())
            if n_actual == 0:
                continue

            tf_embs = None
            if n_actual > 1:
                tf_embs = embed_fn(gen_ids[: n_actual - 1].unsqueeze(0)).to(dtype)
                t_embeds_ext = torch.cat([t_embeds, tf_embs], dim=1)
            else:
                t_embeds_ext = t_embeds

            with torch.no_grad():
                t_out = generator(inputs_embeds=t_embeds_ext, use_cache=False)
            t_logits = t_out.logits[0, -n_actual:, :].float()

        except Exception as e:
            logger.warning(f"Text distill teacher failed: {e}; skipping sample.")
            continue

        # ── Student: prefix + evidence latents + suffix ──────────────────────
        parts = [_e(prefix_str)]
        for k, lat in enumerate(sample_proj):
            label = f"Latent context {k+1}: " if k == 0 else f"\nLatent context {k+1}: "
            parts.append(_e(label))
            parts.append(lat.reshape(-1, lat.shape[-1]).to(dtype).unsqueeze(0))
        parts.append(_e(suffix_str))
        s_embeds = torch.cat(parts, dim=1)

        if tf_embs is not None:
            s_embeds = torch.cat([s_embeds, tf_embs], dim=1)

        s_out = generator(inputs_embeds=s_embeds, use_cache=False)
        s_logits = s_out.logits[0, -n_actual:, :].float()

        kl = F.kl_div(
            F.log_softmax(s_logits.reshape(-1, s_logits.shape[-1]), dim=-1),
            F.softmax(t_logits.reshape(-1, t_logits.shape[-1]).detach(), dim=-1),
            reduction="sum",
        ) / n_actual
        total_kl = total_kl + kl
        count += 1

    return total_kl / max(count, 1)


def _llava_distill_loss(
        generator,
        processor,
        questions: List[str],
        captions: List[str],
        pos_pil: List,           # actual PIL images (parallel to flat pos list)
        student_proj_latents: List[torch.Tensor],
        pos_counts: List[int],
        device: str,
        n_distill_tokens: int,
) -> torch.Tensor:
    """KL divergence for image evidence distillation, aligned per multimodal sample."""
    import torch.nn.functional as F

    image_token = "<image>"

    embed_fn = generator.get_input_embeddings()
    dtype = next(generator.parameters()).dtype
    eos_id = processor.tokenizer.eos_token_id

    def _e(text: str, max_len: int = 512):
        return embed_fn(_tok(processor.tokenizer, text, device, max_len)).to(dtype)

    # ── Per-sample quadruples ────────────────────────────────────────────────
    valid = []
    offset = 0
    for idx, (q, n_pos) in enumerate(zip(questions, pos_counts)):
        if n_pos > 0:
            valid.append((
                q,
                captions[offset: offset + n_pos],
                student_proj_latents[idx],
                pos_pil[offset: offset + n_pos],
            ))
        offset += n_pos

    if not valid:
        return torch.tensor(0.0, device=device)

    total_kl = torch.tensor(0.0, device=device)
    count = 0

    for q, sample_caps, sample_proj, sample_pil in valid:
        prefix_str, suffix_str = _split_chat_template(processor.tokenizer, q)

        try:
            teacher_ctx = "".join(
                f"Context {i + 1}: {image_token}\nTitle: {cap}\n"
                for i, cap in enumerate(sample_caps)
            )
            teacher_text = prefix_str + teacher_ctx + suffix_str
            t_enc = processor(
                text=teacher_text,
                images=list(sample_pil),
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                gen_out = generator.generate(
                    **t_enc,
                    max_new_tokens=n_distill_tokens,
                    do_sample=False,
                    pad_token_id=eos_id,
                )
            l_teacher = t_enc.input_ids.shape[1]
            gen_ids = gen_out[0, l_teacher:] if gen_out.shape[1] > l_teacher else gen_out[0]
            gen_ids = gen_ids[:n_distill_tokens]
            n_actual = int(gen_ids.numel())
            if n_actual == 0:
                continue

            ext_ids = t_enc.input_ids
            ext_mask = t_enc.attention_mask
            tf_embs = None
            if n_actual > 1:
                tf_ids = gen_ids[: n_actual - 1].unsqueeze(0)
                ext_ids = torch.cat([t_enc.input_ids, tf_ids], dim=1)
                ext_mask = torch.cat(
                    [
                        t_enc.attention_mask,
                        torch.ones(
                            1,
                            n_actual - 1,
                            device=device,
                            dtype=t_enc.attention_mask.dtype,
                        ),
                    ],
                    dim=1,
                )
                tf_embs = embed_fn(tf_ids).to(dtype)

            with torch.no_grad():
                t_out = generator(
                    input_ids=ext_ids,
                    pixel_values=t_enc.pixel_values,
                    attention_mask=ext_mask,
                    use_cache=False,
                )
            t_logits = t_out.logits[0, -n_actual:, :].float()

        except Exception as e:
            logger.warning(f"LLaVA distill teacher failed: {e}; skipping sample.")
            continue

        student_parts = [_e(prefix_str)]
        for i, lat in enumerate(sample_proj):
            label = f"Latent context {i + 1}: " if i == 0 else f"\nLatent context {i + 1}: "
            student_parts.append(_e(label))
            student_parts.append(lat.reshape(-1, lat.shape[-1]).to(dtype).unsqueeze(0))
        student_parts.append(_e(suffix_str))
        s_embeds = torch.cat(student_parts, dim=1)

        if tf_embs is not None:
            s_embeds = torch.cat([s_embeds, tf_embs], dim=1)

        s_out = generator(inputs_embeds=s_embeds, use_cache=False)
        s_logits = s_out.logits[0, -n_actual:, :].float()

        kl = F.kl_div(
            F.log_softmax(s_logits.reshape(-1, s_logits.shape[-1]), dim=-1),
            F.softmax(t_logits.reshape(-1, t_logits.shape[-1]).detach(), dim=-1),
            reduction="sum",
        ) / n_actual
        total_kl = total_kl + kl
        count += 1

    return total_kl / max(count, 1)


def multimodal_train_step(
        compressor: LLaVACompressor,
        retriever: ContrastiveRetriever,
        batch: List[Dict],
        device: str,
        lam_recon: float,
        lam_embed_recon: float,
        lam_cont: float,
        lam_distill: float,
        lam_img_distill: float,
        generator=None,
        generator_proc=None,
        n_distill_tokens: int = 8,
        recon_negatives: bool = False,
        recon_query: bool = False,
        recon_title: bool = True,
        student_distill_hard_negatives_max: int = 0,
        rng: Optional[random.Random] = None,
) -> Dict:
    if rng is None:
        rng = random.Random()
    if not batch:
        z = torch.tensor(0.0, device=device)
        return {
            "total": z,
            "contrast": 0.0,
            "txt_recon_sent": 0.0,
            "txt_recon_query": 0.0,
            "txt_distill": 0.0,
            "img_recon_title": 0.0,
            "img_recon_query": 0.0,
            "img_recon_clip": 0.0,
            "img_distill": 0.0,
        }

    questions = [b["question"] for b in batch]
    q_latents = compressor.embed_query_batch(questions, with_grad=True)
    q_embs = compressor.get_retrieval_embedding(q_latents)

    pos_texts = [s for b in batch for s in b["pos_sentences"]]
    neg_texts = [s for b in batch for s in b["neg_sentences"]]
    pos_pil = [r[0] for b in batch for r in b["pos_regions"]]
    pos_titles = [r[2] for b in batch for r in b["pos_regions"]]
    neg_pil = [r[0] for b in batch for r in b["neg_regions"]]
    neg_titles = [r[2] for b in batch for r in b["neg_regions"]]

    pos_text_counts = [len(b["pos_sentences"]) for b in batch]
    pos_img_counts = [len(b["pos_regions"]) for b in batch]
    neg_text_counts = [len(b["neg_sentences"]) for b in batch]
    neg_img_counts = [len(b["neg_regions"]) for b in batch]

    pos_text_embs = pos_text_latents = None
    if pos_texts:
        pos_text_prompts = [LLaVACompressor.format_compress_prompt(s, False) for s in pos_texts]
        pos_text_latents = compressor.compress_batch(pos_text_prompts, with_grad=True, adapter="compress")
        pos_text_embs = compressor.get_retrieval_embedding(pos_text_latents)

    pos_img_embs = pos_img_latents = None
    if pos_pil:
        pos_img_prompts = [LLaVACompressor.format_compress_prompt(t, True) for t in pos_titles]
        pos_img_latents = compressor.compress_batch(
            pos_img_prompts, images=pos_pil, with_grad=True, adapter="compress"
        )
        pos_img_embs = compressor.get_retrieval_embedding(pos_img_latents)

    neg_text_latents_list = []
    neg_text_embs_list = []
    if neg_texts:
        neg_text_prompts = [LLaVACompressor.format_compress_prompt(s, False) for s in neg_texts]
        neg_text_latents = compressor.compress_batch(neg_text_prompts, with_grad=True, adapter="compress")
        neg_text_embs = compressor.get_retrieval_embedding(neg_text_latents)
        offset = 0
        for c in neg_text_counts:
            neg_text_latents_list.append(neg_text_latents[offset:offset + c])
            neg_text_embs_list.append(neg_text_embs[offset:offset + c])
            offset += c
    else:
        if pos_text_latents is not None:
            neg_text_latents_list = [
                pos_text_latents.new_zeros((0, pos_text_latents.shape[-1]))
                for _ in batch
            ]
            neg_text_embs_list = [
                pos_text_embs.new_zeros((0, pos_text_embs.shape[-1]))
                for _ in batch
            ]
        else:
            neg_text_latents_list = []
            neg_text_embs_list = []

    neg_img_latents_list = []
    neg_img_embs_list = []
    if neg_pil:
        neg_img_prompts = [LLaVACompressor.format_compress_prompt(t, True) for t in neg_titles]
        neg_img_latents = compressor.compress_batch(
            neg_img_prompts, images=neg_pil, with_grad=True, adapter="compress"
        )
        neg_img_embs = compressor.get_retrieval_embedding(neg_img_latents)
        offset = 0
        for c in neg_img_counts:
            neg_img_latents_list.append(neg_img_latents[offset:offset + c])
            neg_img_embs_list.append(neg_img_embs[offset:offset + c])
            offset += c
    else:
        if pos_img_latents is not None:
            neg_img_latents_list = [
                pos_img_latents.new_zeros((0, pos_img_latents.shape[-1]))
                for _ in batch
            ]
            neg_img_embs_list = [
                pos_img_embs.new_zeros((0, pos_img_embs.shape[-1]))
                for _ in batch
            ]
        else:
            neg_img_latents_list = []
            neg_img_embs_list = []

    sampled_mixed_neg_latents = []
    sampled_mixed_neg_embs = []
    for i in range(len(batch)):
        latent_chunks = []
        emb_chunks = []
        if i < len(neg_text_latents_list) and neg_text_latents_list[i].shape[0] > 0:
            latent_chunks.append(neg_text_latents_list[i])
            emb_chunks.append(neg_text_embs_list[i])
        if i < len(neg_img_latents_list) and neg_img_latents_list[i].shape[0] > 0:
            latent_chunks.append(neg_img_latents_list[i])
            emb_chunks.append(neg_img_embs_list[i])

        if latent_chunks:
            all_neg_latents = torch.cat(latent_chunks, dim=0)
            all_neg_embs = torch.cat(emb_chunks, dim=0)
            if student_distill_hard_negatives_max > 0:
                max_extra = min(student_distill_hard_negatives_max, all_neg_latents.shape[0])
                n_extra = rng.randint(0, max_extra)
                if n_extra > 0:
                    selected = rng.sample(range(all_neg_latents.shape[0]), k=n_extra)
                    sampled_mixed_neg_latents.append(all_neg_latents[selected])
                    sampled_mixed_neg_embs.append(all_neg_embs[selected])
                else:
                    sampled_mixed_neg_latents.append(all_neg_latents.new_zeros((0, all_neg_latents.shape[-1])))
                    sampled_mixed_neg_embs.append(all_neg_embs.new_zeros((0, all_neg_embs.shape[-1])))
            else:
                sampled_mixed_neg_latents.append(all_neg_latents.new_zeros((0, all_neg_latents.shape[-1])))
                sampled_mixed_neg_embs.append(all_neg_embs.new_zeros((0, all_neg_embs.shape[-1])))
        else:
            ref_latent = pos_text_latents if pos_text_latents is not None else pos_img_latents
            ref_emb = pos_text_embs if pos_text_embs is not None else pos_img_embs
            if ref_latent is not None:
                sampled_mixed_neg_latents.append(ref_latent.new_zeros((0, ref_latent.shape[-1])))
            else:
                sampled_mixed_neg_latents.append(torch.empty(0, 0, device=device))
            if ref_emb is not None:
                sampled_mixed_neg_embs.append(ref_emb.new_zeros((0, ref_emb.shape[-1])))
            else:
                sampled_mixed_neg_embs.append(torch.empty(0, 0, device=device))

    positive_embs_list = []
    txt_offset = 0
    img_offset = 0
    for txt_c, img_c in zip(pos_text_counts, pos_img_counts):
        chunks = []
        if txt_c > 0 and pos_text_embs is not None:
            chunks.append(pos_text_embs[txt_offset: txt_offset + txt_c])
        if img_c > 0 and pos_img_embs is not None:
            chunks.append(pos_img_embs[img_offset: img_offset + img_c])
        positive_embs_list.append(torch.cat(chunks, dim=0))
        txt_offset += txt_c
        img_offset += img_c

    hard_neg_embs = None
    neg_emb_chunks = [embs for embs in sampled_mixed_neg_embs if embs.numel() > 0]
    if neg_emb_chunks:
        hard_neg_embs = torch.cat(neg_emb_chunks, dim=0)

    l_contrast = retriever.contrastive_loss(q_embs, positive_embs_list, hard_neg_embs)

    recon_texts = pos_texts + neg_texts if recon_negatives else pos_texts
    l_txt_recon_sent = (
        compressor.reconstruction_loss_batch(recon_texts, adapter="compress", decode_adapter="decode")
        if recon_texts else torch.tensor(0.0, device=device)
    )
    l_txt_recon_query = (
        compressor.reconstruction_loss_batch(questions, adapter="query", decode_adapter="query_decode")
        if recon_query else torch.tensor(0.0, device=device)
    )

    if recon_title:
        recon_titles = pos_titles + neg_titles if (recon_negatives and neg_pil) else pos_titles
        recon_pil = pos_pil + neg_pil if (recon_negatives and neg_pil) else pos_pil
        recon_prompts = [LLaVACompressor.format_compress_prompt(t, True) for t in recon_titles]
        l_img_recon_title = (
            compressor.reconstruction_loss_batch(
                recon_prompts, images=recon_pil, adapter="compress", decode_adapter="image_decode",
                target_texts=recon_titles,
            ) if recon_titles else torch.tensor(0.0, device=device)
        )
    else:
        l_img_recon_title = torch.tensor(0.0, device=device)

    l_img_recon_query = (
        compressor.reconstruction_loss_batch(questions, adapter="query", decode_adapter="query_decode")
        if recon_query else torch.tensor(0.0, device=device)
    )
    clip_titles = pos_titles + neg_titles if (recon_negatives and neg_pil) else pos_titles
    clip_pil = pos_pil + neg_pil if (recon_negatives and neg_pil) else pos_pil
    clip_prompts = [LLaVACompressor.format_compress_prompt(t, True) for t in clip_titles]
    l_img_recon_clip = (
        compressor.embed_reconstruction_loss_batch(clip_prompts, images=clip_pil, adapter="compress")
        if lam_embed_recon > 0 and clip_pil else torch.tensor(0.0, device=device)
    )

    l_txt_distill = torch.tensor(0.0, device=device)
    if lam_distill > 0 and generator is not None and pos_text_latents is not None and pos_texts:
        student_proj_text = []
        offset = 0
        for i, n_pos in enumerate(pos_text_counts):
            student_latents = pos_text_latents[offset:offset + n_pos]
            offset += n_pos
            if sampled_mixed_neg_latents[i].numel() > 0:
                student_latents = torch.cat([student_latents, sampled_mixed_neg_latents[i]], dim=0)
            student_proj_text.append(compressor.project_for_generator(student_latents))
        l_txt_distill = _text_distill_loss(
            generator, generator_proc,
            questions, pos_texts, student_proj_text,
            pos_text_counts, device, n_distill_tokens,
        )

    l_img_distill = torch.tensor(0.0, device=device)
    if lam_img_distill > 0 and generator is not None and pos_img_latents is not None and pos_pil:
        student_proj_img = []
        offset = 0
        for i, n_pos in enumerate(pos_img_counts):
            student_latents = pos_img_latents[offset:offset + n_pos]
            offset += n_pos
            if sampled_mixed_neg_latents[i].numel() > 0:
                student_latents = torch.cat([student_latents, sampled_mixed_neg_latents[i]], dim=0)
            student_proj_img.append(compressor.project_for_generator(student_latents))
        l_img_distill = _llava_distill_loss(
            generator, generator_proc,
            questions, pos_titles, pos_pil, student_proj_img,
            pos_img_counts, device, n_distill_tokens,
        )

    total = (
        lam_cont * l_contrast
        + lam_recon * (l_txt_recon_sent + l_txt_recon_query + l_img_recon_title + l_img_recon_query)
        + lam_embed_recon * l_img_recon_clip
        + lam_distill * l_txt_distill
        + lam_img_distill * l_img_distill
    )
    return {
        "total": total,
        "contrast": l_contrast.item(),
        "txt_recon_sent": l_txt_recon_sent.item(),
        "txt_recon_query": l_txt_recon_query.item(),
        "txt_distill": l_txt_distill.item() if isinstance(l_txt_distill, torch.Tensor) else 0.0,
        "img_recon_title": l_img_recon_title.item(),
        "img_recon_query": l_img_recon_query.item(),
        "img_recon_clip": l_img_recon_clip.item(),
        "img_distill": l_img_distill.item() if isinstance(l_img_distill, torch.Tensor) else 0.0,
    }


# ---------------------------------------------------------------------------
# Validation — EM / F1 / ROUGE-L + Recall@k
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation_multi_k_legacy_split(
        compressor: LLaVACompressor,
        generator,
        generator_proc,
        image_samples: List[Dict],
        text_samples: List[Dict],
        scale: float,
        device: str,
        top_k_values: List[int],
        max_new_tokens: int = 64,
        max_samples: int = 200,
        gen_batch_size: int = 8,
        compress_batch_size: int = 32,
) -> Dict[int, Dict[str, float]]:
    """
    End-to-end validation: retrieve → project → generate → EM/F1/ROUGE-L.

    Image samples: CLIP-based retrieval (positives + negatives per sample).
    Text  samples: sentence-compression retrieval (txt_pos_facts pool).

    Also computes Recall@k over the candidate pool for each requested k.

    Returns
    -------
    dict mapping k -> metrics dict
    """
    from src.evaluation import evaluate

    compressor.eval()
    k_list = sorted(set(top_k_values))
    if not k_list:
        return {}
    max_k = max(k_list)

    # Pre-compute static embedding state once for all _generate calls
    embed_fn = generator.get_input_embeddings()
    dtype = next(generator.parameters()).dtype

    def _e(text: str, max_len: int = 512):
        return embed_fn(_tok(generator_proc.tokenizer, text, device, max_len)).to(dtype)

    # Cache static label embeddings (question-independent)
    _label_embs = [_e(f"Latent context {i + 1}: ") for i in range(max_k)]

    def _build_seq_embeds(proj_tokens: torch.Tensor, question: str) -> torch.Tensor:
        prefix_str, suffix_str = _split_chat_template(generator_proc.tokenizer, question)
        parts = [_e(prefix_str)]
        for i, lat in enumerate(proj_tokens):
            if i == 0:
                parts.append(_label_embs[i])
            else:
                parts.append(_e(f"\nLatent context {i + 1}: "))
            parts.append(lat.reshape(-1, lat.shape[-1]).to(dtype).unsqueeze(0).to(device))
        parts.append(_e(suffix_str))
        return torch.cat(parts, dim=1).squeeze(0)

    def _batch_generate(
            projected_list: List[torch.Tensor],
            questions: List[str],
    ) -> tuple:
        seq_embeds = [_build_seq_embeds(proj, q) for proj, q in zip(projected_list, questions)]

        batch_size = len(seq_embeds)
        max_len = max(e.shape[0] for e in seq_embeds)
        hidden = seq_embeds[0].shape[-1]
        padded = torch.zeros(batch_size, max_len, hidden, dtype=seq_embeds[0].dtype, device=device)
        attn_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        position_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        input_lens = []

        for i, emb in enumerate(seq_embeds):
            seq_len = emb.shape[0]
            padded[i, max_len - seq_len:] = emb
            attn_mask[i, max_len - seq_len:] = 1
            position_ids[i, max_len - seq_len:] = torch.arange(seq_len, device=device)
            input_lens.append(seq_len)

        out = generator.generate(
            inputs_embeds=padded,
            attention_mask=attn_mask,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=generator_proc.tokenizer.eos_token_id,
        )

        answers = []
        for i in range(batch_size):
            gen_ids = out[i, max_len:] if out.shape[1] > max_len else out[i]
            answers.append(generator_proc.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
        return answers, input_lens

    # ── Image samples ──────────────────────────────────────────────────────
    img_pool = list(image_samples)
    random.Random(42).shuffle(img_pool)
    if max_samples > 0:
        img_pool = img_pool[:max_samples]

    img_valid = []
    for s in tqdm(img_pool, desc="Val-image", leave=False):
        img_dir = s.get("image_dir", "")
        pos_ids = s.get("pos_image_ids", [])
        neg_ids = s.get("neg_image_ids", [])
        question = s.get("question", "")
        answers = s.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]
        if not pos_ids or not question or not answers:
            continue

        # Load + encode all candidates (pos + neg)
        all_ids = pos_ids + neg_ids
        cand_pils, cand_titles = [], []
        for cid, cap in zip(pos_ids + neg_ids,
                            s.get("pos_captions", []) + s.get("neg_captions", [])):
            path = os.path.join(img_dir, f"{cid}.jpg")
            if not os.path.exists(path):
                continue
            try:
                pil = Image.open(path).convert("RGB")
            except Exception:
                continue
            cand_pils.append(resize_image(pil, scale))
            cand_titles.append(cap)

        if not cand_pils:
            continue

        # Compress all candidates
        cand_prompts = [LLaVACompressor.format_compress_prompt(t, True) for t in cand_titles]
        cand_latents = compressor.compress_batch(cand_prompts, images=cand_pils,
                                                 with_grad=False, adapter="compress")
        cand_embs = compressor.get_retrieval_embedding(cand_latents)  # (N, D)

        img_valid.append(
            {
                "source": s.get("source", "webqa"),
                "modality": s.get("modality", "image"),
                "webqa_qcate": s.get("webqa_qcate"),
                "webqa_keywords": s.get("webqa_keywords"),
                "question": question,
                "answers": answers,
                "pos_ids": pos_ids,
                "img_dir": img_dir,
                "cand_pils": cand_pils,
                "cand_latents": cand_latents,
                "cand_embs": cand_embs,
            }
        )

    if img_valid:
        img_questions = [s["question"] for s in img_valid]
        img_q_lat = compressor.embed_query_batch(img_questions, with_grad=False)
        img_q_embs = compressor.get_retrieval_embedding(img_q_lat)
    else:
        img_questions = []
        img_q_embs = None

    # ── Text samples ───────────────────────────────────────────────────────
    txt_pool = list(text_samples)
    random.Random(42).shuffle(txt_pool)
    if max_samples > 0:
        txt_pool = txt_pool[:max_samples]

    txt_valid = []
    txt_offsets = []
    all_txt_prompts = []
    for s in tqdm(txt_pool, desc="Val-text", leave=False):
        pos_facts = s.get("txt_pos_facts", [])
        neg_facts = s.get("txt_neg_facts", [])[:max_k * 2]  # cap using the largest requested k
        question = s.get("question", "")
        answers = s.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]
        if not pos_facts or not question or not answers:
            continue

        all_facts = pos_facts + neg_facts
        start = len(all_txt_prompts)
        all_txt_prompts.extend(LLaVACompressor.format_compress_prompt(f, False) for f in all_facts)
        txt_offsets.append((start, len(all_txt_prompts)))

        txt_valid.append(
            {
                "source": s.get("source", "webqa"),
                "modality": s.get("modality", "text"),
                "webqa_qcate": s.get("webqa_qcate"),
                "webqa_keywords": s.get("webqa_keywords"),
                "question": question,
                "answers": answers,
                "pos_facts": pos_facts,
                "all_facts": all_facts,
            }
        )

    if all_txt_prompts:
        txt_latent_chunks = []
        for start in tqdm(range(0, len(all_txt_prompts), compress_batch_size),
                          total=(len(all_txt_prompts) + compress_batch_size - 1) // compress_batch_size,
                          desc="Compress-text", leave=False):
            txt_latent_chunks.append(
                compressor.compress_batch(
                    all_txt_prompts[start: start + compress_batch_size],
                    with_grad=False,
                    adapter="compress",
                )
            )
        all_txt_latents = torch.cat(txt_latent_chunks, dim=0)

        txt_emb_chunks = []
        for start in tqdm(range(0, all_txt_latents.shape[0], compress_batch_size),
                          total=(all_txt_latents.shape[0] + compress_batch_size - 1) // compress_batch_size,
                          desc="Text-ret", leave=False):
            txt_emb_chunks.append(
                compressor.get_retrieval_embedding(
                    all_txt_latents[start: start + compress_batch_size]
                )
            )
        all_txt_embs = torch.cat(txt_emb_chunks, dim=0)
    else:
        all_txt_latents = None
        all_txt_embs = None

    if txt_valid:
        txt_questions = [s["question"] for s in txt_valid]
        txt_q_lat = compressor.embed_query_batch(txt_questions, with_grad=False)
        txt_q_embs = compressor.get_retrieval_embedding(txt_q_lat)
    else:
        txt_questions = []
        txt_q_embs = None

    results = {}
    for top_k in k_list:
        predictions, references, token_counts = [], [], []
        sample_meta = []
        recall_scores = []
        img_preds, img_refs, txt_preds, txt_refs = [], [], [], []

        img_projected, img_gen_questions, img_gen_refs = [], [], []
        img_meta_pending = []
        for idx, s in enumerate(img_valid):
            q_emb = img_q_embs[idx: idx + 1]
            sims = (s["cand_embs"] @ q_emb.T).squeeze(-1)
            k = min(top_k, len(s["cand_pils"]))
            top_i = torch.topk(sims, k=k).indices.tolist()

            n_pos_loaded = sum(1 for cid in s["pos_ids"]
                               if os.path.exists(os.path.join(s["img_dir"], f"{cid}.jpg")))
            pos_set = set(range(min(n_pos_loaded, len(s["cand_pils"]))))
            hits = len(set(top_i) & pos_set)
            if pos_set:
                recall_scores.append(hits / len(pos_set))

            top_latents = s["cand_latents"][top_i]
            img_projected.append(compressor.project_for_generator(top_latents))
            img_gen_questions.append(s["question"])
            img_gen_refs.append(s["answers"])
            img_meta_pending.append({
                "source": s.get("source", "webqa"),
                "modality": s.get("modality", "image"),
                "webqa_qcate": s.get("webqa_qcate"),
                "webqa_keywords": s.get("webqa_keywords"),
            })

        n_img_batches = (len(img_projected) + gen_batch_size - 1) // gen_batch_size if img_projected else 0
        for b_start in tqdm(range(0, len(img_projected), gen_batch_size),
                            total=n_img_batches, desc=f"Gen-image k={top_k}", leave=False):
            b_proj = img_projected[b_start: b_start + gen_batch_size]
            b_qs = img_gen_questions[b_start: b_start + gen_batch_size]
            b_refs = img_gen_refs[b_start: b_start + gen_batch_size]
            try:
                b_answers, b_lens = _batch_generate(b_proj, b_qs)
            except Exception as e:
                logger.warning(f"Image batch generation failed (k={top_k}): {e}")
                b_answers = [""] * len(b_proj)
                b_lens = [0] * len(b_proj)

            predictions.extend(b_answers)
            references.extend(b_refs)
            token_counts.extend(b_lens)
            sample_meta.extend(img_meta_pending[b_start: b_start + gen_batch_size])
            img_preds.extend(b_answers)
            img_refs.extend(b_refs)

        txt_projected, txt_gen_questions, txt_gen_refs = [], [], []
        txt_meta_pending = []
        for idx, (s, (start, end)) in enumerate(zip(txt_valid, txt_offsets)):
            q_emb = txt_q_embs[idx: idx + 1]
            fact_embs = all_txt_embs[start:end]
            fact_latents = all_txt_latents[start:end]
            sims = (fact_embs @ q_emb.T).squeeze(-1)
            k = min(top_k, len(s["all_facts"]))
            top_i = torch.topk(sims, k=k).indices.tolist()

            pos_set = set(range(len(s["pos_facts"])))
            hits = len(set(top_i) & pos_set)
            if pos_set:
                recall_scores.append(hits / len(pos_set))

            top_latents = fact_latents[top_i]
            txt_projected.append(compressor.project_for_generator(top_latents))
            txt_gen_questions.append(s["question"])
            txt_gen_refs.append(s["answers"])
            txt_meta_pending.append({
                "source": s.get("source", "webqa"),
                "modality": s.get("modality", "text"),
                "webqa_qcate": s.get("webqa_qcate"),
                "webqa_keywords": s.get("webqa_keywords"),
            })

        n_txt_batches = (len(txt_projected) + gen_batch_size - 1) // gen_batch_size if txt_projected else 0
        for b_start in tqdm(range(0, len(txt_projected), gen_batch_size),
                            total=n_txt_batches, desc=f"Gen-text k={top_k}", leave=False):
            b_proj = txt_projected[b_start: b_start + gen_batch_size]
            b_qs = txt_gen_questions[b_start: b_start + gen_batch_size]
            b_refs = txt_gen_refs[b_start: b_start + gen_batch_size]
            try:
                b_answers, b_lens = _batch_generate(b_proj, b_qs)
            except Exception as e:
                logger.warning(f"Text batch generation failed (k={top_k}): {e}")
                b_answers = [""] * len(b_proj)
                b_lens = [0] * len(b_proj)

            predictions.extend(b_answers)
            references.extend(b_refs)
            token_counts.extend(b_lens)
            sample_meta.extend(txt_meta_pending[b_start: b_start + gen_batch_size])
            txt_preds.extend(b_answers)
            txt_refs.extend(b_refs)

        if not predictions:
            logger.warning(f"Validation: no predictions for k={top_k}.")
            results[top_k] = {
                "em": 0.0, "f1": 0.0, "rouge_l": 0.0, "recall_at_k": 0.0,
                "avg_tokens": 0.0, "img_em": 0.0, "img_f1": 0.0,
                "txt_em": 0.0, "txt_f1": 0.0,
            }
            continue

        metrics = evaluate(predictions, references, token_counts, sample_metadata=sample_meta)
        img_meta = [m for m in sample_meta if m["modality"] == "image"]
        txt_meta = [m for m in sample_meta if m["modality"] == "text"]
        img_metrics = evaluate(img_preds, img_refs, [0] * len(img_preds), sample_metadata=img_meta) if img_preds else {"em": 0.0, "f1": 0.0}
        txt_metrics = evaluate(txt_preds, txt_refs, [0] * len(txt_preds), sample_metadata=txt_meta) if txt_preds else {"em": 0.0, "f1": 0.0}
        metrics["recall_at_k"] = float(sum(recall_scores) / len(recall_scores)) if recall_scores else 0.0
        metrics["img_em"] = img_metrics["em"]
        metrics["img_f1"] = img_metrics["f1"]
        metrics["txt_em"] = txt_metrics["em"]
        metrics["txt_f1"] = txt_metrics["f1"]
        if "acc" in img_metrics:
            metrics["img_acc"] = img_metrics["acc"]
            metrics["img_webqa_acc"] = img_metrics["acc"]
        if "acc" in txt_metrics:
            metrics["txt_acc"] = txt_metrics["acc"]
            metrics["txt_webqa_acc"] = txt_metrics["acc"]

        acc_part = f" Acc={metrics['acc']:.4f} |" if "acc" in metrics else ""
        img_acc_part = f" img Acc={metrics['img_acc']:.4f} |" if "img_acc" in metrics else ""
        txt_acc_part = f" txt Acc={metrics['txt_acc']:.4f} |" if "txt_acc" in metrics else ""
        logger.info(
            f"Val k={top_k} | n={len(predictions)} "
            f"(img={len(img_preds)}, txt={len(txt_preds)}) | "
            f"EM={metrics['em']:.4f} F1={metrics['f1']:.4f} "
            f"ROUGE-L={metrics['rouge_l']:.4f} |{acc_part} "
            f"img EM={metrics['img_em']:.4f} F1={metrics['img_f1']:.4f} |{img_acc_part} "
            f"txt EM={metrics['txt_em']:.4f} F1={metrics['txt_f1']:.4f} |{txt_acc_part} "
            f"Recall@{top_k}={metrics['recall_at_k']:.4f} | "
            f"avg_tokens={metrics.get('avg_tokens', 0):.1f}"
        )
        results[top_k] = metrics

    compressor.train()
    return results


def run_validation(
        compressor: LLaVACompressor,
        generator,
        generator_proc,
        image_samples: List[Dict],
        text_samples: List[Dict],
        scale: float,
        device: str,
        top_k: int = 5,
        max_new_tokens: int = 64,
        max_samples: int = 200,
        gen_batch_size: int = 8,
        compress_batch_size: int = 32,
    ) -> Dict[str, float]:
    """Single-k validation wrapper for backward compatibility."""
    return run_validation_multi_k_unified(
        compressor,
        generator,
        generator_proc,
        image_samples,
        text_samples,
        scale,
        device,
        top_k_values=[top_k],
        max_new_tokens=max_new_tokens,
        max_samples=max_samples,
        gen_batch_size=gen_batch_size,
        compress_batch_size=compress_batch_size,
    )[top_k]


@torch.no_grad()
def run_validation_multi_k_unified(
        compressor: LLaVACompressor,
        generator,
        generator_proc,
        image_samples: List[Dict],
        text_samples: List[Dict],
        scale: float,
        device: str,
        top_k_values: List[int],
        max_new_tokens: int = 64,
        max_samples: int = 200,
        gen_batch_size: int = 8,
        compress_batch_size: int = 32,
) -> Dict[int, Dict[str, float]]:
    from src.evaluation import evaluate

    compressor.eval()
    k_list = sorted(set(top_k_values))
    if not k_list:
        return {}
    max_k = max(k_list)

    embed_fn = generator.get_input_embeddings()
    dtype = next(generator.parameters()).dtype

    def _e(text: str, max_len: int = 512):
        return embed_fn(_tok(generator_proc.tokenizer, text, device, max_len)).to(dtype)

    _label_embs = [_e(f"Latent context {i + 1}: ") for i in range(max_k)]

    def _build_seq_embeds(proj_tokens: torch.Tensor, question: str) -> torch.Tensor:
        prefix_str, suffix_str = _split_chat_template(generator_proc.tokenizer, question)
        parts = [_e(prefix_str)]
        for i, lat in enumerate(proj_tokens):
            if i == 0:
                parts.append(_label_embs[i])
            else:
                parts.append(_e(f"\nLatent context {i + 1}: "))
            parts.append(lat.reshape(-1, lat.shape[-1]).to(dtype).unsqueeze(0).to(device))
        parts.append(_e(suffix_str))
        return torch.cat(parts, dim=1).squeeze(0)

    def _batch_generate(projected_list: List[torch.Tensor], questions: List[str]) -> tuple:
        seq_embeds = [_build_seq_embeds(proj, q) for proj, q in zip(projected_list, questions)]
        batch_size = len(seq_embeds)
        max_len = max(e.shape[0] for e in seq_embeds)
        hidden = seq_embeds[0].shape[-1]
        padded = torch.zeros(batch_size, max_len, hidden, dtype=seq_embeds[0].dtype, device=device)
        attn_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        position_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        input_lens = []

        for i, emb in enumerate(seq_embeds):
            seq_len = emb.shape[0]
            padded[i, max_len - seq_len:] = emb
            attn_mask[i, max_len - seq_len:] = 1
            position_ids[i, max_len - seq_len:] = torch.arange(seq_len, device=device)
            input_lens.append(seq_len)

        out = generator.generate(
            inputs_embeds=padded,
            attention_mask=attn_mask,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=generator_proc.tokenizer.eos_token_id,
        )

        answers = []
        for i in range(batch_size):
            gen_ids = out[i, max_len:] if out.shape[1] > max_len else out[i]
            answers.append(generator_proc.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
        return answers, input_lens

    val_pool = list(image_samples) + list(text_samples)
    random.Random(42).shuffle(val_pool)
    if max_samples > 0:
        val_pool = val_pool[:max_samples]

    valid = []
    all_txt_prompts = []
    all_img_prompts = []
    all_img_pils = []
    txt_offsets = []
    img_offsets = []
    txt_offset = 0
    img_offset = 0
    for s in tqdm(val_pool, desc="Prep-val", leave=False):
        question = s.get("question", "")
        answers = s.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]
        if not question or not answers:
            continue

        txt_pos = list(s.get("txt_pos_facts", []))
        txt_neg = list(s.get("txt_neg_facts", []))
        txt_facts = txt_pos + txt_neg

        img_dir = s.get("image_dir", "")
        img_pos_ids = list(s.get("pos_image_ids", []))
        img_neg_ids = list(s.get("neg_image_ids", []))
        img_caps = list(s.get("pos_captions", [])) + list(s.get("neg_captions", []))
        img_pils, loaded_pos = [], 0
        for idx, (cid, cap) in enumerate(zip(img_pos_ids + img_neg_ids, img_caps)):
            path = os.path.join(img_dir, f"{cid}.jpg")
            if not os.path.exists(path):
                continue
            try:
                pil = Image.open(path).convert("RGB")
            except Exception:
                continue
            img_pils.append((resize_image(pil, scale), cap))
            if idx < len(img_pos_ids):
                loaded_pos += 1

        if not txt_facts and not img_pils:
            continue

        txt_start = txt_offset
        txt_end = txt_offset + len(txt_facts)
        all_txt_prompts.extend(LLaVACompressor.format_compress_prompt(f, False) for f in txt_facts)
        txt_offsets.append((txt_start, txt_end))
        txt_offset = txt_end

        img_start = img_offset
        img_end = img_offset + len(img_pils)
        all_img_prompts.extend(
            LLaVACompressor.format_compress_prompt(title, True)
            for _, title in img_pils
        )
        all_img_pils.extend(pil for pil, _ in img_pils)
        img_offsets.append((img_start, img_end))
        img_offset = img_end

        valid.append({
            "modality": s.get("modality", "image" if img_pos_ids else "text"),
            "question": question,
            "answers": answers,
            "source": s.get("source", "webqa"),
            "webqa_qcate": s.get("webqa_qcate"),
            "webqa_keywords": s.get("webqa_keywords"),
            "txt_pos_count": len(txt_pos),
            "img_pos_count": loaded_pos,
        })

    if not valid:
        logger.warning("Validation: no predictions.")
        return {
            k: {
                "em": 0.0, "f1": 0.0, "rouge_l": 0.0, "recall_at_k": 0.0,
                "precision_at_k": 0.0, "avg_tokens": 0.0, "img_em": 0.0, "img_f1": 0.0,
                "txt_em": 0.0, "txt_f1": 0.0, "img_recall_at_k": 0.0,
                "txt_recall_at_k": 0.0, "img_precision_at_k": 0.0,
                "txt_precision_at_k": 0.0, "img_avg_tokens": 0.0, "txt_avg_tokens": 0.0,
            } for k in k_list
        }

    if all_txt_prompts:
        txt_latent_chunks = []
        for start in tqdm(range(0, len(all_txt_prompts), compress_batch_size),
                          total=(len(all_txt_prompts) + compress_batch_size - 1) // compress_batch_size,
                          desc="Compress-text", leave=False):
            txt_latent_chunks.append(
                compressor.compress_batch(
                    all_txt_prompts[start: start + compress_batch_size],
                    with_grad=False,
                    adapter="compress",
                )
            )
        all_txt_latents = torch.cat(txt_latent_chunks, dim=0)

        txt_emb_chunks = []
        for start in tqdm(range(0, all_txt_latents.shape[0], compress_batch_size),
                          total=(all_txt_latents.shape[0] + compress_batch_size - 1) // compress_batch_size,
                          desc="Text-ret", leave=False):
            txt_emb_chunks.append(
                compressor.get_retrieval_embedding(all_txt_latents[start: start + compress_batch_size])
            )
        all_txt_embs = torch.cat(txt_emb_chunks, dim=0)
    else:
        all_txt_latents = None
        all_txt_embs = None

    if all_img_prompts:
        img_latent_chunks = []
        for start in tqdm(range(0, len(all_img_prompts), compress_batch_size),
                          total=(len(all_img_prompts) + compress_batch_size - 1) // compress_batch_size,
                          desc="Compress-image", leave=False):
            img_latent_chunks.append(
                compressor.compress_batch(
                    all_img_prompts[start: start + compress_batch_size],
                    images=all_img_pils[start: start + compress_batch_size],
                    with_grad=False,
                    adapter="compress",
                )
            )
        all_img_latents = torch.cat(img_latent_chunks, dim=0)

        img_emb_chunks = []
        for start in tqdm(range(0, all_img_latents.shape[0], compress_batch_size),
                          total=(all_img_latents.shape[0] + compress_batch_size - 1) // compress_batch_size,
                          desc="Image-ret", leave=False):
            img_emb_chunks.append(
                compressor.get_retrieval_embedding(all_img_latents[start: start + compress_batch_size])
            )
        all_img_embs = torch.cat(img_emb_chunks, dim=0)
    else:
        all_img_latents = None
        all_img_embs = None

    q_embs = compressor.get_retrieval_embedding(
        compressor.embed_query_batch([s["question"] for s in valid], with_grad=False)
    )

    results = {}
    for top_k in k_list:
        predictions, references, token_counts = [], [], []
        sample_meta = []
        recall_scores = []
        precision_scores = []
        img_recalls, txt_recalls = [], []
        img_precisions, txt_precisions = [], []
        img_tokens, txt_tokens = [], []
        img_preds, img_refs, txt_preds, txt_refs = [], [], [], []

        projected_list, questions_list, refs_list, mods_list = [], [], [], []
        meta_list = []
        for s, q_emb, (txt_start, txt_end), (img_start, img_end) in zip(valid, q_embs, txt_offsets, img_offsets):
            cand_latents = []
            cand_embs = []
            pos_indices = []
            cursor = 0

            if img_end > img_start and all_img_latents is not None:
                img_latents = all_img_latents[img_start:img_end]
                img_embs = all_img_embs[img_start:img_end]
                cand_latents.append(img_latents)
                cand_embs.append(img_embs)
                if s["modality"] == "image":
                    pos_indices.extend(range(cursor, cursor + s["img_pos_count"]))
                cursor += img_latents.shape[0]

            if txt_end > txt_start and all_txt_latents is not None:
                fact_latents = all_txt_latents[txt_start:txt_end]
                fact_embs = all_txt_embs[txt_start:txt_end]
                cand_latents.append(fact_latents)
                cand_embs.append(fact_embs)
                if s["modality"] == "text":
                    pos_indices.extend(range(cursor, cursor + s["txt_pos_count"]))
                cursor += fact_latents.shape[0]

            if not cand_latents:
                continue

            cand_latents = torch.cat(cand_latents, dim=0)
            cand_embs = torch.cat(cand_embs, dim=0)
            sims = (cand_embs @ q_emb.unsqueeze(-1)).squeeze(-1)
            k = min(top_k, cand_latents.shape[0])
            top_i = torch.topk(sims, k=k).indices.tolist()

            pos_set = set(pos_indices)
            hits = len(set(top_i) & pos_set)
            if pos_set:
                recall = hits / len(pos_set)
                recall_scores.append(recall)
                if s["modality"] == "image":
                    img_recalls.append(recall)
                else:
                    txt_recalls.append(recall)
            precision = hits / len(top_i) if top_i else 0.0
            precision_scores.append(precision)
            if s["modality"] == "image":
                img_precisions.append(precision)
            else:
                txt_precisions.append(precision)

            projected_list.append(compressor.project_for_generator(cand_latents[top_i]))
            questions_list.append(s["question"])
            refs_list.append(s["answers"])
            mods_list.append(s["modality"])
            meta_list.append({
                "source": s.get("source", "webqa"),
                "modality": s["modality"],
                "webqa_qcate": s.get("webqa_qcate"),
                "webqa_keywords": s.get("webqa_keywords"),
            })

        n_batches = (len(projected_list) + gen_batch_size - 1) // gen_batch_size if projected_list else 0
        for b_start in tqdm(range(0, len(projected_list), gen_batch_size),
                            total=n_batches, desc=f"Gen-val k={top_k}", leave=False):
            b_proj = projected_list[b_start: b_start + gen_batch_size]
            b_qs = questions_list[b_start: b_start + gen_batch_size]
            b_refs = refs_list[b_start: b_start + gen_batch_size]
            b_mods = mods_list[b_start: b_start + gen_batch_size]
            try:
                b_answers, b_lens = _batch_generate(b_proj, b_qs)
            except Exception as e:
                logger.warning(f"Validation batch generation failed (k={top_k}): {e}")
                b_answers = [""] * len(b_proj)
                b_lens = [0] * len(b_proj)

            predictions.extend(b_answers)
            references.extend(b_refs)
            token_counts.extend(b_lens)
            sample_meta.extend(meta_list[b_start: b_start + gen_batch_size])
            for ans, ref, mod, n_tok in zip(b_answers, b_refs, b_mods, b_lens):
                if mod == "image":
                    img_preds.append(ans)
                    img_refs.append(ref)
                    img_tokens.append(n_tok)
                else:
                    txt_preds.append(ans)
                    txt_refs.append(ref)
                    txt_tokens.append(n_tok)

        if not predictions:
            logger.warning(f"Validation: no predictions for k={top_k}.")
            results[top_k] = {
                "em": 0.0, "f1": 0.0, "rouge_l": 0.0, "recall_at_k": 0.0,
                "precision_at_k": 0.0, "avg_tokens": 0.0, "img_em": 0.0, "img_f1": 0.0,
                "txt_em": 0.0, "txt_f1": 0.0, "img_recall_at_k": 0.0,
                "txt_recall_at_k": 0.0, "img_precision_at_k": 0.0,
                "txt_precision_at_k": 0.0, "img_avg_tokens": 0.0, "txt_avg_tokens": 0.0,
            }
            continue

        metrics = evaluate(predictions, references, token_counts, sample_metadata=sample_meta)
        img_meta = [m for m in sample_meta if m["modality"] == "image"]
        txt_meta = [m for m in sample_meta if m["modality"] == "text"]
        img_metrics = evaluate(img_preds, img_refs, [0] * len(img_preds), sample_metadata=img_meta) if img_preds else {"em": 0.0, "f1": 0.0}
        txt_metrics = evaluate(txt_preds, txt_refs, [0] * len(txt_preds), sample_metadata=txt_meta) if txt_preds else {"em": 0.0, "f1": 0.0}
        metrics["recall_at_k"] = float(sum(recall_scores) / len(recall_scores)) if recall_scores else 0.0
        metrics["precision_at_k"] = float(sum(precision_scores) / len(precision_scores)) if precision_scores else 0.0
        metrics["img_em"] = img_metrics["em"]
        metrics["img_f1"] = img_metrics["f1"]
        metrics["txt_em"] = txt_metrics["em"]
        metrics["txt_f1"] = txt_metrics["f1"]
        metrics["img_recall_at_k"] = float(sum(img_recalls) / len(img_recalls)) if img_recalls else 0.0
        metrics["txt_recall_at_k"] = float(sum(txt_recalls) / len(txt_recalls)) if txt_recalls else 0.0
        metrics["img_precision_at_k"] = float(sum(img_precisions) / len(img_precisions)) if img_precisions else 0.0
        metrics["txt_precision_at_k"] = float(sum(txt_precisions) / len(txt_precisions)) if txt_precisions else 0.0
        metrics["img_avg_tokens"] = float(sum(img_tokens) / len(img_tokens)) if img_tokens else 0.0
        metrics["txt_avg_tokens"] = float(sum(txt_tokens) / len(txt_tokens)) if txt_tokens else 0.0
        if "acc" in img_metrics:
            metrics["img_acc"] = img_metrics["acc"]
            metrics["img_webqa_acc"] = img_metrics["acc"]
        if "acc" in txt_metrics:
            metrics["txt_acc"] = txt_metrics["acc"]
            metrics["txt_webqa_acc"] = txt_metrics["acc"]

        acc_part = f" Acc={metrics['acc']:.4f} |" if "acc" in metrics else ""
        img_acc_part = f" img Acc={metrics['img_acc']:.4f} |" if "img_acc" in metrics else ""
        txt_acc_part = f" txt Acc={metrics['txt_acc']:.4f} |" if "txt_acc" in metrics else ""
        logger.info(
            f"Val k={top_k} | n={len(predictions)} "
            f"(img={len(img_preds)}, txt={len(txt_preds)}) | "
            f"EM={metrics['em']:.4f} F1={metrics['f1']:.4f} "
            f"ROUGE-L={metrics['rouge_l']:.4f} |{acc_part} "
            f"img EM={metrics['img_em']:.4f} F1={metrics['img_f1']:.4f} |{img_acc_part} "
            f"txt EM={metrics['txt_em']:.4f} F1={metrics['txt_f1']:.4f} |{txt_acc_part} "
            f"Recall@{top_k}={metrics['recall_at_k']:.4f} | "
            f"avg_tokens={metrics.get('avg_tokens', 0):.1f}"
        )
        results[top_k] = metrics

    compressor.train()
    return results


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = cfg["model"]
    comp_cfg = cfg["compressor"]
    dec_cfg = cfg.get("decoder", {})
    train_cfg = cfg["training"]
    img_cfg = cfg.get("image", {})

    compressor_name = str(model_cfg["compressor_name"]).lower()
    generator_name = str(model_cfg.get("generator_name", "")).lower()
    if "qwen" in compressor_name or "qwen" in generator_name:
        raise ValueError(
            "This config looks like a Qwen-VL setup, but you launched scripts/train_llava.py. "
            "Use scripts/train_qwen.py with config_qwen.yaml instead."
        )

    # ---- compressor ----
    compressor = LLaVACompressor(
        model_name=model_cfg["compressor_name"],
        decoder_name=model_cfg.get("decoder_name", "meta-llama/Llama-3.2-1B-Instruct"),
        lora_r=comp_cfg["lora_r"],
        lora_alpha=comp_cfg["lora_alpha"],
        lora_dropout=comp_cfg.get("lora_dropout", 0.05),
        target_modules=comp_cfg.get("target_modules"),
        decode_lora_r=dec_cfg.get("lora_r", comp_cfg["lora_r"]),
        decode_lora_alpha=dec_cfg.get("lora_alpha", comp_cfg["lora_alpha"]),
        decode_lora_dropout=dec_cfg.get("lora_dropout", comp_cfg.get("lora_dropout", 0.05)),
        decode_target_modules=dec_cfg.get("target_modules", comp_cfg.get("target_modules")),
        retrieval_dim=comp_cfg["retrieval_dim"],
        generator_hidden=model_cfg.get("generator_hidden", 5120),
        num_latent_tokens=comp_cfg.get("num_latent_tokens", 1),
    ).to(device)

    # Gradient checkpointing: recompute activations during backward instead of
    # storing them.  Cuts activation memory ~4-6× at the cost of ~30% more compute.
    # enable_input_require_grads() is required for PEFT + gradient checkpointing.
    # Use compressor._lm() which resolves the correct language_model attribute
    # regardless of transformers version layout.
    lm = compressor._lm()
    lm.enable_input_require_grads()
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ---- generator (LLaVA-13b, frozen, optional) ----
    generator, generator_proc = None, None
    if train_cfg.get("lambda_image_distill", 0) > 0 or train_cfg.get("lambda_distill", 0) > 0:
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        gen_name = model_cfg["generator_name"]
        logger.info(f"Loading frozen generator: {gen_name}")
        generator_proc = AutoProcessor.from_pretrained(gen_name)
        generator = LlavaForConditionalGeneration.from_pretrained(
            gen_name, torch_dtype=torch.bfloat16, device_map=device
        )
        generator.eval()
        for p in generator.parameters():
            p.requires_grad_(False)

    retriever = ContrastiveRetriever(
        temperature=train_cfg.get("temperature", 0.07)
    ).to(device)

    # ---- data paths: CLI args override config ----
    data_cfg = cfg.get("data", {})
    train_path = args.data or data_cfg.get("train")
    val_path = args.val or data_cfg.get("val")

    if not train_path:
        raise ValueError("No training data path — set data.train in config or pass --data")

    # ---- load WebQA samples and split by modality ----
    logger.info(f"Loading WebQA data: {train_path}")
    all_samples = load_webqa_samples(train_path)
    # Infer modality if missing (older JSON without modality field)
    for s in all_samples:
        if "modality" not in s:
            s["modality"] = "image" if s.get("pos_image_ids") else "text"
    image_samples = [s for s in all_samples if s.get("modality") == "image"]
    text_samples = [s for s in all_samples if s.get("modality") == "text"]
    logger.info(f"  image: {len(image_samples)}  text: {len(text_samples)}")

    # ---- validation split ----
    val_image, val_text = [], []
    if val_path and os.path.exists(val_path):
        val_all = load_webqa_samples(val_path)
        for s in val_all:
            if "modality" not in s:
                s["modality"] = "image" if s.get("pos_image_ids") else "text"
        val_image = [s for s in val_all if s.get("modality") == "image"]
        val_text = [s for s in val_all if s.get("modality") == "text"]
        logger.info(f"  val image: {len(val_image)}  val text: {len(val_text)}")
    else:
        rng_s = random.Random(0)

        def _split(lst, frac=0.1):
            idx = list(range(len(lst)))
            rng_s.shuffle(idx)
            n = max(1, int(len(lst) * frac))
            return [lst[i] for i in idx[n:]], [lst[i] for i in idx[:n]]

        image_samples, val_image = _split(image_samples)
        text_samples, val_text = _split(text_samples)

    # ---- output paths: CLI arg > config paths.checkpoint > config training.save_dir ----
    paths_cfg = cfg.get("paths", {})
    save_dir = train_cfg.get("save_dir") or paths_cfg.get("results_dir", "checkpoints/")
    output = args.output or paths_cfg.get("checkpoint") or os.path.join(save_dir, "llava_model.pt")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    # ---- optimizer + scheduler ----
    params = list(compressor.parameters()) + list(retriever.parameters())
    optimizer = AdamW(params, lr=train_cfg["learning_rate"])

    n_img_steps = len(image_samples) // max(train_cfg.get("image_batch_size", 2), 1)
    n_txt_steps = len(text_samples) // max(train_cfg["batch_size"], 1)
    n_steps_per_epoch = max(n_img_steps, n_txt_steps, 1)
    grad_ac = train_cfg["gradient_accumulation_steps"]
    total_steps = n_steps_per_epoch * train_cfg["num_epochs"] // grad_ac
    warmup_steps = train_cfg.get("warmup_steps", 0)

    from torch.optim.lr_scheduler import LinearLR, SequentialLR
    if warmup_steps > 0:
        warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=1e-6)

    # ---- TensorBoard ----
    log_dir = getattr(args, "logdir", None) or os.path.join(os.path.dirname(os.path.abspath(output)), "curvellava")
    writer = SummaryWriter(log_dir=log_dir) if _TB else None

    # ---- config values ----
    lam_r = train_cfg.get("lambda_recon", 1.0)
    lam_c = train_cfg.get("lambda_contrast", 0.5)
    lam_d = train_cfg.get("lambda_distill", 0.0)
    lam_id = train_cfg.get("lambda_image_distill", 0.3)
    lambda_embed_recon = train_cfg.get("lambda_embed_recon", 0.0)

    # ---- image config ----
    scale = img_cfg.get("scale", 1 / 20)
    recon_negatives = train_cfg.get("recon_negatives", False)
    recon_query = train_cfg.get("recon_query", True)
    recon_title = train_cfg.get("recon_title", True)
    log_ev = train_cfg.get("log_every", 10)
    val_ev = train_cfg.get("val_every", 500)
    n_dist = train_cfg.get("n_distill_tokens", 8)
    max_neg = train_cfg.get("max_hard_negatives", 8)
    img_bs = train_cfg.get("image_batch_size", train_cfg["batch_size"])
    max_neg_img = train_cfg.get("max_neg_images", 16)
    retrieval_cfg = cfg.get("retrieval", {})
    top_k = retrieval_cfg.get("top_k", 5)
    top_k_values = sorted(set(retrieval_cfg.get("top_k_values", [top_k])))
    primary_k = top_k if top_k in top_k_values else top_k_values[-1]
    max_new_toks = cfg.get("generation", {}).get("max_new_tokens", 64)
    val_samples_n = train_cfg.get("val_samples", 200)
    student_distill_hard_negatives_max = train_cfg.get("student_distill_hard_negatives_max", 0)

    best_path = output.replace(".pt", "_best.pt")
    best_acc = -1.0
    global_step = 0
    rng = random.Random(42)

    running_mm = {
        k: 0.0 for k in [
            "total", "contrast", "txt_recon_sent", "txt_recon_query", "txt_distill",
            "img_recon_title", "img_recon_query", "img_recon_clip", "img_distill",
        ]
    }
    running_count = 0
    running_total = 0.0

    def _validate(step):
        nonlocal best_acc
        if generator is None:
            return
        metrics_by_k = run_validation_multi_k_unified(
            compressor, generator, generator_proc,
            val_image, val_text, scale, device,
            top_k_values=top_k_values,
            max_new_tokens=max_new_toks,
            max_samples=val_samples_n,
        )
        metrics = metrics_by_k[primary_k]
        if writer:
            for k, k_metrics in metrics_by_k.items():
                for name, value in k_metrics.items():
                    writer.add_scalar(f"val_k{k}/{name}", value, step)
            for name, value in metrics.items():
                writer.add_scalar(f"val/{name}", value, step)
            writer.flush()
        score = metrics.get("acc", metrics.get("f1", 0.0))
        if score > best_acc:
            best_acc = score
            compressor.save(best_path)
            score_name = "Acc" if "acc" in metrics else "F1"
            logger.info(f"  New best {score_name}={best_acc:.4f} at k={primary_k} — saved {best_path}")

    if val_image or val_text:
        _validate(0)

    for epoch in range(train_cfg["num_epochs"]):
        logger.info(f"Epoch {epoch + 1}/{train_cfg['num_epochs']}")
        optimizer.zero_grad()

        for step_idx in tqdm(range(n_steps_per_epoch), desc=f"Epoch {epoch + 1}"):
            step_loss = torch.tensor(0.0, device=device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                mb = build_multimodal_batch(
                    all_samples, max(train_cfg["batch_size"], img_bs), max_neg, max_neg_img, scale, rng
                )
                if mb:
                    md = multimodal_train_step(
                        compressor, retriever, mb, device=device,
                        lam_recon=lam_r,
                        lam_embed_recon=lambda_embed_recon,
                        lam_cont=lam_c,
                        lam_distill=lam_d,
                        lam_img_distill=lam_id,
                        generator=generator,
                        generator_proc=generator_proc,
                        n_distill_tokens=n_dist,
                        recon_negatives=recon_negatives,
                        recon_query=recon_query,
                        recon_title=recon_title,
                        student_distill_hard_negatives_max=student_distill_hard_negatives_max,
                        rng=rng,
                    )
                    step_loss = step_loss + md["total"] / grad_ac
                    for k in running_mm:
                        running_mm[k] += md.get(k, 0.0)
                    running_count += 1

            running_total += step_loss.item() * grad_ac

            if step_loss.requires_grad:
                step_loss.backward()

            if (global_step + 1) % grad_ac == 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if global_step > 0 and global_step % val_ev == 0 and (val_image or val_text):
                _validate(global_step)

            if global_step % log_ev == 0:
                lr = scheduler.get_last_lr()[0]
                n = max(running_count, 1)
                avg_m = {k: v / n for k, v in running_mm.items()}
                avg_total = running_total / max(log_ev, 1)
                logger.info(
                    f"Step {global_step} | lr={lr:.2e} | total={avg_total:.4f} | "
                    f"contrast={avg_m['contrast']:.4f} | "
                    f"txt: recon_s={avg_m['txt_recon_sent']:.4f} recon_q={avg_m['txt_recon_query']:.4f} "
                    f"distill={avg_m['txt_distill']:.4f} | "
                    f"img: recon_title={avg_m['img_recon_title']:.4f} recon_q={avg_m['img_recon_query']:.4f} "
                    f"recon_clip={avg_m['img_recon_clip']:.4f} distill={avg_m['img_distill']:.4f}"
                )
                if writer:
                    writer.add_scalar("train/loss_total", avg_total, global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                    writer.add_scalar("train/mm_loss_total", avg_m["total"], global_step)
                    writer.add_scalar("train/mm_contrast", avg_m["contrast"], global_step)
                    writer.add_scalar("train/txt_recon_sent", avg_m["txt_recon_sent"], global_step)
                    writer.add_scalar("train/txt_recon_query", avg_m["txt_recon_query"], global_step)
                    writer.add_scalar("train/txt_distill", avg_m["txt_distill"], global_step)
                    writer.add_scalar("train/img_recon_title", avg_m["img_recon_title"], global_step)
                    writer.add_scalar("train/img_recon_query", avg_m["img_recon_query"], global_step)
                    writer.add_scalar("train/img_recon_clip", avg_m["img_recon_clip"], global_step)
                    writer.add_scalar("train/img_distill", avg_m["img_distill"], global_step)
                    writer.flush()
                running_mm = {k: 0.0 for k in running_mm}
                running_count = 0
                running_total = 0.0

            global_step += 1

        # End-of-epoch save + validation
        compressor.save(output.replace(".pt", f"_epoch{epoch + 1}.pt"))
        if val_image or val_text:
            _validate(global_step)

    compressor.save(output)
    logger.info(f"Training complete. Saved: {output}")
    if writer:
        writer.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LLaVA-based latent retrieval on WebQA.")
    parser.add_argument("--config", default="config_llava.yaml")
    parser.add_argument("--data", default=None,
                        help="WebQA training JSON (overrides data.train in config)")
    parser.add_argument("--val", default=None,
                        help="WebQA validation JSON (overrides data.val in config)")
    parser.add_argument("--output", default=None,
                        help="Output checkpoint path (overrides paths.checkpoint in config)")
    parser.add_argument("--logdir", default=None)
    args = parser.parse_args()
    train(args)

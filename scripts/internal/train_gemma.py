"""
Qwen2.5-VL Latent Retrieval Training
=====================================
Training script for the Qwen2.5-VL-3B compressor on WebQA.
Uses Qwen2.5-VL-7B as the frozen teacher for distillation.

Differs from train_llava.py:
  - QwenVLCompressor instead of LLaVACompressor
  - Qwen chat template throughout (no Vicuna "USER:/ASSISTANT:")
  - Generator is Qwen2VLForConditionalGeneration (+ image_grid_thw in forward)
  - lambda_embed_recon disabled (no CLIP CLS equivalent in Qwen2VL)
  - compress_batch receives raw content strings (template applied internally)

Usage
-----
python scripts/train_gemma.py \\
    --config config_gemma3_4B_12B.yaml \\
    --data   data/webqa_train.json \\
    --val    data/webqa_val.json \\
    --output checkpoints/qwen_model.pt
"""

import argparse
import json
import logging
import os
import random
import sys

_RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _RELEASE_ROOT not in sys.path:
    sys.path.insert(0, _RELEASE_ROOT)
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
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
from src.gemma_compressor import QwenVLCompressor, resolve_qwen_vl_model_class
from src.region_encoder import image_doc_id, resize_image
from src.retriever import ContrastiveRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Qwen2.5-VL-7B system prompt
_QWEN_SYS = (
    "You are a helpful assistant. "
    "Answer the question concisely (a few words or a short phrase) "
    "based on the provided context."
)
_CTX_PLACEHOLDER = "ZZZCTXZZZ"


def _extend_processor_forward_kwargs(enc, input_ids, attention_mask, dtype=torch.bfloat16):
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
    }
    for key in ("pixel_values", "image_grid_thw", "token_type_ids"):
        if key not in enc:
            continue
        value = enc[key]
        if key == "pixel_values":
            value = value.to(dtype)
        elif key == "token_type_ids" and value.shape[1] != input_ids.shape[1]:
            pad_len = input_ids.shape[1] - value.shape[1]
            value = torch.cat(
                [
                    value,
                    torch.zeros(value.shape[0], pad_len, device=value.device, dtype=value.dtype),
                ],
                dim=1,
            )
        kwargs[key] = value
    return kwargs


def _is_gemma3_processor(processor) -> bool:
    return "gemma3" in processor.__class__.__name__.lower()


def _vl_system_message(processor):
    if _is_gemma3_processor(processor):
        return {"role": "system", "content": [{"type": "text", "text": _QWEN_SYS}]}
    return {"role": "system", "content": _QWEN_SYS}


# ---------------------------------------------------------------------------
# Batch builders (identical to train_llava.py)
# ---------------------------------------------------------------------------

def build_text_batch(
        samples: List[Dict],
        batch_size: int,
        max_hard_negatives: int = 32,
        rng: Optional[random.Random] = None,
) -> List[Dict]:
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

        pos_images = []
        for img_id, title in zip(s["pos_image_ids"], s["pos_captions"]):
            item = _load_image(img_id, title)
            if item is not None:
                pos_images.append(item)

        if not pos_images:
            continue

        neg_images = []
        for img_id, title in zip(s["neg_image_ids"][:max_neg_images],
                                 s["neg_captions"][:max_neg_images]):
            item = _load_image(img_id, title)
            if item is not None:
                neg_images.append(item)

        batch.append({
            "question": s["question"],
            "answers": s["answers"],
            "pos_regions": pos_images,   # list of (PIL, doc_id, title)
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
# Distillation losses (Qwen chat template)
# ---------------------------------------------------------------------------

def _text_distill_loss(
        generator,
        gen_tokenizer,
        questions: List[str],
        pos_sents: List[str],
        proj_latents: torch.Tensor,   # (total_pos, generator_hidden)
        pos_counts: List[int],
        device: str,
        n_distill_tokens: int,
) -> torch.Tensor:
    """
    KL distillation for text evidence.
    Teacher: Qwen2.5-VL-7B sees raw Evidence N: sentences as inputs_embeds.
    Student: Qwen2.5-VL-7B sees projected [MEM] latents in the same positions.
    Both use Qwen chat template (no pixel_values = text-only mode).

    Mirrors compute_distillation_loss_multi from src/distillation.py:
      1. Embed teacher prompt → generate N tokens (no_grad)
      2. Extend both teacher and student with first N-1 token embeddings
      3. One forward pass each → logits[-N:] → KL sum / n_distill_tokens
    """
    embed_fn = generator.get_input_embeddings()
    dtype    = next(generator.parameters()).dtype
    eos_id   = gen_tokenizer.eos_token_id

    def _e(text: str, add_bos: bool) -> torch.Tensor:
        if add_bos:
            ids = gen_tokenizer(text, return_tensors="pt").input_ids.to(device)
        else:
            ids = gen_tokenizer(
                text, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(device)
        return embed_fn(ids).to(dtype)

    # Group pos_sents and proj_latents by question
    valid = []
    offset = 0
    for q, n_pos in zip(questions, pos_counts):
        if n_pos > 0:
            valid.append((q,
                          pos_sents[offset: offset + n_pos],
                          proj_latents[offset: offset + n_pos]))
        offset += n_pos

    if not valid:
        return torch.tensor(0.0, device=device)

    total_kl = torch.tensor(0.0, device=device)
    count = 0

    for q, facts, sample_proj in valid:
        try:
            # ── Teacher prompt (Qwen chat format, text-only) ─────────────────
            evidence_text = "\n".join(
                f"Evidence {k+1}: {f}" for k, f in enumerate(facts)
            )
            teacher_text = (
                f"<|im_start|>system\n{_QWEN_SYS}<|im_end|>\n"
                f"<|im_start|>user\n{evidence_text}\nQuestion: {q}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            t_embeds = _e(teacher_text, add_bos=True)   # (1, L_t, H)

            # Step 1 – generate N tokens
            with torch.no_grad():
                gen_out = generator.generate(
                    inputs_embeds=t_embeds,
                    max_new_tokens=n_distill_tokens,
                    do_sample=False,
                    pad_token_id=eos_id,
                )
            L_t     = t_embeds.shape[1]
            gen_ids = (gen_out[0, L_t:] if gen_out.shape[1] > L_t else gen_out[0])
            gen_ids = gen_ids[:n_distill_tokens]

            # Step 2 – extend teacher with first N-1 token embeddings
            if n_distill_tokens > 1:
                tf_embs      = embed_fn(gen_ids[:n_distill_tokens - 1].unsqueeze(0)).to(dtype)
                t_embeds_ext = torch.cat([t_embeds, tf_embs], dim=1)
            else:
                t_embeds_ext = t_embeds

            # Step 3 – teacher forward (no grad)
            with torch.no_grad():
                t_out = generator(inputs_embeds=t_embeds_ext, use_cache=False)
            t_logits = t_out.logits[0, -n_distill_tokens:, :].float()   # (N, V)

        except Exception as e:
            logger.warning(f"Text distill teacher failed: {e}; skipping sample.")
            continue

        # ── Student: Qwen system prefix + interleaved latents + question ─────
        parts = [_e(
            f"<|im_start|>system\n{_QWEN_SYS}<|im_end|>\n<|im_start|>user\n",
            add_bos=True
        )]
        for k, lat in enumerate(sample_proj):
            sep = "" if k == 0 else "\n"
            parts.append(_e(f"{sep}Evidence {k+1}: ", add_bos=False))
            parts.append(lat.reshape(-1, lat.shape[-1]).to(dtype).unsqueeze(0))
        parts.append(_e(f"\nQuestion: {q}<|im_end|>\n<|im_start|>assistant\n", add_bos=False))
        s_embeds = torch.cat(parts, dim=1)

        if n_distill_tokens > 1:
            tf_embs  = embed_fn(gen_ids[:n_distill_tokens - 1].unsqueeze(0)).to(dtype)
            s_embeds = torch.cat([s_embeds, tf_embs], dim=1)

        s_out    = generator(inputs_embeds=s_embeds, use_cache=False)
        s_logits = s_out.logits[0, -n_distill_tokens:, :].float()   # (N, V)

        kl = F.kl_div(
            F.log_softmax(s_logits.reshape(-1, s_logits.shape[-1]), dim=-1),
            F.softmax(t_logits.reshape(-1, t_logits.shape[-1]).detach(), dim=-1),
            reduction="sum",
        ) / n_distill_tokens
        total_kl = total_kl + kl
        count += 1

    return total_kl / max(count, 1)


def _qwen_distill_loss(
        generator,
        gen_processor,
        questions: List[str],
        captions: List[str],
        pos_pil: List,               # PIL images (parallel to flat pos list)
        proj_latents: torch.Tensor,  # (total_pos, generator_hidden)
        pos_counts: List[int],
        device: str,
        n_distill_tokens: int,
) -> torch.Tensor:
    """
    Legacy Qwen image-distillation implementation kept for reference.

    Important: the current training pipeline does NOT call this function.
    Active training uses `_qwen_distill_loss_hotpot(...)` below, where both
    teacher and student are constructed from the same split chat scaffold and
    the student inserts `Latent context i:` labels plus latent tokens.

    This older path uses a different prompt realization for the student
    (`Evidence 1 (latent): ...`) and is retained only to preserve the earlier
    experimental implementation.

    KL distillation for image evidence.
    Teacher: Qwen2.5-VL-7B sees the real image via pixel_values + image_grid_thw.
    Student: Qwen2.5-VL-7B sees projected [MEM] latent as inputs_embeds (no image).

    Teacher forward: extend input_ids (text-space) with first N-1 generated token
    ids; LLaVA-style — image_grid_thw is passed unchanged in the extended forward
    since it describes the original image patches, not the appended text tokens.

    Student forward: inputs_embeds only (no pixel_values), so Qwen2VL treats the
    sequence as text-only — no image-expansion takes place.
    """
    embed_fn  = generator.get_input_embeddings()
    dtype     = next(generator.parameters()).dtype
    eos_id    = gen_processor.tokenizer.eos_token_id

    def _e(text: str, add_bos: bool) -> torch.Tensor:
        if add_bos:
            ids = gen_processor.tokenizer(text, return_tensors="pt").input_ids.to(device)
        else:
            ids = gen_processor.tokenizer(
                text, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(device)
        return embed_fn(ids).to(dtype)

    # Group by question
    valid = []
    offset = 0
    for q, n_pos in zip(questions, pos_counts):
        if n_pos > 0:
            valid.append((
                q,
                captions[offset: offset + n_pos],
                proj_latents[offset: offset + n_pos],
                pos_pil[offset],          # use first positive image for teacher
            ))
        offset += n_pos

    if not valid:
        return torch.tensor(0.0, device=device)

    total_kl = torch.tensor(0.0, device=device)
    count = 0

    for q, sample_caps, sample_proj, pil_img in valid:
        try:
            # ── Teacher: Qwen2VL with real image ─────────────────────────────
            messages = [
                _vl_system_message(gen_processor),
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text",  "text": (
                            "Evidence 1: [image]\n"
                            + "\n".join(f"Evidence {k+2}: {c}" for k, c in enumerate(sample_caps))
                            + f"\nQuestion: {q}"
                        )},
                    ],
                },
            ]
            teacher_text = gen_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            t_enc = gen_processor(
                text=teacher_text, images=[pil_img], return_tensors="pt"
            ).to(device)

            # Step 1 – generate N tokens
            with torch.no_grad():
                gen_out = generator.generate(
                    **t_enc,
                    max_new_tokens=n_distill_tokens,
                    do_sample=False,
                    pad_token_id=eos_id,
                )
            L_t     = t_enc.input_ids.shape[1]
            gen_ids = (gen_out[0, L_t:] if gen_out.shape[1] > L_t else gen_out[0])
            gen_ids = gen_ids[:n_distill_tokens]

            # Step 2 – extend teacher input_ids with first N-1 generated tokens
            if n_distill_tokens > 1:
                tf_ids   = gen_ids[:n_distill_tokens - 1].unsqueeze(0)   # (1, N-1)
                ext_ids  = torch.cat([t_enc.input_ids, tf_ids], dim=1)
                ext_mask = torch.cat([
                    t_enc.attention_mask,
                    torch.ones(1, n_distill_tokens - 1, device=device,
                               dtype=t_enc.attention_mask.dtype),
                ], dim=1)
            else:
                ext_ids  = t_enc.input_ids
                ext_mask = t_enc.attention_mask

            # Step 3 – teacher forward (image_grid_thw describes original image)
            with torch.no_grad():
                t_out = generator(
                    **_extend_processor_forward_kwargs(t_enc, ext_ids, ext_mask, dtype)
                )
            t_logits = t_out.logits[0, -n_distill_tokens:, :].float()   # (N, V)

        except Exception as e:
            logger.warning(f"Qwen image distill teacher failed: {e}; skipping sample.")
            continue

        # ── Student: projected latents + captions + question (no image) ──────
        parts = [_e(
            f"<|im_start|>system\n{_QWEN_SYS}<|im_end|>\n<|im_start|>user\n",
            add_bos=True
        )]
        parts.append(_e("Evidence 1 (latent): ", add_bos=False))
        parts.append(sample_proj[0].reshape(-1, sample_proj.shape[-1]).to(dtype).unsqueeze(0))
        for k, cap in enumerate(sample_caps):
            parts.append(_e(f"\nEvidence {k+2}: {cap}", add_bos=False))
        parts.append(_e(f"\nQuestion: {q}<|im_end|>\n<|im_start|>assistant\n", add_bos=False))
        s_embeds = torch.cat(parts, dim=1)

        if n_distill_tokens > 1:
            tf_embs  = embed_fn(gen_ids[:n_distill_tokens - 1].unsqueeze(0)).to(dtype)
            s_embeds = torch.cat([s_embeds, tf_embs], dim=1)

        # Student uses inputs_embeds only — no pixel_values, text-only forward
        s_out    = generator(inputs_embeds=s_embeds, use_cache=False)
        s_logits = s_out.logits[0, -n_distill_tokens:, :].float()   # (N, V)

        kl = F.kl_div(
            F.log_softmax(s_logits.reshape(-1, s_logits.shape[-1]), dim=-1),
            F.softmax(t_logits.reshape(-1, t_logits.shape[-1]).detach(), dim=-1),
            reduction="sum",
        ) / n_distill_tokens
        total_kl = total_kl + kl
        count += 1

    return total_kl / max(count, 1)


def _split_qwen_image_chat_template(processor, query: str):
    full = processor.apply_chat_template(
        [
            _vl_system_message(processor),
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Context:\n"},
                    {"type": "image"},
                    {"type": "text", "text": f"Title: {_CTX_PLACEHOLDER}\nQuestion: {query}"},
                ],
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix_str, suffix_str = full.split(_CTX_PLACEHOLDER)
    return prefix_str, suffix_str


def _text_distill_loss_hotpot(
        generator,
        gen_tokenizer,
        questions: List[str],
        pos_sents: List[str],
        student_proj_latents: List[torch.Tensor],
        pos_counts: List[int],
        device: str,
        n_distill_tokens: int,
) -> torch.Tensor:
    embed_fn = generator.get_input_embeddings()
    dtype = next(generator.parameters()).dtype
    eos_id = gen_tokenizer.eos_token_id

    def _e(text: str, max_len: int = 512):
        return embed_fn(_tok(gen_tokenizer, text, device, max_len)).to(dtype)

    valid = []
    offset = 0
    for idx, (q, n_pos) in enumerate(zip(questions, pos_counts)):
        if n_pos > 0:
            valid.append((q, pos_sents[offset: offset + n_pos], student_proj_latents[idx]))
        offset += n_pos

    if not valid:
        return torch.tensor(0.0, device=device)

    total_kl = torch.tensor(0.0, device=device)
    count = 0
    for q, facts, sample_proj in valid:
        prefix_str, suffix_str = _split_chat_template(gen_tokenizer, q)
        try:
            ctx_text = "\n".join(f"Context {i + 1}: {fact}" for i, fact in enumerate(facts))
            t_embeds = _e(prefix_str + ctx_text + suffix_str, max_len=1024)
            with torch.no_grad():
                gen_out = generator.generate(
                    inputs_embeds=t_embeds,
                    max_new_tokens=n_distill_tokens,
                    do_sample=False,
                    pad_token_id=eos_id,
                )
            l_teacher = t_embeds.shape[1]
            gen_ids = gen_out[0, l_teacher:] if gen_out.shape[1] > l_teacher else gen_out[0]
            gen_ids = gen_ids[:n_distill_tokens]
            n_actual = int(gen_ids.numel())
            if n_actual == 0:
                continue

            tf_embs = None
            if n_actual > 1:
                tf_embs = embed_fn(gen_ids[: n_actual - 1].unsqueeze(0)).to(dtype)
                t_embeds = torch.cat([t_embeds, tf_embs], dim=1)

            with torch.no_grad():
                t_out = generator(inputs_embeds=t_embeds, use_cache=False)
            t_logits = t_out.logits[0, -n_actual:, :].float()
        except Exception as e:
            logger.warning(f"Text distill teacher failed: {e}; skipping sample.")
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


def _qwen_distill_loss_hotpot(
        generator,
        gen_processor,
        questions: List[str],
        captions: List[str],
        pos_pil: List,
        student_proj_latents: List[torch.Tensor],
        pos_counts: List[int],
        device: str,
        n_distill_tokens: int,
) -> torch.Tensor:
    embed_fn = generator.get_input_embeddings()
    dtype = next(generator.parameters()).dtype
    eos_id = gen_processor.tokenizer.eos_token_id

    def _e(text: str, max_len: int = 512):
        return embed_fn(_tok(gen_processor.tokenizer, text, device, max_len)).to(dtype)

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
        prefix_str, suffix_str = _split_chat_template(gen_processor.tokenizer, q)
        try:
            content = []
            for i, (cap, pil_img) in enumerate(zip(sample_caps, sample_pil)):
                content.append({"type": "text", "text": f"Context {i + 1}:\n"})
                content.append({"type": "image", "image": pil_img})
                content.append({"type": "text", "text": f"Title: {cap}\n"})
            content.append({"type": "text", "text": f"Question: {q}"})
            teacher_text = gen_processor.apply_chat_template(
                [
                    _vl_system_message(gen_processor),
                    {"role": "user", "content": content},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            t_enc = gen_processor(
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

            tf_embs = None
            ext_ids = t_enc.input_ids
            ext_mask = t_enc.attention_mask
            if n_actual > 1:
                tf_ids = gen_ids[: n_actual - 1].unsqueeze(0)
                tf_embs = embed_fn(tf_ids).to(dtype)
                ext_ids = torch.cat([t_enc.input_ids, tf_ids], dim=1)
                ext_mask = torch.cat(
                    [t_enc.attention_mask,
                     torch.ones(1, n_actual - 1, device=device, dtype=t_enc.attention_mask.dtype)],
                    dim=1,
                )

            with torch.no_grad():
                t_out = generator(
                    **_extend_processor_forward_kwargs(t_enc, ext_ids, ext_mask, dtype)
                )
            t_logits = t_out.logits[0, -n_actual:, :].float()
        except Exception as e:
            logger.warning(f"Qwen image distill teacher failed: {e}; skipping sample.")
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


# ---------------------------------------------------------------------------
# Training steps
# ---------------------------------------------------------------------------

def text_train_step(
        compressor: QwenVLCompressor,
        retriever: ContrastiveRetriever,
        batch: List[Dict],
        lam_recon: float,
        lam_cont: float,
        device: str,
        lam_distill: float = 0.0,
        generator=None,
        gen_tokenizer=None,
        n_distill_tokens: int = 8,
        recon_negatives: bool = False,
        recon_query: bool = False,
        student_distill_hard_negatives_max: int = 0,
        rng: Optional[random.Random] = None,
) -> Dict:
    """Text step: L_recon + L_contrast [+ L_distill]."""
    if rng is None:
        rng = random.Random()
    questions = [b["question"] for b in batch]
    pos_sents = [s for b in batch for s in b["pos_sentences"]]
    neg_sents = [s for b in batch for s in b["hard_neg_sents"]]

    # compress_batch applies Qwen chat template internally — pass raw text
    recon_sents = pos_sents + neg_sents if recon_negatives else pos_sents
    l_recon_sent = compressor.reconstruction_loss_batch(
        recon_sents, adapter="compress", decode_adapter="decode"
    )
    l_recon_query = (
        compressor.reconstruction_loss_batch(
            questions, adapter="query", decode_adapter="query_decode"
        ) if recon_query else torch.tensor(0.0, device=device)
    )
    l_recon = l_recon_sent + l_recon_query

    pos_latents = compressor.compress_batch(pos_sents, with_grad=True, adapter="compress")
    pos_embs    = compressor.get_retrieval_embedding(pos_latents)

    q_latents = compressor.embed_query_batch(questions, with_grad=True)
    q_embs    = compressor.get_retrieval_embedding(q_latents)

    pos_counts = [len(b["pos_sentences"]) for b in batch]
    pos_embs_list, offset = [], 0
    for c in pos_counts:
        pos_embs_list.append(pos_embs[offset:offset + c])
        offset += c

    neg_embs = None
    neg_latents_list = []
    if neg_sents:
        neg_latents = compressor.compress_batch(neg_sents, with_grad=True, adapter="compress")
        neg_embs = compressor.get_retrieval_embedding(neg_latents)
        offset = 0
        for b in batch:
            n = len(b["hard_neg_sents"])
            neg_latents_list.append(neg_latents[offset:offset + n])
            offset += n
    else:
        neg_latents_list = [
            pos_latents.new_zeros((0, pos_latents.shape[-1]))
            for _ in batch
        ]

    l_contrast = retriever.contrastive_loss(q_embs, pos_embs_list, neg_embs)

    l_distill = torch.tensor(0.0, device=device)
    if lam_distill > 0 and generator is not None and pos_latents.shape[0] > 0:
        student_proj_latents = []
        offset = 0
        for i, n_pos in enumerate(pos_counts):
            student_latents = pos_latents[offset:offset + n_pos]
            offset += n_pos

            if student_distill_hard_negatives_max > 0 and neg_latents_list[i].shape[0] > 0:
                max_extra = min(student_distill_hard_negatives_max, neg_latents_list[i].shape[0])
                n_extra = rng.randint(0, max_extra)
                if n_extra > 0:
                    selected = rng.sample(range(neg_latents_list[i].shape[0]), k=n_extra)
                    student_latents = torch.cat([student_latents, neg_latents_list[i][selected]], dim=0)

            student_proj_latents.append(compressor.project_for_generator(student_latents))

        l_distill = _text_distill_loss_hotpot(
            generator, gen_tokenizer,
            questions, pos_sents, student_proj_latents,
            pos_counts, device, n_distill_tokens,
        )

    total = lam_recon * l_recon + lam_cont * l_contrast + lam_distill * l_distill
    return {
        "total": total,
        "recon_sent": l_recon_sent.item(),
        "recon_query": l_recon_query.item(),
        "contrast": l_contrast.item(),
        "distill": l_distill.item() if isinstance(l_distill, torch.Tensor) else 0.0,
    }


def image_train_step(
        compressor: QwenVLCompressor,
        retriever: ContrastiveRetriever,
        batch: List[Dict],
        lam_cont: float,
        lam_distill: float,
        device: str,
        generator=None,
        gen_processor=None,
        n_distill_tokens: int = 8,
        lam_recon: float = 0.5,
        lam_embed_recon: float = 0.0,
        recon_query: bool = False,
        recon_title: bool = True,
        recon_negatives: bool = False,
        student_distill_hard_negatives_max: int = 0,
        rng: Optional[random.Random] = None,
) -> Dict:
    """Image step: L_contrast + L_recon_title + L_recon_query [+ L_distill]."""
    if rng is None:
        rng = random.Random()
    questions  = [b["question"] for b in batch]
    pos_pil    = [r[0] for b in batch for r in b["pos_regions"]]
    pos_titles = [r[2] for b in batch for r in b["pos_regions"]]

    if not pos_pil:
        z = torch.tensor(0.0, device=device)
        return {"total": z, "contrast": 0.0, "distill": 0.0,
                "recon_title": 0.0, "recon_query": 0.0, "recon_clip": 0.0}

    neg_pil    = [r[0] for b in batch for r in b["neg_regions"]]
    neg_titles = [r[2] for b in batch for r in b["neg_regions"]]

    all_titles = pos_titles + neg_titles
    all_pil = pos_pil + neg_pil
    all_latents = compressor.compress_batch(
        all_titles, images=all_pil, with_grad=True, adapter="compress"
    )
    all_embs = compressor.get_retrieval_embedding(all_latents)

    n_pos_total = len(pos_pil)
    pos_latents = all_latents[:n_pos_total]
    pos_embs = all_embs[:n_pos_total]
    neg_embs = all_embs[n_pos_total:] if neg_pil else None

    pos_counts = [len(b["pos_regions"]) for b in batch]
    neg_counts = [len(b["neg_regions"]) for b in batch]
    pos_embs_list, offset = [], 0
    for c in pos_counts:
        pos_embs_list.append(pos_embs[offset:offset + c])
        offset += c

    neg_latents_list = []
    if neg_pil:
        neg_latents = all_latents[n_pos_total:]
        offset = 0
        for c in neg_counts:
            neg_latents_list.append(neg_latents[offset:offset + c])
            offset += c
    else:
        neg_latents_list = [
            pos_latents.new_zeros((0, pos_latents.shape[-1]))
            for _ in batch
        ]

    if recon_title:
        rt_titles = pos_titles
        rt_pil    = pos_pil
        if recon_negatives and neg_pil:
            rt_titles = pos_titles + neg_titles
            rt_pil    = pos_pil + neg_pil
        l_recon_title = compressor.reconstruction_loss_batch(
            rt_titles, images=rt_pil,
            adapter="compress", decode_adapter="image_decode",
            target_texts=rt_titles,
        )
    else:
        l_recon_title = torch.tensor(0.0, device=device)

    clip_titles = pos_titles + neg_titles if (recon_negatives and neg_pil) else pos_titles
    clip_pil = pos_pil + neg_pil if (recon_negatives and neg_pil) else pos_pil
    l_recon_clip = (
        compressor.embed_reconstruction_loss_batch(
            clip_titles,
            images=clip_pil,
            adapter="compress",
        ) if lam_embed_recon > 0 and clip_pil else torch.tensor(0.0, device=device)
    )

    l_recon_query = (
        compressor.reconstruction_loss_batch(
            questions, adapter="query", decode_adapter="query_decode"
        ) if recon_query else torch.tensor(0.0, device=device)
    )

    q_latents = compressor.embed_query_batch(questions, with_grad=True)
    q_embs    = compressor.get_retrieval_embedding(q_latents)

    l_contrast = retriever.contrastive_loss(q_embs, pos_embs_list, neg_embs)

    l_distill = torch.tensor(0.0, device=device)
    if lam_distill > 0 and generator is not None:
        student_proj_latents = []
        offset = 0
        for i, n_pos in enumerate(pos_counts):
            student_latents = pos_latents[offset:offset + n_pos]
            offset += n_pos

            if student_distill_hard_negatives_max > 0 and neg_latents_list[i].shape[0] > 0:
                max_extra = min(student_distill_hard_negatives_max, neg_latents_list[i].shape[0])
                n_extra = rng.randint(0, max_extra)
                if n_extra > 0:
                    selected = rng.sample(range(neg_latents_list[i].shape[0]), k=n_extra)
                    student_latents = torch.cat([student_latents, neg_latents_list[i][selected]], dim=0)

            student_proj_latents.append(compressor.project_for_generator(student_latents))

        l_distill = _qwen_distill_loss_hotpot(
            generator, gen_processor,
            questions, pos_titles, pos_pil, student_proj_latents,
            pos_counts, device, n_distill_tokens,
        )

    total = (
        lam_cont * l_contrast
        + lam_distill * l_distill
        + lam_recon * (l_recon_title + l_recon_query)
        + lam_embed_recon * l_recon_clip
    )
    return {
        "total": total,
        "recon_title": l_recon_title.item(),
        "recon_query": l_recon_query.item(),
        "recon_clip": l_recon_clip.item(),
        "contrast": l_contrast.item(),
        "distill": l_distill.item() if isinstance(l_distill, torch.Tensor) else 0.0,
    }


def multimodal_train_step(
        compressor: QwenVLCompressor,
        retriever: ContrastiveRetriever,
        batch: List[Dict],
        device: str,
        lam_recon: float,
        lam_embed_recon: float,
        lam_cont: float,
        lam_distill: float,
        lam_img_distill: float,
        generator=None,
        gen_processor=None,
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

    # ---- query embeddings
    q_latents = compressor.embed_query_batch(questions, with_grad=True)
    q_embs = compressor.get_retrieval_embedding(q_latents)

    # ---- flatten text/image candidates
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
        pos_text_latents = compressor.compress_batch(pos_texts, with_grad=True, adapter="compress")
        pos_text_embs = compressor.get_retrieval_embedding(pos_text_latents)

    pos_img_embs = pos_img_latents = None
    if pos_pil:
        pos_img_latents = compressor.compress_batch(pos_titles, images=pos_pil, with_grad=True, adapter="compress")
        pos_img_embs = compressor.get_retrieval_embedding(pos_img_latents)

    neg_text_latents_list = []
    neg_text_embs_list = []
    if neg_texts:
        neg_text_latents = compressor.compress_batch(neg_texts, with_grad=True, adapter="compress")
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
        neg_img_latents = compressor.compress_batch(neg_titles, images=neg_pil, with_grad=True, adapter="compress")
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
            ref_latent = None
            if pos_text_latents is not None:
                ref_latent = pos_text_latents
            elif pos_img_latents is not None:
                ref_latent = pos_img_latents
            if ref_latent is not None:
                sampled_mixed_neg_latents.append(ref_latent.new_zeros((0, ref_latent.shape[-1])))
            else:
                sampled_mixed_neg_latents.append(torch.empty(0, 0, device=device))

            ref_emb = None
            if pos_text_embs is not None:
                ref_emb = pos_text_embs
            elif pos_img_embs is not None:
                ref_emb = pos_img_embs
            if ref_emb is not None:
                sampled_mixed_neg_embs.append(ref_emb.new_zeros((0, ref_emb.shape[-1])))
            else:
                sampled_mixed_neg_embs.append(torch.empty(0, 0, device=device))

    # ---- joint positive pool per query
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

    # ---- reconstruction
    recon_texts = pos_texts + neg_texts if recon_negatives else pos_texts
    l_txt_recon_sent = (
        compressor.reconstruction_loss_batch(
            recon_texts,
            adapter="compress",
            decode_adapter="decode",
        ) if recon_texts else torch.tensor(0.0, device=device)
    )
    l_txt_recon_query = (
        compressor.reconstruction_loss_batch(
            questions, adapter="query", decode_adapter="query_decode"
        ) if recon_query else torch.tensor(0.0, device=device)
    )

    if recon_title:
        recon_titles = pos_titles + neg_titles if (recon_negatives and neg_pil) else pos_titles
        recon_pil = pos_pil + neg_pil if (recon_negatives and neg_pil) else pos_pil
        l_img_recon_title = (
            compressor.reconstruction_loss_batch(
                recon_titles,
                images=recon_pil,
                adapter="compress",
                decode_adapter="image_decode",
                target_texts=recon_titles,
            ) if recon_titles else torch.tensor(0.0, device=device)
        )
    else:
        l_img_recon_title = torch.tensor(0.0, device=device)

    l_img_recon_query = (
        compressor.reconstruction_loss_batch(
            questions, adapter="query", decode_adapter="query_decode"
        ) if recon_query else torch.tensor(0.0, device=device)
    )
    clip_titles = pos_titles + neg_titles if (recon_negatives and neg_pil) else pos_titles
    clip_pil = pos_pil + neg_pil if (recon_negatives and neg_pil) else pos_pil
    l_img_recon_clip = (
        compressor.embed_reconstruction_loss_batch(
            clip_titles,
            images=clip_pil,
            adapter="compress",
        ) if lam_embed_recon > 0 and clip_pil else torch.tensor(0.0, device=device)
    )

    # ---- distillation only on positive modality evidence
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

        l_txt_distill = _text_distill_loss_hotpot(
            generator, gen_processor.tokenizer,
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

        l_img_distill = _qwen_distill_loss_hotpot(
            generator, gen_processor,
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
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation_legacy_split(
        compressor: QwenVLCompressor,
        generator,
        gen_processor,
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
    from src.evaluation import evaluate

    compressor.eval()

    embed_fn = generator.get_input_embeddings()
    dtype    = next(generator.parameters()).dtype
    gen_tok  = gen_processor.tokenizer

    def _e(text: str, add_bos: bool) -> torch.Tensor:
        if add_bos:
            ids = gen_tok(text, return_tensors="pt").input_ids.to(device)
        else:
            ids = gen_tok(
                text, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(device)
        return embed_fn(ids).to(dtype)

    _prefix_emb = _e(
        f"<|im_start|>system\n{_QWEN_SYS}<|im_end|>\n<|im_start|>user\n",
        add_bos=True,
    )

    def _build_seq_embeds(proj_tokens: torch.Tensor, question: str) -> torch.Tensor:
        parts = [_prefix_emb]
        for i, lat in enumerate(proj_tokens):
            label = f"{'' if i == 0 else chr(10)}Evidence {i + 1}: "
            parts.append(_e(label, add_bos=False))
            parts.append(lat.reshape(-1, lat.shape[-1]).to(dtype).unsqueeze(0).to(device))
        parts.append(_e(f"\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n", add_bos=False))
        return torch.cat(parts, dim=1).squeeze(0)

    def _batch_generate(projected_list: List[torch.Tensor], questions: List[str]) -> tuple:
        seq_embeds = [_build_seq_embeds(proj, q) for proj, q in zip(projected_list, questions)]
        batch_size = len(seq_embeds)
        max_len = max(e.shape[0] for e in seq_embeds)
        hidden = seq_embeds[0].shape[-1]

        input_embeds = torch.zeros(batch_size, max_len, hidden, dtype=dtype, device=device)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        position_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        prompt_lens = []

        for i, emb in enumerate(seq_embeds):
            cur_len = emb.shape[0]
            input_embeds[i, max_len - cur_len:, :] = emb
            attention_mask[i, max_len - cur_len:] = 1
            position_ids[i, max_len - cur_len:] = torch.arange(cur_len, device=device)
            prompt_lens.append(cur_len)

        out = generator.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=gen_tok.eos_token_id,
        )

        answers, token_counts = [], []
        for i in range(batch_size):
            gen_ids = out[i][max_len:] if out.shape[1] > max_len else out[i]
            answers.append(gen_tok.decode(gen_ids, skip_special_tokens=True).strip())
            token_counts.append(prompt_lens[i])
        return answers, token_counts

    predictions, references, token_counts = [], [], []
    sample_meta = []
    recall_scores = []
    img_preds, img_refs, txt_preds, txt_refs = [], [], [], []

    # ── Image samples ───────────────────────────────────────────────────────
    img_pool = list(image_samples)
    random.Random(42).shuffle(img_pool)
    if max_samples > 0:
        img_pool = img_pool[:max_samples]

    img_valid = []
    for s in tqdm(img_pool, desc="Prep-image", leave=False):
        img_dir = s.get("image_dir", "")
        pos_ids = s.get("pos_image_ids", [])
        question = s.get("question", "")
        answers  = s.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]
        if not pos_ids or not question or not answers:
            continue

        cand_pils, cand_titles = [], []
        for cid, cap in zip(pos_ids + s.get("neg_image_ids", []),
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

        img_valid.append({
            "question": question,
            "answers": answers,
            "webqa_qcate": s.get("webqa_qcate"),
            "webqa_keywords": s.get("webqa_keywords"),
            "cand_pils": cand_pils,
            "cand_titles": cand_titles,
            "n_pos_loaded": sum(
                1 for cid in pos_ids
                if os.path.exists(os.path.join(img_dir, f"{cid}.jpg"))
            ),
        })

    img_projected, img_questions, img_answers = [], [], []
    img_meta_pending = []
    if img_valid:
        img_questions_all = [s["question"] for s in img_valid]
        img_q_lat = compressor.embed_query_batch(img_questions_all, with_grad=False)
        img_q_embs = compressor.get_retrieval_embedding(img_q_lat)

        for s, q_emb in tqdm(list(zip(img_valid, img_q_embs)), desc="Retrieve-image", leave=False):
            cand_latents = compressor.compress_batch(
                s["cand_titles"], images=s["cand_pils"], with_grad=False, adapter="compress"
            )
            cand_embs = compressor.get_retrieval_embedding(cand_latents)

            sims = cand_embs @ q_emb.unsqueeze(-1)
            sims = sims.squeeze(-1)
            k = min(top_k, cand_latents.shape[0])
            top_i = torch.topk(sims, k=k).indices.tolist()

            pos_set = set(range(min(s["n_pos_loaded"], cand_latents.shape[0])))
            if pos_set:
                recall_scores.append(len(set(top_i) & pos_set) / len(pos_set))

            img_projected.append(compressor.project_for_generator(cand_latents[top_i]))
            img_questions.append(s["question"])
            img_answers.append(s["answers"])
            img_meta_pending.append({
                "source": "webqa",
                "modality": "image",
                "webqa_qcate": s.get("webqa_qcate"),
                "webqa_keywords": s.get("webqa_keywords"),
            })

        n_img_batches = (len(img_projected) + gen_batch_size - 1) // gen_batch_size if img_projected else 0
        for b_start in tqdm(range(0, len(img_projected), gen_batch_size),
                            total=n_img_batches, desc="Gen-image", leave=False):
            b_proj = img_projected[b_start: b_start + gen_batch_size]
            b_qs = img_questions[b_start: b_start + gen_batch_size]
            b_refs = img_answers[b_start: b_start + gen_batch_size]
            try:
                b_answers, b_lens = _batch_generate(b_proj, b_qs)
            except Exception as e:
                logger.warning(f"Image batch generation failed: {e}")
                b_answers = [""] * len(b_proj)
                b_lens = [0] * len(b_proj)

            predictions.extend(b_answers)
            references.extend(b_refs)
            token_counts.extend(b_lens)
            sample_meta.extend(img_meta_pending[b_start: b_start + gen_batch_size])
            img_preds.extend(b_answers)
            img_refs.extend(b_refs)

    # ── Text samples ────────────────────────────────────────────────────────
    txt_pool = list(text_samples)
    random.Random(42).shuffle(txt_pool)
    if max_samples > 0:
        txt_pool = txt_pool[:max_samples]

    txt_valid = []
    for s in tqdm(txt_pool, desc="Prep-text", leave=False):
        pos_facts = s.get("txt_pos_facts", [])
        neg_facts = s.get("txt_neg_facts", [])[:top_k * 2]
        question  = s.get("question", "")
        answers   = s.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]
        if not pos_facts or not question or not answers:
            continue

        txt_valid.append({
            "question": question,
            "answers": answers,
            "webqa_qcate": s.get("webqa_qcate"),
            "webqa_keywords": s.get("webqa_keywords"),
            "all_facts": pos_facts + neg_facts,
            "n_pos": len(pos_facts),
        })

    txt_projected, txt_questions, txt_answers = [], [], []
    txt_meta_pending = []
    if txt_valid:
        all_txt_facts = []
        txt_offsets = []
        offset = 0
        for s in txt_valid:
            facts = s["all_facts"]
            all_txt_facts.extend(facts)
            txt_offsets.append((offset, offset + len(facts)))
            offset += len(facts)

        txt_latent_chunks = []
        for start in tqdm(range(0, len(all_txt_facts), compress_batch_size),
                          total=(len(all_txt_facts) + compress_batch_size - 1) // compress_batch_size,
                          desc="Compress-text", leave=False):
            txt_latent_chunks.append(
                compressor.compress_batch(
                    all_txt_facts[start: start + compress_batch_size],
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

        txt_questions_all = [s["question"] for s in txt_valid]
        txt_q_lat = compressor.embed_query_batch(txt_questions_all, with_grad=False)
        txt_q_embs = compressor.get_retrieval_embedding(txt_q_lat)

        for s, (start, end), q_emb in tqdm(list(zip(txt_valid, txt_offsets, txt_q_embs)),
                                           desc="Retrieve-text", leave=False):
            fact_latents = all_txt_latents[start:end]
            fact_embs = all_txt_embs[start:end]
            sims = fact_embs @ q_emb.unsqueeze(-1)
            sims = sims.squeeze(-1)
            k = min(top_k, fact_latents.shape[0])
            top_i = torch.topk(sims, k=k).indices.tolist()

            pos_set = set(range(s["n_pos"]))
            if pos_set:
                recall_scores.append(len(set(top_i) & pos_set) / len(pos_set))

            txt_projected.append(compressor.project_for_generator(fact_latents[top_i]))
            txt_questions.append(s["question"])
            txt_answers.append(s["answers"])
            txt_meta_pending.append({
                "source": "webqa",
                "modality": "text",
                "webqa_qcate": s.get("webqa_qcate"),
                "webqa_keywords": s.get("webqa_keywords"),
            })

        n_txt_batches = (len(txt_projected) + gen_batch_size - 1) // gen_batch_size if txt_projected else 0
        for b_start in tqdm(range(0, len(txt_projected), gen_batch_size),
                            total=n_txt_batches, desc="Gen-text", leave=False):
            b_proj = txt_projected[b_start: b_start + gen_batch_size]
            b_qs = txt_questions[b_start: b_start + gen_batch_size]
            b_refs = txt_answers[b_start: b_start + gen_batch_size]
            try:
                b_answers, b_lens = _batch_generate(b_proj, b_qs)
            except Exception as e:
                logger.warning(f"Text batch generation failed: {e}")
                b_answers = [""] * len(b_proj)
                b_lens = [0] * len(b_proj)

            predictions.extend(b_answers)
            references.extend(b_refs)
            token_counts.extend(b_lens)
            sample_meta.extend(txt_meta_pending[b_start: b_start + gen_batch_size])
            txt_preds.extend(b_answers)
            txt_refs.extend(b_refs)

    compressor.train()

    if not predictions:
        logger.warning("Validation: no predictions.")
        return {"em": 0.0, "f1": 0.0, "rouge_l": 0.0, "recall_at_k": 0.0,
                "avg_tokens": 0.0, "img_em": 0.0, "img_f1": 0.0,
                "txt_em": 0.0, "txt_f1": 0.0}

    metrics     = evaluate(predictions, references, token_counts, sample_metadata=sample_meta)
    img_meta = [m for m in sample_meta if m["modality"] == "image"]
    txt_meta = [m for m in sample_meta if m["modality"] == "text"]
    img_metrics = evaluate(img_preds, img_refs, [0]*len(img_preds), sample_metadata=img_meta) if img_preds else {"em": 0.0, "f1": 0.0}
    txt_metrics = evaluate(txt_preds, txt_refs, [0]*len(txt_preds), sample_metadata=txt_meta) if txt_preds else {"em": 0.0, "f1": 0.0}
    metrics["recall_at_k"] = float(sum(recall_scores)/len(recall_scores)) if recall_scores else 0.0
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
        f"Val | n={len(predictions)} (img={len(img_preds)}, txt={len(txt_preds)}) | "
        f"EM={metrics['em']:.4f} F1={metrics['f1']:.4f} |{acc_part} "
        f"img EM={metrics['img_em']:.4f} F1={metrics['img_f1']:.4f} |{img_acc_part} "
        f"txt EM={metrics['txt_em']:.4f} F1={metrics['txt_f1']:.4f} |{txt_acc_part} "
        f"Recall@{top_k}={metrics['recall_at_k']:.4f} | "
        f"avg_tokens={metrics.get('avg_tokens', 0):.1f}"
    )
    return metrics


@torch.no_grad()
def run_validation_hotpot(
        compressor: QwenVLCompressor,
        generator,
        gen_processor,
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
    from src.evaluation import evaluate

    compressor.eval()
    embed_fn = generator.get_input_embeddings()
    dtype = next(generator.parameters()).dtype
    gen_tok = gen_processor.tokenizer

    def _e(text: str, max_len: int = 512):
        return embed_fn(_tok(gen_tok, text, device, max_len)).to(dtype)

    def _build_seq_embeds(proj_tokens: torch.Tensor, question: str) -> torch.Tensor:
        prefix_str, suffix_str = _split_chat_template(gen_tok, question)
        parts = [_e(prefix_str)]
        for i, lat in enumerate(proj_tokens):
            label = f"Latent context {i + 1}: " if i == 0 else f"\nLatent context {i + 1}: "
            parts.append(_e(label))
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
            pad_token_id=gen_tok.eos_token_id,
        )

        answers = []
        for i in range(batch_size):
            gen_ids = out[i, max_len:] if out.shape[1] > max_len else out[i]
            answers.append(gen_tok.decode(gen_ids, skip_special_tokens=True).strip())
        return answers, input_lens

    predictions, references, token_counts = [], [], []
    sample_meta = []
    recall_scores = []
    precision_scores = []
    img_recalls, txt_recalls = [], []
    img_precisions, txt_precisions = [], []
    img_tokens, txt_tokens = [], []
    img_preds, img_refs, txt_preds, txt_refs = [], [], [], []
    val_pool = list(image_samples) + list(text_samples)
    random.Random(42).shuffle(val_pool)
    if max_samples > 0:
        val_pool = val_pool[:max_samples]

    valid = []
    all_txt_facts = []
    all_img_titles = []
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
        all_txt_facts.extend(txt_facts)
        txt_offsets.append((txt_start, txt_end))
        txt_offset = txt_end

        img_start = img_offset
        img_end = img_offset + len(img_pils)
        all_img_pils.extend(pil for pil, _ in img_pils)
        all_img_titles.extend(title for _, title in img_pils)
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
        return {"em": 0.0, "f1": 0.0, "rouge_l": 0.0, "recall_at_k": 0.0,
                "precision_at_k": 0.0, "avg_tokens": 0.0, "img_em": 0.0, "img_f1": 0.0,
                "txt_em": 0.0, "txt_f1": 0.0, "img_recall_at_k": 0.0,
                "txt_recall_at_k": 0.0, "img_precision_at_k": 0.0,
                "txt_precision_at_k": 0.0, "img_avg_tokens": 0.0, "txt_avg_tokens": 0.0}

    if all_txt_facts:
        txt_latent_chunks = []
        for start in tqdm(range(0, len(all_txt_facts), compress_batch_size),
                          total=(len(all_txt_facts) + compress_batch_size - 1) // compress_batch_size,
                          desc="Compress-text", leave=False):
            txt_latent_chunks.append(
                compressor.compress_batch(
                    all_txt_facts[start: start + compress_batch_size],
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

    if all_img_titles:
        img_latent_chunks = []
        for start in tqdm(range(0, len(all_img_titles), compress_batch_size),
                          total=(len(all_img_titles) + compress_batch_size - 1) // compress_batch_size,
                          desc="Compress-image", leave=False):
            img_latent_chunks.append(
                compressor.compress_batch(
                    all_img_titles[start: start + compress_batch_size],
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
                        total=n_batches, desc="Gen-val", leave=False):
        b_proj = projected_list[b_start: b_start + gen_batch_size]
        b_qs = questions_list[b_start: b_start + gen_batch_size]
        b_refs = refs_list[b_start: b_start + gen_batch_size]
        b_mods = mods_list[b_start: b_start + gen_batch_size]
        try:
            b_answers, b_lens = _batch_generate(b_proj, b_qs)
        except Exception as e:
            logger.warning(f"Validation batch generation failed: {e}")
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

    compressor.train()

    if not predictions:
        logger.warning("Validation: no predictions.")
        return {"em": 0.0, "f1": 0.0, "rouge_l": 0.0, "recall_at_k": 0.0,
                "precision_at_k": 0.0, "avg_tokens": 0.0, "img_em": 0.0, "img_f1": 0.0,
                "txt_em": 0.0, "txt_f1": 0.0, "img_recall_at_k": 0.0,
                "txt_recall_at_k": 0.0, "img_precision_at_k": 0.0,
                "txt_precision_at_k": 0.0, "img_avg_tokens": 0.0, "txt_avg_tokens": 0.0}

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
        f"Val | n={len(predictions)} (img={len(img_preds)}, txt={len(txt_preds)}) | "
        f"EM={metrics['em']:.4f} F1={metrics['f1']:.4f} |{acc_part} "
        f"img EM={metrics['img_em']:.4f} F1={metrics['img_f1']:.4f} |{img_acc_part} "
        f"txt EM={metrics['txt_em']:.4f} F1={metrics['txt_f1']:.4f} |{txt_acc_part} "
        f"Recall@{top_k}={metrics['recall_at_k']:.4f} "
        f"Precision@{top_k}={metrics['precision_at_k']:.4f} | "
        f"img R@{top_k}={metrics['img_recall_at_k']:.4f} P@{top_k}={metrics['img_precision_at_k']:.4f} "
        f"txt R@{top_k}={metrics['txt_recall_at_k']:.4f} P@{top_k}={metrics['txt_precision_at_k']:.4f} | "
        f"avg_tokens={metrics.get('avg_tokens', 0):.1f}"
    )
    return metrics


def dump_train_debug_examples(
        samples: List[Dict],
        output_path: str,
        scale: float,
        batch_size: int,
        max_hard_negatives: int,
        max_neg_images: int,
        generator_name: str,
        seed: int = 0,
):
    rng = random.Random(seed)
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(generator_name)
    batch = build_multimodal_batch(
        samples,
        batch_size=batch_size,
        max_hard_negatives=max_hard_negatives,
        max_neg_images=max_neg_images,
        scale=scale,
        rng=rng,
    )

    dumped = []
    for item in batch:
        question = item["question"]
        pos_sentences = list(item.get("pos_sentences", []))
        neg_sentences = list(item.get("neg_sentences", []))
        pos_regions = list(item.get("pos_regions", []))
        neg_regions = list(item.get("neg_regions", []))

        teacher_prompt = None
        if pos_regions:
            content = []
            for i, region in enumerate(pos_regions):
                content.append({"type": "text", "text": f"Context {i + 1}:\n"})
                content.append({"type": "image"})
                content.append({"type": "text", "text": f"Title: {region[2]}\n"})
            content.append({"type": "text", "text": f"Question: {question}"})
            teacher_messages = [
                _vl_system_message(proc),
                {"role": "user", "content": content},
            ]
            teacher_prompt = proc.apply_chat_template(
                teacher_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        latent_prompt = {
            "system": _QWEN_SYS,
            "user_prefix": "Latent context 1...n",
            "question": question,
        }

        dumped.append({
            "question": question,
            "answers": item.get("answers", []),
            "positive_modality": item.get("positive_modality"),
            "n_pos_text": len(pos_sentences),
            "n_neg_text": len(neg_sentences),
            "n_pos_images": len(pos_regions),
            "n_neg_images": len(neg_regions),
            "pos_texts": pos_sentences,
            "neg_texts": neg_sentences,
            "pos_image_titles": [r[2] for r in pos_regions],
            "neg_image_titles": [r[2] for r in neg_regions],
            "image_teacher_messages": teacher_prompt,
            "latent_student_template": latent_prompt,
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "batch_size": batch_size,
                "n_examples": len(dumped),
                "examples": dumped,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"Saved train debug dump: {output_path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = cfg["model"]
    comp_cfg  = cfg["compressor"]
    dec_cfg   = cfg.get("decoder", {})
    train_cfg = cfg["training"]
    img_cfg   = cfg.get("image", {})

    compressor_name = str(model_cfg["compressor_name"]).lower()
    generator_name = str(model_cfg.get("generator_name", "")).lower()
    if "llava" in compressor_name or "llava" in generator_name:
        raise ValueError(
            "This config looks like an LLaVA setup, but you launched scripts/train_gemma.py. "
            "Use scripts/train_llava.py with config_llava.yaml instead."
        )

    # ---- compressor ----
    compressor = QwenVLCompressor(
        model_name=model_cfg["compressor_name"],
        decoder_name=model_cfg.get("decoder_name", "meta-llama/Llama-3.2-1B-Instruct"),
        clip_model_name=model_cfg.get("clip_model_name", "openai/clip-vit-large-patch14-336"),
        lora_r=comp_cfg["lora_r"],
        lora_alpha=comp_cfg["lora_alpha"],
        lora_dropout=comp_cfg.get("lora_dropout", 0.05),
        target_modules=comp_cfg.get("target_modules"),
        decode_lora_r=dec_cfg.get("lora_r", comp_cfg["lora_r"]),
        decode_lora_alpha=dec_cfg.get("lora_alpha", comp_cfg["lora_alpha"]),
        decode_lora_dropout=dec_cfg.get("lora_dropout", comp_cfg.get("lora_dropout", 0.05)),
        decode_target_modules=dec_cfg.get("target_modules", comp_cfg.get("target_modules")),
        retrieval_dim=comp_cfg["retrieval_dim"],
        generator_hidden=model_cfg.get("generator_hidden", 3584),
        num_latent_tokens=comp_cfg.get("num_latent_tokens", 1),
        load_clip_model=train_cfg.get("lambda_embed_recon", 0.0) > 0,
    ).to(device)
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        compressor.load(args.checkpoint)

    compressor.enable_gradient_checkpointing()

    # ---- generator (Qwen2.5-VL-7B, frozen) ----
    generator, gen_processor = None, None
    lam_distill     = train_cfg.get("lambda_distill",       0.0)
    lam_img_distill = train_cfg.get("lambda_image_distill", 0.0)
    if lam_distill > 0 or lam_img_distill > 0 or args.validate_only:
        from transformers import AutoProcessor
        gen_name = model_cfg["generator_name"]
        gen_model_cls = resolve_qwen_vl_model_class(gen_name)
        logger.info(f"Loading frozen VL generator with {gen_model_cls.__name__}: {gen_name}")
        gen_processor = AutoProcessor.from_pretrained(gen_name, trust_remote_code=True)
        generator = gen_model_cls.from_pretrained(
            gen_name, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
        )
        generator.eval()
        for p in generator.parameters():
            p.requires_grad_(False)

    retriever = ContrastiveRetriever(
        temperature=train_cfg.get("temperature", 0.07)
    ).to(device)

    # ---- data ----
    data_cfg   = cfg.get("data", {})
    train_path = args.data or data_cfg.get("train")
    val_path   = args.val  or data_cfg.get("val")

    if not train_path:
        raise ValueError("No training data path — set data.train in config or pass --data")

    logger.info(f"Loading WebQA data: {train_path}")
    all_samples = load_webqa_samples(train_path)
    for s in all_samples:
        if "modality" not in s:
            s["modality"] = "image" if s.get("pos_image_ids") else "text"
    image_samples = [s for s in all_samples if s.get("modality") == "image"]
    text_samples  = [s for s in all_samples if s.get("modality") == "text"]
    train_samples = list(all_samples)
    logger.info(f"  image: {len(image_samples)}  text: {len(text_samples)}")

    val_image, val_text = [], []
    if val_path and os.path.exists(val_path):
        val_all = load_webqa_samples(val_path)
        for s in val_all:
            if "modality" not in s:
                s["modality"] = "image" if s.get("pos_image_ids") else "text"
        val_image = [s for s in val_all if s.get("modality") == "image"]
        val_text  = [s for s in val_all if s.get("modality") == "text"]
    else:
        rng_s = random.Random(0)

        def _split(lst, frac=0.1):
            idx = list(range(len(lst)))
            rng_s.shuffle(idx)
            n = max(1, int(len(lst) * frac))
            return [lst[i] for i in idx[n:]], [lst[i] for i in idx[:n]]

        image_samples, val_image = _split(image_samples)
        text_samples,  val_text  = _split(text_samples)
        train_samples = image_samples + text_samples

    if args.validate_only:
        if generator is None or gen_processor is None:
            raise ValueError("Validation requires generator_name to be loadable.")
        metrics = run_validation_hotpot(
            compressor, generator, gen_processor,
            val_image, val_text, img_cfg.get("scale", 1 / 10), device,
            top_k=cfg.get("retrieval", {}).get("top_k", 5),
            max_new_tokens=cfg.get("generation", {}).get("max_new_tokens", 64),
            max_samples=train_cfg.get("val_samples", 200),
            gen_batch_size=train_cfg.get("val_gen_batch_size", 8),
            compress_batch_size=train_cfg.get("val_compress_batch_size", 32),
        )
        logger.info("Validation-only metrics:\n%s", json.dumps(metrics, indent=2, sort_keys=True))
        return

    if args.dump_train_debug:
        dump_train_debug_examples(
            train_samples,
            output_path=args.dump_train_debug,
            scale=img_cfg.get("scale", 1 / 10),
            batch_size=train_cfg.get("batch_size", 4),
            max_hard_negatives=train_cfg.get("max_hard_negatives", 8),
            max_neg_images=train_cfg.get("max_neg_images", 8),
            generator_name=model_cfg["generator_name"],
            seed=0,
        )
        return

    # ---- output paths: CLI arg > config paths.checkpoint > config training.save_dir ----
    paths_cfg = cfg.get("paths", {})
    save_dir = train_cfg.get("save_dir") or paths_cfg.get("results_dir", "checkpoints/")
    out_path = args.output or paths_cfg.get("checkpoint") or os.path.join(save_dir, "qwen_model.pt")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    best_path = out_path.replace(".pt", "_best.pt")

    # ---- optimiser ----
    trainable = [p for p in compressor.parameters() if p.requires_grad]
    trainable += list(retriever.parameters())
    optimizer = AdamW(trainable, lr=train_cfg["learning_rate"], weight_decay=0.01)
    mixed_bs = max(train_cfg.get("batch_size", 1), train_cfg.get("image_batch_size", 1))
    total_steps = (
        train_cfg["num_epochs"]
        * max(len(train_samples), 1)
        // max(mixed_bs * train_cfg["gradient_accumulation_steps"], 1)
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    writer = None
    if _TB:
        log_dir = args.logdir or os.path.join(os.path.dirname(os.path.abspath(out_path)), "curveqwen")
        writer  = SummaryWriter(log_dir=log_dir)

    # ---- training ----
    scale         = img_cfg.get("scale", 1 / 10)
    lam_recon     = train_cfg.get("lambda_recon",         0.5)
    lam_embed_recon = train_cfg.get("lambda_embed_recon", 0.0)
    lam_cont      = train_cfg.get("lambda_contrast",      0.2)
    lam_img_cont  = train_cfg.get("lambda_image_contrast",0.2)
    n_distill     = train_cfg.get("n_distill_tokens",     16)
    log_every     = train_cfg.get("log_every",            10)
    val_every     = train_cfg.get("val_every",            500)
    val_samples   = train_cfg.get("val_samples",          200)
    val_gen_bs    = train_cfg.get("val_gen_batch_size",   8)
    val_comp_bs   = train_cfg.get("val_compress_batch_size", 32)
    grad_accum    = train_cfg.get("gradient_accumulation_steps", 4)
    bs_text       = train_cfg.get("batch_size",           4)
    bs_image      = train_cfg.get("image_batch_size",     4)
    max_neg_img   = train_cfg.get("max_neg_images",       8)
    max_hard_neg  = train_cfg.get("max_hard_negatives",   8)
    recon_neg     = train_cfg.get("recon_negatives",      False)
    recon_query   = train_cfg.get("recon_query",          False)
    recon_title   = train_cfg.get("recon_title",          True)
    student_distill_hard_negatives_max = train_cfg.get("student_distill_hard_negatives_max", 0)

    rng       = random.Random(42)
    global_step = 0
    best_acc  = -1.0
    running_mm = {
        k: 0.0 for k in [
            "total", "contrast", "txt_recon_sent", "txt_recon_query", "txt_distill",
            "img_recon_title", "img_recon_query", "img_recon_clip", "img_distill",
        ]
    }
    running_count = 0
    running_total = 0.0

    def _validate(step: int):
        nonlocal best_acc
        if generator is None:
            return
        metrics = run_validation_hotpot(
            compressor, generator, gen_processor,
            val_image, val_text, scale, device,
            top_k=cfg.get("retrieval", {}).get("top_k", 5),
            max_new_tokens=cfg.get("generation", {}).get("max_new_tokens", 64),
            max_samples=val_samples,
            gen_batch_size=val_gen_bs,
            compress_batch_size=val_comp_bs,
        )
        if writer:
            for k, v in metrics.items():
                writer.add_scalar(f"val/{k}", v, step)
            writer.flush()
        compressor.save(out_path)
        score = metrics.get("acc", metrics.get("f1", 0.0))
        if score > best_acc:
            best_acc = score
            compressor.save(best_path)
            score_name = "Acc" if "acc" in metrics else "F1"
            logger.info(f"New best {score_name}: {best_acc:.4f} -> {best_path}")

    if (val_image or val_text) and generator is not None:
        _validate(0)

    for epoch in range(train_cfg["num_epochs"]):
        logger.info(f"Epoch {epoch + 1}/{train_cfg['num_epochs']}")
        compressor.train()
        optimizer.zero_grad()

        n_steps = max(len(train_samples) // max(mixed_bs, 1), 1)

        for step in tqdm(range(n_steps), desc=f"Epoch {epoch+1}"):
            mm_batch = build_multimodal_batch(
                train_samples, mixed_bs, max_hard_neg, max_neg_img, scale, rng
            )
            mm_out = multimodal_train_step(
                compressor, retriever, mm_batch, device=device,
                lam_recon=lam_recon,
                lam_embed_recon=lam_embed_recon,
                lam_cont=lam_cont,
                lam_distill=lam_distill,
                lam_img_distill=lam_img_distill,
                generator=generator,
                gen_processor=gen_processor,
                n_distill_tokens=n_distill,
                recon_negatives=recon_neg,
                recon_query=recon_query,
                recon_title=recon_title,
                student_distill_hard_negatives_max=student_distill_hard_negatives_max,
                rng=rng,
            )
            (mm_out["total"] / grad_accum).backward()

            running_total += mm_out["total"].item()
            for key in running_mm:
                running_mm[key] += mm_out[key] if key != "total" else mm_out["total"].item()
            running_count += 1

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % log_every == 0:
                    lr = scheduler.get_last_lr()[0]
                    n = max(running_count, 1)
                    avg_m = {k: v / n for k, v in running_mm.items()}
                    avg_total = running_total / max(log_every, 1)
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

                if global_step % val_every == 0 and generator is not None:
                    _validate(global_step)

        # End-of-epoch save
        epoch_path = out_path.replace(".pt", f"_epoch{epoch+1}.pt")
        compressor.save(epoch_path)
        logger.info(f"Epoch {epoch+1} done → {epoch_path}")

    if val_image or val_text:
        _validate(global_step)
    compressor.save(out_path)
    logger.info(f"Training complete. Saved: {out_path}")
    if writer:
        writer.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Qwen2.5-VL Latent Retrieval")
    parser.add_argument("--config",  required=True, help="Path to config_gemma3_4B_12B.yaml")
    parser.add_argument("--data",    default=None,  help="Override data.train path")
    parser.add_argument("--val",     default=None,  help="Override data.val path")
    parser.add_argument("--checkpoint", default=None, help="Load a compressor checkpoint before training/validation")
    parser.add_argument("--output",  default=None,  help="Override checkpoint output path")
    parser.add_argument("--logdir",  default=None,  help="TensorBoard log directory")
    parser.add_argument("--validate_only", action="store_true", help="Run validation once and exit")
    parser.add_argument("--dump_train_debug", default=None, help="Write one sampled training batch structure/prompts to JSON and exit")
    args = parser.parse_args()
    train(args)

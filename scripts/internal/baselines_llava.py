"""
WebQA LLaVA Baselines
=====================
Retrieval baselines evaluated against a frozen LLaVA-13B generator on WebQA.
Handles both image-modality and text-modality samples.

Baselines
---------
1. BM25Retrieval    : BM25 over captions (image) / text facts (text) → LLaVA-13B
2. DenseRetrieval   : sentence-transformers over captions / text facts → LLaVA-13B
3. VisRAGRetrieval  : CLIP visual embedding for image retrieval (arXiv 2410.10594),
                      dense for text → LLaVA-13B
4. LatentRetrieval  : LLaVACompressor (compress adapter) → single latent token per
                      image/fact, cosine retrieval, cross_proj into LLaVA-13B
                      (requires --checkpoint)

Unified samples : candidate pool = txt_pos_facts/txt_neg_facts plus
                  pos_image_ids/neg_image_ids when present

Usage
-----
python scripts/baselines_llava.py \\
    --config  config_llava.yaml \\
    --val     data/webqa_val.json \\
    --methods bm25 dense visrag latent \\
    --checkpoint checkpoints/llava_model_best.pt \\
    --top_k   5 \\
    --max_samples 200 \\
    --output  results/baselines_llava.json
"""

import argparse
import gc
import json
import logging
import os
import random
import sys

_RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _RELEASE_ROOT not in sys.path:
    sys.path.insert(0, _RELEASE_ROOT)
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from data.prepare_webqa import load_webqa_samples
from src.distillation import _split_chat_template, _tok
from src.evaluation import evaluate
from src.region_encoder import resize_image
from scripts.internal.baselines import _count_tokens_from_marker, _count_tokens_between_markers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _disable_tqdm_progress() -> None:
    global tqdm
    _orig_tqdm = tqdm

    def _quiet_tqdm(*args, **kwargs):
        kwargs["disable"] = True
        return _orig_tqdm(*args, **kwargs)

    tqdm = _quiet_tqdm


def _simple_tqdm_progress() -> None:
    global tqdm
    _orig_tqdm = tqdm
    visible = {"prep-unified", "unified-baselines"}

    def _simple_tqdm(*args, **kwargs):
        if kwargs.get("desc") not in visible:
            kwargs["disable"] = True
        return _orig_tqdm(*args, **kwargs)

    tqdm = _simple_tqdm

_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the question concisely (a few words or a short phrase) "
    "based on the provided context."
)


# ---------------------------------------------------------------------------
# Generator — frozen LLaVA-13B
# ---------------------------------------------------------------------------

class LLaVAGenerator:
    """
    Frozen LLaVA-v1.5-13B for answer generation.

    Supports two input modes:
      - with_images : prompt contains <image> slots; actual PIL images passed
      - text_only   : no images; evidence is injected as Evidence N: labels
    """

    def __init__(self, model_name: str, device: str = "cuda", max_new_tokens: int = 64):
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        logger.info(f"Loading LLaVA-13B generator: {model_name}")
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_type = getattr(cfg, "model_type", None)
        if model_type != "llava":
            raise ValueError(
                f"scripts/baselines_llava.py requires a LLaVA generator, but got "
                f"model_type={model_type} from {model_name}. "
                f"Use config_llava.yaml / a llava-1.5-13b-hf path, not Qwen-VL."
            )
        self.processor = AutoProcessor.from_pretrained(model_name)
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            self.processor.tokenizer.padding_side = "left"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.max_new_tokens = max_new_tokens

    def to(self, device: str):
        self.model.to(device)
        self.device = device
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    def _get_vision_config(self):
        if hasattr(self.model, "vision_tower"):
            return self.model.vision_tower.config
        if hasattr(self.model, "model") and hasattr(self.model.model, "vision_tower"):
            return self.model.model.vision_tower.config
        vision_cfg = getattr(self.model.config, "vision_config", None)
        if vision_cfg is not None:
            return vision_cfg
        raise AttributeError(
            "Cannot locate LLaVA vision config. "
            f"Top-level attrs: {[n for n, _ in self.model.named_children()]}"
        )

    def _effective_image_input_tokens(self, enc, n_imgs_used: int) -> int:
        n_in = enc["input_ids"].shape[-1]
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        n_image_token_ids = int((enc["input_ids"] == image_token_id).sum().item())
        vt_cfg = self._get_vision_config()
        n_patches_per_image = (vt_cfg.image_size // vt_cfg.patch_size) ** 2
        if n_image_token_ids > n_imgs_used:
            return n_in
        return n_in - n_imgs_used + n_imgs_used * n_patches_per_image

    def _image_expansion_extra_tokens(self, enc, n_imgs_used: int) -> int:
        """Return visual-token cost beyond raw ``<image>`` placeholders."""
        if n_imgs_used <= 0:
            return 0
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        n_image_token_ids = int((enc["input_ids"] == image_token_id).sum().item())
        vt_cfg = self._get_vision_config()
        n_patches_per_image = (vt_cfg.image_size // vt_cfg.patch_size) ** 2
        if n_image_token_ids > n_imgs_used:
            return max(0, n_image_token_ids - n_imgs_used)
        return max(0, n_imgs_used * n_patches_per_image - n_imgs_used)

    def _encode_evidence_prompt(self, prompt: str, images: List[Image.Image]):
        if images:
            enc = self.processor(
                text=prompt,
                images=images,
                return_tensors="pt",
            ).to(self.device)
            return enc, self._effective_image_input_tokens(enc, len(images))

        ids = self.processor.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )["input_ids"].to(self.device)
        return {"input_ids": ids}, ids.shape[-1]

    def _prompt_token_stats(self, prompt: str, image_extra_tokens: int = 0, latent: bool = False) -> Tuple[int, int]:
        if latent:
            no_system = _count_tokens_from_marker(self.processor.tokenizer, prompt, ["Latent context 1:"])
            context_only = _count_tokens_between_markers(
                self.processor.tokenizer, prompt, ["Latent context 1:"], ["Question:"]
            )
        else:
            no_system = _count_tokens_from_marker(self.processor.tokenizer, prompt, ["Evidence 1:"])
            context_only = _count_tokens_between_markers(
                self.processor.tokenizer, prompt, ["Evidence 1:"], ["Question:"]
            )
        return int(no_system + image_extra_tokens), int(context_only + image_extra_tokens)

    def build_evidence_prompt_text(self, question: str, evidences: List[Dict]) -> str:
        lines = []
        for i, ev in enumerate(evidences):
            if ev["kind"] == "image":
                lines.append(f"Evidence {i + 1}: <image>\nTitle: {ev['title']}")
            else:
                lines.append(f"Evidence {i + 1}: {ev['text']}")
        return (
            f"SYSTEM: {_SYSTEM_PROMPT}\n"
            + "USER: "
            + "\n".join(lines)
            + f"\nQuestion: {question}\nASSISTANT:"
        )

    def build_latent_prompt_text(self, question: str, proj_tokens: torch.Tensor) -> str:
        parts = []
        prefix_str, suffix_str = _split_chat_template(self.processor.tokenizer, question)
        parts.append(prefix_str)
        for i in range(proj_tokens.shape[0]):
            label = f"Latent context {i + 1}: [LATENT]"
            parts.append(label if i == 0 else f"\n{label}")
        parts.append(suffix_str)
        return "".join(parts)

    @torch.no_grad()
    def generate_with_images(
        self,
        question: str,
        images: List[Image.Image],
        titles: List[str],
    ) -> Tuple[str, int, int, int]:
        """
        Feed top-k retrieved images directly to LLaVA-13B.

        Prompt structure (LLaVA-1.5 convention — <image> must come first):
          USER: Evidence 1: <image>\nTitle: {t1}
                Evidence 2: <image>\nTitle: {t2}
                ...
                Question: {q}
          ASSISTANT:
        """
        if not images:
            return "", 0, 0, 0
        # Build prompt: one <image> token per retrieved image
        lines = []
        for i, title in enumerate(titles):
            lines.append(f"Evidence {i + 1}: <image>\nTitle: {title}")
        prompt = (
            f"SYSTEM: {_SYSTEM_PROMPT}\n"
            + "USER: "
            + "\n".join(lines)
            + f"\nQuestion: {question}\nASSISTANT:"
        )

        n_imgs_used = len(images)
        try:
            enc = self.processor(
                text=prompt,
                images=images,
                return_tensors="pt",
            ).to(self.device)
        except Exception as e:
            logger.warning(f"Processor failed with {len(images)} images: {e}; using first image only.")
            prompt = (
                f"SYSTEM: {_SYSTEM_PROMPT}\n"
                f"USER: Evidence 1: <image>\nTitle: {titles[0]}\nQuestion: {question}\nASSISTANT:"
            )
            enc = self.processor(
                text=prompt,
                images=[images[0]],
                return_tensors="pt",
            ).to(self.device)
            n_imgs_used = 1

        n_in = enc["input_ids"].shape[-1]
        n_effective = self._effective_image_input_tokens(enc, n_imgs_used)
        image_extra_tokens = self._image_expansion_extra_tokens(enc, n_imgs_used)
        n_no_system, n_context_only = self._prompt_token_stats(
            prompt, image_extra_tokens=image_extra_tokens, latent=False
        )

        out  = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        return self.processor.tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip(), n_effective, n_no_system, n_context_only

    @torch.no_grad()
    def generate_with_evidence(
        self,
        question: str,
        evidences: List[Dict],
    ) -> Tuple[str, int, int, int]:
        if not evidences:
            return "", 0, 0, 0
        prompt = self.build_evidence_prompt_text(question, evidences)
        images = [ev["image"] for ev in evidences if ev["kind"] == "image"]
        enc, n_effective = self._encode_evidence_prompt(prompt, images)
        n_in = enc["input_ids"].shape[-1]
        image_extra_tokens = self._image_expansion_extra_tokens(enc, len(images))
        n_no_system, n_context_only = self._prompt_token_stats(
            prompt, image_extra_tokens=image_extra_tokens, latent=False
        )
        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        return self.processor.tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip(), n_effective, n_no_system, n_context_only

    @torch.no_grad()
    def count_with_evidence(
        self,
        question: str,
        evidences: List[Dict],
    ) -> Tuple[int, int, int]:
        if not evidences:
            return 0, 0, 0
        prompt = self.build_evidence_prompt_text(question, evidences)
        images = [ev["image"] for ev in evidences if ev["kind"] == "image"]
        enc, n_effective = self._encode_evidence_prompt(prompt, images)
        image_extra_tokens = self._image_expansion_extra_tokens(enc, len(images))
        n_no_system, n_context_only = self._prompt_token_stats(
            prompt, image_extra_tokens=image_extra_tokens, latent=False
        )
        return int(n_effective), int(n_no_system), int(n_context_only)

    @torch.no_grad()
    def generate_with_latents(
        self,
        question: str,
        proj_tokens: torch.Tensor,  # (k, gen_hidden) — output of project_for_generator
    ) -> Tuple[str, int, int, int]:
        """
        Inject projected latent tokens into LLaVA-13B via input_embeds.

        Prompt structure (matches training validation):
          BOS USER: Latent context 1: [lat_1] Latent context 2: [lat_2] … Question: {q} ASSISTANT:
        """
        embed_fn = self.model.get_input_embeddings()
        dtype    = next(self.model.parameters()).dtype

        def _e(text: str, max_len: int = 512):
            return embed_fn(_tok(self.processor.tokenizer, text, self.device, max_len)).to(dtype)

        prefix_str, suffix_str = _split_chat_template(self.processor.tokenizer, question)
        parts = [_e(prefix_str)]
        middle_len = 0
        for i, lat in enumerate(proj_tokens):
            label = f"Latent context {i + 1}: " if i == 0 else f"\nLatent context {i + 1}: "
            label_emb = _e(label)
            lat_emb = lat.reshape(-1, lat.shape[-1]).to(dtype).unsqueeze(0).to(self.device)
            middle_len += int(label_emb.shape[1] + lat_emb.shape[1])
            parts.append(label_emb)
            parts.append(lat_emb)
        suffix = _e(suffix_str)
        parts.append(suffix)
        input_embs = torch.cat(parts, dim=1)
        n_tok = input_embs.shape[1]
        n_no_system = n_tok - int(parts[0].shape[1])
        n_context_only = middle_len

        out = self.model.generate(
            inputs_embeds=input_embs,
            attention_mask=torch.ones(1, n_tok, dtype=torch.long, device=self.device),
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        # generate() with inputs_embeds returns only the newly generated token IDs
        # (not the input embeddings), so out[0] is always the generated sequence.
        return self.processor.tokenizer.decode(out[0], skip_special_tokens=True).strip(), n_tok, n_no_system, n_context_only

    @torch.no_grad()
    def count_with_latents(
        self,
        question: str,
        proj_tokens: torch.Tensor,
    ) -> Tuple[int, int, int]:
        embed_fn = self.model.get_input_embeddings()
        dtype = next(self.model.parameters()).dtype

        def _e(text: str, max_len: int = 512):
            return embed_fn(_tok(self.processor.tokenizer, text, self.device, max_len)).to(dtype)

        prefix_str, suffix_str = _split_chat_template(self.processor.tokenizer, question)
        parts = [_e(prefix_str)]
        middle_len = 0
        for i, lat in enumerate(proj_tokens):
            label = f"Latent context {i + 1}: " if i == 0 else f"\nLatent context {i + 1}: "
            label_emb = _e(label)
            lat_emb = lat.reshape(-1, lat.shape[-1]).to(dtype).unsqueeze(0).to(self.device)
            middle_len += int(label_emb.shape[1] + lat_emb.shape[1])
            parts.append(label_emb)
            parts.append(lat_emb)
        suffix = _e(suffix_str)
        parts.append(suffix)
        input_embs = torch.cat(parts, dim=1)
        n_tok = int(input_embs.shape[1])
        n_no_system = n_tok - int(parts[0].shape[1])
        n_context_only = middle_len
        return n_tok, n_no_system, n_context_only

    @torch.no_grad()
    def generate_with_text(
        self,
        question: str,
        facts: List[str],
    ) -> Tuple[str, int, int, int]:
        """
        Feed top-k retrieved text facts to LLaVA-13B (no image).

        Prompt structure:
          USER: Evidence 1: {fact1}
                Evidence 2: {fact2}
                ...
                Question: {q}
          ASSISTANT:
        """
        if not facts:
            return "", 0, 0, 0
        lines = [f"Evidence {i + 1}: {f}" for i, f in enumerate(facts)]
        prompt = (
            f"SYSTEM: {_SYSTEM_PROMPT}\n"
            + "USER: "
            + "\n".join(lines)
            + f"\nQuestion: {question}\nASSISTANT:"
        )

        ids = self.processor.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )["input_ids"].to(self.device)
        n_in = ids.shape[-1]
        n_no_system, n_context_only = self._prompt_token_stats(prompt, image_extra_tokens=0, latent=False)
        out  = self.model.generate(
            input_ids=ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        return self.processor.tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip(), n_in, n_no_system, n_context_only

    @torch.no_grad()
    def batch_generate_with_latents(
        self,
        questions: List[str],
        projected_list: List[torch.Tensor],
    ) -> Tuple[List[str], List[int], List[int], List[int]]:
        if not projected_list:
            return [], [], [], []

        embed_fn = self.model.get_input_embeddings()
        dtype = next(self.model.parameters()).dtype

        seq_embeds = []
        no_system_lens = []
        context_only_lens = []
        for question, proj_tokens in zip(questions, projected_list):
            def _e(text: str, max_len: int = 512):
                return embed_fn(_tok(self.processor.tokenizer, text, self.device, max_len)).to(dtype)

            prefix_str, suffix_str = _split_chat_template(self.processor.tokenizer, question)
            parts = [_e(prefix_str)]
            middle_len = 0
            for i, lat in enumerate(proj_tokens):
                label = f"Latent context {i + 1}: " if i == 0 else f"\nLatent context {i + 1}: "
                label_emb = _e(label)
                lat_emb = lat.reshape(-1, lat.shape[-1]).to(dtype).unsqueeze(0).to(self.device)
                middle_len += int(label_emb.shape[1] + lat_emb.shape[1])
                parts.append(label_emb)
                parts.append(lat_emb)
            suffix = _e(suffix_str)
            parts.append(suffix)
            input_embs = torch.cat(parts, dim=1).squeeze(0)
            seq_embeds.append(input_embs)
            no_system_lens.append(int(input_embs.shape[0] - parts[0].shape[1]))
            context_only_lens.append(middle_len)

        batch_size = len(seq_embeds)
        max_len = max(e.shape[0] for e in seq_embeds)
        hidden = seq_embeds[0].shape[-1]
        padded = torch.zeros(batch_size, max_len, hidden, dtype=seq_embeds[0].dtype, device=self.device)
        attn_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        position_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        input_lens = []
        for i, emb in enumerate(seq_embeds):
            seq_len = emb.shape[0]
            padded[i, max_len - seq_len:] = emb
            attn_mask[i, max_len - seq_len:] = 1
            position_ids[i, max_len - seq_len:] = torch.arange(seq_len, device=self.device)
            input_lens.append(seq_len)

        out = self.model.generate(
            inputs_embeds=padded,
            attention_mask=attn_mask,
            position_ids=position_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        answers = []
        for i in range(batch_size):
            gen_ids = out[i, max_len:] if out.shape[1] > max_len else out[i]
            answers.append(self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
        return answers, input_lens, no_system_lens, context_only_lens

    @torch.no_grad()
    def batch_generate_with_evidence(
        self,
        questions: List[str],
        evidences_batch: List[List[Dict]],
    ) -> Tuple[List[str], List[int], List[int], List[int]]:
        if not evidences_batch:
            return [], [], [], []

        prompts = [self.build_evidence_prompt_text(q, evs) for q, evs in zip(questions, evidences_batch)]
        images_batch = [[ev["image"] for ev in evs if ev["kind"] == "image"] for evs in evidences_batch]
        has_any_images = any(len(imgs) > 0 for imgs in images_batch)
        if has_any_images:
            answers, lengths, no_system_lens, context_only_lens = [], [], [], []
            for q, evs in zip(questions, evidences_batch):
                ans, n_tok, n_no_system, n_context_only = self.generate_with_evidence(q, evs)
                answers.append(ans)
                lengths.append(n_tok)
                no_system_lens.append(n_no_system)
                context_only_lens.append(n_context_only)
            return answers, lengths, no_system_lens, context_only_lens

        try:
            enc = self.processor.tokenizer(
                list(prompts), return_tensors="pt", padding=True, truncation=True, max_length=2048
            ).to(self.device)
            lengths = enc["attention_mask"].sum(dim=1).tolist()
            no_system_lens = []
            context_only_lens = []
            for prompt in prompts:
                n_no_system, n_context_only = self._prompt_token_stats(prompt, image_extra_tokens=0, latent=False)
                no_system_lens.append(n_no_system)
                context_only_lens.append(n_context_only)
            out = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
            max_raw_len = enc["input_ids"].shape[1]
            answers = []
            for i in range(len(prompts)):
                gen_ids = out[i, max_raw_len:] if out.shape[1] > max_raw_len else out[i]
                answers.append(self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
            return answers, [int(x) for x in lengths], no_system_lens, context_only_lens
        except Exception as exc:
            logger.warning(f"Batch evidence generation failed, falling back to serial: {exc}")
            answers, lengths, no_system_lens, context_only_lens = [], [], [], []
            for q, evs in zip(questions, evidences_batch):
                ans, n_tok, n_no_system, n_context_only = self.generate_with_evidence(q, evs)
                answers.append(ans)
                lengths.append(n_tok)
                no_system_lens.append(n_no_system)
                context_only_lens.append(n_context_only)
            return answers, lengths, no_system_lens, context_only_lens


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def _bm25_retrieve(corpus: List[str], query: str, top_k: int) -> List[int]:
    """Return indices of top-k BM25 hits."""
    from rank_bm25 import BM25Okapi
    tokenized = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    k = min(top_k, len(corpus))
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]


def _dense_retrieve(
    corpus: List[str],
    query: str,
    top_k: int,
    encoder,
) -> List[int]:
    """Return indices of top-k dense cosine hits."""
    import numpy as np
    doc_embs = encoder.encode(
        corpus, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
    )
    q_emb    = encoder.encode(
        [query],  convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
    )
    scores   = (doc_embs @ q_emb.T).squeeze(-1).cpu().numpy()
    k = min(top_k, len(corpus))
    return list(map(int, (-scores).argsort()[:k]))


def _nemo_multimodal_retrieve(
    candidates: List[Dict],
    query: str,
    top_k: int,
    encoder,
) -> List[int]:
    text_pairs = [(idx, c["text"]) for idx, c in enumerate(candidates) if c["kind"] == "text"]
    image_pairs = [
        (idx, c["image"], c.get("title", ""))
        for idx, c in enumerate(candidates)
        if c["kind"] == "image"
    ]
    if not text_pairs and not image_pairs:
        return []

    emb_chunks = []
    global_indices = []
    if text_pairs:
        emb_chunks.append(encoder.encode_document(
            [text for _, text in text_pairs],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ))
        global_indices.extend(idx for idx, _ in text_pairs)
    if image_pairs:
        image_docs = [
            {"image": img, "text": title}
            for _, img, title in image_pairs
        ]
        emb_chunks.append(encoder.encode(
            image_docs,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ))
        global_indices.extend(idx for idx, _, _ in image_pairs)

    doc_embs = torch.cat(emb_chunks, dim=0).float()
    q_emb = encoder.encode_query(
        [query],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).float()
    scores = (doc_embs @ q_emb.T).squeeze(-1).cpu().numpy()
    k = min(top_k, len(global_indices))
    local_top = list(map(int, (-scores).argsort()[:k]))
    return [global_indices[i] for i in local_top]


_QWEN3VL_QUERY_PROMPT = "Retrieve relevant documents for the query."


def _qwen3vl_multimodal_retrieve(
    candidates: List[Dict],
    query: str,
    top_k: int,
    encoder,
) -> List[int]:
    documents = []
    global_indices = []
    for idx, c in enumerate(candidates):
        if c["kind"] == "text":
            documents.append(c["text"])
            global_indices.append(idx)
        elif c["kind"] == "image":
            documents.append({"image": c["image"], "text": c.get("title", "")})
            global_indices.append(idx)
    if not documents:
        return []

    q_emb = encoder.encode(
        [query],
        prompt=_QWEN3VL_QUERY_PROMPT,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    doc_embs = encoder.encode(documents, convert_to_tensor=True, show_progress_bar=False)
    if hasattr(encoder, "similarity"):
        scores = encoder.similarity(q_emb, doc_embs).squeeze(0).float().cpu().numpy()
    else:
        q_emb = torch.nn.functional.normalize(q_emb.float(), dim=-1)
        doc_embs = torch.nn.functional.normalize(doc_embs.float(), dim=-1)
        scores = (doc_embs @ q_emb.T).squeeze(-1).cpu().numpy()
    k = min(top_k, len(global_indices))
    local_top = list(map(int, (-scores).argsort()[:k]))
    return [global_indices[i] for i in local_top]


def _clip_image_retrieve(
    images: List[Image.Image],
    query: str,
    top_k: int,
    clip_model,
    clip_processor,
    device: str,
) -> List[int]:
    """
    VisRAG-style retrieval: CLIP visual embeddings for images, CLIP text for query.
    Cosine similarity between image patches and query text.
    """
    import torch.nn.functional as F

    # Encode images
    img_inputs = clip_processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        img_out = clip_model.vision_model(pixel_values=img_inputs["pixel_values"])
        img_feats = clip_model.visual_projection(img_out.pooler_output)
    img_feats = F.normalize(img_feats.float(), dim=-1)   # (N, D)

    # Encode query
    txt_inputs = clip_processor(text=[query], return_tensors="pt", truncation=True, max_length=77).to(device)
    with torch.no_grad():
        txt_out = clip_model.text_model(input_ids=txt_inputs["input_ids"],
                                        attention_mask=txt_inputs.get("attention_mask"))
        txt_feat = clip_model.text_projection(txt_out.pooler_output)
    txt_feat = F.normalize(txt_feat.float(), dim=-1)     # (1, D)

    scores = (img_feats @ txt_feat.T).squeeze(-1).cpu()  # (N,)
    k = min(top_k, len(images))
    return sorted(range(len(scores)), key=lambda i: scores[i].item(), reverse=True)[:k]


# ---------------------------------------------------------------------------
# Per-sample evaluation helpers
# ---------------------------------------------------------------------------

def _load_image(img_dir: str, img_id: str, scale: float) -> Optional[Image.Image]:
    path = os.path.join(img_dir, f"{img_id}.jpg")
    if not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert("RGB")
        return resize_image(img, scale)
    except Exception:
        return None


def _prepare_eval_pool(samples: List[Dict], max_samples: int) -> List[Dict]:
    pool = list(samples)
    random.Random(42).shuffle(pool)
    if max_samples > 0:
        pool = pool[:max_samples]
    return pool


def _serialize_evidences(evidences: List[Dict]) -> List[Dict]:
    out = []
    for ev in evidences:
        if ev["kind"] == "image":
            out.append({"kind": "image", "title": ev["title"]})
        else:
            out.append({"kind": "text", "text": ev["text"]})
    return out


def _get_text_retrieval_units(sample: Dict) -> Tuple[List[str], List[int]]:
    txt_pos = [x for x in list(sample.get("txt_pos_facts", [])) if (x or "").strip()]
    txt_neg = [x for x in list(sample.get("txt_neg_facts", [])) if (x or "").strip()]
    return txt_pos + txt_neg, list(range(len(txt_pos)))


def _build_sample_candidates(sample: Dict, scale: float) -> Optional[Dict]:
    question = sample.get("question", "")
    answers = sample.get("answers", [])
    if isinstance(answers, str):
        answers = [answers]
    if not question or not answers:
        return None

    modality = sample.get("modality", "image" if sample.get("pos_image_ids") else "text")
    txt_pos = [x for x in list(sample.get("txt_pos_facts", [])) if (x or "").strip()]
    txt_neg = [x for x in list(sample.get("txt_neg_facts", [])) if (x or "").strip()]
    txt_pos_evidence = list(sample.get("txt_pos_evidence", txt_pos))
    txt_neg_evidence = list(sample.get("txt_neg_evidence", txt_neg))
    text_retrieval_units, text_retrieval_pos = _get_text_retrieval_units(sample)

    img_dir = sample.get("image_dir", "")
    pos_ids = list(sample.get("pos_image_ids", []))
    neg_ids = list(sample.get("neg_image_ids", []))
    all_img_ids = pos_ids + neg_ids
    all_caps = list(sample.get("pos_captions", [])) + list(sample.get("neg_captions", []))

    full_context_candidates = []
    retrieval_candidates = []
    full_pos_indices = []
    retrieval_pos_indices = []

    for idx, fact in enumerate(txt_pos_evidence + txt_neg_evidence):
        if not (fact or "").strip():
            continue
        if idx < len(txt_pos_evidence) and modality == "text":
            full_pos_indices.append(len(full_context_candidates))
        full_context_candidates.append({"kind": "text", "text": fact, "search_text": fact})

    text_pos_set = set(text_retrieval_pos)
    for idx, fact in enumerate(text_retrieval_units):
        if idx in text_pos_set and modality == "text":
            retrieval_pos_indices.append(len(retrieval_candidates))
        retrieval_candidates.append({"kind": "text", "text": fact, "search_text": fact})

    for idx, (img_id, cap) in enumerate(zip(all_img_ids, all_caps)):
        img = _load_image(img_dir, img_id, scale)
        if img is None:
            continue
        if idx < len(pos_ids) and modality == "image":
            full_pos_indices.append(len(full_context_candidates))
            retrieval_pos_indices.append(len(retrieval_candidates))
        image_candidate = {"kind": "image", "image": img, "title": cap, "search_text": cap}
        full_context_candidates.append(image_candidate)
        retrieval_candidates.append(image_candidate)

    if not full_context_candidates:
        return None

    return {
        "id": sample.get("id", sample.get("qid", sample.get("question_id"))),
        "source": sample.get("source", "webqa"),
        "modality": modality,
        "question": question,
        "answers": answers,
        "webqa_qcate": sample.get("webqa_qcate"),
        "webqa_keywords": sample.get("webqa_keywords"),
        "full_context_candidates": full_context_candidates,
        "retrieval_candidates": retrieval_candidates,
        "full_pos_indices": full_pos_indices,
        "pos_indices": retrieval_pos_indices,
    }


def _finalize_method_metrics(acc: Dict[str, List]) -> Dict[str, float]:
    preds = acc["preds"]
    refs = acc["refs"]
    recalls = acc["recalls"]
    precisions = acc["precisions"]
    tokens = acc["tokens"]
    tokens_no_system = acc.get("tokens_no_system", [])
    tokens_context_only = acc.get("tokens_context_only", [])
    mods = acc["mods"]
    meta = acc["meta"]
    if not preds:
        return {}
    m = evaluate(preds, refs, tokens, token_counts_no_system=tokens_no_system, token_counts_context_only=tokens_context_only, sample_metadata=meta)
    m["recall_at_k"] = sum(recalls) / len(recalls) if recalls else 0.0
    m["precision_at_k"] = sum(precisions) / len(precisions) if precisions else 0.0
    m["n"] = len(preds)
    img_preds = [p for p, mod in zip(preds, mods) if mod == "image"]
    img_refs = [r for r, mod in zip(refs, mods) if mod == "image"]
    txt_preds = [p for p, mod in zip(preds, mods) if mod == "text"]
    txt_refs = [r for r, mod in zip(refs, mods) if mod == "text"]
    img_recalls = [v for v, mod in zip(recalls, mods) if mod == "image"]
    txt_recalls = [v for v, mod in zip(recalls, mods) if mod == "text"]
    img_precisions = [v for v, mod in zip(precisions, mods) if mod == "image"]
    txt_precisions = [v for v, mod in zip(precisions, mods) if mod == "text"]
    img_tokens = [v for v, mod in zip(tokens, mods) if mod == "image"]
    txt_tokens = [v for v, mod in zip(tokens, mods) if mod == "text"]
    img_tokens_no_system = [v for v, mod in zip(tokens_no_system, mods) if mod == "image"]
    txt_tokens_no_system = [v for v, mod in zip(tokens_no_system, mods) if mod == "text"]
    img_tokens_context_only = [v for v, mod in zip(tokens_context_only, mods) if mod == "image"]
    txt_tokens_context_only = [v for v, mod in zip(tokens_context_only, mods) if mod == "text"]
    img_meta = [mm for mm, mod in zip(meta, mods) if mod == "image"]
    txt_meta = [mm for mm, mod in zip(meta, mods) if mod == "text"]
    img_metrics = evaluate(img_preds, img_refs, [0] * len(img_preds), sample_metadata=img_meta) if img_preds else {"em": 0.0, "f1": 0.0}
    txt_metrics = evaluate(txt_preds, txt_refs, [0] * len(txt_preds), sample_metadata=txt_meta) if txt_preds else {"em": 0.0, "f1": 0.0}
    m["img_em"] = img_metrics["em"]
    m["img_f1"] = img_metrics["f1"]
    m["txt_em"] = txt_metrics["em"]
    m["txt_f1"] = txt_metrics["f1"]
    if "acc" in img_metrics:
        m["img_acc"] = img_metrics["acc"]
        m["img_webqa_acc"] = img_metrics["acc"]
    if "acc" in txt_metrics:
        m["txt_acc"] = txt_metrics["acc"]
        m["txt_webqa_acc"] = txt_metrics["acc"]
    m["img_recall_at_k"] = sum(img_recalls) / len(img_recalls) if img_recalls else 0.0
    m["txt_recall_at_k"] = sum(txt_recalls) / len(txt_recalls) if txt_recalls else 0.0
    m["img_precision_at_k"] = sum(img_precisions) / len(img_precisions) if img_precisions else 0.0
    m["txt_precision_at_k"] = sum(txt_precisions) / len(txt_precisions) if txt_precisions else 0.0
    m["img_avg_tokens"] = sum(img_tokens) / len(img_tokens) if img_tokens else 0.0
    m["txt_avg_tokens"] = sum(txt_tokens) / len(txt_tokens) if txt_tokens else 0.0
    m["img_avg_tokens_no_system"] = sum(img_tokens_no_system) / len(img_tokens_no_system) if img_tokens_no_system else 0.0
    m["txt_avg_tokens_no_system"] = sum(txt_tokens_no_system) / len(txt_tokens_no_system) if txt_tokens_no_system else 0.0
    m["img_avg_tokens_context_only"] = sum(img_tokens_context_only) / len(img_tokens_context_only) if img_tokens_context_only else 0.0
    m["txt_avg_tokens_context_only"] = sum(txt_tokens_context_only) / len(txt_tokens_context_only) if txt_tokens_context_only else 0.0
    return m


def _finalize_token_only_metrics(acc: Dict[str, List]) -> Dict[str, float]:
    tokens = acc["tokens"]
    tokens_no_system = acc.get("tokens_no_system", [])
    tokens_context_only = acc.get("tokens_context_only", [])
    mods = acc["mods"]
    if not tokens:
        return {}
    m = {
        "avg_tokens": sum(tokens) / len(tokens),
        "avg_tokens_no_system": sum(tokens_no_system) / len(tokens_no_system) if tokens_no_system else 0.0,
        "avg_tokens_context_only": sum(tokens_context_only) / len(tokens_context_only) if tokens_context_only else 0.0,
        "n": len(tokens),
    }
    img_tokens = [v for v, mod in zip(tokens, mods) if mod == "image"]
    txt_tokens = [v for v, mod in zip(tokens, mods) if mod == "text"]
    img_tokens_no_system = [v for v, mod in zip(tokens_no_system, mods) if mod == "image"]
    txt_tokens_no_system = [v for v, mod in zip(tokens_no_system, mods) if mod == "text"]
    img_tokens_context_only = [v for v, mod in zip(tokens_context_only, mods) if mod == "image"]
    txt_tokens_context_only = [v for v, mod in zip(tokens_context_only, mods) if mod == "text"]
    m["img_avg_tokens"] = sum(img_tokens) / len(img_tokens) if img_tokens else 0.0
    m["txt_avg_tokens"] = sum(txt_tokens) / len(txt_tokens) if txt_tokens else 0.0
    m["img_avg_tokens_no_system"] = sum(img_tokens_no_system) / len(img_tokens_no_system) if img_tokens_no_system else 0.0
    m["txt_avg_tokens_no_system"] = sum(txt_tokens_no_system) / len(txt_tokens_no_system) if txt_tokens_no_system else 0.0
    m["img_avg_tokens_context_only"] = sum(img_tokens_context_only) / len(img_tokens_context_only) if img_tokens_context_only else 0.0
    m["txt_avg_tokens_context_only"] = sum(txt_tokens_context_only) / len(txt_tokens_context_only) if txt_tokens_context_only else 0.0
    return m


def run_unified_baselines_multik(
    samples: List[Dict],
    generator: LLaVAGenerator,
    top_k_values: List[int],
    scale: float,
    max_samples: int,
    methods: List[str],
    dense_model_name: str,
    device: str,
    dense_encoder=None,
    nemo_encoder=None,
    qwen3vl_encoder=None,
    clip_model=None,
    clip_processor=None,
    compressor=None,
    pool: Optional[List[Dict]] = None,
    gen_batch_size: int = 8,
    dump_examples: int = 0,
    token_count_only: bool = False,
    offload_generator_during_latent: bool = False,
) -> Tuple[Dict[str, Dict], Dict[str, List[Dict]]]:
    if not methods:
        return {}, {}
    if "dense" in methods and dense_encoder is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading dense encoder: {dense_model_name}")
        dense_encoder = SentenceTransformer(dense_model_name)
    if pool is None:
        pool = _prepare_eval_pool(samples, max_samples)

    prepared_pool = []
    for s in tqdm(pool, desc="prep-unified", leave=False):
        prepared = _build_sample_candidates(s, scale)
        if prepared is not None:
            prepared_pool.append(prepared)

    max_k = max(top_k_values)
    method_keys = []
    for method in methods:
        if method == "full_context":
            method_keys.append("full_context")
        else:
            method_keys.extend([f"{method}_k{k}" for k in top_k_values])
    acc = {
        key: {"preds": [], "refs": [], "recalls": [], "precisions": [], "tokens": [], "tokens_no_system": [], "tokens_context_only": [], "mods": [], "meta": []}
        for key in method_keys
    }
    pending = {key: [] for key in method_keys}
    dumped = {key: [] for key in method_keys}

    def _record_generated(key: str, chunk: List[Dict], answers, lengths, lengths_no_system, lengths_context_only):
        for ans, n_tok, n_tok_no_system, n_tok_context_only, item in zip(
            answers, lengths, lengths_no_system, lengths_context_only, chunk
        ):
            acc[key]["preds"].append(ans)
            acc[key]["refs"].append(item["refs"])
            acc[key]["recalls"].append(item["recall"])
            acc[key]["precisions"].append(item["precision"])
            acc[key]["tokens"].append(n_tok)
            acc[key]["tokens_no_system"].append(n_tok_no_system)
            acc[key]["tokens_context_only"].append(n_tok_context_only)
            acc[key]["mods"].append(item["modality"])
            acc[key]["meta"].append(item["meta"])

    def _flush_key(key: str, force: bool = False):
        if not pending[key]:
            return
        if not force and len(pending[key]) < gen_batch_size:
            return

        chunk = pending[key] if force else pending[key][:gen_batch_size]
        if force:
            pending[key] = []
        else:
            del pending[key][:gen_batch_size]

        if token_count_only:
            answers, lengths = [], []
            lengths_no_system, lengths_context_only = [], []
            for item in chunk:
                if item["mode"] == "latent":
                    n_tok, n_tok_no_system, n_tok_context_only = generator.count_with_latents(
                        item["question"], item["payload"]
                    )
                else:
                    n_tok, n_tok_no_system, n_tok_context_only = generator.count_with_evidence(
                        item["question"], item["payload"]
                    )
                answers.append("")
                lengths.append(n_tok)
                lengths_no_system.append(n_tok_no_system)
                lengths_context_only.append(n_tok_context_only)
            _record_generated(key, chunk, answers, lengths, lengths_no_system, lengths_context_only)
            return

        questions = [it["question"] for it in chunk]
        try:
            if chunk[0]["mode"] == "latent":
                answers, lengths, lengths_no_system, lengths_context_only = generator.batch_generate_with_latents(
                    questions, [it["payload"] for it in chunk]
                )
            else:
                answers, lengths, lengths_no_system, lengths_context_only = generator.batch_generate_with_evidence(
                    questions, [it["payload"] for it in chunk]
                )
        except Exception as e:
            logger.warning(f"{key} batch generation failed, falling back to serial: {e}")
            answers, lengths, lengths_no_system, lengths_context_only = [], [], [], []
            for item in chunk:
                try:
                    if item["mode"] == "latent":
                        ans, n_tok, n_no_system, n_context_only = generator.generate_with_latents(
                            item["question"], item["payload"]
                        )
                    else:
                        ans, n_tok, n_no_system, n_context_only = generator.generate_with_evidence(
                            item["question"], item["payload"]
                        )
                except Exception as inner_e:
                    logger.warning(f"{key} generation failed: {inner_e}")
                    ans, n_tok, n_no_system, n_context_only = "", 0, 0, 0
                answers.append(ans)
                lengths.append(n_tok)
                lengths_no_system.append(n_no_system)
                lengths_context_only.append(n_context_only)

        _record_generated(key, chunk, answers, lengths, lengths_no_system, lengths_context_only)
        del chunk

    for s in tqdm(prepared_pool, desc="unified-baselines", leave=False):
        question = s["question"]
        answers = s["answers"]
        full_context_candidates = s["full_context_candidates"]
        candidates = s["retrieval_candidates"]
        pos_indices = set(s["pos_indices"])
        full_pos_indices = set(s["full_pos_indices"])
        modality = s["modality"]
        meta = {
            "source": s.get("source", "webqa"),
            "modality": modality,
            "webqa_qcate": s.get("webqa_qcate"),
            "webqa_keywords": s.get("webqa_keywords"),
        }

        if "full_context" in methods:
            hits = len(full_pos_indices)
            precision = hits / len(full_context_candidates) if full_context_candidates else 0.0
            pending["full_context"].append({
                "mode": "evidence",
                "sample_id": s.get("id"),
                "question": question,
                "payload": full_context_candidates,
                "refs": answers,
                "recall": 1.0 if full_pos_indices else 0.0,
                "precision": precision,
                "modality": modality,
                "meta": meta,
            })
            _flush_key("full_context")
            if dump_examples > 0 and len(dumped["full_context"]) < dump_examples:
                dumped["full_context"].append({
                    "sample_id": s.get("id"),
                    "modality": modality,
                    "question": question,
                    "answers": answers,
                    "retrieved": _serialize_evidences(full_context_candidates),
                    "prompt": generator.build_evidence_prompt_text(question, full_context_candidates),
                })

        ranked_indices: Dict[str, List[int]] = {}
        search_corpus = [c["search_text"] for c in candidates]
        if "bm25" in methods:
            ranked_indices["bm25"] = _bm25_retrieve(search_corpus, question, max_k)
        if "dense" in methods:
            ranked_indices["dense"] = _dense_retrieve(search_corpus, question, max_k, dense_encoder)
        if "nemo" in methods:
            ranked_indices["nemo"] = _nemo_multimodal_retrieve(candidates, question, max_k, nemo_encoder)
        if "qwen3vl" in methods:
            ranked_indices["qwen3vl"] = _qwen3vl_multimodal_retrieve(candidates, question, max_k, qwen3vl_encoder)
        if "visrag" in methods:
            image_pairs = [(idx, c) for idx, c in enumerate(candidates) if c["kind"] == "image"]
            if image_pairs and clip_model is not None and clip_processor is not None:
                local_top = _clip_image_retrieve(
                    [c["image"] for _, c in image_pairs],
                    question,
                    min(max_k, len(image_pairs)),
                    clip_model,
                    clip_processor,
                    device,
                )
                ranked_indices["visrag"] = [image_pairs[i][0] for i in local_top]
            else:
                ranked_indices["visrag"] = []

        latent_projected = None
        latent_ranked = None
        if "latent" in methods:
            if compressor is None:
                raise ValueError("latent baseline requires a LLaVA compressor checkpoint")
            generator_was_offloaded = False
            if offload_generator_during_latent and device == "cuda":
                generator.to("cpu")
                generator_was_offloaded = True
                gc.collect()
                torch.cuda.empty_cache()
            image_indices = [idx for idx, c in enumerate(candidates) if c["kind"] == "image"]
            text_indices = [idx for idx, c in enumerate(candidates) if c["kind"] == "text"]
            latent_candidate_indices = image_indices + text_indices
            image_items = [candidates[idx] for idx in image_indices]
            text_items = [candidates[idx] for idx in text_indices]
            latent_chunks = []
            try:
                if text_items:
                    text_prompts = [c["text"] for c in text_items]
                    text_latents = compressor.compress_batch(
                        [type(compressor).format_compress_prompt(t, False) for t in text_prompts],
                        with_grad=False,
                        adapter="compress",
                    )
                if image_items:
                    img_latents = compressor.compress_batch(
                        [type(compressor).format_compress_prompt(c["title"], True) for c in image_items],
                        images=[c["image"] for c in image_items],
                        with_grad=False,
                        adapter="compress",
                    )
                    latent_chunks.append(img_latents)
                if text_items:
                    latent_chunks.append(text_latents)
                if latent_chunks:
                    cand_latents = torch.cat(latent_chunks, dim=0)
                    q_lat = compressor.embed_query_batch([question], with_grad=False)
                    q_emb = compressor.get_retrieval_embedding(q_lat)
                    cand_embs = compressor.get_retrieval_embedding(cand_latents)
                    sims = (cand_embs @ q_emb.T).squeeze(-1)
                    latent_ranked = torch.topk(
                        sims, k=min(max_k, cand_latents.shape[0])
                    ).indices.tolist()
                    latent_projected = compressor.project_for_generator(cand_latents).detach().cpu()
                    del cand_latents, q_lat, q_emb, cand_embs, sims
                else:
                    latent_ranked = []
            finally:
                if generator_was_offloaded:
                    generator.to(device)
                    torch.cuda.empty_cache()

        for method in methods:
            if method == "full_context":
                continue
            for k in top_k_values:
                key = f"{method}_k{k}"
                if method == "latent":
                    local_top_i = latent_ranked[: min(k, len(latent_ranked))]
                    top_i = [latent_candidate_indices[i] for i in local_top_i]
                    payload = latent_projected[local_top_i] if local_top_i else None
                else:
                    top_i = ranked_indices.get(method, [])[: min(k, len(ranked_indices.get(method, [])))]
                    payload = [candidates[i] for i in top_i]
                if not top_i:
                    continue

                hits = len(set(top_i) & pos_indices)
                recall = hits / len(pos_indices) if pos_indices else 0.0
                precision = hits / len(top_i) if top_i else 0.0
                pending[key].append({
                    "mode": "latent" if method == "latent" else "evidence",
                    "sample_id": s.get("id"),
                    "question": question,
                    "payload": payload,
                    "refs": answers,
                    "recall": recall,
                    "precision": precision,
                    "modality": modality,
                    "meta": meta,
                })
                _flush_key(key)
                if dump_examples > 0 and len(dumped[key]) < dump_examples:
                    retrieved = [candidates[i] for i in top_i]
                    dumped[key].append({
                        "sample_id": s.get("id"),
                        "modality": modality,
                        "question": question,
                        "answers": answers,
                        "retrieved_indices": top_i,
                        "retrieved": _serialize_evidences(retrieved),
                        "prompt": (
                            generator.build_latent_prompt_text(question, payload)
                            if method == "latent"
                            else generator.build_evidence_prompt_text(question, retrieved)
                        ),
                    })

    for key in pending:
        _flush_key(key, force=True)

    finalizer = _finalize_token_only_metrics if token_count_only else _finalize_method_metrics
    results = {key: finalizer(val) for key, val in acc.items()}
    return results, dumped


def debug_token_counts(
    samples: List[Dict],
    generator: LLaVAGenerator,
    top_k: int,
    scale: float,
    max_samples: int,
) -> None:
    pool = _prepare_eval_pool(samples, max_samples)
    rows = []
    for s in tqdm(pool, desc="debug-tokens", leave=False):
        prepared = _build_sample_candidates(s, scale)
        if prepared is None:
            continue
        candidates = prepared["retrieval_candidates"]
        full_context_candidates = prepared["full_context_candidates"]
        top_candidates = candidates[: min(top_k, len(candidates))]
        for name, payload in [("full_context", full_context_candidates), (f"first_k{top_k}", top_candidates)]:
            prompt = generator.build_evidence_prompt_text(prepared["question"], payload)
            images = [ev["image"] for ev in payload if ev["kind"] == "image"]
            try:
                _, n_tok = generator._encode_evidence_prompt(prompt, images)
            except Exception as e:
                logger.warning(f"debug token encode failed ({name}): {e}")
                continue
            rows.append({
                "method": name,
                "modality": prepared["modality"],
                "tokens": n_tok,
                "n_images": len(images),
                "n_text": sum(1 for ev in payload if ev["kind"] == "text"),
            })

    if not rows:
        print("No token debug rows.")
        return
    print("\nToken Debug")
    print("method modality n tokens avg_images avg_text")
    keys = sorted(set((r["method"], r["modality"]) for r in rows))
    for method, modality in keys:
        vals = [r for r in rows if r["method"] == method and r["modality"] == modality]
        print(
            f"{method} {modality} {len(vals)} "
            f"{sum(r['tokens'] for r in vals) / len(vals):.2f} "
            f"{sum(r['n_images'] for r in vals) / len(vals):.2f} "
            f"{sum(r['n_text'] for r in vals) / len(vals):.2f}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _format_table(title: str, results: Dict[str, Dict], top_k: int) -> str:
    W = 78
    has_acc = any("acc" in m for m in results.values() if m)
    lines = []
    lines.append("─" * W)
    lines.append(f"  {title}")
    lines.append("─" * W)
    header = f"  {'Method':<18} {'n':>5} {'EM':>7} {'F1':>7} {'ROUGE-L':>9}"
    if has_acc:
        header += f" {'Acc':>7}"
    header += f" {'R@k':>6} {'AvgTok':>8}"
    lines.append(header)
    lines.append("─" * W)
    for method, m in results.items():
        if not m:
            lines.append(f"  {method:<18}  (no results)")
            continue
        row = (
            f"  {method:<18} {m['n']:5d} {m['em']:7.4f} {m['f1']:7.4f} "
            f"{m.get('rouge_l', 0):9.4f}"
        )
        if has_acc:
            row += f" {m.get('acc', float('nan')):7.4f}"
        row += f" {m['recall_at_k']:6.4f} {m.get('avg_tokens', 0):8.1f}"
        lines.append(row)
    lines.append("─" * W)
    return "\n".join(lines)



def _print_table(title: str, results: Dict[str, Dict], top_k: int):
    print("\n" + _format_table(title, results, top_k))


def _fmt_metric(metrics: Dict, key: str, scale: float = 1.0) -> str:
    val = metrics.get(key)
    if val is None:
        return "NA"
    return f"{val * scale:.2f}"


def _print_split_report(results: Dict[str, Dict]) -> None:
    if not results:
        print("No results.")
        return

    print("\nOverall")
    print("method f1 acc r@k p@k tokens n")
    for method, metrics in results.items():
        if not metrics:
            print(f"{method} empty")
            continue
        print(" ".join([
            method,
            _fmt_metric(metrics, "f1", 100.0),
            _fmt_metric(metrics, "acc", 100.0),
            _fmt_metric(metrics, "recall_at_k", 100.0),
            _fmt_metric(metrics, "precision_at_k", 100.0),
            _fmt_metric(metrics, "avg_tokens"),
            str(metrics.get("n", metrics.get("n_samples", "NA"))),
        ]))

    print("\nImage")
    print("method f1 acc r@k p@k tokens")
    for method, metrics in results.items():
        if not metrics:
            continue
        print(" ".join([
            method,
            _fmt_metric(metrics, "img_f1", 100.0),
            _fmt_metric(metrics, "img_acc", 100.0),
            _fmt_metric(metrics, "img_recall_at_k", 100.0),
            _fmt_metric(metrics, "img_precision_at_k", 100.0),
            _fmt_metric(metrics, "img_avg_tokens"),
        ]))

    print("\nText")
    print("method f1 acc r@k p@k tokens")
    for method, metrics in results.items():
        if not metrics:
            continue
        print(" ".join([
            method,
            _fmt_metric(metrics, "txt_f1", 100.0),
            _fmt_metric(metrics, "txt_acc", 100.0),
            _fmt_metric(metrics, "txt_recall_at_k", 100.0),
            _fmt_metric(metrics, "txt_precision_at_k", 100.0),
            _fmt_metric(metrics, "txt_avg_tokens"),
        ]))


def _print_token_only_report(results: Dict[str, Dict]) -> None:
    if not results:
        print("No token results.")
        return
    print("\nToken Counts")
    print("method avg_tokens avg_tokens_no_system avg_tokens_context_only n")
    for method, metrics in results.items():
        if not metrics:
            print(f"{method} empty")
            continue
        print(" ".join([
            method,
            _fmt_metric(metrics, "avg_tokens"),
            _fmt_metric(metrics, "avg_tokens_no_system"),
            _fmt_metric(metrics, "avg_tokens_context_only"),
            str(metrics.get("n", "NA")),
        ]))


def main(args):
    if args.disable_tqdm:
        _disable_tqdm_progress()
    elif args.simple_tqdm:
        _simple_tqdm_progress()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = cfg["model"]
    img_cfg   = cfg.get("image", {})
    scale     = img_cfg.get("scale", 1 / 20)

    gen_name       = model_cfg["generator_name"]
    max_new_tokens = cfg.get("generation", {}).get("max_new_tokens", 64)
    dense_model    = cfg.get("visrag", {}).get("dense_model",
                     "sentence-transformers/all-MiniLM-L6-v2")
    nemo_model     = args.nemo_model or cfg.get("visrag", {}).get("nemo_model",
                     "nvidia/llama-nemotron-embed-vl-1b-v2")
    qwen3vl_model  = args.qwen3vl_model or cfg.get("visrag", {}).get("qwen3vl_model",
                     "Qwen/Qwen3-VL-Embedding-8B")
    clip_model     = cfg.get("visrag", {}).get("clip_model",
                     "openai/clip-vit-large-patch14")

    # top_k_values: CLI overrides config; default [1, 2, 5]
    if args.top_k_values:
        top_k_values = sorted(set(args.top_k_values))
    else:
        top_k_values = sorted(set(
            cfg.get("retrieval", {}).get("top_k_values", [1, 2, 5])
        ))

    # ── Load validation data ───────────────────────────────────────────────
    data_cfg = cfg.get("data", {})
    val_path = args.val or data_cfg.get("val")
    if not val_path or not os.path.exists(val_path):
        raise ValueError(f"Val data not found: {val_path}")

    logger.info(f"Loading val data: {val_path}")
    val_all = load_webqa_samples(val_path)
    if args.image_dir:
        for s in val_all:
            s["image_dir"] = args.image_dir
    for s in val_all:
        if "modality" not in s:
            s["modality"] = "image" if s.get("pos_image_ids") else "text"
    val_image = [s for s in val_all if s["modality"] == "image"]
    val_text  = [s for s in val_all if s["modality"] == "text"]
    logger.info(f"  val image: {len(val_image)}  val text: {len(val_text)}")

    # ── Parse requested methods ────────────────────────────────────────────
    requested = [m.lower() for m in args.methods]
    valid = {"bm25", "dense", "nemo", "qwen3vl", "visrag", "latent", "full_context"}
    for m in requested:
        if m not in valid:
            logger.warning(f"Unknown method '{m}' — ignoring.")

    # full_context runs once (k is irrelevant); retrieval methods run per k
    fc_requested      = "full_context" in requested
    retrieval_methods = [m for m in requested if m in valid and m != "full_context"]

    image_retrieval = [m for m in retrieval_methods if m in valid]
    text_retrieval  = [m for m in retrieval_methods if m in ("bm25", "dense", "nemo", "qwen3vl", "latent")]

    if not fc_requested and not retrieval_methods:
        raise ValueError("No valid methods specified.")

    if "latent" in requested and not args.checkpoint:
        raise ValueError("--checkpoint is required for the 'latent' method.")

    # ── Load generator once (shared by all methods and all k) ─────────────
    generator = LLaVAGenerator(gen_name, device=device, max_new_tokens=max_new_tokens)
    if args.debug_token_counts:
        debug_token_counts(
            val_all,
            generator,
            top_k=max(top_k_values),
            scale=scale,
            max_samples=args.max_samples,
        )
        return {}

    dense_encoder = None
    if "dense" in retrieval_methods:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading shared dense encoder: {dense_model}")
        dense_encoder = SentenceTransformer(dense_model)
    nemo_encoder = None
    if "nemo" in retrieval_methods:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading shared NeMo multimodal encoder: {nemo_model}")
        nemo_encoder = SentenceTransformer(nemo_model, trust_remote_code=True)
    qwen3vl_encoder = None
    if "qwen3vl" in retrieval_methods:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading shared Qwen3-VL embedding encoder: {qwen3vl_model}")
        qwen3vl_encoder = SentenceTransformer(qwen3vl_model, trust_remote_code=True)

    clip_model_obj = None
    clip_proc = None
    if "visrag" in image_retrieval:
        from transformers import CLIPModel, CLIPProcessor
        logger.info(f"Loading shared CLIP model for VisRAG: {clip_model}")
        clip_proc = CLIPProcessor.from_pretrained(clip_model)
        clip_model_obj = CLIPModel.from_pretrained(
            clip_model, torch_dtype=torch.float32
        ).to(device)
        clip_model_obj.eval()

    # ── Load compressor if needed ──────────────────────────────────────────
    compressor = None
    if "latent" in requested:
        from src.llava_compressor import LLaVACompressor
        comp_cfg = cfg.get("compressor", {})
        dec_cfg = cfg.get("decoder", {})
        logger.info(f"Loading LLaVACompressor from {args.checkpoint}")
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
        compressor.load(args.checkpoint)
        compressor.eval()

    run_methods = []
    if fc_requested:
        run_methods.append("full_context")
    run_methods.extend(retrieval_methods)
    unified_pool = _prepare_eval_pool(val_all, args.max_samples)
    unified_results, dumped_examples = run_unified_baselines_multik(
        val_all, generator, top_k_values, scale, args.max_samples, run_methods,
        dense_model, device,
        dense_encoder=dense_encoder,
        nemo_encoder=nemo_encoder,
        qwen3vl_encoder=qwen3vl_encoder,
        clip_model=clip_model_obj,
        clip_processor=clip_proc,
        compressor=compressor,
        pool=unified_pool,
        gen_batch_size=args.gen_batch_size,
        dump_examples=args.dump_examples,
        token_count_only=args.token_count_only,
        offload_generator_during_latent=args.latent_offload,
    )

    if args.token_count_only:
        _print_token_only_report(unified_results)
    else:
        _print_table(
            f"UNIFIED baselines  (k∈{top_k_values}, n≤{args.max_samples})",
            unified_results,
            top_k_values[-1],
        )
        _print_split_report(unified_results)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "top_k_values": top_k_values,
                "results": unified_results,
                "unified": unified_results,
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {args.output}")
        if args.dump_examples > 0:
            dump_path = args.output.replace(".json", ".examples.json")
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump({"unified": dumped_examples}, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved examples: {dump_path}")

        txt_path = args.output.replace(".json", ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join([
                "WebQA LLaVA Unified Baseline Results",
                f"top_k_values={top_k_values}  max_samples={args.max_samples}",
                "",
                _format_table(
                    f"UNIFIED retrieval baselines  (k∈{top_k_values}, n≤{args.max_samples})",
                    unified_results,
                    top_k_values[-1],
                ),
            ]))
        logger.info(f"Text report saved: {txt_path}")

    return {"results": unified_results, "unified": unified_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebQA LLaVA baselines — image (BM25/Dense/VisRAG) and text (BM25/Dense)."
    )
    parser.add_argument("--config",       default="config_llava.yaml")
    parser.add_argument("--val",          default=None,
                        help="WebQA val JSON (overrides data.val in config)")
    parser.add_argument("--methods",      nargs="+", default=["bm25", "dense", "visrag", "latent"],
                        help="Methods to run: bm25 dense visrag latent full_context")
    parser.add_argument("--checkpoint",   default=None,
                        help="Path to LLaVACompressor checkpoint (required for 'latent')")
    parser.add_argument("--top_k_values", type=int, nargs="+", default=None,
                        help="One or more k values, e.g. --top_k_values 1 2 5  "
                             "(default: [1, 2, 5])")
    parser.add_argument("--max_samples",  type=int, default=200,
                        help="Max val samples per modality (0 = all)")
    parser.add_argument("--image_dir", default=None,
                        help="Optional override for sample['image_dir'] when local JSON paths are stale")
    parser.add_argument("--nemo_model", default=None,
                        help="NeMo multimodal embedding model id or local path. Overrides config visrag.nemo_model.")
    parser.add_argument("--qwen3vl_model", default=None,
                        help="Qwen3-VL embedding model id or local path. Overrides config visrag.qwen3vl_model.")
    parser.add_argument("--gen_batch_size", type=int, default=8,
                        help="Batch size for unified generation")
    parser.add_argument("--dump_examples", type=int, default=0,
                        help="Dump first N examples per method to a sidecar JSON")
    parser.add_argument("--debug_token_counts", action="store_true",
                        help="Only estimate unified full-context and first-k token counts, then exit.")
    parser.add_argument("--token_count_only", action="store_true",
                        help="Run retrieval/prompt construction only and report token counts without generation.")
    parser.add_argument("--latent_offload", action="store_true",
                        help="Move the LLaVA generator to CPU during latent compression. Saves VRAM, but is much slower.")
    parser.add_argument("--disable_tqdm", action="store_true",
                        help="Disable tqdm progress bars in baseline loops.")
    parser.add_argument("--simple_tqdm", action="store_true",
                        help="Show only outer sample-level tqdm bars, hiding batch-level bars.")
    parser.add_argument("--output",       default="results/baselines_llava.json",
                        help="Path to save JSON results")
    args = parser.parse_args()
    main(args)

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
import json
import logging
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from data.prepare_webqa import load_webqa_samples
from src.evaluation import evaluate
from src.region_encoder import resize_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.max_new_tokens = max_new_tokens

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

    def build_evidence_prompt_text(self, question: str, evidences: List[Dict]) -> str:
        lines = []
        for i, ev in enumerate(evidences):
            if ev["kind"] == "image":
                lines.append(f"Evidence {i + 1}: <image>\nTitle: {ev['title']}")
            else:
                lines.append(f"Evidence {i + 1}: {ev['text']}")
        return "USER: " + "\n".join(lines) + f"\nQuestion: {question}\nASSISTANT:"

    def build_latent_prompt_text(self, question: str, proj_tokens: torch.Tensor) -> str:
        lines = [f"Latent {i + 1}: [LATENT]" for i in range(proj_tokens.shape[0])]
        return "USER: " + "\n".join(lines) + f"\nQuestion: {question}\nASSISTANT:"

    @torch.no_grad()
    def generate_with_images(
        self,
        question: str,
        images: List[Image.Image],
        titles: List[str],
    ) -> str:
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
            return "", 0
        # Build prompt: one <image> token per retrieved image
        lines = []
        for i, title in enumerate(titles):
            lines.append(f"Evidence {i + 1}: <image>\nTitle: {title}")
        prompt = "USER: " + "\n".join(lines) + f"\nQuestion: {question}\nASSISTANT:"

        n_imgs_used = len(images)
        try:
            enc = self.processor(
                text=prompt,
                images=images,
                return_tensors="pt",
            ).to(self.device)
        except Exception as e:
            logger.warning(f"Processor failed with {len(images)} images: {e}; using first image only.")
            prompt = f"USER: Evidence 1: <image>\nTitle: {titles[0]}\nQuestion: {question}\nASSISTANT:"
            enc = self.processor(
                text=prompt,
                images=[images[0]],
                return_tensors="pt",
            ).to(self.device)
            n_imgs_used = 1

        n_in = enc["input_ids"].shape[-1]
        n_effective = self._effective_image_input_tokens(enc, n_imgs_used)

        out  = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        return self.processor.tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip(), n_effective

    @torch.no_grad()
    def generate_with_evidence(
        self,
        question: str,
        evidences: List[Dict],
    ) -> Tuple[str, int]:
        if not evidences:
            return "", 0
        prompt = self.build_evidence_prompt_text(question, evidences)
        images = [ev["image"] for ev in evidences if ev["kind"] == "image"]
        enc, n_effective = self._encode_evidence_prompt(prompt, images)
        n_in = enc["input_ids"].shape[-1]
        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        return self.processor.tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip(), n_effective

    @torch.no_grad()
    def generate_with_latents(
        self,
        question: str,
        proj_tokens: torch.Tensor,  # (k, gen_hidden) — output of project_for_generator
    ) -> str:
        """
        Inject projected latent tokens into LLaVA-13B via input_embeds.

        Prompt structure (matches training validation):
          BOS USER: Latent 1: [lat_1] Latent 2: [lat_2] … Question: {q} ASSISTANT:
        """
        embed_fn = self.model.get_input_embeddings()
        dtype    = next(self.model.parameters()).dtype

        def _e(text, add_bos: bool):
            if add_bos:
                ids = self.processor(text=text, return_tensors="pt")["input_ids"].to(self.device)
            else:
                ids = self.processor.tokenizer(
                    text, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(self.device)
            return embed_fn(ids).to(dtype)

        parts = [_e("USER: ", add_bos=True)]
        for i, lat in enumerate(proj_tokens):
            parts.append(_e(f"Latent {i + 1}: ", add_bos=False))
            parts.append(lat.reshape(-1, lat.shape[-1]).to(dtype).unsqueeze(0).to(self.device))
        parts.append(_e(f"\nQuestion: {question}\nASSISTANT:", add_bos=False))
        input_embs = torch.cat(parts, dim=1)
        n_tok = input_embs.shape[1]

        out = self.model.generate(
            inputs_embeds=input_embs,
            attention_mask=torch.ones(1, n_tok, dtype=torch.long, device=self.device),
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        # generate() with inputs_embeds returns only the newly generated token IDs
        # (not the input embeddings), so out[0] is always the generated sequence.
        return self.processor.tokenizer.decode(out[0], skip_special_tokens=True).strip(), n_tok

    @torch.no_grad()
    def generate_with_text(
        self,
        question: str,
        facts: List[str],
    ) -> str:
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
            return "", 0
        lines = [f"Evidence {i + 1}: {f}" for i, f in enumerate(facts)]
        prompt = "USER: " + "\n".join(lines) + f"\nQuestion: {question}\nASSISTANT:"

        ids = self.processor.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )["input_ids"].to(self.device)
        n_in = ids.shape[-1]
        out  = self.model.generate(
            input_ids=ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        return self.processor.tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip(), n_in


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
    doc_embs = encoder.encode(corpus, convert_to_tensor=True, normalize_embeddings=True)
    q_emb    = encoder.encode([query],  convert_to_tensor=True, normalize_embeddings=True)
    scores   = (doc_embs @ q_emb.T).squeeze(-1).cpu().numpy()
    k = min(top_k, len(corpus))
    return list(map(int, (-scores).argsort()[:k]))


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


def _eval_image_sample(
    sample: Dict,
    generator: LLaVAGenerator,
    top_k: int,
    scale: float,
    method: str,
    dense_encoder=None,
    clip_model=None,
    clip_processor=None,
    compressor=None,
    device: str = "cuda",
) -> Optional[Tuple[str, List[str], float, int]]:
    """
    Retrieval + generation for one image-modality sample.
    Returns (prediction, references, recall@k, n_tokens) or None if candidate pool is empty.
    """
    from src.llava_compressor import LLaVACompressor

    img_dir   = sample.get("image_dir", "")
    pos_ids   = sample.get("pos_image_ids", [])
    neg_ids   = sample.get("neg_image_ids", [])
    all_ids   = pos_ids + neg_ids
    all_caps  = sample.get("pos_captions", []) + sample.get("neg_captions", [])
    question  = sample["question"]
    answers   = sample["answers"]
    if isinstance(answers, str):
        answers = [answers]

    # Load all candidate images; track which loaded indices are positives
    cand_imgs   : List[Image.Image] = []
    cand_titles : List[str]         = []
    pos_loaded_indices: List[int]   = []
    pos_id_set = set(pos_ids)
    for img_id, cap in zip(all_ids, all_caps):
        img = _load_image(img_dir, img_id, scale)
        if img is not None:
            if img_id in pos_id_set:
                pos_loaded_indices.append(len(cand_imgs))
            cand_imgs.append(img)
            cand_titles.append(cap)

    if not cand_imgs:
        return None

    recall  = 0.0
    n_tokens = 0

    if method == "full_context":
        recall = 1.0 if pos_loaded_indices else 0.0
        pred, n_tokens = generator.generate_with_images(question, cand_imgs, cand_titles)
        return pred, answers, recall, n_tokens

    if method == "latent":
        cand_prompts = [LLaVACompressor.format_compress_prompt(t, True) for t in cand_titles]
        cand_latents = compressor.compress_batch(cand_prompts, images=cand_imgs,
                                                 with_grad=False, adapter="compress")
        cand_embs    = compressor.get_retrieval_embedding(cand_latents)
        q_lat        = compressor.embed_query_batch([question], with_grad=False)
        q_emb        = compressor.get_retrieval_embedding(q_lat)
        sims         = (cand_embs @ q_emb.T).squeeze(-1)
        k            = min(top_k, len(cand_imgs))
        top_i        = torch.topk(sims, k=k).indices.tolist()

        if pos_loaded_indices:
            hits = len(set(top_i) & set(pos_loaded_indices))
            recall = hits / len(pos_loaded_indices)

        top_latents = cand_latents[top_i]
        projected   = compressor.project_for_generator(top_latents)
        try:
            pred, n_tokens = generator.generate_with_latents(question, projected)
        except Exception as e:
            logger.warning(f"Latent image generation failed: {e}")
            pred, n_tokens = "", 0
    else:
        if method == "bm25":
            top_i = _bm25_retrieve(cand_titles, question, top_k)
        elif method == "dense":
            top_i = _dense_retrieve(cand_titles, question, top_k, dense_encoder)
        elif method == "visrag":
            top_i = _clip_image_retrieve(cand_imgs, question, top_k,
                                         clip_model, clip_processor, device)
        else:
            raise ValueError(f"Unknown method: {method}")

        if pos_loaded_indices:
            hits = len(set(top_i) & set(pos_loaded_indices))
            recall = hits / len(pos_loaded_indices)

        retrieved_imgs   = [cand_imgs[i]   for i in top_i]
        retrieved_titles = [cand_titles[i] for i in top_i]
        pred, n_tokens = generator.generate_with_images(question, retrieved_imgs, retrieved_titles)

    return pred, answers, recall, n_tokens


def _eval_text_sample(
    sample: Dict,
    generator: LLaVAGenerator,
    top_k: int,
    method: str,           # "bm25", "dense", or "latent"
    dense_encoder=None,
    compressor=None,
) -> Optional[Tuple[str, List[str], float, int]]:
    """
    Retrieval + generation for one text-modality sample.
    Returns (prediction, references, recall@k, n_tokens) or None if candidate pool is empty.
    """
    from src.llava_compressor import LLaVACompressor

    pos_facts = list(sample.get("txt_pos_facts", []))
    neg_facts = list(sample.get("txt_neg_facts", []))
    pos_evidence = list(sample.get("txt_pos_evidence", pos_facts))
    neg_evidence = list(sample.get("txt_neg_evidence", neg_facts))
    all_facts = pos_facts + neg_facts
    all_evidence = pos_evidence + neg_evidence
    question  = sample["question"]
    answers   = sample["answers"]
    if isinstance(answers, str):
        answers = [answers]

    if not all_facts:
        return None

    recall   = 0.0
    n_tokens = 0

    if method == "full_context":
        recall = 1.0 if pos_evidence else 0.0
        pred, n_tokens = generator.generate_with_text(question, all_evidence)
        return pred, answers, recall, n_tokens

    if method == "latent":
        fact_prompts = [LLaVACompressor.format_compress_prompt(f, False) for f in all_facts]
        fact_latents = compressor.compress_batch(fact_prompts, with_grad=False, adapter="compress")
        fact_embs    = compressor.get_retrieval_embedding(fact_latents)
        q_lat        = compressor.embed_query_batch([question], with_grad=False)
        q_emb        = compressor.get_retrieval_embedding(q_lat)
        sims         = (fact_embs @ q_emb.T).squeeze(-1)
        k            = min(top_k, len(all_facts))
        top_i        = torch.topk(sims, k=k).indices.tolist()

        if pos_facts:
            pos_set = set(range(len(pos_facts)))
            recall  = len(set(top_i) & pos_set) / len(pos_set)

        top_latents = fact_latents[top_i]
        projected   = compressor.project_for_generator(top_latents)
        try:
            pred, n_tokens = generator.generate_with_latents(question, projected)
        except Exception as e:
            logger.warning(f"Latent text generation failed: {e}")
            pred, n_tokens = "", 0
    else:
        if method == "bm25":
            top_i = _bm25_retrieve(all_facts, question, top_k)
        elif method == "dense":
            top_i = _dense_retrieve(all_facts, question, top_k, dense_encoder)
        else:
            raise ValueError(f"Unsupported text retrieval method: {method}")

        if pos_facts:
            pos_set = set(range(len(pos_facts)))
            recall  = len(set(top_i) & pos_set) / len(pos_set)

        retrieved_facts = [all_facts[i] for i in top_i]
        pred, n_tokens = generator.generate_with_text(question, retrieved_facts)

    return pred, answers, recall, n_tokens


def _build_image_prompt_text(question: str, titles: List[str]) -> str:
    lines = [f"Evidence {i + 1}: <image>\nTitle: {title}" for i, title in enumerate(titles)]
    return "USER: " + "\n".join(lines) + f"\nQuestion: {question}\nASSISTANT:"


def _build_text_prompt_text(question: str, facts: List[str]) -> str:
    lines = [f"Evidence {i + 1}: {fact}" for i, fact in enumerate(facts)]
    return "USER: " + "\n".join(lines) + f"\nQuestion: {question}\nASSISTANT:"


def _serialize_text_facts(facts: List[str]) -> List[Dict]:
    return [{"kind": "text", "text": fact} for fact in facts]


def _serialize_image_titles(titles: List[str]) -> List[Dict]:
    return [{"kind": "image", "title": title} for title in titles]


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
    mods = acc["mods"]
    meta = acc["meta"]
    if not preds:
        return {}
    m = evaluate(preds, refs, tokens, sample_metadata=meta)
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
    clip_model=None,
    clip_processor=None,
    compressor=None,
    pool: Optional[List[Dict]] = None,
    dump_examples: int = 0,
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
        key: {"preds": [], "refs": [], "recalls": [], "precisions": [], "tokens": [], "mods": [], "meta": []}
        for key in method_keys
    }
    dumped = {key: [] for key in method_keys}

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
            try:
                pred, n_tok = generator.generate_with_evidence(question, full_context_candidates)
            except Exception as e:
                logger.warning(f"full_context generation failed: {e}")
                pred, n_tok = "", 0
            acc["full_context"]["preds"].append(pred)
            acc["full_context"]["refs"].append(answers)
            acc["full_context"]["recalls"].append(1.0 if full_pos_indices else 0.0)
            acc["full_context"]["precisions"].append(precision)
            acc["full_context"]["tokens"].append(n_tok)
            acc["full_context"]["mods"].append(modality)
            acc["full_context"]["meta"].append(meta)
            if dump_examples > 0 and len(dumped["full_context"]) < dump_examples:
                dumped["full_context"].append({
                    "sample_id": s.get("id"),
                    "modality": modality,
                    "question": question,
                    "answers": answers,
                    "retrieved": _serialize_evidences(full_context_candidates),
                    "prompt": generator.build_evidence_prompt_text(question, full_context_candidates),
                    "prediction": pred,
                })

        ranked_indices: Dict[str, List[int]] = {}
        search_corpus = [c["search_text"] for c in candidates]
        if "bm25" in methods:
            ranked_indices["bm25"] = _bm25_retrieve(search_corpus, question, max_k)
        if "dense" in methods:
            ranked_indices["dense"] = _dense_retrieve(search_corpus, question, max_k, dense_encoder)
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
            text_items = [c for c in candidates if c["kind"] == "text"]
            image_items = [c for c in candidates if c["kind"] == "image"]
            latent_chunks = []
            emb_chunks = []
            if text_items:
                text_prompts = [c["text"] for c in text_items]
                text_latents = compressor.compress_batch(
                    [type(compressor).format_compress_prompt(t, False) for t in text_prompts],
                    with_grad=False,
                    adapter="compress",
                )
                latent_chunks.append(text_latents)
                emb_chunks.append(compressor.get_retrieval_embedding(text_latents))
            if image_items:
                img_latents = compressor.compress_batch(
                    [type(compressor).format_compress_prompt(c["title"], True) for c in image_items],
                    images=[c["image"] for c in image_items],
                    with_grad=False,
                    adapter="compress",
                )
                latent_chunks.append(img_latents)
                emb_chunks.append(compressor.get_retrieval_embedding(img_latents))
            if latent_chunks:
                cand_latents = torch.cat(latent_chunks, dim=0)
                cand_embs = torch.cat(emb_chunks, dim=0)
                q_lat = compressor.embed_query_batch([question], with_grad=False)
                q_emb = compressor.get_retrieval_embedding(q_lat)
                sims = (cand_embs @ q_emb.T).squeeze(-1)
                latent_ranked = torch.topk(sims, k=min(max_k, cand_latents.shape[0])).indices.tolist()
                latent_projected = compressor.project_for_generator(cand_latents)
            else:
                latent_ranked = []

        for method in methods:
            if method == "full_context":
                continue
            for k in top_k_values:
                key = f"{method}_k{k}"
                if method == "latent":
                    top_i = latent_ranked[: min(k, len(latent_ranked))]
                    payload = latent_projected[top_i] if top_i else None
                else:
                    top_i = ranked_indices.get(method, [])[: min(k, len(ranked_indices.get(method, [])))]
                    payload = [candidates[i] for i in top_i]
                if not top_i:
                    continue

                hits = len(set(top_i) & pos_indices)
                recall = hits / len(pos_indices) if pos_indices else 0.0
                precision = hits / len(top_i) if top_i else 0.0
                try:
                    if method == "latent":
                        pred, n_tok = generator.generate_with_latents(question, payload)
                    else:
                        pred, n_tok = generator.generate_with_evidence(question, payload)
                except Exception as e:
                    logger.warning(f"{key} generation failed: {e}")
                    pred, n_tok = "", 0

                acc[key]["preds"].append(pred)
                acc[key]["refs"].append(answers)
                acc[key]["recalls"].append(recall)
                acc[key]["precisions"].append(precision)
                acc[key]["tokens"].append(n_tok)
                acc[key]["mods"].append(modality)
                acc[key]["meta"].append(meta)
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
                        "prediction": pred,
                    })

    results = {key: _finalize_method_metrics(val) for key, val in acc.items()}
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
# Per-modality evaluation loops
# ---------------------------------------------------------------------------

def run_image_baselines(
    val_image:       List[Dict],
    generator:       LLaVAGenerator,
    top_k:           int,
    scale:           float,
    max_samples:     int,
    image_methods:   List[str],        # subset of ["bm25", "dense", "visrag", "latent"]
    dense_model_name: str,
    clip_model_name:  str,
    device:          str,
    compressor=None,
    pool: Optional[List[Dict]] = None,
    dense_encoder=None,
    clip_model_obj=None,
    clip_proc=None,
    dump_examples: int = 0,
) -> Tuple[Dict[str, Dict], Dict[str, List[Dict]]]:
    """
    Evaluate image-retrieval baselines on image-modality samples.

    image_methods may include "bm25", "dense", "visrag", and/or "latent".
    Returns {method: metrics_dict}.
    """
    # ── Load retrieval models ──────────────────────────────────────────────
    if "dense" in image_methods and dense_encoder is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading dense encoder: {dense_model_name}")
        dense_encoder = SentenceTransformer(dense_model_name)

    if "visrag" in image_methods and (clip_model_obj is None or clip_proc is None):
        from transformers import CLIPModel, CLIPProcessor
        logger.info(f"Loading CLIP model for VisRAG: {clip_model_name}")
        clip_proc      = CLIPProcessor.from_pretrained(clip_model_name)
        clip_model_obj = CLIPModel.from_pretrained(
            clip_model_name, torch_dtype=torch.float32
        ).to(device)
        clip_model_obj.eval()

    # ── Fixed-shuffle val subset ───────────────────────────────────────────
    if pool is None:
        pool = _prepare_eval_pool(val_image, max_samples)

    # Accumulators per method
    acc: Dict[str, Dict] = {m: {"preds": [], "refs": [], "recalls": [], "tokens": [], "meta": []}
                             for m in image_methods}
    dumped: Dict[str, List[Dict]] = {m: [] for m in image_methods}

    for s in tqdm(pool, desc="image-baselines", leave=False):
        img_dir = s.get("image_dir", "")
        pos_ids = s.get("pos_image_ids", [])
        neg_ids = s.get("neg_image_ids", [])
        all_ids = pos_ids + neg_ids
        all_caps = s.get("pos_captions", []) + s.get("neg_captions", [])
        cand_imgs = []
        cand_titles = []
        pos_loaded_indices = []
        pos_id_set = set(pos_ids)
        for img_id, cap in zip(all_ids, all_caps):
            img = _load_image(img_dir, img_id, scale)
            if img is not None:
                if img_id in pos_id_set:
                    pos_loaded_indices.append(len(cand_imgs))
                cand_imgs.append(img)
                cand_titles.append(cap)
        for method in image_methods:
            result = _eval_image_sample(
                s, generator, top_k, scale, method,
                dense_encoder=dense_encoder,
                clip_model=clip_model_obj,
                clip_processor=clip_proc,
                compressor=compressor,
                device=device,
            )
            if result is None:
                continue
            pred, refs, recall, n_tok = result
            acc[method]["preds"].append(pred)
            acc[method]["refs"].append(refs)
            acc[method]["recalls"].append(recall)
            acc[method]["tokens"].append(n_tok)
            acc[method]["meta"].append({
                "source": s.get("source", "webqa"),
                "modality": s.get("modality", "image"),
                "webqa_qcate": s.get("webqa_qcate"),
                "webqa_keywords": s.get("webqa_keywords"),
            })
            if dump_examples > 0 and len(dumped[method]) < dump_examples:
                if method == "full_context":
                    retrieved_titles = cand_titles
                elif method == "bm25":
                    top_i = _bm25_retrieve(cand_titles, s["question"], top_k)
                    retrieved_titles = [cand_titles[i] for i in top_i]
                elif method == "dense":
                    top_i = _dense_retrieve(cand_titles, s["question"], top_k, dense_encoder)
                    retrieved_titles = [cand_titles[i] for i in top_i]
                elif method == "visrag":
                    top_i = _clip_image_retrieve(cand_imgs, s["question"], top_k, clip_model_obj, clip_proc, device)
                    retrieved_titles = [cand_titles[i] for i in top_i]
                elif method == "latent":
                    retrieved_titles = cand_titles
                else:
                    retrieved_titles = cand_titles[:top_k]
                dumped[method].append({
                    "sample_id": s.get("id"),
                    "modality": "image",
                    "question": s["question"],
                    "answers": refs,
                    "retrieved": _serialize_image_titles(retrieved_titles),
                    "prompt": _build_image_prompt_text(s["question"], retrieved_titles),
                    "prediction": pred,
                })

    # ── Compute metrics ────────────────────────────────────────────────────
    results: Dict[str, Dict] = {}
    for method in image_methods:
        preds   = acc[method]["preds"]
        refs    = acc[method]["refs"]
        recalls = acc[method]["recalls"]
        tokens  = acc[method]["tokens"]
        meta    = acc[method]["meta"]
        if not preds:
            logger.warning(f"image/{method}: no predictions.")
            results[method] = {}
            continue
        m = evaluate(preds, refs, tokens, sample_metadata=meta)
        m["recall_at_k"] = sum(recalls) / len(recalls) if recalls else 0.0
        m["n"]           = len(preds)
        acc_str = f"  Acc={m['acc']:.4f}" if "acc" in m else ""
        logger.info(
            f"IMAGE {method.upper()} | n={len(preds)} | "
            f"EM={m['em']:.4f}  F1={m['f1']:.4f}  ROUGE-L={m.get('rouge_l', 0):.4f}  "
            f"Recall@{top_k}={m['recall_at_k']:.4f}{acc_str}"
        )
        results[method] = m

    return results, dumped


def run_text_baselines(
    val_text:        List[Dict],
    generator:       LLaVAGenerator,
    top_k:           int,
    max_samples:     int,
    text_methods:    List[str],        # ["bm25", "dense", "latent"]
    dense_model_name: str,
    device:          str,
    compressor=None,
    pool: Optional[List[Dict]] = None,
    dense_encoder=None,
    dump_examples: int = 0,
) -> Tuple[Dict[str, Dict], Dict[str, List[Dict]]]:
    """
    Evaluate text-retrieval baselines on text-modality samples.

    text_methods may include "bm25", "dense", and/or "latent".
    VisRAG is NOT applicable to text samples.
    Returns {method: metrics_dict}.
    """
    if "dense" in text_methods and dense_encoder is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading dense encoder: {dense_model_name}")
        dense_encoder = SentenceTransformer(dense_model_name)

    if pool is None:
        pool = _prepare_eval_pool(val_text, max_samples)

    acc: Dict[str, Dict] = {m: {"preds": [], "refs": [], "recalls": [], "tokens": [], "meta": []}
                             for m in text_methods}
    dumped: Dict[str, List[Dict]] = {m: [] for m in text_methods}

    for s in tqdm(pool, desc="text-baselines", leave=False):
        pos_facts = list(s.get("txt_pos_facts", []))
        neg_facts = list(s.get("txt_neg_facts", []))
        pos_evidence = list(s.get("txt_pos_evidence", pos_facts))
        neg_evidence = list(s.get("txt_neg_evidence", neg_facts))
        all_facts = pos_facts + neg_facts
        all_evidence = pos_evidence + neg_evidence
        for method in text_methods:
            result = _eval_text_sample(
                s, generator, top_k, method,
                dense_encoder=dense_encoder,
                compressor=compressor,
            )
            if result is None:
                continue
            pred, refs, recall, n_tok = result
            acc[method]["preds"].append(pred)
            acc[method]["refs"].append(refs)
            acc[method]["recalls"].append(recall)
            acc[method]["tokens"].append(n_tok)
            acc[method]["meta"].append({
                "source": s.get("source", "webqa"),
                "modality": s.get("modality", "text"),
                "webqa_qcate": s.get("webqa_qcate"),
                "webqa_keywords": s.get("webqa_keywords"),
            })
            if dump_examples > 0 and len(dumped[method]) < dump_examples:
                if method == "full_context":
                    retrieved_facts = all_evidence
                elif method == "bm25":
                    top_i = _bm25_retrieve(all_facts, s["question"], top_k)
                    retrieved_facts = [all_facts[i] for i in top_i]
                elif method == "dense":
                    top_i = _dense_retrieve(all_facts, s["question"], top_k, dense_encoder)
                    retrieved_facts = [all_facts[i] for i in top_i]
                elif method == "latent":
                    retrieved_facts = all_facts
                else:
                    retrieved_facts = all_facts[:top_k]
                dumped[method].append({
                    "sample_id": s.get("id"),
                    "modality": "text",
                    "question": s["question"],
                    "answers": refs,
                    "retrieved": _serialize_text_facts(retrieved_facts),
                    "prompt": _build_text_prompt_text(s["question"], retrieved_facts),
                    "prediction": pred,
                })

    results: Dict[str, Dict] = {}
    for method in text_methods:
        preds   = acc[method]["preds"]
        refs    = acc[method]["refs"]
        recalls = acc[method]["recalls"]
        tokens  = acc[method]["tokens"]
        meta    = acc[method]["meta"]
        if not preds:
            logger.warning(f"text/{method}: no predictions.")
            results[method] = {}
            continue
        m = evaluate(preds, refs, tokens, sample_metadata=meta)
        m["recall_at_k"] = sum(recalls) / len(recalls) if recalls else 0.0
        m["n"]           = len(preds)
        acc_str = f"  Acc={m['acc']:.4f}" if "acc" in m else ""
        logger.info(
            f"TEXT  {method.upper()} | n={len(preds)} | "
            f"EM={m['em']:.4f}  F1={m['f1']:.4f}  ROUGE-L={m.get('rouge_l', 0):.4f}  "
            f"Recall@{top_k}={m['recall_at_k']:.4f}{acc_str}"
        )
        results[method] = m

    return results, dumped


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


def _save_text_report(
    output_path: str,
    img_results: Dict[str, Dict],
    txt_results: Dict[str, Dict],
    top_k_values: List[int],
    max_samples: int,
) -> None:
    lines = [
        "WebQA LLaVA Baseline Results",
        f"top_k_values={top_k_values}  max_samples={max_samples}",
        "",
    ]
    if img_results:
        lines.append(_format_table(
            f"IMAGE retrieval baselines  (k∈{top_k_values}, n≤{max_samples})",
            img_results, top_k_values[-1],
        ))
        lines.append("")
    if txt_results:
        lines.append(_format_table(
            f"TEXT  retrieval baselines  (k∈{top_k_values}, n≤{max_samples})",
            txt_results, top_k_values[-1],
        ))
        lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Text report saved: {output_path}")


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


def main(args):
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
    for s in val_all:
        if "modality" not in s:
            s["modality"] = "image" if s.get("pos_image_ids") else "text"
    val_image = [s for s in val_all if s["modality"] == "image"]
    val_text  = [s for s in val_all if s["modality"] == "text"]
    logger.info(f"  val image: {len(val_image)}  val text: {len(val_text)}")

    # ── Parse requested methods ────────────────────────────────────────────
    requested = [m.lower() for m in args.methods]
    valid = {"bm25", "dense", "visrag", "latent", "full_context"}
    for m in requested:
        if m not in valid:
            logger.warning(f"Unknown method '{m}' — ignoring.")

    # full_context runs once (k is irrelevant); retrieval methods run per k
    fc_requested      = "full_context" in requested
    retrieval_methods = [m for m in requested if m in valid and m != "full_context"]

    image_retrieval = [m for m in retrieval_methods if m in valid]
    text_retrieval  = [m for m in retrieval_methods if m in ("bm25", "dense", "latent")]

    if not fc_requested and not retrieval_methods:
        raise ValueError("No valid methods specified.")

    if "latent" in requested and not args.checkpoint:
        raise ValueError("--checkpoint is required for the 'latent' method.")

    # ── Load generator once (shared by all methods and all k) ─────────────
    generator = LLaVAGenerator(gen_name, device=device, max_new_tokens=max_new_tokens)
    val_image_pool = _prepare_eval_pool(val_image, args.max_samples) if val_image else []
    val_text_pool = _prepare_eval_pool(val_text, args.max_samples) if val_text else []

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
        clip_model=clip_model_obj,
        clip_processor=clip_proc,
        compressor=compressor,
        pool=unified_pool,
        dump_examples=args.dump_examples,
    )

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

    # ── Run evaluations ────────────────────────────────────────────────────
    # Results keyed as "method_k{k}" for retrieval methods, "full_context" for fc.
    img_results: Dict[str, Dict] = {}
    txt_results: Dict[str, Dict] = {}
    dumped_examples: Dict[str, Dict[str, List[Dict]]] = {"image": {}, "text": {}}

    # full_context — run once
    if fc_requested:
        if val_image:
            r, dumps = run_image_baselines(
                val_image, generator, top_k_values[-1], scale, args.max_samples,
                ["full_context"], dense_model, clip_model, device,
                pool=val_image_pool,
                dense_encoder=dense_encoder,
                clip_model_obj=clip_model_obj,
                clip_proc=clip_proc,
                dump_examples=args.dump_examples,
            )
            img_results["full_context"] = r.get("full_context", {})
            if args.dump_examples > 0:
                dumped_examples["image"]["full_context"] = dumps.get("full_context", [])
        if val_text:
            r, dumps = run_text_baselines(
                val_text, generator, top_k_values[-1], args.max_samples,
                ["full_context"], dense_model, device,
                pool=val_text_pool,
                dense_encoder=dense_encoder,
                dump_examples=args.dump_examples,
            )
            txt_results["full_context"] = r.get("full_context", {})
            if args.dump_examples > 0:
                dumped_examples["text"]["full_context"] = dumps.get("full_context", [])

    # retrieval methods — one pass per k value
    for k in top_k_values:
        logger.info(f"── k={k} ──────────────────────────────────────────")

        if val_image and image_retrieval:
            r, dumps = run_image_baselines(
                val_image, generator, k, scale, args.max_samples,
                image_retrieval, dense_model, clip_model, device,
                compressor=compressor,
                pool=val_image_pool,
                dense_encoder=dense_encoder,
                clip_model_obj=clip_model_obj,
                clip_proc=clip_proc,
                dump_examples=args.dump_examples,
            )
            for method, metrics in r.items():
                img_results[f"{method}_k{k}"] = metrics
                if args.dump_examples > 0:
                    dumped_examples["image"][f"{method}_k{k}"] = dumps.get(method, [])

        if val_text and text_retrieval:
            r, dumps = run_text_baselines(
                val_text, generator, k, args.max_samples,
                text_retrieval, dense_model, device,
                compressor=compressor,
                pool=val_text_pool,
                dense_encoder=dense_encoder,
                dump_examples=args.dump_examples,
            )
            for method, metrics in r.items():
                txt_results[f"{method}_k{k}"] = metrics
                if args.dump_examples > 0:
                    dumped_examples["text"][f"{method}_k{k}"] = dumps.get(method, [])

    # ── Print tables ───────────────────────────────────────────────────────
    if img_results:
        _print_table(
            f"IMAGE baselines  (k∈{top_k_values}, n≤{args.max_samples})",
            img_results, top_k_values[-1],
        )
    if txt_results:
        _print_table(
            f"TEXT  baselines  (k∈{top_k_values}, n≤{args.max_samples})",
            txt_results, top_k_values[-1],
        )

    # ── Save results ───────────────────────────────────────────────────────
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "top_k_values": top_k_values,
                "image":        img_results,
                "text":         txt_results,
            }, f, indent=2)
        logger.info(f"Saved: {args.output}")
        if args.dump_examples > 0:
            dump_path = args.output.replace(".json", ".examples.json")
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(dumped_examples, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved examples: {dump_path}")

        txt_path = args.output.replace(".json", ".txt")
        _save_text_report(txt_path, img_results, txt_results, top_k_values, args.max_samples)

    return {"image": img_results, "text": txt_results}


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
    parser.add_argument("--dump_examples", type=int, default=0,
                        help="Dump first N examples per method to a sidecar JSON")
    parser.add_argument("--debug_token_counts", action="store_true",
                        help="Only estimate unified full-context and first-k token counts, then exit.")
    parser.add_argument("--output",       default="results/baselines_llava.json",
                        help="Path to save JSON results")
    args = parser.parse_args()
    main(args)

"""
WebQA Qwen Baselines
====================
Unified retrieval baselines evaluated against a frozen Qwen2.5-VL generator on
WebQA. Candidate pools contain both text facts and image contexts, matching the
current Qwen validation setup.

Supports:
  1. full_context
  2. bm25
  3. dense
  4. latent
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
from transformers import AutoConfig, AutoProcessor

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from data.prepare_webqa import load_webqa_samples
from src.evaluation import evaluate
from src.distillation import _split_chat_template, _tok
from src.qwen_compressor import QwenVLCompressor, resolve_qwen_vl_model_class
from src.region_encoder import resize_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class QwenGenerator:
    def __init__(self, model_name: str, device: str = "cuda", max_new_tokens: int = 64):
        logger.info(f"Loading Qwen VL generator: {model_name}")
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_type = getattr(cfg, "model_type", None)
        if model_type not in {"qwen2_vl", "qwen2_5_vl"}:
            raise ValueError(
                f"scripts/baselines_qwen.py requires a Qwen-VL generator, got "
                f"model_type={model_type} from {model_name}."
            )
        model_cls = resolve_qwen_vl_model_class(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"
        self.model = model_cls.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.max_new_tokens = max_new_tokens

    def _effective_multimodal_length(self, enc, num_images: int) -> int:
        if num_images <= 0 or "image_grid_thw" not in enc:
            return int(enc["input_ids"].shape[-1])
        grid = enc["image_grid_thw"]
        merge_size = getattr(getattr(self.processor, "image_processor", None), "merge_size", None)
        if merge_size is None:
            merge_size = getattr(getattr(self.model.config, "vision_config", None), "spatial_merge_size", 2)
        merge_tokens = int((grid.prod(dim=-1) // (merge_size ** 2)).sum().item())
        return int(enc["input_ids"].shape[-1]) - int(num_images) + merge_tokens

    def _build_latent_seq_embeds(self, question: str, proj_tokens: torch.Tensor) -> torch.Tensor:
        embed_fn = self.model.get_input_embeddings()
        dtype = next(self.model.parameters()).dtype
        tok = self.processor.tokenizer

        def _e(text: str, max_len: int = 512):
            return embed_fn(_tok(tok, text, self.device, max_len)).to(dtype)

        prefix_str, suffix_str = _split_chat_template(tok, question)
        parts = [_e(prefix_str)]
        for i, lat in enumerate(proj_tokens):
            label = f"Latent context {i + 1}: " if i == 0 else f"\nLatent context {i + 1}: "
            parts.append(_e(label))
            parts.append(lat.reshape(-1, lat.shape[-1]).to(dtype).unsqueeze(0).to(self.device))
        parts.append(_e(suffix_str))
        return torch.cat(parts, dim=1).squeeze(0)

    def _build_evidence_prompt(self, question: str, evidences: List[Dict]) -> Tuple[str, List[Image.Image]]:
        # Align raw-evidence prompting with the teacher-side wording/order used in
        # Qwen distillation: image first, then "Context/Title" text.
        content = []
        images = []
        for i, ev in enumerate(evidences):
            if ev["kind"] == "image":
                content.append({"type": "text", "text": f"Context {i + 1}:\n"})
                content.append({"type": "image", "image": ev["image"]})
                images.append(ev["image"])
                content.append({"type": "text", "text": f"Title: {ev['title']}\n"})
            else:
                content.append({"type": "text", "text": f"Context {i + 1}: {ev['text']}\n"})
        content.append({"type": "text", "text": f"Question: {question}"})
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt, images

    def build_latent_prompt_text(self, question: str, proj_tokens: torch.Tensor) -> str:
        parts = []
        prefix_str, suffix_str = _split_chat_template(self.processor.tokenizer, question)
        parts.append(prefix_str)
        for i in range(proj_tokens.shape[0]):
            n_slots = proj_tokens[i].numel() // proj_tokens.shape[-1]
            suffix = "" if n_slots == 1 else f" x{n_slots}"
            label = f"Latent context {i + 1}: [LATENT_{i + 1}{suffix}]"
            parts.append(label if i == 0 else f"\n{label}")
        parts.append(suffix_str)
        return "".join(parts)

    def build_evidence_prompt_text(self, question: str, evidences: List[Dict]) -> str:
        prompt, _ = self._build_evidence_prompt(question, evidences)
        return prompt

    @torch.no_grad()
    def generate_with_latents(self, question: str, proj_tokens: torch.Tensor) -> Tuple[str, int]:
        tok = self.processor.tokenizer
        input_embeds = self._build_latent_seq_embeds(question, proj_tokens).unsqueeze(0)
        seq_len = input_embeds.shape[1]
        out = self.model.generate(
            inputs_embeds=input_embeds,
            attention_mask=torch.ones(1, seq_len, dtype=torch.long, device=self.device),
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        gen_ids = out[0, seq_len:] if out.shape[1] > seq_len else out[0]
        return tok.decode(gen_ids, skip_special_tokens=True).strip(), seq_len

    @torch.no_grad()
    def generate_with_evidence(self, question: str, evidences: List[Dict]) -> Tuple[str, int]:
        if not evidences:
            return "", 0
        prompt, images = self._build_evidence_prompt(question, evidences)
        if images:
            enc = self.processor(text=prompt, images=images, return_tensors="pt").to(self.device)
        else:
            enc = self.processor(text=prompt, return_tensors="pt").to(self.device)
        n_in = self._effective_multimodal_length(enc, len(images))
        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        raw_prompt_len = enc["input_ids"].shape[-1]
        gen_ids = out[0, raw_prompt_len:] if out.shape[1] > raw_prompt_len else out[0]
        return self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip(), n_in

    @torch.no_grad()
    def batch_generate_with_latents(self, questions: List[str], projected_list: List[torch.Tensor]) -> Tuple[List[str], List[int]]:
        if not projected_list:
            return [], []
        tok = self.processor.tokenizer
        seq_embeds = [self._build_latent_seq_embeds(q, proj) for q, proj in zip(questions, projected_list)]
        batch_size = len(seq_embeds)
        max_len = max(e.shape[0] for e in seq_embeds)
        hidden = seq_embeds[0].shape[-1]
        dtype = seq_embeds[0].dtype
        padded = torch.zeros(batch_size, max_len, hidden, dtype=dtype, device=self.device)
        attn_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        position_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        for i, emb in enumerate(seq_embeds):
            seq_len = emb.shape[0]
            padded[i, max_len - seq_len:] = emb
            attn_mask[i, max_len - seq_len:] = 1
            position_ids[i, max_len - seq_len:] = torch.arange(seq_len, device=self.device)
        out = self.model.generate(
            inputs_embeds=padded,
            attention_mask=attn_mask,
            position_ids=position_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        answers = []
        lengths = [int(mask.sum().item()) for mask in attn_mask]
        for i in range(batch_size):
            gen_ids = out[i, max_len:] if out.shape[1] > max_len else out[i]
            answers.append(tok.decode(gen_ids, skip_special_tokens=True).strip())
        return answers, lengths

    @torch.no_grad()
    def batch_generate_with_evidence(self, questions: List[str], evidences_batch: List[List[Dict]]) -> Tuple[List[str], List[int]]:
        if not evidences_batch:
            return [], []
        prompts, images_batch = zip(*(self._build_evidence_prompt(q, evs) for q, evs in zip(questions, evidences_batch)))
        has_any_images = any(len(imgs) > 0 for imgs in images_batch)
        if has_any_images:
            answers, lengths = [], []
            for q, evs in zip(questions, evidences_batch):
                ans, n_tok = self.generate_with_evidence(q, evs)
                answers.append(ans)
                lengths.append(n_tok)
            return answers, lengths
        try:
            enc = self.processor(text=list(prompts), padding=True, return_tensors="pt").to(self.device)
            lengths = enc["attention_mask"].sum(dim=1).tolist()
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
            max_raw_len = enc["input_ids"].shape[1]
            answers = []
            for i in range(len(prompts)):
                gen_ids = out[i, max_raw_len:] if out.shape[1] > max_raw_len else out[i]
                answers.append(self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
            return answers, [int(x) for x in lengths]
        except Exception as exc:
            logger.warning(f"Batch evidence generation failed, falling back to serial: {exc}")
            answers, lengths = [], []
            for q, evs in zip(questions, evidences_batch):
                ans, n_tok = self.generate_with_evidence(q, evs)
                answers.append(ans)
                lengths.append(n_tok)
            return answers, lengths


def _bm25_retrieve(corpus: List[str], query: str, top_k: int) -> List[int]:
    from rank_bm25 import BM25Okapi
    tokenized = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    k = min(top_k, len(corpus))
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]


def _dense_retrieve(corpus: List[str], query: str, top_k: int, encoder) -> List[int]:
    doc_embs = encoder.encode(corpus, convert_to_tensor=True, normalize_embeddings=True)
    q_emb = encoder.encode([query], convert_to_tensor=True, normalize_embeddings=True)
    scores = (doc_embs @ q_emb.T).squeeze(-1).cpu().numpy()
    k = min(top_k, len(corpus))
    return list(map(int, (-scores).argsort()[:k]))


def _prepare_eval_pool(samples: List[Dict], max_samples: int) -> List[Dict]:
    pool = list(samples)
    random.Random(42).shuffle(pool)
    if max_samples > 0:
        pool = pool[:max_samples]
    return pool


def _load_image(img_dir: str, img_id: str, scale: float) -> Optional[Image.Image]:
    path = os.path.join(img_dir, f"{img_id}.jpg")
    if not os.path.exists(path):
        return None
    try:
        return resize_image(Image.open(path).convert("RGB"), scale)
    except Exception:
        return None


def _ensure_title_sentence(sentence: str, title: str = "") -> str:
    sent = (sentence or "").strip()
    title = (title or "").strip()
    if not sent:
        return ""
    if title and not sent.startswith(f"{title}:"):
        return f"{title}: {sent}"
    return sent


def _get_text_retrieval_units(sample: Dict) -> Tuple[List[str], List[int]]:
    """
    Build text retrieval units from WebQA facts.

    BM25, dense, and latent baselines should all use the same text pool:
    `txt_pos_facts + txt_neg_facts`. Do not use `sentences` here; that field can
    come from other text pipelines and may not preserve the WebQA fact formatting
    expected by these multimodal baselines.
    """
    txt_pos = list(sample.get("txt_pos_facts", []))
    txt_neg = list(sample.get("txt_neg_facts", []))
    units = [x for x in (txt_pos + txt_neg) if (x or "").strip()]
    return units, list(range(len([x for x in txt_pos if (x or "").strip()])))


def _build_sample_candidates(sample: Dict, scale: float) -> Optional[Dict]:
    question = sample.get("question", "")
    answers = sample.get("answers", [])
    if isinstance(answers, str):
        answers = [answers]
    if not question or not answers:
        return None

    txt_pos = list(sample.get("txt_pos_facts", []))
    txt_neg = list(sample.get("txt_neg_facts", []))
    text_retrieval_units, text_retrieval_pos = _get_text_retrieval_units(sample)
    txt_pos_evidence = list(sample.get("txt_pos_evidence", txt_pos))
    txt_neg_evidence = list(sample.get("txt_neg_evidence", txt_neg))
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
        if idx < len(txt_pos_evidence) and sample.get("modality") == "text":
            full_pos_indices.append(len(full_context_candidates))
        full_context_candidates.append({"kind": "text", "text": fact, "search_text": fact})

    for idx, fact in enumerate(text_retrieval_units):
        if idx in set(text_retrieval_pos) and sample.get("modality") == "text":
            retrieval_pos_indices.append(len(retrieval_candidates))
        retrieval_candidates.append({"kind": "text", "text": fact, "search_text": fact})

    for idx, (img_id, cap) in enumerate(zip(all_img_ids, all_caps)):
        img = _load_image(img_dir, img_id, scale)
        if img is None:
            continue
        if idx < len(pos_ids) and sample.get("modality") == "image":
            full_pos_indices.append(len(full_context_candidates))
            retrieval_pos_indices.append(len(retrieval_candidates))
        image_candidate = {"kind": "image", "image": img, "title": cap, "search_text": cap}
        full_context_candidates.append(image_candidate)
        retrieval_candidates.append(image_candidate)

    if not full_context_candidates:
        return None

    return {
        "id": sample.get("id", sample.get("qid", sample.get("question_id", None))),
        "source": sample.get("source", "webqa"),
        "modality": sample.get("modality", "image" if pos_ids else "text"),
        "question": question,
        "answers": answers,
        "webqa_qcate": sample.get("webqa_qcate"),
        "webqa_keywords": sample.get("webqa_keywords"),
        "full_context_candidates": full_context_candidates,
        "retrieval_candidates": retrieval_candidates,
        "full_pos_indices": full_pos_indices,
        "pos_indices": retrieval_pos_indices,
    }


def _serialize_evidences(evidences: List[Dict]) -> List[Dict]:
    out = []
    for ev in evidences:
        if ev["kind"] == "image":
            out.append({"kind": "image", "title": ev["title"]})
        else:
            out.append({"kind": "text", "text": ev["text"]})
    return out


def _eval_sample(
    prepared: Dict,
    generator: QwenGenerator,
    top_k: int,
    method: str,
    dense_encoder=None,
    compressor: Optional[QwenVLCompressor] = None,
) -> Optional[Tuple[str, List[str], float, int]]:
    question = prepared["question"]
    answers = prepared["answers"]
    full_context_candidates = prepared["full_context_candidates"]
    candidates = prepared["retrieval_candidates"]
    pos_indices = set(prepared["pos_indices"])
    full_pos_indices = set(prepared["full_pos_indices"])

    if method == "full_context":
        pred, n_tok = generator.generate_with_evidence(question, full_context_candidates)
        recall = 1.0 if full_pos_indices else 0.0
        return pred, answers, recall, n_tok

    if method == "latent":
        if compressor is None:
            raise ValueError("latent baseline requires a Qwen compressor checkpoint")
        text_items = [c for c in candidates if c["kind"] == "text"]
        image_items = [c for c in candidates if c["kind"] == "image"]
        latent_chunks = []
        emb_chunks = []
        item_order = []
        if text_items:
            text_latents = compressor.compress_batch(
                [c["text"] for c in text_items], with_grad=False, adapter="compress"
            )
            latent_chunks.append(text_latents)
            emb_chunks.append(compressor.get_retrieval_embedding(text_latents))
            item_order.extend(text_items)
        if image_items:
            img_latents = compressor.compress_batch(
                [c["title"] for c in image_items],
                images=[c["image"] for c in image_items],
                with_grad=False,
                adapter="compress",
            )
            latent_chunks.append(img_latents)
            emb_chunks.append(compressor.get_retrieval_embedding(img_latents))
            item_order.extend(image_items)
        cand_latents = torch.cat(latent_chunks, dim=0)
        cand_embs = torch.cat(emb_chunks, dim=0)
        q_lat = compressor.embed_query_batch([question], with_grad=False)
        q_emb = compressor.get_retrieval_embedding(q_lat)
        sims = (cand_embs @ q_emb.T).squeeze(-1)
        k = min(top_k, len(item_order))
        top_i = torch.topk(sims, k=k).indices.tolist()
        recall = len(set(top_i) & pos_indices) / len(pos_indices) if pos_indices else 0.0
        pred, n_tok = generator.generate_with_latents(question, compressor.project_for_generator(cand_latents[top_i]))
        return pred, answers, recall, n_tok

    search_corpus = [c["search_text"] for c in candidates]
    if method == "bm25":
        top_i = _bm25_retrieve(search_corpus, question, top_k)
    elif method == "dense":
        top_i = _dense_retrieve(search_corpus, question, top_k, dense_encoder)
    else:
        raise ValueError(f"Unsupported method: {method}")

    recall = len(set(top_i) & pos_indices) / len(pos_indices) if pos_indices else 0.0
    pred, n_tok = generator.generate_with_evidence(question, [candidates[i] for i in top_i])
    return pred, answers, recall, n_tok


def run_unified_baselines(
    samples: List[Dict],
    generator: QwenGenerator,
    top_k: int,
    scale: float,
    max_samples: int,
    methods: List[str],
    dense_model_name: str,
    dense_encoder=None,
    pool: Optional[List[Dict]] = None,
    compressor: Optional[QwenVLCompressor] = None,
) -> Dict[str, Dict]:
    if "dense" in methods and dense_encoder is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading dense encoder: {dense_model_name}")
        dense_encoder = SentenceTransformer(dense_model_name)
    if pool is None:
        pool = _prepare_eval_pool(samples, max_samples)

    prepared_pool = []
    for s in tqdm(pool, desc="prep-baselines", leave=False):
        prepared = _build_sample_candidates(s, scale)
        if prepared is not None:
            prepared_pool.append(prepared)

    acc = {m: {"preds": [], "refs": [], "recalls": [], "tokens": [], "mods": [], "meta": []} for m in methods}
    for s in tqdm(prepared_pool, desc="unified-baselines", leave=False):
        for method in methods:
            result = _eval_sample(
                s, generator, top_k, method,
                dense_encoder=dense_encoder, compressor=compressor,
            )
            if result is None:
                continue
            pred, refs, recall, n_tok = result
            acc[method]["preds"].append(pred)
            acc[method]["refs"].append(refs)
            acc[method]["recalls"].append(recall)
            acc[method]["tokens"].append(n_tok)
            acc[method]["mods"].append(s["modality"])
            acc[method]["meta"].append({
                "source": s.get("source", "webqa"),
                "modality": s["modality"],
                "webqa_qcate": s.get("webqa_qcate"),
                "webqa_keywords": s.get("webqa_keywords"),
            })

    results = {}
    for method in methods:
        preds = acc[method]["preds"]
        refs = acc[method]["refs"]
        recalls = acc[method]["recalls"]
        tokens = acc[method]["tokens"]
        mods = acc[method]["mods"]
        meta = acc[method]["meta"]
        if not preds:
            results[method] = {}
            continue
        m = evaluate(preds, refs, tokens, sample_metadata=meta)
        m["recall_at_k"] = sum(recalls) / len(recalls) if recalls else 0.0
        m["n"] = len(preds)
        img_preds = [p for p, mod in zip(preds, mods) if mod == "image"]
        img_refs = [r for r, mod in zip(refs, mods) if mod == "image"]
        txt_preds = [p for p, mod in zip(preds, mods) if mod == "text"]
        txt_refs = [r for r, mod in zip(refs, mods) if mod == "text"]
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
        results[method] = m
    return results


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
    generator: QwenGenerator,
    top_k_values: List[int],
    scale: float,
    max_samples: int,
    methods: List[str],
    dense_model_name: str,
    dense_encoder=None,
    pool: Optional[List[Dict]] = None,
    compressor: Optional[QwenVLCompressor] = None,
    gen_batch_size: int = 8,
    latent_query_batch_size: int = 128,
    dump_examples: int = 0,
) -> Dict[str, Dict]:
    if not methods:
        return {}
    if "dense" in methods and dense_encoder is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading dense encoder: {dense_model_name}")
        dense_encoder = SentenceTransformer(dense_model_name)
    if pool is None:
        pool = _prepare_eval_pool(samples, max_samples)

    prepared_pool = []
    for s in tqdm(pool, desc="prep-baselines", leave=False):
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
    pending = {key: [] for key in method_keys}
    dumped = {key: [] for key in method_keys}

    latent_q_embs = None
    if "latent" in methods and compressor is not None and prepared_pool:
        q_emb_chunks = []
        questions_all = [s["question"] for s in prepared_pool]
        for start in tqdm(
            range(0, len(questions_all), latent_query_batch_size),
            desc="latent-query",
            leave=False,
        ):
            q_lat = compressor.embed_query_batch(
                questions_all[start: start + latent_query_batch_size],
                with_grad=False,
            )
            q_emb_chunks.append(compressor.get_retrieval_embedding(q_lat))
        latent_q_embs = torch.cat(q_emb_chunks, dim=0)

    for sample_idx, s in enumerate(tqdm(prepared_pool, desc="unified-baselines", leave=False)):
        question = s["question"]
        answers = s["answers"]
        full_context_candidates = s["full_context_candidates"]
        candidates = s["retrieval_candidates"]
        pos_indices = set(s["pos_indices"])
        full_pos_indices = set(s["full_pos_indices"])
        modality = s["modality"]

        if "full_context" in methods:
            hits = len(full_pos_indices)
            precision = (hits / len(full_context_candidates)) if full_context_candidates else 0.0
            item = {
                "mode": "evidence",
                "sample_id": s.get("id"),
                "question": question,
                "payload": full_context_candidates,
                "refs": answers,
                "recall": 1.0 if full_pos_indices else 0.0,
                "precision": precision,
                "modality": modality,
                "meta": {
                    "source": s.get("source", "webqa"),
                    "modality": modality,
                    "webqa_qcate": s.get("webqa_qcate"),
                    "webqa_keywords": s.get("webqa_keywords"),
                },
            }
            pending["full_context"].append(item)
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

        if "bm25" in methods:
            search_corpus = [c["search_text"] for c in candidates]
            ranked_indices["bm25"] = _bm25_retrieve(search_corpus, question, max_k)

        if "dense" in methods:
            search_corpus = [c["search_text"] for c in candidates]
            ranked_indices["dense"] = _dense_retrieve(search_corpus, question, max_k, dense_encoder)

        latent_projected = None
        latent_ranked = None
        if "latent" in methods:
            if compressor is None:
                raise ValueError("latent baseline requires a Qwen compressor checkpoint")
            text_items = [c for c in candidates if c["kind"] == "text"]
            image_items = [c for c in candidates if c["kind"] == "image"]
            latent_chunks = []
            emb_chunks = []
            item_order = []
            if text_items:
                text_latents = compressor.compress_batch(
                    [c["text"] for c in text_items], with_grad=False, adapter="compress"
                )
                latent_chunks.append(text_latents)
                emb_chunks.append(compressor.get_retrieval_embedding(text_latents))
                item_order.extend(text_items)
            if image_items:
                img_latents = compressor.compress_batch(
                    [c["title"] for c in image_items],
                    images=[c["image"] for c in image_items],
                    with_grad=False,
                    adapter="compress",
                )
                latent_chunks.append(img_latents)
                emb_chunks.append(compressor.get_retrieval_embedding(img_latents))
                item_order.extend(image_items)
            cand_latents = torch.cat(latent_chunks, dim=0)
            cand_embs = torch.cat(emb_chunks, dim=0)
            q_emb = latent_q_embs[sample_idx]
            sims = (cand_embs @ q_emb.unsqueeze(-1)).squeeze(-1)
            latent_ranked = torch.topk(sims, k=min(max_k, len(item_order))).indices.tolist()
            latent_projected = compressor.project_for_generator(cand_latents)

        for method in methods:
            if method == "full_context":
                continue
            for k in top_k_values:
                key = f"{method}_k{k}"
                if method == "latent":
                    top_i = latent_ranked[: min(k, len(latent_ranked))]
                    hits = len(set(top_i) & pos_indices)
                    recall = hits / len(pos_indices) if pos_indices else 0.0
                    precision = hits / len(top_i) if top_i else 0.0
                    item = {
                        "mode": "latent",
                        "sample_id": s.get("id"),
                        "question": question,
                        "payload": latent_projected[top_i],
                        "refs": answers,
                        "recall": recall,
                        "precision": precision,
                        "modality": modality,
                        "meta": {
                            "source": s.get("source", "webqa"),
                            "modality": modality,
                            "webqa_qcate": s.get("webqa_qcate"),
                            "webqa_keywords": s.get("webqa_keywords"),
                        },
                    }
                    pending[key].append(item)
                    if dump_examples > 0 and len(dumped[key]) < dump_examples:
                        dumped[key].append({
                            "sample_id": s.get("id"),
                            "modality": modality,
                            "question": question,
                            "answers": answers,
                            "retrieved_indices": top_i,
                            "retrieved": _serialize_evidences([candidates[i] for i in top_i]),
                            "prompt": generator.build_latent_prompt_text(question, latent_projected[top_i]),
                        })
                else:
                    top_i = ranked_indices[method][: min(k, len(ranked_indices[method]))]
                    hits = len(set(top_i) & pos_indices)
                    recall = hits / len(pos_indices) if pos_indices else 0.0
                    precision = hits / len(top_i) if top_i else 0.0
                    item = {
                        "mode": "evidence",
                        "sample_id": s.get("id"),
                        "question": question,
                        "payload": [candidates[i] for i in top_i],
                        "refs": answers,
                        "recall": recall,
                        "precision": precision,
                        "modality": modality,
                        "meta": {
                            "source": s.get("source", "webqa"),
                            "modality": modality,
                            "webqa_qcate": s.get("webqa_qcate"),
                            "webqa_keywords": s.get("webqa_keywords"),
                        },
                    }
                    pending[key].append(item)
                    if dump_examples > 0 and len(dumped[key]) < dump_examples:
                        retrieved = [candidates[i] for i in top_i]
                        dumped[key].append({
                            "sample_id": s.get("id"),
                            "modality": modality,
                            "question": question,
                            "answers": answers,
                            "retrieved_indices": top_i,
                            "retrieved": _serialize_evidences(retrieved),
                            "prompt": generator.build_evidence_prompt_text(question, retrieved),
                        })

    for key, items in pending.items():
        if not items:
            continue
        for start in tqdm(range(0, len(items), gen_batch_size), desc=f"gen-{key}", leave=False):
            chunk = items[start: start + gen_batch_size]
            questions = [it["question"] for it in chunk]
            if chunk[0]["mode"] == "latent":
                answers, lengths = generator.batch_generate_with_latents(
                    questions, [it["payload"] for it in chunk]
                )
            else:
                answers, lengths = generator.batch_generate_with_evidence(
                    questions, [it["payload"] for it in chunk]
                )
            for ans, n_tok, item in zip(answers, lengths, chunk):
                acc[key]["preds"].append(ans)
                acc[key]["refs"].append(item["refs"])
                acc[key]["recalls"].append(item["recall"])
                acc[key]["precisions"].append(item["precision"])
                acc[key]["tokens"].append(n_tok)
                acc[key]["mods"].append(item["modality"])
                acc[key]["meta"].append(item["meta"])
            if dump_examples > 0 and dumped.get(key):
                start_idx = start
                for local_idx, ans in enumerate(answers):
                    global_idx = start_idx + local_idx
                    if global_idx < len(dumped[key]):
                        dumped[key][global_idx]["prediction"] = ans

    results = {key: _finalize_method_metrics(val) for key, val in acc.items()}
    return results, dumped


def _fmt_metric(metrics: Dict, key: str, scale: float = 1.0) -> str:
    val = metrics.get(key)
    if val is None:
        return "NA"
    return f"{val * scale:.2f}"


def _print_results_report(results: Dict[str, Dict]) -> None:
    """Print a compact metrics report for job logs."""
    if not results:
        print("No results.")
        return

    columns = ["method", "f1", "acc", "r@k", "p@k", "tokens", "n"]
    print("\nResults")
    print(" ".join(columns))
    for method, metrics in results.items():
        if not metrics:
            print(f"{method} empty")
            continue
        print(
            " ".join([
                method,
                _fmt_metric(metrics, "f1", 100.0),
                _fmt_metric(metrics, "acc", 100.0),
                _fmt_metric(metrics, "recall_at_k", 100.0),
                _fmt_metric(metrics, "precision_at_k", 100.0),
                _fmt_metric(metrics, "avg_tokens"),
                str(metrics.get("n", metrics.get("n_samples", "NA"))),
            ])
        )

    has_split = any(
        any(k.startswith("img_") or k.startswith("txt_") for k in metrics)
        for metrics in results.values()
        if metrics
    )
    if not has_split:
        return

    print("\nImage")
    print("method f1 acc r@k p@k tokens")
    for method, metrics in results.items():
        if not metrics:
            continue
        print(
            " ".join([
                method,
                _fmt_metric(metrics, "img_f1", 100.0),
                _fmt_metric(metrics, "img_acc", 100.0),
                _fmt_metric(metrics, "img_recall_at_k", 100.0),
                _fmt_metric(metrics, "img_precision_at_k", 100.0),
                _fmt_metric(metrics, "img_avg_tokens"),
            ])
        )

    print("\nText")
    print("method f1 acc r@k p@k tokens")
    for method, metrics in results.items():
        if not metrics:
            continue
        print(
            " ".join([
                method,
                _fmt_metric(metrics, "txt_f1", 100.0),
                _fmt_metric(metrics, "txt_acc", 100.0),
                _fmt_metric(metrics, "txt_recall_at_k", 100.0),
                _fmt_metric(metrics, "txt_precision_at_k", 100.0),
                _fmt_metric(metrics, "txt_avg_tokens"),
            ])
        )


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = cfg["model"]
    scale = cfg.get("image", {}).get("scale", 0.1)
    gen_name = model_cfg["generator_name"]
    max_new_tokens = cfg.get("generation", {}).get("max_new_tokens", 64)
    dense_model = cfg.get("visrag", {}).get("dense_model", "sentence-transformers/all-MiniLM-L6-v2")

    top_k_values = sorted(set(args.top_k_values or cfg.get("retrieval", {}).get("top_k_values", [1, 2, 5])))
    val_path = args.val or cfg.get("data", {}).get("val")
    if not val_path or not os.path.exists(val_path):
        raise ValueError(f"Val data not found: {val_path}")

    logger.info(f"Loading val data: {val_path}")
    val_all = load_webqa_samples(val_path)
    for s in val_all:
        if "modality" not in s:
            s["modality"] = "image" if s.get("pos_image_ids") else "text"
    val_image = [s for s in val_all if s["modality"] == "image"]
    val_text = [s for s in val_all if s["modality"] == "text"]
    logger.info(f"  val image: {len(val_image)}  val text: {len(val_text)}")

    requested = [m.lower() for m in args.methods]
    valid = {"bm25", "dense", "full_context", "latent"}
    methods = [m for m in requested if m in valid]
    fc_requested = "full_context" in methods
    retrieval_methods = [m for m in methods if m != "full_context"]
    if not methods:
        raise ValueError("No valid methods specified.")

    generator = QwenGenerator(gen_name, device=device, max_new_tokens=max_new_tokens)
    compressor = None
    if "latent" in retrieval_methods:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for the latent baseline")
        comp_cfg = cfg["compressor"]
        dec_cfg = cfg.get("decoder", {})
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
            load_clip_model=False,
        ).to(device)
        compressor.load(args.checkpoint)
        compressor.eval()
    dense_encoder = None
    if "dense" in retrieval_methods:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading shared dense encoder: {dense_model}")
        dense_encoder = SentenceTransformer(dense_model)

    val_pool = _prepare_eval_pool(val_all, args.max_samples)
    run_methods = []
    if fc_requested:
        run_methods.append("full_context")
    run_methods.extend(retrieval_methods)
    results, dumped = run_unified_baselines_multik(
        val_all, generator, top_k_values, scale, args.max_samples, run_methods,
        dense_model, dense_encoder=dense_encoder, pool=val_pool, compressor=compressor,
        gen_batch_size=args.gen_batch_size,
        latent_query_batch_size=args.latent_query_batch_size,
        dump_examples=args.dump_examples,
    )

    result_obj = {"top_k_values": top_k_values, "results": results}
    _print_results_report(results)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result_obj, f, indent=2)
        logger.info(f"Saved: {args.output}")
        if args.dump_examples > 0:
            dump_path = args.output.replace(".json", ".examples.json")
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(dumped, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved examples: {dump_path}")
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebQA Qwen baselines")
    parser.add_argument("--config", required=True)
    parser.add_argument("--val", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--methods", nargs="+", default=["full_context", "bm25", "dense"])
    parser.add_argument("--top_k_values", type=int, nargs="+", default=None)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--gen_batch_size", type=int, default=8)
    parser.add_argument("--latent_query_batch_size", type=int, default=128)
    parser.add_argument("--dump_examples", type=int, default=0)
    parser.add_argument("--output", default="results/baselines_qwen.json")
    args = parser.parse_args()
    main(args)

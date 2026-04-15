"""
Validation Runner
=================
Runs end-to-end evaluation of the full latent retrieval pipeline on HotpotQA
validation samples without computing gradients.

Pipeline for each sample
------------------------
1. Compress ALL context sentences → (N, 2048) via SentenceAutoencoder.compress_batch()
2. Embed the query → (1, retrieval_dim) via SentenceAutoencoder.embed_query_batch()
3. Get retrieval embeddings for all sentences → (N, retrieval_dim) via get_retrieval_embedding()
4. Cosine similarity → pick top_k sentence indices
5. Project top_k compressed embeddings → (k, 4096) via project_for_generator()
6. Build input_embeds = cat([projected_tokens, query_embeds], dim=1)
7. Call frozen 8B generator.generate(inputs_embeds=...) → answer string
8. Compute EM / F1 / ROUGE-L vs. ground-truth answers

Aggregate metrics returned
--------------------------
em, f1, rouge_l, avg_tokens, avg_retrieved_sentences
"""

import logging
import random
from typing import Dict, List

import torch
from tqdm import tqdm

from src.evaluation import evaluate
from src.distillation import _split_chat_template, _tok

logger = logging.getLogger(__name__)


class ValidationRunner:
    """
    Evaluates the full 1B-compressor → projection → frozen-8B-generator pipeline.

    Efficient multi-k evaluation via run_multi_k():
      - All sentences across all samples are compressed in one batched pass.
      - All queries are embedded in one batched pass.
      - For each k: retrieval is a cheap cosine-sim + topk; generation runs in
        batches of gen_batch_size using left-padded inputs_embeds.
      - No redundant compression when sweeping k=1,2,5,10.

    Parameters
    ----------
    autoencoder : SentenceAutoencoder
    generator : AutoModelForCausalLM  (frozen 8B)
    generator_tokenizer : PreTrainedTokenizer
    top_k : int
        Default k used by run() for single-k evaluation.
    max_new_tokens : int
    device : str
    compress_batch_size : int
        Max sentences per compress_batch() call (avoids OOM on large contexts).
    gen_batch_size : int
        Samples per batched generate() call.
    """

    def __init__(
        self,
        autoencoder,
        generator,
        generator_tokenizer,
        top_k: int = 5,
        max_new_tokens: int = 64,
        device: str = "cuda",
        compress_batch_size: int = 128,
        gen_batch_size: int = 16,
    ):
        self.autoencoder = autoencoder
        self.generator = generator
        self.generator_tokenizer = generator_tokenizer
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.compress_batch_size = compress_batch_size
        self.gen_batch_size = gen_batch_size

        if self.generator_tokenizer.pad_token is None:
            self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_seq_embeds(
        self,
        projected_tokens: torch.Tensor,  # (k, H_gen)
        query: str,
    ) -> torch.Tensor:
        """Return input_embeds for one sample as a 1-D sequence (L, H_gen)."""
        embed = self.generator.get_input_embeddings()
        dtype = next(self.generator.parameters()).dtype

        def _e(text):
            return embed(_tok(self.generator_tokenizer, text, self.device)).to(dtype)

        prefix_str, suffix_str = _split_chat_template(self.generator_tokenizer, query)
        parts = [_e(prefix_str)]
        for i, lat in enumerate(projected_tokens):
            label = f"Latent context {i+1}: " if i == 0 else f"\nLatent context {i+1}: "
            parts.append(_e(label))
            parts.append(lat.reshape(-1, lat.shape[-1]).to(dtype).unsqueeze(0))
        parts.append(_e(suffix_str))
        return torch.cat(parts, dim=1).squeeze(0)   # (L, H)

    def _batch_generate(
        self,
        projected_list: List[torch.Tensor],  # list of (k_i, H_gen)
        questions: List[str],
    ):
        """
        Generate answers for a batch of (projected_tokens, question) pairs.

        Left-pads inputs_embeds so all sequences share the same length.
        Passes explicit position_ids so each sample's real tokens are at
        positions 0..L_i-1, independent of padding offset.

        Returns
        -------
        answers : list[str]
        input_lens : list[int]  (real sequence length per sample, for token counting)
        """
        seq_embeds = [
            self._build_seq_embeds(proj, q)
            for proj, q in zip(projected_list, questions)
        ]

        B      = len(seq_embeds)
        max_L  = max(e.shape[0] for e in seq_embeds)
        H      = seq_embeds[0].shape[-1]
        dtype  = seq_embeds[0].dtype

        padded      = torch.zeros(B, max_L, H,    dtype=dtype,        device=self.device)
        attn_mask   = torch.zeros(B, max_L,       dtype=torch.long,   device=self.device)
        position_ids = torch.zeros(B, max_L,      dtype=torch.long,   device=self.device)
        input_lens  = []

        for i, emb in enumerate(seq_embeds):
            L = emb.shape[0]
            padded[i, max_L - L:]      = emb
            attn_mask[i, max_L - L:]   = 1
            position_ids[i, max_L - L:] = torch.arange(L, device=self.device)
            input_lens.append(L)

        out = self.generator.generate(
            inputs_embeds=padded,
            attention_mask=attn_mask,
            position_ids=position_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.generator_tokenizer.eos_token_id,
        )

        # older transformers (<4.43): out = (B, n_new)
        # newer transformers (>=4.43): out = (B, max_L + n_new)
        answers = []
        for i in range(B):
            gen_ids = out[i, max_L:] if out.shape[1] > max_L else out[i]
            answers.append(
                self.generator_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            )
        return answers, input_lens

    # ------------------------------------------------------------------
    # Public: multi-k evaluation (compress once, generate per k)
    # ------------------------------------------------------------------

    def run_multi_k(
        self,
        samples: List[Dict],
        k_list: List[int],
        n_samples: int = None,
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluate at every k in k_list in a single compression pass.

        Phase 1  — compress all sentences from all samples once.
        Phase 2  — embed all queries once.
        Phase 3  — for each k: cosine-sim retrieval + project + batch generate.

        Returns
        -------
        dict mapping k → metrics dict (em, f1, rouge_l, recall_at_k, …)
        """
        samples = list(samples)
        random.Random(42).shuffle(samples)
        if n_samples is not None:
            samples = samples[:n_samples]

        valid = [
            s for s in samples
            if s.get("sentences") and s.get("question") and s.get("answers")
        ]
        if not valid:
            empty = {"em": 0.0, "f1": 0.0, "rouge_l": 0.0,
                     "avg_tokens": 0.0, "avg_retrieved_sentences": 0.0,
                     "recall_at_k": 0.0, "precision_at_k": 0.0}
            return {k: empty for k in k_list}

        was_training = self.autoencoder.training
        self.autoencoder.eval()

        with torch.no_grad():
            # ── Phase 1: compress all sentences (sub-batched to avoid OOM) ──
            all_sentences: List[str] = []
            offsets: List[tuple] = []          # (start, end) index into all_sentences
            for s in valid:
                start = len(all_sentences)
                all_sentences.extend(s["sentences"])
                offsets.append((start, len(all_sentences)))

            n_sent_batches = (len(all_sentences) + self.compress_batch_size - 1) // self.compress_batch_size
            compressed_chunks = []
            for i in tqdm(range(0, len(all_sentences), self.compress_batch_size),
                          total=n_sent_batches, desc="Compressing sentences", leave=False):
                compressed_chunks.append(
                    self.autoencoder.compress_batch(
                        all_sentences[i: i + self.compress_batch_size],
                        with_grad=False,
                    ).cpu()
                )
            all_compressed = torch.cat(compressed_chunks, dim=0)   # (total_sents, H_1B)

            # ── Phase 2: retrieval embeddings for all sentences ──────────────
            n_ret_batches = (all_compressed.shape[0] + self.compress_batch_size - 1) // self.compress_batch_size
            ret_emb_chunks = []
            for i in tqdm(range(0, all_compressed.shape[0], self.compress_batch_size),
                          total=n_ret_batches, desc="Retrieval embeddings", leave=False):
                ret_emb_chunks.append(
                    self.autoencoder.get_retrieval_embedding(
                        all_compressed[i: i + self.compress_batch_size].to(self.device)
                    ).cpu()
                )
            all_sent_embs = torch.cat(ret_emb_chunks, dim=0)       # (total_sents, D)

            # ── Phase 3: embed all queries ───────────────────────────────────
            questions = [s["question"] for s in valid]
            n_q_batches = (len(questions) + self.compress_batch_size - 1) // self.compress_batch_size
            q_emb_chunks = []
            for i in tqdm(range(0, len(questions), self.compress_batch_size),
                          total=n_q_batches, desc="Embedding queries", leave=False):
                q_emb_chunks.append(
                    self.autoencoder.embed_query_batch(
                        questions[i: i + self.compress_batch_size]
                    ).cpu()
                )
            all_q_embs = torch.cat(q_emb_chunks, dim=0)            # (N, D)

            # ── Phase 4: per-k retrieval + batched generation ────────────────
            results: Dict[int, Dict[str, float]] = {}

            for k in k_list:
                predictions: List[str]       = []
                references:  List[List[str]] = []
                token_counts: List[int]      = []
                retrieved_counts: List[int]  = []
                recall_scores: List[float]   = []
                precision_scores: List[float]= []

                # Retrieve on CPU; delay projection to the generation batch
                top_indices_by_sample: List[torch.Tensor] = []
                for idx, (s, (start, end)) in enumerate(zip(valid, offsets)):
                    actual_k    = min(k, end - start)
                    q_emb       = all_q_embs[idx: idx + 1]          # (1, D)
                    sent_embs   = all_sent_embs[start:end]           # (n, D)
                    sims        = (sent_embs @ q_emb.T).squeeze(-1)  # (n,)
                    top_indices = torch.topk(sims, k=actual_k).indices.cpu()
                    top_indices_by_sample.append(top_indices)

                    pos_indices = s.get("positive_indices", [])
                    if pos_indices:
                        retrieved_set = set(top_indices.tolist())
                        pos_set       = set(pos_indices)
                        hits = len(retrieved_set & pos_set)
                        recall_scores.append(hits / len(pos_set))
                        precision_scores.append(hits / len(retrieved_set))

                    retrieved_counts.append(actual_k)
                    ans = s["answers"]
                    references.append(ans if isinstance(ans, list) else [ans])

                # Batch generate
                n_gen_batches = (len(valid) + self.gen_batch_size - 1) // self.gen_batch_size
                for b_start in tqdm(range(0, len(valid), self.gen_batch_size),
                                    total=n_gen_batches, desc=f"Generating k={k}", leave=False):
                    b_proj = []
                    b_qs   = questions[b_start: b_start + self.gen_batch_size]
                    batch_offsets = offsets[b_start: b_start + self.gen_batch_size]
                    for local_idx, (start, end) in enumerate(batch_offsets):
                        top_indices = top_indices_by_sample[b_start + local_idx]
                        top_compressed = all_compressed[start:end][top_indices].to(self.device)
                        b_proj.append(self.autoencoder.project_for_generator(top_compressed))
                    try:
                        b_answers, b_lens = self._batch_generate(b_proj, b_qs)
                    except Exception as exc:
                        logger.warning(f"Batch generation failed (k={k}): {exc}")
                        b_answers = [""] * len(b_proj)
                        b_lens    = [0]   * len(b_proj)
                    predictions.extend(b_answers)
                    token_counts.extend(b_lens)

                metrics = evaluate(predictions, references, token_counts)
                metrics["avg_retrieved_sentences"] = (
                    sum(retrieved_counts) / len(retrieved_counts)
                )
                metrics["recall_at_k"]    = float(sum(recall_scores)    / len(recall_scores))    if recall_scores    else 0.0
                metrics["precision_at_k"] = float(sum(precision_scores) / len(precision_scores)) if precision_scores else 0.0

                logger.info(
                    f"Validation k={k} | n={len(predictions)} | "
                    f"EM={metrics['em']:.4f} F1={metrics['f1']:.4f} "
                    f"Recall@{k}={metrics['recall_at_k']:.4f}"
                )
                results[k] = metrics

        if was_training:
            self.autoencoder.train()

        return results

    # ------------------------------------------------------------------
    # Public: single-k convenience wrapper (backward-compatible)
    # ------------------------------------------------------------------

    def run(self, samples: List[Dict], n_samples: int = None) -> Dict[str, float]:
        """Single-k evaluation. Delegates to run_multi_k for efficiency."""
        return self.run_multi_k(samples, [self.top_k], n_samples)[self.top_k]


class WebQAValidationRunner:
    """
    End-to-end validation for WebQA (multimodal: image + optional text evidence).

    Pipeline per sample
    -------------------
    1. Load all candidate images (pos_image_ids + neg_image_ids) from disk
    2. Encode via ImageEncoder → (N, compressor_hidden)
    3. Get retrieval embeddings via autoencoder.get_retrieval_embedding() → (N, D)
    4. Encode query text via autoencoder.embed_query_batch() → (1, D)
    5. Cosine similarity → top_k image indices
    6. If txt_pos_facts exist, compress them and append as extra latent tokens
    7. Project retrieved embeddings → (k[+T], generator_hidden) via cross_proj
    8. Generate answer with frozen 8B, compute EM / F1 / ROUGE-L
    9. Recall@k: fraction of pos_image_ids in top-k retrieved set

    Metrics returned
    ----------------
    em, f1, rouge_l, recall_at_k, precision_at_k, avg_tokens, n_samples
    """

    def __init__(
        self,
        autoencoder,
        image_encoder,
        generator,
        generator_tokenizer,
        top_k: int = 5,
        max_new_tokens: int = 64,
        device: str = "cuda",
    ):
        self.autoencoder = autoencoder
        self.image_encoder = image_encoder
        self.generator = generator
        self.generator_tokenizer = generator_tokenizer
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.device = device

        if self.generator_tokenizer.pad_token is None:
            self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token

    def run(self, samples: List[Dict], n_samples: int = None) -> Dict[str, float]:
        """
        Evaluate on up to n_samples WebQA validation samples.

        Each sample must have:
          - "question"       : str
          - "answers"        : List[str]
          - "pos_image_ids"  : List[str]
          - "neg_image_ids"  : List[str]
          - "image_dir"      : str
          - "txt_pos_facts"  : List[str]  (optional text evidence)
        """
        samples = list(samples)
        random.Random(42).shuffle(samples)
        if n_samples is not None:
            samples = samples[:n_samples]

        was_training = self.autoencoder.training
        self.autoencoder.eval()
        self.image_encoder.eval()

        predictions: List[str] = []
        references: List[List[str]] = []
        token_counts: List[int] = []
        recall_scores: List[float] = []
        precision_scores: List[float] = []

        embed = self.generator.get_input_embeddings()
        dtype = next(self.generator.parameters()).dtype

        def _e(text):
            return embed(_tok(self.generator_tokenizer, text, self.device)).to(dtype)

        with torch.no_grad():
            for sample in tqdm(samples, desc="WebQA Val", leave=False):
                question  = sample.get("question", "")
                answers   = sample.get("answers", [])
                pos_ids   = sample.get("pos_image_ids", [])
                image_dir = sample.get("image_dir", "")

                if isinstance(answers, str):
                    answers = [answers]
                if not question or not answers:
                    continue

                # ── Route by modality ─────────────────────────────────────
                if pos_ids:
                    # IMAGE-modality sample
                    neg_ids   = sample.get("neg_image_ids", [])
                    txt_facts = sample.get("txt_pos_facts", [])

                    all_ids = pos_ids + neg_ids
                    pil_images, valid_ids, _ = self.image_encoder.load_images(
                        all_ids, image_dir
                    )
                    if not pil_images:
                        continue

                    pixel_values = self.image_encoder.preprocess(pil_images)
                    img_encoded  = self.image_encoder.encode(pixel_values, with_grad=False)
                    img_ret_embs = self.autoencoder.get_retrieval_embedding(img_encoded)

                    q_emb = self.autoencoder.embed_query_batch([question])
                    sims  = (img_ret_embs @ q_emb.T).squeeze(-1)
                    k     = min(self.top_k, len(valid_ids))
                    top_local     = torch.topk(sims, k=k).indices.tolist()
                    retrieved_ids = {valid_ids[i] for i in top_local}

                    pos_set = set(pos_ids)
                    hits = len(retrieved_ids & pos_set)
                    recall_scores.append(hits / len(pos_set))
                    precision_scores.append(hits / len(retrieved_ids))

                    top_compressed = img_encoded[top_local]
                    projected_all  = self.autoencoder.project_for_generator(top_compressed)
                    if txt_facts:
                        txt_comp      = self.autoencoder.compress_batch(
                            txt_facts, with_grad=False, adapter="compress"
                        )
                        projected_txt = self.autoencoder.project_for_generator(txt_comp)
                        projected_all = torch.cat([projected_all, projected_txt], dim=0)

                else:
                    # TEXT-modality sample — use sentence retrieval pipeline
                    sentences = sample.get("sentences", [])
                    pos_indices = sample.get("positive_indices", [])
                    if not sentences:
                        continue

                    compressed  = self.autoencoder.compress_batch(
                        sentences, with_grad=False
                    )
                    q_emb       = self.autoencoder.embed_query_batch([question])
                    sent_embs   = self.autoencoder.get_retrieval_embedding(compressed)
                    sims        = (sent_embs @ q_emb.T).squeeze(-1)
                    k           = min(self.top_k, len(sentences))
                    top_indices = torch.topk(sims, k=k).indices.tolist()

                    if pos_indices:
                        retrieved_set = set(top_indices)
                        pos_set       = set(pos_indices)
                        hits = len(retrieved_set & pos_set)
                        recall_scores.append(hits / len(pos_set))
                        precision_scores.append(hits / len(retrieved_set))

                    top_compressed = compressed[top_indices]
                    projected_all  = self.autoencoder.project_for_generator(top_compressed)

                # ── Generate answer (shared for both modalities) ──────────
                prefix_str, suffix_str = _split_chat_template(
                    self.generator_tokenizer, question
                )
                parts = [_e(prefix_str)]
                for i, lat in enumerate(projected_all):
                    label = f"Latent context {i+1}: " if i == 0 else f"\nLatent context {i+1}: "
                    parts.append(_e(label))
                    parts.append(lat.reshape(-1, lat.shape[-1]).to(dtype).unsqueeze(0).to(self.device))
                parts.append(_e(suffix_str))

                input_embeds   = torch.cat(parts, dim=1)
                n_input_tokens = input_embeds.shape[1]

                try:
                    out = self.generator.generate(
                        inputs_embeds=input_embeds,
                        attention_mask=torch.ones(
                            1, n_input_tokens, dtype=torch.long, device=self.device
                        ),
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.generator_tokenizer.eos_token_id,
                    )
                    if out.shape[1] > n_input_tokens:
                        generated_ids = out[0][n_input_tokens:]
                    else:
                        generated_ids = out[0]
                    answer = self.generator_tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    ).strip()
                except Exception as exc:
                    logger.warning(f"WebQA generation failed: {exc}")
                    answer = ""
                    n_input_tokens = 0

                predictions.append(answer)
                references.append(answers)
                token_counts.append(n_input_tokens)

        if was_training:
            self.autoencoder.train()
        self.image_encoder.train()

        if not predictions:
            logger.warning("WebQA: no predictions — returning zero metrics.")
            return {"em": 0.0, "f1": 0.0, "rouge_l": 0.0,
                    "recall_at_k": 0.0, "precision_at_k": 0.0,
                    "avg_tokens": 0.0, "n_samples": 0}

        metrics = evaluate(predictions, references, token_counts)
        metrics["recall_at_k"]    = float(sum(recall_scores)    / len(recall_scores))
        metrics["precision_at_k"] = float(sum(precision_scores) / len(precision_scores))

        logger.info(
            f"WebQA Val | n={len(predictions)} | "
            f"EM={metrics['em']:.4f} | F1={metrics['f1']:.4f} | "
            f"ROUGE-L={metrics['rouge_l']:.4f} | "
            f"Recall@{self.top_k}={metrics['recall_at_k']:.4f} | "
            f"Precision@{self.top_k}={metrics['precision_at_k']:.4f} | "
            f"avg_tokens={metrics.get('avg_tokens', 0):.1f}"
        )
        return metrics

"""
Baseline Evaluation Script
===========================
Three text-retrieval baselines, all using a shared frozen 8B LLM for generation.

FullContextBaseline
    Feed the entire context (all sentences concatenated) as raw text.
    Represents the upper bound on retrieval quality — the model sees everything.

BM25RetrievalBaseline
    Retrieve top-k sentences using BM25 (no neural embedding needed).
    Classical sparse retrieval baseline.

DenseRetrievalBaseline
    Retrieve top-k sentences using sentence-transformers dense embeddings.
    Strong off-the-shelf dense retrieval baseline.

All baselines share the same prompt format and generation settings so that
differences in score purely reflect retrieval quality, not generation setup.

Usage
-----
python scripts/baselines.py \
    --generator meta-llama/Llama-3.1-8B-Instruct \
    --data data/hotpotqa_val.json \
    --top_k 5 \
    --max_samples 500 \
    --output results/baselines/
"""

import argparse
import json
import logging
import os
import re
import string
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.evaluation import compare_results, evaluate, print_results

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MAX_CONTEXT_TOKENS = 4096


def _normalize_bm25(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _bm25_scores(query: str, sentences: List[str]) -> np.ndarray:
    """
    Minimal BM25 implementation with no external dependencies.

    Parameters
    ----------
    query : str
    sentences : list of str

    Returns
    -------
    scores : np.ndarray of shape (len(sentences),)
    """
    k1, b = 1.5, 0.75
    q_terms = _normalize_bm25(query).split()

    tf: List[Dict[str, int]] = []
    dl: List[int] = []
    for s in sentences:
        words = _normalize_bm25(s).split()
        dl.append(len(words))
        freq: Dict[str, int] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        tf.append(freq)

    avgdl = float(np.mean(dl)) if dl else 1.0
    N = len(sentences)
    scores = np.zeros(N)

    for term in q_terms:
        df = sum(1 for f in tf if term in f)
        idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
        for i, (f, d) in enumerate(zip(tf, dl)):
            freq_val = f.get(term, 0)
            scores[i] += idf * (freq_val * (k1 + 1.0)) / (
                freq_val + k1 * (1.0 - b + b * d / avgdl)
            )

    return scores


_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the question concisely (a few words or a short phrase) "
    "based only on the provided context."
)


def _build_prompt(tokenizer: AutoTokenizer, context: str, question: str) -> str:
    """Build prompt using the model's chat template (Llama-3 Instruct style)."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _split_title_prefixed_sentence(text: str) -> Tuple[Optional[str], str]:
    """
    Heuristically split strings like 'Title: sentence' into (title, sentence).
    Avoid breaking ordinary prose such as timestamps or generic clauses.
    """
    if not text:
        return None, ""

    match = re.match(r"^\s*([^:\n]{1,120}):\s+(.+?)\s*$", text)
    if not match:
        return None, text.strip()

    title = match.group(1).strip()
    sentence = match.group(2).strip()
    if not title or not sentence:
        return None, text.strip()

    if title.lower() in {"context", "question", "answer", "title", "evidence"}:
        return None, text.strip()

    if re.search(r"\d", title):
        return None, text.strip()

    return title, sentence


def _format_context(sentences: List[str], paragraph_map: Optional[List[str]] = None) -> str:
    """
    Make evidence more readable for instruction-tuned generators.

    Preference order:
      1. Use paragraph_map to group sentences under their source title.
      2. Fall back to parsing 'Title: sentence' patterns.
      3. Fall back to one evidence item per sentence.
    """
    if not sentences:
        return ""

    blocks: List[str] = []

    if paragraph_map and len(paragraph_map) == len(sentences):
        grouped: Dict[str, List[str]] = {}
        order: List[str] = []
        for sent, title in zip(sentences, paragraph_map):
            clean_sent = (sent or "").strip()
            clean_title = (title or "").strip() or "Untitled"
            if not clean_sent:
                continue
            parsed_title, parsed_sent = _split_title_prefixed_sentence(clean_sent)
            if parsed_title and parsed_title == clean_title:
                clean_sent = parsed_sent
            if clean_title not in grouped:
                grouped[clean_title] = []
                order.append(clean_title)
            grouped[clean_title].append(clean_sent)

        for idx, title in enumerate(order, start=1):
            joined = " ".join(grouped[title])
            blocks.append(f"Document {idx}\nTitle: {title}\nEvidence: {joined}")
        return "\n\n".join(blocks)

    parsed_blocks: List[Tuple[Optional[str], str]] = [
        _split_title_prefixed_sentence(s) for s in sentences if (s or "").strip()
    ]

    titled_count = sum(1 for title, _ in parsed_blocks if title)
    if titled_count >= max(2, len(parsed_blocks) // 2):
        current_title: Optional[str] = None
        current_sents: List[str] = []
        doc_idx = 0

        def flush_current() -> None:
            nonlocal current_title, current_sents, doc_idx
            if not current_sents:
                return
            doc_idx += 1
            title = current_title or f"Untitled {doc_idx}"
            blocks.append(f"Document {doc_idx}\nTitle: {title}\nEvidence: {' '.join(current_sents)}")
            current_title = None
            current_sents = []

        for title, sent in parsed_blocks:
            if title and title != current_title:
                flush_current()
                current_title = title
            current_sents.append(sent)
        flush_current()

        if blocks:
            return "\n\n".join(blocks)

    return "\n".join(
        f"Evidence {idx}: {sent.strip()}"
        for idx, sent in enumerate(sentences, start=1)
        if (sent or "").strip()
    )


def _generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> Tuple[str, int]:
    """
    Tokenise prompt, truncate to _MAX_CONTEXT_TOKENS, generate greedily.

    Returns
    -------
    answer : str
    n_input_tokens : int
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=_MAX_CONTEXT_TOKENS,
    ).to(device)

    n_input_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = out[0, n_input_tokens:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return answer, n_input_tokens


def _generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int,
    device: str,
) -> Tuple[List[str], List[int]]:
    """
    Tokenise prompts, truncate to _MAX_CONTEXT_TOKENS, and generate greedily in batch.

    Returns
    -------
    answers : list[str]
    n_input_tokens : list[int]
    """
    if not prompts:
        return [], []

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=_MAX_CONTEXT_TOKENS,
    ).to(device)

    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    prompt_padded_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    answers: List[str] = []
    for row in out:
        generated_ids = row[prompt_padded_len:]
        answers.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())

    return answers, [int(n) for n in input_lengths]


# ---------------------------------------------------------------------------
# Baseline 1: Full Context
# ---------------------------------------------------------------------------

class FullContextBaseline:
    """
    Feed ALL context sentences as raw text to the 8B LLM.

    This is the upper bound on retrieval quality: the model sees every sentence
    and must extract the answer by itself.  It is bounded only by the model's
    ability to handle long contexts and the 4096-token truncation limit.
    """

    def __init__(
        self,
        generator: AutoModelForCausalLM,
        generator_tokenizer: AutoTokenizer,
        max_new_tokens: int = 64,
        device: str = "cuda",
        batch_size: int = 1,
        dump_examples: int = 0,
        dump_dir: Optional[str] = None,
    ):
        self.generator = generator
        self.generator_tokenizer = generator_tokenizer
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.batch_size = max(1, batch_size)
        self.dump_examples = max(0, dump_examples)
        self.dump_dir = dump_dir
        self._dump_rows: List[Dict] = []

    def answer(
        self,
        question: str,
        sentences: List[str],
        paragraph_map: Optional[List[str]] = None,
    ) -> Tuple[str, int]:
        """
        Build a full-context prompt and generate an answer.

        Parameters
        ----------
        question : str
        sentences : list[str]

        Returns
        -------
        answer : str
        n_tokens : int  — number of input tokens consumed
        """
        if not sentences:
            return "", 0

        context = _format_context(sentences, paragraph_map)
        prompt = _build_prompt(self.generator_tokenizer, context, question)
        return _generate(
            self.generator,
            self.generator_tokenizer,
            prompt,
            self.max_new_tokens,
            self.device,
        )

    def _maybe_dump_example(
        self,
        sample: Dict,
        context: str,
        prompt: str,
        prediction: str,
        n_tokens: int,
    ) -> None:
        if len(self._dump_rows) >= self.dump_examples:
            return
        self._dump_rows.append({
            "id": sample.get("id"),
            "question": sample.get("question"),
            "answers": sample.get("answers", []),
            "paragraph_map_present": bool(sample.get("paragraph_map")),
            "raw_sentences_head": sample.get("sentences", [])[:8],
            "formatted_context": context,
            "prompt_head": prompt[:2000],
            "prediction": prediction,
            "input_tokens": n_tokens,
        })

    def _flush_dump(self) -> None:
        if not self.dump_examples or not self.dump_dir:
            return
        os.makedirs(self.dump_dir, exist_ok=True)
        dump_path = os.path.join(self.dump_dir, "full_context_debug_examples.json")
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(self._dump_rows, f, indent=2, ensure_ascii=False)
        logger.info(f"FullContext debug dump saved to {dump_path}")

    def run(self, samples: List[Dict]) -> Dict[str, float]:
        """
        Evaluate over a list of HotpotQA-format samples.

        Returns
        -------
        dict with em, f1, rouge_l, avg_tokens, recall_at_k, precision_at_k
        """
        prompts: List[str] = []
        contexts: List[str] = []
        kept_samples: List[Dict] = []
        references: List[List[str]] = []
        recall_scores: List[float] = []

        for sample in tqdm(samples, desc="FullContext"):
            sentences = sample.get("sentences", [])
            paragraph_map = sample.get("paragraph_map")
            question = sample["question"]
            answers = sample.get("answers", [])
            if isinstance(answers, str):
                answers = [answers]

            if not sentences or not question or not answers:
                continue

            context = _format_context(sentences, paragraph_map)
            prompt = _build_prompt(self.generator_tokenizer, context, question)
            prompts.append(prompt)
            contexts.append(context)
            kept_samples.append(sample)
            references.append(answers)

            pos_indices = sample.get("positive_indices", [])
            if pos_indices:
                recall_scores.append(1.0)  # full context always sees all positives

        if not prompts:
            return {"em": 0.0, "f1": 0.0, "rouge_l": 0.0, "avg_tokens": 0.0,
                    "recall_at_k": 0.0, "precision_at_k": 0.0}

        predictions: List[str] = []
        token_counts: List[int] = []
        for start in tqdm(range(0, len(prompts), self.batch_size), desc="FullContext Generate"):
            batch_prompts = prompts[start:start + self.batch_size]
            batch_answers, batch_tokens = _generate_batch(
                self.generator,
                self.generator_tokenizer,
                batch_prompts,
                self.max_new_tokens,
                self.device,
            )
            predictions.extend(batch_answers)
            token_counts.extend(batch_tokens)
            if self.dump_examples > 0:
                batch_contexts = contexts[start:start + self.batch_size]
                batch_samples = kept_samples[start:start + self.batch_size]
                for sample, context, prompt, pred, n_tok in zip(
                    batch_samples, batch_contexts, batch_prompts, batch_answers, batch_tokens
                ):
                    self._maybe_dump_example(sample, context, prompt, pred, n_tok)

        self._flush_dump()

        metrics = evaluate(predictions, references, token_counts)
        metrics["recall_at_k"]    = float(sum(recall_scores) / len(recall_scores)) if recall_scores else 0.0
        metrics["precision_at_k"] = 0.0  # not meaningful for full context
        return metrics


# ---------------------------------------------------------------------------
# Baseline 2: BM25 Retrieval
# ---------------------------------------------------------------------------

class BM25RetrievalBaseline:
    """
    Retrieve top-k sentences with BM25, then feed raw text to the 8B LLM.

    Uses a minimal, dependency-free BM25 implementation (k1=1.5, b=0.75).
    """

    def __init__(
        self,
        generator: AutoModelForCausalLM,
        generator_tokenizer: AutoTokenizer,
        top_k: int = 5,
        max_new_tokens: int = 64,
        device: str = "cuda",
        batch_size: int = 1,
    ):
        self.generator = generator
        self.generator_tokenizer = generator_tokenizer
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.batch_size = max(1, batch_size)

    def answer(
        self,
        question: str,
        sentences: List[str],
        paragraph_map: Optional[List[str]] = None,
    ) -> Tuple[str, int]:
        """
        BM25-retrieve top_k sentences, concatenate, generate answer.

        Parameters
        ----------
        question : str
        sentences : list[str]

        Returns
        -------
        answer : str
        n_tokens : int
        """
        if not sentences:
            return "", 0

        k = min(self.top_k, len(sentences))
        if k == 0:
            return "", 0

        scores = _bm25_scores(question, sentences)
        top_indices = np.argsort(scores)[::-1][:k]
        # Preserve document order for readability
        top_indices_ordered = sorted(top_indices.tolist())
        retrieved = [sentences[i] for i in top_indices_ordered]

        context = " ".join(retrieved)
        prompt = _build_prompt(self.generator_tokenizer, context, question)
        return _generate(
            self.generator,
            self.generator_tokenizer,
            prompt,
            self.max_new_tokens,
            self.device,
        )

    def run(self, samples: List[Dict]) -> Dict[str, float]:
        """
        Evaluate over a list of HotpotQA-format samples.

        Returns
        -------
        dict with em, f1, rouge_l, avg_tokens, recall_at_k, precision_at_k
        """
        prompts: List[str] = []
        references: List[List[str]] = []
        recall_scores:    List[float] = []
        precision_scores: List[float] = []

        for sample in tqdm(samples, desc="BM25"):
            sentences = sample.get("sentences", [])
            question = sample["question"]
            answers = sample.get("answers", [])
            if isinstance(answers, str):
                answers = [answers]

            if not sentences or not question or not answers:
                continue

            k = min(self.top_k, len(sentences))
            if k == 0:
                continue

            scores = _bm25_scores(question, sentences)
            top_indices = np.argsort(scores)[::-1][:k]
            top_indices_ordered = sorted(top_indices.tolist())
            retrieved = [sentences[i] for i in top_indices_ordered]
            prompts.append(_build_prompt(self.generator_tokenizer, " ".join(retrieved), question))
            references.append(answers)

            pos_indices = sample.get("positive_indices", [])
            if pos_indices:
                top_indices = set(np.argsort(scores)[::-1][:k].tolist())
                pos_set = set(pos_indices)
                hits = len(top_indices & pos_set)
                recall_scores.append(hits / len(pos_set))
                # Precision@k uses fixed denominator = top_k (not adaptive k)
                precision_scores.append(hits / self.top_k)

        if not prompts:
            return {"em": 0.0, "f1": 0.0, "rouge_l": 0.0, "avg_tokens": 0.0,
                    "recall_at_k": 0.0, "precision_at_k": 0.0}

        predictions: List[str] = []
        token_counts: List[int] = []
        for start in tqdm(range(0, len(prompts), self.batch_size), desc="BM25 Generate"):
            batch_prompts = prompts[start:start + self.batch_size]
            batch_answers, batch_tokens = _generate_batch(
                self.generator,
                self.generator_tokenizer,
                batch_prompts,
                self.max_new_tokens,
                self.device,
            )
            predictions.extend(batch_answers)
            token_counts.extend(batch_tokens)

        metrics = evaluate(predictions, references, token_counts)
        metrics["recall_at_k"]    = float(sum(recall_scores)    / len(recall_scores))    if recall_scores    else 0.0
        metrics["precision_at_k"] = float(sum(precision_scores) / len(precision_scores)) if precision_scores else 0.0
        return metrics


# ---------------------------------------------------------------------------
# Baseline 3: Dense Retrieval (sentence-transformers)
# ---------------------------------------------------------------------------

class DenseRetrievalBaseline:
    """
    Retrieve top-k sentences with a sentence-transformers dense encoder,
    then feed retrieved raw text to the 8B LLM.

    Requires the `sentence-transformers` package:
        pip install sentence-transformers
    """

    def __init__(
        self,
        generator: AutoModelForCausalLM,
        generator_tokenizer: AutoTokenizer,
        embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 5,
        max_new_tokens: int = 64,
        device: str = "cuda",
        batch_size: int = 1,
    ):
        self.generator = generator
        self.generator_tokenizer = generator_tokenizer
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.batch_size = max(1, batch_size)

        logger.info(f"Loading sentence-transformers embedder: {embedder_name}")
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(embedder_name, device=device)
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for DenseRetrievalBaseline. "
                "Install it with: pip install sentence-transformers"
            ) from exc

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Encode texts into L2-normalised dense vectors."""
        embeddings = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings  # (N, dim)

    def answer(self, question: str, sentences: List[str]) -> Tuple[str, int]:
        """
        Dense-retrieve top_k sentences, concatenate, generate answer.

        Parameters
        ----------
        question : str
        sentences : list[str]

        Returns
        -------
        answer : str
        n_tokens : int
        """
        if not sentences:
            return "", 0

        k = min(self.top_k, len(sentences))
        if k == 0:
            return "", 0

        # Embed query and all sentences
        all_texts = [question] + sentences
        all_embs = self._embed(all_texts)  # (1+N, dim)
        q_emb = all_embs[0:1]             # (1, dim)
        sent_embs = all_embs[1:]          # (N, dim)

        # Cosine similarity (vectors already L2-normalised)
        sims = (sent_embs @ q_emb.T).squeeze(-1)  # (N,)
        top_indices = np.argsort(sims)[::-1][:k]
        top_indices_ordered = sorted(top_indices.tolist())
        retrieved = [sentences[i] for i in top_indices_ordered]

        context = " ".join(retrieved)
        prompt = _build_prompt(self.generator_tokenizer, context, question)
        return _generate(
            self.generator,
            self.generator_tokenizer,
            prompt,
            self.max_new_tokens,
            self.device,
        )

    def run(self, samples: List[Dict]) -> Dict[str, float]:
        """
        Evaluate over a list of HotpotQA-format samples.

        Returns
        -------
        dict with em, f1, rouge_l, avg_tokens, recall_at_k, precision_at_k
        """
        prompts: List[str] = []
        references: List[List[str]] = []
        recall_scores:    List[float] = []
        precision_scores: List[float] = []

        for sample in tqdm(samples, desc="DenseRetrieval"):
            sentences = sample.get("sentences", [])
            question = sample["question"]
            answers = sample.get("answers", [])
            if isinstance(answers, str):
                answers = [answers]

            if not sentences or not question or not answers:
                continue

            k = min(self.top_k, len(sentences))
            if k == 0:
                continue

            all_embs = self._embed([question] + sentences)
            sims = (all_embs[1:] @ all_embs[0:1].T).squeeze(-1)
            top_indices = np.argsort(sims)[::-1][:k]
            top_indices_ordered = sorted(top_indices.tolist())
            retrieved = [sentences[i] for i in top_indices_ordered]
            prompts.append(_build_prompt(self.generator_tokenizer, " ".join(retrieved), question))
            references.append(answers)

            pos_indices = sample.get("positive_indices", [])
            if pos_indices:
                top_indices = set(np.argsort(sims)[::-1][:k].tolist())
                pos_set = set(pos_indices)
                hits = len(top_indices & pos_set)
                recall_scores.append(hits / len(pos_set))
                # Precision@k uses fixed denominator = top_k (not adaptive k)
                precision_scores.append(hits / self.top_k)

        if not prompts:
            return {"em": 0.0, "f1": 0.0, "rouge_l": 0.0, "avg_tokens": 0.0,
                    "recall_at_k": 0.0, "precision_at_k": 0.0}

        predictions: List[str] = []
        token_counts: List[int] = []
        for start in tqdm(range(0, len(prompts), self.batch_size), desc="Dense Generate"):
            batch_prompts = prompts[start:start + self.batch_size]
            batch_answers, batch_tokens = _generate_batch(
                self.generator,
                self.generator_tokenizer,
                batch_prompts,
                self.max_new_tokens,
                self.device,
            )
            predictions.extend(batch_answers)
            token_counts.extend(batch_tokens)

        metrics = evaluate(predictions, references, token_counts)
        metrics["recall_at_k"]    = float(sum(recall_scores)    / len(recall_scores))    if recall_scores    else 0.0
        metrics["precision_at_k"] = float(sum(precision_scores) / len(precision_scores)) if precision_scores else 0.0
        return metrics


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all_baselines(
    samples: List[Dict],
    generator: AutoModelForCausalLM,
    generator_tokenizer: AutoTokenizer,
    top_k: int = 5,
    device: str = "cuda",
    output_dir: str = "results/baselines/",
    max_new_tokens: int = 64,
    compression_cfg: Optional[Dict] = None,
    skip_standard: bool = False,
    batch_size: int = 1,
    dump_full_examples: int = 0,
) -> Dict[str, Dict]:
    """
    Run baselines, print a comparison table, and save results to JSON.

    Parameters
    ----------
    skip_standard : bool
        When True, skip FullContext / BM25 / Dense and run only compression baselines.
    """
    results_dict = {}

    if not skip_standard:
        logger.info("Running FullContextBaseline...")
        full_ctx = FullContextBaseline(
            generator,
            generator_tokenizer,
            max_new_tokens,
            device,
            batch_size=batch_size,
            dump_examples=dump_full_examples,
            dump_dir=output_dir,
        )
        results_dict["FullContext"] = full_ctx.run(samples)
        print_results(results_dict["FullContext"], "FullContext")

        logger.info("Running BM25RetrievalBaseline...")
        bm25 = BM25RetrievalBaseline(
            generator, generator_tokenizer, top_k, max_new_tokens, device, batch_size=batch_size
        )
        results_dict["BM25Retrieval"] = bm25.run(samples)
        print_results(results_dict["BM25Retrieval"], "BM25Retrieval")

        logger.info("Running DenseRetrievalBaseline...")
        try:
            dense = DenseRetrievalBaseline(
                generator, generator_tokenizer, top_k=top_k,
                max_new_tokens=max_new_tokens, device=device, batch_size=batch_size,
            )
            results_dict["DenseRetrieval"] = dense.run(samples)
        except ImportError as exc:
            logger.warning(f"DenseRetrievalBaseline skipped: {exc}")
            results_dict["DenseRetrieval"] = {"em": float("nan"), "f1": float("nan"), "rouge_l": float("nan")}
        print_results(results_dict["DenseRetrieval"], "DenseRetrieval")

    # --- Optional compression-based baselines ---
    if compression_cfg:
        from src.compression_baselines import (
            LLMLinguaBaseline, LCCBaseline, XRAGBaseline
        )

        if compression_cfg.get("llmlingua"):
            logger.info("Running LLMLinguaBaseline (20% and 10%)...")
            try:
                llmlingua = LLMLinguaBaseline(
                    generator, generator_tokenizer,
                    llmlingua_model=compression_cfg["llmlingua"].get(
                        "model", "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
                    ),
                    compression_ratio=0.2,
                    max_new_tokens=max_new_tokens, device=device,
                )
                results_dict["LLMLingua (20%)"] = llmlingua.run(samples, desc="LLMLingua (20%)")
                print_results(results_dict["LLMLingua (20%)"], "LLMLingua (20%)")
                llmlingua.compression_ratio = 0.1
                results_dict["LLMLingua (10%)"] = llmlingua.run(samples, desc="LLMLingua (10%)")
                print_results(results_dict["LLMLingua (10%)"], "LLMLingua (10%)")
                llmlingua.compression_ratio = 0.05
                results_dict["LLMLingua (5%)"] = llmlingua.run(samples, desc="LLMLingua (5%)")
                print_results(results_dict["LLMLingua (5%)"], "LLMLingua (5%)")
            except Exception as exc:
                logger.warning(f"LLMLinguaBaseline skipped: {exc}")
                nan = {"em": float("nan"), "f1": float("nan"), "rouge_l": float("nan")}
                results_dict["LLMLingua (20%)"] = results_dict["LLMLingua (10%)"] = results_dict["LLMLingua (5%)"] = nan

        if compression_cfg.get("lcc"):
            logger.info("Running LCCBaseline...")
            try:
                lcc = LCCBaseline(
                    compiler_checkpoint=compression_cfg["lcc"]["checkpoint"],
                    generator=generator,
                    generator_tokenizer=generator_tokenizer,
                    compressor_name=compression_cfg["lcc"].get("compressor_name", ""),
                    num_buffer_tokens=compression_cfg["lcc"].get("num_buffer_tokens", 64),
                    max_new_tokens=max_new_tokens, device=device,
                )
                results_dict["LCC"] = lcc.run(samples)
            except Exception as exc:
                logger.warning(f"LCCBaseline skipped: {exc}")
                results_dict["LCC"] = {"em": float("nan"), "f1": float("nan"), "rouge_l": float("nan")}
            print_results(results_dict["LCC"], "LCC")

        if compression_cfg.get("xrag"):
            logger.info("Running XRAGBaseline...")
            try:
                xrag = XRAGBaseline(
                    xrag_model_name=compression_cfg["xrag"].get("model", "Hannibal046/xrag-7b"),
                    retrieval_encoder_name=compression_cfg["xrag"].get(
                        "retrieval_encoder", "Salesforce/SFR-Embedding-Mistral"
                    ),
                    xrag_src=compression_cfg["xrag"].get("xrag_src", ""),
                    max_new_tokens=max_new_tokens, device=device,
                )
                results_dict["xRAG"] = xrag.run(samples)
            except Exception as exc:
                logger.warning(f"XRAGBaseline skipped: {exc}")
                results_dict["xRAG"] = {"em": float("nan"), "f1": float("nan"), "rouge_l": float("nan")}
            print_results(results_dict["xRAG"], "xRAG")

    compare_results(results_dict)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"Baseline results saved to {out_path}")

    return results_dict


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval and compression baselines across multiple datasets."
    )
    parser.add_argument("--config",      default="config.yaml",
                        help="config.yaml — reads generator_name and compression model paths.")
    parser.add_argument("--datasets",    default="hotpotqa",
                        help="Comma-separated primary+OOD datasets "
                             "(e.g. hotpotqa,2wikimultihopqa,musique,wikihop).")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples per dataset (0 = all).")
    parser.add_argument("--top_k",       type=int, default=5,
                        help="k for BM25/Dense/xRAG retrieval.")
    parser.add_argument("--output",      default="results/baselines/",
                        help="Root directory; per-dataset subdirs are created automatically.")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--compression_baselines", default=None,
                        help="Comma-separated: llmlingua,autocompressor,lcc,xrag")
    parser.add_argument("--skip_standard", action="store_true",
                        help="Skip FullContext/BM25/Dense; run only compression baselines.")
    # Model path overrides (fall back to config.yaml values)
    parser.add_argument("--llmlingua_model",      default=None)
    parser.add_argument("--autocompressor_model", default=None)
    parser.add_argument("--xrag_model",           default=None)
    parser.add_argument("--lcc_checkpoint",       default=None)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import sys, yaml
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    args = _parse_args()

    # --- Read config ---
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg_comp  = cfg.get("compression_baselines", {})
    gen_name  = cfg["model"]["generator_name"]
    cache_dir = cfg.get("data", {}).get("cache_dir", "data/cache/")

    # --- Load generator once ---
    logger.info(f"Loading generator: {gen_name}")
    generator_tokenizer = AutoTokenizer.from_pretrained(gen_name)
    if generator_tokenizer.pad_token is None:
        generator_tokenizer.pad_token = generator_tokenizer.eos_token
    generator_tokenizer.padding_side = "left"
    generator = AutoModelForCausalLM.from_pretrained(
        gen_name, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    generator.eval()
    for p in generator.parameters():
        p.requires_grad_(False)
    logger.info("Generator loaded and frozen.")

    # --- Build compression_cfg from config + CLI overrides ---
    compression_cfg = None
    if args.compression_baselines:
        requested = [b.strip().lower() for b in args.compression_baselines.split(",") if b.strip()]
        compression_cfg = {}
        if "llmlingua" in requested:
            entry = dict(cfg_comp.get("llmlingua", {}))
            if args.llmlingua_model:
                entry["model"] = args.llmlingua_model
            compression_cfg["llmlingua"] = entry
        if "autocompressor" in requested:
            entry = dict(cfg_comp.get("autocompressor", {}))
            if args.autocompressor_model:
                entry["model"] = args.autocompressor_model
            compression_cfg["autocompressor"] = entry
        if "autocompressor_512_50" in requested:
            compression_cfg["autocompressor_512_50"] = dict(cfg_comp.get("autocompressor_512_50", cfg_comp.get("autocompressor", {})))
        if "autocompressor_full_50" in requested:
            compression_cfg["autocompressor_full_50"] = dict(cfg_comp.get("autocompressor_full_50", cfg_comp.get("autocompressor", {})))
        if "lcc" in requested:
            entry = dict(cfg_comp.get("lcc", {}))
            if args.lcc_checkpoint:
                entry["checkpoint"] = args.lcc_checkpoint
            compression_cfg["lcc"] = entry
        if "xrag" in requested:
            entry = dict(cfg_comp.get("xrag", {}))
            if args.xrag_model:
                entry["model"] = args.xrag_model
            compression_cfg["xrag"] = entry

    # --- Load datasets and run ---
    from data.prepare_data import load_dataset_by_name, load_hotpotqa

    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    max_s = args.max_samples if args.max_samples > 0 else None
    all_results = {}

    for ds_name in dataset_names:
        logger.info(f"Loading dataset: {ds_name}")
        if ds_name == "hotpotqa":
            samples = load_hotpotqa(split="validation", max_samples=max_s, cache_dir=cache_dir)
        else:
            samples = load_dataset_by_name(ds_name, split=None, max_samples=max_s or 500,
                                           cache_dir=cache_dir)
            samples = [s for s in samples if s.get("sentences")]
        if not samples:
            logger.warning(f"No usable samples for {ds_name}, skipping.")
            continue
        logger.info(f"  {len(samples)} samples")

        out_dir = os.path.join(args.output, ds_name)
        results = run_all_baselines(
            samples=samples,
            generator=generator,
            generator_tokenizer=generator_tokenizer,
            top_k=args.top_k,
            device=args.device,
            output_dir=out_dir,
            max_new_tokens=args.max_new_tokens,
            compression_cfg=compression_cfg,
            skip_standard=args.skip_standard,
        )
        all_results[ds_name] = results

    import json as _json
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "results_summary.json"), "w") as f:
        _json.dump(all_results, f, indent=2)
    logger.info(f"Summary saved to {args.output}/results_summary.json")

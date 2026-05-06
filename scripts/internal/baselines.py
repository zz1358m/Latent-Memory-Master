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

_RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _RELEASE_ROOT not in sys.path:
    sys.path.insert(0, _RELEASE_ROOT)
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from jinja2.exceptions import TemplateError
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.evaluation import compare_results, evaluate, print_results

logger = logging.getLogger(__name__)


def _disable_tqdm_progress() -> None:
    global tqdm
    _orig_tqdm = tqdm

    def _quiet_tqdm(*args, **kwargs):
        kwargs["disable"] = True
        return _orig_tqdm(*args, **kwargs)

    tqdm = _quiet_tqdm

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
    "based on the provided context."
)


def _count_tokens(tokenizer: AutoTokenizer, text: str) -> int:
    return int(len(tokenizer(text, add_special_tokens=False)["input_ids"]))


def _count_tokens_from_marker(
    tokenizer: AutoTokenizer,
    prompt: str,
    markers: List[str],
) -> int:
    for marker in markers:
        idx = prompt.find(marker)
        if idx >= 0:
            return _count_tokens(tokenizer, prompt[idx:])
    return _count_tokens(tokenizer, prompt)


def _count_tokens_between_markers(
    tokenizer: AutoTokenizer,
    prompt: str,
    start_markers: List[str],
    end_markers: List[str],
) -> int:
    start_idx = None
    for marker in start_markers:
        idx = prompt.find(marker)
        if idx >= 0:
            start_idx = idx
            break
    if start_idx is None:
        return _count_tokens(tokenizer, prompt)

    end_idx = None
    for marker in end_markers:
        idx = prompt.find(marker, start_idx)
        if idx >= 0:
            end_idx = idx
            break
    if end_idx is None:
        return _count_tokens(tokenizer, prompt[start_idx:])
    return _count_tokens(tokenizer, prompt[start_idx:end_idx])


def _build_prompt(tokenizer: AutoTokenizer, context: str, question: str) -> str:
    """Build prompt using the model's chat template (Llama-3 Instruct style)."""
    user_text = f"{_SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except TemplateError as exc:
            if "System role not supported" not in str(exc):
                raise
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
    return (
        f"System: {_SYSTEM_PROMPT}\n\n"
        f"User: Context:\n{context}\n\nQuestion: {question}\n\n"
        "Assistant:"
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
        batch_size: int = 16,
        dump_examples: int = 0,
        dump_dir: Optional[str] = None,
        dump_name_suffix: str = "",
    ):
        self.generator = generator
        self.generator_tokenizer = generator_tokenizer
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.batch_size = max(1, batch_size)
        self.dump_examples = max(0, dump_examples)
        self.dump_dir = dump_dir
        self.dump_name_suffix = dump_name_suffix
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
            "prompt": prompt,
            "prediction": prediction,
            "input_tokens": n_tokens,
            "input_tokens_no_system": _count_tokens_from_marker(
                self.generator_tokenizer, prompt, ["Context:"]
            ),
            "input_tokens_context_only": _count_tokens_between_markers(
                self.generator_tokenizer, prompt, ["Context:"], ["Question:"]
            ),
        })

    def _flush_dump(self) -> None:
        if not self.dump_examples or not self.dump_dir:
            return
        os.makedirs(self.dump_dir, exist_ok=True)
        dump_path = os.path.join(self.dump_dir, f"full_context_debug_examples{self.dump_name_suffix}.json")
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
        token_counts_no_system: List[int] = []
        token_counts_context_only: List[int] = []
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
            token_counts_no_system.extend(
                _count_tokens_from_marker(self.generator_tokenizer, p, ["Context:"])
                for p in batch_prompts
            )
            token_counts_context_only.extend(
                _count_tokens_between_markers(self.generator_tokenizer, p, ["Context:"], ["Question:"])
                for p in batch_prompts
            )
            if self.dump_examples > 0:
                batch_contexts = contexts[start:start + self.batch_size]
                batch_samples = kept_samples[start:start + self.batch_size]
                for sample, context, prompt, pred, n_tok in zip(
                    batch_samples, batch_contexts, batch_prompts, batch_answers, batch_tokens
                ):
                    self._maybe_dump_example(sample, context, prompt, pred, n_tok)

        self._flush_dump()

        metrics = evaluate(
            predictions,
            references,
            token_counts,
            token_counts_no_system=token_counts_no_system,
            token_counts_context_only=token_counts_context_only,
        )
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
        batch_size: int = 16,
        dump_examples: int = 0,
        dump_dir: Optional[str] = None,
        dump_name_suffix: str = "",
    ):
        self.generator = generator
        self.generator_tokenizer = generator_tokenizer
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.batch_size = max(1, batch_size)
        self.dump_examples = max(0, dump_examples)
        self.dump_dir = dump_dir
        self.dump_name_suffix = dump_name_suffix
        self._dump_rows: List[Dict] = []

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
        retrieved_texts: List[List[str]] = []
        kept_samples: List[Dict] = []
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
            retrieved_texts.append(retrieved)
            kept_samples.append(sample)
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
        token_counts_no_system: List[int] = []
        token_counts_context_only: List[int] = []
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
            token_counts_no_system.extend(
                _count_tokens_from_marker(self.generator_tokenizer, p, ["Context:"])
                for p in batch_prompts
            )
            token_counts_context_only.extend(
                _count_tokens_between_markers(self.generator_tokenizer, p, ["Context:"], ["Question:"])
                for p in batch_prompts
            )
            if self.dump_examples > 0 and len(self._dump_rows) < self.dump_examples:
                batch_samples = kept_samples[start:start + self.batch_size]
                batch_retrieved = retrieved_texts[start:start + self.batch_size]
                for sample, retrieved, prompt, pred, n_tok in zip(
                    batch_samples, batch_retrieved, batch_prompts, batch_answers, batch_tokens
                ):
                    if len(self._dump_rows) >= self.dump_examples:
                        break
                    self._dump_rows.append({
                        "id": sample.get("id"),
                        "question": sample.get("question"),
                        "answers": sample.get("answers", []),
                        "retrieved_sentences": retrieved,
                        "prompt": prompt,
                        "prediction": pred,
                        "input_tokens": n_tok,
                        "input_tokens_no_system": _count_tokens_from_marker(
                            self.generator_tokenizer, prompt, ["Context:"]
                        ),
                        "input_tokens_context_only": _count_tokens_between_markers(
                            self.generator_tokenizer, prompt, ["Context:"], ["Question:"]
                        ),
                    })

        if self.dump_examples and self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)
            dump_path = os.path.join(self.dump_dir, f"bm25_debug_examples{self.dump_name_suffix}.json")
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(self._dump_rows, f, indent=2, ensure_ascii=False)
            logger.info(f"BM25 debug dump saved to {dump_path}")

        metrics = evaluate(
            predictions,
            references,
            token_counts,
            token_counts_no_system=token_counts_no_system,
            token_counts_context_only=token_counts_context_only,
        )
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
        batch_size: int = 16,
        dump_examples: int = 0,
        dump_dir: Optional[str] = None,
        dump_name_suffix: str = "",
    ):
        self.generator = generator
        self.generator_tokenizer = generator_tokenizer
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.batch_size = max(1, batch_size)
        self.dump_examples = max(0, dump_examples)
        self.dump_dir = dump_dir
        self.dump_name_suffix = dump_name_suffix
        self._dump_rows: List[Dict] = []

        logger.info(f"Loading sentence-transformers embedder: {embedder_name}")
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(embedder_name, device=device)
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for DenseRetrievalBaseline. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        self._supports_query_prompt = "qwen3-embedding" in embedder_name.lower()

    def _embed_queries(self, queries: List[str]) -> np.ndarray:
        encode_kwargs = dict(
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.batch_size,
        )
        if self._supports_query_prompt:
            encode_kwargs["prompt_name"] = "query"
        return self.embedder.encode(queries, **encode_kwargs)

    def _embed_query(self, query: str) -> np.ndarray:
        return self._embed_queries([query])

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Encode texts into L2-normalised dense vectors."""
        embeddings = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.batch_size,
        )
        return embeddings  # (N, dim)

    def _prepare_ranked_records(self, samples: List[Dict], desc: str = "DenseRetrieval Rank") -> List[Dict]:
        records: List[Dict] = []
        valid_samples: List[Dict] = []
        for sample in samples:
            sentences = sample.get("sentences", [])
            question = sample["question"]
            answers = sample.get("answers", [])
            if isinstance(answers, str):
                answers = [answers]
            if sentences and question and answers:
                valid_samples.append({
                    "sample": sample,
                    "question": question,
                    "answers": answers,
                    "sentences": sentences,
                })

        for start in tqdm(range(0, len(valid_samples), self.batch_size), desc=desc):
            batch = valid_samples[start:start + self.batch_size]
            questions = [item["question"] for item in batch]
            sentence_lengths = [len(item["sentences"]) for item in batch]
            flat_sentences = [sent for item in batch for sent in item["sentences"]]

            q_embs = self._embed_queries(questions)
            sent_embs = self._embed(flat_sentences)

            offset = 0
            for item, q_emb, sent_count in zip(batch, q_embs, sentence_lengths):
                sample_sent_embs = sent_embs[offset:offset + sent_count]
                sims = sample_sent_embs @ q_emb
                ranked_indices = np.argsort(sims)[::-1].tolist()
                records.append({
                    "sample": item["sample"],
                    "question": item["question"],
                    "answers": item["answers"],
                    "sentences": item["sentences"],
                    "ranked_indices": ranked_indices,
                })
                offset += sent_count
        return records

    def _run_from_ranked_records(
        self,
        records: List[Dict],
        top_k: int,
        desc_prefix: str = "DenseRetrieval",
    ) -> Dict[str, float]:
        prompts: List[str] = []
        retrieved_texts: List[List[str]] = []
        kept_samples: List[Dict] = []
        references: List[List[str]] = []
        recall_scores: List[float] = []
        precision_scores: List[float] = []
        self._dump_rows = []

        for record in records:
            sentences = record["sentences"]
            k = min(top_k, len(sentences))
            if k == 0:
                continue

            top_indices = record["ranked_indices"][:k]
            top_indices_ordered = sorted(top_indices)
            retrieved = [sentences[i] for i in top_indices_ordered]
            prompts.append(_build_prompt(self.generator_tokenizer, " ".join(retrieved), record["question"]))
            retrieved_texts.append(retrieved)
            kept_samples.append(record["sample"])
            references.append(record["answers"])

            pos_indices = record["sample"].get("positive_indices", [])
            if pos_indices:
                top_index_set = set(top_indices)
                pos_set = set(pos_indices)
                hits = len(top_index_set & pos_set)
                recall_scores.append(hits / len(pos_set))
                precision_scores.append(hits / top_k)

        if not prompts:
            return {"em": 0.0, "f1": 0.0, "rouge_l": 0.0, "avg_tokens": 0.0,
                    "recall_at_k": 0.0, "precision_at_k": 0.0}

        predictions: List[str] = []
        token_counts: List[int] = []
        token_counts_no_system: List[int] = []
        token_counts_context_only: List[int] = []
        for start in tqdm(range(0, len(prompts), self.batch_size), desc=f"{desc_prefix} Generate@{top_k}"):
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
            token_counts_no_system.extend(
                _count_tokens_from_marker(self.generator_tokenizer, p, ["Context:"])
                for p in batch_prompts
            )
            token_counts_context_only.extend(
                _count_tokens_between_markers(self.generator_tokenizer, p, ["Context:"], ["Question:"])
                for p in batch_prompts
            )
            if self.dump_examples > 0 and len(self._dump_rows) < self.dump_examples:
                batch_samples = kept_samples[start:start + self.batch_size]
                batch_retrieved = retrieved_texts[start:start + self.batch_size]
                for sample, retrieved, prompt, pred, n_tok in zip(
                    batch_samples, batch_retrieved, batch_prompts, batch_answers, batch_tokens
                ):
                    if len(self._dump_rows) >= self.dump_examples:
                        break
                    self._dump_rows.append({
                        "id": sample.get("id"),
                        "question": sample.get("question"),
                        "answers": sample.get("answers", []),
                        "retrieved_sentences": retrieved,
                        "prompt": prompt,
                        "prediction": pred,
                        "input_tokens": n_tok,
                        "input_tokens_no_system": _count_tokens_from_marker(
                            self.generator_tokenizer, prompt, ["Context:"]
                        ),
                        "input_tokens_context_only": _count_tokens_between_markers(
                            self.generator_tokenizer, prompt, ["Context:"], ["Question:"]
                        ),
                    })

        if self.dump_examples and self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)
            dump_path = os.path.join(self.dump_dir, f"dense_debug_examples{self.dump_name_suffix}.json")
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(self._dump_rows, f, indent=2, ensure_ascii=False)
            logger.info(f"Dense debug dump saved to {dump_path}")

        metrics = evaluate(
            predictions,
            references,
            token_counts,
            token_counts_no_system=token_counts_no_system,
            token_counts_context_only=token_counts_context_only,
        )
        metrics["recall_at_k"] = float(sum(recall_scores) / len(recall_scores)) if recall_scores else 0.0
        metrics["precision_at_k"] = float(sum(precision_scores) / len(precision_scores)) if precision_scores else 0.0
        return metrics

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

        q_emb = self._embed_query(question)  # (1, dim)
        sent_embs = self._embed(sentences)   # (N, dim)

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
        records = self._prepare_ranked_records(samples)
        return self._run_from_ranked_records(records, self.top_k)

    def run_multi_k(self, samples: List[Dict], top_k_values: List[int]) -> Dict[int, Dict[str, float]]:
        results: Dict[int, Dict[str, float]] = {}
        deduped_top_k = []
        seen = set()
        for top_k in top_k_values:
            if top_k not in seen:
                deduped_top_k.append(top_k)
                seen.add(top_k)
        records = self._prepare_ranked_records(samples)
        for top_k in deduped_top_k:
            results[top_k] = self._run_from_ranked_records(records, top_k)
        return results


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
    batch_size: int = 16,
    dump_full_examples: int = 0,
    qwen3_dense_model: str = "Qwen/Qwen3-Embedding-0.6B",
    standard_methods: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Run baselines, print a comparison table, and save results to JSON.

    Parameters
    ----------
    skip_standard : bool
        When True, skip FullContext / BM25 / Dense and run only compression baselines.
    """
    results_dict = {}
    requested_standard = set(m.lower() for m in (standard_methods or ["full_context", "bm25", "dense", "qwen3_dense"]))

    if not skip_standard:
        if "full_context" in requested_standard:
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

        if "bm25" in requested_standard:
            logger.info("Running BM25RetrievalBaseline...")
            bm25 = BM25RetrievalBaseline(
                generator, generator_tokenizer, top_k, max_new_tokens, device,
                batch_size=batch_size, dump_examples=dump_full_examples, dump_dir=output_dir
            )
            results_dict["BM25Retrieval"] = bm25.run(samples)
            print_results(results_dict["BM25Retrieval"], "BM25Retrieval")

        if "dense" in requested_standard:
            logger.info("Running DenseRetrievalBaseline...")
            try:
                dense = DenseRetrievalBaseline(
                    generator, generator_tokenizer, top_k=top_k,
                    max_new_tokens=max_new_tokens, device=device, batch_size=batch_size,
                    dump_examples=dump_full_examples, dump_dir=output_dir,
                )
                results_dict["DenseRetrieval"] = dense.run(samples)
            except ImportError as exc:
                logger.warning(f"DenseRetrievalBaseline skipped: {exc}")
                results_dict["DenseRetrieval"] = {"em": float("nan"), "f1": float("nan"), "rouge_l": float("nan")}
            print_results(results_dict["DenseRetrieval"], "DenseRetrieval")

        if "qwen3_dense" in requested_standard:
            logger.info("Running Qwen3DenseRetrievalBaseline...")
            try:
                qwen3_dense = DenseRetrievalBaseline(
                    generator,
                    generator_tokenizer,
                    embedder_name=qwen3_dense_model,
                    top_k=top_k,
                    max_new_tokens=max_new_tokens,
                    device=device,
                    batch_size=batch_size,
                    dump_examples=dump_full_examples,
                    dump_dir=output_dir,
                    dump_name_suffix="_qwen3",
                )
                results_dict["Qwen3DenseRetrieval"] = qwen3_dense.run(samples)
            except ImportError as exc:
                logger.warning(f"Qwen3DenseRetrievalBaseline skipped: {exc}")
                results_dict["Qwen3DenseRetrieval"] = {"em": float("nan"), "f1": float("nan"), "rouge_l": float("nan")}
            print_results(results_dict["Qwen3DenseRetrieval"], "Qwen3DenseRetrieval")

    # --- Optional compression-based baselines ---
    if compression_cfg:
        from src.compression_baselines import (
            LLMLinguaBaseline, AutoCompressorBaseline, LCCBaseline, XRAGBaseline
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

        if compression_cfg.get("autocompressor"):
            logger.info("Running AutoCompressorBaseline...")
            try:
                ac_cfg = compression_cfg["autocompressor"]
                autocompressor = AutoCompressorBaseline(
                    model_name=ac_cfg.get("model", "princeton-nlp/AutoCompressor-Llama-2-7b-6k"),
                    segment_length=ac_cfg.get("segment_length", 512),
                    summary_length=ac_cfg.get("summary_length", 50),
                    max_new_tokens=max_new_tokens,
                    device=device,
                    top_k=top_k,
                )
                results_dict["AutoCompressor"] = autocompressor.run(samples, batch_size=max(1, batch_size // 4))
            except Exception as exc:
                logger.warning(f"AutoCompressorBaseline skipped: {exc}")
                results_dict["AutoCompressor"] = {"em": float("nan"), "f1": float("nan"), "rouge_l": float("nan")}
            print_results(results_dict["AutoCompressor"], "AutoCompressor")

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
                             "(e.g. hotpotqa,2wikimultihopqa,musique,nq,triviaqa,qasper_longbench,wice).")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples per dataset (0 = all).")
    parser.add_argument("--top_k",       default="5",
                        help="Single k or comma-separated k list for retrieval baselines "
                             "(e.g. 5 or 1,2,5,10).")
    parser.add_argument("--output",      default="results/baselines/",
                        help="Root directory; per-dataset subdirs are created automatically.")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for standard dense/full-context generation; AutoCompressor uses batch_size//4.")
    parser.add_argument("--compression_baselines", default=None,
                        help="Comma-separated: llmlingua,autocompressor,lcc,xrag")
    parser.add_argument("--skip_standard", action="store_true",
                        help="Skip FullContext/BM25/Dense; run only compression baselines.")
    parser.add_argument("--standard_methods", default="full_context,bm25,dense,qwen3_dense",
                        help="Comma-separated standard baselines: full_context,bm25,dense,qwen3_dense")
    parser.add_argument("--disable_tqdm", action="store_true",
                        help="Disable tqdm progress bars in baseline loops.")
    # Model path overrides (fall back to config.yaml values)
    parser.add_argument("--llmlingua_model",      default=None)
    parser.add_argument("--autocompressor_model", default=None)
    parser.add_argument("--xrag_model",           default=None)
    parser.add_argument("--lcc_checkpoint",       default=None)
    parser.add_argument("--qwen3_dense_model",    default="Qwen/Qwen3-Embedding-0.6B")
    return parser.parse_args()


def _parse_top_k_values(raw_top_k: str) -> List[int]:
    values: List[int] = []
    for part in str(raw_top_k).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid --top_k value '{part}'. Use an integer or comma-separated integers."
            ) from exc
        if value <= 0:
            raise argparse.ArgumentTypeError("--top_k values must be positive integers.")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("--top_k must contain at least one positive integer.")
    return values


def _save_baseline_results(output_dir: str, results_dict: Dict[str, Dict]) -> None:
    compare_results(results_dict)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"Baseline results saved to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import sys, yaml
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    args = _parse_args()
    if args.disable_tqdm:
        _disable_tqdm_progress()

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
    standard_methods = [m.strip().lower() for m in args.standard_methods.split(",") if m.strip()]
    top_k_values = _parse_top_k_values(args.top_k)
    max_s = args.max_samples if args.max_samples > 0 else 0
    all_results = {}

    for ds_name in dataset_names:
        logger.info(f"Loading dataset: {ds_name}")
        if ds_name == "hotpotqa":
            samples = load_hotpotqa(split="validation", max_samples=max_s, cache_dir=cache_dir)
        else:
            samples = load_dataset_by_name(ds_name, split=None, max_samples=max_s,
                                           cache_dir=cache_dir)
            samples = [s for s in samples if s.get("sentences")]
        if not samples:
            logger.warning(f"No usable samples for {ds_name}, skipping.")
            continue
        logger.info(f"  {len(samples)} samples")

        if (
            len(top_k_values) > 1
            and not args.skip_standard
            and not compression_cfg
            and set(standard_methods).issubset({"dense", "qwen3_dense"})
        ):
            all_results[ds_name] = {}
            multi_k_results: Dict[str, Dict[int, Dict[str, float]]] = {}

            if "dense" in standard_methods:
                logger.info(f"Running DenseRetrieval once for all k on {ds_name}")
                dense = DenseRetrievalBaseline(
                    generator,
                    generator_tokenizer,
                    top_k=max(top_k_values),
                    max_new_tokens=args.max_new_tokens,
                    device=args.device,
                    batch_size=args.batch_size,
                )
                multi_k_results["DenseRetrieval"] = dense.run_multi_k(samples, top_k_values)

            if "qwen3_dense" in standard_methods:
                logger.info(f"Running Qwen3DenseRetrieval once for all k on {ds_name}")
                qwen3_dense = DenseRetrievalBaseline(
                    generator,
                    generator_tokenizer,
                    embedder_name=args.qwen3_dense_model,
                    top_k=max(top_k_values),
                    max_new_tokens=args.max_new_tokens,
                    device=args.device,
                    batch_size=args.batch_size,
                    dump_name_suffix="_qwen3",
                )
                multi_k_results["Qwen3DenseRetrieval"] = qwen3_dense.run_multi_k(samples, top_k_values)

            for top_k in top_k_values:
                results = {
                    method_name: method_results[top_k]
                    for method_name, method_results in multi_k_results.items()
                }
                out_dir = os.path.join(args.output, ds_name, f"k{top_k}")
                _save_baseline_results(out_dir, results)
                all_results[ds_name][f"k{top_k}"] = results
        elif len(top_k_values) == 1:
            out_dir = os.path.join(args.output, ds_name)
            results = run_all_baselines(
                samples=samples,
                generator=generator,
                generator_tokenizer=generator_tokenizer,
                top_k=top_k_values[0],
                device=args.device,
                output_dir=out_dir,
                max_new_tokens=args.max_new_tokens,
                compression_cfg=compression_cfg,
                skip_standard=args.skip_standard,
                batch_size=args.batch_size,
                qwen3_dense_model=args.qwen3_dense_model,
                standard_methods=standard_methods,
            )
            all_results[ds_name] = results
        else:
            all_results[ds_name] = {}
            for top_k in top_k_values:
                logger.info(f"Running baselines for {ds_name} with top_k={top_k}")
                out_dir = os.path.join(args.output, ds_name, f"k{top_k}")
                results = run_all_baselines(
                    samples=samples,
                    generator=generator,
                    generator_tokenizer=generator_tokenizer,
                    top_k=top_k,
                    device=args.device,
                    output_dir=out_dir,
                    max_new_tokens=args.max_new_tokens,
                    compression_cfg=compression_cfg,
                    skip_standard=args.skip_standard,
                    batch_size=args.batch_size,
                    qwen3_dense_model=args.qwen3_dense_model,
                    standard_methods=standard_methods,
                )
                all_results[ds_name][f"k{top_k}"] = results

    import json as _json
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "results_summary.json"), "w") as f:
        _json.dump(all_results, f, indent=2)
    logger.info(f"Summary saved to {args.output}/results_summary.json")

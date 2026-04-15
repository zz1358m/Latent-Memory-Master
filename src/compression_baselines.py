"""
Compression-based Baselines
============================
Four context-compression baselines, all using the same evaluation interface
as scripts/baselines.py.

LLMLinguaBaseline
    Token-prunes retrieved sentences with LLMLingua-2 before feeding to a
    frozen 8B LLM.  Approach: dense-retrieve top_k*2 sentences, compress to
    target_ratio, then generate with the same Llama-3.1-8B generator.
    Requires: pip install llmlingua
    Checkpoint (set in config): microsoft/llmlingua-2-xlm-roberta-large-meetingbank

AutoCompressorBaseline
    Compresses long context into soft summary vectors via
    princeton-nlp/AutoCompressor-Llama-2-7b-6k (Llama-2-7B fine-tuned).
    The model is self-contained — it both compresses context and generates answers.
    Requires: pip install auto-compressor
    Checkpoint: princeton-nlp/AutoCompressor-Llama-2-7b-6k

LCCBaseline
    Legacy Latent Context Compilation (LCC): compiles each document into
    N buffer tokens using src/compiler.py's LCCCompiler, then injects the
    buffer tokens as a soft prefix into a frozen generator.
    Requires: a trained LCCCompiler checkpoint (--lcc_checkpoint).

XRAGBaseline
    eXtreme RAG — each retrieved passage is compressed to a single projected
    embedding token by a fine-tuned Mistral-7B (OFA-Sys/xrag-7b).
    The projector maps a dense retrieval embedding (768-dim DRAGON+) to the
    LM hidden space (4096-dim).
    Requires: pip install xrag  or clone OFA-Sys/xRAG
    Checkpoints: OFA-Sys/xrag-7b  (generator)  +  facebook/dragon-plus-context-encoder (retrieval)
"""

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from scripts.baselines import (
    _build_prompt,
    _generate,
    _bm25_scores,
    _MAX_CONTEXT_TOKENS,
    _SYSTEM_PROMPT,
)
from src.evaluation import evaluate, print_results
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Baseline 4: LLMLingua
# ---------------------------------------------------------------------------

class LLMLinguaBaseline:
    """
    Compress the full context with LLMLingua-2 token pruning, then generate
    with the same frozen 8B generator as other baselines.

    All sentences are joined and fed directly to LLMLingua-2; no sentence-level
    pre-filtering is performed.  The compression_ratio controls how many tokens
    are retained (0.5 = 50% of tokens kept, 0.2 = 20% kept).

    Parameters
    ----------
    generator / generator_tokenizer : shared frozen LLM (Llama-3.1-8B-Instruct)
    llmlingua_model : HuggingFace model id or local path for LLMLingua-2
    compression_ratio : fraction of tokens to keep (0.0–1.0)
    """

    def __init__(
        self,
        generator,
        generator_tokenizer,
        llmlingua_model: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        compression_ratio: float = 0.5,
        max_new_tokens: int = 64,
        device: str = "cuda",
    ):
        self.generator           = generator
        self.generator_tokenizer = generator_tokenizer
        self.compression_ratio   = compression_ratio
        self.max_new_tokens      = max_new_tokens
        self.device              = device

        logger.info(f"Loading LLMLingua-2: {llmlingua_model}")
        try:
            from llmlingua import PromptCompressor
            self.compressor = PromptCompressor(
                llmlingua_model,
                use_llmlingua2=True,
                device_map=device,
            )
        except ImportError as exc:
            raise ImportError(
                "llmlingua is required for LLMLinguaBaseline. "
                "Install it with: pip install llmlingua"
            ) from exc

    def answer(self, question: str, sentences: List[str]) -> Tuple[str, int]:
        if not sentences:
            return "", 0

        context_raw = " ".join(sentences)

        # Compress all context tokens to the target ratio
        try:
            result = self.compressor.compress_prompt(
                context_raw,
                rate=self.compression_ratio,
                force_tokens=["\n", "?"],
            )
            context = result["compressed_prompt"]
        except Exception as e:
            logger.warning(f"LLMLingua compression failed ({e}); using raw context.")
            context = context_raw

        prompt = _build_prompt(self.generator_tokenizer, context, question)
        return _generate(self.generator, self.generator_tokenizer, prompt, self.max_new_tokens, self.device)

    def run(self, samples: List[Dict], desc: str = "LLMLingua") -> Dict[str, float]:
        predictions, references, token_counts = [], [], []

        for sample in tqdm(samples, desc=desc):
            sentences = sample.get("sentences", [])
            question  = sample["question"]
            answers   = sample.get("answers", [])
            if isinstance(answers, str):
                answers = [answers]
            if not sentences or not question or not answers:
                continue

            ans, n_tok = self.answer(question, sentences)
            predictions.append(ans)
            references.append(answers)
            token_counts.append(n_tok)

        if not predictions:
            return {"em": 0.0, "f1": 0.0, "rouge_l": 0.0, "avg_tokens": 0.0,
                    "recall_at_k": 0.0, "precision_at_k": 0.0}
        metrics = evaluate(predictions, references, token_counts)
        metrics["recall_at_k"]    = 0.0
        metrics["precision_at_k"] = 0.0
        return metrics


# ---------------------------------------------------------------------------
# Baseline 5: AutoCompressor
# ---------------------------------------------------------------------------

class AutoCompressorBaseline:
    """
    Compresses long context into soft summary vectors (princeton-nlp/AutoCompressor-*).

    The correct two-step API (from https://github.com/princeton-nlp/AutoCompressors):
      1. summary_vectors = model(context_ids, output_softprompt=True).softprompt
      2. model.generate(prompt_ids, softprompt=summary_vectors, ...)

    Parameters
    ----------
    model_name : HuggingFace model id, e.g. "princeton-nlp/AutoCompressor-Llama-2-7b-6k"
    segment_length : tokens per compression segment (default 512)
    summary_length : number of summary vectors per segment (default 50)
    """

    def __init__(
        self,
        model_name: str = "princeton-nlp/AutoCompressor-Llama-2-7b-6k",
        segment_length: int = 512,
        summary_length: int = 50,
        max_new_tokens: int = 64,
        device: str = "cuda",
        top_k: int = 5,
    ):
        self.segment_length  = segment_length
        self.summary_length  = summary_length
        self.max_new_tokens  = max_new_tokens
        self.device          = device
        self.top_k           = top_k

        logger.info(f"Loading AutoCompressor: {model_name}")
        LlamaAutoCompressorModel = self._import_llama_ac_model(model_name)
        from transformers import AutoTokenizer
        self.model = LlamaAutoCompressorModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    @staticmethod
    def _import_llama_ac_model(model_name: str):
        """
        Import LlamaAutoCompressorModel. Tries in order:
          1. pip-installed auto_compressor package
          2. auto_compressor.py inside the local model directory
        Raises ImportError if neither is available.
        """
        try:
            from auto_compressor import LlamaAutoCompressorModel
            return LlamaAutoCompressorModel
        except ImportError:
            pass

        import importlib.util
        ac_path = os.path.join(model_name, "auto_compressor.py")
        if os.path.isfile(ac_path):
            logger.info(f"Loading auto_compressor from local model dir: {ac_path}")
            spec = importlib.util.spec_from_file_location("auto_compressor", ac_path)
            mod  = importlib.util.module_from_spec(spec)
            sys.modules.setdefault("auto_compressor", mod)
            spec.loader.exec_module(mod)
            return mod.LlamaAutoCompressorModel

        raise ImportError(
            "auto_compressor package not found and auto_compressor.py not present in "
            f"'{model_name}'. Install with: pip install auto-compressor"
        )

    def _get_context(self, question: str, sentences: List[str]) -> str:
        k = min(self.top_k * 3, len(sentences))
        scores  = _bm25_scores(question, sentences)
        top_idx = sorted(np.argsort(scores)[::-1][:k].tolist())
        return " ".join(sentences[i] for i in top_idx)

    def answer(self, question: str, sentences: List[str]) -> Tuple[str, int]:
        if not sentences:
            return "", 0

        context = self._get_context(question, sentences)

        try:
            ctx_ids = self.tokenizer(
                context, add_special_tokens=False, return_tensors="pt",
                truncation=True, max_length=_MAX_CONTEXT_TOKENS - 64,
            ).input_ids.to(self.device)

            prompt_ids = self.tokenizer(
                f"Question: {question}\nAnswer:",
                add_special_tokens=False, return_tensors="pt",
            ).input_ids.to(self.device)

            with torch.no_grad():
                summary_vectors = self.model(ctx_ids, output_softprompt=True).softprompt
                out = self.model.generate(
                    prompt_ids,
                    softprompt=summary_vectors,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_ids = out[0, prompt_ids.shape[1]:]
            answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            return answer, ctx_ids.shape[1] + prompt_ids.shape[1]

        except Exception as e:
            logger.warning(f"AutoCompressor inference failed: {e}")
            return "", 0

    def answer_batch(self, questions: List[str], sentences_list: List[List[str]]) -> List[Tuple[str, int]]:
        """Compress each context individually, then batch the generation step."""
        B = len(questions)
        results = [("", 0)] * B

        try:
            # Step 1: compress each context individually → fixed-size softprompt per sample
            softprompts = []
            prompt_strs = []
            ctx_lens    = []
            for q, sentences in zip(questions, sentences_list):
                context = self._get_context(q, sentences)
                ctx_ids = self.tokenizer(
                    context, add_special_tokens=False, return_tensors="pt",
                    truncation=True, max_length=_MAX_CONTEXT_TOKENS - 64,
                ).input_ids.to(self.device)
                with torch.no_grad():
                    sv = self.model(ctx_ids, output_softprompt=True).softprompt  # (1, summary_length, H)
                softprompts.append(sv)
                prompt_strs.append(f"Question: {q}\nAnswer:")
                ctx_lens.append(ctx_ids.shape[1])

            # Step 2: batch generate — stack softprompts (all same size: summary_length vectors)
            softprompt_batch = torch.cat(softprompts, dim=0)  # (B, summary_length, H)

            qry_enc = self.tokenizer(
                prompt_strs, return_tensors="pt", padding=True,
                truncation=True, max_length=128, add_special_tokens=False,
            ).to(self.device)

            with torch.no_grad():
                out = self.model.generate(
                    qry_enc["input_ids"],
                    attention_mask=qry_enc["attention_mask"],
                    softprompt=softprompt_batch,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            qry_len = qry_enc["input_ids"].shape[1]
            for i in range(B):
                gen_ids = out[i, qry_len:]
                ans = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                results[i] = (ans, ctx_lens[i] + qry_len)

        except Exception as e:
            logger.warning(f"AutoCompressor batch inference failed: {e}")

        return results

    def run(self, samples: List[Dict], batch_size: int = 4) -> Dict[str, float]:
        predictions, references, token_counts = [], [], []

        valid = [
            s for s in samples
            if s.get("question") and s.get("answers")
        ]

        for i in tqdm(range(0, len(valid), batch_size), desc="AutoCompressor"):
            batch    = valid[i : i + batch_size]
            qs       = [s["question"] for s in batch]
            sents    = [s.get("sentences", []) for s in batch]
            ans_list = self.answer_batch(qs, sents)
            for s, (ans, n_tok) in zip(batch, ans_list):
                refs = s["answers"] if isinstance(s["answers"], list) else [s["answers"]]
                predictions.append(ans)
                references.append(refs)
                token_counts.append(n_tok)

        if not predictions:
            return {"em": 0.0, "f1": 0.0, "rouge_l": 0.0, "avg_tokens": 0.0,
                    "recall_at_k": 0.0, "precision_at_k": 0.0}
        metrics = evaluate(predictions, references, token_counts)
        metrics["recall_at_k"]    = 0.0
        metrics["precision_at_k"] = 0.0
        return metrics


# ---------------------------------------------------------------------------
# Baseline 6: LCC (Latent Context Compilation — legacy)
# ---------------------------------------------------------------------------

class LCCBaseline:
    """
    Uses the original LCC approach (N buffer tokens per document).
    Compiles a per-sample document into buffer tokens with LCCCompiler, then
    injects them as a soft prefix into a frozen generator via LatentInference.

    Parameters
    ----------
    compiler_checkpoint : path to a trained LCCCompiler checkpoint (.pt)
    generator / generator_tokenizer : frozen LLM shared with other baselines
    num_buffer_tokens : number of buffer tokens per document (from checkpoint config)
    """

    def __init__(
        self,
        compiler_checkpoint: str,
        generator,
        generator_tokenizer,
        compressor_name: str,
        num_buffer_tokens: int = 64,
        max_new_tokens: int = 64,
        device: str = "cuda",
    ):
        self.generator           = generator
        self.generator_tokenizer = generator_tokenizer
        self.max_new_tokens      = max_new_tokens
        self.device              = device

        logger.info(f"Loading LCCCompiler from {compiler_checkpoint}")
        from src.compiler import LCCCompiler
        self.compiler = LCCCompiler(
            model_name=compressor_name,
            num_buffer_tokens=num_buffer_tokens,
        ).to(device)
        self.compiler.eval()

        if os.path.exists(compiler_checkpoint):
            self.compiler.load(compiler_checkpoint)
        else:
            logger.warning(
                f"LCC checkpoint not found at '{compiler_checkpoint}'. "
                "Running with untrained weights."
            )

    def answer(self, question: str, sentences: List[str]) -> Tuple[str, int]:
        if not sentences:
            return "", 0

        document = " ".join(sentences)
        try:
            with torch.no_grad():
                # Compile document → buffer tokens (N, hidden)
                memory = self.compiler.compile(document)   # (N, hidden)
                # Project to generator space if needed
                if hasattr(self.compiler, "cross_proj"):
                    projected = self.compiler.cross_proj(memory.float())  # (N, gen_hidden)
                else:
                    projected = memory  # assume same space

            # Build input_embeds: [buffer_tokens | query_tokens]
            embed_fn  = self.generator.get_input_embeddings()
            dtype     = next(self.generator.parameters()).dtype
            query_str = f"Question: {question}\nAnswer:"
            q_ids     = self.generator_tokenizer(
                query_str, return_tensors="pt", add_special_tokens=True
            )["input_ids"].to(self.device)
            q_embs    = embed_fn(q_ids).to(dtype)           # (1, Lq, H)

            buf_embs  = projected.unsqueeze(0).to(dtype)    # (1, N, H)
            input_emb = torch.cat([buf_embs, q_embs], dim=1)
            n_input   = input_emb.shape[1]

            with torch.no_grad():
                out = self.generator.generate(
                    inputs_embeds=input_emb,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.generator_tokenizer.eos_token_id,
                )
            generated_ids = out[0, n_input:]
            answer = self.generator_tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
            return answer, n_input

        except Exception as e:
            logger.warning(f"LCC inference failed: {e}")
            return "", 0

    def run(self, samples: List[Dict]) -> Dict[str, float]:
        predictions, references, token_counts = [], [], []

        for sample in tqdm(samples, desc="LCC"):
            sentences = sample.get("sentences", [])
            question  = sample["question"]
            answers   = sample.get("answers", [])
            if isinstance(answers, str):
                answers = [answers]
            if not question or not answers:
                continue

            ans, n_tok = self.answer(question, sentences)
            predictions.append(ans)
            references.append(answers)
            token_counts.append(n_tok)

        if not predictions:
            return {"em": 0.0, "f1": 0.0, "rouge_l": 0.0, "avg_tokens": 0.0,
                    "recall_at_k": 0.0, "precision_at_k": 0.0}
        metrics = evaluate(predictions, references, token_counts)
        metrics["recall_at_k"]    = 0.0
        metrics["precision_at_k"] = 0.0
        return metrics


# ---------------------------------------------------------------------------
# Baseline 7: xRAG
# ---------------------------------------------------------------------------

# Special token the xRAG model uses as a placeholder for each retrieved passage
_XRAG_TOKEN = "<xRAG>"
# Prompt template matching the xRAG training format (Mistral [INST] style)
_XRAG_TEMPLATE = (
    "[INST] Refer to the background document and answer the questions:\n\n"
    "Background: {document}\n\n"
    "Question: {question} [/INST] The answer is:"
)


class XRAGBaseline:
    """
    eXtreme RAG: retrieved passage is compressed to a single soft token
    fed to a fine-tuned Mistral-7B (Hannibal046/xrag-7b).

    Inference pipeline (https://github.com/Hannibal046/xRAG tutorial):
      1. Encode documents (full paragraphs) and query with SFR-Embedding-Mistral
         using last-token pooling — NO normalization.
      2. Dot-product retrieval → top-1 document.
      3. Build prompt with a single <xRAG> placeholder in the Background field.
      4. model.generate(input_ids=..., retrieval_embeds=doc_emb.unsqueeze(0))
         XMistralForCausalLM internally projects the embedding and replaces the
         <xRAG> token; output contains only generated tokens.

    Parameters
    ----------
    xrag_model_name        : HuggingFace id or local path (Hannibal046/xrag-7b)
    retrieval_encoder_name : HuggingFace id or local path (Salesforce/SFR-Embedding-Mistral)
    """

    def __init__(
        self,
        xrag_model_name: str = "Hannibal046/xrag-7b",
        retrieval_encoder_name: str = "Salesforce/SFR-Embedding-Mistral",
        xrag_src: str = "",
        max_new_tokens: int = 64,
        device: str = "cuda",
    ):
        self.max_new_tokens = max_new_tokens
        self.device         = device

        from transformers import AutoModel, AutoTokenizer

        # --- Load SFR-Embedding-Mistral retrieval encoder ---
        logger.info(f"Loading SFR retrieval encoder: {retrieval_encoder_name}")
        self.ret_tokenizer = AutoTokenizer.from_pretrained(retrieval_encoder_name)
        self.ret_encoder   = AutoModel.from_pretrained(
            retrieval_encoder_name, torch_dtype=torch.bfloat16,
        ).to(device).eval()
        for p in self.ret_encoder.parameters():
            p.requires_grad_(False)

        # --- Load xRAG LLM (XMistralForCausalLM from xRAG source) ---
        # The HuggingFace checkpoint has no custom model code, so trust_remote_code
        # only loads plain MistralForCausalLM. Must import from the xRAG repo source.
        logger.info(f"Loading xRAG model: {xrag_model_name}")
        XMistralForCausalLM = self._import_xmistral(xrag_src)

        self.xrag_tokenizer = AutoTokenizer.from_pretrained(
            xrag_model_name,
            add_eos_token=False,
            use_fast=False,
            padding_side="left",
        )
        if self.xrag_tokenizer.pad_token is None:
            self.xrag_tokenizer.pad_token = self.xrag_tokenizer.eos_token

        self.xrag_model = XMistralForCausalLM.from_pretrained(
            xrag_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()
        for p in self.xrag_model.parameters():
            p.requires_grad_(False)

        # Register the <xRAG> token ID so the model knows where to inject embeddings
        xrag_token_id = self.xrag_tokenizer.convert_tokens_to_ids(_XRAG_TOKEN)
        self.xrag_model.set_xrag_token_id(xrag_token_id)

    @staticmethod
    def _import_xmistral(xrag_src: str):
        """
        Import XMistralForCausalLM from the xRAG source repo.
        The HuggingFace checkpoint contains no custom model code, so
        trust_remote_code=True only loads plain MistralForCausalLM.

        Adds the repo root to sys.path so all internal xRAG imports resolve,
        then does a normal package import.

        Tries in order:
          1. xrag_src argument (path to cloned xRAG repo)
          2. XRAG_SRC environment variable
          3. Common sibling/parent directory names ('xRAG', 'xrag')
        """
        import importlib.util

        def _try_load(repo_root: str):
            model_file = os.path.join(repo_root, "src", "model", "xMistral", "modeling_xmistral.py")
            if not os.path.isfile(model_file):
                return None
            spec = importlib.util.spec_from_file_location("xrag_modeling_xmistral", model_file)
            mod  = importlib.util.module_from_spec(spec)
            sys.modules["xrag_modeling_xmistral"] = mod
            spec.loader.exec_module(mod)
            return mod.XMistralForCausalLM

        # 1. explicit argument
        if xrag_src:
            cls = _try_load(xrag_src)
            if cls is not None:
                return cls
            raise ImportError(f"modeling_xmistral.py not found under xrag_src='{xrag_src}'")

        # 2. environment variable
        env_src = os.environ.get("XRAG_SRC", "")
        if env_src:
            cls = _try_load(env_src)
            if cls is not None:
                return cls

        # 3. common locations relative to this file
        _here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for candidate in ["xRAG", "xrag", "../xRAG", "../xrag"]:
            cls = _try_load(os.path.join(_here, candidate))
            if cls is not None:
                return cls

        raise ImportError(
            "XMistralForCausalLM not found. Clone https://github.com/Hannibal046/xRAG "
            "and pass xrag_src='/path/to/xRAG' or set the XRAG_SRC environment variable."
        )

    @staticmethod
    def _last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Last-token pooling matching SFR modeling_sfr.py exactly."""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_state[:, -1]
        else:
            seq_len = attention_mask.sum(dim=1) - 1
            bs      = last_hidden_state.shape[0]
            return last_hidden_state[torch.arange(bs, device=last_hidden_state.device), seq_len]

    def _encode(self, texts: List[str]) -> torch.Tensor:
        """Raw unnormalized SFR embeddings → (N, 4096). No normalization (matches xRAG training)."""
        enc = self.ret_tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=180,
        ).to(self.device)
        with torch.no_grad():
            out = self.ret_encoder(**enc)
        return self._last_token_pool(out.last_hidden_state, enc["attention_mask"])

    @staticmethod
    def _build_paragraphs(sentences: List[str], paragraph_map: Optional[List[str]]) -> List[str]:
        """
        Reconstruct full paragraphs from sentences + paragraph_map.
        Returns 'Title | sentence1 sentence2 ...' strings matching xRAG document format.
        Falls back to raw sentences if paragraph_map is absent.
        """
        if not paragraph_map or len(paragraph_map) != len(sentences):
            return sentences
        from collections import OrderedDict
        paras: dict = OrderedDict()
        for sent, title in zip(sentences, paragraph_map):
            paras.setdefault(title, []).append(sent)
        return [f"{title} | {' '.join(sents)}" for title, sents in paras.items()]

    def answer(self, question: str, sentences: List[str],
               paragraph_map: Optional[List[str]] = None) -> Tuple[str, int]:
        if not sentences:
            return "", 0

        try:
            # Reconstruct full paragraphs — xRAG was trained on passage-level documents
            documents = self._build_paragraphs(sentences, paragraph_map)
            # Step 1: encode query and documents — NO normalization (matches xRAG training)
            q_emb   = self._encode([question])          # (1, D)
            d_embs  = self._encode(documents)           # (N, D)

            # Dot-product retrieval — top-1 only (xRAG uses a single <xRAG> token)
            sims    = torch.matmul(q_emb, d_embs.T).squeeze(0)   # (N,)
            top_idx = torch.topk(sims, 1).indices[0].item()

            # Step 2: single retrieved document embedding, shape (1, D)
            doc_embs = d_embs[top_idx].unsqueeze(0)     # (1, D) — unnormalized

            # Step 3: single <xRAG> placeholder
            xrag_placeholders = _XRAG_TOKEN
            prompt    = _XRAG_TEMPLATE.format(document=xrag_placeholders, question=question)
            input_ids = self.xrag_tokenizer(
                prompt, return_tensors="pt"
            ).input_ids.to(self.device)

            # Step 4: generate — XMistralForCausalLM uses inputs_embeds internally,
            # so output contains only generated tokens; decode the full output directly
            with torch.no_grad():
                out = self.xrag_model.generate(
                    input_ids=input_ids,
                    retrieval_embeds=doc_embs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.xrag_tokenizer.eos_token_id,
                )
            answer = self.xrag_tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
            return answer, input_ids.shape[1]

        except Exception as e:
            logger.warning(f"xRAG inference failed: {e}")
            return "", 0

    def run(self, samples: List[Dict]) -> Dict[str, float]:
        predictions, references, token_counts = [], [], []
        recall_scores: List[float] = []

        for sample in tqdm(samples, desc="xRAG"):
            sentences     = sample.get("sentences", [])
            question      = sample["question"]
            answers       = sample.get("answers", [])
            paragraph_map = sample.get("paragraph_map", None)
            if isinstance(answers, str):
                answers = [answers]
            if not question or not answers:
                continue

            ans, n_tok = self.answer(question, sentences, paragraph_map)
            predictions.append(ans)
            references.append(answers)
            token_counts.append(n_tok)

            # Recall@1: did the retrieved paragraph contain any positive sentence?
            pos_indices = sample.get("positive_indices", [])
            if pos_indices and sentences:
                try:
                    documents = self._build_paragraphs(sentences, paragraph_map)
                    q_emb  = self._encode([question])
                    d_embs = self._encode(documents)
                    sims   = torch.matmul(q_emb, d_embs.T).squeeze(0)
                    top_para_idx = torch.topk(sims, 1).indices[0].item()

                    # Collect sentence indices that belong to the retrieved paragraph
                    if paragraph_map and len(paragraph_map) == len(sentences):
                        retrieved_title = list(dict.fromkeys(paragraph_map))[top_para_idx]
                        retrieved_sent_indices = {
                            i for i, t in enumerate(paragraph_map) if t == retrieved_title
                        }
                    else:
                        # No paragraph_map: each "document" is a single sentence
                        retrieved_sent_indices = {top_para_idx}

                    hit = int(bool(retrieved_sent_indices & set(pos_indices)))
                    recall_scores.append(hit)
                except Exception:
                    pass

        if not predictions:
            return {"em": 0.0, "f1": 0.0, "rouge_l": 0.0, "avg_tokens": 0.0, "recall_at_1": 0.0}
        metrics = evaluate(predictions, references, token_counts)
        metrics["recall_at_1"] = float(np.mean(recall_scores)) if recall_scores else 0.0
        return metrics

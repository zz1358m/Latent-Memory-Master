"""
RAG Baseline
============
Standard Retrieval-Augmented Generation pipeline for comparison:

  1. Chunk documents into fixed-size text segments
  2. Embed chunks with a lightweight sentence encoder
  3. Store in FAISS
  4. At query time: retrieve top-k chunks → concat as text → feed to LLM

This represents the conventional approach that the latent retrieval method
aims to improve upon (fewer tokens, same or better accuracy).
"""

import logging
from typing import Dict, List, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def chunk_text(text: str, tokenizer, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """
    Split text into overlapping token-level chunks.

    Args
    ----
    text       : raw document text
    tokenizer  : HuggingFace tokenizer (for accurate token counting)
    chunk_size : max tokens per chunk
    overlap    : overlap between consecutive chunks (in tokens)

    Returns
    -------
    chunks : list of text strings
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


class RAGBaseline:
    """
    Dense retrieval RAG baseline.

    Usage
    -----
    rag = RAGBaseline(llm_name="meta-llama/Llama-3.2-1B-Instruct",
                      embedder_name="sentence-transformers/all-MiniLM-L6-v2")
    rag.add_documents(doc_list)   # list of {"id": ..., "text": ...}
    rag.build_index()

    answer, n_tokens = rag.answer(query, top_k=3)
    """

    def __init__(
        self,
        llm_name: str,
        embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        max_new_tokens: int = 128,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_new_tokens = max_new_tokens

        # Lightweight sentence encoder for retrieval
        logger.info(f"Loading embedder: {embedder_name}")
        self.embedder = SentenceTransformer(embedder_name)
        self.embed_dim = self.embedder.get_sentence_embedding_dimension()

        # LLM for generation
        logger.info(f"Loading LLM: {llm_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.llm.eval()

        # Chunk storage
        self.chunks: List[str] = []         # raw text
        self.chunk_doc_ids: List[str] = []  # which doc each chunk came from
        self.index: faiss.Index = None

    @property
    def device(self):
        return next(self.llm.parameters()).device

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Dict]):
        """
        Chunk and index a list of documents.

        Args
        ----
        documents : list of {"id": str, "text": str}
        """
        logger.info(f"Chunking {len(documents)} documents...")
        for doc in documents:
            doc_chunks = chunk_text(
                doc["text"], self.tokenizer, self.chunk_size, self.chunk_overlap
            )
            self.chunks.extend(doc_chunks)
            self.chunk_doc_ids.extend([doc["id"]] * len(doc_chunks))

        logger.info(f"Total chunks: {len(self.chunks)}")

    def build_index(self):
        """Embed all chunks and build a FAISS flat IP index."""
        logger.info("Embedding chunks...")
        embeddings = self.embedder.encode(
            self.chunks, batch_size=64, show_progress_bar=False, normalize_embeddings=True
        )
        embeddings = embeddings.astype(np.float32)

        self.index = faiss.IndexFlatIP(self.embed_dim)
        self.index.add(embeddings)
        logger.info(f"RAG index built: {len(self.chunks)} chunks, dim={self.embed_dim}")

    # ------------------------------------------------------------------
    # Retrieval + Generation
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3) -> Tuple[List[str], List[float]]:
        """Retrieve top-k text chunks for a query."""
        if self.index is None:
            raise RuntimeError("Call build_index() before retrieve().")

        q_emb = self.embedder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
        scores, indices = self.index.search(q_emb, top_k)

        retrieved_chunks = [self.chunks[i] for i in indices[0] if i >= 0]
        retrieved_scores = [float(s) for s in scores[0] if s > -1]
        return retrieved_chunks, retrieved_scores

    def answer(self, query: str, top_k: int = 3) -> Tuple[str, int]:
        """
        Full RAG pipeline: retrieve → prompt → generate.

        Returns
        -------
        answer   : generated text string
        n_tokens : number of input tokens (context length used)
        """
        chunks, _ = self.retrieve(query, top_k)
        context = "\n\n".join(chunks)

        prompt = (
            f"Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        ).to(self.device)
        n_tokens = inputs.input_ids.shape[1]

        with torch.no_grad():
            out = self.llm.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        generated = out[0, n_tokens:]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return answer, n_tokens

"""
Memory Bank
===========
Stores compiled latent memories and provides fast similarity retrieval via FAISS.

Each entry in the bank is:
  - doc_id      : str, unique identifier
  - memory      : (N, H) float32 tensor — the compiled latent memory
  - retrieval_vec : (D,) float32 — L2-normalized projection of memory (for FAISS)
  - text        : str, original document text (for RAG comparison / debug)
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    doc_id: str
    memory: torch.Tensor          # (N, H) — the actual latent memory
    retrieval_vec: np.ndarray     # (D,)   — FAISS retrieval vector
    text: str = ""                # original document text


class MemoryBank:
    """
    FAISS-backed store for compiled latent memories.

    Usage
    -----
    bank = MemoryBank(retrieval_dim=256)
    bank.add("doc_0", memory_tensor, retrieval_vec, text="...")
    bank.build_index()

    ids, memories = bank.search(query_vec, k=3)
    """

    def __init__(self, retrieval_dim: int = 256, index_type: str = "flat"):
        self.retrieval_dim = retrieval_dim
        self.index_type = index_type
        self.entries: List[MemoryEntry] = []
        self._id_to_idx: Dict[str, int] = {}
        self.index: Optional[faiss.Index] = None

    # ------------------------------------------------------------------
    # Adding entries
    # ------------------------------------------------------------------

    def add(
        self,
        doc_id: str,
        memory: torch.Tensor,
        retrieval_vec: torch.Tensor,
        text: str = "",
    ):
        """
        Add one compiled memory to the bank.

        Args
        ----
        doc_id       : unique string identifier
        memory       : (N, H) compiled latent memory tensor
        retrieval_vec: (D,) normalized retrieval embedding (from retrieval_head)
        text         : original document text (optional, stored for reference)
        """
        if doc_id in self._id_to_idx:
            logger.warning(f"doc_id '{doc_id}' already in bank — skipping.")
            return

        vec = retrieval_vec.detach().cpu().float().numpy()
        entry = MemoryEntry(
            doc_id=doc_id,
            memory=memory.detach().cpu().float(),
            retrieval_vec=vec,
            text=text,
        )
        self._id_to_idx[doc_id] = len(self.entries)
        self.entries.append(entry)

    def add_batch(
        self,
        doc_ids: List[str],
        memories: List[torch.Tensor],
        retrieval_vecs: torch.Tensor,
        texts: List[str] = None,
    ):
        """Add multiple entries at once."""
        if texts is None:
            texts = [""] * len(doc_ids)
        for i, (did, mem, rvec) in enumerate(zip(doc_ids, memories, retrieval_vecs)):
            self.add(did, mem, rvec, texts[i])

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self):
        """
        Build FAISS index over all stored retrieval vectors.
        Call this after adding all documents.
        """
        if len(self.entries) == 0:
            raise ValueError("No entries in memory bank. Add documents first.")

        vecs = np.stack([e.retrieval_vec for e in self.entries], axis=0)
        # Normalize just in case (retrieval_head already does this, but be safe)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.clip(norms, 1e-8, None)

        if self.index_type == "flat":
            # Exact inner product search (equiv. to cosine similarity on normalized vecs)
            self.index = faiss.IndexFlatIP(self.retrieval_dim)
        elif self.index_type == "ivf":
            # Approximate search — faster for large corpora (>100k docs)
            nlist = min(100, len(self.entries) // 10)
            quantizer = faiss.IndexFlatIP(self.retrieval_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.retrieval_dim, nlist, faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(vecs)
        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")

        self.index.add(vecs)
        logger.info(f"FAISS index built: {len(self.entries)} entries, dim={self.retrieval_dim}")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self, query_vec: np.ndarray, k: int = 3
    ) -> Tuple[List[str], List[torch.Tensor], List[float]]:
        """
        Retrieve top-k latent memories by cosine similarity.

        Args
        ----
        query_vec : (D,) float32 numpy array, L2-normalized
        k         : number of results

        Returns
        -------
        doc_ids  : list of matched doc_id strings
        memories : list of (N, H) tensors — the compiled latent memories
        scores   : list of similarity scores
        """
        if self.index is None:
            raise RuntimeError("Call build_index() before search().")

        # Normalize query
        query_vec = query_vec / np.clip(np.linalg.norm(query_vec), 1e-8, None)
        query_vec = query_vec.reshape(1, -1).astype(np.float32)

        k = min(k, len(self.entries))
        scores, indices = self.index.search(query_vec, k)

        doc_ids, memories, sim_scores = [], [], []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            entry = self.entries[idx]
            doc_ids.append(entry.doc_id)
            memories.append(entry.memory)
            sim_scores.append(float(score))

        return doc_ids, memories, sim_scores

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, save_dir: str):
        """Save the memory bank to disk."""
        os.makedirs(save_dir, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, os.path.join(save_dir, "faiss.index"))

        # Save memories and metadata
        data = {
            "doc_ids": [e.doc_id for e in self.entries],
            "memories": [e.memory for e in self.entries],
            "retrieval_vecs": [e.retrieval_vec for e in self.entries],
            "texts": [e.text for e in self.entries],
            "retrieval_dim": self.retrieval_dim,
            "index_type": self.index_type,
        }
        torch.save(data, os.path.join(save_dir, "memories.pt"))
        logger.info(f"Memory bank saved to {save_dir} ({len(self.entries)} entries)")

    @classmethod
    def load(cls, save_dir: str) -> "MemoryBank":
        """Load a memory bank from disk."""
        data = torch.load(os.path.join(save_dir, "memories.pt"), map_location="cpu")
        bank = cls(retrieval_dim=data["retrieval_dim"], index_type=data["index_type"])

        for did, mem, rvec, txt in zip(
            data["doc_ids"], data["memories"], data["retrieval_vecs"], data["texts"]
        ):
            entry = MemoryEntry(doc_id=did, memory=mem, retrieval_vec=rvec, text=txt)
            bank._id_to_idx[did] = len(bank.entries)
            bank.entries.append(entry)

        bank.index = faiss.read_index(os.path.join(save_dir, "faiss.index"))
        logger.info(f"Memory bank loaded from {save_dir} ({len(bank.entries)} entries)")
        return bank

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return f"MemoryBank(n={len(self.entries)}, dim={self.retrieval_dim})"

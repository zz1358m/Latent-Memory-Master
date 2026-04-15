"""
Contrastive Latent Retriever
=============================
Trains a retriever to match query embeddings against compressed sentence embeddings.

Loss: NT-Xent (InfoNCE)
-----------------------
For a batch of (query, positive_sentence) pairs:

  L = -1/N * Σ_i log(
      exp(sim(q_i, s_i^+) / τ)
      ─────────────────────────────────────────────────────
      Σ_j [exp(sim(q_i, s_j^+) / τ) + exp(sim(q_i, s_j^-) / τ)]
  )

where:
  q_i   = L2-normalized query embedding
  s_i^+ = L2-normalized compressed embedding of a positive sentence
  s_j^- = L2-normalized compressed embeddings of negative sentences
  τ     = temperature (default 0.07)

Negatives:
  - In-batch negatives: positives of OTHER questions in the same batch
  - Hard negatives:     other sentences from the SAME context as q_i

At inference:
  - Query is embedded via embed_query() → (retrieval_dim,)
  - All sentence embeddings are pre-computed and stored in FAISS
  - Top-k nearest neighbors are retrieved
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def nt_xent_loss(
    query_embs: torch.Tensor,          # (B, D)  L2-normalized
    positive_embs: torch.Tensor,       # (B, D)  L2-normalized
    negative_embs: torch.Tensor = None,  # (B*K, D) hard negatives, optional
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    NT-Xent (InfoNCE) contrastive loss.

    Uses in-batch negatives (positives of other queries) + optional hard negatives.

    Args
    ----
    query_embs    : (B, D) query embeddings, L2-normalized
    positive_embs : (B, D) positive sentence embeddings, L2-normalized
    negative_embs : (N, D) hard negative embeddings (optional), L2-normalized
    temperature   : softmax temperature τ

    Returns
    -------
    loss : scalar tensor
    """
    B = query_embs.shape[0]

    # Similarity matrix between queries and all positive keys
    # sim[i, j] = cosine_sim(q_i, s_j^+)
    sim_matrix = torch.matmul(query_embs, positive_embs.T) / temperature  # (B, B)

    # If hard negatives provided, append them as extra columns
    if negative_embs is not None and negative_embs.shape[0] > 0:
        sim_neg = torch.matmul(query_embs, negative_embs.T) / temperature  # (B, N)
        sim_matrix = torch.cat([sim_matrix, sim_neg], dim=1)               # (B, B+N)

    # The diagonal is the positive pair (q_i, s_i^+)
    labels = torch.arange(B, device=query_embs.device)

    loss = F.cross_entropy(sim_matrix, labels)
    return loss


def nt_xent_loss_multi_positive(
    query_embs: torch.Tensor,              # (B, D)  L2-normalized
    positive_embs_list: List[torch.Tensor],  # list of B tensors, each (K_i, D)
    negative_embs: torch.Tensor = None,    # (N, D) hard negatives, optional
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Multi-positive NT-Xent (InfoNCE) loss.

    Each query q_i may have K_i >= 1 ground-truth positive sentences.
    For each positive p_ij the loss is:

        -log( exp(sim(q_i, p_ij) / τ) / Σ_{all keys} exp(sim(q_i, key) / τ) )

    All positives across all queries share the same denominator (full key pool).
    Loss is averaged over all (query, positive) pairs.

    Args
    ----
    query_embs         : (B, D)
    positive_embs_list : list of length B; element i is (K_i, D)
    negative_embs      : (N, D) optional hard negatives
    temperature        : τ

    Returns
    -------
    loss : scalar tensor
    """
    # Build the full key pool: all positives + hard negatives
    all_pos = torch.cat(positive_embs_list, dim=0)   # (total_pos, D)
    if negative_embs is not None and negative_embs.shape[0] > 0:
        keys = torch.cat([all_pos, negative_embs], dim=0)  # (total_pos+N, D)
    else:
        keys = all_pos                                       # (total_pos, D)

    # Precompute log-partition for each query: log Σ_k exp(sim(q_i, k) / τ)
    all_sims = torch.matmul(query_embs, keys.T) / temperature  # (B, total_pos+N)
    log_Z = torch.logsumexp(all_sims, dim=-1)                  # (B,)

    # For each query, for each of its positives, compute log numerator and subtract log_Z
    losses = []
    offset = 0
    for i, pos_embs in enumerate(positive_embs_list):
        K_i = pos_embs.shape[0]
        q_i = query_embs[i:i+1]                              # (1, D)
        pos_sims = torch.matmul(q_i, pos_embs.T) / temperature  # (1, K_i)
        # loss per positive: -(log_num - log_Z_i)
        loss_i = -(pos_sims - log_Z[i]).mean()
        losses.append(loss_i)
        offset += K_i

    return torch.stack(losses).mean()


class ContrastiveRetriever(nn.Module):
    """
    Wraps the contrastive loss computation.

    The actual query and sentence encoders live in SentenceAutoencoder;
    this module only handles the loss and provides a clean interface.

    In practice, during training we call:
        retriever.contrastive_loss(query_embs, pos_embs, hard_neg_embs)

    and during inference we use:
        retriever.search(query_emb, sentence_bank)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def contrastive_loss(
        self,
        query_embs: torch.Tensor,                        # (B, D)
        positive_embs_list: List[torch.Tensor],          # list of B tensors, each (K_i, D)
        hard_negative_embs: Optional[torch.Tensor] = None,  # (N, D)
    ) -> torch.Tensor:
        """
        Compute multi-positive NT-Xent loss for a batch.

        Each query may have K_i >= 1 ground-truth positive sentences.
        All embeddings must be L2-normalized before calling this.
        """
        return nt_xent_loss_multi_positive(
            query_embs, positive_embs_list, hard_negative_embs, self.temperature
        )

    @staticmethod
    def search(
        query_emb: np.ndarray,
        sentence_embs: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[List[int], List[float]]:
        """
        Brute-force cosine similarity search (used when FAISS is not available).

        Args
        ----
        query_emb     : (D,) numpy array, L2-normalized
        sentence_embs : (N, D) numpy array, L2-normalized
        top_k         : number of results

        Returns
        -------
        indices : list of top-k sentence indices
        scores  : list of cosine similarity scores
        """
        scores = sentence_embs @ query_emb   # (N,)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return list(top_idx), [float(scores[i]) for i in top_idx]

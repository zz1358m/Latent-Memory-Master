"""
Sentence Compressor & Decoder
==============================
Core components of the latent sentence retrieval system.

SentenceCompressor
------------------
Compresses one sentence into exactly 1 token embedding.
  - Uses Llama-3.2-1B-Instruct + LoRA adapter named "compress"
  - One learnable [MEM] token prepended to the sentence
  - Takes the [MEM] hidden state at the last layer as the compressed vector

    [MEM | s_1 | s_2 | ... | s_n]
       ↓ (LoRA-LLM)
    h_MEM  ← compressed embedding  (hidden_dim,)

SentenceDecoder
---------------
Reconstructs the original sentence from 1 token embedding.
  - Same frozen base LLM + separate LoRA adapter named "decode"
  - Input: compressed embedding as the first input embedding
  - Trained autoregressively to predict the original sentence tokens

    [h_MEM | <bos>]  → predict [s_1 | s_2 | ... | s_n | <eos>]

Both adapters share the same frozen base weights (parameter-efficient).

CrossModelProjection
--------------------
Projects the 1B compressed embedding into the 8B generator's embedding space.

    compressed (2048,) → CrossModelProjection → projected (4096,)

A two-layer MLP + LayerNorm, trained jointly with the rest of the system.
It bridges the two model families so the frozen 8B LLM can directly consume
latent tokens produced by the 1B compressor.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cross-model projection: 1B hidden space → 8B hidden space
# ---------------------------------------------------------------------------

class CrossModelProjection(nn.Module):
    """
    Learnable projection from the 1B compressor's hidden space to the
    8B generator's embedding space.

    Architecture: Linear → GELU → Linear → LayerNorm
    This MLP is expressive enough to remap the distribution while being
    small relative to the full models.

    Trained jointly with the compressor via:
      - Distillation loss (align projected latent with 8B text hidden states)
      - Contrastive loss gradients (through retrieval_proj, independent)
    """

    def __init__(self, in_dim: int = 2048, out_dim: int = 4096):
        super().__init__()
        mid_dim = (in_dim + out_dim) // 2   # 3072 for 2048→4096
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(mid_dim, out_dim, dtype=torch.float32),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : (..., in_dim) float32

        Returns
        -------
        projected : (..., out_dim) float32
        """
        return self.net(x.float())


class SentenceAutoencoder(nn.Module):
    """
    Wraps the compressor (LoRA "compress") and decoder (LoRA "decode")
    on top of a single frozen Llama-1B base model.

    The two LoRA adapters are switched via PEFT's set_adapter() API,
    so base model weights are loaded only once.

    Also contains CrossModelProjection to bridge 1B → 8B embedding spaces.
    The 8B generator model itself is NOT loaded here — it lives in inference.py
    and distillation.py.
    """

    def __init__(
        self,
        model_name: str,
        compress_lora_r: int = 16,
        compress_lora_alpha: int = 32,
        decode_lora_r: int = 16,
        decode_lora_alpha: int = 32,
        query_lora_r: int = 16,
        query_lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        compress_target_modules: list = None,
        decode_target_modules: list = None,
        query_target_modules: list = None,
        retrieval_dim: int = 256,
        generator_hidden: int = 4096,   # hidden dim of the 8B generator
        num_latent_tokens: int = 1,
    ):
        super().__init__()
        self.model_name = model_name
        self.retrieval_dim = retrieval_dim
        self.num_latent_tokens = int(num_latent_tokens)
        if self.num_latent_tokens < 1:
            raise ValueError("num_latent_tokens must be >= 1")

        if compress_target_modules is None:
            compress_target_modules = ["q_proj", "v_proj"]
        if decode_target_modules is None:
            decode_target_modules = ["q_proj", "v_proj", "o_proj"]
        if query_target_modules is None:
            query_target_modules = ["q_proj", "v_proj"]

        # --- Tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- Frozen base model ---
        logger.info(f"Loading base model: {model_name}")
        base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.hidden_size = base.config.hidden_size

        # --- Add first LoRA adapter: "compress" ---
        compress_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=compress_lora_r,
            lora_alpha=compress_lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=compress_target_modules,
            bias="none",
        )
        self.model = get_peft_model(base, compress_cfg, adapter_name="compress")

        # --- Add second LoRA adapter: "decode" ---
        decode_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=decode_lora_r,
            lora_alpha=decode_lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=decode_target_modules,
            bias="none",
        )
        self.model.add_adapter("decode", decode_cfg)

        # --- Add third LoRA adapter: "query" ---
        # Dedicated query encoder — trained only by L_contrast (query side).
        # Shares frozen base weights but has NO reconstruction gradient,
        # so it can specialise in retrieval without conflicting with L_recon.
        query_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=query_lora_r,
            lora_alpha=query_lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=query_target_modules,
            bias="none",
        )
        self.model.add_adapter("query", query_cfg)

        # --- Add fourth LoRA adapter: "query_decode" ---
        # Dedicated decoder for query reconstruction (CE loss).
        # Kept separate from "decode" so sentence and query decoders
        # specialise independently without gradient interference.
        query_decode_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=decode_lora_r,
            lora_alpha=decode_lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=decode_target_modules,
            bias="none",
        )
        self.model.add_adapter("query_decode", query_decode_cfg)

        # Activate "compress" by default
        self._set_adapter("compress")
        self.model.print_trainable_parameters()

        # --- Learnable [MEM] token (shared across all sentences) ---
        # Initialized small so it starts in the model's embedding distribution
        self.mem_embedding = nn.Parameter(
            torch.randn(self.num_latent_tokens, self.hidden_size) * 0.02
        )

        # --- Projection head: compressed embedding → retrieval space ---
        # Used for FAISS indexing and contrastive learning (1B dim → retrieval_dim)
        self.retrieval_proj = nn.Sequential(
            nn.Linear(self.hidden_size, retrieval_dim, dtype=torch.float32),
            nn.LayerNorm(retrieval_dim),
        )

        # --- Cross-model projection: 1B hidden → 8B generator hidden ---
        # Bridges the compressor and the generator at inference + distillation time.
        # Input:  compressed embedding  (self.hidden_size = 2048 for 1B)
        # Output: projected embedding   (generator_hidden = 4096 for 8B)
        self.cross_proj = CrossModelProjection(
            in_dim=self.hidden_size,
            out_dim=generator_hidden,
        )
        self.generator_hidden = generator_hidden

    @property
    def device(self):
        return self.mem_embedding.device

    def _set_adapter(self, adapter: str):
        """Switch active adapter and re-enable all LoRA params.

        PEFT's set_adapter() disables requires_grad on non-active adapters in
        some versions, which would silently drop them from the optimizer update.
        We always re-enable every lora_ param after switching.
        """
        self.model.set_adapter(adapter)
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)

    # ------------------------------------------------------------------
    # Compression:  sentence → 1-token embedding
    # ------------------------------------------------------------------

    def compress(self, sentence: str, with_grad: bool = False) -> torch.Tensor:
        """
        Compress a sentence into one embedding vector.

        Steps:
          1. Tokenize sentence
          2. Prepend learnable [MEM]: [MEM | s_1 ... s_n]
          3. Forward through LoRA-"compress" model
          4. Return last hidden state at [MEM] position → (hidden_size,)

        Args
        ----
        sentence   : raw sentence string
        with_grad  : if True, keeps gradients (for training)

        Returns
        -------
        compressed : (hidden_size,) float32 tensor
        """
        self._set_adapter("compress")

        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            add_special_tokens=True,
        ).to(self.device)

        # sentence token embeddings
        sent_embeds = self.model.get_input_embeddings()(inputs.input_ids)  # (1, L, H)

        # append [MEM] at the END so it attends to all sentence tokens (causal LM)
        mem = self.mem_embedding.unsqueeze(0).to(sent_embeds.dtype)        # (1, T, H)
        full_input = torch.cat([sent_embeds, mem], dim=1)                  # (1, L+T, H)

        backbone = self.model.base_model.model
        ctx = torch.no_grad() if not with_grad else torch.enable_grad()
        with ctx:
            out = backbone(
                inputs_embeds=full_input,
                output_hidden_states=True,
                use_cache=False,
            )

        # Hidden state at the LAST position = [MEM] = compressed representation
        compressed = out.hidden_states[-1][0, -self.num_latent_tokens :, :].float()   # (T, H)
        return compressed

    def compress_batch(
        self, sentences: list, with_grad: bool = False, adapter: str = "compress"
    ) -> torch.Tensor:
        """
        Compress a list of sentences.  Pads to same length for batch processing.

        Parameters
        ----------
        adapter : "compress" (sentences) or "query" (queries)

        Returns
        -------
        embeddings : (B, hidden_size) float32 tensor
        """
        self._set_adapter(adapter)

        enc = self.tokenizer(
            sentences,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
            add_special_tokens=True,
        ).to(self.device)

        sent_embeds = self.model.get_input_embeddings()(enc.input_ids)   # (B, L, H)
        B = sent_embeds.shape[0]
        mem = self.mem_embedding.unsqueeze(0).expand(B, -1, -1).to(sent_embeds.dtype)  # (B,T,H)

        # [MEM] at END so it attends to all sentence tokens via causal mask
        full_input = torch.cat([sent_embeds, mem], dim=1)               # (B, L+T, H)

        # Extend attention mask: sentence mask + 1 for [MEM]
        mem_mask = torch.ones(B, self.num_latent_tokens, dtype=enc.attention_mask.dtype, device=self.device)
        full_mask = torch.cat([enc.attention_mask, mem_mask], dim=1)    # (B, L+T)

        # Call the transformer backbone directly (skip lm_head) to avoid
        # allocating a huge (B*L, vocab_size) logits tensor we don't need.
        backbone = self.model.base_model.model
        ctx = torch.no_grad() if not with_grad else torch.enable_grad()
        with ctx:
            out = backbone(
                inputs_embeds=full_input,
                attention_mask=full_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        compressed = out.hidden_states[-1][:, -self.num_latent_tokens :, :].float()   # (B, T, H)
        return compressed

    # ------------------------------------------------------------------
    # Retrieval embedding
    # ------------------------------------------------------------------

    def get_retrieval_embedding(self, compressed: torch.Tensor) -> torch.Tensor:
        """
        Project compressed embedding to the retrieval space.

        Args
        ----
        compressed : (H,), (T, H), or (B, T, H)

        Returns
        -------
        embedding  : (retrieval_dim,) or (B, retrieval_dim), L2-normalized
        """
        if compressed.dim() == 3:
            compressed = compressed.mean(dim=1)
        emb = self.retrieval_proj(compressed)
        return F.normalize(emb, dim=-1)

    def project_for_generator(self, compressed: torch.Tensor) -> torch.Tensor:
        """
        Project 1B compressed embedding(s) into the 8B generator's space.

        Args
        ----
        compressed : (H_1B,), (T, H_1B), or (B, T, H_1B)

        Returns
        -------
        projected : same leading dimensions with final dim H_8B
        """
        return self.cross_proj(compressed)

    # ------------------------------------------------------------------
    # Reconstruction loss  (L_recon)
    # ------------------------------------------------------------------

    def reconstruction_loss_batch(
        self, sentences: list,
        adapter: str = "compress",
        decode_adapter: str = "decode",
    ) -> torch.Tensor:
        """
        Truly batched reconstruction loss — one forward pass per step.

        Steps
        -----
        1. Compress all B sentences in one forward pass  → (B, H)
        2. Tokenize all sentences with padding            → (B, L)
        3. Decoder forward: [h_MEM | sent_tokens[:-1]]   → (B, L, V)
        4. Cross-entropy on sent_tokens[1:], ignoring pad

        Parameters
        ----------
        adapter        : "compress" for sentences, "query" for queries
        decode_adapter : "decode" for sentences, "query_decode" for queries
                         (keeps sentence and query decoders separate)
        """
        # --- Step 1: compress (with grad) using the specified adapter ---
        compressed = self.compress_batch(sentences, with_grad=True, adapter=adapter)   # (B, T, H)

        # --- Step 2: tokenize target sentences with BOS ---
        self._set_adapter(decode_adapter)
        enc = self.tokenizer(
            sentences,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
            add_special_tokens=True,
        ).to(self.device)
        target_ids = enc.input_ids          # (B, L)
        B, L = target_ids.shape

        target_embeds = self.model.get_input_embeddings()(target_ids)  # (B, L, H)

        # --- Step 3: build decoder input [h_MEM | BOS | x_1 .. x_{n-1}] ---
        mem = compressed.to(target_embeds.dtype)                       # (B, T, H)
        input_embeds = torch.cat([mem, target_embeds[:, :-1, :]], dim=1)  # (B, T+L-1, H)

        mem_mask = torch.ones(B, self.num_latent_tokens, dtype=enc.attention_mask.dtype, device=self.device)
        full_mask = torch.cat([mem_mask, enc.attention_mask[:, :-1]], dim=1)  # (B, T+L-1)

        out = self.model(
            inputs_embeds=input_embeds,
            attention_mask=full_mask,
            use_cache=False,
        )
        logits = out.logits   # (B, L, V)
        V = logits.shape[-1]

        # --- Step 4: skip latent->BOS, supervise x_1 ... x_n ---
        sent_logits  = logits[:, self.num_latent_tokens :, :].reshape(-1, V)   # (B*(L-1), V)
        sent_targets = target_ids[:, 1:].reshape(-1)     # (B*(L-1),)

        loss = F.cross_entropy(
            sent_logits, sent_targets,
            ignore_index=self.tokenizer.pad_token_id,
        )
        return loss

    # ------------------------------------------------------------------
    # Query embedding  (for contrastive retrieval)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed_query(self, query: str) -> torch.Tensor:
        """Single query → (retrieval_dim,) L2-normalized embedding."""
        return self.embed_query_batch([query])[0]

    @torch.no_grad()
    def embed_query_batch(self, queries: list) -> torch.Tensor:
        """
        Batch query encoding using the same [MEM] compress mechanism as sentences.

        Both query and sentence go through the same compress LoRA + [MEM] path,
        so they live in the same embedding space from the start.

        Returns
        -------
        query_embs : (B, retrieval_dim) L2-normalized float32 tensor
        """
        compressed = self.compress_batch(queries, with_grad=False, adapter="query")   # (B, H)
        return self.get_retrieval_embedding(compressed)                                # (B, retrieval_dim)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str):
        sd = self.model.state_dict()

        def _f(name):
            tag = f".{name}."
            return {k: v for k, v in sd.items() if tag in k}

        torch.save(
            {
                "mem_embedding":  self.mem_embedding.data,
                "retrieval_proj": self.retrieval_proj.state_dict(),
                "cross_proj":     self.cross_proj.state_dict(),
                "lora_compress":     _f("compress"),
                "lora_decode":       _f("decode"),
                "lora_query":        _f("query"),
                "lora_query_decode": _f("query_decode"),
                "config": {
                    "model_name":       self.model_name,
                    "retrieval_dim":    self.retrieval_dim,
                    "hidden_size":      self.hidden_size,
                    "generator_hidden": self.generator_hidden,
                    "num_latent_tokens": self.num_latent_tokens,
                },
            },
            path,
        )
        logger.info(f"Saved to {path}")

    def _copy_mem_from_checkpoint(self, saved_mem: torch.Tensor):
        if saved_mem.shape == self.mem_embedding.data.shape:
            self.mem_embedding.data.copy_(saved_mem)
            return
        if saved_mem.dim() == 2 and saved_mem.shape[0] == 1 and saved_mem.shape[1] == self.hidden_size:
            self.mem_embedding.data.copy_(saved_mem.expand_as(self.mem_embedding.data))
            logger.warning(
                "Loaded single-token mem_embedding into %d latent slots by replication.",
                self.num_latent_tokens,
            )
            return
        n = min(saved_mem.shape[0], self.mem_embedding.data.shape[0])
        self.mem_embedding.data[:n].copy_(saved_mem[:n])
        if n < self.mem_embedding.data.shape[0]:
            self.mem_embedding.data[n:].copy_(
                saved_mem[:1].expand(self.mem_embedding.data.shape[0] - n, -1)
            )
        logger.warning(
            "Checkpoint mem_embedding shape %s differs from model shape %s; copied %d slots.",
            tuple(saved_mem.shape),
            tuple(self.mem_embedding.data.shape),
            n,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self._copy_mem_from_checkpoint(ckpt["mem_embedding"])
        self.retrieval_proj.load_state_dict(ckpt["retrieval_proj"])
        self.cross_proj.load_state_dict(ckpt["cross_proj"])
        sd = self.model.state_dict()
        sd.update(ckpt["lora_compress"])
        sd.update(ckpt["lora_decode"])
        if "lora_query" in ckpt:
            sd.update(ckpt["lora_query"])
        if "lora_query_decode" in ckpt:
            sd.update(ckpt["lora_query_decode"])
        self.model.load_state_dict(sd)
        logger.info(f"Loaded from {path}")

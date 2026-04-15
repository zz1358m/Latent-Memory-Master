"""
LCC Compiler
============
Implements the Latent Context Compilation (LCC) idea:
  - A disposable LoRA module acts as a "compiler"
  - It distills a long document into N compact buffer tokens (latent memory)
  - At inference, the LoRA is discarded; only the buffer token representations
    are injected into the frozen base LLM

Extension for retrieval:
  - A retrieval_head projects pooled buffer tokens → a normalized vector
    used to index and query a FAISS memory bank

Reference: "Latent Context Compilation: Distilling Long Context into
Compact Portable Memory" (LCC paper)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class LCCCompiler(nn.Module):
    """
    LCC Compiler: compiles a document into latent buffer tokens.

    Architecture
    ------------
    buffer_embeddings : nn.Parameter  shape (N, H)
        Shared learnable "memory slots". The LoRA contextualizes them
        per-document during compilation.

    model : PeftModel (LoRA on top of frozen LLM)
        The compiler. Only LoRA weights + buffer_embeddings are trained.
        Discarded after compilation at deployment time.

    retrieval_head : nn.Sequential
        Maps pooled buffer tokens (H,) → normalized retrieval vector (D,).
        Trained jointly with the compiler.

    Training objective (self-aligned, no synthetic QA needed)
    ---------------------------------------------------------
    1. Reconstruction loss:
       [compiled_memory | doc[:-1]] → frozen LLM → predict doc[1:]
    2. (Optional) QA loss:
       [compiled_memory | query[:-1]] → frozen LLM → predict answer

    Compilation (inference)
    -----------------------
    [buffer_tokens | document] → LoRA-LLM → last hidden states of buffer positions
    → store as latent memory M_i ∈ R^{N x H}
    """

    def __init__(
        self,
        model_name: str,
        num_buffer_tokens: int = 64,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: list = None,
        retrieval_dim: int = 256,
        max_doc_length: int = 2048,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_buffer_tokens = num_buffer_tokens
        self.retrieval_dim = retrieval_dim
        self.max_doc_length = max_doc_length

        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]

        # --- Tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- Base LLM (frozen) ---
        logger.info(f"Loading base model: {model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        self.hidden_size = base_model.config.hidden_size
        self.vocab_size = base_model.config.vocab_size

        # --- LoRA compiler (trainable) ---
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.model = get_peft_model(base_model, lora_cfg)
        self.model.print_trainable_parameters()

        # --- Buffer token embeddings (trainable) ---
        # Initialized small so they start near the model's embedding distribution
        self.buffer_embeddings = nn.Parameter(
            torch.randn(num_buffer_tokens, self.hidden_size) * 0.02
        )

        # --- Retrieval head (trainable) ---
        self.retrieval_head = nn.Sequential(
            nn.Linear(self.hidden_size, retrieval_dim, dtype=torch.float32),
            nn.LayerNorm(retrieval_dim),
        )

    @property
    def device(self):
        return self.buffer_embeddings.device

    # ------------------------------------------------------------------
    # Core compilation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compile(self, document: str) -> torch.Tensor:
        """
        Compile a document into latent memory (inference-time, no grad).

        Steps:
          1. Tokenize document
          2. Prepend learnable buffer tokens: [B | doc]
          3. Forward through LoRA-LLM
          4. Extract hidden states at buffer positions → compiled memory

        Returns
        -------
        compiled_memory : (num_buffer_tokens, hidden_size) float32 tensor
        """
        inputs = self.tokenizer(
            document,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_doc_length,
        ).to(self.device)

        doc_embeds = self.model.get_input_embeddings()(inputs.input_ids)
        buf = self.buffer_embeddings.unsqueeze(0).to(doc_embeds.dtype)
        full_input = torch.cat([buf, doc_embeds], dim=1)

        outputs = self.model(
            inputs_embeds=full_input,
            output_hidden_states=True,
            use_cache=False,
        )
        # (1, N+L, H) → take buffer positions → (N, H)
        compiled = outputs.hidden_states[-1][0, : self.num_buffer_tokens, :].float()
        return compiled.detach()

    def compile_with_grad(self, document: str) -> torch.Tensor:
        """Same as compile() but keeps gradients (for training)."""
        inputs = self.tokenizer(
            document,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_doc_length,
        ).to(self.device)

        doc_embeds = self.model.get_input_embeddings()(inputs.input_ids)
        buf = self.buffer_embeddings.unsqueeze(0).to(doc_embeds.dtype)
        full_input = torch.cat([buf, doc_embeds], dim=1)

        outputs = self.model(
            inputs_embeds=full_input,
            output_hidden_states=True,
            use_cache=False,
        )
        compiled = outputs.hidden_states[-1][0, : self.num_buffer_tokens, :]
        return compiled

    # ------------------------------------------------------------------
    # Retrieval embedding
    # ------------------------------------------------------------------

    def get_retrieval_embedding(self, compiled_memory: torch.Tensor) -> torch.Tensor:
        """
        Produce a normalized retrieval vector from compiled memory.

        Args
        ----
        compiled_memory : (N, H) or (B, N, H)

        Returns
        -------
        embedding : (retrieval_dim,) or (B, retrieval_dim), L2-normalized
        """
        if compiled_memory.dim() == 2:
            pooled = compiled_memory.float().mean(dim=0)        # (H,)
        else:
            pooled = compiled_memory.float().mean(dim=1)        # (B, H)
        emb = self.retrieval_head(pooled)
        return F.normalize(emb, dim=-1)

    # ------------------------------------------------------------------
    # Training losses
    # ------------------------------------------------------------------

    def reconstruction_loss(self, document: str, alpha: float = 1.0) -> torch.Tensor:
        """
        Self-aligned reconstruction loss (LCC's core training signal).

        Objective: given ONLY the compiled memory (no raw document at
        "inference"), predict the document tokens autoregressively.

        This forces the buffer tokens to encode all document content,
        while the context-agnostic frozen LLM acts as a regularizer.
        """
        compiled = self.compile_with_grad(document)

        target_inputs = self.tokenizer(
            document,
            return_tensors="pt",
            truncation=True,
            max_length=min(512, self.max_doc_length),
        ).to(self.device)
        target_ids = target_inputs.input_ids                      # (1, L)
        target_embeds = self.model.get_input_embeddings()(target_ids)

        # Simulate inference: frozen LLM receives [compiled_memory | doc[:-1]]
        mem = compiled.unsqueeze(0).to(target_embeds.dtype)       # (1, N, H)
        input_embeds = torch.cat([mem, target_embeds[:, :-1, :]], dim=1)

        with self.model.disable_adapter():
            outputs = self.model(inputs_embeds=input_embeds, use_cache=False)

        logits = outputs.logits[0]                                # (N+L-1, V)
        doc_logits = logits[self.num_buffer_tokens :]             # (L-1, V)
        doc_targets = target_ids[0, 1:]                           # (L-1,)

        return F.cross_entropy(doc_logits, doc_targets) * alpha

    def qa_loss(
        self,
        document: str,
        query: str,
        answer: str,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """
        Optional supervised QA loss for fine-tuning with labeled data.

        Objective: [compiled_memory | query] → predict answer tokens.
        """
        compiled = self.compile_with_grad(document)

        qa_text = f"{query} {answer}"
        qa_ids = self.tokenizer(
            qa_text, return_tensors="pt", truncation=True, max_length=256
        ).input_ids.to(self.device)
        q_len = self.tokenizer(
            query, return_tensors="pt", truncation=True, max_length=128
        ).input_ids.shape[1]

        qa_embeds = self.model.get_input_embeddings()(qa_ids)
        mem = compiled.unsqueeze(0).to(qa_embeds.dtype)
        input_embeds = torch.cat([mem, qa_embeds[:, :-1, :]], dim=1)

        with self.model.disable_adapter():
            outputs = self.model(inputs_embeds=input_embeds, use_cache=False)

        logits = outputs.logits[0]                               # (N+Lqa-1, V)
        ans_start = self.num_buffer_tokens + q_len
        if ans_start >= logits.shape[0]:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        ans_logits = logits[ans_start:]
        ans_targets = qa_ids[0, q_len + 1 :]
        min_len = min(ans_logits.shape[0], ans_targets.shape[0])
        if min_len == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return F.cross_entropy(ans_logits[:min_len], ans_targets[:min_len]) * alpha

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(
            {
                "buffer_embeddings": self.buffer_embeddings.data,
                "retrieval_head": self.retrieval_head.state_dict(),
                "lora": self.model.state_dict(),
                "config": {
                    "model_name": self.model_name,
                    "num_buffer_tokens": self.num_buffer_tokens,
                    "retrieval_dim": self.retrieval_dim,
                    "max_doc_length": self.max_doc_length,
                },
            },
            path,
        )
        logger.info(f"Compiler saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.buffer_embeddings.data.copy_(ckpt["buffer_embeddings"])
        self.retrieval_head.load_state_dict(ckpt["retrieval_head"])
        self.model.load_state_dict(ckpt["lora"])
        logger.info(f"Compiler loaded from {path}")

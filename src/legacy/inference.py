"""
Inference with Latent Memory Injection (1B → 8B pipeline)
===========================================================
At query time:
  1. Retrieve top-k compressed sentence embeddings (from 1B compressor, dim=2048)
  2. Project each through CrossModelProjection → 8B embedding space (dim=4096)
  3. Prepend projected latent tokens to query → frozen 8B LLM → answer

The 8B LLM receives:
  [projected_sent_1 | projected_sent_2 | ... | projected_sent_k | query_tokens]

Each projected_sent is a SINGLE token (dim=4096), so k retrieved sentences
cost only k tokens total — far fewer than pasting raw text.
"""

import logging
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class LatentInference:
    """
    Frozen LLM that accepts compiled latent memories as soft-prompt prefix.

    The LoRA compiler is NOT used here — it was discarded after compilation.
    We only use the frozen base LLM.
    """

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        logger.info(f"Loading frozen LLM for inference: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model.eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _count_memory_tokens(self, memories: List[torch.Tensor]) -> int:
        """Count total tokens occupied by injected memories."""
        return sum(m.shape[0] for m in memories)

    @torch.no_grad()
    def generate(
        self,
        query: str,
        memories: List[torch.Tensor],  # list of (N, H) tensors
        prompt_template: Optional[str] = None,
    ) -> Tuple[str, int]:
        """
        Generate an answer given a query and retrieved latent memories.

        Args
        ----
        query    : question string
        memories : list of compiled memory tensors, each (N, H)
                   retrieved from the MemoryBank
        prompt_template : optional format string with {query} placeholder

        Returns
        -------
        answer   : generated text
        n_tokens : total input tokens (memory + query), for efficiency tracking
        """
        if prompt_template is not None:
            query_text = prompt_template.format(query=query)
        else:
            query_text = f"Question: {query}\nAnswer:"

        # Tokenize query
        q_inputs = self.tokenizer(query_text, return_tensors="pt").to(self.device)
        q_embeds = self.model.get_input_embeddings()(q_inputs.input_ids)   # (1, Lq, H)

        # Build input: [memory_1 | memory_2 | ... | query]
        parts = []
        for mem in memories:
            # Move memory to correct device and dtype
            mem = mem.to(self.device).to(q_embeds.dtype)
            parts.append(mem.unsqueeze(0))   # (1, N, H)
        parts.append(q_embeds)

        input_embeds = torch.cat(parts, dim=1)   # (1, total_len, H)
        n_tokens = input_embeds.shape[1]

        # Build attention mask (all ones — no padding)
        attention_mask = torch.ones(
            1, n_tokens, dtype=torch.long, device=self.device
        )

        # Generate
        do_sample = self.temperature > 0
        out = self.model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decode only newly generated tokens
        generated_ids = out[0, n_tokens:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        logger.debug(
            f"Generated {len(generated_ids)} tokens | "
            f"Input: {n_tokens} tokens "
            f"({self._count_memory_tokens(memories)} from memories, "
            f"{q_inputs.input_ids.shape[1]} from query)"
        )
        return answer, n_tokens

    @torch.no_grad()
    def generate_batch(
        self,
        queries: List[str],
        batch_memories: List[List[torch.Tensor]],
    ) -> List[Tuple[str, int]]:
        """Generate answers for a batch (sequentially, since memory lengths vary)."""
        results = []
        for query, memories in zip(queries, batch_memories):
            results.append(self.generate(query, memories))
        return results

"""
LLaVA Compressor
================
Replaces SentenceAutoencoder with a LLaVA-v1.5-7B based compressor that
handles BOTH text-only and image+text inputs in a unified sequence.

Architecture
------------
  Base model : liuhaotian/llava-v1.5-7b  (via HuggingFace llava-hf format)
               → LlavaForConditionalGeneration

  Vision tower       : frozen CLIP ViT-L/14@336px  (inside LLaVA)
  MM projector       : frozen 2-layer MLP           (inside LLaVA)
  Language model     : Vicuna-7B  ← five LoRA adapters applied here

  LoRA adapters on Vicuna-7B language_model (compression only):
    "compress"     : (title + optional image) → [MEM] latent
    "query"        : question text → retrieval embedding

  Separate decoder LM (LLaMA-3.2-1B) with LoRA adapters (reconstruction only):
    "decode"       : [MEM] latent → reconstruct sentence/passage text  (CE loss)
    "query_decode" : [MEM] query latent → reconstruct question text  (CE loss)
    "image_decode" : [MEM] image latent → reconstruct image caption/title  (CE loss)

  Additional trainable parameters:
    mem_embedding        : (1, 4096)  learnable [MEM] token
    retrieval_proj       : Linear(4096, retrieval_dim) → LayerNorm → L2-norm
    cross_proj           : Linear(4096, mid) → GELU → Linear(mid, 5120) → LayerNorm
                           bridges 7B → 13B hidden spaces
    decode_proj          : Linear(4096, 2048) → LayerNorm
                           bridges Vicuna-7B latent space → LLaMA-1B decoder input space
    img_embed_decode_proj: Linear(4096→2560) → GELU → LayerNorm → Linear(2560→1024)
                           maps [MEM] latent to CLIP CLS space for visual alignment

Input format
------------
  Image+text sample:
    "USER: Title: {title}\\nPic: <image>\\nASSISTANT:"  + [MEM]
    (LLaVA processor replaces <image> with CLIP visual token sequence)

  Text-only sample:
    "USER: {sentence}\\nASSISTANT:"  + [MEM]
    (no image; same code path, images=None)

  Query:
    "USER: {question}\\nASSISTANT:"  + [MEM]

Decoder target (L_recon):
  Autoregressive CE on title text tokens only (not image tokens, which cannot
  be reconstructed as raw pixels from a single latent token).

Model IDs
---------
  Compressor : llava-hf/llava-1.5-7b-hf   (transformers-native format of
               liuhaotian/llava-v1.5-7b, identical weights)
  Generator  : llava-hf/llava-1.5-13b-hf  (liuhaotian/llava-v1.5-13b)

Checkpoint format
-----------------
  {
    "mem_embedding"        : tensor,
    "retrieval_proj"       : state_dict,
    "cross_proj"           : state_dict,
    "decode_proj"          : state_dict,
    "img_embed_decode_proj": state_dict,
    "lora_compress"        : state_dict,   # from Vicuna-7B
    "lora_query"           : state_dict,   # from Vicuna-7B
    "lora_decode"          : state_dict,   # from LLaMA-1B decoder
    "lora_query_decode"    : state_dict,   # from LLaMA-1B decoder
    "lora_image_decode"    : state_dict,   # from LLaMA-1B decoder
    "config"               : dict,
  }
"""

import logging
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor, AutoTokenizer, AutoModelForCausalLM,
    LlavaForConditionalGeneration,
)

logger = logging.getLogger(__name__)

_IMAGE_TOKEN = "<image>"
_MEM_TOKEN   = "[MEM]"


# ---------------------------------------------------------------------------
# Cross-model projection: 7B (4096) → 13B (5120)
# ---------------------------------------------------------------------------

class CrossModelProjection(nn.Module):
    def __init__(self, in_dim: int = 4096, out_dim: int = 5120):
        super().__init__()
        mid_dim = (in_dim + out_dim) // 2
        self.net = nn.Sequential(
            nn.Linear(in_dim,  mid_dim, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(mid_dim, out_dim, dtype=torch.float32),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


# ---------------------------------------------------------------------------
# Main compressor
# ---------------------------------------------------------------------------

class LLaVACompressor(nn.Module):
    """
    LLaVA-v1.5-7B based compressor with three LoRA adapters.

    Parameters
    ----------
    model_name      : HuggingFace model ID (llava-hf/llava-1.5-7b-hf or local path)
    lora_r          : LoRA rank (shared across all three adapters)
    lora_alpha      : LoRA alpha
    lora_dropout    : LoRA dropout
    target_modules  : LM layer names to apply LoRA (default: attention projections)
    retrieval_dim   : dimension of the L2-normalised retrieval embedding
    generator_hidden: hidden size of the 13B generator (5120 for llava-1.5-13b-hf)
    """

    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        decoder_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        decode_lora_r: Optional[int] = None,
        decode_lora_alpha: Optional[int] = None,
        decode_lora_dropout: Optional[float] = None,
        decode_target_modules: Optional[List[str]] = None,
        retrieval_dim: int = 512,
        generator_hidden: int = 5120,
        num_latent_tokens: int = 1,
    ):
        super().__init__()
        self.model_name    = model_name
        self.decoder_name  = decoder_name
        self.retrieval_dim = retrieval_dim
        self.num_latent_tokens = int(num_latent_tokens)
        if self.num_latent_tokens < 1:
            raise ValueError("num_latent_tokens must be >= 1")

        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if decode_lora_r is None:
            decode_lora_r = lora_r
        if decode_lora_alpha is None:
            decode_lora_alpha = lora_alpha
        if decode_lora_dropout is None:
            decode_lora_dropout = lora_dropout
        if decode_target_modules is None:
            decode_target_modules = list(target_modules)

        # ------------------------------------------------------------------ processor
        logger.info(f"Loading LLaVA processor: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer  = self.processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add [MEM] as a new special token
        self.tokenizer.add_special_tokens({"additional_special_tokens": [_MEM_TOKEN]})
        self.mem_token_id   = self.tokenizer.convert_tokens_to_ids(_MEM_TOKEN)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(_IMAGE_TOKEN)

        # ------------------------------------------------------------------ base model
        logger.info(f"Loading LLaVA base model: {model_name}")
        base = LlavaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        # Resize embeddings to include [MEM] token
        base.resize_token_embeddings(len(self.tokenizer))

        # Resolve language_model sub-module — handle both transformers layouts:
        #   Old: LlavaForConditionalGeneration.language_model
        #   New: LlavaForConditionalGeneration.model.language_model
        if hasattr(base, "language_model"):
            self._lm_parent = base
            self._lm_attr   = "language_model"
        elif hasattr(base, "model") and hasattr(base.model, "language_model"):
            self._lm_parent = base.model
            self._lm_attr   = "language_model"
        else:
            raise AttributeError(
                "Cannot locate language_model in LlavaForConditionalGeneration. "
                f"Top-level attrs: {[n for n, _ in base.named_children()]}"
            )

        lm = getattr(self._lm_parent, self._lm_attr)

        # Freeze everything except the language model
        for name, p in base.named_parameters():
            if "language_model" not in name:
                p.requires_grad_(False)

        self.hidden_size = lm.config.hidden_size   # 4096 for 7B

        # ------------------------------------------------------------------ LoRA adapters on Vicuna-7B (compress + query only)
        # task_type=None: language_model may be bare LlamaModel (no LM head)
        # in newer transformers; we only need hidden states, not generation wrappers.
        def _lora_cfg(task_type=None):
            return LoraConfig(
                task_type=task_type,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
            )

        def _decode_lora_cfg(task_type=None):
            return LoraConfig(
                task_type=task_type,
                r=decode_lora_r,
                lora_alpha=decode_lora_alpha,
                lora_dropout=decode_lora_dropout,
                target_modules=decode_target_modules,
                bias="none",
            )

        lm_peft = get_peft_model(lm, _lora_cfg(), adapter_name="compress")
        lm_peft.add_adapter("query", _lora_cfg())   # query compressor

        # Cast LoRA params to bfloat16 to match base model dtype
        for param in lm_peft.parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)

        setattr(self._lm_parent, self._lm_attr, lm_peft)

        self.model = base

        # ------------------------------------------------------------------ Separate decoder LM (LLaMA-3.2-1B)
        # Decode adapters live here, completely decoupled from the Vicuna-7B backbone.
        logger.info(f"Loading decoder LM: {decoder_name}")
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_name)
        if self.decoder_tokenizer.pad_token is None:
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token

        decoder_base = AutoModelForCausalLM.from_pretrained(
            decoder_name, torch_dtype=torch.bfloat16
        )
        self.decoder_hidden_size = decoder_base.config.hidden_size   # 2048 for LLaMA-1B

        # Freeze the decoder base; only LoRA adapters will be trained
        for p in decoder_base.parameters():
            p.requires_grad_(False)

        from peft import TaskType
        decoder_peft = get_peft_model(
            decoder_base, _decode_lora_cfg(task_type=TaskType.CAUSAL_LM), adapter_name="decode"
        )
        decoder_peft.add_adapter("query_decode", _decode_lora_cfg(task_type=TaskType.CAUSAL_LM))
        decoder_peft.add_adapter("image_decode", _decode_lora_cfg(task_type=TaskType.CAUSAL_LM))

        for param in decoder_peft.parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)

        self.decoder_lm = decoder_peft
        self._set_decode_adapter("decode")

        # Resolve vision_tower — handle both transformers layouts:
        #   Old: LlavaForConditionalGeneration.vision_tower
        #   New: LlavaForConditionalGeneration.model.vision_tower
        if hasattr(base, "vision_tower"):
            self._vision_tower = base.vision_tower
        elif hasattr(base, "model") and hasattr(base.model, "vision_tower"):
            self._vision_tower = base.model.vision_tower
        else:
            raise AttributeError(
                "Cannot locate vision_tower in LlavaForConditionalGeneration. "
                f"Top-level attrs: {[n for n, _ in base.named_children()]}"
            )

        self._set_adapter("compress")

        # ------------------------------------------------------------------ trainable heads
        self.mem_embedding = nn.Parameter(
            torch.randn(self.num_latent_tokens, self.hidden_size, dtype=torch.float32) * 0.02
        )
        self.retrieval_proj = nn.Sequential(
            nn.Linear(self.hidden_size, retrieval_dim, dtype=torch.float32),
            nn.LayerNorm(retrieval_dim),
        )
        # decode_proj: bridges Vicuna-7B latent space (4096) → LLaMA-1B decoder input (2048)
        self.decode_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.decoder_hidden_size, dtype=torch.float32),
            nn.LayerNorm(self.decoder_hidden_size),
        )
        self.cross_proj = CrossModelProjection(
            in_dim=self.hidden_size, out_dim=generator_hidden
        )
        # Projects MEM embedding → CLIP CLS hidden space (1024-dim).
        # 2-layer MLP: hidden_size → mid → 1024 (CLIP CLS hidden dim).
        # Target at training time: raw CLS hidden state from the frozen CLIP vision tower.
        _mid = (self.hidden_size + 1024) // 2   # 2560 for 7B (4096+1024)//2
        self.img_embed_decode_proj = nn.Sequential(
            nn.Linear(self.hidden_size, _mid, dtype=torch.float32),
            nn.GELU(),
            nn.LayerNorm(_mid, dtype=torch.float32),
            nn.Linear(_mid, 1024, dtype=torch.float32),
        )

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"LLaVACompressor trainable params: {n_train / 1e6:.1f}M")

    # ------------------------------------------------------------------ adapter switching

    def _lm(self):
        return getattr(self._lm_parent, self._lm_attr)

    def _set_adapter(self, name: str):
        lm = self._lm()
        lm.set_adapter(name)
        for n, p in lm.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)

    def _set_decode_adapter(self, name: str):
        self.decoder_lm.set_adapter(name)
        for n, p in self.decoder_lm.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)

    # ------------------------------------------------------------------ preprocessing

    def _build_inputs(
        self,
        texts: List[str],
        images: Optional[List[Optional[Image.Image]]] = None,
    ) -> dict:
        """
        Build processor inputs for a batch.
        texts  : list of prompt strings (already formatted with USER:/ASSISTANT:)
        images : parallel list of PIL Images or None for text-only samples.
        """
        if images is None:
            images = [None] * len(texts)

        # Separate image vs text-only samples
        has_image = [img is not None for img in images]

        # Process samples with images
        img_texts   = [t for t, h in zip(texts, has_image) if h]
        img_images  = [img for img, h in zip(images, has_image) if h]

        text_texts  = [t for t, h in zip(texts, has_image) if not h]

        device = self.mem_embedding.device
        dtype  = torch.bfloat16

        # We build a combined input_embeds + attention_mask batch manually
        # by processing each sample individually and padding to the same length.
        all_input_ids    = []
        all_pixel_values = []
        all_has_image    = has_image

        # Tokenize + get pixel values for image samples
        for txt, img in zip(img_texts, img_images):
            enc = self.processor(text=txt, images=img, return_tensors="pt")
            all_input_ids.append(enc["input_ids"].squeeze(0))
            all_pixel_values.append(enc["pixel_values"].squeeze(0))

        # Tokenize text-only samples
        for txt in text_texts:
            enc = self.tokenizer(txt, return_tensors="pt")
            all_input_ids.append(enc["input_ids"].squeeze(0))
            all_pixel_values.append(None)

        # Reorder to original batch order
        img_counter  = 0
        txt_counter  = len(img_texts)
        ordered_ids  = []
        ordered_pix  = []
        for h in has_image:
            if h:
                ordered_ids.append(all_input_ids[img_counter])
                ordered_pix.append(all_pixel_values[img_counter])
                img_counter += 1
            else:
                ordered_ids.append(all_input_ids[txt_counter])
                ordered_pix.append(None)
                txt_counter += 1

        return ordered_ids, ordered_pix, has_image

    # ------------------------------------------------------------------ compress

    def compress_batch(
        self,
        texts: List[str],
        images: Optional[List[Optional[Image.Image]]] = None,
        with_grad: bool = True,
        adapter: str = "compress",
    ) -> torch.Tensor:
        """
        Compress a batch of (text, optional image) pairs into latent tokens.

        For each sample:
          input_ids → LLaVA forward (with image tokens injected) + [MEM] appended
          The hidden state at the [MEM] position is the compressed embedding.

        Returns
        -------
        latents : (B, hidden_size) float32
        """
        self._set_adapter(adapter)
        device = self.mem_embedding.device

        input_ids_list, pixel_values_list, has_image = self._build_inputs(texts, images)
        B = len(input_ids_list)

        latents = []

        ctx = torch.enable_grad if with_grad else torch.no_grad
        with ctx():
            for i in range(B):
                ids = input_ids_list[i].to(device)                    # (L,)
                pv  = pixel_values_list[i]

                # Append [MEM] token ids
                mem_id = torch.full(
                    (self.num_latent_tokens,),
                    self.mem_token_id,
                    device=device,
                    dtype=ids.dtype,
                )
                ids_with_mem = torch.cat([ids, mem_id]).unsqueeze(0)   # (1, L+1)

                # Build attention mask
                attn_mask = torch.ones_like(ids_with_mem)

                if pv is not None:
                    pv = pv.unsqueeze(0).to(device, dtype=torch.bfloat16)
                    out = self.model(
                        input_ids=ids_with_mem,
                        pixel_values=pv,
                        attention_mask=attn_mask,
                        output_hidden_states=True,
                    )
                else:
                    out = self._lm()(
                        input_ids=ids_with_mem,
                        attention_mask=attn_mask,
                        output_hidden_states=True,
                    )

                # Hidden states at final positions = [MEM] latent tokens
                h_mem = out.hidden_states[-1][0, -self.num_latent_tokens :, :].float()    # (T, hidden_size)

                # Add learnable [MEM] residual
                h_mem = h_mem + self.mem_embedding.float()
                latents.append(h_mem)

        return torch.stack(latents, dim=0)   # (B, T, hidden_size)

    # ------------------------------------------------------------------ reconstruction loss

    def reconstruction_loss_batch(
        self,
        texts: List[str],
        images: Optional[List[Optional[Image.Image]]] = None,
        adapter: str = "compress",
        decode_adapter: str = "decode",
        target_texts: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        L_recon: compress→latent→decode autoregressive CE loss on text tokens.

        Mirrors SentenceAutoencoder.reconstruction_loss_batch:
          1. Compress all inputs (text + optional images) → latents z  [adapter]
          2. Switch to decode_adapter
          3. For each sample: build [z | text_tokens[:-1]] as inputs_embeds,
             forward through the full model, CE loss on text_tokens[1:]

        Parameters
        ----------
        adapter        : "compress" for sentences/images, "query" for queries
        decode_adapter : "decode" for sentences, "query_decode" for queries,
                         "image_decode" for image captions
                         (keeps sentence, query, and image decoders separate)
        target_texts   : optional separate CE targets (e.g. raw titles when
                         compress inputs are full prompts with <image> tokens)
        """
        if not texts:
            return torch.tensor(0.0, device=self.mem_embedding.device)

        device = self.mem_embedding.device
        decode_targets = target_texts if target_texts is not None else texts

        # Step 1: compress → latents via Vicuna-7B (with gradient)
        latents = self.compress_batch(texts, images=images, with_grad=True, adapter=adapter)
        # (B, 4096)

        # Step 2: switch to the appropriate decode adapter on the LLaMA-1B decoder
        self._set_decode_adapter(decode_adapter)
        embed_fn = self.decoder_lm.get_input_embeddings()

        total_loss = torch.tensor(0.0, device=device)
        count = 0

        for i, text in enumerate(decode_targets):
            # Project Vicuna-7B latent (4096) → LLaMA-1B input space (decoder_hidden_size)
            z_proj = self.decode_proj(latents[i].float())          # (T, decoder_hidden_size)
            z = z_proj.unsqueeze(0)                                # (1, T, decoder_hidden_size)

            # Tokenise target text with the decoder's own tokenizer
            enc = self.decoder_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                add_special_tokens=True,
            ).to(device)
            target_ids = enc.input_ids
            L = target_ids.shape[1]

            text_embeds = embed_fn(target_ids)          # (1, L, decoder_hidden_size)

            # Decoder input: [z_proj_1 ... z_proj_T | text_embeds[:-1]]
            input_embeds = torch.cat(
                [z.to(text_embeds.dtype), text_embeds[:, :-1, :]], dim=1
            )   # (1, T+L-1, decoder_hidden_size)
            attn_mask = torch.ones(1, self.num_latent_tokens + L - 1, device=device, dtype=torch.long)

            # Forward through LLaMA-1B decoder with the active decode adapter
            out    = self.decoder_lm(inputs_embeds=input_embeds, attention_mask=attn_mask)
            logits = out.logits     # (1, L, V)

            V = logits.shape[-1]
            shift_logits  = logits[0, self.num_latent_tokens :, :].reshape(-1, V)
            shift_targets = target_ids[0, 1:].reshape(-1)

            loss = F.cross_entropy(
                shift_logits.float(), shift_targets,
                ignore_index=self.decoder_tokenizer.pad_token_id,
            )
            total_loss = total_loss + loss
            count += 1

        return total_loss / max(count, 1)

    # ------------------------------------------------------------------ embedding-space reconstruction loss

    @torch.no_grad()
    def get_clip_cls_batch(
        self,
        pil_images: List[Image.Image],
    ) -> List[Optional[torch.Tensor]]:
        """
        Extract CLIP ViT CLS token for each PIL image using the frozen vision tower.
        Returns a list of (1024,) float32 CPU tensors (or None if processing fails).
        Used to batch-extract CLIP CLS tokens (e.g. for offline corpus compilation).
        """
        device = self.mem_embedding.device
        results = []
        for pil in pil_images:
            try:
                _, pv_list, _ = self._build_inputs(["dummy"], [pil])
                pv = pv_list[0]
                if pv is None:
                    results.append(None)
                    continue
                pv_t = pv.unsqueeze(0).to(device, dtype=torch.bfloat16)
                clip_out = self._vision_tower(pv_t)
                if hasattr(clip_out, "last_hidden_state"):
                    clip_feats = clip_out.last_hidden_state
                else:
                    clip_feats = clip_out[0] if isinstance(clip_out, (tuple, list)) else clip_out
                results.append(clip_feats[0, 0, :].float().cpu())
            except Exception:
                results.append(None)
        return results

    def embed_reconstruction_loss_batch(
        self,
        texts: List[str],
        images: Optional[List[Optional[Image.Image]]] = None,
        adapter: str = "compress",
    ) -> torch.Tensor:
        """
        Embedding-space reconstruction loss — image samples only.

        MEM → img_embed_decode_proj → predicted CLIP CLS hidden (1024-dim).
        MSE vs. frozen CLIP CLS from LLaVA's vision tower.
        Text samples (pv is None) are skipped.
        All image pixel values are batched into a single vision-tower forward pass.
        """
        if not texts:
            return torch.tensor(0.0, device=self.mem_embedding.device)

        device = self.mem_embedding.device

        # Get MEM latents (with gradient)
        latents = self.compress_batch(texts, images=images, with_grad=True, adapter=adapter)
        _, pixel_values_list, _ = self._build_inputs(texts, images)

        # Collect image indices and stack pixel values for a single batched forward
        img_indices = [i for i, pv in enumerate(pixel_values_list) if pv is not None]
        if not img_indices:
            return torch.tensor(0.0, device=device)

        pv_batch = torch.stack(
            [pixel_values_list[i].to(device, dtype=torch.bfloat16) for i in img_indices]
        )  # (N_img, C, H, W)

        with torch.no_grad():
            clip_out = self._vision_tower(pv_batch)
            if hasattr(clip_out, "last_hidden_state"):
                clip_feats = clip_out.last_hidden_state
            else:
                clip_feats = clip_out[0] if isinstance(clip_out, (tuple, list)) else clip_out
            clip_cls_batch = clip_feats[:, 0, :].float()  # (N_img, 1024)

        total_loss = torch.tensor(0.0, device=device)
        for j, i in enumerate(img_indices):
            pred_cls = self.img_embed_decode_proj(latents[i].float().mean(dim=0))  # (1024,)
            total_loss = total_loss + F.mse_loss(pred_cls, clip_cls_batch[j])

        return total_loss / len(img_indices)

    # ------------------------------------------------------------------ retrieval

    def get_retrieval_embedding(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Project latents to the L2-normalised retrieval space.

        Parameters
        ----------
        latents : (B, T, hidden_size) float32

        Returns
        -------
        embeddings : (B, retrieval_dim) float32, L2-normalised
        """
        if latents.dim() == 3:
            latents = latents.mean(dim=1)
        proj = self.retrieval_proj(latents.float())    # (B, retrieval_dim) or (retrieval_dim,)
        return F.normalize(proj, dim=-1)

    def project_for_generator(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Project latents into the 13B generator's embedding space via cross_proj.

        Parameters
        ----------
        latents : (B, hidden_size) float32

        Returns
        -------
        projected : (B, generator_hidden) float32
        """
        return self.cross_proj(latents.float())

    # ------------------------------------------------------------------ query embedding

    def embed_query_batch(
        self, questions: List[str], with_grad: bool = True
    ) -> torch.Tensor:
        """
        Embed questions using the query adapter.

        Returns
        -------
        latents : (B, hidden_size) float32
        """
        prompts = [f"USER: {q}\nASSISTANT:" for q in questions]
        return self.compress_batch(prompts, images=None, with_grad=with_grad, adapter="query")

    # ------------------------------------------------------------------ checkpoint

    def save(self, path: str):
        def _filter(sd, adapter_name):
            # Use dot-bounded match so "decode" doesn't capture "query_decode"/"image_decode"
            tag = f".{adapter_name}."
            return {k: v for k, v in sd.items() if tag in k}

        compressor_sd = self._lm().state_dict()
        decoder_sd    = self.decoder_lm.state_dict()
        torch.save({
            "mem_embedding"        : self.mem_embedding.data,
            "retrieval_proj"       : self.retrieval_proj.state_dict(),
            "cross_proj"           : self.cross_proj.state_dict(),
            "decode_proj"          : self.decode_proj.state_dict(),
            "img_embed_decode_proj": self.img_embed_decode_proj.state_dict(),
            # Compression LoRAs — from Vicuna-7B
            "lora_compress"        : _filter(compressor_sd, "compress"),
            "lora_query"           : _filter(compressor_sd, "query"),
            # Decode LoRAs — from LLaMA-1B decoder
            "lora_decode"          : _filter(decoder_sd, "decode"),
            "lora_query_decode"    : _filter(decoder_sd, "query_decode"),
            "lora_image_decode"    : _filter(decoder_sd, "image_decode"),
            "config": {
                "model_name"      : self.model_name,
                "decoder_name"    : self.decoder_name,
                "retrieval_dim"   : self.retrieval_dim,
                "hidden_size"     : self.hidden_size,
                "decoder_hidden_size": self.decoder_hidden_size,
                "num_latent_tokens": self.num_latent_tokens,
            },
        }, path)
        logger.info(f"LLaVACompressor saved to {path}")

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
        ckpt = torch.load(path, map_location=self.mem_embedding.device)
        self._copy_mem_from_checkpoint(ckpt["mem_embedding"])
        self.retrieval_proj.load_state_dict(ckpt["retrieval_proj"])
        self.cross_proj.load_state_dict(ckpt["cross_proj"])
        if "decode_proj" in ckpt:
            self.decode_proj.load_state_dict(ckpt["decode_proj"])
        if "img_embed_decode_proj" in ckpt:
            self.img_embed_decode_proj.load_state_dict(ckpt["img_embed_decode_proj"])

        # Load compression LoRAs into Vicuna-7B
        lm = self._lm()
        comp_sd = lm.state_dict()
        comp_sd.update(ckpt["lora_compress"])
        comp_sd.update(ckpt["lora_query"])
        lm.load_state_dict(comp_sd, strict=False)

        # Load decode LoRAs into LLaMA-1B decoder
        dec_sd = self.decoder_lm.state_dict()
        if "lora_decode" in ckpt:
            dec_sd.update(ckpt["lora_decode"])
        if "lora_query_decode" in ckpt:
            dec_sd.update(ckpt["lora_query_decode"])
        if "lora_image_decode" in ckpt:
            dec_sd.update(ckpt["lora_image_decode"])
        self.decoder_lm.load_state_dict(dec_sd, strict=False)
        logger.info(f"LLaVACompressor loaded from {path}")

    # ------------------------------------------------------------------ prompt helpers

    @staticmethod
    def format_compress_prompt(title: str, has_image: bool) -> str:
        if has_image:
            # <image> must come FIRST so visual patches precede the caption tokens,
            # matching LLaVA-1.5's training convention.
            return f"USER: {_IMAGE_TOKEN}\nTitle: {title}\nASSISTANT:"
        return f"USER: {title}\nASSISTANT:"

    @staticmethod
    def format_query_prompt(question: str) -> str:
        return f"USER: {question}\nASSISTANT:"

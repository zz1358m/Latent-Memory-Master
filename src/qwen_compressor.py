"""
Qwen2.5-VL Compressor
=====================
Drop-in replacement for LLaVACompressor using Qwen2.5-VL-3B as the
compressor backbone and (optionally) Qwen2.5-VL-7B as the generator.

Architecture
------------
  Base model  : Qwen/Qwen2.5-VL-3B-Instruct  → Qwen2VLForConditionalGeneration
  Vision enc  : Qwen2VL ViT (inside the model) — frozen
  Language LM : Qwen2.5-3B — two LoRA adapters: "compress", "query"

  Decoder LM  : Qwen/Qwen2.5-VL-3B-Instruct — three LoRA adapters:
                  "decode"        : [MEM] → reconstruct text sentence
                  "query_decode"  : [MEM] → reconstruct query text
                  "image_decode"  : [MEM] → reconstruct image caption title

  Trainable heads (auto-sized from loaded model):
    mem_embedding   : (1, hidden_size)  learnable [MEM] parameter
    retrieval_proj  : Linear(hidden_size, retrieval_dim) → LayerNorm → L2-norm
    decode_proj     : Linear(hidden_size, decoder_hidden) → LayerNorm
    cross_proj      : CrossModelProjection(hidden_size → generator_hidden)

  Note: img_embed_decode_proj (CLIP CLS MSE) is omitted.  Qwen2VL's visual
  encoder merges patches directly into LM hidden space via PatchMerger, so
  there is no clean CLIP-CLS equivalent.  Set lambda_embed_recon=0.0 in
  config_qwen.yaml (the default).

LoRA target modules
-------------------
  ["q_proj", "k_proj", "v_proj", "o_proj"] are applied to the *entire*
  Qwen2VLForConditionalGeneration, but Qwen2VL's vision encoder uses a
  single fused "qkv" projection, so these names only match the language-
  model attention layers — the vision encoder is untouched.

Prompt convention
-----------------
  compress_batch() applies the Qwen chat template internally.
  Pass raw content strings (sentence, caption, or question); do NOT
  pre-wrap with "USER:" / "ASSISTANT:" as in LLaVACompressor.

Checkpoint format
-----------------
  {
    "mem_embedding"   : tensor,
    "retrieval_proj"  : state_dict,
    "cross_proj"      : state_dict,
    "decode_proj"     : state_dict,
    "lora_compress"   : state_dict,   # from Qwen2.5-3B
    "lora_query"      : state_dict,   # from Qwen2.5-3B
    "lora_decode"     : state_dict,   # from Qwen2.5-VL-3B decoder
    "lora_query_decode": state_dict,  # from Qwen2.5-VL-3B decoder
    "lora_image_decode": state_dict,  # from Qwen2.5-VL-3B decoder
    "config"          : dict,
  }
"""

import logging
import os
import json
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    CLIPModel,
    Qwen2VLForConditionalGeneration,
)

logger = logging.getLogger(__name__)

_MEM_TOKEN = "[MEM]"


def _summarize_config(cfg) -> str:
    vision_cfg = getattr(cfg, "vision_config", None)
    try:
        text_hidden = get_qwen_vl_text_hidden_size(cfg)
    except Exception:
        text_hidden = getattr(cfg, "hidden_size", None)
    if vision_cfg is None:
        vision_summary = "None"
    else:
        keys = ["model_type", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads"]
        parts = []
        for key in keys:
            if hasattr(vision_cfg, key):
                parts.append(f"{key}={getattr(vision_cfg, key)}")
        vision_summary = ", ".join(parts) if parts else str(type(vision_cfg).__name__)
    return (
        f"model_type={getattr(cfg, 'model_type', None)}, "
        f"hidden_size={text_hidden}, "
        f"vision_config=({vision_summary})"
    )


def _log_local_config_json(model_name: str, tag: str):
    if not os.path.isdir(model_name):
        return
    cfg_path = os.path.join(model_name, "config.json")
    if not os.path.exists(cfg_path):
        logger.warning(f"{tag} local model dir has no config.json: {model_name}")
        return
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        vision_raw = raw.get("vision_config", {})
        vision_summary = {
            k: vision_raw.get(k)
            for k in ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads")
            if k in vision_raw
        }
        logger.info(
            f"{tag} local config.json ({cfg_path}): "
            f"model_type={raw.get('model_type')}, "
            f"hidden_size={raw.get('hidden_size')}, "
            f"vision_config={vision_summary}"
        )
    except Exception as exc:
        logger.warning(f"Failed reading {tag} local config.json at {cfg_path}: {exc}")


def resolve_qwen_vl_model_class(model_name: str):
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_type = getattr(cfg, "model_type", None)
    if model_type == "qwen2_5_vl":
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            return Qwen2_5_VLForConditionalGeneration
        except Exception as exc:
            raise RuntimeError(
                "This checkpoint is qwen2_5_vl, but your transformers build does not expose "
                "Qwen2_5_VLForConditionalGeneration. Upgrade transformers to a version that "
                "supports Qwen2.5-VL."
            ) from exc
    if model_type == "qwen2_vl":
        return Qwen2VLForConditionalGeneration
    raise ValueError(f"Unsupported Qwen VL model_type={model_type} for model {model_name}")


def resolve_decoder_model_class(model_name: str):
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_type = getattr(cfg, "model_type", None)
    if model_type in {"qwen2_vl", "qwen2_5_vl"}:
        return "qwen_vl", resolve_qwen_vl_model_class(model_name)
    return "text", AutoModelForCausalLM


def get_qwen_vl_text_hidden_size(cfg) -> int:
    hidden_size = getattr(cfg, "hidden_size", None)
    if hidden_size is not None:
        return hidden_size

    for attr in ("text_config", "language_config", "llm_config"):
        sub_cfg = getattr(cfg, attr, None)
        if sub_cfg is None:
            continue
        sub_hidden = getattr(sub_cfg, "hidden_size", None)
        if sub_hidden is not None:
            return sub_hidden

    if isinstance(cfg, dict):
        if cfg.get("hidden_size") is not None:
            return cfg["hidden_size"]
        for key in ("text_config", "language_config", "llm_config"):
            sub_cfg = cfg.get(key)
            if isinstance(sub_cfg, dict) and sub_cfg.get("hidden_size") is not None:
                return sub_cfg["hidden_size"]

    raise AttributeError(
        f"Could not determine language-model hidden size from config type {type(cfg).__name__}"
    )


# ---------------------------------------------------------------------------
# Cross-model projection (compressor hidden → generator hidden)
# ---------------------------------------------------------------------------

class CrossModelProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
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

class QwenVLCompressor(nn.Module):
    """
    Qwen2.5-VL-3B based compressor with two LoRA adapters on the base LM,
    plus a separate LLaMA-1B decoder with three LoRA adapters.

    Parameters
    ----------
    model_name       : HuggingFace model ID for the Qwen2.5-VL compressor
    decoder_name     : HuggingFace model ID for the Qwen2.5-VL text decoder
    lora_r           : LoRA rank (shared across all adapters)
    lora_alpha       : LoRA alpha
    lora_dropout     : LoRA dropout
    target_modules   : LM layer names to apply LoRA (default: attention projections)
    retrieval_dim    : dimension of the L2-normalised retrieval embedding
    generator_hidden : hidden size of the generator model for cross_proj
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        decoder_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        clip_model_name: str = "openai/clip-vit-large-patch14-336",
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        decode_lora_r: Optional[int] = None,
        decode_lora_alpha: Optional[int] = None,
        decode_lora_dropout: Optional[float] = None,
        decode_target_modules: Optional[List[str]] = None,
        retrieval_dim: int = 512,
        generator_hidden: int = 3584,  # Qwen2.5-VL-7B hidden size
        num_latent_tokens: int = 1,
        load_clip_model: bool = False,
        clip_hidden_size: int = 1024,
    ):
        super().__init__()
        self.model_name    = model_name
        self.decoder_name  = decoder_name
        self.clip_model_name = clip_model_name
        self.retrieval_dim = retrieval_dim
        self.num_latent_tokens = int(num_latent_tokens)
        if self.num_latent_tokens < 1:
            raise ValueError("num_latent_tokens must be >= 1")

        if target_modules is None:
            # These names only hit LM transformer layers; Qwen2VL vision encoder
            # uses a fused "qkv" projection so it is unaffected.
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
        logger.info(f"Loading Qwen2VL processor: {model_name}")
        _log_local_config_json(model_name, "Qwen base")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer  = self.processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            base_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            logger.info(f"Qwen base config summary ({model_name}): {_summarize_config(base_cfg)}")
        except Exception as exc:
            logger.warning(f"Failed to inspect base config for {model_name}: {exc}")

        # Register [MEM] as a new special token
        self.tokenizer.add_special_tokens({"additional_special_tokens": [_MEM_TOKEN]})
        self.mem_token_id = self.tokenizer.convert_tokens_to_ids(_MEM_TOKEN)

        # ------------------------------------------------------------------ base model
        base_model_cls = resolve_qwen_vl_model_class(model_name)
        logger.info(f"Loading Qwen VL base model with {base_model_cls.__name__}: {model_name}")
        base = base_model_cls.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        base.resize_token_embeddings(len(self.tokenizer))

        # Freeze everything first; LoRA will unfreeze only the added delta params
        for p in base.parameters():
            p.requires_grad_(False)

        # Read hidden size from config
        self.hidden_size = get_qwen_vl_text_hidden_size(base.config)

        # ------------------------------------------------------------------ LoRA adapters
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

        # Apply LoRA to the full Qwen2VL model.
        # target_modules only match LM attention; vision encoder is safe.
        peft_model = get_peft_model(base, _lora_cfg(), adapter_name="compress")
        peft_model.add_adapter("query", _lora_cfg())

        # Cast new LoRA params to bfloat16
        for param in peft_model.parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)

        self.model = peft_model
        self._set_adapter("compress")

        # ------------------------------------------------------------------ decoder LM (Qwen-VL or text-only causal LM)
        logger.info(f"Loading decoder LM: {decoder_name}")
        _log_local_config_json(decoder_name, "Qwen decoder")
        try:
            decoder_cfg = AutoConfig.from_pretrained(decoder_name, trust_remote_code=True)
            logger.info(f"Qwen decoder config summary ({decoder_name}): {_summarize_config(decoder_cfg)}")
        except Exception as exc:
            logger.warning(f"Failed to inspect decoder config for {decoder_name}: {exc}")
        self.decoder_kind, decoder_model_cls = resolve_decoder_model_class(decoder_name)
        if self.decoder_kind == "qwen_vl":
            self.decoder_processor = AutoProcessor.from_pretrained(decoder_name)
            self.decoder_tokenizer = getattr(self.decoder_processor, "tokenizer", None)
            if self.decoder_tokenizer is None:
                logger.warning(
                    "Decoder processor for %s has no tokenizer attribute; "
                    "falling back to AutoTokenizer.",
                    decoder_name,
                )
                self.decoder_tokenizer = AutoTokenizer.from_pretrained(
                    decoder_name, trust_remote_code=True
                )
        else:
            self.decoder_processor = None
            self.decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_name, trust_remote_code=True)
        if self.decoder_tokenizer.pad_token is None:
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token

        logger.info(
            f"Loading decoder LM with {decoder_model_cls.__name__} "
            f"(kind={self.decoder_kind}): {decoder_name}"
        )
        decoder_base = decoder_model_cls.from_pretrained(decoder_name, torch_dtype=torch.bfloat16)
        if self.decoder_kind == "qwen_vl":
            self.decoder_hidden_size = get_qwen_vl_text_hidden_size(decoder_base.config)
        else:
            self.decoder_hidden_size = decoder_base.config.hidden_size

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

        # ------------------------------------------------------------------ optional external CLIP target model
        self.clip_processor = None
        self.clip_model = None
        self.clip_hidden_size = int(clip_hidden_size)
        if load_clip_model:
            logger.info(f"Loading CLIP target model: {clip_model_name}")
            self.clip_processor = AutoProcessor.from_pretrained(clip_model_name)
            self.clip_model = CLIPModel.from_pretrained(clip_model_name, torch_dtype=torch.float32)
            self.clip_model.eval()
            for p in self.clip_model.parameters():
                p.requires_grad_(False)
            self.clip_hidden_size = int(self.clip_model.vision_model.config.hidden_size)
        else:
            logger.info(
                "Skipping CLIP target model load; using clip_hidden_size=%d for projection heads.",
                self.clip_hidden_size,
            )

        # ------------------------------------------------------------------ trainable heads
        self.mem_embedding = nn.Parameter(
            torch.randn(self.num_latent_tokens, self.hidden_size, dtype=torch.float32) * 0.02
        )
        self.retrieval_proj = nn.Sequential(
            nn.Linear(self.hidden_size, retrieval_dim, dtype=torch.float32),
            nn.LayerNorm(retrieval_dim),
        )
        # Bridges Qwen-3B latent (hidden_size) → Qwen2.5-VL-3B decoder input
        self.decode_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.decoder_hidden_size, dtype=torch.float32),
            nn.LayerNorm(self.decoder_hidden_size),
        )
        # Bridges compressor hidden → generator hidden (e.g. 2048 → 3584)
        self.cross_proj = CrossModelProjection(
            in_dim=self.hidden_size, out_dim=generator_hidden
        )
        _mid = (self.hidden_size + self.clip_hidden_size) // 2
        self.img_embed_decode_proj = nn.Sequential(
            nn.Linear(self.hidden_size, _mid, dtype=torch.float32),
            nn.GELU(),
            nn.LayerNorm(_mid),
            nn.Linear(_mid, self.clip_hidden_size, dtype=torch.float32),
        )

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"QwenVLCompressor trainable params: {n_train / 1e6:.1f}M")

    # ------------------------------------------------------------------ adapter switching

    def _set_adapter(self, name: str):
        self.model.set_adapter(name)
        for n, p in self.model.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)

    def _set_decode_adapter(self, name: str):
        self.decoder_lm.set_adapter(name)
        for n, p in self.decoder_lm.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)

    # ------------------------------------------------------------------ gradient checkpointing

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on the Qwen2VL base model."""
        base = self.model.base_model.model   # unwrap PEFT → Qwen2VLForConditionalGeneration
        base.enable_input_require_grads()
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # ------------------------------------------------------------------ compress

    def compress_batch(
        self,
        texts: List[str],
        images: Optional[List[Optional[Image.Image]]] = None,
        with_grad: bool = True,
        adapter: str = "compress",
    ) -> torch.Tensor:
        """
        Compress a batch of (text, optional image) pairs into [MEM] latent tokens.

        Parameters
        ----------
        texts  : raw content strings (sentence, caption, or question).
                 Do NOT pre-format with Vicuna "USER:" / "ASSISTANT:".
                 Qwen chat template is applied internally.
        images : parallel list of PIL Images, or None for text-only samples.

        Returns
        -------
        latents : (B, hidden_size) float32
        """
        self._set_adapter(adapter)
        device = self.mem_embedding.device
        dtype  = torch.bfloat16

        if images is None:
            images = [None] * len(texts)

        latents: List[Optional[torch.Tensor]] = [None] * len(texts)
        ctx = torch.enable_grad if with_grad else torch.no_grad

        with ctx():
            image_indices = [i for i, img in enumerate(images) if img is not None]
            if image_indices:
                image_texts = []
                image_pils = []
                for i in image_indices:
                    image_texts.append(
                        self.processor.apply_chat_template(
                            [{
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": images[i]},
                                    {"type": "text", "text": texts[i]},
                                ],
                            }],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    )
                    image_pils.append(images[i])

                enc = self.processor(
                    text=image_texts,
                    images=image_pils,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                mem_ids = torch.full(
                    (enc.input_ids.shape[0], self.num_latent_tokens),
                    self.mem_token_id,
                    device=device,
                    dtype=enc.input_ids.dtype,
                )
                ids_ext = torch.cat([enc.input_ids, mem_ids], dim=1)
                mask_ext = torch.cat(
                    [
                        enc.attention_mask,
                        torch.ones(
                            enc.input_ids.shape[0],
                            self.num_latent_tokens,
                            device=device,
                            dtype=enc.attention_mask.dtype,
                        ),
                    ],
                    dim=1,
                )

                out = self.model(
                    input_ids=ids_ext,
                    pixel_values=enc.pixel_values.to(dtype),
                    image_grid_thw=enc.image_grid_thw,
                    attention_mask=mask_ext,
                    output_hidden_states=True,
                )
                batch_mem = (
                    out.hidden_states[-1][:, -self.num_latent_tokens :, :].float()
                    + self.mem_embedding.float().unsqueeze(0)
                )
                for row, idx in enumerate(image_indices):
                    latents[idx] = batch_mem[row]

            text_indices = [i for i, img in enumerate(images) if img is None]
            if text_indices:
                formatted = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": texts[i]}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for i in text_indices
                ]
                enc = self.tokenizer(formatted, padding=True, return_tensors="pt").to(device)
                mem_ids = torch.full(
                    (enc.input_ids.shape[0], self.num_latent_tokens),
                    self.mem_token_id,
                    device=device,
                    dtype=enc.input_ids.dtype,
                )
                ids_ext = torch.cat([enc.input_ids, mem_ids], dim=1)
                mask_ext = torch.cat(
                    [
                        enc.attention_mask,
                        torch.ones(
                            enc.input_ids.shape[0],
                            self.num_latent_tokens,
                            device=device,
                            dtype=enc.attention_mask.dtype,
                        ),
                    ],
                    dim=1,
                )

                out = self.model(
                    input_ids=ids_ext,
                    attention_mask=mask_ext,
                    output_hidden_states=True,
                )
                batch_mem = (
                    out.hidden_states[-1][:, -self.num_latent_tokens :, :].float()
                    + self.mem_embedding.float().unsqueeze(0)
                )
                for row, idx in enumerate(text_indices):
                    latents[idx] = batch_mem[row]

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
        L_recon: compress → latent → decode autoregressive CE loss.

        texts        : raw content strings (formatted by compress_batch internally)
        target_texts : CE targets for the decoder (raw text, no templates).
                       If None, uses `texts` as targets.
        """
        if not texts:
            return torch.tensor(0.0, device=self.mem_embedding.device)

        device = self.mem_embedding.device
        decode_targets = target_texts if target_texts is not None else texts

        latents = self.compress_batch(texts, images=images, with_grad=True, adapter=adapter)

        self._set_decode_adapter(decode_adapter)
        embed_fn = self.decoder_lm.get_input_embeddings()

        total_loss = torch.tensor(0.0, device=device)
        count = 0

        for i, text in enumerate(decode_targets):
            z_proj = self.decode_proj(latents[i].float())         # (T, decoder_hidden_size)
            z = z_proj.unsqueeze(0)                               # (1, T, decoder_hidden_size)

            enc = self.decoder_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                add_special_tokens=True,
            ).to(device)
            target_ids = enc.input_ids
            L = target_ids.shape[1]

            text_embeds  = embed_fn(target_ids)                   # (1, L, decoder_hidden_size)
            input_embeds = torch.cat(
                [z.to(text_embeds.dtype), text_embeds[:, :-1, :]], dim=1
            )
            attn_mask = torch.ones(1, self.num_latent_tokens + L - 1, device=device, dtype=torch.long)

            out    = self.decoder_lm(inputs_embeds=input_embeds, attention_mask=attn_mask)
            logits = out.logits    # (1, L, V)

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

    # ------------------------------------------------------------------ embed_recon stub

    def embed_reconstruction_loss_batch(
        self,
        texts: List[str],
        images: Optional[List[Optional[Image.Image]]] = None,
        adapter: str = "compress",
    ) -> torch.Tensor:
        """
        Not implemented for Qwen2VL (lambda_embed_recon should be 0.0).
        Qwen2VL's PatchMerger projects visual tokens directly into LM hidden
        space; there is no independent CLIP-CLS equivalent.
        """
        return torch.tensor(0.0, device=self.mem_embedding.device)

    def get_clip_cls_batch(self, pil_images: List[Image.Image]):
        """Stub — not implemented for Qwen2VL. Returns list of None."""
        return [None] * len(pil_images)

    # ------------------------------------------------------------------ retrieval / projection

    def get_retrieval_embedding(self, latents: torch.Tensor) -> torch.Tensor:
        """Project latents to L2-normalised retrieval space."""
        if latents.dim() == 3:
            latents = latents.mean(dim=1)
        proj = self.retrieval_proj(latents.float())
        return F.normalize(proj, dim=-1)

    def project_for_generator(self, latents: torch.Tensor) -> torch.Tensor:
        """Project latents into the generator's embedding space via cross_proj."""
        return self.cross_proj(latents.float())

    def embed_query_batch(
        self, questions: List[str], with_grad: bool = True
    ) -> torch.Tensor:
        """Embed questions using the 'query' adapter."""
        return self.compress_batch(
            questions, images=None, with_grad=with_grad, adapter="query"
        )

    # ------------------------------------------------------------------ checkpoint

    def save(self, path: str):
        def _filter(sd, adapter_name):
            tag = f".{adapter_name}."
            return {k: v for k, v in sd.items() if tag in k}

        comp_sd = self.model.state_dict()
        dec_sd  = self.decoder_lm.state_dict()
        torch.save({
            "mem_embedding"    : self.mem_embedding.data,
            "retrieval_proj"   : self.retrieval_proj.state_dict(),
            "cross_proj"       : self.cross_proj.state_dict(),
            "decode_proj"      : self.decode_proj.state_dict(),
            "lora_compress"    : _filter(comp_sd, "compress"),
            "lora_query"       : _filter(comp_sd, "query"),
            "lora_decode"      : _filter(dec_sd,  "decode"),
            "lora_query_decode": _filter(dec_sd,  "query_decode"),
            "lora_image_decode": _filter(dec_sd,  "image_decode"),
            "config": {
                "model_name"         : self.model_name,
                "decoder_name"       : self.decoder_name,
                "retrieval_dim"      : self.retrieval_dim,
                "hidden_size"        : self.hidden_size,
                "decoder_hidden_size": self.decoder_hidden_size,
                "num_latent_tokens"  : self.num_latent_tokens,
            },
        }, path)
        logger.info(f"QwenVLCompressor saved to {path}")

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

        comp_sd = self.model.state_dict()
        comp_sd.update(ckpt.get("lora_compress", {}))
        comp_sd.update(ckpt.get("lora_query",    {}))
        self.model.load_state_dict(comp_sd, strict=False)

        dec_sd = self.decoder_lm.state_dict()
        for key in ("lora_decode", "lora_query_decode", "lora_image_decode"):
            if key in ckpt:
                dec_sd.update(ckpt[key])
        self.decoder_lm.load_state_dict(dec_sd, strict=False)
        logger.info(f"QwenVLCompressor loaded from {path}")

    # ------------------------------------------------------------------ prompt helpers

    @staticmethod
    def format_compress_prompt(text: str, has_image: bool) -> str:
        """
        Identity helper — compress_batch applies Qwen chat template internally.
        Returns the raw content string unchanged.
        Provided for interface compatibility; callers should pass raw text.
        """
        return text

    @staticmethod
    def format_query_prompt(question: str) -> str:
        """Identity — compress_batch applies the template internally."""
        return question


def _qwen_embed_reconstruction_loss_batch(
    self,
    texts: List[str],
    images: Optional[List[Optional[Image.Image]]] = None,
    adapter: str = "compress",
) -> torch.Tensor:
    if not texts:
        return torch.tensor(0.0, device=self.mem_embedding.device)

    device = self.mem_embedding.device
    latents = self.compress_batch(texts, images=images, with_grad=True, adapter=adapter)
    img_indices = [i for i, img in enumerate(images or []) if img is not None]
    if not img_indices:
        return torch.tensor(0.0, device=device)
    if self.clip_model is None or self.clip_processor is None:
        raise RuntimeError(
            "CLIP target model is not loaded. Instantiate QwenVLCompressor with "
            "load_clip_model=True when lambda_embed_recon > 0."
        )

    pil_batch = [images[i] for i in img_indices]
    clip_inputs = self.clip_processor(images=pil_batch, return_tensors="pt").to(device)
    with torch.no_grad():
        clip_out = self.clip_model.vision_model(
            pixel_values=clip_inputs["pixel_values"].to(dtype=torch.float32)
        )
        clip_cls_batch = clip_out.last_hidden_state[:, 0, :].float()

    total_loss = torch.tensor(0.0, device=device)
    for j, i in enumerate(img_indices):
        pred_cls = self.img_embed_decode_proj(latents[i].float().mean(dim=0))
        total_loss = total_loss + F.mse_loss(pred_cls, clip_cls_batch[j])
    return total_loss / len(img_indices)


def _qwen_get_clip_cls_batch(self, pil_images: List[Image.Image]):
    if not pil_images:
        return []
    if self.clip_model is None or self.clip_processor is None:
        raise RuntimeError(
            "CLIP target model is not loaded. Instantiate QwenVLCompressor with "
            "load_clip_model=True before calling get_clip_cls_batch()."
        )
    device = self.mem_embedding.device
    outputs: List[Optional[torch.Tensor]] = []
    for pil in pil_images:
        try:
            clip_inputs = self.clip_processor(images=pil, return_tensors="pt").to(device)
            with torch.no_grad():
                clip_out = self.clip_model.vision_model(
                    pixel_values=clip_inputs["pixel_values"].to(dtype=torch.float32)
                )
            outputs.append(clip_out.last_hidden_state[0, 0, :].float().cpu())
        except Exception:
            outputs.append(None)
    return outputs


def _qwen_save(self, path: str):
    def _filter(sd, adapter_name):
        tag = f".{adapter_name}."
        return {k: v for k, v in sd.items() if tag in k}

    comp_sd = self.model.state_dict()
    dec_sd = self.decoder_lm.state_dict()
    torch.save({
        "mem_embedding": self.mem_embedding.data,
        "retrieval_proj": self.retrieval_proj.state_dict(),
        "cross_proj": self.cross_proj.state_dict(),
        "decode_proj": self.decode_proj.state_dict(),
        "img_embed_decode_proj": self.img_embed_decode_proj.state_dict(),
        "lora_compress": _filter(comp_sd, "compress"),
        "lora_query": _filter(comp_sd, "query"),
        "lora_decode": _filter(dec_sd, "decode"),
        "lora_query_decode": _filter(dec_sd, "query_decode"),
        "lora_image_decode": _filter(dec_sd, "image_decode"),
        "config": {
            "model_name": self.model_name,
            "decoder_name": self.decoder_name,
            "clip_model_name": self.clip_model_name,
            "retrieval_dim": self.retrieval_dim,
            "hidden_size": self.hidden_size,
            "decoder_hidden_size": self.decoder_hidden_size,
            "clip_hidden_size": self.clip_hidden_size,
            "num_latent_tokens": self.num_latent_tokens,
        },
    }, path)
    logger.info(f"QwenVLCompressor saved to {path}")


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


def _qwen_load(self, path: str):
    ckpt = torch.load(path, map_location=self.mem_embedding.device)
    _copy_mem_from_checkpoint(self, ckpt["mem_embedding"])
    self.retrieval_proj.load_state_dict(ckpt["retrieval_proj"])
    self.cross_proj.load_state_dict(ckpt["cross_proj"])
    if "decode_proj" in ckpt:
        self.decode_proj.load_state_dict(ckpt["decode_proj"])
    if "img_embed_decode_proj" in ckpt:
        self.img_embed_decode_proj.load_state_dict(ckpt["img_embed_decode_proj"])

    comp_sd = self.model.state_dict()
    comp_sd.update(ckpt.get("lora_compress", {}))
    comp_sd.update(ckpt.get("lora_query", {}))
    self.model.load_state_dict(comp_sd, strict=False)

    dec_sd = self.decoder_lm.state_dict()
    for key in ("lora_decode", "lora_query_decode", "lora_image_decode"):
        if key in ckpt:
            dec_sd.update(ckpt[key])
    self.decoder_lm.load_state_dict(dec_sd, strict=False)
    logger.info(f"QwenVLCompressor loaded from {path}")


QwenVLCompressor.embed_reconstruction_loss_batch = _qwen_embed_reconstruction_loss_batch
QwenVLCompressor.get_clip_cls_batch = _qwen_get_clip_cls_batch
QwenVLCompressor.save = _qwen_save
QwenVLCompressor.load = _qwen_load

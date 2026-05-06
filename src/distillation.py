"""
Distillation Loss  (L_distill)
================================
Aligns the frozen 8B generator's output distribution when processing:

  Teacher (text context, chat-template formatted):
    <BOS><system>...</system><user>Context: {sentence}\nQuestion: {q}</user><assistant>

  Student (latent context, same template split around the context slot):
    embed(<BOS><system>...</system><user>) + embed("Latent context: ") + lat
    + embed(\nQuestion: {q}</user><assistant>)

For multi-sentence (K positives):
  Teacher: ... <user>Context 1: {s1}\nContext 2: {s2}\n...\nQuestion: {q}</user><assistant>
  Student: prefix_embeds + embed("Latent context 1: ") + lat1
           + embed("\nLatent context 2: ") + lat2 + ... + suffix_embeds

The chat template is split with a placeholder so we never hardcode
model-specific special tokens.

Loss: KL(student || teacher),  stop_gradient on teacher side.

Gradient flows: KL_loss → student logits → 8B forward (frozen weights,
grad through input_embeds) → latent_token → CrossModelProjection → 1B compress
"""

import logging
import torch
import torch.nn.functional as F
from jinja2.exceptions import TemplateError

logger = logging.getLogger(__name__)

_SYSTEM_MSG = (
    "You are a helpful assistant. "
    "Answer the question concisely (a few words or a short phrase) "
    "based on the provided context."
)
_PLACEHOLDER = "ZZZCTXZZZ"


def _latent_token_rows(latent_projected: torch.Tensor) -> torch.Tensor:
    """Return latent context as (T, H), preserving token order."""
    if latent_projected.dim() == 1:
        return latent_projected.unsqueeze(0)
    return latent_projected.reshape(-1, latent_projected.shape[-1])


def _split_chat_template(generator_tokenizer, query: str):
    """
    Build the chat-template string with a placeholder where the context goes,
    then split it into (prefix_str, suffix_str).

    prefix_str : everything from <BOS> up to (not including) the context
    suffix_str : from after the context through <assistant> generation prompt
                 (starts with "\\nQuestion: {query}...")

    Both strings, when tokenized with add_special_tokens=False, correctly
    include all special tokens because apply_chat_template already wrote them
    as literal strings (e.g. "<|begin_of_text|>").
    """
    chat_template = getattr(generator_tokenizer, "chat_template", None)
    if chat_template:
        tok_name = str(getattr(generator_tokenizer, "name_or_path", "")).lower()
        tok_cls = generator_tokenizer.__class__.__name__.lower()
        if "gemma-3" in tok_name or "gemma3" in tok_cls:
            user_content = [{
                "type": "text",
                "text": f"{_SYSTEM_MSG}\n\n{_PLACEHOLDER}\nQuestion: {query}",
            }]
            full = generator_tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            system_content = _SYSTEM_MSG
            user_content = f"{_PLACEHOLDER}\nQuestion: {query}"
            try:
                full = generator_tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except TemplateError as exc:
                if "System role not supported" not in str(exc):
                    raise
                full = generator_tokenizer.apply_chat_template(
                    [{"role": "user", "content": f"{_SYSTEM_MSG}\n\n{user_content}"}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
    else:
        bos = generator_tokenizer.bos_token or ""
        full = (
            f"{bos}{_SYSTEM_MSG}\n"
            f"USER: {_PLACEHOLDER}\n"
            f"Question: {query}\n"
            f"ASSISTANT:"
        )
    prefix_str, suffix_str = full.split(_PLACEHOLDER)
    return prefix_str, suffix_str


def _tok(generator_tokenizer, text, device, max_len=512):
    return generator_tokenizer(
        text, return_tensors="pt", add_special_tokens=False,
        truncation=True, max_length=max_len,
    ).input_ids.to(device)


def compute_distillation_loss(
    generator,
    generator_tokenizer,
    query: str,
    sentence: str,
    latent_projected: torch.Tensor,   # (H_8B,) output of CrossModelProjection
    device: str = "cuda",
) -> torch.Tensor:
    """
    KL-divergence distillation between teacher (text context) and student (latent context).

    Both use the full chat template; only the context slot differs:
      Teacher: prefix + "Context: {sentence}" + suffix
      Student: embed(prefix) + embed("Latent context: ") + lat + embed(suffix)

    The logits at the LAST position (first generated answer token) are compared.
    """
    embed = generator.get_input_embeddings()
    dtype = next(generator.parameters()).dtype

    prefix_str, suffix_str = _split_chat_template(generator_tokenizer, query)

    def _e(text, max_len=512):
        return embed(_tok(generator_tokenizer, text, device, max_len)).to(dtype)

    # ---- Teacher: prefix + "Context: {sentence}" + suffix  (all text) ----
    teacher_text = prefix_str + f"Context: {sentence}" + suffix_str
    t_embeds = _e(teacher_text, max_len=512)                        # (1, L, H)

    # ---- Student: embed(prefix) + embed("Latent context: ") + lat + embed(suffix) ----
    s_embeds = torch.cat([
        _e(prefix_str),                                             # (1, Lp, H)
        _e("Latent context: "),                                     # (1, Ll, H)
        _latent_token_rows(latent_projected).to(dtype).unsqueeze(0), # (1, T, H)
        _e(suffix_str),                                             # (1, Ls, H)
    ], dim=1)

    # ---- Teacher forward (no grad) ----
    with torch.no_grad():
        t_out = generator(inputs_embeds=t_embeds, use_cache=False)
    t_probs = F.softmax(t_out.logits[0, -1, :].float(), dim=-1).detach()

    # ---- Student forward (grad flows through lat → cross_proj → compress) ----
    s_out = generator(inputs_embeds=s_embeds, use_cache=False)
    s_log_probs = F.log_softmax(s_out.logits[0, -1, :].float(), dim=-1)

    return F.kl_div(s_log_probs, t_probs, reduction="sum")


def batch_distillation_loss(
    generator,
    generator_tokenizer,
    queries: list,
    sentences: list,
    latents_projected: torch.Tensor,   # (B, H_8B)
    device: str = "cuda",
) -> torch.Tensor:
    """Compute distillation loss for a batch, averaged over samples."""
    losses = [
        compute_distillation_loss(generator, generator_tokenizer, q, s, lat, device)
        for q, s, lat in zip(queries, sentences, latents_projected)
    ]
    return torch.stack(losses).mean()


def compute_distillation_loss_multi(
    generator,
    generator_tokenizer,
    query: str,
    sentences: list,                    # K ground-truth positive sentences
    latents_projected: torch.Tensor,    # (K, H_8B) — one latent per sentence
    device: str = "cuda",
    n_distill_tokens: int = 1,          # number of answer tokens to supervise
) -> torch.Tensor:
    """
    Multi-sentence distillation loss.

    Teacher: prefix + "Context 1: {s1}\\nContext 2: {s2}\\n..." + suffix
    Student: embed(prefix)
             + embed("Latent context 1: ") + lat1
             + embed("\\nLatent context 2: ") + lat2
             + ...
             + embed(suffix)

    KL(student || teacher) summed over n_distill_tokens answer positions.

    For n_distill_tokens > 1:
      - Greedily generate n_distill_tokens from the teacher (no grad).
      - Append the first n_distill_tokens-1 generated token embeddings to
        both teacher and student inputs (teacher-forcing).
      - One forward pass each; compare logits at the last n_distill_tokens positions.
    """
    embed = generator.get_input_embeddings()
    dtype = next(generator.parameters()).dtype

    prefix_str, suffix_str = _split_chat_template(generator_tokenizer, query)

    def _e(text, max_len=512):
        return embed(_tok(generator_tokenizer, text, device, max_len)).to(dtype)

    # ---- Teacher: prefix + all context sentences + suffix ----
    ctx_text = "\n".join(f"Context {i+1}: {s}" for i, s in enumerate(sentences))
    t_embeds = _e(prefix_str + ctx_text + suffix_str, max_len=1024)  # (1, L_t, H)

    # ---- Student: interleave "Latent context i: " labels with latent tokens ----
    student_parts = [_e(prefix_str)]
    for i, lat in enumerate(latents_projected):
        label = f"Latent context {i+1}: " if i == 0 else f"\nLatent context {i+1}: "
        student_parts.append(_e(label))
        student_parts.append(_latent_token_rows(lat).to(dtype).unsqueeze(0))
    student_parts.append(_e(suffix_str))
    s_embeds = torch.cat(student_parts, dim=1)                       # (1, L_s, H)

    # ---- For n_distill_tokens > 1: extend both inputs with teacher-generated tokens ----
    if n_distill_tokens > 1:
        with torch.no_grad():
            L_t = t_embeds.shape[1]
            gen_out = generator.generate(
                inputs_embeds=t_embeds,
                max_new_tokens=n_distill_tokens,
                do_sample=False,
                pad_token_id=generator_tokenizer.eos_token_id,
            )
            # Handle both transformers <4.43 (new tokens only) and >=4.43 (full seq)
            gen_ids = gen_out[0, L_t:] if gen_out.shape[1] > L_t else gen_out[0]
            gen_ids = gen_ids[:n_distill_tokens]                     # (N,)

        # Append first N-1 generated tokens to both inputs for teacher-forcing
        prefix_embs = embed(gen_ids[:n_distill_tokens - 1].unsqueeze(0)).to(dtype)  # (1, N-1, H)
        t_embeds = torch.cat([t_embeds, prefix_embs], dim=1)        # (1, L_t+N-1, H)
        s_embeds = torch.cat([s_embeds, prefix_embs], dim=1)        # (1, L_s+N-1, H)

    # ---- Teacher forward (no grad) ----
    with torch.no_grad():
        t_out = generator(inputs_embeds=t_embeds, use_cache=False)
    t_probs = F.softmax(
        t_out.logits[0, -n_distill_tokens:, :].float(), dim=-1
    ).detach()                                                       # (N, V)

    # ---- Student forward (grad flows through lats → cross_proj → compress) ----
    s_out = generator(inputs_embeds=s_embeds, use_cache=False)
    s_log_probs = F.log_softmax(
        s_out.logits[0, -n_distill_tokens:, :].float(), dim=-1
    )                                                                # (N, V)

    return F.kl_div(
        s_log_probs.reshape(-1, s_log_probs.shape[-1]),
        t_probs.reshape(-1, t_probs.shape[-1]),
        reduction="sum",
    ) / n_distill_tokens


def batch_distillation_loss_multi(
    generator,
    generator_tokenizer,
    queries: list,
    sentences_list: list,              # list of B lists, each with K_i sentences
    latents_list: list,                # list of B tensors, each (K_i, H_8B)
    device: str = "cuda",
    n_distill_tokens: int = 1,
) -> torch.Tensor:
    """Multi-sentence distillation loss for a batch, averaged over samples."""
    losses = [
        compute_distillation_loss_multi(
            generator, generator_tokenizer, q, sents, lats, device, n_distill_tokens
        )
        for q, sents, lats in zip(queries, sentences_list, latents_list)
    ]
    return torch.stack(losses).mean()

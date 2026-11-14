"""Gemma SDPA with explicit grouped-query KV expansion.

Transformers 5.10.1 enables Torch's GQA flag for an unmasked SDPA call. Torch
2.10's CUDA memory-efficient backend does not implement that flag and requires
equal Q/K/V head counts. Expanding KV before SDPA preserves the native grouped
ordering and gradients while allowing that backend at the required fp32 dtype.
The expansion is linear in sequence length; no attention score matrix or math
fallback is introduced. Native Gemma mask construction and cache storage remain
unchanged. Registering this named interface does not replace global ``sdpa``.
"""
from __future__ import annotations

from types import SimpleNamespace
import warnings

import torch


BACKEND = "r1b_g_sdpa"


def gemma_sdpa(module, query, key, value, attention_mask, dropout=0.0,
               scaling=None, is_causal=None, **kwargs):
    """Native SDPA semantics with an explicit equal-head input boundary."""
    from transformers.integrations.sdpa_attention import repeat_kv

    q_heads, kv_heads = query.shape[1], key.shape[1]
    if kv_heads < 1 or q_heads % kv_heads or value.shape[1] != kv_heads:
        raise ValueError("Gemma SDPA needs Q heads divisible by equal K/V heads")
    groups = q_heads // kv_heads
    key, value = repeat_kv(key, groups), repeat_kv(value, groups)
    if kwargs.get("output_attentions", False):
        warnings.warn("G SDPA does not return attention matrices; use explicit readouts.", stacklevel=2)
    causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    causal = query.shape[2] > 1 and attention_mask is None and causal
    if torch.jit.is_tracing() and isinstance(causal, torch.Tensor):
        causal = causal.item()
    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=dropout,
        scale=scaling, is_causal=causal)
    return output.transpose(1, 2).contiguous(), None


def configure_attention(model):
    """Use the public attention and mask registries only for this G model."""
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import sdpa_mask

    AttentionInterface.register(BACKEND, gemma_sdpa)
    AttentionMaskInterface.register(BACKEND, sdpa_mask)
    model.set_attn_implementation(BACKEND)
    return model


def attention_shapes(config):
    """Distinct native Gemma local/global Q/KV/head-dimension triples."""
    c = config if isinstance(config, dict) else config.to_dict()
    heads = int(c["num_attention_heads"])
    result = set()
    for kind in c["layer_types"]:
        full = kind == "full_attention"
        dim = c.get("global_head_dim") if full else None
        dim = int(dim or c["head_dim"])
        kv = (c.get("num_global_key_value_heads") if full and c.get("attention_k_eq_v")
              else c["num_key_value_heads"])
        kv = int(kv)
        if heads < 1 or kv < 1 or heads % kv or dim < 1:
            raise ValueError("Invalid pinned Gemma attention head layout")
        result.add((heads, kv, dim))
    return sorted(result)


def preflight_cuda_attention(config, dtype, device):
    """Exercise actual head shapes on tiny tensors before downloading weights.

    Scientific long-context calls retain the efficient-only backend guard. This
    separate, bounded engineering probe compares the efficient result against a
    tiny math reference and checks odd-length masked/cache calls as well.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError("G efficient attention preflight requires CUDA")
    cases = []
    for heads, kv_heads, dim in attention_shapes(config):
        for query_length, key_length, masked in ((17, 17, False), (5, 17, True), (1, 17, True)):
            def tensor(h, n, shift):
                # No RNG-state change during this engineering check.
                data = torch.arange(h * n * dim, device=dev, dtype=torch.float32)
                return (((data + shift).remainder(31) - 15) / 64).reshape(1, h, n, dim).to(dtype)
            q, k, v = tensor(heads, query_length, 0), tensor(kv_heads, key_length, 7), tensor(kv_heads, key_length, 13)
            mask = None
            if masked:
                positions = torch.arange(query_length, device=dev) + key_length - query_length
                mask = (torch.arange(key_length, device=dev)[None] <= positions[:, None])[None, None]
            try:
                with torch.no_grad(), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    actual, _ = gemma_sdpa(SimpleNamespace(is_causal=True), q, k, v, mask, scaling=1.0)
                with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                    expected, _ = gemma_sdpa(SimpleNamespace(is_causal=True), q, k, v, mask, scaling=1.0)
                torch.cuda.synchronize(dev)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"G CUDA memory-efficient SDPA preflight failed for {dtype}, "
                    f"Q/KV heads={heads}/{kv_heads}, head_dim={dim}, "
                    f"Q/K lengths={query_length}/{key_length}. No math fallback was enabled. "
                    f"Check the pinned CUDA PyTorch environment: {exc}") from exc
            tol = 8 * torch.finfo(dtype).eps if dtype in (torch.float16, torch.bfloat16) else 1e-5
            if not torch.allclose(actual, expected, atol=tol, rtol=tol):
                raise RuntimeError("G CUDA memory-efficient SDPA preflight differs from its small math reference")
            cases.append({"query_heads": heads, "kv_heads": kv_heads, "head_dim": dim,
                          "query_tokens": query_length, "key_tokens": key_length,
                          "masked": masked, "max_abs_error": float((actual - expected).abs().max())})
    return {"backend": BACKEND, "cuda_kernel": "EFFICIENT_ATTENTION", "dtype": str(dtype),
            "kv_policy": "explicit repeat before SDPA; native KV cache unchanged", "cases": cases}

"""Offline, randomly initialized Gemma 4 fixtures; never downloads a model.

The actual Transformers decoder classes exercise PLE, unequal MLP widths, native
attention masks, KV sharing and softcapping. These fixtures are code tests, not
scientific observations. Importing this module does not import torch.
"""
from __future__ import annotations


def make_tokenizer():
    """Lossless tiny byte tokenizer with Gemma's turn delimiters (no real assets)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    specials = ["<pad>", "<eos>", "<bos>", "<unk>", "<|turn>", "<turn|>",
                "<|channel>", "<channel|>", "<|think|>"]
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {s: i for i, s in enumerate(specials + alphabet)}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                                 eos_token="<eos>", bos_token="<bos>", unk_token="<unk>",
                                 additional_special_tokens=specials[4:], padding_side="left")
    tok.chat_template = (
        "{{ bos_token }}"
        "{% for m in messages %}"
        "{{ '<|turn>' + ('model' if m['role'] == 'assistant' else m['role']) + '\n' }}"
        "{{ m['content'] }}{{ '<turn|>\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|turn>model\n' }}{% endif %}"
    )
    return tok


def _text_kwargs(vocab_size):
    return dict(vocab_size=vocab_size, hidden_size=32, intermediate_size=48,
                num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                num_global_key_value_heads=2, head_dim=8, global_head_dim=8,
                max_position_embeddings=256, sliding_window=8,
                layer_types=["sliding_attention", "full_attention"] * 2,
                attention_k_eq_v=False, num_kv_shared_layers=2,
                final_logit_softcapping=30.0, pad_token_id=0, eos_token_id=[1, 5],
                bos_token_id=2, attention_dropout=0.0, use_bidirectional_attention=None)


def make_mock(kind="e2b", seed=1701):
    """Return (tokenizer, tiny random native HF Gemma LM), deterministically.

    ``kind`` accepts the registry aliases as well as e2b/e4b/12b/31b. QAT
    aliases select the same *architecture*: quantization itself has a separate
    synthetic compressed-tensors test and is never pretended here.
    """
    import torch
    from transformers import (Gemma4ForCausalLM, Gemma4TextConfig,
                              Gemma4UnifiedForCausalLM, Gemma4UnifiedTextConfig)

    kind = kind.lower().replace("gemma4-", "").replace("-qat", "")
    if kind not in {"e2b", "e4b", "12b", "31b"}:
        raise ValueError(f"unknown mock target: {kind}")
    tok = make_tokenizer()
    kw = _text_kwargs(len(tok))
    if kind == "12b":
        cfg = Gemma4UnifiedTextConfig(**kw)
        cls = Gemma4UnifiedForCausalLM
    else:
        kw.update(hidden_size_per_layer_input=8 if kind in {"e2b", "e4b"} else 0,
                  vocab_size_per_layer_input=len(tok) if kind in {"e2b", "e4b"} else 0,
                  use_double_wide_mlp=(kind == "e2b"))
        cfg = Gemma4TextConfig(**kw)
        cls = Gemma4ForCausalLM
    cfg._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = cls(cfg).float().eval()
    model.requires_grad_(False)
    model._r1b_g_mock = True
    model._r1b_g_name = "gemma4-" + kind
    return tok, model


def make_assistant(target, seed=1702):
    """Random native assistant whose KV dimensions match ``target`` exactly."""
    import torch
    from transformers import Gemma4AssistantConfig, Gemma4AssistantForCausalLM, Gemma4TextConfig

    t = target.config.get_text_config()
    kw = _text_kwargs(t.vocab_size)
    kw.update(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
              layer_types=["sliding_attention", "full_attention"], num_kv_shared_layers=2,
              hidden_size_per_layer_input=0, vocab_size_per_layer_input=0)
    cfg = Gemma4AssistantConfig(text_config=Gemma4TextConfig(**kw),
                               backbone_hidden_size=t.hidden_size,
                               use_ordered_embeddings=False)
    cfg._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Gemma4AssistantForCausalLM(cfg).float().eval()
    model.requires_grad_(False)
    model._r1b_g_mock = True
    return model

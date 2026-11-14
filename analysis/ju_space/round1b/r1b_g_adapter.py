"""Gemma 4 adapter for part G; no changes or monkeypatches to the shared core.

Real models are CUDA-only and require a complete, rechecked manifest.  The core's
MOCK loader deliberately bypasses registrations, so G calls ``load_model`` here
instead.  Its tiny random models never need a network connection or model files.
``register_models`` is the sole mutation of core registries and is explicit.
"""
from __future__ import annotations

import bisect
import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path


PINNED_VERSIONS = {"torch": "2.10.0", "transformers": "5.10.1", "compressed-tensors": "0.15.0",
                   "accelerate": "1.12.0"}


def _spec(name, freeze=None):
    from r1b_g_models import get_spec
    spec = get_spec(name)
    override = ((freeze or {}).get("models") or {}).get(name, {})
    if override:
        # Identity and scientific precision cannot be changed through a run file.
        for key in ("repo", "family", "role", "dtype", "quantized"):
            if key in override and override[key] != spec[key]:
                raise ValueError(f"{name}: freeze changes model identity/precision ({key})")
        if "sha256" in override and not set(spec["sha256"]).issubset(override["sha256"]):
            raise ValueError(f"{name}: freeze removes reviewed model/tokenizer files from the manifest")
        spec.update({key: copy.deepcopy(override[key]) for key in ("revision", "sha256") if key in override})
    return spec


def check_versions(quantized=False):
    packages = ("torch", "transformers", "accelerate") + (("compressed-tensors",) if quantized else ())
    found = {package: version(package) for package in packages}
    for package, actual in found.items():
        if actual.split("+", 1)[0] != PINNED_VERSIONS[package]:
            raise RuntimeError(f"G requires {package}=={PINNED_VERSIONS[package]}; found {actual}")
    return found


def _file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_snapshot(directory, pins, *, require_weights=True):
    """Rehash every pinned file, including cached snapshots; never trust a marker.

    All weights selected by a safetensors index must be pinned.  Extra model
    weights/configs cannot silently become an alternative load path.
    """
    directory = Path(directory).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", str(pins.get("revision", ""))):
        raise ValueError("G requires an immutable 40-character model revision")
    hashes = pins.get("sha256") or {}
    if "config.json" not in hashes or not hashes:
        raise ValueError("G requires a complete sha256 manifest including config.json")
    if require_weights and not any(path.endswith(".safetensors") for path in hashes):
        raise ValueError("G manifest has no model weights")
    for rel, expected in hashes.items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected)):
            raise ValueError(f"invalid sha256 for {rel}")
        p = (directory / rel).resolve()
        if Path(rel).is_absolute() or ".." in Path(rel).parts or not p.is_relative_to(directory):
            raise ValueError(f"unsafe manifest path: {rel}")
        if not p.is_file() or _file_hash(p) != expected:
            raise RuntimeError(f"G pinned-file verification failed: {rel}")
    for index in directory.glob("*.safetensors.index.json"):
        if index.name not in hashes:
            raise ValueError(f"unpinned weight index: {index.name}")
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if not shards.issubset(hashes):
            raise ValueError(f"unpinned weight shards: {sorted(shards - hashes.keys())}")
    if require_weights:
        for pattern in ("*.safetensors", "pytorch_model*.bin", "*.safetensors.index.json"):
            for p in directory.glob(pattern):
                if p.name not in hashes:
                    raise ValueError(f"unexpected model file: {p.name}")
    auto_files = ("config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
                  "tokenizer.model", "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
                  "chat_template.jinja")
    for rel in auto_files:
        if (directory / rel).is_file() and rel not in hashes:
            raise ValueError(f"unpinned automatically loaded file: {rel}")
    return {"revision": pins["revision"], "sha256": dict(hashes), "ok": True}


def ensure_snapshot(name, root, freeze=None, download=False):
    spec = _spec(name, freeze)
    path = Path(root) / "models" / name
    if download:
        from huggingface_hub import snapshot_download
        # Pass no token: HF's environment/secret mechanism handles credentials.
        snapshot_download(repo_id=spec["repo"], revision=spec["revision"],
                          local_dir=str(path), allow_patterns=list(spec["sha256"]))
    validate_snapshot(path, spec)
    return path


def text_config(model):
    cfg = model.config
    return cfg.get_text_config() if hasattr(cfg, "get_text_config") else getattr(cfg, "text_config", cfg)


def text_model(model):
    for path in ("model.language_model", "model", "language_model.model"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if hasattr(obj, "layers") and hasattr(obj, "norm"):
                return obj
        except AttributeError:
            pass
    raise TypeError("Gemma text decoder (layers and final norm) not found")


def layers_of(model):
    return text_model(model).layers


@dataclass(frozen=True)
class NeuronLayout:
    """Ragged MLP coordinates; flat IDs use prefix sums, never layer * width."""
    widths: tuple[int, ...]

    def __post_init__(self):
        if not self.widths or any(not isinstance(n, int) or n < 1 for n in self.widths):
            raise ValueError("neuron widths must be positive integers")

    @classmethod
    def from_model(cls, model):
        return cls(tuple(int(layer.mlp.down_proj.in_features) for layer in layers_of(model)))

    @property
    def offsets(self):
        out = [0]
        for width in self.widths:
            out.append(out[-1] + width)
        return tuple(out)

    @property
    def total(self):
        return sum(self.widths)

    def encode(self, layer, neuron):
        if not isinstance(layer, int) or not 0 <= layer < len(self.widths):
            raise IndexError(f"invalid layer {layer}")
        if not isinstance(neuron, int) or not 0 <= neuron < self.widths[layer]:
            raise IndexError(f"invalid neuron {layer}:{neuron}")
        return self.offsets[layer] + neuron

    def decode(self, flat):
        if not isinstance(flat, int) or not 0 <= flat < self.total:
            raise IndexError(f"invalid flat neuron {flat}")
        layer = bisect.bisect_right(self.offsets, flat) - 1
        return layer, flat - self.offsets[layer]

    def as_dict(self):
        return {"widths": list(self.widths), "offsets": list(self.offsets), "n_neurons": self.total,
                "indexing": "zero-based layer, neuron; flat = offsets[layer] + neuron"}


def attention_layout(model):
    cfg = text_config(model)
    kinds = tuple(cfg.layer_types)
    if len(kinds) != len(layers_of(model)) or set(kinds) - {"full_attention", "sliding_attention"}:
        raise ValueError("unsupported or inconsistent Gemma attention layout")
    return {"layer_types": list(kinds), "sliding_window": int(cfg.sliding_window),
            "max_position_embeddings": int(cfg.max_position_embeddings),
            "num_kv_shared_layers": int(getattr(cfg, "num_kv_shared_layers", 0))}


def chat_text(tok, user, system=None):
    messages = ([{"role": "system", "content": system}] if system is not None else [])
    messages.append({"role": "user", "content": user})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def chat_ids(tok, user, system=None):
    # The template already includes BOS; adding special tokens again changes positions.
    return tok(chat_text(tok, user, system), add_special_tokens=False)["input_ids"]


def user_start(tok, user="", system=None):
    """First token overlapping actual user content, including a boundary BPE token."""
    marker = "R1B_G_BOUNDARY_7ad29f0b"
    template = chat_text(tok, marker, system)
    if template.count(marker) != 1:
        raise ValueError("chat template does not preserve user text")
    prefix = template[:template.index(marker)]
    rendered = chat_text(tok, user if user.strip() else marker, system)
    if not rendered.startswith(prefix):
        raise ValueError("chat template has a content-dependent user prefix")
    if not getattr(tok, "is_fast", False):
        raise ValueError("exact Gemma user boundary requires a fast tokenizer")
    encoded = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    for i, (begin, end) in enumerate(encoded["offset_mapping"]):
        if end > len(prefix):
            return i
    raise ValueError("user text has no tokens")


def stop_tokens(tok, model):
    e = getattr(model.generation_config, "eos_token_id", None)
    eos = set(e if isinstance(e, (list, tuple)) else [e])
    eos.add(getattr(tok, "eos_token_id", None))
    vocab = tok.get_vocab()
    if "<turn|>" in vocab:
        eos.add(vocab["<turn|>"])
    eos = {int(x) for x in eos if x is not None and int(x) >= 0}
    if not eos:
        raise ValueError("Gemma has no valid stop token")
    stop = eos | ({int(tok.pad_token_id)} if tok.pad_token_id is not None else set())
    return sorted(eos), stop


def unembed(model, hidden, *, normalized=False):
    """Raw block output -> final RMSNorm -> tied head -> Gemma logit softcap."""
    import torch
    if getattr(model, "_r1b_g_forward_only", False):
        _forward_only_guard(model, ())
    head = model.get_output_embeddings()
    # Shared Hooks.capture deliberately stores fp32 (often on CPU), including
    # when the native decoder executes in fp16. Restore the execution boundary.
    native_hidden = hidden.to(device=head.weight.device, dtype=head.weight.dtype)
    h = native_hidden if normalized else text_model(model).norm(native_hidden)
    logits = head(h)
    softcap = getattr(text_config(model), "final_logit_softcapping", None)
    return softcap * torch.tanh(logits / softcap) if softcap is not None else logits


def require_gradients(model, measurement):
    if getattr(model, "_r1b_g_forward_only", False):
        raise ValueError(f"{measurement}: gradient measurements are excluded for QAT targets")


def _forward_only_guard(module, args):
    import torch
    if torch.is_grad_enabled():
        raise RuntimeError("G QAT targets are forward-only; call within torch.no_grad() or inference_mode()")


def _cast_parameters(model, dtype):
    """Enforce execution precision without reducing native fp32 RoPE buffers."""
    import torch
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.is_floating_point() and parameter.dtype != dtype:
                parameter.data = parameter.data.to(dtype)
    model.config.dtype = dtype


def _tag_model(model, name, spec, *, mock=False):
    model._r1b_g_name = name
    model._r1b_g_forward_only = bool(spec.get("forward_only", spec.get("quantized", False)))
    model._r1b_g_mock = bool(mock)
    model._r1b_g_pins = {k: copy.deepcopy(spec[k]) for k in ("repo", "revision", "sha256") if k in spec}
    if model._r1b_g_forward_only:
        model.register_forward_pre_hook(_forward_only_guard)
        # A stopped-state read commonly calls model.model directly. It must not
        # bypass the forward-only rule attached to the causal-LM wrapper.
        decoder = text_model(model)
        decoder._r1b_g_forward_only = True
        decoder.register_forward_pre_hook(_forward_only_guard)
    model.eval().requires_grad_(False)
    return model


def _load_real(name, root, device, freeze=None, download=False):
    import torch
    from transformers import AutoTokenizer
    spec = _spec(name, freeze)
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Real Gemma models are CUDA-only; CPU tests must use mock=True")
    check_versions(spec["quantized"])
    path = ensure_snapshot(name, root, freeze, download=download)
    raw = json.loads((path / "config.json").read_text())
    if raw["model_type"] != spec["family"]:
        raise ValueError("pinned config does not match the Gemma architecture")
    kwargs = dict(local_files_only=True, trust_remote_code=False, attn_implementation="sdpa",
                  dtype=getattr(torch, spec["dtype"]), device_map={"": str(device)}, output_loading_info=True)
    if spec["role"] == "assistant":
        from transformers import Gemma4AssistantForCausalLM
        cls = Gemma4AssistantForCausalLM
        # Assistants may omit their tokenizer; use only the pinned target's files.
        token_path = path
        if not (path / "tokenizer.json").is_file():
            token_path = ensure_snapshot(spec["target"], root, freeze, download=download)
    else:
        if spec["family"] == "gemma4_unified":
            from transformers import Gemma4UnifiedForCausalLM, Gemma4UnifiedTextConfig
            cls, cfg_cls = Gemma4UnifiedForCausalLM, Gemma4UnifiedTextConfig
        else:
            from transformers import Gemma4ForCausalLM, Gemma4TextConfig
            cls, cfg_cls = Gemma4ForCausalLM, Gemma4TextConfig
        cfg = cfg_cls.from_dict(raw["text_config"])
        # Both full checkpoints store the decoder below model.language_model.
        kwargs.update(config=cfg, key_mapping={r"^model\.language_model\.": "model."})
        token_path = path
        if spec["quantized"]:
            from transformers import CompressedTensorsConfig
            q = copy.deepcopy(raw["quantization_config"])
            # CPU/GPU decompression retains the quantized reconstruction, not original weights.
            q["run_compressed"] = False
            cfg.quantization_config = copy.deepcopy(q)
            kwargs["quantization_config"] = CompressedTensorsConfig.from_dict(q)
    tok = AutoTokenizer.from_pretrained(str(token_path), local_files_only=True, trust_remote_code=False)
    model, info = cls.from_pretrained(str(path), **kwargs)
    missing = set(info.get("missing_keys", ()))
    if missing:
        raise RuntimeError(f"G text-only checkpoint conversion left missing weights: {sorted(missing)}")
    if info.get("mismatched_keys") or info.get("error_msgs"):
        raise RuntimeError("G checkpoint conversion has mismatched weights or loading errors")
    # CT 0.15 can preserve checkpoint dtypes on ignored/unquantized parameters
    # despite from_pretrained(dtype=...). Enforce the declared execution precision
    # on parameters, while retaining the model's native fp32 RoPE buffers.
    execution_dtype = getattr(torch, spec["dtype"])
    _cast_parameters(model, execution_dtype)
    if spec["role"] == "target":
        layout = NeuronLayout.from_model(model)
        if tuple(spec["widths"]) != layout.widths or int(text_config(model).hidden_size) != spec["hidden"]:
            raise ValueError("loaded decoder shape differs from the reviewed model config")
        attention_layout(model)
    tok.padding_side = "left"
    return tok, _tag_model(model, name, spec)


def register_models(freeze=None, *, download=False):
    """Register loaders without changing any core functions or existing entries."""
    import r1b_core as core
    from r1b_g_models import MODEL_SPECS
    names = []
    for name in MODEL_SPECS:
        spec = _spec(name, freeze)

        def loader(root, device, mock=False, name=name):
            return load_model(name, root=root, device=device, mock=mock, freeze=freeze, download=download)

        core.register_loader(name, loader, layers=spec["layers"], hidden=spec["hidden"], repo=spec["repo"],
                             revision=spec["revision"], sha256=spec["sha256"])
        names.append(name)
    return names


def load_model(name, root=None, device=None, mock=None, freeze=None, download=False):
    """G loading entry point; unlike the core, registered mocks are Gemma-shaped."""
    import torch
    import r1b_core as core
    core.numerics()
    spec = _spec(name, freeze)
    mock = os.environ.get("MOCK") == "1" if mock is None else bool(mock)
    device = device or ("cpu" if mock else "cuda")
    if mock:
        from r1b_g_mock import make_mock
        kind = next(k for k in ("e2b", "e4b", "12b", "31b") if k in spec["repo"].lower())
        tok, model = make_mock(kind=kind)
        if spec["role"] == "assistant":
            from r1b_g_mock import make_assistant
            model = make_assistant(model)
        _cast_parameters(model, getattr(torch, spec["dtype"]))
        tok.padding_side = "left"
        return tok, _tag_model(model.to(device), name, spec, mock=True)
    root = root or os.environ.get("R1B_ROOT", "/content/root")
    return _load_real(name, root, device, freeze=freeze, download=download)


@dataclass
class AssistantState:
    hidden: object
    shared_kv_states: dict
    position_ids: object
    attention_mask: object


class AssistantBridge:
    """Native Gemma MTP recursion, with a fixed target context and RoPE anchor.

    ``prefill`` returns state h(t), KV(<=t), and target logits for seed x(t+1).
    ``step(seed, state)`` predicts x(t+2), returning the projected hidden state for
    the next draft step.  The target KV and anchor do not advance during a draft.
    """
    def __init__(self, target, assistant):
        from r1b_g_models import get_spec
        target_name = getattr(target, "_r1b_g_name", None)
        assistant_name = getattr(assistant, "_r1b_g_name", None)
        if target_name is None or assistant_name is None:
            raise ValueError("MTP bridge needs identified G target and assistant models")
        ts, ds = get_spec(target_name), get_spec(assistant_name)
        if ts["role"] != "target" or ds["role"] != "assistant" or ds["target"] != target_name:
            raise ValueError("Gemma target/assistant pair does not match")
        if ts["quantized"] or ds["quantized"]:
            raise ValueError("G MTP calibration uses only matching non-quantized originals")
        if int(assistant.config.backbone_hidden_size) != int(text_config(target).hidden_size):
            raise ValueError("assistant backbone dimension mismatch")
        if int(text_config(assistant).vocab_size) != int(text_config(target).vocab_size):
            raise ValueError("assistant vocabulary dimension mismatch")
        self.target, self.assistant = target, assistant

    def prefill(self, input_ids, attention_mask=None):
        import torch
        mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        if mask.shape != input_ids.shape or (mask[:, -1] != 1).any():
            raise ValueError("MTP prefill requires nonempty, left-padded sequences")
        pos = (mask.long().cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            output = self.target(input_ids=input_ids, attention_mask=mask, position_ids=pos, use_cache=True,
                                 output_hidden_states=True, return_shared_kv_states=True, logits_to_keep=0)
        if not output.shared_kv_states or not output.hidden_states:
            raise RuntimeError("target did not expose hidden states and shared KV states")
        state = AssistantState(output.hidden_states[-1][:, -1:], output.shared_kv_states,
                               mask.long().sum(-1, keepdim=True), mask)
        return state, output.logits[:, -1:]

    def step(self, seed, state):
        import torch
        if seed.ndim == 1:
            seed = seed[:, None]
        if seed.shape != state.hidden.shape[:2] or seed.shape[1] != 1:
            raise ValueError("MTP expects one seed per state")
        with torch.no_grad():
            embedded = self.target.get_input_embeddings()(seed)
            output = self.assistant(inputs_embeds=torch.cat([embedded, state.hidden], dim=-1),
                                    position_ids=state.position_ids, attention_mask=state.attention_mask,
                                    shared_kv_states=state.shared_kv_states, use_cache=False)
        next_state = AssistantState(output.last_hidden_state, state.shared_kv_states,
                                    state.position_ids, state.attention_mask)
        return output.logits, next_state


def selftest():
    import torch
    import r1b_core as core
    from r1b_g_models import MODEL_SPECS
    for name, spec in MODEL_SPECS.items():
        if spec["role"] != "target":
            continue
        tok, model = load_model(name, mock=True)
        ids = torch.tensor([chat_ids(tok, "Name a tree.")])
        layout = NeuronLayout.from_model(model)
        assert all(layout.encode(*layout.decode(i)) == i for i in range(layout.total))
        with torch.no_grad():
            base = model(ids, logits_to_keep=0).logits
            with core.Hooks(model).neuron(0, [0], "scale", 1):
                same = model(ids, logits_to_keep=0).logits
            assert torch.equal(base, same)
        assert 0 < user_start(tok, "Name a tree.") < ids.shape[1]
        assert tok.pad_token_id in stop_tokens(tok, model)[1]
    print("Gemma adapter selftest: PASS (random CPU models only)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["selftest"])
    parser.parse_args()
    selftest()

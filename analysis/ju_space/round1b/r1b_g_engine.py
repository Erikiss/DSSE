"""Independent Gemma measurement engine: native forwards, ragged neurons and CRN.

Only model-independent shared numerical definitions are imported. No Qwen model,
tokens, result files or runner are required. Returned tensors live on the CPU.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
from pathlib import Path

import r1b_g_adapter as A


def settings(ctx):
    return dict((getattr(ctx, "freeze", {}) or {}).get("g", {}))


def text_rows(ctx, key, *, minimum=1, limit=None):
    """Raw-text inputs, with explicit identity; no model-specific token bank."""
    import r1b_core as core
    data = ctx.data()
    values = data.get(key)
    if not values:
        raise core.NotRun(f"G input missing: {key}")
    if not isinstance(values, list):
        raise ValueError(f"{key} must be a list of texts or {{id,text}} records")
    rows = []
    for i, value in enumerate(values):
        row = {"id": f"{key}:{i}", "text": value} if isinstance(value, str) else dict(value)
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"{key}[{i}] has no text")
        row.setdefault("id", f"{key}:{i}")
        rows.append(row)
    if len(rows) < minimum:
        raise core.NotRun(f"{key}: need {minimum} rows, received {len(rows)}")
    return rows if limit is None else rows[:limit]


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _manager(setup):
    return contextlib.nullcontext() if setup is None else setup() if callable(setup) else setup


def _kernel(device, length):
    # The inherited long-context SG9 condition prohibits a silent math fallback.
    # Short Jacobian windows choose their separate differentiable kernel.
    import r1b_core as core
    return core.sdpa_efficient(device) if length > 2048 else contextlib.nullcontext()


class ModelEngine:
    def __init__(self, ctx):
        import torch
        self.ctx = ctx
        self.tok, self.model = ctx.model()
        self.device = next(self.model.parameters()).device
        self.layout = A.NeuronLayout.from_model(self.model)
        self.layers = A.layers_of(self.model)
        self.cfg = A.text_config(self.model)
        self.wnorm = {}
        self.logit_chunk = int(settings(ctx).get("logit_chunk", 32))
        if self.logit_chunk < 1:
            raise ValueError("logit_chunk must be positive")

    def encode(self, text, chat=True):
        return A.chat_ids(self.tok, text) if chat else self.tok(text, add_special_tokens=False)["input_ids"]

    def _sequence(self, value):
        ids = self.encode(value) if isinstance(value, str) else list(map(int, value))
        if not ids or len(ids) > int(self.cfg.max_position_embeddings):
            raise ValueError("empty sequence or context exceeds native Gemma limit")
        if min(ids) < 0 or max(ids) >= int(self.cfg.vocab_size):
            raise ValueError("token ID outside the target vocabulary")
        return ids

    def _weights(self, layer):
        if layer not in self.wnorm:
            self.wnorm[layer] = self.layers[layer].mlp.down_proj.weight.detach().float().norm(dim=0)
        return self.wnorm[layer]

    def _population(self, populations, all_if_none=False):
        if populations is None:
            populations = {"all": list(range(self.layout.total))} if all_if_none else {}
        sets = {str(k): sorted(set(map(int, v))) for k, v in populations.items()}
        selected = sorted(set(j for v in sets.values() for j in v))
        per_layer = {}
        for column, flat in enumerate(selected):
            layer, neuron = self.layout.decode(flat)
            per_layer.setdefault(layer, []).append((column, neuron))
        return sets, selected, per_layer

    @contextlib.contextmanager
    def intervene(self, neurons, op="scale", value=1.0, positions=None):
        """Native down-projection pre-hooks; optional absolute positions per call.

        This context uses complete-prefix forwards, so absolute position masks
        have identical semantics for generation and teacher forcing.
        """
        import torch
        handles = []
        pairs = {}
        neurons = list(neurons)
        for order, flat in enumerate(neurons):
            layer, neuron = self.layout.decode(int(flat))
            pairs.setdefault(layer, []).append((order, neuron))
        try:
            for layer, pairs_ in pairs.items():
                order, ix = zip(*pairs_)
                def hook(module, args, ix=tuple(ix), order=tuple(order)):
                    x = args[0]
                    out = x.clone()
                    chosen = out[..., list(ix)]
                    val = torch.as_tensor(value, device=x.device, dtype=x.dtype)
                    if val.ndim == 1 and val.numel() == len(neurons):
                        val = val[list(order)]
                    if op == "scale":
                        changed = chosen * val
                    elif op == "add":
                        changed = chosen + val
                    elif op in {"zero", "mean", "clamp"}:
                        changed = torch.zeros_like(chosen) if op == "zero" else val.expand_as(chosen)
                    else:
                        raise ValueError(f"unknown neuron operation {op}")
                    if positions is not None:
                        mask = torch.zeros(x.shape[1], device=x.device, dtype=torch.bool)
                        ixpos = torch.as_tensor(positions, device=x.device, dtype=torch.long)
                        ixpos = ixpos[(ixpos >= 0) & (ixpos < x.shape[1])]
                        mask[ixpos] = True
                        changed = torch.where(mask[None, :, None], changed, chosen)
                    out[..., list(ix)] = changed
                    return (out,) + args[1:]
                handles.append(self.layers[layer].mlp.down_proj.register_forward_pre_hook(hook))
            yield self
        finally:
            for handle in handles:
                handle.remove()

    @contextlib.contextmanager
    def patch_state(self, block, state, position, *, add=False):
        import torch
        if not 0 <= int(block) < len(self.layers):
            raise IndexError("invalid source block")
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if not -hidden.shape[1] <= position < hidden.shape[1]:
                raise IndexError("patch position outside current prefix")
            modified = hidden.clone()
            value = torch.as_tensor(state, device=hidden.device, dtype=hidden.dtype)
            modified[:, position] = hidden[:, position] + value if add else value
            return (modified,) + output[1:] if isinstance(output, tuple) else modified
        handle = self.layers[int(block)].register_forward_hook(hook)
        try:
            yield self
        finally:
            handle.remove()

    def _collect(self, seq, populations=None, blocks=(), positions=None, means=False, setup=None, score=True):
        import torch
        sets, selected, per_layer = self._population(populations, all_if_none=means)
        row = {"ids": list(seq), "neuron_ids": selected, "hidden": {}, "channels": {}}
        pos = list(range(len(seq))) if positions is None else list(map(int, positions))
        if any(i < 0 or i >= len(seq) for i in pos) or (means and not pos):
            raise ValueError("invalid/empty feature positions")
        shape = (len(selected),) if means else (len(pos), len(selected))
        row["z"], row["cett"] = torch.zeros(shape), torch.zeros(shape)
        handles = []
        with _manager(setup), _kernel(self.device, len(seq)), torch.no_grad():
            try:
                for layer, mapping in per_layer.items():
                    columns, neuron_ix = zip(*mapping)
                    def cap(module, args, output, layer=layer, columns=columns, neuron_ix=neuron_ix):
                        z = args[0][0, pos][:, list(neuron_ix)].float()
                        den = output[0, pos].float().norm(dim=-1).clamp(min=1e-9)
                        c = z.abs() * self._weights(layer)[list(neuron_ix)] / den[:, None]
                        if means:
                            z, c = z.mean(0), c.mean(0)
                        row["z"][..., list(columns)] = z.detach().cpu()
                        row["cett"][..., list(columns)] = c.detach().cpu()
                    handles.append(self.layers[layer].mlp.down_proj.register_forward_hook(cap))
                for block in blocks:
                    def cap_hidden(module, args, output, block=int(block)):
                        h = output[0] if isinstance(output, tuple) else output
                        row["hidden"][block] = h[0, pos].detach().float().cpu()
                    handles.append(self.layers[int(block)].register_forward_hook(cap_hidden))
                ids = torch.tensor([seq], device=self.device)
                outputs = A.text_model(self.model)(input_ids=ids, use_cache=False)
                if score:
                    normalized = outputs.last_hidden_state[0]
                    lp, entropy, top1 = [], [], []
                    for start in range(0, max(len(seq) - 1, 0), self.logit_chunk):
                        end = min(start + self.logit_chunk, len(seq) - 1)
                        logits = A.unembed(self.model, normalized[start:end], normalized=True).float()
                        logp = logits.log_softmax(-1)
                        truth = ids[0, start + 1:end + 1]
                        lp.append(logp.gather(-1, truth[:, None])[:, 0].cpu())
                        entropy.append(-(logp.exp() * logp).sum(-1).cpu())
                        top1.append(logp.argmax(-1).cpu())
                    row["next_logprobs"] = torch.cat(lp) if lp else torch.empty(0)
                    row["entropy"] = torch.cat(entropy) if entropy else torch.empty(0)
                    row["top1"] = torch.cat(top1) if top1 else torch.empty(0, dtype=torch.long)
            finally:
                for handle in handles:
                    handle.remove()
        col = {n: i for i, n in enumerate(selected)}
        for name, values in sets.items():
            ix = [col[n] for n in values]
            row["channels"][name] = {k: row[k][..., ix].mean(-1) if ix else torch.zeros(row[k].shape[:-1])
                                      for k in ("z", "cett")}
        row["positions"] = pos
        return row

    def teacher_force(self, seqs, populations=None, blocks=(), positions=None, setup=None):
        return {"rows": [self._collect(self._sequence(s), populations, blocks, positions, setup=setup) for s in seqs],
                "layout": self.layout.as_dict()}

    def features(self, seqs, spans, populations=None, setup=None):
        import torch
        if len(seqs) != len(spans):
            raise ValueError("one feature span is required per sequence")
        rows = [self._collect(self._sequence(s), populations, positions=p, means=True, setup=setup, score=False)
                for s, p in zip(seqs, spans)]
        if not rows:
            raise ValueError("no feature rows")
        return {"z": torch.stack([r["z"] for r in rows]), "cett": torch.stack([r["cett"] for r in rows]),
                "neuron_ids": rows[0]["neuron_ids"], "layout": self.layout.as_dict()}

    def hidden(self, seqs, blocks):
        return [self._collect(self._sequence(s), blocks=blocks, score=False)["hidden"] for s in seqs]

    def next_logits(self, prefix, setup=None):
        import torch
        seq = self._sequence(prefix)
        with _manager(setup), _kernel(self.device, len(seq)), torch.no_grad():
            ids = torch.tensor([seq], device=self.device)
            # Keep every decoder position (including KV-sharing layers) while
            # projecting only the last one through the large vocabulary head.
            h = A.text_model(self.model)(input_ids=ids, use_cache=False).last_hidden_state[:, -1:]
            return A.unembed(self.model, h, normalized=True)[0, -1].float()

    @staticmethod
    def choose(logits, generator=None, temperature=0.0, topk=None, topp=None):
        import torch
        if temperature < 0:
            raise ValueError("temperature must be nonnegative")
        s = logits.float().clone()
        if topk is not None:
            if int(topk) < 1:
                raise ValueError("topk must be positive")
            s[s < s.topk(min(int(topk), len(s))).values[-1]] = -torch.inf
        if topp is not None:
            if not 0 < float(topp) <= 1:
                raise ValueError("topp must be in (0,1]")
            vals, ix = s.sort(descending=True)
            remove = vals.softmax(-1).cumsum(-1) - vals.softmax(-1) > float(topp)
            s[ix[remove]] = -torch.inf
        if not torch.isfinite(s).any():
            raise ValueError("sampling mask removed every token")
        if temperature > 0:
            if generator is None:
                raise ValueError("stochastic generation requires an explicit keyed generator")
            u = torch.rand(s.shape, device=s.device, generator=generator).clamp(1e-7, 1 - 1e-7)
            s = s / temperature - torch.log(-torch.log(u))
        return int(s.argmax())

    def decode(self, prompts, keys=None, max_new=40, temperature=0.7, setup=None, topk=None, topp=None):
        import torch
        if max_new < 0 or (keys is not None and len(keys) != len(prompts)):
            raise ValueError("invalid generation length/keys")
        if temperature > 0 and keys is None:
            keys = [self.ctx.key("decode", content_hash(p), i) for i, p in enumerate(prompts)]
        _, stop = A.stop_tokens(self.tok, self.model)
        rows = []
        for i, prompt in enumerate(prompts):
            prefix = self._sequence(prompt)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(keys[i]) if keys is not None else 0)
            tokens, probabilities = [], []
            for k in range(int(max_new)):
                if len(prefix) + len(tokens) >= int(self.cfg.max_position_embeddings):
                    raise ValueError("generation would exceed native context limit")
                logits = self.next_logits(prefix + tokens, setup=setup)
                token = self.choose(logits, generator, temperature, topk, topp)
                if token in stop:
                    break
                tokens.append(token)
                probabilities.append(float(logits.log_softmax(-1)[token]))
            rows.append({"ids": tokens, "text": self.tok.decode(tokens, skip_special_tokens=True),
                         "logprobs": probabilities, "prompt_ids": prefix})
        return rows

    def damage(self, reftexts, setup=None):
        """Shared SO-DMG KL definition on target-tokenized raw reference texts."""
        import r1b_core as core
        from r1b_g_l_erasure import Damage
        seqs = [self.encode(t, chat=False) if isinstance(t, str) else self._sequence(t) for t in reftexts]
        refs = core.RefSet([(ids, list(range(len(ids) - 1))) for ids in seqs if len(ids) > 1], "G-REF")
        if not len(refs):
            raise ValueError("empty damage reference set")
        return Damage(self.model, refs, chunk=self.logit_chunk)(setup)

    def calibrate(self, reftexts, neurons, start=2.0, target=0.05, op="scale"):
        import r1b_core as core
        from r1b_g_l_erasure import Damage
        seqs = [self.encode(t, chat=False) for t in reftexts]
        refs = core.RefSet([(s, list(range(len(s) - 1))) for s in seqs if len(s) > 1], "G-REF")
        dm = Damage(self.model, refs, chunk=self.logit_chunk)
        return core.calibrate(lambda alpha: dm(lambda: self.intervene(neurons, op, alpha))["damage"],
                              start, target, form="gain" if op == "scale" else "push")

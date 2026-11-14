"""SO-TE (part K): central-difference transfer, frozen-QK attention (the attention probabilities of every layer stored on the
unperturbed pass and replayed), VJP edge attribution per head, and the FD-against-autograd check (u . FD(v) against (J^T u) . v).
Forward and backward sensitivities only (main §2.5: not reconstructions)."""
import contextlib
import r1b_core as RC
from r1b_k_base import T_

_ST = dict(mode=None, probs={}, keep=False, heads={})


def _attn(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    """Explicit attention: 'record' stores the probabilities per layer, 'replay' uses the stored ones with the current values;
    'grad' keeps the per-head outputs (requires grad) and probabilities for the VJP attribution."""
    torch = T_()
    rep = query.shape[1] // key.shape[1]
    k = key.repeat_interleave(rep, 1); v = value.repeat_interleave(rep, 1); li = module.layer_idx
    if _ST["mode"] == "replay" and li in _ST["probs"]:
        p = _ST["probs"][li].to(v.dtype)
    else:
        s = (query.float() @ k.float().transpose(-1, -2)) * scaling
        T, S = s.shape[-2], s.shape[-1]
        if attention_mask is None:
            causal = torch.ones(T, S, dtype=torch.bool, device=s.device).tril(S - T)
            s = s.masked_fill(~causal, float("-inf"))
        elif attention_mask.dtype == torch.bool:
            s = s.masked_fill(~attention_mask[..., :S], float("-inf"))
        else:
            s = s + attention_mask[..., :S].float()
        p = s.softmax(-1).to(v.dtype)
        if _ST["mode"] == "record":
            _ST["probs"][li] = p.detach()
    o = p @ v                                                     # [B, H, T, d]
    if _ST["mode"] == "grad":
        o.retain_grad(); _ST["heads"][li] = (o, p.detach(), v.detach())
    return o.transpose(1, 2).contiguous(), p


def _register():
    from transformers import AttentionInterface
    AttentionInterface.register("k_te", _attn)
    try:
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask
        ALL_MASK_ATTENTION_FUNCTIONS.register("k_te", sdpa_mask)
    except Exception:
        pass


@contextlib.contextmanager
def explicit(model, mode=None):
    """Run the model with the explicit attention function (mode None | 'record' | 'replay' | 'grad'); sdpa restored on exit."""
    _register(); old = model.config._attn_implementation
    model.set_attn_implementation("k_te"); _ST["mode"] = mode
    if mode in ("record", "grad"):
        _ST["probs"] = {} if mode == "record" else _ST["probs"]; _ST["heads"] = {}
    try:
        yield _ST
    finally:
        _ST["mode"] = None; model.set_attn_implementation(old)


def states_at(model, ids, block, pos, add, read_blocks):
    """Block outputs at read_blocks (full sequences) with `add` [D] added to the block-`block` output at position pos."""
    torch = T_()
    L = RC.layers_of(model)

    def hook(m, a, o):
        x = o[0] if isinstance(o, tuple) else o
        x = x.clone(); x[:, pos] = x[:, pos] + add.to(x.device, x.dtype)
        return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
    h = L[block].register_forward_hook(hook)
    try:
        with RC.Hooks(model) as hk, torch.no_grad():
            hk.capture(sorted(set(read_blocks)), cpu=False); model.model(input_ids=ids)
            return {b: hk.states[b][0] for b in read_blocks}
    finally:
        h.remove()


def cdiff(model, ids, block, pos, v, read_blocks, frozen=False):
    """Central difference (+- v at the block output, position pos) of the read blocks' outputs: dict b -> [B, T, D] / 2."""
    torch = T_()
    if frozen:
        with explicit(model, "record"):
            states_at(model, ids, block, pos, torch.zeros_like(v), read_blocks)
        with explicit(model, "replay"):
            a = states_at(model, ids, block, pos, v, read_blocks); b = states_at(model, ids, block, pos, -v, read_blocks)
    else:
        a = states_at(model, ids, block, pos, v, read_blocks); b = states_at(model, ids, block, pos, -v, read_blocks)
    return {k: (a[k] - b[k]) / 2 for k in read_blocks}


def frozen_qk_check(model, ids):
    """G-K10: the explicit pass with replayed probabilities reproduces the logits (relative max difference)."""
    torch = T_()
    with torch.no_grad():
        base = model(input_ids=ids).logits.float()
        with explicit(model, "record"):
            model(input_ids=ids)
        with explicit(model, "replay"):
            rep = model(input_ids=ids).logits.float()
    return float((rep - base).abs().max() / base.abs().max().clamp(min=1e-9))


def fd_vs_vjp(model, ids, block, pos, v, tgt_block, tpos, u, eps=1e-3):
    """u . [h_tgt(tpos | +eps v) - h_tgt(tpos | -eps v)] / (2 eps) against (J^T u) . v by autograd; relative error."""
    torch = T_()
    d = cdiff(model, ids, block, pos, eps * v, [tgt_block])[tgt_block][0, tpos] / eps
    L = RC.layers_of(model); box2 = {}

    def hook2(m, a, o):
        x = o[0] if isinstance(o, tuple) else o
        x = x.detach().requires_grad_(True); box2["x"] = x
        return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
    tgt = {}
    h1 = L[block].register_forward_hook(hook2)
    h2 = L[tgt_block].register_forward_hook(lambda m, a, o: tgt.__setitem__("y", o[0] if isinstance(o, tuple) else o))
    try:
        with torch.enable_grad():
            model.requires_grad_(False); RC.Hooks.run(lambda: model.model(input_ids=ids))
            g, = torch.autograd.grad((tgt["y"][0, tpos] * u.to(tgt["y"].dtype)).sum(), box2["x"])
    finally:
        h1.remove(); h2.remove()
    a = float(u @ d.float()); b = float(g[0, pos].float() @ v.float())
    return abs(a - b) / max(abs(b), 1e-9), a, b


def head_vjp(model, ids, F_fn, tprime):
    """VJP edge attribution: e_{m,h}(r <- s) = dF/do_{m,h}(r) . p_{m,h}(r, s) v_{m,h}(s) for F = F_fn(hidden states dict) at t';
    returns per layer [H, T, T] edges (absolute) and the routing share per lag (sum over r - s = lag of |e| / sum |e|)."""
    torch = T_()
    with explicit(model, "grad") as st, torch.enable_grad():
        emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)     # a grad leaf (the weights are frozen)
        o = model.model(inputs_embeds=emb, output_hidden_states=True)
        F = F_fn(o.hidden_states, tprime)
        heads = st["heads"]; outs = [heads[l][0] for l in sorted(heads)]
        grads = torch.autograd.grad(F, outs, allow_unused=True)
    edges, lag = {}, {}
    for l, gr in zip(sorted(heads), grads):
        if gr is None:
            continue
        _, p, v = heads[l]                                         # p [B, H, T, S], v [B, H, S, d], gr [B, H, T, d]
        e = (gr[0][:, :, None, :] * (p[0][..., None] * v[0][:, None, :, :])).sum(-1).abs()     # [H, T, S]
        edges[l] = e.cpu()
        T = e.shape[1]
        for r in range(T):
            for s in range(min(r + 1, e.shape[2])):
                lag[r - s] = lag.get(r - s, 0.0) + float(e[:, r, s].sum())
    tot = sum(lag.values()) or 1.0
    return edges, {k: v / tot for k, v in sorted(lag.items())}

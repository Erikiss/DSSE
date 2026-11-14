"""Independent native Gemma port of the frozen part-L model engine: the down_proj probe (SO-HN hook manager form: z, CETT, zCETT channels a_X(t) of many populations, spins,
all-neuron CETT means, rotated-basis CETT, neuron operations with row / position / window control), keyed Gumbel-max decoding with
generation-time capture (rk_loop's key and Gumbel draw per row and step), teacher-forced passes (channels, token log-probs, entropies,
KL between contexts), and the B = 1 chunked long-context scorer for L2a (SO-LCE form, memory-efficient kernel asserted on the GPU)."""
import contextlib, math
import numpy as np

import r1b_core as K
import r1b_g_adapter as GA


def _t():
    return K._torch()


class Probe:
    """Hooks on mlp.down_proj of the layers that carry tracked neurons (forward pre-hook: operations; forward hook: capture).
    chans: [(name, flat ids)] -> buf['a'] (mean zCETT) and buf['a2'] (mean standardised z) [B, T, C];
    spins: flat ids (ordered) -> buf['zs'] = |z| [B, T, S];
    cett_all: per row the masked mean of CETT over positions (wpos [B, T]) and CETT at lastpos [B] for all neurons (+ REF-W sums);
    rot: {layer: (R [I, I], colnorm [I])} rotated-basis CETT means;
    ops: dicts(ids (flat), kind 'scale' | 'mean' | 'zero', val (scalar, [B] or [n]), lam, rows [B] bool, pos [B, T] bool, on)."""

    def __init__(self, model, refw, chans=(), spins=(), cett_all=False, rot=None, refw_acc=False):
        torch = _t()
        self.model, self.blocks = model, GA.layers_of(model)
        self.layout = GA.NeuronLayout.from_model(model)
        self.offsets = np.asarray(self.layout.offsets)
        self.L = len(self.layout.widths)
        dev = K.model_device(model); self.dev = dev
        self.rw = {k: v.to(dev) for k, v in refw.items()} if refw is not None else None
        self.names = [n for n, _ in chans]
        self.cl, self.sl, self.ops, self.h = {}, {}, [], []
        self.cett_all, self.rot, self.refw_acc = cett_all, rot, refw_acc
        C = len(chans)
        for c, (_, ids) in enumerate(chans):
            ids = np.asarray(ids, dtype=np.int64)
            layer_of = np.searchsorted(self.offsets, ids, side="right") - 1
            for l in np.unique(layer_of):
                loc = ids[layer_of == l] - self.offsets[l]
                d = self.cl.setdefault(int(l), {})
                for i in loc:
                    d.setdefault(int(i), np.zeros(C, dtype=np.float32))[c] += 1.0 / len(ids)
        self.cl = {l: (torch.tensor(sorted(d), device=dev), torch.tensor(np.stack([d[i] for i in sorted(d)]), device=dev))
                   for l, d in self.cl.items()}
        sp = np.asarray(list(spins), dtype=np.int64)
        self.nspin = len(sp)
        spin_layers = np.searchsorted(self.offsets, sp, side="right") - 1
        for l in np.unique(spin_layers) if len(sp) else []:
            cols = np.flatnonzero(spin_layers == l)
            self.sl[int(l)] = (torch.tensor(sp[cols] - self.offsets[l], device=dev), torch.tensor(cols, device=dev))
        self.wn = [blk.mlp.down_proj.weight.detach().float().norm(dim=0) for blk in self.blocks]
        layers = set(self.cl) | set(self.sl) | (set(range(self.L)) if cett_all or refw_acc else set()) | set(rot or {})
        for l in sorted(layers):
            self.h.append(self.blocks[l].mlp.down_proj.register_forward_hook(self._cap(l)))
        self._pre = {}
        self.buf = None
        self.wpos = self.lastpos = None

    # ---- operations
    def add_op(self, ids, kind, val=None, lam=None, rows=None, pos=None, on=True, per_neuron=None):
        """val: scalar or per-row [B]; per_neuron: gains in the order of ids (X4, X5); lam: partial mean ablation (scalar or [B])."""
        torch = _t()
        ids = np.asarray(ids, dtype=np.int64)
        layer_of = np.searchsorted(self.offsets, ids, side="right") - 1
        if len(ids) and (ids.min() < 0 or ids.max() >= self.layout.total):
            raise ValueError("neuron outside Gemma ragged layout")
        per = {int(l): torch.tensor(ids[layer_of == l] - self.offsets[l], device=self.dev) for l in np.unique(layer_of)}
        pn = None if per_neuron is None else {int(l): torch.tensor(np.asarray(per_neuron, dtype=np.float32)[layer_of == l],
                                                                    device=self.dev) for l in np.unique(layer_of)}
        op = dict(per=per, kind=kind, val=val, lam=lam, rows=rows, pos=pos, on=on, ids=ids, pn=pn)
        self.ops.append(op)
        for l in per:
            if l not in self._pre:
                self._pre[l] = self.blocks[l].mlp.down_proj.register_forward_pre_hook(self._op(l))
        return op

    def clear_ops(self):
        self.ops = []

    def _op(self, l):
        torch = _t()

        def pre(mod, args):
            act = [o for o in self.ops if o["on"] and l in o["per"]]
            if not act:
                return None
            x = args[0].clone()
            for o in act:
                i = o["per"][l]; sel = x[..., i]
                if o["kind"] == "scale":
                    if o["pn"] is not None:
                        v = o["pn"][l].to(x.dtype)
                    else:
                        v = torch.as_tensor(o["val"], dtype=x.dtype, device=x.device)
                        if v.dim() == 1:
                            v = v[:, None, None]
                    new = sel * v
                elif o["kind"] == "mean":
                    mu = self.rw["mu_z"][int(self.offsets[l]) + i].to(x.dtype)
                    lam = torch.as_tensor(o["lam"], dtype=x.dtype, device=x.device)
                    if lam.dim() == 1:
                        lam = lam[:, None, None]
                    new = mu + lam * (sel - mu)
                else:
                    new = torch.zeros_like(sel)
                m = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
                if o["rows"] is not None:
                    m = m & torch.as_tensor(o["rows"], device=x.device)[:, None]
                if o["pos"] is not None and x.shape[1] > 1:
                    m = m & torch.as_tensor(o["pos"], device=x.device)[:, -x.shape[1]:]
                x[..., i] = torch.where(m[..., None], new, sel)
            return (x,) + tuple(args[1:])
        return pre

    # ---- capture
    def begin(self, wpos=None, lastpos=None):
        self.buf = {}
        self.wpos, self.lastpos = wpos, lastpos

    def _cap(self, l):
        torch = _t()

        def hook(mod, inp, out):
            z = inp[0]
            B, T = z.shape[:2]
            b = self.buf
            if b is None:
                return
            on = out.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            if l in self.cl:
                U, M = self.cl[l]
                zs = z[..., U].float(); g = int(self.offsets[l]) + U
                c = zs.abs() * self.wn[l][U] / on.float()
                zc = (c - self.rw["mu_cett"][g]) / self.rw["sd_cett"][g]
                zz = (zs - self.rw["mu_z"][g]) / self.rw["sd_z"][g]
                if "a" not in b:
                    C = M.shape[1]
                    b["a"] = torch.zeros(B, T, C, device=z.device); b["a2"] = torch.zeros(B, T, C, device=z.device)
                b["a"] += zc @ M; b["a2"] += zz @ M
            if l in self.sl:
                S, cols = self.sl[l]
                if "zs" not in b:
                    b["zs"] = torch.zeros(B, T, self.nspin, device=z.device, dtype=torch.float16)
                b["zs"][..., cols] = z[..., S].abs().to(torch.float16)
            if self.cett_all or self.refw_acc or (self.rot and l in self.rot):
                c = z.float().abs() * self.wn[l] / on.float()
                w = self.wpos.to(z.device).float() if self.wpos is not None else torch.ones(B, T, device=z.device)
                ws = w.sum(1, keepdim=True).clamp_min(1e-9)
                sl = slice(self.offsets[l], self.offsets[l + 1])
                if self.cett_all:
                    b.setdefault("cmean", torch.zeros(B, self.layout.total, device=z.device))[:, sl] = (c * w[..., None]).sum(1) / ws
                    if self.lastpos is not None:
                        b.setdefault("clast", torch.zeros(B, self.layout.total, device=z.device))[:, sl] = \
                            c[torch.arange(B, device=z.device), self.lastpos.to(z.device)]
                if self.refw_acc:
                    zf = z.float()
                    for k, v in (("sz", zf), ("szz", zf * zf), ("sc", c), ("scc", c * c)):
                        b.setdefault(k, torch.zeros(self.layout.total, device=z.device, dtype=torch.float64))[sl] += \
                            (v * w[..., None]).sum((0, 1)).double()
                    b["n"] = b.get("n", 0.0) + (float(w.sum()) if l == 0 else 0.0)
                if self.rot and l in self.rot:
                    R, cn = self.rot[l]
                    cr = (z.float() @ R.T).abs() * cn / on.float()
                    b.setdefault("rmean", torch.zeros(B, self.layout.total, device=z.device))[:, sl] = (cr * w[..., None]).sum(1) / ws
        return hook

    def remove(self):
        for h in self.h + list(self._pre.values()):
            h.remove()
        self.h, self._pre = [], {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.remove()


@contextlib.contextmanager
def maybe(probe):
    yield probe


def _gumbel(gens, V, dev):
    torch = _t()
    return torch.stack([-torch.log(-torch.log(torch.rand(V, generator=G, device=dev).clamp_(1e-12, 1 - 1e-7))) for G in gens])


def decode(model, tok, prompts, keys, T=0.7, max_new=64, probe=None, mask_eos=False, advance=None, greedy=False, on_step=None,
           spin_fn=None, stop=True):
    """Keyed Gumbel-max decoding (rk_loop's draw: one torch.rand(V) per row and step on a generator seeded with the key; scores =
    logits / T + g; argmax), KV cache, left padding, explicit positions. Step 0 is the prefill; step k >= 1 processes x_k.
    Per row (length n = number of generated tokens before the stop token): tokens x_1..x_n; per position k = 1..n: a, a2 [n, C] (probe
    channels while processing x_k), e (entropy at T 1 of the next-token distribution), peos (its EOS mass), s (-log q(x_(k+1)), the
    stop token or the extra draw at the end); zs via spin_fn(step, zs [B, S]) (per-step reduction, e.g. packed spins).
    advance [B]: draws discarded per row before decoding (the twin protocol's draw-and-discard). on_step(step, probe) switches windows."""
    torch = _t()
    from transformers import DynamicCache
    dev = K.model_device(model)
    eos, stops = GA.stop_tokens(tok, model)
    B = len(prompts)
    if B == 0:
        return []
    if len(keys) != B or any(not p for p in prompts):
        raise ValueError("native L decode needs nonempty prompts and one key per row")
    if max(map(len, prompts)) + max_new > GA.text_config(model).max_position_embeddings:
        raise K.NotRun("native Gemma context limit exceeded; no silent history truncation")
    ids, am, pad = K.pad_left(prompts, tok.pad_token_id)
    ids, am = ids.to(dev), am.to(dev)
    V = model.get_output_embeddings().weight.shape[0]
    gens = [torch.Generator(device=dev).manual_seed(int(k)) for k in keys]
    if advance is not None:
        for r, n in enumerate(advance):
            for _ in range(int(n)):
                torch.rand(V, generator=gens[r], device=dev)
    eos_t = torch.tensor(sorted(eos), device=dev)
    cache = None
    pos = (am.cumsum(-1) - 1).clamp_min(0)
    cur, full_am, cur_pos = ids, am, pos
    alive = torch.ones(B, dtype=torch.bool, device=dev)
    length = torch.zeros(B, dtype=torch.long, device=dev)
    rec = dict(tok=[], a=[], a2=[], e=[], peos=[], s=[], z=[])
    s0 = e0 = None
    for step in range(max_new + 1):
        if on_step is not None:
            on_step(step, probe)
        if probe is not None:
            probe.begin()
        with torch.no_grad():
            out = GA.text_model(model)(input_ids=cur, attention_mask=full_am, position_ids=cur_pos,
                                       past_key_values=cache, use_cache=True, logits_to_keep=0)
            lg = GA.unembed(model, out.last_hidden_state[:, -1], normalized=True).float()
        cache = out.past_key_values
        lp = torch.log_softmax(lg, -1); p = lp.exp()
        e = -(p * lp).sum(-1); pe = p[:, eos_t].sum(-1)
        if step >= 1:
            rec["e"].append(e.cpu()); rec["peos"].append(pe.cpu())
            if probe is not None and probe.buf and "a" in probe.buf:
                rec["a"].append(probe.buf["a"][:, -1].cpu()); rec["a2"].append(probe.buf["a2"][:, -1].cpu())
            if probe is not None and probe.buf and "zs" in probe.buf and spin_fn is not None:
                rec["z"].append(spin_fn(step, probe.buf["zs"][:, -1]))
        else:
            e0 = e.cpu()
        sc = lg.clone()
        if mask_eos:
            sc[:, eos_t] = -float("inf")
        if greedy:
            nxt = sc.argmax(-1)
        else:
            nxt = (sc / T + _gumbel(gens, V, dev)).argmax(-1)
        s = -lp.gather(1, nxt[:, None])[:, 0]
        if step >= 1:
            rec["s"].append(s.cpu())
        else:
            s0 = s.cpu()
        is_stop = torch.tensor([int(t) in stops for t in nxt.tolist()], device=dev) if stop else torch.zeros(B, dtype=torch.bool, device=dev)
        newly = alive & is_stop
        if step < max_new:
            length = torch.where(alive & ~is_stop, length + 1, length)
        alive = alive & ~is_stop
        if step < max_new:
            rec["tok"].append(torch.where(alive, nxt, torch.full_like(nxt, tok.pad_token_id)).cpu())
        if step == max_new or not bool(alive.any()):
            break
        cur = torch.where(alive, nxt, torch.full_like(nxt, tok.pad_token_id))[:, None]
        full_am = torch.cat([full_am, torch.ones(B, 1, dtype=full_am.dtype, device=dev)], 1)
        cur_pos = cur_pos[:, -1:] + 1
    if probe is not None:
        probe.buf = None
    n = length.cpu().tolist()
    stack = lambda L_: torch.stack(L_, 1) if L_ else None
    TOK, A, A2, E, PE, S_ = (stack(rec[k]) for k in ("tok", "a", "a2", "e", "peos", "s"))
    res = []
    for r in range(B):
        k = n[r]
        d = dict(tokens=TOK[r, :k].tolist() if TOK is not None else [], n=k, ended=k < max_new,
                 e=E[r, :k].numpy() if E is not None else np.zeros(0), peos=PE[r, :k].numpy() if PE is not None else np.zeros(0),
                 s=S_[r, :k].numpy() if S_ is not None else np.zeros(0), e0=float(e0[r]), s0=float(s0[r]))
        if A is not None:
            d["a"], d["a2"] = A[r, :k].numpy(), A2[r, :k].numpy()
        elif probe is not None and probe.names:
            d["a"], d["a2"] = np.zeros((0, len(probe.names))), np.zeros((0, len(probe.names)))
        if rec["z"]:
            d["z"] = [tuple(x[r] for x in z) if isinstance(z, tuple) else z[r] for z in rec["z"][:k]]
        res.append(d)
    return res


def tf_pass(model, tok, seqs, starts, probe=None, batch=16, chunk=256, lastpos=False, wpos_fn=None, eff=False, kl_with=None):
    """Teacher-forced reads: seqs = full id lists, starts[r] = index of the first continuation token. Per row, for continuation
    tokens k = 1..n: lp (log q(x_k | prefix)), e (entropy at the position of x_k), s (-log q(x_(k+1)); last = NaN), a / a2 [n, C]
    (probe channels at x_k), plus the probe's cmean / clast / rmean / zs when requested (cett_all: wpos_fn(ids, pad, starts) -> [B, T]
    weights, lastpos -> index of the last prompt token)."""
    torch = _t()
    dev = K.model_device(model); out = []
    for i in range(0, len(seqs), batch):
        sb, st = seqs[i:i + batch], starts[i:i + batch]
        ids, am, pad = K.pad_left(sb, tok.pad_token_id)
        ids, am = ids.to(dev), am.to(dev)
        pos = (am.cumsum(-1) - 1).clamp_min(0)
        p0 = pad.to(dev) + torch.tensor(st, device=dev)
        if probe is not None:
            probe.begin(wpos=None if wpos_fn is None else wpos_fn(ids, pad, st), lastpos=(p0 - 1) if lastpos else None)
        cm = K.sdpa_efficient(dev) if (eff or ids.shape[1] > 2048) else contextlib.nullcontext()
        with cm, torch.no_grad():
            h = GA.text_model(model)(input_ids=ids, attention_mask=am, position_ids=pos, use_cache=False, logits_to_keep=0).last_hidden_state
        Tn = ids.shape[1]
        for r in range(len(sb)):
            P0 = int(p0[r]); n = Tn - P0
            lp = np.full(n, np.nan); e = np.full(n, np.nan)
            for c0 in range(P0 - 1, Tn, chunk):
                c1 = min(c0 + chunk, Tn)
                with torch.no_grad():
                    ls = torch.log_softmax(GA.unembed(model, h[r, c0:c1], normalized=True).float(), -1)
                js = torch.arange(c0, c1, device=dev)
                m = js + 1 < Tn                               # position j predicts token j + 1 (continuation index j + 1 - P0)
                if bool(m.any()):
                    lp[(js[m] + 1 - P0).cpu().numpy()] = ls[m].gather(1, ids[r, js[m] + 1][:, None])[:, 0].cpu().numpy()
                me = js >= P0
                if bool(me.any()):
                    e[(js[me] - P0).cpu().numpy()] = (-(ls[me].exp() * ls[me]).sum(-1)).cpu().numpy()
            d = dict(lp=lp, e=e, s=np.append(-lp[1:], np.nan), n=n)
            if probe is not None and probe.buf:
                b = probe.buf
                if "a" in b:
                    d["a"] = b["a"][r, int(p0[r]):].cpu().numpy(); d["a2"] = b["a2"][r, int(p0[r]):].cpu().numpy()
                for k in ("cmean", "clast", "rmean"):
                    if k in b:
                        d[k] = b[k][r].cpu()
                if "zs" in b:
                    d["zs"] = b["zs"][r, int(p0[r]):].cpu()
            out.append(d)
        if probe is not None and probe.buf is not None and probe.refw_acc:
            out.append(dict(refw_sums={k: probe.buf[k].cpu() for k in ("sz", "szz", "sc", "scc") if k in probe.buf},
                            n=probe.buf.get("n", 0.0)))
        if probe is not None:
            probe.buf = None
    return out


def ctx_logprobs(model, tok, groups, batch=12, chunk=128):
    """Anchor information (L2b): groups = [(ctx_ids_list, cont)] with ctx_ids_list = [A, A', A''] (prompts) and the shared continuation.
    Per group: lp [3, n] (log q(z_k | ctx, z_<k)) and kl [2, n]: KL(q(.|A) || q(.|A')) and KL(q(.|A'') || q(.|A')) at every z_k
    (the expected anchor information iota*, validity fix; realised iota = lp[0] - lp[1])."""
    torch = _t()
    dev = K.model_device(model); res = []
    rows = [(g, j) for g in range(len(groups)) for j in range(3)]
    per = max(1, batch // 3)
    for i in range(0, len(groups), per):
        gs = groups[i:i + per]
        seqs = [list(c[j]) + list(z) for c, z in gs for j in range(3)]
        st = [len(c[j]) for c, z in gs for j in range(3)]
        ids, am, pad = K.pad_left(seqs, tok.pad_token_id)
        ids, am = ids.to(dev), am.to(dev)
        pos = (am.cumsum(-1) - 1).clamp_min(0)
        with torch.no_grad():
            h = GA.text_model(model)(input_ids=ids, attention_mask=am, position_ids=pos, use_cache=False, logits_to_keep=0).last_hidden_state
        Tn = ids.shape[1]
        for g, (c, z) in enumerate(gs):
            n = len(z)
            lp = np.zeros((3, n)); kl = np.zeros((2, n))
            for c0 in range(0, n, chunk):
                c1 = min(c0 + chunk, n)
                L3 = []
                for j in range(3):
                    r = 3 * g + j; a0 = int(pad[r]) + st[3 * g + j] - 1
                    with torch.no_grad():
                        ls = torch.log_softmax(GA.unembed(model, h[r, a0 + c0:a0 + c1], normalized=True).float(), -1)
                    L3.append(ls)
                    tg = torch.tensor(z[c0:c1], device=dev)
                    lp[j, c0:c1] = ls.gather(1, tg[:, None])[:, 0].cpu().numpy()
                kl[0, c0:c1] = (L3[0].exp() * (L3[0] - L3[1])).sum(-1).cpu().numpy()
                kl[1, c0:c1] = (L3[2].exp() * (L3[2] - L3[1])).sum(-1).cpu().numpy()
            res.append(dict(lp=lp, kl=kl))
    return res


def lce_logprobs(model, ids, chunk=2048, need_from=1):
    """B = 1 chunked prefill with a DynamicCache and explicit positions (SO-LCE form): log q(x_j | x_<j) for j = need_from..len-1
    (index j of the returned array; earlier entries NaN). lm_head only on the needed positions."""
    torch = _t()
    from transformers import DynamicCache
    dev = K.model_device(model)
    x = torch.tensor(ids, device=dev)[None]
    n = x.shape[1]; out = np.full(n, np.nan)
    cache = None
    for s in range(0, n, chunk):
        e_ = min(s + chunk, n)
        with K.sdpa_efficient(dev), torch.no_grad():
            result = GA.text_model(model)(input_ids=x[:, s:e_], position_ids=torch.arange(s, e_, device=dev)[None],
                                         attention_mask=torch.ones((1, e_), dtype=torch.long, device=dev),
                                         past_key_values=cache, use_cache=True, logits_to_keep=0)
            cache = result.past_key_values
            h = result.last_hidden_state[0]
        js = [j for j in range(max(s, need_from - 1), e_) if j + 1 < n]          # position j predicts x_(j+1)
        for c0 in range(0, len(js), 32):
            jj = js[c0:c0 + 32]
            if not jj:
                continue
            with torch.no_grad():
                ls = torch.log_softmax(GA.unembed(model, h[torch.tensor(jj, device=dev) - s], normalized=True).float(), -1)
            tg = x[0, torch.tensor(jj, device=dev) + 1]
            out[np.array(jj) + 1] = ls.gather(1, tg[:, None])[:, 0].double().cpu().numpy()
    return out


def full_logprobs(model, ids, need_from=1):
    """One full causal pass (parity reference for G-L2a-LCE)."""
    torch = _t()
    dev = K.model_device(model)
    x = torch.tensor(ids, device=dev)[None]; n = x.shape[1]; out = np.full(n, np.nan)
    with K.sdpa_efficient(dev), torch.no_grad():
        h = GA.text_model(model)(input_ids=x, position_ids=torch.arange(n, device=dev)[None], use_cache=False, logits_to_keep=0).last_hidden_state[0]
    for c0 in range(max(0, need_from - 1), n - 1, 32):
        c1 = min(c0 + 32, n - 1)
        with torch.no_grad():
            ls = torch.log_softmax(GA.unembed(model, h[c0:c1], normalized=True).float(), -1)
        out[c0 + 1:c1 + 1] = ls.gather(1, x[0, c0 + 1:c1 + 1][:, None])[:, 0].double().cpu().numpy()
    return out


def chat_sys_ids(tok, system, user):
    """Chat template with a system message, thinking off (the empty think block), as ids."""
    m = ([dict(role="system", content=system)] if system else []) + [dict(role="user", content=user)]
    return K.enc(tok, tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False))


def turn_suffix_ids(tok, user):
    """Native Gemma suffix after an assistant reply, derived from its template."""
    marker = "R1B_G_TURN_BOUNDARY_a3fc"
    messages = [{"role": "user", "content": "initial"}, {"role": "assistant", "content": marker},
                {"role": "user", "content": user}]
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if rendered.count(marker) != 1:
        raise ValueError("Gemma chat template did not preserve the assistant turn boundary")
    return tok(rendered.split(marker)[1], add_special_tokens=False)["input_ids"]


def logit_choice(model, tok, ctx_ids, cands, batch=32):
    """Teacher-forced forced choice: log q(first token of each candidate | ctx) for every context row -> [N, len(cands)]."""
    torch = _t()
    dev = K.model_device(model)
    ct = [K.enc(tok, c)[0] for c in cands]
    out = []
    for i in range(0, len(ctx_ids), batch):
        ids, am, pad = K.pad_left(ctx_ids[i:i + batch], tok.pad_token_id)
        with torch.no_grad():
            lg = model(input_ids=ids.to(dev), attention_mask=am.to(dev),
                       position_ids=(am.cumsum(-1) - 1).clamp_min(0).to(dev), logits_to_keep=1).logits[:, -1].float()
        ls = torch.log_softmax(lg, -1)
        out.append(ls[:, ct].cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(cands)))


def choice_rows(model, tok, ctx_ids, cand_ids, batch=32):
    """Batched forced choice with per-row candidate token ids: log q(cand_ids[r][j] | ctx_ids[r]) -> [N, k]."""
    torch = _t()
    dev = K.model_device(model)
    out = []
    for i in range(0, len(ctx_ids), batch):
        ids, am, pad = K.pad_left(ctx_ids[i:i + batch], tok.pad_token_id)
        with torch.no_grad():
            lg = model(input_ids=ids.to(dev), attention_mask=am.to(dev),
                       position_ids=(am.cumsum(-1) - 1).clamp_min(0).to(dev), logits_to_keep=1).logits[:, -1].float()
        ls = torch.log_softmax(lg, -1)
        out.append(ls.gather(1, torch.tensor(cand_ids[i:i + batch], device=dev)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(cand_ids[0]) if cand_ids else 0))

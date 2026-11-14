"""SO-CL (part K): the chain lens CL17 (self-rollout from a patched placeholder, two streams real / prior with contrastive decoding,
re-injection off unless asked), its per-step channels (log p, e(s), ENT, CDM, TOP1, FOV_att exactly from q and the cached k, a_X of
named neuron sets, DLA shares), the inverse-free verifier (forward round trip through the unmodified model), J-lens seeds, priors,
V7's covariance-matched null, MTP-style heads (CL35 native Qwen3.5 MTP; RK-1's trained heads) and the self-consistency KL of
Teoh 2026 (arXiv 2610.38331): KL(q || p) between the model's own next-token distribution at the aligned position and the lens's."""
import math, sys
import numpy as np
import r1b_core as RC
from r1b_k_base import T_, Tap, Interv, SetReader, specials, lm_parts, Gumbel, by_layer

_PIECES, _VALID = {}, {}


def pieces(tok):
    k = id(tok)
    if k not in _PIECES:
        _PIECES[k] = [tok.decode([i]) for i in range(len(tok))]
    return _PIECES[k]


def valid_seed_mask(tok, V, dev):
    """Tokens with at least one letter or digit (J-lens seeds)."""
    k = (id(tok), V)
    if k not in _VALID:
        torch = T_(); p = pieces(tok); m = torch.zeros(V, dtype=torch.bool)
        for i, s in enumerate(p[:V]):
            m[i] = any(c.isalnum() for c in s)
        _VALID[k] = m
    return _VALID[k].to(dev)


def jmap(J, conv, l):
    """The J-lens map that transports a block-l OUTPUT state (K4j convention: 'output' -> J[l]; 'input' -> J[l + 1]); None past 26."""
    i = l if conv == "output" else l + 1
    return J["maps"][i] if i < J["maps"].shape[0] else None


def transport(J, conv, l, h):
    """J_l h = h @ J[l].T (the logit lens beyond the last map)."""
    m = jmap(J, conv, l)
    return h if m is None else h @ m.to(h.device, h.dtype).T


def jlens_seeds(model, tok, J, conv, l, h, k=10):
    """J-lens top-k tokens of unembed(J_l h) after dropping tokens without a letter or digit; [N, k] ids."""
    torch = T_()
    norm, W, _ = lm_parts(model)
    with torch.no_grad():
        lg = norm(transport(J, conv, l, h.to(W.device, W.dtype))) @ W.T
        lg = lg.masked_fill(~valid_seed_mask(tok, lg.shape[-1], lg.device) | specials(tok, lg.shape[-1], lg.device), float("-inf"))
    return lg.topk(k, -1).indices.cpu()


def v7_states(mu, h, sig_half, keys):
    """V7: r = mu + ||h - mu|| unit(Sigma^{1/2} g), g ~ N(0, I) seeded per cell (keys)."""
    torch = T_()
    out = []
    for m, x, k in zip(mu, h, keys):
        g = torch.randn(m.shape[0], generator=torch.Generator().manual_seed(int(k) % (2 ** 63)), dtype=torch.float64)
        v = (sig_half @ g).float(); out.append(m + (x - m).norm() * v / v.norm())
    return torch.stack(out)


class Lens:
    """CL17 on a decoder: frame = P0 (16 tokens) + placeholder ' ...' (index 16), seed at index 17, then the chain."""

    def __init__(self, model, tok, frames, wn=None, ref=None, sets=None):
        torch = T_()
        self.model, self.tok, self.dev = model, tok, RC.model_device(model)
        self.frame = list(frames["P0_ids"]) + [frames["ph_id"]]; self.ph = len(self.frame) - 1
        self.L = RC.layers_of(model); self.nL = len(self.L)
        self.norm, self.WU, self.WUbar = lm_parts(model)
        self.V = self.WU.shape[0]; self.spec = specials(tok, self.V, self.dev)
        self.wn, self.ref, self.sets = wn, ref, sets or {}
        self.rope = sys.modules[type(self.L[0].self_attn).__module__].apply_rotary_pos_emb

    # ---- hooks
    def _patch(self, block, vecs, pos):
        def hook(m, a, o):
            x = o[0] if isinstance(o, tuple) else o
            if x.shape[1] > max(pos, 1):          # the prefill chunk only (decode steps have one token)
                x = x.clone(); x[:, pos] = vecs.to(x.device, x.dtype)
                return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
        return self.L[block].register_forward_hook(hook)

    def _reinj(self, block, vecs, rows, st):
        def hook(m, a, o):
            x = o[0] if isinstance(o, tuple) else o
            p = st["p0"] + T_().arange(x.shape[1], device=x.device)
            msk = (p >= 0)[None, :, None] & rows.to(x.device)[:, None, None]
            x = x + msk * vecs.to(x.device, x.dtype)[:, None, :]
            return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
        return self.L[block].register_forward_hook(hook)

    def _fov(self, layers, key_pos, acc):
        torch = T_()
        hs = []
        for l in layers:
            at = self.L[l].self_attn

            def hook(m, args, kw, out, l=l):
                h = kw.get("hidden_states", args[0] if args else None)[:, -1:]
                cos, sin = kw["position_embeddings"]; pkv = kw.get("past_key_values")
                B = h.shape[0]; hd = m.head_dim
                q = m.q_norm(m.q_proj(h).view(B, 1, -1, hd)).transpose(1, 2)
                q, _ = self.rope(q, q, cos[:, -1:], sin[:, -1:])
                k = pkv.layers[m.layer_idx].keys
                k = k.repeat_interleave(q.shape[1] // k.shape[1], 1)
                p = torch.softmax((q.float() @ k.float().transpose(-1, -2)) * m.scaling, -1)[..., 0, :]
                acc.append(p[..., key_pos].mean(1))
            hs.append(at.register_forward_hook(hook, with_kwargs=True))
        return hs

    # ---- the rollout
    def rollout(self, block, Hreal, Hprior, seeds, J=16, variant="E", lam=1.0, plaus=0.1, greedy=False, T=None, keys=None,
                ops=None, reinj=None, forced=None, sets=True, dla=False, fov=True, prefix=None, bias=None, keep_first=False):
        """Hreal / Hprior [N, D]: the vectors written at the placeholder (block-l output for 'E'; block-26 output for 'L', already
        n26 * unit(J_l h)). prefix: (DynamicCache for B = 1, length) of a shared filler prefix (SO-LCE copy path, K6f / K6e-d).
        ops: neuron interventions (r1b_k_base.Interv) with 'when' on the chain position (state['p0']); reinj: (beta * (h - mu) [N, D]).
        Returns tokens [N, J] and per-step records (index k = chain position k = forward whose output selects token k + 1)."""
        torch = T_()
        from transformers import DynamicCache
        N = len(seeds); B = 2 * N; dev = self.dev
        pb = block if variant == "E" else self.nL - 2
        vec = torch.cat([Hreal, Hprior]).float()
        if prefix is None:
            ids = torch.tensor([self.frame + [int(s)] for s in seeds] * 2, device=dev); ph_abs, off = self.ph, 0; cache = DynamicCache()
            p0 = -len(self.frame)
        else:
            pc, Lp = prefix; cache = DynamicCache()
            for li, lay in enumerate(pc.layers):
                cache.update(lay.keys.repeat(B, 1, 1, 1), lay.values.repeat(B, 1, 1, 1), li)
            ids = torch.tensor([[self.frame[-1], int(s)] for s in seeds] * 2, device=dev); ph_abs, off = Lp, Lp; p0 = -1
        st = dict(p0=p0); rec = {k: [] for k in ("lpr", "lpp", "ent", "cdm", "top1", "alt", "fov")}
        rec.update(aX={k: [] for k in self.sets} if sets else {}, share={k: [] for k in self.sets} if dla else {})
        hooks, fovacc, buf, zfull = [], [], {}, {}
        sr = SetReader(self.sets, self.wn, self.ref["c_mu"], self.ref["c_sd"]) if (sets and self.sets) else None
        try:
            hooks.append(self._patch(pb, vec, 0 if prefix is not None else self.ph))
            if reinj is not None:
                hooks.append(self._reinj(block, torch.cat([reinj, torch.zeros_like(reinj)]), torch.arange(B) < N, st))
            if fov:
                hooks += self._fov(range(block + 1 if variant == "E" else pb + 1, self.nL), ph_abs, fovacc)
            if sr is not None or dla:
                def tapf(l, z, den):
                    if sr is not None:
                        sr(l, z, den)
                    if dla:
                        zfull[l] = z[:, -1].float()
                tap = Tap(self.model, tapf)
            else:
                tap = None
            if dla:
                hooks.append(self.L[-1].register_forward_hook(lambda m, a, o: buf.__setitem__("h27", (o[0] if isinstance(o, tuple) else o)[:, -1].float())))
            if ops:                                    # rows 'real': the expert (real) stream of this call's 2N rows
                ops = [dict(o, rows=(torch.arange(B) < N)) if isinstance(o.get("rows"), str) else o for o in ops]
            iv = Interv(self.model, ops, st) if ops else None
            G = Gumbel(dev, T, list(keys) * 2) if keys is not None else None
            toks = torch.zeros(N, J, dtype=torch.long); first = None
            pos = torch.arange(ids.shape[1], device=dev)[None].repeat(B, 1) + off
            with torch.no_grad(), RC.sdpa_efficient(dev):            # K.0: the memory-efficient kernel is asserted
                for k in range(J):
                    fovacc.clear()
                    o = self.model(input_ids=ids, position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1)
                    lg = o.logits[:, -1].float(); lr, lp = lg[:N].log_softmax(-1), lg[N:].log_softmax(-1)
                    lr_m = lr.masked_fill(self.spec, float("-inf"))
                    if greedy:
                        s = lr_m.clone()
                    else:
                        s = (lr - lam * lp).masked_fill((lr < lr_m.max(-1, keepdim=True).values + math.log(plaus)) | self.spec, float("-inf"))
                    if keep_first and k == 0:
                        first = dict(lr=lr.cpu(), lp=lp.cpu(), plaus=torch.isfinite(s).cpu())
                    if bias is not None:
                        s = s + (bias(k) if callable(bias) else bias)
                    sel = s / T + G(torch.zeros(B, s.shape[1], device=dev))[:N] if G is not None else s
                    top2 = sel.topk(2, -1); x = top2.indices[:, 0]
                    if forced is not None:
                        for i, f in enumerate(forced):
                            if k < len(f):
                                x[i] = int(f[k])
                    toks[:, k] = x.cpu()
                    sx = s.gather(1, x[:, None])[:, 0]
                    rec["lpr"].append(lr.gather(1, x[:, None])[:, 0].cpu()); rec["lpp"].append(lp.gather(1, x[:, None])[:, 0].cpu())
                    pr = lr.exp(); rec["ent"].append(-(pr * lr.nan_to_num(neginf=0.0)).sum(-1).cpu()); rec["top1"].append(pr.max(-1).values.cpu())
                    rec["cdm"].append((sx - torch.where(top2.indices[:, 0] == x, top2.values[:, 1], top2.values[:, 0])).cpu())
                    rec["alt"].append(torch.where(top2.indices[:, 0] == x, top2.indices[:, 1], top2.indices[:, 0]).cpu())
                    rec["fov"].append(torch.stack(fovacc).mean(0)[:N].cpu() if fovacc else torch.full((N,), float("nan")))
                    if sr is not None:
                        for nm in self.sets:
                            a = sr.aX(nm); rec["aX"][nm].append(a[:N, -1].cpu() if a is not None else torch.full((N,), float("nan")))
                    if dla:
                        u = self.norm.weight.float() * (self.WU[x].float() - self.WUbar.float())          # [N, D] (same token both streams)
                        h27 = buf["h27"]; rms = h27.pow(2).mean(-1).add(1e-6).sqrt()                        # [2N]
                        tot = torch.zeros(N, device=dev); part = {nm: torch.zeros(N, device=dev) for nm in self.sets}
                        for l in range(self.nL):
                            Wd = self.L[l].mlp.down_proj.weight.float()                                   # [D, n]
                            c = u @ Wd                                                                    # [N, n]
                            d = zfull[l][:N] * c / rms[:N, None] - zfull[l][N:] * c / rms[N:, None]
                            tot += d.abs().sum(-1)
                            for nm, nl in self.sets.items():
                                ix = by_layer(nl).get(l)
                                if ix:
                                    part[nm] += d[:, ix].sum(-1)
                        for nm in self.sets:
                            rec["share"][nm].append((part[nm] / tot.clamp(min=1e-12)).cpu())
                    if k == J - 1:
                        break
                    st["p0"] = k + 1
                    ids = torch.cat([x, x])[:, None]; pos = pos[:, -1:] + 1
        finally:
            for h in hooks:
                h.remove()
            if tap is not None:
                tap.__exit__()
            if iv is not None:
                iv.__exit__()
        out = {k: torch.stack(v, 1) for k, v in rec.items() if isinstance(v, list) and v}
        out["aX"] = {k: torch.stack(v, 1) for k, v in rec["aX"].items() if v}
        out["share"] = {k: torch.stack(v, 1) for k, v in rec["share"].items() if v}
        out["e"] = out["lpr"] - out["lpp"]; out["tok"] = toks; out["first"] = first
        return out

    # ---- verifier (inverse-free, SO-GVRA): the chain fed through the unmodified model
    def verify(self, block, chains, H, mu, batch=64):
        """score = cos(s(c) - mu, h - mu), s(c) = mean block-l state over the chain positions of P0 + chain (no placeholder)."""
        torch = T_()
        P0 = self.frame[:-1]; out = []
        for i in range(0, len(chains), batch):
            cb = chains[i:i + batch]; L_ = max(len(c) for c in cb)
            ids = torch.tensor([P0 + list(c) + [self.tok.pad_token_id] * (L_ - len(c)) for c in cb], device=self.dev)
            with RC.Hooks(self.model) as hk, torch.no_grad():
                hk.capture([block], cpu=False).stop_after(block); hk.run(lambda: self.model.model(input_ids=ids))
                hs = hk.states[block][0][:, len(P0):]
            for r, c in enumerate(cb):
                s = hs[r, :len(c)].mean(0) - mu[i + r].to(hs.device); v = H[i + r].to(hs.device) - mu[i + r].to(hs.device)
                out.append(float(torch.nn.functional.cosine_similarity(s, v, 0)))
        return np.array(out)

    # ---- teacher-forced log-probs of the real stream along given chains (self-consistency KL)
    def tf_logp(self, block, Hreal, seeds, chains, variant="E"):
        """log p_real [N, J, V] along the chains (patched placeholder), J = len(chain)."""
        torch = T_()
        pb = block if variant == "E" else self.nL - 2
        ids = torch.tensor([self.frame + [int(s)] + list(c[:-1]) for s, c in zip(seeds, chains)], device=self.dev)
        h = self._patch(pb, Hreal.float(), self.ph)
        try:
            with torch.no_grad():
                lg = self.model(input_ids=ids).logits[:, self.ph + 1:].float()
        finally:
            h.remove()
        return lg.log_softmax(-1)


def backbone_logq(model, ctx_ids, seeds, chains):
    """The model's own next-token log-probs at the aligned positions: item context up to t0, then the seed and the chain
    (teacher-forced); [N, J, V] for chain tokens 1..J."""
    torch = T_()
    dev = RC.model_device(model); seqs = [list(c0) + [int(s)] + list(c[:-1]) for c0, s, c in zip(ctx_ids, seeds, chains)]
    ids, am, pad = RC.pad_left(seqs, 0); J = len(chains[0])
    with torch.no_grad():
        lg = model(input_ids=ids.to(dev), attention_mask=am.to(dev), position_ids=(am.cumsum(-1) - 1).clamp(min=0).to(dev),
                   logits_to_keep=J).logits.float()
    return lg.log_softmax(-1)


def kl_rows(lq, lp):
    """KL(q || p) per row and step from log-probs [N, J, V]."""
    return (lq.exp() * (lq - lp)).sum(-1)


# ---------------------------------------------------------------------------------------------------------------------------------
# MTP-style heads: CL35 (Qwen3.5-2B native MTP, vLLM qwen3_5_mtp wiring) and RK-1's trained heads (DeepSeek-V3 style)
# ---------------------------------------------------------------------------------------------------------------------------------
def rotary_of(model):
    for path in ("model.rotary_emb", "model.language_model.rotary_emb", "model.model.rotary_emb"):
        m = model
        try:
            for a in path.split("."):
                m = getattr(m, a)
            return m
        except AttributeError:
            continue
    raise AttributeError("rotary embedding not found")


def make_head(model, layer_cls_cfg=None, init_block=None):
    """An MTP head: fc([norm_e(emb); norm_h(hidden)]) -> one decoder layer -> norm -> the model's (tied) head. The decoder layer is a
    copy of block init_block (RK-1) or a fresh layer of the model's class (CL35 loads the mtp.* tensors into it)."""
    torch = T_()
    import copy
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    D = cfg.hidden_size; L = RC.layers_of(model)
    lay = copy.deepcopy(L[init_block if init_block is not None else -1])
    lay.self_attn.layer_idx = 0
    Norm = type(L[0].input_layernorm)

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2 * D, D, bias=False); self.pre_fc_norm_embedding = Norm(D, eps=cfg.rms_norm_eps)
            self.pre_fc_norm_hidden = Norm(D, eps=cfg.rms_norm_eps); self.layer = lay; self.norm = copy.deepcopy(lm_parts(model)[0])
            self.wiring = ("eh", "post")

        def step(self, e, h, cache, pos, rot):
            a, b = self.pre_fc_norm_embedding(e), self.pre_fc_norm_hidden(h)
            x = self.fc(torch.cat([a, b] if self.wiring[0] == "eh" else [b, a], -1))[:, None]
            pe = rot(x, pos)
            y = self.layer(hidden_states=x, attention_mask=None, position_ids=pos, past_key_values=cache, use_cache=True,
                           position_embeddings=pe)
            y = y[0] if isinstance(y, tuple) else y
            return self.norm(y[:, -1])
    return Head()


def head_rollout(model, tok, head, h0r, h0p, seeds, J=16, lam=1.0, plaus=0.1, pos0=0, beta=None, bias=None, forced=None,
                 return_logp=False):
    """CL35 / RK-1 chain: step k feeds emb(x_k) and the previous hidden (real: from h0r, prior: from h0p), contrastive selection as
    CL17; beta (variant beta_N = 1): h0r - h0p added to the real hidden input at every step >= 2. Returns tokens [N, J] and records."""
    torch = T_()
    from transformers import DynamicCache
    dev = RC.model_device(model); N = len(seeds); emb = model.get_input_embeddings(); W = model.get_output_embeddings().weight
    rot = rotary_of(model); spec = specials(tok, W.shape[0], dev); cache = DynamicCache()
    hid = torch.cat([h0r, h0p]).to(dev).float(); x = torch.tensor([int(s) for s in seeds] * 2, device=dev)
    toks = torch.zeros(N, J, dtype=torch.long); rec = {k: [] for k in ("lpr", "lpp", "ent", "top1")}; lps = []
    with torch.no_grad():
        for k in range(J):
            inp = hid.clone()
            if beta is not None and k >= 1:
                inp[:N] = inp[:N] + beta * (h0r.to(dev) - h0p.to(dev))
            pos = torch.full((2 * N, 1), pos0 + k, device=dev, dtype=torch.long)
            out = head.step(emb(x).float(), inp, cache, pos, rot); lg = (out @ W.T.float()).float()
            lr, lp = lg[:N].log_softmax(-1), lg[N:].log_softmax(-1); lrm = lr.masked_fill(spec, float("-inf"))
            s = (lr - lam * lp).masked_fill((lr < lrm.max(-1, keepdim=True).values + math.log(plaus)) | spec, float("-inf"))
            if bias is not None:
                s = s + (bias(k) if callable(bias) else bias)
            xn = s.argmax(-1)
            if forced is not None:
                for i, f in enumerate(forced):
                    if k < len(f):
                        xn[i] = int(f[k])
            toks[:, k] = xn.cpu(); pr = lr.exp()
            rec["lpr"].append(lr.gather(1, xn[:, None])[:, 0].cpu()); rec["lpp"].append(lp.gather(1, xn[:, None])[:, 0].cpu())
            rec["ent"].append(-(pr * lr.nan_to_num(neginf=0.0)).sum(-1).cpu()); rec["top1"].append(pr.max(-1).values.cpu())
            if return_logp:
                lps.append(lr.half().cpu())
            hid = out; x = torch.cat([xn, xn])
    r = {k: torch.stack(v, 1) for k, v in rec.items()}; r["e"] = r["lpr"] - r["lpp"]; r["tok"] = toks
    if return_logp:
        r["logp"] = torch.stack(lps, 1)
    return r


def head_tf_logp(model, head, h0, seeds, chains, pos0=0):
    """Teacher-forced head log-probs along given chains [N, J, V] (real stream only; for the self-consistency KL)."""
    torch = T_()
    from transformers import DynamicCache
    dev = RC.model_device(model); emb = model.get_input_embeddings(); W = model.get_output_embeddings().weight; rot = rotary_of(model)
    cache, hid, out = DynamicCache(), h0.to(dev).float(), []
    x = torch.tensor([int(s) for s in seeds], device=dev)
    with torch.no_grad():
        for k in range(len(chains[0])):
            o = head.step(emb(x).float(), hid, cache, torch.full((len(seeds), 1), pos0 + k, device=dev, dtype=torch.long), rot)
            out.append((o @ W.T.float()).float().log_softmax(-1)); hid = o
            x = torch.tensor([int(c[k]) for c in chains], device=dev)
    return torch.stack(out, 1)

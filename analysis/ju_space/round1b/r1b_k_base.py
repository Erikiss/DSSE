"""Part K base (round 1b): sizes (real / MOCK), the K1 label bank, keyed decoding with truncation, teacher-forced batches, neuron
taps (z, CETT, zCETT) and neuron interventions on the down_proj input, part-steps, and K.4's statistics helpers (item bootstrap,
logistic fits, AUROC, IUT, equivalence, mixture LR, carry). Inverse-free (main §2.5): the only linear solves are named fitting steps
(logistic Newton steps, ridge probes), marked '# §2.5 fitting step'."""
import contextlib, hashlib, json, math, os
import numpy as np
import r1b_core as RC
from r1b_core import coded, val, interval, p_one_sided, p_rank, p_pi, holm, dist, labelled

NEG = -1e30


def N(ctx, real, mock):
    """Unit counts: the plan's value, or the MOCK value (code tests on the CPU)."""
    return mock if ctx.mock else real


def H(s):
    return int(hashlib.sha256(str(s).encode()).hexdigest()[:8], 16)


def lazy(fn, **kw):
    """A Step callable that imports 'module:function' only when the step runs (the registry stays light) and passes kw."""
    def run(ctx):
        import importlib
        m, f = fn.split(":")
        return getattr(importlib.import_module(m), f)(ctx, **kw)
    run.__name__ = fn.replace(":", ".") + ("" if not kw else "(" + ",".join(f"{k}={v}" for k, v in kw.items()) + ")")
    return run


# ---------------------------------------------------------------------------------------------------------------------------------
# K1 bank (package file, frozen before the freeze; MOCK: built under <root>/mock_pkg)
# ---------------------------------------------------------------------------------------------------------------------------------
_BANK = {}


def bank(ctx):
    import r1b_k_labels as KL
    real = os.path.join(RC.REPO, RC.C.PLAN_DIR, "k1_label_bank")
    if os.path.exists(os.path.join(real, "bank.json")):
        d = real
    elif ctx.mock:
        d = os.path.join(ctx.root, "mock_pkg", "k1_label_bank")
        if not os.path.exists(os.path.join(d, "bank.json")):
            tok, _ = ctx.model()
            KL.build_mock_bank(d, tok)
    else:
        raise RC.NotRun("K1 label bank missing (package file k1_label_bank/)")
    if d not in _BANK:
        _BANK[d] = KL.load_bank(d)
    return _BANK[d]


def jl(ctx, model):
    """SO-JLENS through ctx.jlens(); MOCK: maps of the mock model's size when the file's maps do not fit (code tests only)."""
    J = ctx.jlens()
    D = model.config.get_text_config().hidden_size if hasattr(model.config, "get_text_config") else model.config.hidden_size
    if J["maps"].shape[-1] != D:
        if not ctx.mock:
            raise RuntimeError(f"J-lens maps {tuple(J['maps'].shape)} do not fit the model (hidden {D})")
        J = RC.load_jlens(None, mock=True, D=D, n=max(len(RC.layers_of(model)) - 1, 1)); ctx.jlens_sha256 = J["sha256"]
    return J


def ref_k16(ctx):
    p = os.path.join(RC.REPO, RC.C.PLAN_DIR, "ref_k16.json")
    if not os.path.exists(p):
        p = os.path.join(os.path.dirname(bank(ctx)["_dir"]), "ref_k16.json")
    return RC.RefSet.from_json(p, "REF-K16")


def ref_r16(ctx):
    """REF-R16 (account B: r1b_xfer_A; account A: r1b_R_S0's xfer copy). MOCK placeholder -> REF-K16's first 8 items (labelled)."""
    p = None
    for f in (lambda: ctx.xfer("dmg_ref_R16.json"), lambda: ctx.input("r1b_R_S0", "xfer/dmg_ref_R16.json", required=False)):
        try:
            p = f()
        except (FileNotFoundError, RC.NotRun):
            p = None
        if p:
            break
    d = RC.load_json(p) if p else None
    if not d or not d.get("items"):
        if ctx.mock:
            r = ref_k16(ctx); return RC.RefSet(r.items[:8], "REF-R16(MOCK)")
        return None
    return RC.RefSet.from_json(p, "REF-R16")


def ref_u32_damage(ctx, model, setups, name):
    """SO-DMG reporting: damage of calibrated interventions on REF-U32 (descriptive). setups: {label: setup()}."""
    r16 = ref_r16(ctx)
    if r16 is None:
        return {k: coded("NA:not_run:REF-R16 missing") for k in setups}
    dm = RC.Damage(model, r16 + ref_k16(ctx))
    return {k: val(dm(s)["damage"]) for k, s in setups.items()} | {"_refs": dm.refs.name, "_name": name}


# ---------------------------------------------------------------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------------------------------------------------------------
def T_():
    import torch
    return torch


def specials(tok, V, dev):
    """Bool mask [V]: special / added / padding ids (>= the tokenizer's base vocabulary)."""
    torch = T_()
    m = torch.zeros(V, dtype=torch.bool, device=dev); m[int(tok.vocab_size):] = True
    for i in tok.all_special_ids:
        if i < V:
            m[i] = True
    return m


def digit_ids(tok):
    return [tok.convert_tokens_to_ids(str(d)) for d in range(10)]


def lm_parts(model):
    """final norm, unembedding W_U [V, D] and the mean row of W_U."""
    W = model.get_output_embeddings().weight
    base = model.model
    norm = getattr(base, "norm", None) or getattr(getattr(base, "language_model", None), "norm", None)
    return norm, W, W.mean(0)


def unembed(model, h):
    norm, W, _ = lm_parts(model)
    return norm(h.to(W.dtype)) @ W.T


class Gumbel:
    """rk_loop's keyed Gumbel (verbatim source via r1b_core.gumbel_class) for one batch of keys; draw(scores) = scores / T + g."""

    def __init__(self, dev, T, keys):
        self.g = RC.gumbel_class(dev, T)(list(keys))

    def __call__(self, scores):
        return self.g(None, scores)


def decode(model, tok, prompts, keys=None, T=1.0, max_new=24, topk=None, topp=None, batch=32, setup=None, on_fwd=None,
           bias=None, return_lp=False):
    """Keyed Gumbel-max (keys) or greedy (keys None) decoding with a KV cache, left padding and explicit position ids (token-identical
    to rk_loop.generate's rule). Truncation (top-k, then top-p) runs before the keyed draw (K2's Gao protocol). setup(b0, B) -> context
    manager active for the batch (interventions, taps); on_fwd(b0, step) is called after every forward (step 0 = prefill).
    Returns continuations without the stop token (and per-token log-probs / entropies if return_lp)."""
    torch = T_()
    from transformers import DynamicCache
    dev = RC.model_device(model); eos, stop = RC.stop_tokens(tok, model); out, lps = [], []
    stop_t = torch.tensor(sorted(stop), device=dev)
    for b0 in range(0, len(prompts), batch):
        P = prompts[b0:b0 + batch]; B = len(P)
        ids, am, _ = RC.pad_left(P, tok.pad_token_id); ids, am = ids.to(dev), am.to(dev)
        G = Gumbel(dev, T, keys[b0:b0 + B]) if keys is not None else None
        cache, done, conts, lpc = DynamicCache(), torch.zeros(B, dtype=torch.bool, device=dev), [[] for _ in range(B)], [[] for _ in range(B)]
        pos = (am.cumsum(-1) - 1).clamp(min=0); cur = pos[:, -1:].clone()
        with (setup(b0, B) if setup else contextlib.nullcontext()), torch.no_grad():
            o = model(input_ids=ids, attention_mask=am, position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1)
            if on_fwd:
                on_fwd(b0, 0)
            for s in range(max_new):
                lg = o.logits[:, -1].float()
                if bias is not None:
                    lg = lg + bias
                if topk:
                    lg = lg.masked_fill(lg < lg.topk(topk, -1).values[:, -1:], float("-inf"))
                if topp:
                    sp, ix = lg.softmax(-1).sort(-1, descending=True); cs = sp.cumsum(-1)
                    lg = lg.masked_fill(torch.zeros_like(lg, dtype=torch.bool).scatter(1, ix, (cs - sp) > topp), float("-inf"))
                x = (G(lg) if G is not None else lg).argmax(-1)
                if return_lp:
                    l = lg.log_softmax(-1); ent = -(l.exp() * l.nan_to_num(neginf=0.0)).sum(-1); lx = l.gather(1, x[:, None])[:, 0]
                fin = torch.isin(x, stop_t)
                for r in range(B):
                    if not done[r] and not fin[r]:
                        conts[r].append(int(x[r]))
                        if return_lp:
                            lpc[r].append((float(lx[r]), float(ent[r])))
                done |= fin
                if bool(done.all()) or s == max_new - 1:
                    break
                am = torch.cat([am, torch.ones(B, 1, dtype=am.dtype, device=dev)], 1); cur = cur + 1
                o = model(input_ids=torch.where(done, tok.pad_token_id, x)[:, None], attention_mask=am, position_ids=cur,
                          past_key_values=cache, use_cache=True)
                if on_fwd:
                    on_fwd(b0, s + 1)
        out += conts; lps += lpc
    return (out, lps) if return_lp else out


def tf_batches(model, seqs, pad_id, batch=16, keep=None, setup=None, on_fwd=None, logits=True):
    """Teacher-forced left-padded forwards; yields (b0, ids, am, pad, logits or None) per batch. keep: logits_to_keep."""
    torch = T_()
    dev = RC.model_device(model)
    for b0 in range(0, len(seqs), batch):
        ids, am, pad = RC.pad_left(seqs[b0:b0 + batch], pad_id); ids, am = ids.to(dev), am.to(dev)
        pos = (am.cumsum(-1) - 1).clamp(min=0)
        with (setup(b0, ids.shape[0]) if setup else contextlib.nullcontext()), torch.no_grad():
            if logits:
                o = model(input_ids=ids, attention_mask=am, position_ids=pos, **({} if keep is None else dict(logits_to_keep=keep))).logits
            else:
                o = None; RC.Hooks.run(lambda: model.model(input_ids=ids, attention_mask=am, position_ids=pos))
        if on_fwd:
            on_fwd(b0, ids, am, pad)
        yield b0, ids, am, pad, o


# ---------------------------------------------------------------------------------------------------------------------------------
# Neuron taps and interventions on the down_proj input z = SiLU(W_gate x) * W_up x (SO-HN hook manager)
# ---------------------------------------------------------------------------------------------------------------------------------
def wnorms(model):
    """||W_down[:, j]|| per layer [L, n]."""
    torch = T_()
    return torch.stack([l.mlp.down_proj.weight.float().norm(dim=0) for l in RC.layers_of(model)])


class Tap:
    """Forward hooks on every down_proj. fn(layer, z [B, T, n], den [B, T]) is called per forward with den = ||W_down z_t|| (the FFN
    output norm, CETT's denominator); z is the (possibly intervened) input."""

    def __init__(self, model, fn, layers=None):
        self.h = []
        for i, l in enumerate(RC.layers_of(model)):
            if layers is None or i in layers:
                self.h.append(l.mlp.down_proj.register_forward_hook(lambda m, a, o, i=i: fn(i, a[0], o.norm(dim=-1))))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        for h in self.h:
            h.remove()


class Interv:
    """Neuron interventions on z (pre-hooks on down_proj): ops = [dict(sel={layer: idx}, kind 'scale' | 'clamp' | 'add', value
    (scalar or {layer: tensor [n]}), rows (bool [B] or None), when (callable(T, state) -> bool mask [B or 1, T] or None = always))].
    state: dict shared with the caller (e.g. the current chain position)."""

    def __init__(self, model, ops, state=None):
        torch = T_()
        self.h, self.state = [], state if state is not None else {}
        by = {}
        for k, op in enumerate(ops):
            for l, idx in op["sel"].items():
                by.setdefault(int(l), []).append((op, torch.as_tensor(idx, dtype=torch.long)))
        L = RC.layers_of(model)
        for l, lst in by.items():
            def pre(m, args, lst=lst, l=l):
                x = args[0]; x = x.clone()
                for op, ix in lst:
                    i = ix.to(x.device); sel = x[..., i]; v = op.get("value", 1.0)
                    v = (v[l] if isinstance(v, dict) else torch.as_tensor(v)).to(x.device, x.dtype)
                    new = {"scale": lambda: sel * v, "clamp": lambda: v.expand_as(sel), "add": lambda: sel + v}[op["kind"]]()
                    m_ = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
                    if op.get("rows") is not None:
                        m_ = m_ & op["rows"].to(x.device)[:, None]
                    if op.get("when") is not None:
                        w = op["when"](x.shape[1], self.state)
                        if w is None:
                            continue
                        m_ = m_ & torch.as_tensor(w, device=x.device).bool()
                    x[..., i] = torch.where(m_[..., None], new, sel)
                return (x,) + tuple(args[1:])
            self.h.append(L[l].mlp.down_proj.register_forward_pre_hook(pre))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        for h in self.h:
            h.remove()


def by_layer(neurons):
    """[[layer, index], ...] -> {layer: [index, ...]}."""
    d = {}
    for l, j in neurons:
        d.setdefault(int(l), []).append(int(j))
    return d


def set_values(neurons, arr):
    """{layer: tensor} of per-neuron values from an [L, n] array (e.g. the REF-W mean of z: the mean-clamp value)."""
    torch = T_()
    return {l: torch.as_tensor(np.asarray(arr)[l, ix], dtype=torch.float32) for l, ix in by_layer(neurons).items()}


class SetReader:
    """zCETT of named neuron sets at the last position (or all positions) of each forward: a_X = mean over X of
    (CETT - mu) / sd with REF-W's per-neuron CETT mean / SD; also raw z of the sets (pole-separate reads)."""

    def __init__(self, sets, wn, ref_mu, ref_sd, last=True):
        torch = T_()
        self.sets, self.last, self.buf = sets, last, {}
        u = {}
        for name, nl in sets.items():
            for l, ix in by_layer(nl).items():
                u.setdefault(l, set()).update(ix)
        self.u = {l: torch.tensor(sorted(s)) for l, s in u.items()}
        self.pos = {l: {int(j): k for k, j in enumerate(self.u[l].tolist())} for l in self.u}
        self.wn = {l: wn[l, self.u[l]].float() for l in self.u}
        self.mu = {l: torch.as_tensor(np.asarray(ref_mu)[l, self.u[l].numpy()]).float() for l in self.u}
        self.sd = {l: torch.as_tensor(np.asarray(ref_sd)[l, self.u[l].numpy()]).float().clamp(min=1e-6) for l in self.u}
        self.layers = sorted(self.u)

    def __call__(self, l, z, den):
        if l not in self.u:
            return
        ix = self.u[l].to(z.device)
        zz = z[:, -1:, ix] if self.last else z[..., ix]; dd = den[:, -1:] if self.last else den
        c = zz.abs().float() * self.wn[l].to(z.device) / dd.float().clamp(min=1e-9)[..., None]
        self.buf[l] = ((c - self.mu[l].to(z.device)) / self.sd[l].to(z.device), zz.float())

    def aX(self, name):
        """[B, T'] mean zCETT over the set (from the last forward)."""
        torch = T_()
        acc, n = 0, 0
        for l, ix in by_layer(self.sets[name]).items():
            if l in self.buf:
                k = torch.tensor([self.pos[l][j] for j in ix], device=self.buf[l][0].device)
                acc = acc + self.buf[l][0][..., k].sum(-1); n += len(ix)
        return acc / max(n, 1) if n else None

    def zcols(self, name):
        torch = T_()
        out = []
        for l, ix in by_layer(self.sets[name]).items():
            if l in self.buf:
                out.append(self.buf[l][1][..., torch.tensor([self.pos[l][j] for j in ix], device=self.buf[l][1].device)])
        return torch.cat(out, -1) if out else None


# ---------------------------------------------------------------------------------------------------------------------------------
# Statistics (K.4)
# ---------------------------------------------------------------------------------------------------------------------------------
SEED_BOOT = 1301


def auroc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float); ok = np.isfinite(s)
    y, s = y[ok], s[ok]; n1, n0 = (y == 1).sum(), (y == 0).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logit_fit(X, y, l2=1e-4, w=None, iters=50, std=True, C=None):
    """L2 logistic by Newton steps (# §2.5 fitting step: a classifier / regression fit, not a reconstruction). C given: sklearn's
    convention C * sum(loss) + 0.5 |beta|^2 (intercept unpenalised); else ridge l2 per row. Returns dict(beta, b, mu, sd, ok)."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    if X.ndim == 1:
        X = X[:, None]
    n, d = X.shape
    mu = X.mean(0) if std else np.zeros(d); sd = X.std(0) if std else np.ones(d); sd[sd < 1e-12] = 1.0
    Z = np.c_[np.ones(n), (X - mu) / sd]; w = np.ones(n) if w is None else np.asarray(w, float)
    lam = np.r_[0.0, np.full(d, (1.0 / C) if C else l2 * n)]; beta = np.zeros(d + 1); ok = True
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ beta, -30, 30)))
        g = Z.T @ (w * (p - y)) * (C or 1.0) + lam * beta
        Hm = (Z * (w * p * (1 - p) * (C or 1.0))[:, None]).T @ Z + np.diag(lam + 1e-9)
        try:
            step = np.linalg.solve(Hm, g)            # §2.5 fitting step (Newton step of a logistic fit)
        except np.linalg.LinAlgError:
            ok = False; break
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return dict(beta=beta[1:] / sd, b=float(beta[0] - (beta[1:] / sd) @ mu), mu=mu, sd=sd, ok=ok and bool(np.isfinite(beta).all()))


def logit_score(fit, X):
    X = np.asarray(X, float)
    return (X if X.ndim > 1 else X[:, None]) @ fit["beta"] + fit["b"]


def crossfit_auroc(X, y, folds, C=1.0):
    """Cross-fitted AUROC of an L2 logistic (5 folds given as fold labels)."""
    s = np.full(len(y), np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if len(np.unique(y[tr])) < 2:
            continue
        s[te] = logit_score(logit_fit(X[tr], y[tr], C=C), X[te])
    return s


def boot_stat(n, fn, B=RC.N_BOOT, seed=SEED_BOOT, groups=None):
    """Unit bootstrap of fn(idx) -> float (or array); groups: cluster labels (units = clusters)."""
    g = np.random.default_rng(seed)
    if groups is None:
        return np.array([fn(g.integers(0, n, n)) for _ in range(B)], dtype=float)
    groups = np.asarray(groups); ug = np.unique(groups); members = {u: np.where(groups == u)[0] for u in ug}
    out = []
    for _ in range(B):
        pick = g.integers(0, len(ug), len(ug))
        out.append(fn(np.concatenate([members[ug[k]] for k in pick])))
    return np.array(out, dtype=float)


def ci(draws):
    lo, hi = interval(draws)
    return dict(lo=lo, hi=hi)


def fin(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def inside(lohi, m):
    lo, hi = lohi["lo"], lohi["hi"]
    return fin(lo) and fin(hi) and lo >= -m and hi <= m


def gt(lohi, t=0.0):
    return fin(lohi["lo"]) and lohi["lo"] > t


def lt(lohi, t=0.0):
    return fin(lohi["hi"]) and lohi["hi"] < t


def p_iut(*ps):
    """p_IUT = max of the component p (coded components count as 1)."""
    return float(max(p if fin(p) else 1.0 for p in ps))


def holm_family(members):
    """members: {name: p}; returns {name: dict(p, p_holm, reject)} (Holm at 0.05)."""
    ks = list(members); ps = [members[k] if fin(members[k]) else 1.0 for k in ks]
    if not ks:
        return {}
    adj, rej = holm(ps)
    return {k: dict(p=ps[i], p_holm=adj[i], reject=bool(rej[i])) for i, k in enumerate(ks)}


def mixture_lr(z, items, B=RC.N_BOOT, seed=SEED_BOOT):
    """K6e-b / K8b bistability: (1 - pi) N(0, 1) + pi N(Delta, s1^2) against a monostable N(delta, s^2) by EM; LR = 2 (ll_b - ll_m);
    p from B parametric draws of the fitted monostable model with its between- and within-item variance. z: the DIVERGED units only
    (undiverged CRN pairs, z = 0 exactly, are excluded and counted by the caller: finding validity-3 / stats-2, K.12)."""
    from scipy.stats import norm
    z = np.asarray(z, float); items = np.asarray(items)
    if z.size < 10:
        return dict(LR=coded("NaN:too_few_diverged_units"), p=coded("NaN:too_few_diverged_units"), n=int(z.size))

    def fit(x):
        d, s = x.mean(), max(x.std(), 1e-3); llm = norm.logpdf(x, d, s).sum()
        pi, D, s1 = 0.2, max(np.percentile(x, 90), 1.0), 1.0
        for _ in range(200):
            a = (1 - pi) * norm.pdf(x, 0, 1); b = pi * norm.pdf(x, D, s1); r = b / np.maximum(a + b, 1e-300)
            pi = float(np.clip(r.mean(), 1e-3, 1 - 1e-3)); D = float((r * x).sum() / max(r.sum(), 1e-9))
            s1 = float(np.clip(np.sqrt((r * (x - D) ** 2).sum() / max(r.sum(), 1e-9)), 0.05, 50))
        llb = np.log(np.maximum((1 - pi) * norm.pdf(x, 0, 1) + pi * norm.pdf(x, D, s1), 1e-300)).sum()
        return float(2 * (llb - llm)), dict(pi=pi, Delta=D, s1=s1, delta=float(d), s=float(s))
    LR, par = fit(z)
    ui = np.unique(items); mi = np.array([z[items == u].mean() for u in ui]); vb = max(mi.var() - z.var() / max(len(z) / len(ui), 1), 0)
    vw = max(z.var() - vb, 1e-6); g = np.random.default_rng(seed); cnt = 0
    for _ in range(B):
        im = g.normal(par["delta"], math.sqrt(vb), len(ui)); xs = np.concatenate([g.normal(im[k], math.sqrt(vw), (items == u).sum()) for k, u in enumerate(ui)])
        cnt += fit(xs)[0] >= LR
    return dict(LR=val(LR, "NaN:LR_undefined"), p=(1 + cnt) / (B + 1), n=int(z.size), **par)


def robust_tail_share(z, items, thr=2.0, B=RC.N_BOOT, seed=SEED_BOOT):
    """'no within-item bistability' statistic (finding validity-4, K.12): share of units whose robustly standardised value
    (z - median) / (1.4826 MAD) exceeds thr, with its item-bootstrap interval (a monostable shift or scale no longer counts)."""
    z = np.asarray(z, float)

    def st(ix):
        x = z[ix]
        if x.size < 5:
            return np.nan
        m = np.median(x); s = 1.4826 * np.median(np.abs(x - m))
        return float(((x - m) / max(s, 1e-9) > thr).mean())
    return dict(share=val(st(np.arange(z.size))), ci=ci(boot_stat(z.size, st, B, seed, groups=items)))


def ratio_carry(num, den, units, B=RC.N_BOOT, seed=SEED_BOOT):
    """R6K carry K = sum(num) / sum(den) over units, defined iff den's 90 % interval excludes 0; log K draws for the PI test."""
    num = np.asarray(num, float); den = np.asarray(den, float)
    dd = boot_stat(len(den), lambda ix: den[ix].mean(), B, seed, groups=units)
    kd = boot_stat(len(den), lambda ix: num[ix].sum() / den[ix].sum() if den[ix].sum() != 0 else np.nan, B, seed, groups=units)
    dci = ci(dd); defined = gt(dci) or lt(dci)
    K = num.sum() / den.sum() if den.sum() != 0 else float("nan")
    return dict(K=val(K), defined=bool(defined), den_ci=dci, K_ci=ci(kd), p_le1=p_one_sided(kd, 1.0, "greater"),
                p_ge1=p_one_sided(kd, 1.0, "less"), logK=val(math.log(K)) if fin(K) and K > 0 else coded("NaN:K_not_positive"))


def write(ctx, rel, obj, config=None, labels=(), seeds=None):
    return ctx.write_json(rel, RC.jsonable(obj), config=config, seeds=seeds, labels=list(labels) + (["MOCK"] if ctx.mock else []))


def save(ctx, rel, obj, config=None, labels=()):
    return ctx.save_pt(rel, obj, config=config, labels=list(labels) + (["MOCK"] if ctx.mock else []))


def load_pt(p):
    return T_().load(p, map_location="cpu", weights_only=False)


def own(ctx, rel):
    """A file written earlier in this session (path or None)."""
    p = os.path.join(ctx.work, rel)
    return p if os.path.exists(p) else None


def up(ctx, session, rel, required=True):
    """A file of an upstream session (copied input) or, if the session is this one, the own file."""
    return own(ctx, rel) if session == ctx.session else ctx.input(session, rel, required=required)


def parts(units, i, n):
    return units[i::n]


def merge_parts(ctx, meas, n):
    out = []
    for i in range(n):
        p = own(ctx, f"k/{meas}/part{i}.pt")
        if p is None:
            raise RuntimeError(f"{meas}: part {i} missing")
        out += load_pt(p)["rows"]
    return out

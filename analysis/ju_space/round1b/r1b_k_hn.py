"""SO-HN (part K, K2): REF-W per-neuron statistics (z and CETT), 99.8th WikiText percentiles (both poles), mu_gen and Sigma_l, CETT
answer / other / pre-generation features, the GPU FISTA L1-logistic selector select(X, y, C, exclude_mask), L2 detectors, the
random-set generator (same count per layer, mean answer-CETT within +-20 %) and damage matching. Importable by part L (r1b_hn.py)."""
import math
import numpy as np
import r1b_core as RC
from r1b_k_base import T_, Tap, Interv, SetReader, wnorms, by_layer, set_values, logit_fit, logit_score, auroc, N


def nlayers(model):
    return len(RC.layers_of(model))


def refw_stats(model, seqs, lo=16, hi=126, mu_blocks=True, batch=16):
    """REF-W: per-neuron mean / SD of z and CETT over positions lo..hi of 128-token windows; mu_gen[l] (mean block-l output)."""
    torch = T_()
    dev = RC.model_device(model); wn = wnorms(model).to(dev); L = wn.shape[0]
    acc = {k: torch.zeros_like(wn, dtype=torch.float64) for k in ("z", "z2", "c", "c2")}; cnt = [0]
    D = model.config.get_text_config().hidden_size if hasattr(model.config, "get_text_config") else model.config.hidden_size
    mus = torch.zeros(L, D, dtype=torch.float64, device=dev)

    def fn(l, z, den):
        zz = z[:, lo:hi + 1].double(); c = zz.abs() * wn[l].double() / den[:, lo:hi + 1, None].double().clamp(min=1e-9)
        acc["z"][l] += zz.sum((0, 1)); acc["z2"][l] += (zz ** 2).sum((0, 1)); acc["c"][l] += c.sum((0, 1)); acc["c2"][l] += (c ** 2).sum((0, 1))
        if l == 0:
            cnt[0] += zz.shape[0] * zz.shape[1]
    with Tap(model, fn), RC.Hooks(model) as hk, torch.no_grad():
        hk.capture(list(range(L)), cpu=False)
        for i in range(0, len(seqs), batch):
            ids = torch.tensor(seqs[i:i + batch], device=dev); model.model(input_ids=ids)
            for l in range(L):
                mus[l] += hk.states[l][-1][:, lo:hi + 1].double().sum((0, 1))
            for l in range(L):
                hk.states[l].clear()
    n = cnt[0]; out = {}
    for k in ("z", "c"):
        m = acc[k] / n; out[f"{k}_mu"] = m.cpu().numpy(); out[f"{k}_sd"] = (acc[k + "2"] / n - m ** 2).clamp(min=0).sqrt().cpu().numpy()
    out["mu_gen"] = (mus / n).float().cpu(); out["n_tokens"] = n
    return out


def percentiles(model, ids, T=128, batch=64, q=0.998, sig_blocks=(10, 14, 18, 22)):
    """99.8th percentile of z (positive pole) and of -z (negative pole) per neuron over the reference tokens (exact, running top-k per
    layer), and Sigma_l (covariance of block-l outputs, float64 accumulators) at sig_blocks."""
    torch = T_()
    dev = RC.model_device(model); L = nlayers(model)
    n_tok = (len(ids) // T) * T; K = int(math.ceil((1 - q) * n_tok)) + 1
    top = {}; S = {}

    def fn(l, z, den):
        x = z.reshape(-1, z.shape[-1]).float().T          # [n, tokens]
        for pole, v in (("pos", x), ("neg", -x)):
            cur = top.get((l, pole))
            v = v if cur is None else torch.cat([cur, v], 1)
            top[(l, pole)] = v.topk(min(K, v.shape[1]), 1).values
    with Tap(model, fn), RC.Hooks(model) as hk, torch.no_grad():
        hk.capture(list(sig_blocks), cpu=False)
        arr = torch.tensor(np.asarray(ids[:n_tok]).reshape(-1, T), device=dev)
        for i in range(0, arr.shape[0], batch):
            model.model(input_ids=arr[i:i + batch])
            for b in sig_blocks:
                h = hk.states[b].pop(); h = h.reshape(-1, h.shape[-1]).double()
                s = S.setdefault(b, [torch.zeros(h.shape[1], dtype=torch.float64, device=dev),
                                     torch.zeros(h.shape[1], h.shape[1], dtype=torch.float64, device=dev), 0])
                s[0] += h.sum(0); s[1] += h.T @ h; s[2] += h.shape[0]
    p = {pole: np.stack([top[(l, pole)][:, -1].cpu().numpy() * (1 if pole == "pos" else -1) for l in range(L)]) for pole in ("pos", "neg")}
    sig = {b: ((s[1] / s[2]) - torch.outer(s[0] / s[2], s[0] / s[2])).float().cpu() for b, s in S.items()}
    mean = {b: (s[0] / s[2]).float().cpu() for b, s in S.items()}
    return dict(p998_pos=p["pos"], p998_neg=p["neg"], n_tokens=n_tok, sigma=sig, sigma_mean=mean)


def sqrtm_psd(S):
    """Sigma^{1/2} by eigendecomposition (# §2.5: a fitting step for V7's null generation, not an inversion)."""
    torch = T_()
    w, V = torch.linalg.eigh(S.double())
    return (V * w.clamp(min=0).sqrt()) @ V.T


def spans(tok, prompt_ids):
    """(first, end) of the user content in a chat prompt (template tokens excluded)."""
    a = RC.user_start(tok); suf = len(RC.chat_ids(tok, "x")) - a - len(RC.enc(tok, "x"))
    return a, len(prompt_ids) - suf


def cett_features(model, tok, rows, batch=16, sets=None):
    """rows: (prompt_ids, cont_ids, n_answer_tokens). Returns ans, oth, pre: [n, L * n_neur] float16 CPU (CETT means over the answer
    span, over user-content prompt tokens and non-answer response tokens (positions >= 16, template tokens excluded), and at the last
    prompt token). sets: only these neurons (dict layer -> idx) -> smaller columns in the same layer-major order."""
    torch = T_()
    dev = RC.model_device(model); wn = wnorms(model).to(dev); L = wn.shape[0]
    out = {k: [] for k in ("ans", "oth", "pre")}
    for i in range(0, len(rows), batch):
        rb = rows[i:i + batch]; seqs = [p + c for p, c, _ in rb]
        ids, am, pad = RC.pad_left(seqs, tok.pad_token_id); T = ids.shape[1]
        ma = torch.zeros(len(rb), T, device=dev); mo = torch.zeros_like(ma); last = torch.zeros(len(rb), dtype=torch.long, device=dev)
        for r, (p, c, na) in enumerate(rb):
            o = int(pad[r]); a, e = spans(tok, p)
            ma[r, o + len(p):o + len(p) + na] = 1
            for t in list(range(max(a, 16), e)) + list(range(len(p) + na, len(p) + len(c))):
                if t >= 16:
                    mo[r, o + t] = 1
            last[r] = o + len(p) - 1
        buf = {k: [None] * L for k in out}

        def fn(l, z, den):
            ix = None if sets is None else torch.as_tensor(sets.get(l, []), dtype=torch.long, device=z.device)
            zz = z if ix is None else z[..., ix]; w_ = wn[l] if ix is None else wn[l][ix]
            c = zz.abs().float() * w_ / den.float().clamp(min=1e-9)[..., None]
            buf["ans"][l] = ((c * ma[..., None]).sum(1) / ma.sum(1, keepdim=True).clamp(min=1)).half().cpu()
            buf["oth"][l] = ((c * mo[..., None]).sum(1) / mo.sum(1, keepdim=True).clamp(min=1)).half().cpu()
            buf["pre"][l] = c[torch.arange(c.shape[0], device=c.device), last].half().cpu()
        with Tap(model, fn), torch.no_grad():
            model.model(input_ids=ids.to(dev), attention_mask=am.to(dev), position_ids=(am.cumsum(-1) - 1).clamp(min=0).to(dev))
        for k in out:
            out[k].append(torch.cat([b for b in buf[k] if b is not None and b.shape[-1]], -1))
    return {k: torch.cat(v, 0) for k, v in out.items()}


# ---------------------------------------------------------------------------------------------------------------------------------
# L1 logistic (GPU FISTA) and the selector
# ---------------------------------------------------------------------------------------------------------------------------------
def _lip(X, C, it=30):
    torch = T_()
    v = torch.randn(X.shape[1] + 1, device=X.device, generator=torch.Generator(device=X.device).manual_seed(0)); v /= v.norm()
    for _ in range(it):
        u = X @ v[:-1] + v[-1]; v = torch.cat([X.T @ u, u.sum()[None]]); s = v.norm(); v = v / s
    return 0.25 * C * float(s) + 1e-6


def fista(X, y, C, w0=None, b0=None, iters=800, tol=1e-7, exclude=None):
    """min_w,b  C * sum_i log(1 + exp(-s_i (x_i w + b))) + ||w||_1, intercept unpenalised (sklearn's convention), fp32 on X's device.
    exclude: bool [d] mask of features forced to 0."""
    torch = T_()
    s = (2 * y.float() - 1); L = _lip(X, C)
    w = torch.zeros(X.shape[1], device=X.device) if w0 is None else w0.clone(); b = torch.zeros((), device=X.device) if b0 is None else b0.clone()
    zw, zb, t = w.clone(), b.clone(), 1.0
    for it in range(iters):
        m = X @ zw + zb; g = -s * torch.sigmoid(-s * m) * C
        gw, gb = X.T @ g, g.sum()
        wn = zw - gw / L; wn = torch.sign(wn) * (wn.abs() - 1.0 / L).clamp(min=0)
        if exclude is not None:
            wn = wn.masked_fill(exclude, 0.0)
        bn = zb - gb / L
        tn = (1 + math.sqrt(1 + 4 * t * t)) / 2
        zw = wn + (t - 1) / tn * (wn - w); zb = bn + (t - 1) / tn * (bn - b)
        done = float((wn - w).abs().max()) < tol * max(1.0, float(wn.abs().max()))
        w, b, t = wn, bn, tn
        if done and it > 10:
            break
    return w, b


def objective(X, y, C, w, b):
    torch = T_()
    s = 2 * y.float() - 1
    return float(C * torch.nn.functional.softplus(-s * (X @ w + b)).sum() + w.abs().sum())


def c0(X, y):
    """Smallest C with >= 1 non-zero weight (the intercept-only fit is optimal iff C ||X^T (y - p0)||_inf <= 1)."""
    p0 = y.float().mean()
    return 1.0001 / float((X.T @ (y.float() - p0)).abs().max().clamp(min=1e-12))


def select(X, y, C, exclude_mask=None, **kw):
    """SO-HN selector (importable, part L): the FISTA fit at C; returns (theta [d], intercept)."""
    return fista(X, y, C, exclude=exclude_mask, **kw)


def fista_check(X, y, C):
    """FISTA against sklearn liblinear (intercept_scaling 1e4 = practically unpenalised) on <= 2,000 features: relative objective
    difference (code check, tolerance 1e-4)."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return dict(ok=None, note="sklearn missing")
    torch = T_()
    Xs = X[:, :2000]; w, b = fista(Xs, y, C, iters=5000, tol=1e-10)
    m = LogisticRegression(penalty="l1", C=C, solver="liblinear", intercept_scaling=1e4, tol=1e-10, max_iter=10000)
    m.fit(Xs.cpu().double().numpy(), y.cpu().numpy())
    ws = torch.tensor(m.coef_[0], dtype=torch.float32, device=X.device); bs = torch.tensor(float(m.intercept_[0]), device=X.device)
    f1, f2 = objective(Xs, y, C, w, b), objective(Xs, y, C, ws, bs)
    r = abs(f1 - f2) / max(abs(f2), 1e-9)
    return dict(ok=bool(r <= 1e-4 or f1 <= f2), rel_obj_diff=r, fista=f1, sklearn=f2)


def neurons_of(w, n_per_layer, sign=1):
    """Feature indices with sign * w > 0 -> [[layer, index], ...] (layer-major columns)."""
    ix = ((w * sign) > 0).nonzero().flatten().tolist()
    return [[int(i // n_per_layer), int(i % n_per_layer)] for i in ix]


def cols_of(neurons, n_per_layer):
    return [l * n_per_layer + j for l, j in neurons]


def jaccard(a, b):
    a, b = {tuple(x) for x in a}, {tuple(x) for x in b}
    return len(a & b) / max(len(a | b), 1)


def l2_detector(Xtr, ytr, C=1.0):
    """K3a's refit detector: L2 logistic, C = 1, standardised features (# §2.5 fitting step)."""
    return logit_fit(np.asarray(Xtr, float), np.asarray(ytr, float), C=C)


def l2_full(X, y, lam=1e-3, iters=300, lr=0.5):
    """Dg3: L2 logistic on all features (GPU gradient descent on standardised columns); returns weights [d]."""
    torch = T_()
    mu, sd = X.mean(0), X.std(0).clamp(min=1e-6); Z = (X - mu) / sd; w = torch.zeros(X.shape[1], device=X.device); b = torch.zeros((), device=X.device)
    yf = y.float()
    for _ in range(iters):
        p = torch.sigmoid(Z @ w + b); g = Z.T @ (p - yf) / len(yf) + lam * w; w -= lr * g; b -= lr * (p - yf).mean()
    return w / sd


# ---------------------------------------------------------------------------------------------------------------------------------
# Random sets and damage matching
# ---------------------------------------------------------------------------------------------------------------------------------
def rand_sets(target, cett_mean, n_per_layer, n_valid=400, seed_base=31, tol=0.2, max_draws=20000, exclude=()):
    """Candidates k = 0, 1, ...: torch.Generator(seed_base + k), the same count per layer as target, mean answer-CETT within +-tol of
    the target's; returns the first n_valid valid candidates (as [[layer, index], ...] with their draw number)."""
    torch = T_()
    cm = np.asarray(cett_mean).reshape(-1, n_per_layer); want = np.mean([cm[l, j] for l, j in target]) if target else 0.0
    bl = by_layer(target); ex = {tuple(x) for x in target} | {tuple(x) for x in exclude}; out = []
    for k in range(max_draws):
        g = torch.Generator().manual_seed(seed_base + k); s = []
        for l, ix in bl.items():
            perm = torch.randperm(n_per_layer, generator=g).tolist()
            s += [[l, j] for j in [p for p in perm if (l, p) not in ex][:len(ix)]]
        m = np.mean([cm[l, j] for l, j in s]) if s else 0.0
        if abs(m - want) <= tol * abs(want) + 1e-12:
            out.append(dict(k=k, neurons=s, cett=float(m)))
            if len(out) >= n_valid:
                break
    return out


def dmg_scale(dm, model, neurons, alpha):
    return dm(lambda: Interv(model, [dict(sel=by_layer(neurons), kind="scale", value=float(alpha))]))["damage"]


def dmg_mclamp(dm, model, neurons, zmu):
    return dm(lambda: Interv(model, [dict(sel=by_layer(neurons), kind="clamp", value=set_values(neurons, zmu))]))["damage"]


def match_sets(model, dm, Hset, cands, zmu, n_need=39, kinds=("scale", "mclamp")):
    """RS39 (scaling damage matched at alpha 0 and 3: |d - d_H| <= max(0.3 d_H, 0.01)) and RS39mc (mean-clamp damage matched), first
    n_need among the candidates, with the fallbacks of K2 ('scale-matched random sets', 'loosely matched')."""
    dH = dict(a0=dmg_scale(dm, model, Hset, 0.0), a3=dmg_scale(dm, model, Hset, 3.0), mc=dmg_mclamp(dm, model, Hset, zmu))
    ok = lambda d, ref, w=0.3: abs(d - ref) <= max(w * ref, 0.01)
    rs, rsmc, tried = [], [], []
    for c in cands:
        if len(rs) >= n_need and len(rsmc) >= n_need:
            break
        d = dict(a0=dmg_scale(dm, model, c["neurons"], 0.0), a3=dmg_scale(dm, model, c["neurons"], 3.0),
                 mc=dmg_mclamp(dm, model, c["neurons"], zmu))
        tried.append(dict(k=c["k"], **d))
        if len(rs) < n_need and ok(d["a0"], dH["a0"]) and ok(d["a3"], dH["a3"]):
            rs.append(dict(c, damage=d))
        if len(rsmc) < n_need and ok(d["mc"], dH["mc"]):
            rsmc.append(dict(c, damage=d))
    lab = {"RS39": [], "RS39mc": []}
    if len(rs) < n_need:                      # scale-matched fallback: each candidate's gain calibrated to H's damage at alpha 3
        lab["RS39"].append("scale-matched random sets")
        rs = []
        for c in cands[:n_need]:
            res = RC.calibrate(lambda a: dmg_scale(dm, model, c["neurons"], a), 3.0, dH["a3"], form="gain")
            rs.append(dict(c, damage=dict(a3=res["damage"]), gain3=res["f"], cal_labels=res["labels"]))
    if len(rsmc) < n_need:
        lab["RS39mc"].append("loosely matched")
        for t in tried:
            if len(rsmc) >= n_need:
                break
            if ok(t["mc"], dH["mc"], 0.6) and t["k"] not in {x["k"] for x in rsmc}:
                rsmc.append(dict(next(c for c in cands if c["k"] == t["k"]), damage=t))
    return dict(H_damage=dH, RS39=rs, RS39mc=rsmc, labels=lab, tried=len(tried))

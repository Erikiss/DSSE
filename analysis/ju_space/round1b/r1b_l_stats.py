"""Part L estimators (plan L.2-L.4): pure numpy / torch, no model. Inverse-free (L.0): forward regressions fitted by L-BFGS, cov/var
residual slopes (Gram-Schmidt), 1-D grid searches, power iteration, nearest-neighbour cross-maps, counts; no matrix inversion anywhere.
Used by the session steps (r1b_l_a / r1b_l_b), L-ANA (r1b_l_ana) and the known-answer calibration (r1b_l_cal)."""
import math
import numpy as np

LN2 = math.log(2.0)


def _t():
    import torch
    return torch


def dev_of(x):
    return x.device if hasattr(x, "device") else "cpu"


# ==================================================================================================================================
# bootstrap summaries (SO-PVAL conventions via r1b_core)
# ==================================================================================================================================
def boot_counts(n, B=2000, seed=0):
    """[B, n] multiplicities of a unit bootstrap (r1b_core.boot_idx)."""
    from r1b_core import boot_idx
    idx = boot_idx(n, B=B, seed=seed)
    out = np.zeros((B, n), dtype=np.float64)
    np.add.at(out, (np.repeat(np.arange(B), n), idx.ravel()), 1.0)
    return out


def summ(pt, draws, margin=None):
    """Point, 90 % interval, one-sided p for > 0 and < 0 (SO-PVAL), share of undefined replicates; 'inside' if a margin is given."""
    from r1b_core import interval, p_one_sided, val
    d = np.asarray(draws, dtype=float).ravel()
    lo, hi = interval(d)
    out = dict(pt=val(pt), lo=lo, hi=hi, p_gt0=p_one_sided(d, 0.0, "greater"), p_lt0=p_one_sided(d, 0.0, "less"),
               undefined_share=float((~np.isfinite(d)).mean()) if d.size else 1.0, se=val(np.nanstd(d)) if d.size else val(None))
    if margin is not None:
        out["margin"] = margin
        out["inside"] = bool(isinstance(lo, float) and isinstance(hi, float) and lo > -margin and hi < margin)
    return out


def lo(s):
    v = s.get("lo") if isinstance(s, dict) else None
    return v if isinstance(v, float) else float("nan")


def hi(s):
    v = s.get("hi") if isinstance(s, dict) else None
    return v if isinstance(v, float) else float("nan")


def pt(s):
    v = s.get("pt") if isinstance(s, dict) else s
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else float("nan")


# ==================================================================================================================================
# batched L2 logistic regression (L-BFGS) and the information estimate I-hat (L1)
# ==================================================================================================================================
def logreg(X, y, w, lam, iters=60, W0=None):
    """Batched L2 logistic: X [P, n, m], y [P, n] in {0,1}, w [P, n] training weights, lam [P] penalty on W (not on b).
    Objective per problem: sum_i w_i loss_i / sum_i w_i + lam ||W||^2 / 2. Returns W [P, m], b [P]."""
    torch = _t()
    P, n, m = X.shape
    W = torch.zeros(P, m, dtype=X.dtype, device=X.device) if W0 is None else W0.clone()
    b = torch.zeros(P, dtype=X.dtype, device=X.device)
    W.requires_grad_(True); b.requires_grad_(True)
    ws = w.sum(1).clamp_min(1e-9)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, tolerance_grad=1e-7, tolerance_change=1e-10, history_size=10,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = torch.einsum("pnm,pm->pn", X, W) + b[:, None]
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none") * w).sum(1) / ws
        tot = (loss + 0.5 * lam * (W * W).sum(1)).sum()
        tot.backward()
        return tot
    with torch.enable_grad():
        opt.step(closure)
    return W.detach(), b.detach()


def _std(X, tr):
    """z-score X [P, n, m] with the mean / SD of the training rows tr [n] bool."""
    Xt = X[:, tr]
    mu = Xt.mean(1, keepdim=True); sd = Xt.std(1, keepdim=True).clamp_min(1e-6)
    return (X - mu) / sd


def ihat_rows(X, y, folds, Cs=(1e-3, 1e-2, 1e-1, 1.0), inner=3, iters=60):
    """Held-out log-loss in bits per row [P, n] (L1's I-hat = 1 - mean): X [P, n, m] (torch), y [n] or [P, n] (0/1), folds [n] int
    (by question). Per outer fold: z-score on the training fold, C by inner 3-fold CV from Cs, refit, score the held-out rows."""
    torch = _t()
    P, n, m = X.shape
    Y = (y if y.dim() == 2 else y[None].expand(P, n)).to(X.dtype)
    out = torch.zeros(P, n, dtype=X.dtype, device=X.device)
    nf = int(folds.max()) + 1
    nC = len(Cs)
    for f in range(nf):
        te = folds == f; tr = ~te
        Xs = _std(X, tr)
        tri = torch.nonzero(tr).flatten()
        inn = torch.arange(tri.numel(), device=X.device) % inner
        cv = torch.zeros(P, nC, dtype=X.dtype, device=X.device)
        Xr = Xs.repeat_interleave(nC, 0); Yr = Y.repeat_interleave(nC, 0)
        for g in range(inner):
            wt = torch.zeros(P * nC, n, dtype=X.dtype, device=X.device)
            wt[:, tri[inn != g]] = 1.0
            ntr = float((inn != g).sum())
            lam = torch.tensor([1.0 / (C * ntr) for C in Cs] * P, dtype=X.dtype, device=X.device)
            W, b = logreg(Xr, Yr, wt, lam, iters)
            va = tri[inn == g]
            z = torch.einsum("pnm,pm->pn", Xr[:, va], W) + b[:, None]
            ll = torch.nn.functional.binary_cross_entropy_with_logits(z, Yr[:, va], reduction="none").mean(1)
            cv += ll.view(P, nC)
        best = cv.argmin(1)
        ntr = float(tr.sum())
        lam = torch.tensor([1.0 / (Cs[int(k)] * ntr) for k in best], dtype=X.dtype, device=X.device)
        W, b = logreg(Xs, Y, tr[None].expand(P, n).to(X.dtype), lam, iters)
        z = torch.einsum("pnm,pm->pn", Xs[:, te], W) + b[:, None]
        out[:, te] = torch.nn.functional.binary_cross_entropy_with_logits(z, Y[:, te], reduction="none") / LN2
    return out


def ihat(loss_rows):
    """I-hat = 1 bit - mean held-out log-loss (bits); loss_rows [..., n]."""
    return 1.0 - loss_rows.mean(-1)


def folds_by_question(n, k=5, seed=0):
    g = np.random.default_rng(seed)
    f = np.arange(n) % k
    g.shuffle(f)
    return f


def balance(y, seed=0):
    """Indices of a class-balanced subsample (the larger class subsampled)."""
    y = np.asarray(y)
    i1, i0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    k = min(len(i1), len(i0))
    g = np.random.default_rng(seed)
    return np.sort(np.concatenate([g.choice(i1, k, replace=False), g.choice(i0, k, replace=False)]))


def l1_select(X, y, C, exclude=None, iters=300):
    """L1 logistic with an unpenalised intercept by FISTA (K2's selector form; used when r1b_hn is not importable): X [n, N] torch
    (unstandardised CETT), y [n] 0/1; objective mean loss + ||w||_1 / (C n). Lipschitz bound by power iteration. Returns w [N]."""
    torch = _t()
    n, N = X.shape
    yt = y.to(X.dtype)
    v = torch.randn(N, dtype=X.dtype, device=X.device, generator=torch.Generator(device=X.device).manual_seed(0))
    for _ in range(30):
        v = X.T @ (X @ v); v = v / v.norm().clamp_min(1e-12)
    L = float((X @ v).norm() ** 2) / (4 * n) + 1e-6
    step, thr = 1.0 / L, 1.0 / (C * n * L)
    w = torch.zeros(N, dtype=X.dtype, device=X.device); b = torch.zeros((), dtype=X.dtype, device=X.device)
    zw, zb, t = w.clone(), b.clone(), 1.0
    mask = None if exclude is None else (~exclude.to(X.device))
    for _ in range(iters):
        p = torch.sigmoid(X @ zw + zb)
        g = X.T @ (p - yt) / n; gb = (p - yt).mean()
        wn = zw - step * g
        wn = torch.sign(wn) * (wn.abs() - thr).clamp_min(0)
        if mask is not None:
            wn = wn * mask
        bn = zb - step * gb
        tn = (1 + math.sqrt(1 + 4 * t * t)) / 2
        zw = wn + (t - 1) / tn * (wn - w); zb = bn + (t - 1) / tn * (bn - b)
        w, b, t = wn, bn, tn
    return w


def power_pc1(X, iters=50, seed=0):
    """First principal axis of the column-centred X [n, d] by power iteration (L1(iv))."""
    torch = _t()
    Xc = X - X.mean(0, keepdim=True)
    v = torch.randn(X.shape[1], dtype=X.dtype, device=X.device, generator=torch.Generator(device=X.device).manual_seed(seed))
    for _ in range(iters):
        v = Xc.T @ (Xc @ v); v = v / v.norm().clamp_min(1e-12)
    return v, Xc.mean(0) * 0 + X.mean(0)


# ==================================================================================================================================
# Frisch-Waugh slopes (L2b): cross-product matrices per cluster, Gram-Schmidt partialling (cov / var only)
# ==================================================================================================================================
def crossprod(V):
    """V [n, k] (already demeaned within chain) -> [k, k] sums of products."""
    return V.T @ V


def fw_slope(M, y, x, partial=()):
    """Slope of y on x after partialling the columns in `partial`, from a cross-product matrix M [k, k] (Gram-Schmidt: each step a
    cov / var residualisation)."""
    M = np.array(M, dtype=float)
    idx = list(partial)
    for j in idx:            # residualise every column on column j (sequentially orthogonal after the first)
        vj = M[j, j]
        if vj <= 0:
            continue
        c = M[:, j] / vj
        M = M - np.outer(c, M[j, :]) - np.outer(M[:, j], c) + np.outer(c, c) * vj
    return M[y, x] / M[x, x] if M[x, x] > 0 else float("nan")


# ==================================================================================================================================
# binary state dynamics (L3a): HD, stickiness hazard, Markov surrogates, bistability
# ==================================================================================================================================
def run_lengths(b):
    """Dwell d_t: number of consecutive 1s ending at t (0 where b_t = 0)."""
    d = np.zeros(len(b), dtype=np.int64); c = 0
    for t, v in enumerate(b):
        c = c + 1 if v else 0; d[t] = c
    return d


def hd_mh(chains, bins):
    """Mantel-Haenszel log odds ratio of P(b_t+1 | b_t = 1) against P(b_t+1 | b_t = 0) over strata chain x position bin.
    chains: list of int arrays b; bins: list of int arrays (bin of t)."""
    num = den = 0.0
    for b, bn in zip(chains, bins):
        if len(b) < 2:
            continue
        x0, x1, s = b[:-1], b[1:], bn[:-1]
        for k in np.unique(s):
            m = s == k
            a = float(((x0 == 1) & (x1 == 1) & m).sum()); bb = float(((x0 == 1) & (x1 == 0) & m).sum())
            c = float(((x0 == 0) & (x1 == 1) & m).sum()); d = float(((x0 == 0) & (x1 == 0) & m).sum())
            n = a + bb + c + d
            if n > 0:
                num += a * d / n; den += bb * c / n
    return math.log(num / den) if num > 0 and den > 0 else float("nan")


def hazard_design(chains, bins, ecov=None):
    """Exit-hazard rows over at-risk tokens (b_t = 1, t + 1 in range): chain id, bin, e_t (or 0), log(1 + d), exit."""
    rows = []
    for c, (b, bn) in enumerate(zip(chains, bins)):
        if len(b) < 2:
            continue
        d = run_lengths(b)
        t = np.flatnonzero(b[:-1] == 1)
        if t.size == 0:
            continue
        e = ecov[c][t] if ecov is not None else np.zeros(t.size)
        rows.append(np.stack([np.full(t.size, c), bn[t], e, np.log1p(d[t]), (b[t + 1] == 0).astype(float)], 1))
    return np.concatenate(rows) if rows else np.zeros((0, 5))


def fit_hazards(designs, n_chain, n_bin, use_e=True, weights=None, iters=80, ridge=1e-3, device="cpu"):
    """Batched exit hazards logit P(exit) = chain FE + bin FE + beta_e e - gamma log(1 + d) (L-BFGS on the summed objective).
    designs: list of [r, 5] arrays (one per fit); weights: optional list of [n_chain] multiplicities (cluster bootstrap).
    Returns gamma [F] (nan where a fit has no rows)."""
    torch = _t()
    F = len(designs)
    sizes = [len(d) for d in designs]
    if sum(sizes) == 0:
        return np.full(F, np.nan)
    D = torch.tensor(np.concatenate([d for d in designs if len(d)]), dtype=torch.float64, device=device)
    fid = torch.tensor(np.repeat(np.arange(F), sizes), device=device)
    ch = fid * n_chain + D[:, 0].long(); bn = fid * n_bin + D[:, 1].long()
    w = torch.ones(len(D), dtype=torch.float64, device=device)
    if weights is not None:
        Wt = torch.tensor(np.stack(weights), dtype=torch.float64, device=device)
        w = Wt[fid, D[:, 0].long()]
    fe_c = torch.zeros(F * n_chain, dtype=torch.float64, device=device, requires_grad=True)
    fe_b = torch.zeros(F * n_bin, dtype=torch.float64, device=device, requires_grad=True)
    be = torch.zeros(F, dtype=torch.float64, device=device, requires_grad=True)
    ga = torch.zeros(F, dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([fe_c, fe_b, be, ga], lr=1, max_iter=iters, line_search_fn="strong_wolfe", tolerance_change=1e-12)
    nrow = torch.zeros(F, dtype=torch.float64, device=device).index_add_(0, fid, w).clamp_min(1.0)

    def closure():
        opt.zero_grad()
        z = fe_c[ch] + fe_b[bn] + (be[fid] * D[:, 2] if use_e else 0.0) - ga[fid] * D[:, 3]
        l = torch.nn.functional.binary_cross_entropy_with_logits(z, D[:, 4], reduction="none") * w
        tot = (torch.zeros(F, dtype=torch.float64, device=device).index_add_(0, fid, l) / nrow).sum() \
            + ridge * ((fe_c ** 2).sum() + (fe_b ** 2).sum()) / max(n_chain, 1)
        tot.backward()
        return tot
    with torch.enable_grad():
        opt.step(closure)
    g = ga.detach().cpu().numpy()
    g[np.array(sizes) == 0] = np.nan
    return g


def fit_markov(chains, bins, n_bin, iters=60, ridge=1e-2):
    """Per-chain first-order Markov fit (validity fix: chain effects + shared position profile), entry and exit logits.
    Returns dict(entry_c, entry_b, exit_c, exit_b) numpy."""
    torch = _t()
    out = {}
    for kind in ("entry", "exit"):
        rows = []
        for c, (b, bn) in enumerate(zip(chains, bins)):
            if len(b) < 2:
                continue
            t = np.flatnonzero(b[:-1] == (0 if kind == "entry" else 1))
            y = (b[t + 1] == (1 if kind == "entry" else 0)).astype(float)
            rows.append(np.stack([np.full(t.size, c), bn[t], y], 1))
        R = torch.tensor(np.concatenate(rows) if rows else np.zeros((0, 3)), dtype=torch.float64)
        fc = torch.zeros(len(chains), dtype=torch.float64, requires_grad=True)
        fb = torch.zeros(n_bin, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([fc, fb], lr=1, max_iter=iters, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            if len(R):
                z = fc[R[:, 0].long()] + fb[R[:, 1].long()]
                l = torch.nn.functional.binary_cross_entropy_with_logits(z, R[:, 2], reduction="mean")
            else:
                l = torch.zeros((), dtype=torch.float64)
            tot = l + ridge * ((fc ** 2).mean() + (fb ** 2).mean())
            tot.backward()
            return tot
        with torch.enable_grad():
            opt.step(closure)
        out[kind + "_c"], out[kind + "_b"] = fc.detach().numpy(), fb.detach().numpy()
    return out


def sim_markov(fit, chains, bins, rng, entry_c=None, exit_c=None):
    """One surrogate per chain: time-varying entry / exit probabilities, memoryless dwell, initial state copied."""
    sig = lambda z: 1 / (1 + np.exp(-z))
    out = []
    ec = fit["entry_c"] if entry_c is None else entry_c
    xc = fit["exit_c"] if exit_c is None else exit_c
    for c, (b, bn) in enumerate(zip(chains, bins)):
        n = len(b)
        if n == 0:
            out.append(b.copy()); continue
        pe, px = sig(ec[c] + fit["entry_b"][bn]), sig(xc[c] + fit["exit_b"][bn])
        u = rng.random(n); s = np.empty(n, dtype=np.int8); s[0] = b[0]
        for t in range(n - 1):
            s[t + 1] = (u[t] < pe[t]) if s[t] == 0 else (u[t] >= px[t])
        out.append(s)
    return out


def bi_stat(chains_by_item, k0, k1, min_len, min_seeds=6):
    """Within-item bistability BI: share of items with >= 2 seeds o_s >= 0.3 and >= 2 seeds o_s <= 0.02, o_s = mean b over
    continuation tokens k0..k1 (seeds shorter than min_len excluded; items need >= min_seeds seeds). Returns (BI, n_items)."""
    split = n = 0
    for chains in chains_by_item:
        o = [float(np.mean(b[k0:k1])) for b in chains if len(b) >= min_len]
        if len(o) < min_seeds:
            continue
        n += 1
        o = np.array(o)
        split += int((o >= 0.3).sum() >= 2 and (o <= 0.02).sum() >= 2)
    return (split / n if n else float("nan")), n


# ==================================================================================================================================
# Ising (L3c, X3, X4, X5): pseudo-likelihood, K-pairwise, Gibbs, iterative scaling, descent, modularity, kinetic fits
# ==================================================================================================================================
def tau_int(x, max_lag=200):
    """Integrated autocorrelation time 1 + 2 sum rho(k) up to the first negative rho."""
    x = np.asarray(x, dtype=float); x = x - x.mean(); v = (x * x).mean()
    if v <= 0 or len(x) < 4:
        return 1.0
    s = 1.0
    for k in range(1, min(max_lag, len(x) - 1)):
        r = float((x[:-k] * x[k:]).mean() / v)
        if r <= 0:
            break
        s += 2 * r
    return s


def pl_fit(S, lam, groups=None, n_groups=1, kpair=False, pair=True, iters=100, device="cpu"):
    """Pseudo-likelihood fit of a static model on spins S [T, N] (0/1): logit P(s_i | s_-i) = h_i(g) + sum_j J_ij s_j (+ V(K_-i + 1) -
    V(K_-i) for the K-pairwise model). J symmetrised in the parametrisation (J = (A + A^T) / 2, zero diagonal). pair=False: independent.
    Returns dict(h [G, N], J [N, N], V [N + 1] or None)."""
    torch = _t()
    X = torch.as_tensor(S, dtype=torch.float32, device=device)
    T, N = X.shape
    g = torch.zeros(T, dtype=torch.long, device=device) if groups is None else torch.as_tensor(groups, device=device).long()
    h = torch.zeros(n_groups, N, dtype=torch.float32, device=device, requires_grad=True)
    A = torch.zeros(N, N, dtype=torch.float32, device=device, requires_grad=pair)
    V = torch.zeros(N + 1, dtype=torch.float32, device=device, requires_grad=kpair)
    eye = torch.eye(N, device=device)
    params = [h] + ([A] if pair else []) + ([V] if kpair else [])
    opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, line_search_fn="strong_wolfe")
    p = X.mean(0).clamp(1e-3, 1 - 1e-3)
    with torch.no_grad():
        h[:] = torch.log(p / (1 - p))[None]

    def closure():
        opt.zero_grad()
        z = _pl_logits(X, g, h, A, V if kpair else None, eye, pair)
        l = torch.nn.functional.binary_cross_entropy_with_logits(z, X, reduction="mean")
        tot = l + (lam * (A * A).sum() / N if pair else 0.0)
        tot.backward()
        return tot
    with torch.enable_grad():
        opt.step(closure)
    J = ((A + A.T) / 2 * (1 - eye)).detach() if pair else torch.zeros(N, N, device=device)
    return dict(h=h.detach(), J=J, V=V.detach() if kpair else None)


def _pl_logits(X, g, h, A, V, eye, pair=True):
    torch = _t()
    z = h[g]
    if pair:
        J = (A + A.T) / 2 * (1 - eye)
        z = z + X @ J
    if V is not None:
        K = X.sum(1, keepdim=True) - X          # K_-i
        Ki = K.long()
        z = z + V[Ki + 1] - V[Ki]
    return z


def pl_ll(S, fit, groups=None, device="cpu"):
    """Mean log-likelihood per token and unit [T] -> scalar per token (nats); returns per-token means [T] (numpy)."""
    torch = _t()
    X = torch.as_tensor(S, dtype=torch.float32, device=device)
    N = X.shape[1]
    g = torch.zeros(X.shape[0], dtype=torch.long, device=device) if groups is None else torch.as_tensor(groups, device=device).long()
    z = fit["h"][g] + X @ fit["J"]
    if fit.get("V") is not None:
        K = (X.sum(1, keepdim=True) - X).long(); z = z + fit["V"][K + 1] - fit["V"][K]
    ll = -torch.nn.functional.binary_cross_entropy_with_logits(z, X, reduction="none")
    return ll.mean(1).cpu().numpy()


def pl_select_lam(S, groups_items, lams=(1e-4, 1e-3, 1e-2), groups=None, n_groups=1, device="cpu"):
    """Inner CV (2 folds by item) for the L2 penalty of the pairwise PL fit."""
    it = np.asarray(groups_items); u = np.unique(it); f = np.isin(it, u[::2])
    best, bl = -1e18, lams[0]
    for lam in lams:
        sc = 0.0
        for a in (f, ~f):
            if a.sum() < 10 or (~a).sum() < 10:
                continue
            fit = pl_fit(S[a], lam, None if groups is None else groups[a], n_groups, device=device)
            sc += float(pl_ll(S[~a], fit, None if groups is None else groups[~a], device).mean())
        if sc > best:
            best, bl = sc, lam
    return bl


def gibbs(h, J, n_chains=512, sweeps=4000, burn=1000, seed=0, V=None, clamp=None, thin=10, device="cpu"):
    """Forward Gibbs sampling of a static pairwise model (h [N], J [N, N], optional V); clamp: dict(idx, val) of fixed units.
    Returns samples [n, N] (every `thin` sweeps after burn-in)."""
    torch = _t()
    gen = torch.Generator(device=device).manual_seed(seed)
    h = torch.as_tensor(h, dtype=torch.float32, device=device).flatten(); J = torch.as_tensor(J, dtype=torch.float32, device=device)
    N = h.numel()
    s = (torch.rand(n_chains, N, generator=gen, device=device) < torch.sigmoid(h)).float()
    if clamp is not None:
        s[:, clamp["idx"]] = torch.as_tensor(clamp["val"], dtype=torch.float32, device=device)
    free = torch.ones(N, dtype=torch.bool, device=device)
    if clamp is not None:
        free[clamp["idx"]] = False
    out = []
    for sw in range(sweeps + burn):
        for i in torch.nonzero(free).flatten().tolist():
            z = h[i] + s @ J[:, i]
            if V is not None:
                K = (s.sum(1) - s[:, i]).long(); z = z + V[K + 1] - V[K]
            s[:, i] = (torch.rand(n_chains, generator=gen, device=device) < torch.sigmoid(z)).float()
        if sw >= burn and (sw - burn) % thin == 0:
            out.append(s.clone())
    return torch.cat(out).cpu().numpy() if out else s.cpu().numpy()


def pk(S, N=None):
    """P(K) of the population count K = sum s over K = 0..N."""
    S = np.asarray(S); N = S.shape[1] if N is None else N
    return np.bincount(S.sum(1).astype(int), minlength=N + 1)[:N + 1] / max(len(S), 1)


def triplet_mean(S, n_trip=200, seed=0):
    """Mean connected triplet correlation over random unit triplets."""
    S = np.asarray(S, dtype=float); N = S.shape[1]
    if N < 3:
        return float("nan")
    g = np.random.default_rng(seed); X = S - S.mean(0)
    tr = [g.choice(N, 3, replace=False) for _ in range(n_trip)]
    return float(np.mean([(X[:, a] * X[:, b] * X[:, c]).mean() for a, b, c in tr]))


def maxent_info(S10, ess_scale=1.0, iters=400):
    """I2 / IN for a 10-unit subset: exact pairwise max-ent fit by iterative scaling over the 2^10 states (forward moment matching);
    S_1 independent entropy, S_2 pairwise max-ent entropy, S_N Miller-Madow plug-in with counts scaled to the ESS (nats)."""
    S10 = np.asarray(S10, dtype=np.int64); n, k = S10.shape
    states = ((np.arange(2 ** k)[:, None] >> np.arange(k)[None]) & 1).astype(float)
    code = (S10 * (1 << np.arange(k))).sum(1)
    cnt = np.bincount(code, minlength=2 ** k).astype(float)
    pe = cnt / n
    m1 = S10.mean(0); m2 = (S10.T @ S10) / n
    p = np.full(2 ** k, 1.0 / 2 ** k)
    for _ in range(iters):        # generalised iterative scaling on first and second moments
        q1 = states.T @ p; q2 = (states * p[:, None]).T @ states
        lp = np.zeros(2 ** k)
        for i in range(k):
            lp += states[:, i] * (np.log(max(m1[i], 1e-9) / max(q1[i], 1e-12)))
            lp += (1 - states[:, i]) * (np.log(max(1 - m1[i], 1e-9) / max(1 - q1[i], 1e-12)))
        for i in range(k):
            for j in range(i + 1, k):
                lp += states[:, i] * states[:, j] * np.log(max(m2[i, j], 1e-9) / max(q2[i, j], 1e-12)) / (k - 1)
        p = p * np.exp(lp / (k + 1)); p /= p.sum()
    H = lambda q: float(-(q[q > 0] * np.log(q[q > 0])).sum())
    S1 = float(sum(-(a * math.log(a) + (1 - a) * math.log(1 - a)) for a in np.clip(m1, 1e-12, 1 - 1e-12)))
    S2 = H(p)
    neff = max(n / ess_scale, 2.0)
    SN = H(pe) + (np.count_nonzero(cnt) - 1) / (2 * neff)
    IN, I2 = S1 - SN, S1 - S2
    return dict(I2=I2, IN=IN, ratio=(I2 / IN if IN > 1e-9 else float("nan")), IN_per_unit_bits=IN / k / LN2)


def descend(S, h, J):
    """Zero-temperature single-flip descent of E(s) = -h.s - s.J.s / 2 from every pattern; returns minima codes per row and the
    unique minima (as bit tuples)."""
    S = np.asarray(S, dtype=float).copy(); h = np.asarray(h, dtype=float).flatten(); J = np.asarray(J, dtype=float)
    for _ in range(4 * S.shape[1]):
        f = h[None] + S @ J               # local field; flipping i changes E by -(1 - 2 s_i) f_i
        dE = -(1 - 2 * S) * f
        i = dE.argmin(1); m = dE[np.arange(len(S)), i] < -1e-12
        if not m.any():
            break
        S[np.flatnonzero(m), i[m]] = 1 - S[np.flatnonzero(m), i[m]]
    keys = [tuple(r.astype(int)) for r in S]
    uniq = sorted(set(keys))
    lut = {k: i for i, k in enumerate(uniq)}
    return np.array([lut[k] for k in keys]), uniq


def modularity(W, iters=300, seed=0):
    """Leading split of the modularity matrix of W+ = max(W, 0) by power iteration (shifted to the most positive eigenvalue); Q."""
    W = np.clip(np.asarray(W, dtype=float), 0, None); np.fill_diagonal(W, 0)
    k = W.sum(1); m2 = k.sum()
    if m2 <= 0:
        return dict(Q=0.0, split=[0] * len(W))
    Bm = W - np.outer(k, k) / m2
    shift = np.abs(Bm).sum(1).max()
    v = np.random.default_rng(seed).standard_normal(len(W))
    for _ in range(iters):
        v = Bm @ v + shift * v; v /= np.linalg.norm(v) + 1e-12
    s = np.where(v >= 0, 1.0, -1.0)
    Q = float(s @ Bm @ s / (2 * m2))
    return dict(Q=Q, split=((s > 0).astype(int)).tolist())


def kinetic_fit(S, nxt, groups, n_groups, lam=1e-3, pair=True, iters=100, device="cpu"):
    """Kinetic Ising: per-unit forward logistic regressions of s_(j,t+1) on s_t with fields h_j(group). pair=False keeps only the
    self-coupling (the independent model with self-coupling). S, nxt [T, N]. Returns dict(h [G, N], J [N, N] (J[i, j]: i -> j))."""
    torch = _t()
    X = torch.as_tensor(S, dtype=torch.float32, device=device); Y = torch.as_tensor(nxt, dtype=torch.float32, device=device)
    g = torch.as_tensor(groups, device=device).long()
    N = X.shape[1]
    h = torch.zeros(n_groups, N, device=device, requires_grad=True)
    J = torch.zeros(N, N, device=device, requires_grad=True)
    mask = torch.ones(N, N, device=device) if pair else torch.eye(N, device=device)
    opt = torch.optim.LBFGS([h, J], lr=1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = h[g] + X @ (J * mask)
        l = torch.nn.functional.binary_cross_entropy_with_logits(z, Y, reduction="mean") + lam * ((J * mask) ** 2).sum() / N
        l.backward()
        return l
    with torch.enable_grad():
        opt.step(closure)
    return dict(h=h.detach(), J=(J * mask).detach())


def kinetic_ll(S, nxt, groups, fit, device="cpu"):
    torch = _t()
    X = torch.as_tensor(S, dtype=torch.float32, device=device); Y = torch.as_tensor(nxt, dtype=torch.float32, device=device)
    z = fit["h"][torch.as_tensor(groups, device=device).long()] + X @ fit["J"]
    return (-torch.nn.functional.binary_cross_entropy_with_logits(z, Y, reduction="none")).mean(1).cpu().numpy()


def asym(J):
    J = np.asarray(J, dtype=float)
    d = np.linalg.norm(J + J.T)
    return float(np.linalg.norm(J - J.T) / d) if d > 0 else float("nan")


# ==================================================================================================================================
# L3d: RQA trapping time, irreversibility d-hat, detailed-balance surrogates; Gaussian-copula MI (L4 Psi)
# ==================================================================================================================================
def rqa_tt(x, m=4, tau=1, rr=0.05, theiler=2, vmin=2):
    """Trapping time (mean vertical line length >= vmin) of the recurrence plot of the delay embedding of x [T, d] at a fixed
    recurrence rate (threshold = the rr quantile of pairwise distances outside the Theiler window)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n = len(x) - (m - 1) * tau
    if n < 8:
        return float("nan")
    Y = np.concatenate([x[i * tau: i * tau + n] for i in range(m)], 1)
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
    D = np.sqrt(((Y[:, None] - Y[None]) ** 2).sum(-1))
    off = np.abs(np.arange(n)[:, None] - np.arange(n)[None]) > theiler
    eps = np.quantile(D[off], rr) if off.any() else 0.0
    R = (D <= eps) & off
    L = []
    for j in range(n):
        c = 0
        for v in R[:, j]:
            if v:
                c += 1
            else:
                if c >= vmin:
                    L.append(c)
                c = 0
        if c >= vmin:
            L.append(c)
    return float(np.mean(L)) if L else 0.0


def word_counts(windows, n, k=4):
    """Counts of n-words over symbol windows (pooled), as a dict-free dense array over k^n words."""
    c = np.zeros(k ** n)
    base = k ** np.arange(n)[::-1]
    for w in windows:
        w = np.asarray(w, dtype=np.int64)
        if len(w) >= n:
            idx = np.lib.stride_tricks.sliding_window_view(w, n) @ base
            np.add.at(c, idx, 1)
    return c


def kl_rev(c, n, k=4, a=0.5):
    """D_n = sum_w p(w) ln[p(w) / p(w^R)] with Jeffreys pseudo-count a."""
    p = (c + a) / (c + a).sum()
    words = ((np.arange(k ** n)[:, None] // (k ** np.arange(n)[::-1])[None]) % k)
    rev = (words[:, ::-1] * (k ** np.arange(n)[::-1])[None]).sum(1)
    return float((p * np.log(p / p[rev])).sum())


def dhat(windows, k=4):
    """d-hat = D_3 - D_2 (Roldan and Parrondo 2010) over pooled windows."""
    return kl_rev(word_counts(windows, 3, k), 3, k) - kl_rev(word_counts(windows, 2, k), 2, k)


def db_surrogate(sym_series, rng, k=4):
    """Detailed-balance surrogate of a symbol series: first-order transition counts symmetrised (P_ij pi_i = P_ji pi_j), simulated
    from the copied initial symbol over the same length."""
    s = np.asarray(sym_series, dtype=np.int64)
    C = np.zeros((k, k)) + 0.5
    np.add.at(C, (s[:-1], s[1:]), 1)
    F = (C + C.T) / 2
    P = F / F.sum(1, keepdims=True)
    out = np.empty(len(s), dtype=np.int64); out[0] = s[0]
    u = rng.random(len(s))
    cum = np.cumsum(P, 1)
    for t in range(len(s) - 1):
        out[t + 1] = min(int(np.searchsorted(cum[out[t]], u[t])), k - 1)
    return out


def quartiles(x, ref=None):
    q = np.quantile(np.asarray(x if ref is None else ref, dtype=float), [0.25, 0.5, 0.75])
    return np.searchsorted(q, np.asarray(x, dtype=float), side="right")


def gc_mi(x, y):
    """Gaussian-copula MI (nats) of two 1-D samples: ranks -> normal scores (Phi^-1 on ranks transforms marginals) -> -0.5 ln(1-r^2)."""
    from scipy.stats import norm, rankdata
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4:
        return float("nan")
    a = norm.ppf(rankdata(x) / (len(x) + 1)); b = norm.ppf(rankdata(y) / (len(y) + 1))
    r = np.corrcoef(a, b)[0, 1]
    return float(-0.5 * math.log(max(1 - r * r, 1e-12)))


# ==================================================================================================================================
# L2c simplex cross-map
# ==================================================================================================================================
def embed(x, E, tau=1):
    """Delay embedding of x [T, d] -> [T - (E - 1) tau, d E] (row t holds x_t, x_(t-1), ...)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n = len(x) - (E - 1) * tau
    return np.concatenate([x[(E - 1 - i) * tau:(E - 1 - i) * tau + n] for i in range(E)], 1)


def simplex_nn(M, lib, pred, E, theiler):
    """E + 1 nearest library neighbours of every prediction point (Theiler exclusion |t' - t| <= theiler), exponential weights."""
    D = np.sqrt(((M[pred][:, None] - M[lib][None]) ** 2).sum(-1))
    D[np.abs(pred[:, None] - lib[None]) <= theiler] = np.inf
    k = min(E + 1, len(lib) - 1)
    nn = np.argsort(D, 1)[:, :k]
    d = np.take_along_axis(D, nn, 1)
    d0 = np.maximum(d[:, :1], 1e-9)
    w = np.where(np.isfinite(d), np.exp(-d / d0), 0.0); w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
    return lib[nn], w


def xmap_rho(target, nbr, w, pred):
    """Pearson skill of the cross-map estimate of each target column [T', C] at the prediction points."""
    est = (target[nbr] * w[..., None]).sum(1)
    obs = target[pred]
    ec, oc = est - est.mean(0), obs - obs.mean(0)
    den = np.sqrt((ec ** 2).sum(0) * (oc ** 2).sum(0))
    return np.where(den > 0, (ec * oc).sum(0) / np.maximum(den, 1e-12), np.nan)


def simplex_E(series, Es=(2, 3, 4, 5, 6), theiler_extra=0):
    """E by simplex 1-step self-forecast skill (mean over series of the Pearson skill of B_(t+1) from E-dim neighbours)."""
    best, bE = -2.0, Es[0]
    for E in Es:
        sk = []
        for x in series:
            M = embed(x, E)
            if len(M) < E + 6:
                continue
            idx = np.arange(len(M) - 1)
            half = len(idx) // 2
            lib, pred = idx[:half], idx[half:]
            nb, w = simplex_nn(M, lib, pred, E, E + theiler_extra)
            tgt = M[1:, :np.asarray(x).reshape(len(x), -1).shape[1]]
            r = xmap_rho(np.concatenate([tgt, tgt[-1:]]), nb, w, pred)
            sk.append(np.nanmean(r))
        s = float(np.nanmean(sk)) if sk else -2.0
        if s > best:
            best, bE = s, E
    return bE


# ==================================================================================================================================
# L2a growth classes (1-D grid least squares, BIC)
# ==================================================================================================================================
def growth_fits(L, I):
    """BIC of saturating I_inf (1 - e^(-L/lam)), logarithmic a + b ln L, power a L^beta; each fitted by closed-form 1-D least squares
    over a grid of the nonlinear parameter. Returns dict(class -> dict(bic, params))."""
    L = np.asarray(L, float); I = np.asarray(I, float); n = len(L)
    out = {}

    def bic(rss, k):
        return n * math.log(max(rss, 1e-300) / n) + k * math.log(n)
    best = None
    for lam in np.exp(np.linspace(math.log(L.min() / 4), math.log(L.max() * 8), 200)):
        f = 1 - np.exp(-L / lam); a = float(f @ I / max(f @ f, 1e-300)); r = float(((I - a * f) ** 2).sum())
        if best is None or r < best[0]:
            best = (r, a, lam)
    out["saturating"] = dict(bic=bic(best[0], 2), I_inf=best[1], lam=best[2])
    x = np.log(L); b = float(np.cov(x, I, bias=True)[0, 1] / max(x.var(), 1e-300)); a = float(I.mean() - b * x.mean())
    out["logarithmic"] = dict(bic=bic(float(((I - a - b * x) ** 2).sum()), 2), a=a, b=b)
    best = None
    for be in np.linspace(0.01, 1.5, 300):
        f = L ** be; a = float(f @ I / max(f @ f, 1e-300)); r = float(((I - a * f) ** 2).sum())
        if best is None or r < best[0]:
            best = (r, a, be)
    out["power"] = dict(bic=bic(best[0], 2), a=best[1], beta=best[2])
    return out


def growth_class(L, I):
    f = growth_fits(L, I)
    return min(f, key=lambda k: f[k]["bic"]), f


# ==================================================================================================================================
# L4: forward logistic with item fixed effects (agent kinetic Ising); L5c conditional hazard (same form)
# ==================================================================================================================================
def logit_fe(y, fe, X, n_fe, weights=None, ridge=1e-3, iters=100):
    """logit P(y = 1) = a + FE[fe] + X @ beta; returns beta (numpy), a, FE. Forward regression by L-BFGS (ridge on FE only)."""
    torch = _t()
    y = torch.as_tensor(np.asarray(y, float)); X = torch.as_tensor(np.asarray(X, float)).reshape(len(y), -1)
    fe = torch.as_tensor(np.asarray(fe)).long()
    w = torch.ones(len(y), dtype=torch.float64) if weights is None else torch.as_tensor(np.asarray(weights, float))
    beta = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    F = torch.zeros(max(n_fe, 1), dtype=torch.float64, requires_grad=True)
    a = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([beta, F, a], lr=1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = a + F[fe] + X @ beta
        l = (torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none") * w).sum() / w.sum().clamp_min(1e-9) \
            + ridge * (F ** 2).mean()
        l.backward()
        return l
    with torch.enable_grad():
        opt.step(closure)
    return beta.detach().numpy(), float(a.detach()), F.detach().numpy()


def logit_ll(y, fe, X, beta, a, F):
    y = np.asarray(y, float); X = np.asarray(X, float).reshape(len(y), -1)
    z = a + F[np.asarray(fe, int)] + X @ beta
    return -(np.logaddexp(0, z) - y * z)


def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return float("nan")
    return float(spearmanr(x[m], y[m]).correlation)


def derange(n, rng):
    """A random derangement of range(n) (n >= 2)."""
    while True:
        p = rng.permutation(n)
        if n < 2 or not (p == np.arange(n)).any():
            return p

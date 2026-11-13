"""Part D, session r1b_D_ana (account A; CPU work on the VM): every D verdict (D.4-D.17) from the copied r1b_D_A / r1b_D_B / r1b_D_C
outputs and SO-AX. Steps: ana_d1 (G-M, G-D1-PC, G-deg, G-D1-null, D1-a ... D1-s, coexistence index), ana_d2 (G-M2, G-LCE-3, G-D2-PC,
G-D2-null, D2-a/b/c/g, specificity labels, D2-f, descriptives), ana_d3 (G-D0c, G-D0d, G-D3pc, G-DGN, D3 per length and across lengths,
D4), ana_d5 (G-D6, D5-a/r/g/b, D5-x), ana_comb (D.17: Holm over (a), (b1), (b2); d_verdict.json with the R2 trigger).
Intervals: 90 % percentile over 2,000 replicates (row 0 = point); one-sided p = (1 + #null side) / 2,001 (SO-PVAL). Clip +-15 nats.
Implementation choices where the part file is silent are listed in the part file (D.23)."""
import math, os
import numpy as np
import r1b_core as K
import r1b_contracts as C
import r1b_d_stim as S

CLIP, B = 15.0, K.N_BOOT
LBL_SEC = "secondary (not multiplicity-adjusted)"


# ------------------------------------------------------------------------------------------------------------------------------------
# bootstrap helpers (row 0 = point estimate)
# ------------------------------------------------------------------------------------------------------------------------------------
def weights(groups, seed, two=True):
    """W [B+1, n]: two-level (streams, then units within streams) or one-level unit bootstrap; groups = per stream the unit indices."""
    n = sum(len(g) for g in groups); W = np.zeros((B + 1, n)); W[0] = 1
    if two:
        gs = [g for g in groups if len(g)]
        for b, rep in enumerate(K.boot_two_level([len(g) for g in gs], B, seed)):
            for s, idx in rep:
                np.add.at(W[b + 1], np.asarray(gs[s])[idx], 1)
    else:
        for b, idx in enumerate(K.boot_idx(n, B, seed)):
            np.add.at(W[b + 1], idx, 1)
    return W


def wm(W, X):
    """Weighted means over units (axis 0 of X) per replicate; NaN units carry no weight."""
    X = np.asarray(X, float); ok = np.isfinite(X); Xz = np.where(ok, X, 0.0)
    num = np.tensordot(W, Xz, axes=(1, 0)); den = np.tensordot(W, ok.astype(float), axes=(1, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def S_(x):
    """Summary of a replicate vector: point, 90 % interval, SD, realized MDE (2.49 SD), non-finite share."""
    x = np.asarray(x, float); lo, hi = K.interval(x[1:])
    sd = float(np.nanstd(x[1:], ddof=1)) if np.isfinite(x[1:]).sum() > 1 else float("nan")
    return dict(pt=K.val(x[0], "NaN:undefined_point"), lo=lo, hi=hi, sd=K.val(sd), mde=K.val(2.49 * sd), nonfinite=float((~np.isfinite(x[1:])).mean()))


def lo(x):
    v = S_(x)["lo"]
    return v if isinstance(v, float) else float("nan")


def hi(x):
    v = S_(x)["hi"]
    return v if isinstance(v, float) else float("nan")


def pt(x):
    return float(np.asarray(x, float)[0])


def sd(x):
    return float(np.nanstd(np.asarray(x, float)[1:], ddof=1))


def inside(x, a, b):
    return lo(x) > a and hi(x) < b


def p_gt(x, thr):
    return K.p_one_sided(np.asarray(x, float)[1:], thr, "greater")


def survival(theta, theta_k):
    """D.3 survival rule for one check: 'survives' / 'explained' / 'inconclusive'."""
    t, tk = pt(theta), pt(theta_k)
    if tk >= 0.5 * t and lo(theta_k) > 0:
        return "survives"
    d = np.asarray(theta, float) - np.asarray(theta_k, float)
    if tk < 0.5 * t and (lo(d) > 0 or hi(d) < 0):
        return "explained"
    return "inconclusive"


def clipped(v):
    return float(np.clip(v, -CLIP, CLIP))


# ------------------------------------------------------------------------------------------------------------------------------------
# D1 data
# ------------------------------------------------------------------------------------------------------------------------------------
class D1:
    def __init__(self, recs):
        self.ix = {}
        for r in recs:
            k = (r["arm"], r["fam"], r["item"], r["swap"], r["k"], r["order"], r["arr"], r.get("enc", "can"), r["L"], r["filler"], r["model"])
            self.ix[k] = r

    def get(self, arm, item, swap=0, k=0, order=0, arr=None, L=0, fam="sym", enc="can", filler="coh", model=C.M17):
        arr = arr or {"pure": "pure", "xn": "xn"}.get(arm, "int")
        return self.ix.get((arm, fam, item, swap, k, order, arr, enc, L, filler, model))

    def att(self, rec, mean_assign=True):
        """[6] mean clipped l per attribute (both letter assignments averaged where asked); NaN if absent."""
        out = np.full(6, np.nan)
        if rec is None or not rec.get("l"):
            return out
        acc = {}
        for (j, bf), v in zip(rec["qs"], rec["l"]):
            acc.setdefault(int(j), []).append(clipped(v))
        for j, v in acc.items():
            out[j] = np.mean(v)
        return out

    def A(self, items, **kw):
        return np.array([self.att(self.get(item=i, **kw)) for i in items])


def item_groups(items, n_streams):
    g = [[] for _ in range(n_streams)]
    for u, i in enumerate(items):
        g[i % n_streams].append(u)
    return g


def m_ij(*arrs):
    with np.errstate(all="ignore"):
        return np.nanmean(np.stack(arrs), 0)


def curve(D, items, L, ks, arm="main", orders=(1, 2), swaps=(0, 1), filler="coh", model=C.M17, enc="can", pure_arm="pure", fam="sym"):
    """Per-item arrays: l0 (pure, B-oriented), lk {k: mean over swaps, orders, attributes}."""
    l0 = np.nanmean(np.stack([D.A(items, arm=pure_arm, swap=s, L=L, filler=filler, model=model, enc=enc, fam=fam) for s in swaps]), (0, 2))
    lk = {k: np.nanmean(np.stack([D.A(items, arm=arm, swap=s, k=k, order=o, L=L, filler=filler, model=model, enc=enc, fam=fam)
                                  for s in swaps for o in orders]), (0, 2)) for k in ks}
    return l0, lk


def phi(W, l0, lk, ks):
    """S(k) = [l(k) - l(0)] / [l(10) - l(0)] with l(10) = -l(0) (ratios per replicate); returns (Phi over ks, {k: S(k)})."""
    m0 = wm(W, l0); Sk = {}
    for k in ks:
        with np.errstate(all="ignore"):
            Sk[k] = (wm(W, lk[k]) - m0) / (-2 * m0)
    return np.mean([Sk[k] for k in ks], 0), Sk


# ---- G-D1-null: per-length forward readout fit l = c tanh((beta E + s b)/c), envelope, planted validation
def fit_readout(y, E, s):
    """y [n, m] measured log-odds at m contexts with evidence E [m] and prior sign s [m] per item-attribute (rows); shared c (grid),
    per-row beta >= 0 (capped where the fitted pure context reaches the clip or tanh 0.999) and b (Gauss-Newton). Returns dict."""
    ok_rows = np.isfinite(y).all(1); y = y[ok_rows]; best = None
    for c in np.geomspace(1.0, 60.0, 36):
        cap = c * (math.atanh(min(CLIP / c, 0.999))) / 10.0
        bt = np.clip(-(y[:, E == -10].mean(1) * s[E == -10]).mean() / 10 * np.ones(len(y)) if False else np.full(len(y), 0.3), 0, cap)
        b = np.zeros(len(y))
        for _ in range(40):
            u = bt[:, None] * E[None] + s[None] * b[:, None]; th = np.tanh(u / c); f = c * th; r = y - f; g = 1 - th ** 2
            J1, J2 = g * E[None], g * s[None]
            a11, a12, a22 = (J1 * J1).sum(1) + 1e-6, (J1 * J2).sum(1), (J2 * J2).sum(1) + 1e-6
            g1, g2 = (J1 * r).sum(1), (J2 * r).sum(1); det = a11 * a22 - a12 ** 2
            bt = np.clip(bt + (a22 * g1 - a12 * g2) / det, 0, cap); b = b + (a11 * g2 - a12 * g1) / det
        u = bt[:, None] * E[None] + s[None] * b[:, None]; sse = float(((y - c * np.tanh(u / c)) ** 2).sum())
        if best is None or sse < best["sse"]:
            best = dict(c=float(c), beta=bt, b=b, sse=sse, capped=float((bt >= cap - 1e-9).mean()), cap=cap)
    best["rows"] = ok_rows; best["resid_sd"] = math.sqrt(best["sse"] / max(y.size - 2 * len(y) - 1, 1))
    return best


KERNELS = [(tau, pr) for tau in (0.7, 1.5, 3.0, float("inf")) for pr in (0.0, 0.5)]


def kernel_w(tau, pr):
    s = np.arange(10); w = (np.exp(-(9 - s) / tau) if np.isfinite(tau) else np.ones(10)) + pr * np.exp(-s / 1.5)
    return 10 * w / w.sum()


def evidence(kinds, swap, w):
    A, Bk = ("A1", "A2") if swap == 0 else ("A2", "A1")
    e = np.array([1.0 if x == Bk else (-1.0 if x == A else 0.0) for x in kinds])
    return float((w * e).sum())


def null_curves(D, items, L, fit, mode="all", strata=None):
    """Forward-simulated null l (no capture mechanism) for the main design at L for each kernel: per item l0 and lk (k = 1..4), using the
    actual placements, orders and swaps; mode 'opp' restricts each item-attribute to its prior-opposed swap (per-swap endpoints)."""
    rows = np.zeros((len(items), 6), bool); beta = np.zeros((len(items), 6)); bb = np.zeros((len(items), 6))
    idx = np.where(fit["rows"])[0]; flat = np.zeros(len(items) * 6, bool); flat[idx] = True
    beta.flat[idx] = fit["beta"]; bb.flat[idx] = fit["b"]; c = fit["c"]; out = []
    for tau, pr in KERNELS:
        w = kernel_w(tau, pr); l0 = np.full((len(items), 6, 2), np.nan); lk = {k: np.full((len(items), 6, 2, 2), np.nan) for k in range(1, 5)}
        for u, i in enumerate(items):
            for sw in (0, 1):
                sg = 1 if sw == 0 else -1
                l0[u, :, sw] = c * np.tanh((beta[u] * -10 + sg * bb[u]) / c)
                for k in range(1, 5):
                    for o in (1, 2):
                        r = D.get("main", i, sw, k, o, L=L)
                        if r is not None:
                            E = evidence(r["kinds"], sw, w)
                            lk[k][u, :, sw, o - 1] = c * np.tanh((beta[u] * E + sg * bb[u]) / c)
        ok = flat.reshape(len(items), 6)
        l0[~ok] = np.nan
        for k in lk:
            lk[k][~ok] = np.nan
        out.append(strat(l0, lk, strata, mode))
    return out


def strat(l0, lk, strata, mode):
    """Per-item endpoints and curves: mode 'all' averages swaps; 'opp' / 'al' take each item-attribute's stratum swap with per-swap
    endpoints l_s(10) = -l_(1-s)(0)."""
    with np.errstate(all="ignore"):
        if mode == "all":
            return np.nanmean(l0, (1, 2)), {k: np.nanmean(v, (1, 2, 3)) for k, v in lk.items()}, None
        sw = strata if mode == "opp" else 1 - strata                           # [n, 6] swap index of the stratum
        take = lambda a: np.take_along_axis(a, sw[..., None], 2)[..., 0]
        l0s = take(l0); l10 = -np.take_along_axis(l0, (1 - sw)[..., None], 2)[..., 0]
        lks = {k: np.nanmean(np.take_along_axis(v, sw[..., None, None].repeat(2, -1), 2)[:, :, 0, :], -1) for k, v in lk.items()}
        return np.nanmean(l0s, 1), {k: np.nanmean(x, 1) for k, x in lks.items()}, np.nanmean(l10, 1)


def phi_s(W, l0, lk, l10, ks=(1, 2, 3, 4)):
    m0 = wm(W, l0); m10 = -m0 if l10 is None else wm(W, l10)
    with np.errstate(all="ignore"):
        return np.mean([(wm(W, lk[k]) - m0) / (m10 - m0) for k in ks], 0)


def gm_ok(D, L, arms=None):
    m = [x for r in D.ix.values() if r["L"] == L and r["model"] == C.M17 and r["filler"] == "coh" and (arms is None or r["arm"] in arms)
         for x in r["M"]]
    return (float(np.median(m)) if m else float("nan")), bool(m and np.median(m) >= 0.9)


def _ana_d1(ctx):
    torch = K._torch(); c = S.cfg(ctx); recs = []
    for name in ("ends", "L0", "Ls", "Lf", "s"):
        p = ctx.input("r1b_D_A", f"d1/raw_{name}.pt", required=False)
        if p:
            recs += torch.load(p, weights_only=False)["records"]
    if not recs:
        raise K.NotRun("input missing")
    ls = K.load_json(ctx.input("r1b_D_A", "d1/lstar.json")); Ls = ls["L_star"]; mode = ls.get("mode")
    D = D1(recs); items = list(range(c.n_sym)); W = weights(item_groups(items, c.n_streams), 11); out = dict(L_star=Ls, mode=mode)
    # ---- gates
    Lall = sorted({r["L"] for r in recs if r["model"] == C.M17 and r["filler"] == "coh"})
    out["G-M"] = {str(L): dict(zip(("median", "passed"), gm_ok(D, L))) for L in Lall}
    l0_0, lk0 = curve(D, items, 0, range(1, 6))
    g_pc = dict(i=S_(wm(W, lk0[4]) - wm(W, lk0[1])), ii=K.val(float(-2 * np.nanmean(l0_0))))
    it24 = list(range(min(24 if not ctx.trimmed("T6") else 16, c.n_sym))); W24 = weights(item_groups(it24, c.n_streams), 11)
    _, lr0 = curve(D, it24, 0, (2,), arm="rel", orders=(0,)); _, li0 = curve(D, it24, 0, (2,))
    g_pc["iii"] = S_(wm(W24, lr0[2]) - wm(W24, li0[2]))
    pc_ok = lo(wm(W, lk0[4]) - wm(W, lk0[1])) > 0 and pt(-2 * wm(W, l0_0)) >= 4
    rec_ok = lo(wm(W24, lr0[2]) - wm(W24, li0[2])) > 0
    out["G-D1-PC"] = dict(i_share_response=g_pc["i"], ii_range_L0=g_pc["ii"], iii_recency=g_pc["iii"], passed_i_ii=pc_ok, passed_iii=rec_ok)
    rho = {}
    for L in Lall:
        l0L, _ = curve(D, items, L, ())
        rho[str(L)] = S_(wm(W, l0L) / wm(W, l0_0))
    out["G-deg"] = dict(rho=rho, rule="rho_L = [l(10, L) - l(0, L)] / [l(10, L0) - l(0, L0)] >= 0.5 (point)")
    recomp = None
    for L in c.L_long:
        if str(L) in rho and isinstance(rho[str(L)]["pt"], float) and rho[str(L)]["pt"] >= 0.5 and out["G-M"].get(str(L), {}).get("passed"):
            recomp = L; break
    out["L_star_recomputed"] = dict(value=recomp, agrees=recomp == Ls or mode == "FB-2" or str(mode).startswith("MOCK"))
    gm_bad = lambda L: not out["G-M"].get(str(L), {}).get("passed", False) and not ctx.mock
    # ---- D1-a
    out["D1-a"] = d1a(ctx, D, items, W, Ls, mode, pc_ok, gm_bad, c)
    # ---- D1-b
    out["D1-b"] = d1b(D, it24, W24, Ls, rec_ok)
    out["D1-c"] = {str(L): d1c(D, list(range(min(24, c.n_sym))), c, L) for L in ([0] if (ctx.trimmed("T2") or Ls is None) else [0, Ls])}
    out["D1-d"] = d1d(D, list(range(min(24, c.n_sym))), c, Ls, out["D1-a"])
    out["D1-e"] = d1e(D, list(range(min(24, c.n_asym))), c, Ls)
    out["D1-f"] = d1f(D, list(range(min(24 if not ctx.trimmed("T5") else 16, c.n_sym))), c, Ls, out["D1-a"])
    out["D1-r"] = d1r(D, list(range(min(24, c.n_sym))), list(range(min(24, c.n_asym))), c)
    p = ctx.input("r1b_D_A", "d1/d1v.json", required=False)
    out["D1-v"] = d1v(K.load_json(p)) if p else dict(verdict=K.coded("NA:not_run:input missing"))
    out["D1-s"] = d1s(ctx, D, list(range(min(24 if not ctx.trimmed("T3") else 16, c.n_sym))), c, Ls)
    out["coexistence"] = coexistence(D, items, [0, Ls] if Ls else [0])
    ctx.write_json("d_ana/d1.json", out, config=dict(seeds=dict(d1=11), clip=CLIP, cfg=dict(c)), seeds=dict(bootstrap=11),
                   labels=["MOCK (tiny random model)"] if ctx.mock else [])


def d1a(ctx, D, items, W, Ls, mode, pc_ok, gm_bad, c):
    """D1-a (primary; part a), D.5, with G-D1-null (fit, envelope, planted validation) and the three survival checks."""
    res = dict(labels=[])
    if not pc_ok and not ctx.mock:
        return dict(verdict="not testable (readout insensitive)", support=False, negative=False)
    if Ls is None:
        return dict(verdict="not decidable (no gated long length)", support=False, negative=False)
    if gm_bad(0) or gm_bad(Ls):
        return dict(verdict="not testable (letter readout not committed at L)", support=False, negative=False)
    if mode == "FB-2":
        rows = [r for r in D.ix.values() if r["arm"] == "fb2"]
        return dict(verdict="not decidable (envelope and survival checks not run under FB-2)", support=False, negative=False,
                    labels=["engine fallback FB-2: 16K full recompute, 32 items, one attribute per context"], n_fb2=len(rows))
    l0a, lka = curve(D, items, 0, range(1, 5)); l0b, lkb = curve(D, items, Ls, range(1, 5))
    P0, _ = phi(W, l0a, lka, range(1, 5)); P1, _ = phi(W, l0b, lkb, range(1, 5)); dP = P1 - P0
    P0_24, _ = phi(W, l0a, lka, (2, 4)); P1_24, _ = phi(W, l0b, lkb, (2, 4))
    res.update(Phi_L0=S_(P0), Phi_Ls=S_(P1), dPhi=S_(dP), Phi24_L0=S_(P0_24), Phi24_Ls=S_(P1_24))
    # G-D1-null (i) fits at L0 and L*: X_N (E 0, s +1), (0,5,5) per swap (E -5), pure per swap (E -10)
    fits, env, env_opp, strata = {}, {}, {}, None
    xn0 = D.A(items, arm="xn", L=0)
    bh = np.where(np.isfinite(xn0), xn0, 0.0)
    par = (np.arange(len(items)) % 2 == 0)[:, None].repeat(6, 1)
    strata = np.where(np.abs(bh) < 0.05, np.where(par, 1, 0), np.where(bh > 0, 1, 0))     # opposed swap: B holds the value b_hat disfavours
    for L in (0, Ls):
        cols = [D.A(items, arm="xn", L=L)] + [np.nanmean(np.stack([D.A(items, arm="nc", swap=s, order=o, arr="nc", L=L) for o in (1, 2)]), 0)
                                               for s in (0, 1)] + [D.A(items, arm="pure", swap=s, L=L) for s in (0, 1)]
        y = np.stack([x.reshape(-1) for x in cols], 1); E = np.array([0, -5, -5, -10, -10.0]); sg = np.array([1, 1, -1, 1, -1.0])
        fits[L] = fit_readout(y, E, sg)
        for md, store in (("all", env), ("opp", env_opp)):
            sims = null_curves(D, items, L, fits[L], md, strata)
            mx = np.max([phi_s(W, a, b_, c_) for a, b_, c_ in sims], 0)
            store[L] = float(np.nanpercentile(mx[1:], 95)) if np.isfinite(mx[1:]).any() else float("nan")
    res["G-D1-null"] = dict(fit={str(L): dict(c=f["c"], capped_share=f["capped"], resid_sd=f["resid_sd"], n_rows=int(f["rows"].sum()))
                                 for L, f in fits.items()}, env={str(L): K.val(v) for L, v in env.items()},
                            env_opp={str(L): K.val(v) for L, v in env_opp.items()},
                            kernels=[dict(tau=K.val(t, "NaN:infinite"), primacy=p) for t, p in KERNELS])
    bound = max(0.25, env.get(Ls, 0.25) if np.isfinite(env.get(Ls, np.nan)) else 0.25)
    cond4 = lambda P1_, dP_: lo(P1_) > bound and lo(dP_) > 0
    val = planted(D, items, W, Ls, fits, bound, n=(200 if not ctx.mock else 20))
    res["G-D1-null"]["validation"] = val
    null_ok = val["null_rate"] <= 0.05
    # prior strata
    def st_phi(L, md):
        l0 = np.stack([D.A(items, arm="pure", swap=s, L=L) for s in (0, 1)], 2)
        lk = {k: np.stack([np.stack([D.A(items, arm="main", swap=s, k=k, order=o, L=L) for o in (1, 2)], 2) for s in (0, 1)], 2)
              for k in range(1, 5)}
        a, b_, c_ = strat(l0, lk, strata, md)
        return phi_s(W, a, b_, c_)
    Po0, Po1, Pa0, Pa1 = st_phi(0, "opp"), st_phi(Ls, "opp"), st_phi(0, "al"), st_phi(Ls, "al")
    res.update(Phi_opp_Ls=S_(Po1), dPhi_opp=S_(Po1 - Po0), Phi_al_Ls=S_(Pa1), dPhi_al=S_(Pa1 - Pa0))
    absb = np.abs(bh); med = np.median(absb)
    for half, m in (("low", absb <= med), ("high", absb > med)):
        P_ = {}
        for L in (0, Ls):
            l0 = np.nanmean(np.where(m[None], np.stack([D.A(items, arm="pure", swap=s, L=L) for s in (0, 1)]), np.nan), (0, 2))
            lk = {k: np.nanmean(np.where(m[None], np.stack([D.A(items, arm="main", swap=s, k=k, order=o, L=L) for s in (0, 1) for o in (1, 2)]),
                                         np.nan), (0, 2)) for k in range(1, 5)}
            P_[L] = phi(W, l0, lk, range(1, 5))[0]
        res[f"Phi_{half}_abs_b"] = dict(L0=S_(P_[0]), Ls=S_(P_[Ls]))
    # hedging control (items 0-31) and filler form (items 0-23): theta on each check's own item set (D.23)
    ih = list(range(min(32, len(items)))); Wh = weights(item_groups(ih, c.n_streams), 11)
    def p24(L, arm="main", sub_items=None, Wx=None, filler="coh", l0arm="pure"):
        l0, lk = curve(D, sub_items, L, (2, 4), arm=arm, filler=filler, pure_arm=l0arm)
        return phi(Wx, l0, lk, (2, 4))[0]
    dP24_h = p24(Ls, sub_items=ih, Wx=Wh) - p24(0, sub_items=ih, Wx=Wh)
    l00, lkC0 = curve(D, ih, 0, (2, 4), arm="hedge"); l0L, lkCL = curve(D, ih, Ls, (2, 4), arm="hedge")
    PC0, PC1 = phi(Wh, l00, lkC0, (2, 4))[0], phi(Wh, l0L, lkCL, (2, 4))[0]
    psi = dP24_h - (PC1 - PC0)
    res.update(PhiC24_L0=S_(PC0), PhiC24_Ls=S_(PC1), Psi_Delta=S_(psi))
    iff = list(range(min(24, len(items)))); Wf = weights(item_groups(iff, c.n_streams), 11)
    _, lkf = curve(D, iff, Ls, (2, 4), arm="f", filler="shuf")
    with np.errstate(all="ignore"):
        l0f = np.nanmean(np.stack([D.A(iff, arm="f", swap=s, arr="pure", L=Ls, filler="shuf") for s in (0, 1)]), (0, 2))
    has_f = bool(np.isfinite(l0f).any())
    if has_f:
        Pf = phi(Wf, l0f, lkf, (2, 4))[0]; Pf0 = p24(0, sub_items=iff, Wx=Wf)
        dP24_f = p24(Ls, sub_items=iff, Wx=Wf) - Pf0; thk_f = Pf - Pf0; res.update(Phi24_shuf_Ls=S_(Pf))
    # rule
    res["p_a"] = max(p_gt(P1, bound), p_gt(dP, 0.0)); res["bound"] = bound
    if not null_ok:
        res.update(verdict="not decidable (rule not specific at the measured noise)", support=False, negative=False); return res
    if cond4(P1, dP):
        th_lvl, thk_lvl = P1 - bound, Po1 - max(0.25, env_opp.get(Ls, 0.25) if np.isfinite(env_opp.get(Ls, np.nan)) else 0.25)
        pr = [survival(th_lvl, thk_lvl), survival(dP, Po1 - Po0)]
        prior = "explained" if "explained" in pr else ("survives" if pr == ["survives", "survives"] else "inconclusive")
        hedge = survival(dP24_h, psi)
        form = survival(dP24_f, thk_f) if has_f else "inconclusive"
        res["checks"] = dict(prior=dict(result=prior, components=pr), hedging=hedge, filler_form=form if has_f else "not run")
        if prior == "explained":
            v = "not minority capture: explained by prior x saturation"
        elif hedge == "explained":
            v = "not minority capture: explained by conflict hedging"
        elif form == "explained":
            v = "not minority capture: explained by filler form"
        else:
            inc = [n for n, r in (("prior", prior), ("hedging", hedge), ("filler form", form)) if r != "survives"]
            v = ("length-driven minority capture (survives prior, hedging and filler-form checks)" if not inc else
                 f"length-driven minority capture (inconclusive on {', '.join(inc)})")
        res.update(verdict=v, support=not v.startswith("not minority"), negative=v.startswith("not minority")); return res
    if lo(P0) > bound and lo(P1) > bound and inside(dP, -0.03, 0.03):
        v = "minority over-representation without length dependence"
    elif hi(dP) < 0:
        v = "majority sharpens with length"
    elif hi(dP) < 0.03:
        v = "no length-driven minority capture" if sd(dP) <= 0.012 else (
            "not decidable (underpowered)" if 2.49 * sd(dP) > 0.06 else f"undecided (underpowered: realized MDE {2.49 * sd(dP):.3f})")
    elif lo(dP) > 0 and hi(P1) <= bound:
        v = "length raises minority influence within proportional (compatible with degradation)"
    else:
        v = "undecided"
    neg = v in ("majority sharpens with length", "no length-driven minority capture",
                "length raises minority influence within proportional (compatible with degradation)")
    res.update(verdict=v, support=False, negative=neg, against_length_only=v == "minority over-representation without length dependence")
    return res


def planted(D, items, W, Ls, fits, bound, n=200):
    """G-D1-null (iii): planted-null data sets from the fitted null (a kernel drawn from the eight; context-level residual noise and item
    random effects) and planted-effect sets (+1.5 nats towards B in mixed contexts at L* only); rate of verdict 4's condition."""
    rng = np.random.default_rng(1101); rates = {}
    pre = {}
    for L in (0, Ls):
        f = fits[L]; sims = null_curves(D, items, L, f, "all")
        pre[L] = (sims, f["resid_sd"])
    for tag, bonus in (("null", 0.0), ("effect", 1.5)):
        hits = 0
        for _ in range(n):
            kk = int(rng.integers(len(KERNELS))); P = {}
            for L in (0, Ls):
                (l0, lk, _), sdv = pre[L][0][kk], pre[L][1]
                tau = 0.5 * sdv; u = rng.normal(0, tau, len(items))
                e = lambda: rng.normal(0, sdv / math.sqrt(24), len(items))
                l0p = l0 + u + e(); lkp = {k: lk[k] + u + e() + (bonus if L == Ls else 0.0) for k in lk}
                P[L] = phi(W, l0p, lkp, range(1, 5))[0]
            hits += int(lo(P[Ls]) > bound and lo(P[Ls] - P[0]) > 0)
        rates[tag] = hits / n
    return dict(null_rate=rates["null"], effect_rate_power=rates["effect"], n=n, rule="verdict 4 in <= 5 % of planted nulls")


def d1b(D, items, W, Ls, rec_ok):
    out = {}
    if not rec_ok:
        out["labels"] = ["not testable (recency not detected)"]
    curves = {}
    for L in ([0, Ls] if Ls else [0]):
        l0, lk = curve(D, items, L, range(1, 6), arm="rel", orders=(0,))
        m = np.stack([wm(W, l0)] + [wm(W, lk[k]) for k in range(1, 6)], 1)          # [B+1, 6] l_rel(k), k = 0..5
        a = np.full(len(m), np.nan); cens = m[:, 5] < 0
        for r in range(len(m)):
            for k in range(1, 6):
                if m[r, k - 1] < 0 <= m[r, k]:
                    a[r] = (k - 1 + (0 - m[r, k - 1]) / (m[r, k] - m[r, k - 1])) / 10; break
        a[cens] = 0.55; curves[L] = (a, cens, m)
        g = 1 / (1 + np.exp(-m[:, 5]))
        out[f"L{L}"] = dict(alpha_star_rel=S_(a), censored_share=float(cens[1:].mean()), point_censored=bool(cens[0]), gamma_rec=S_(g),
                            eyal_sirer=S_((1 - g) / (3 - 2 * g)),
                            verdict_L0=("release threshold below 1/2 at L0" if (L == 0 and hi(a) < 0.5 and not cens[0]) else None))
    if Ls:
        (a0, c0, _), (a1, c1, _) = curves[0], curves[Ls]; d = a1 - a0
        out["d_alpha"] = S_(d)
        if c0[1:].mean() > 0.2 or c1[1:].mean() > 0.2:
            v = "not decidable (censored in > 20 % of replicates)"
        elif hi(d) < 0:
            v = "release threshold falls with length"
        elif lo(d) > 0:
            v = "release threshold rises with length"
        elif inside(d, -0.05, 0.05):
            v = "release threshold stable with length"
        else:
            v = "undecided"
        out["verdict"] = labelled_v(v, "length acts through position weighting" if "falls" in v else None, LBL_SEC)
    return out


def labelled_v(v, *labels):
    return K.labelled(v, *labels)


def d1c(D, items, c, L):
    W = weights(item_groups(items, c.n_streams), 11)
    def cell(name):
        return np.nanmean(np.stack([D.A(items, arm="c", swap=s, order=o, arr=name, L=L) for s in (0, 1) for o in (1, 2)]), (0, 2))
    T1, T2, T4 = cell("T1"), cell("T2"), cell("T4")
    l0, lk = curve(D, items, L, (2,)); T3 = lk[2]
    dpc, dcs, dsh, dg = wm(W, T4) - wm(W, T1), wm(W, T2) - wm(W, T1), wm(W, T3) - wm(W, T1), wm(W, T3) - wm(W, l0)
    with np.errstate(all="ignore"):
        rcs, rsh = dcs / dpc, dsh / dpc
    if lo(dg) <= 0 or lo(dpc) <= 0:
        v = "not decidable (readout saturated in the count arm)"
    elif hi(rsh) < -0.4 and hi(rcs) < 0.45:
        v = "count-driven (in-context evidence accumulates; the hash-share mapping fails)"
    elif lo(rsh) > -0.4 and lo(rcs) > 0.2:
        v = "share-driven (evidence is averaged; the hash-share mapping holds)"
    elif hi(rsh) < -0.4 and lo(rcs) > 0.45:
        v = "mixed (count and share both matter)"
    else:
        v = "undecided"
    return dict(verdict=K.labelled(v, LBL_SEC), d_pc=S_(dpc), d_cs=S_(dcs), d_sh=S_(dsh), d_g=S_(dg), r_cs=S_(rcs), r_sh=S_(rsh))


def d1d(D, items, c, Ls, d1a_res):
    if not Ls:
        return dict(verdict=K.coded("NA:not_run:no gated long length"))
    W = weights(item_groups(items, c.n_streams), 11); out = {}
    def per(L, key, f):
        comp = [[np.nan] * 2 for _ in items]; nonc = [[np.nan] * 2 for _ in items]
        for u, i in enumerate(items):
            for s in (0, 1):
                a = [f(D.get("main", i, s, 5, o, L=L)) for o in (1, 2)]; b = [f(D.get("nc", i, s, 0, o, arr="nc", L=L)) for o in (1, 2)]
                comp[u][s] = np.nanmean(a) if np.isfinite(a).any() else np.nan; nonc[u][s] = np.nanmean(b) if np.isfinite(b).any() else np.nan
        with np.errstate(all="ignore"):
            return np.nanmean(np.array(comp) - np.array(nonc), 1)
    get = dict(nll=lambda r: np.mean(r["d"]["nll"]) if r and r.get("d") else np.nan,
               kl=lambda r: r["d"]["kl"] if r and r.get("d") and r["d"]["kl"] is not None else np.nan,
               fact=lambda r: np.mean(r["d"]["fact"]) if r and r.get("d") else np.nan)
    for key, f in get.items():
        C0, C1 = wm(W, per(0, key, f)), wm(W, per(Ls, key, f)); out[key] = dict(C_L0=S_(C0), C_Ls=S_(C1), growth=S_(C1 - C0))
        if key == "nll":
            if lo(C1) > 0.01 and lo(C1 - C0) > 0:
                v = "losing game (competition costs coherence beyond a non-competing second entity, growing with length)"
            elif lo(C1) > 0.01:
                v = "cost without length growth"
            elif inside(C1, -0.01, 0.01):
                v = "no system cost"
            else:
                v = "undecided"
            out["verdict"] = K.labelled(v, LBL_SEC)
            cl = per(Ls, key, f)
    try:
        from scipy.stats import spearmanr
        l0, lk = curve(D, items, Ls, range(1, 5))
        with np.errstate(all="ignore"):
            exc = np.nanmean([(lk[k] - l0) / (-2 * l0) - k / 10 for k in range(1, 5)], 0)
        ok = np.isfinite(cl) & np.isfinite(exc)
        out["link_spearman"] = K.val(spearmanr(cl[ok], exc[ok])[0] if ok.sum() > 2 else None, "NaN:too_few_items")
    except Exception:
        out["link_spearman"] = K.coded("NaN:not_computed")
    return out


def d1e(D, items, c, Ls):
    W = weights(item_groups(items, c.n_streams), 11); out = {}
    cb = np.array([D.att(D.get("ecb", i, 0, 0, 0, arr="closed", fam="asym")) for i in items])      # towards the true value
    keep = cb > 0; out["closed_book"] = dict(retained=int(keep.sum()), of=int(keep.size))
    g = {}
    for L in ([0, Ls] if Ls else [0]):
        par = np.nanmean(np.stack([np.where(keep, D.A(items, arm="e", swap=0, k=5, order=o, L=L, fam="asym"), np.nan) for o in (1, 2)] +
                                  [np.where(keep, -D.A(items, arm="e", swap=1, k=5, order=o, L=L, fam="asym"), np.nan) for o in (1, 2)]), (0, 2))
        g[L] = wm(W, par); out[f"gamma_prior_L{L}"] = S_(g[L])
        for sw, nm in ((0, "true_minority"), (1, "cp_minority")):
            l0 = np.nanmean(np.where(keep, D.A(items, arm="e", swap=sw, arr="pure", L=L, fam="asym"), np.nan), 1)
            lk = {k: np.nanmean(np.stack([np.where(keep, D.A(items, arm="e", swap=sw, k=k, order=o, L=L, fam="asym"), np.nan) for o in (1, 2)]), (0, 2))
                  for k in (2, 4)}
            out[f"Phi24_{nm}_L{L}"] = S_(phi(W, l0, lk, (2, 4))[0])
    out["prior_asymmetry"] = K.labelled("prior asymmetry present" if lo(g[0]) > 0 else "prior asymmetry not detected", "positive control")
    if Ls:
        d = g[Ls] - g[0]; out["d_gamma"] = S_(d)
        v = "prior fallback grows with length" if lo(d) > 0 else ("stable" if inside(d, -0.5, 0.5) else "undecided")
        out["verdict"] = K.labelled(v, LBL_SEC)
    return out


def d1f(D, items, c, Ls, d1a_res):
    if not Ls:
        return dict(verdict=K.coded("NA:not_run:no gated long length"))
    W = weights(item_groups(items, c.n_streams), 11)
    l0s = np.nanmean(np.stack([D.A(items, arm="f", swap=s, arr="pure", L=Ls, filler="shuf") for s in (0, 1)]), (0, 2))
    _, lks = curve(D, items, Ls, (2, 4), arm="f", filler="shuf")
    if not np.isfinite(l0s).any():
        return dict(verdict=K.coded("NA:not_run:D1-f not run"))
    l0c, lkc = curve(D, items, Ls, (2, 4))
    Dd = phi(W, l0s, lks, (2, 4))[0] - phi(W, l0c, lkc, (2, 4))[0]
    dlo = (d1a_res.get("dPhi") or {}).get("lo"); dlo = dlo if isinstance(dlo, float) else float("nan")
    if lo(Dd) > 0 or hi(Dd) < 0:
        v = f"form of the long context matters (sign {'+' if pt(Dd) > 0 else '-'})"
    elif inside(Dd, -0.08, 0.08) and dlo > 0:
        v = "size, not form"
    else:
        v = "undecided"
    return dict(verdict=K.labelled(v, LBL_SEC), D=S_(Dd))


def d1r(D, items, aitems, c):
    W = weights(item_groups(items, c.n_streams), 11); out = {}
    l0c, lkc = curve(D, items, 0, range(1, 5), orders=(1,)); Pc = phi(W, l0c, lkc, range(1, 5))[0]
    for e in ("seg", "char"):
        l0 = np.nanmean(np.stack([D.A(items, arm="r", swap=s, arr="pure", L=0, enc=e) for s in (0, 1)]), (0, 2))
        lk = {k: np.nanmean(np.stack([D.A(items, arm="r", swap=s, k=k, order=1, L=0, enc=e) for s in (0, 1)]), (0, 2)) for k in range(1, 5)}
        Pe = phi(W, l0, lk, range(1, 5))[0]
        with np.errstate(all="ignore"):
            rho = wm(W, l0) / wm(W, l0c)
        out[e] = dict(Phi=S_(Pe), rho=S_(rho), D=S_(Pe - Pc))
        per = {}
        for j in range(6):
            a = np.nanmean(np.stack([D.A(items, arm="r", swap=s, arr="pure", L=0, enc=e)[:, j] for s in (0, 1)]), 0)
            b = np.nanmean(np.stack([D.A(items, arm="pure", swap=s, L=0)[:, j] for s in (0, 1)]), 0)
            with np.errstate(all="ignore"):
                per[S.SLOTS[j]] = K.val(np.nanmean(a) / np.nanmean(b))
        out[e]["rho_per_attribute"] = per
    l0 = np.nanmean(np.stack([D.A(items, arm="r", swap=s, arr="pure", L=0, enc="char") for s in (0, 1)]), (0, 2))
    lk = {k: np.nanmean(np.stack([D.A(items, arm="r", swap=s, k=k, order=1, L=0, enc="char") for s in (0, 1)]), (0, 2)) for k in range(1, 5)}
    Pch = phi(W, l0, lk, range(1, 5))[0]; Dd = Pch - Pc
    with np.errstate(all="ignore"):
        rho = pt(wm(W, l0) / wm(W, l0c))
    if not (rho >= 0.25):
        v = "not decidable (character-level reading fails)"
    elif inside(Dd, -0.07, 0.07):
        v = "capture pattern survives retokenisation (above token identity)"
    elif (lo(Dd) > 0 or hi(Dd) < 0) and abs(pt(Dd)) >= 0.07:
        v = "token-identity-bound"
    else:
        v = "undecided"
    out["verdict"] = K.labelled(v, LBL_SEC)
    Wa = weights(item_groups(aitems, c.n_streams), 11)
    gc = np.nanmean(np.stack([D.A(aitems, arm="r", swap=0, k=5, order=1, fam="asym", enc="char"),
                              -D.A(aitems, arm="r", swap=1, k=5, order=1, fam="asym", enc="char")]), (0, 2))
    gn = np.nanmean(np.stack([D.A(aitems, arm="e", swap=0, k=5, order=1, fam="asym"), -D.A(aitems, arm="e", swap=1, k=5, order=1, fam="asym")]), (0, 2))
    out["gamma_prior_char_vs_can"] = dict(char=S_(wm(Wa, gc)), can=S_(wm(Wa, gn)))
    return out


def d1v(R):
    res = R["results"]; out = {}
    for b in ("b14", "b18"):
        fam_ret = {}
        for fam in ("symmetric_gender", "directional_capital", "directional_partwhole"):
            can, ch = res[f"{fam}|{b}|can"]["forward"]["acc_excl"], res[f"{fam}|{b}|char"]["forward"]["acc_excl"]
            lab = "offset structure survives retokenisation" if ch >= can - 0.15 else ("token-bound" if ch < 0.5 * can else "intermediate")
            r = res[f"{fam}|{b}|char"]["retention"]["forward"]; fam_ret[fam] = r if isinstance(r, float) else float("nan")
            out[f"{fam}|{b}"] = dict(label=lab, canonical=can, char=ch, seg=res[f"{fam}|{b}|seg"]["forward"]["acc_excl"],
                                     reverse_gap_can=can - res[f"{fam}|{b}|can"]["reverse"]["acc_excl"],
                                     floor=res[f"{fam}|{b}|can"]["floor"]["forward"]["acc_excl"])
        d = min(fam_ret["directional_capital"], fam_ret["directional_partwhole"])
        out[f"asymmetry|{b}"] = ("asymmetric relations reach the floor faster" if d <= fam_ret["symmetric_gender"] - 0.2 else
                                 "no faster floor for asymmetric relations")
    out["labels"] = ["descriptive (the user's aside; no H-DUAL verdict)"]
    return out


def d1s(ctx, D, items, c, Ls):
    W = weights(item_groups(items, c.n_streams), 11); L = c.L_long[0]
    p = ctx.input("r1b_D_A", "d1/d1s_glce.json", required=False); g = K.load_json(p) if p else {}
    m06 = [x for r in D.ix.values() if r["model"] == C.M06 for x in r["M"]]
    l0a = np.nanmean(np.stack([D.A(items, arm="s", swap=s, arr="pure", L=0, model=C.M06) for s in (0, 1)]), (0, 2))
    if not m06:
        return dict(verdict=K.coded("NA:not_run:input missing"))
    rng06 = pt(-2 * wm(W, l0a))
    if (np.median(m06) < 0.9 or rng06 < 2) and not ctx.mock:
        return dict(verdict="not testable (0.6B readout)", median_mass=float(np.median(m06)), range_L0=rng06)
    def d24(model, arm, Lx):
        out = []
        for LL in (0, Lx):
            l0 = np.nanmean(np.stack([D.A(items, arm=arm, swap=s, arr="pure", L=LL, model=model) for s in (0, 1)]), (0, 2))
            _, lk = curve(D, items, LL, (2, 4), arm=arm, model=model)
            out.append(phi(W, l0, lk, (2, 4))[0])
        return out[1] - out[0]
    if not g.get("path"):
        return dict(verdict="not testable (engine parity)")
    d06 = d24(C.M06, "s", L)
    l0x, lkx = curve(D, items, 0, (2, 4)); l0y, lky = curve(D, items, Ls, (2, 4)) if Ls else (None, None)
    if Ls is None:
        return dict(verdict=K.coded("NA:not_run:no gated long length"), dPhi24_06=S_(d06))
    d17 = phi(W, l0y, lky, (2, 4))[0] - phi(W, l0x, lkx, (2, 4))[0]; dd = d17 - d06
    v = ("length effect larger in 1.7B" if lo(dd) > 0 else ("larger in 0.6B" if hi(dd) < 0 else
         ("no size difference" if inside(dd, -0.08, 0.08) else "undecided")))
    return dict(verdict=K.labelled(v, "descriptive", *(["1.7B at L* = %d" % Ls] if Ls != L else [])), contrast=S_(dd), dPhi24_17=S_(d17),
                dPhi24_06=S_(d06))


def coexistence(D, items, Ls):
    """CI = a + b from h(X_AB) - h(X_N) = a u_A + b u_B per item-attribute, block, length (parity k = 5; matched letter assignments)."""
    out = {}
    for L in Ls:
        for blk in (14, 18):
            CI, bal, R2 = [], [], []
            for i in items:
                xn = D.get("xn", i, L=L); pa = {s: D.get("pure", i, s, L=L) for s in (0, 1)}
                if xn is None or any(v is None or f"h{blk}" not in v for v in pa.values()) or f"h{blk}" not in xn:
                    continue
                hq = lambda r: {tuple(q): r[f"h{blk}"][n].double().numpy() for n, q in enumerate(r["qs"])}
                HX, HP = hq(xn), {s: hq(pa[s]) for s in (0, 1)}
                for s in (0, 1):
                    for o in (1, 2):
                        r = D.get("main", i, s, 5, o, L=L)
                        if r is None or f"h{blk}" not in r:
                            continue
                        for n, (j, bf) in enumerate(r["qs"]):
                            hab = r[f"h{blk}"][n].double().numpy(); hn = HX[(j, bf if s == 0 else (not bf))]
                            uA, uB = HP[s][(j, bf)] - hn, HP[1 - s][(j, not bf)] - hn; y = hab - hn
                            X = np.stack([uA, uB], 1); co, *_ = np.linalg.lstsq(X, y, rcond=None)
                            pred = X @ co; r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
                            CI.append(co.sum()); bal.append(co[1] / co.sum() if abs(co.sum()) > 1e-9 else np.nan); R2.append(r2)
            m = float(np.mean(CI)) if CI else float("nan")
            out[f"L{L}|b{blk}"] = dict(CI=K.dist(CI), balance=K.dist(bal), R2=K.dist(R2),
                                       label=("superposed" if m >= 1.5 else ("mixed" if m <= 1.2 else "intermediate")) if CI else
                                       K.coded("NaN:no_states"))
    return out


# ------------------------------------------------------------------------------------------------------------------------------------
# D2
# ------------------------------------------------------------------------------------------------------------------------------------
ROW = dict(OPP=("pro|anti", "anti|pro"), NEU=("pro|neutral", "anti|neutral"), ALI=("pro|pro", "anti|anti"), MON=("pro|None", "anti|None"),
           DOM=("pro|dom_anti", "anti|dom_pro"))


def d2_arrays(recs, n_p):
    """V[(length, cond, s1|s2, turn, field)] -> [n_p] arrays."""
    V = {}
    for r in recs:
        k = (r["length"], r["cond"], f"{r['s1']}|{r['s2']}", r["turn"])
        for f in ("D1", "D2", "rep4", "share_lines_S1"):
            x = r[f]; x = x if isinstance(x, (int, float)) else np.nan
            V.setdefault(k + (f,), np.full(n_p, np.nan))[r["p"]] = x
    return V


def d2_effects(V, W, L, n_p):
    nan = np.full(n_p, np.nan); g = lambda c, row, t, f="D1": V.get((L, c, row, t, f), nan)
    E = {}
    for c, (rp, ra) in ROW.items():
        for t in (1, 2, 3):
            E[("e11", c, t)] = wm(W, (g(c, rp, t) - g(c, ra, t)) / 2)
    for c, rows_ in (("OPP", ("anti|pro", "pro|anti")), ("ALI", ("pro|pro", "anti|anti")), ("DOM", ("anti|dom_pro", "pro|dom_anti"))):
        for t in (1, 2, 3):
            E[("e22", c, t)] = wm(W, (g(c, rows_[0], t, "D2") - g(c, rows_[1], t, "D2")) / 2)
    for t in (1, 2, 3):
        d1 = lambda a, b_: g("OPP" if a != b_ else "ALI", f"{a}|{b_}", t); d2 = lambda a, b_: g("OPP" if a != b_ else "ALI", f"{a}|{b_}", t, "D2")
        E[("e11x", t)] = wm(W, ((d1("pro", "pro") - d1("anti", "pro")) + (d1("pro", "anti") - d1("anti", "anti"))) / 4)
        E[("c12", t)] = wm(W, ((d1("pro", "pro") - d1("pro", "anti")) + (d1("anti", "pro") - d1("anti", "anti"))) / 4)
        E[("e22x", t)] = wm(W, ((d2("pro", "pro") - d2("pro", "anti")) + (d2("anti", "pro") - d2("anti", "anti"))) / 4)
        E[("c21x", t)] = wm(W, ((d2("pro", "pro") - d2("anti", "pro")) + (d2("pro", "anti") - d2("anti", "anti"))) / 4)
        E[("c21N", t)] = wm(W, (g("NEU", "pro|neutral", t, "D2") - g("NEU", "anti|neutral", t, "D2")) / 2)
    return E


def persist(E, key):
    e1, e2, e3 = E[key + (1,)], E[key + (2,)], E[key + (3,)]
    defined = lo(e1) > 0 or hi(e1) < 0
    with np.errstate(all="ignore"):
        return (e2 + e3) / (2 * e1), defined


def _ana_d2(ctx):
    torch = K._torch(); c = S.cfg(ctx)
    p = ctx.input("r1b_D_B", "d2/read.pt", required=False)
    if not p:
        raise K.NotRun("input missing")
    recs = torch.load(p, weights_only=False)["records"]; n_p = c.d2_n
    fb = K.load_json(ctx.input("r1b_D_B", "d2/fallback.json", required=False) or "", {}) or {}
    nsec = 48 if (fb.get("secondary_48") or ctx.trimmed("T4")) and not c.mock else c.d2_sec
    V = d2_arrays(recs, n_p); groups = [[p_ for p_ in range(n_p) if p_ % c.n_streams == s] for s in range(c.n_streams)]
    W = weights(groups, 12); Wsec = W * (np.arange(n_p) < nsec)[None]
    out = dict(n_secondary=nsec)
    g3 = K.load_json(ctx.input("r1b_D_B", "d2/glce3.json", required=False) or "", {}) or {}
    rd = K.load_json(ctx.input("r1b_D_B", "d2/d2_read.json", required=False) or "", {}) or {}
    out["G-LCE-3"] = dict(passed=g3.get("passed")); out["G-M2"] = rd.get("G_M2_passed")
    E = {L: d2_effects(V, W, L, n_p) for L in ("long", "short")}; Es = {L: d2_effects(V, Wsec, L, n_p) for L in ("long", "short")}
    pcs = {}
    for L in ("long", "short"):
        e, es = E[L], Es[L]
        i_ok = all(lo(e[("e11", cc, 1)]) > 0 for cc in ("OPP", "NEU")) and all(lo(es[("e11", cc, 1)]) > 0 for cc in ("ALI", "MON", "DOM")) and \
            all(lo((es if cc != "OPP" else e)[("e22", cc, 1)]) > 0 for cc in ("OPP", "ALI", "DOM"))
        ii_ok = abs(pt(es[("c12", 1)])) <= 0.3 * pt(es[("e11x", 1)]) and abs(pt(es[("c21x", 1)])) <= 0.3 * pt(es[("e22x", 1)])
        iii_ok = abs(pt(e[("c21N", 1)])) <= 0.3 * pt(e[("e11", "NEU", 1)])
        pcs[L] = dict(i=bool(i_ok), ii=bool(ii_ok), iii=bool(iii_ok))
    out["G-D2-PC"] = pcs
    eng_ok = bool(g3.get("passed")) or ctx.mock
    m2 = lambda L: bool((rd.get("G_M2_passed") or {}).get(L)) or ctx.mock
    Pi = {}
    for L in ("long", "short"):
        for cc in ("OPP", "NEU"):
            Pi[(L, cc)] = persist(E[L], ("e11", cc))
        for cc in ("ALI", "MON", "DOM"):
            Pi[(L, cc)] = persist(Es[L], ("e11", cc))
        Pi[(L, "OPP2")] = persist(E[L], ("e22", "OPP"))
    dPi = {L: Pi[(L, "OPP")][0] - Pi[(L, "NEU")][0] for L in ("long", "short")}
    out["Pi"] = {f"{L}|{cc}": dict(Pi=S_(v[0]), defined=v[1]) for (L, cc), v in Pi.items()}
    out["dPi1"] = {L: S_(v) for L, v in dPi.items()}
    # D2-a
    null = d2_null(V, W, n_p, ctx)
    out["G-D2-null"] = null
    rep_l = np.nanmean([V.get(("long", "OPP", r, t, "rep4"), np.full(n_p, np.nan)) for r in ROW["OPP"] for t in (2, 3)], 0) - \
        np.nanmean([V.get(("long", "NEU", r, t, "rep4"), np.full(n_p, np.nan)) for r in ROW["NEU"] for t in (2, 3)], 0)
    rep_d = wm(W, rep_l)
    adj = pi_adj(V, W, n_p, "long")
    dl = dPi["long"]
    if not (pcs["long"]["i"] and pcs["long"]["ii"]) or not eng_ok or not m2("long"):
        v = "not testable (two-entity readout failed at L_long)"
    elif null["null_rate"] > 0.10:
        v = "not decidable (rule not specific at the measured noise)"
    elif not (Pi[("long", "OPP")][1] and Pi[("long", "NEU")][1]):
        v = "Pi not defined (turn-1 stance not resolved)"
    elif lo(dl) > 0:
        v = "two entities keep each other going (catfish)"
    elif hi(dl) < 0:
        v = "competitor dilutes (no catfish)"
    elif inside(dl, -0.15, 0.15):
        v = "no catfish effect" if sd(dl) <= 0.075 else ("not decidable (underpowered)" if 2.49 * sd(dl) > 0.30 else "undecided (underpowered)")
    else:
        v = "undecided"
    lab = []
    if v.startswith("two entities"):
        lab += (["small"] if pt(dl) < 0.15 else []) + (["confounded with repetition"] if lo(rep_d) > 0.05 else []) + \
               (["not robust to production share"] if lo(adj) <= 0 else [])
    out["D2-a"] = dict(verdict=K.labelled(v, *lab), dPi1=S_(dl), dPi1_adj=S_(adj), rep4_opp_minus_neu=S_(rep_d), p=p_gt(dl, 0.0))
    spec = {}
    if Pi[("long", "ALI")][1]:
        d_oa, d_an = Pi[("long", "OPP")][0] - Pi[("long", "ALI")][0], Pi[("long", "ALI")][0] - Pi[("long", "NEU")][0]
        spec["OPP-ALI"] = S_(d_oa); spec["ALI-NEU"] = S_(d_an)
        spec["label"] = ("competition-specific" if lo(d_oa) > 0 else ("any engaged partner sustains" if lo(d_an) > 0 and lo(d_oa) <= 0 <= hi(d_oa)
                                                                       else None))
    if Pi[("long", "MON")][1]:
        d_om = Pi[("long", "OPP")][0] - Pi[("long", "MON")][0]; spec["OPP-MON"] = S_(d_om)
        spec["competitor_vs_no_partner"] = ("higher" if lo(d_om) > 0 else ("lower" if hi(d_om) < 0 else
                                            ("no difference" if inside(d_om, -0.3, 0.3) else "undecided")))
    out["specificity"] = K.labelled(spec, LBL_SEC)
    # D2-b
    ddl = dPi["long"] - dPi["short"]
    if not (pcs["long"]["i"] and pcs["long"]["ii"] and pcs["short"]["i"] and pcs["short"]["ii"]):
        vb = "not testable"
    elif lo(ddl) > 0:
        vb = "hyper-long context strengthens two-entity sustainment"
    elif hi(ddl) < 0:
        vb = "weakens"
    elif inside(ddl, -0.25, 0.25):
        vb = "no length interaction" if sd(ddl) <= 0.11 else "undecided (underpowered)"
    else:
        vb = "undecided"
    out["D2-b"] = dict(verdict=vb, ddPi1=S_(ddl), p=p_gt(ddl, 0.0))
    # D2-c
    for L in ("long", "short"):
        es = Es[L]
        with np.errstate(all="ignore"):
            mu = ((es[("c12", 2)] + es[("c12", 3)]) / 2) / es[("e11x", 1)]
        P1o, P2o = Pi[(L, "OPP")][0], Pi[(L, "OPP2")][0]
        if lo(mu) > 0.10:
            vc = "merging (speaker 1 adopts the competitor's stance)"
        elif hi(mu) < -0.10:
            vc = "polarisation (moves away)"
        elif inside(mu, -0.20, 0.20) and lo(P1o) > 0 and lo(P2o) > 0:
            vc = "coexistence"
        elif (hi(P1o) < 0.10 and lo(P2o) > 0.10) or (hi(P2o) < 0.10 and lo(P1o) > 0.10):
            vc = "one entity captured"
        else:
            vc = "undecided"
        out[f"D2-c_{L}"] = dict(verdict=vc, mu=S_(mu), p_mu_below_010=K.p_one_sided(mu[1:], 0.10, "less"))
    out["D2-c"] = out["D2-c_long"]
    # D2-g
    for L in ("long", "short"):
        if Pi[(L, "DOM")][1] and Pi[(L, "OPP")][1]:
            dd = Pi[(L, "DOM")][0] - Pi[(L, "OPP")][0]
            vg = ("a dominant competitor suppresses the other (tournament-type reversal)" if hi(dd) < 0 else
                  ("dominance raises persistence" if lo(dd) > 0 else ("no dominance effect" if inside(dd, -0.3, 0.3) else "undecided")))
            out[f"D2-g_{L}"] = dict(verdict=K.labelled(vg, LBL_SEC, "does not count against H-DUAL"), diff=S_(dd))
        else:
            out[f"D2-g_{L}"] = dict(verdict="Pi not defined (turn-1 stance not resolved)")
    out["D2-f"] = d2f(ctx, recs, W, n_p)
    out["descriptive"] = d2_desc(recs, V, n_p)
    ctx.write_json("d_ana/d2.json", out, config=dict(seed=12, cfg=dict(c)), seeds=dict(bootstrap=12),
                   labels=["MOCK (tiny random model)"] if ctx.mock else [])


def pi_adj(V, W, n_p, L):
    """Matched-presence Delta Pi_1: turn-2 / 3 effects only on prompts whose two rows give speaker 1 25-75 % of the lines."""
    nan = np.full(n_p, np.nan); res = {}
    for cc in ("OPP", "NEU"):
        rp, ra = ROW[cc]; e = {}
        for t in (1, 2, 3):
            d = (V.get((L, cc, rp, t, "D1"), nan) - V.get((L, cc, ra, t, "D1"), nan)) / 2
            if t > 1:
                sh = np.stack([V.get((L, cc, r_, t, "share_lines_S1"), nan) for r_ in (rp, ra)])
                d = np.where(((sh >= 0.25) & (sh <= 0.75)).all(0), d, np.nan)
            e[t] = wm(W, d)
        with np.errstate(all="ignore"):
            res[cc] = (e[2] + e[3]) / (2 * e[1])
    return res["OPP"] - res["NEU"]


def d2_null(V, W, n_p, ctx, n=200):
    """G-D2-null: planted per-prompt chains with the measured turn-1 effects, per-turn carry and noise and no OPP-NEU difference (null)
    or Pi_1^OPP + 0.2 (effect); rate of D2-a's verdict 4 (lower bound of Delta Pi_1 > 0)."""
    rng = np.random.default_rng(1201); nan = np.full(n_p, np.nan); n = n if not ctx.mock else 20
    per = {}
    for cc in ("OPP", "NEU"):
        rp, ra = ROW[cc]
        per[cc] = [(V.get(("long", cc, rp, t, "D1"), nan) - V.get(("long", cc, ra, t, "D1"), nan)) / 2 for t in (1, 2, 3)]
    e1 = np.nanmean(np.concatenate([per[cc][0] for cc in per])); sds = [np.nanstd(np.concatenate([per[cc][t] for cc in per])) for t in range(3)]
    with np.errstate(all="ignore"):
        Pi0 = np.nanmean(np.concatenate([(per[cc][1] + per[cc][2]) for cc in per])) / (2 * e1) if e1 else 0.5
    Pi0 = float(np.clip(Pi0 if np.isfinite(Pi0) else 0.5, -0.4, 1.5)); kk = lambda P: (-1 + math.sqrt(max(1 + 8 * P, 0))) / 2
    rates = {}
    for tag, add in (("null", 0.0), ("effect", 0.2)):
        hits = 0
        for _ in range(n):
            Pis = {}
            for cc in ("OPP", "NEU"):
                K_ = kk(Pi0 + (add if cc == "OPP" else 0.0))
                e = [e1 + rng.normal(0, sds[0] or 1e-6, n_p)]
                e += [e1 * K_ ** t + rng.normal(0, sds[t] or 1e-6, n_p) for t in (1, 2)]
                m = [wm(W, x) for x in e]
                with np.errstate(all="ignore"):
                    Pis[cc] = (m[1] + m[2]) / (2 * m[0])
            hits += int(lo(Pis["OPP"] - Pis["NEU"]) > 0)
        rates[tag] = hits / n
    return dict(null_rate=rates["null"], effect_rate_power=rates["effect"], n=n, Pi_pooled=Pi0, rule="verdict 4 in <= 10 % of planted nulls")


def axes14(ctx):
    """u_T_perp(14), u*_14 (SO-AX, r1b_R_S0) and u_T^v(14) (AX-1b, r1b_R_S2_14); MOCK placeholders -> random unit axes (labelled)."""
    torch = K._torch(); out, lab = {}, []
    p = ctx.input("r1b_R_S0", "so_ax/so_ax_b14.pt", required=False); q = ctx.input("r1b_R_S2_14", "ax1b/ax1b_b14.pt", required=False)
    a = torch.load(p, weights_only=False) if p else {}; b = torch.load(q, weights_only=False) if q else {}
    for name, src, key in (("u_T_perp", a, "u_T_perp"), ("u_star", a, "u_star"), ("u_T_v", b, "u_T_v")):
        if key in src:
            out[name] = torch.as_tensor(src[key]).double().reshape(-1)
        elif ctx.mock:
            g = torch.Generator().manual_seed(len(name)); out[name] = torch.randn(2048 if not ctx.mock else 64, generator=g, dtype=torch.float64)
            lab.append(f"{name}: MOCK random axis")
        else:
            raise K.NotRun("input missing (SO-AX)")
    if (a.get("label") or "") == "SO-AX reproduction failed":
        lab.append("SO-AX reproduction failed")
    return out, lab


def d2f(ctx, recs, W, n_p):
    """D2-f (descriptive): K(t) = u . d^(t+1) / u . d^(t), d^(t) = mean_p (x_14(pro) - x_14(anti)) / 2 (speaker 1), NEU and OPP, on
    u_T_perp(14), u*_14, u_T^v(14); G26 definedness and the R6 strings (r2_theta_verdict.rule_R6; transfer part not applicable)."""
    torch = K._torch()
    try:
        ax, lab = axes14(ctx)
    except K.NotRun as e:
        return dict(verdict=K.coded("NA:not_run:input missing"), note=str(e))
    import r2_theta_verdict as TV
    X = {}
    for r in recs:
        X[(r["length"], r["cond"], f"{r['s1']}|{r['s2']}", r["turn"], r["p"])] = r["x14"].double()
    out = {}
    for L in ("long", "short"):
        for cc in ("OPP", "NEU"):
            rp, ra = ROW[cc]
            for an, u in ax.items():
                u = u / u.norm(); proj = {}
                for t in (1, 2, 3):
                    v = np.full(n_p, np.nan)
                    for p_ in range(n_p):
                        a_, b_ = X.get((L, cc, rp, t, p_)), X.get((L, cc, ra, t, p_))
                        if a_ is not None and b_ is not None and a_.numel() == u.numel():
                            v[p_] = float(((a_ - b_) / 2) @ u)
                    proj[t] = wm(W, v)
                for t in (1, 2):
                    with np.errstate(all="ignore"):
                        Kt = proj[t + 1] / proj[t]
                    r6 = TV.rule_R6(Kt, proj[t], np.full_like(Kt, np.nan))
                    r6.pop("transfer", None)
                    out[f"{L}|{cc}|{an}|t{t}"] = K.jsonable(r6)
    tv = os.path.join(K.REPO, "results/ju_space/runde2_ergebnisse/theta_verdict.json")
    ref = (K.load_json(tv, {}).get("blocks", {}).get("14", {}).get("R6", {}) or {}).get("cells") if os.path.exists(tv) else None
    return dict(cells=out, round2_R6_cells=ref, labels=["descriptive (instruction-induced stance; not 1:1 comparable with push-induced R6)"] + lab)


def d2_desc(recs, V, n_p):
    t3 = [r["D1"] for r in recs if r["length"] == "long" and r["cond"] == "NEU" and r["turn"] == 3]
    x = np.asarray(t3, float); bc = None
    if x.size > 3 and np.std(x) > 0:
        from scipy.stats import kurtosis, skew
        g, k_ = skew(x), kurtosis(x); n = x.size
        bc = (g ** 2 + 1) / (k_ + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    prod = {}
    for r in recs:
        v = r["share_lines_S1"]
        if isinstance(v, (int, float)):
            prod.setdefault(f"{r['length']}|{r['cond']}|t{r['turn']}", []).append(v)
    flux = []
    rng = np.random.default_rng(1202)
    cyc = {(1, 1): 0, (1, -1): 1, (-1, -1): 2, (-1, 1): 3}
    seqs = {}
    for r in recs:
        if r["cond"] in ("OPP", "NEU", "ALI", "DOM"):
            seqs.setdefault((r["length"], r["rid"]), {})[r["turn"]] = (int(np.sign(r["D1"]) or 1), int(np.sign(r["D2"]) or 1))
    def fl(ss):
        f, n_ = 0, 0
        for s in ss:
            for a, b_ in zip(s[:-1], s[1:]):
                d = (cyc[b_] - cyc[a]) % 4
                f += (d == 1) - (d == 3); n_ += 1
        return f / max(n_, 1)
    ss = [[v[t] for t in sorted(v)] for v in seqs.values() if len(v) == 3]
    obs = fl(ss); floor = [fl([list(rng.permutation(s)) for s in ss]) for _ in range(200)]
    return dict(D1_turn3_NEU_long=K.dist(t3), bimodality_coefficient=K.val(bc, "NaN:too_few_texts"),
                production_share_S1={k: K.dist(v) for k, v in sorted(prod.items())},
                flux=K.labelled(dict(net_cyclic_flux=obs, shuffle_floor_p95=float(np.percentile(np.abs(floor), 95)) if floor else None),
                                "provisional (UNVERIFIED energy notes)", "no verdict uses it"))


# ------------------------------------------------------------------------------------------------------------------------------------
# D3, D4
# ------------------------------------------------------------------------------------------------------------------------------------
def em2(x, starts=3, iters=200, seed=0):
    """Equal-variance two-component Gaussian mixture by EM (three starts) on rows of x [m, n]: returns dict of arrays (dBIC = BIC1 - BIC2,
    weights, means, sd, Ashman D)."""
    x = np.asarray(x, float); m, n = x.shape; rng = np.random.default_rng(seed)
    mu1, s1 = x.mean(1), x.std(1) + 1e-9
    ll1 = (-0.5 * ((x - mu1[:, None]) / s1[:, None]) ** 2 - np.log(s1[:, None] * math.sqrt(2 * math.pi))).sum(1)
    best = None
    for st in range(starts):
        q = np.sort(x, 1); mA = q[:, max(0, n // 4 - st)] if st < 2 else rng.choice(n, m) * 0 + x.min(1)
        mB = q[:, min(n - 1, 3 * n // 4 + st)] if st < 2 else x.max(1)
        w, s = np.full(m, 0.5), s1.copy()
        for _ in range(iters):
            pa = w[:, None] * np.exp(-0.5 * ((x - mA[:, None]) / s[:, None]) ** 2)
            pb = (1 - w[:, None]) * np.exp(-0.5 * ((x - mB[:, None]) / s[:, None]) ** 2)
            r = pa / np.maximum(pa + pb, 1e-300); w = np.clip(r.mean(1), 1e-6, 1 - 1e-6)
            mA = (r * x).sum(1) / np.maximum(r.sum(1), 1e-12); mB = ((1 - r) * x).sum(1) / np.maximum((1 - r).sum(1), 1e-12)
            s = np.sqrt(((r * (x - mA[:, None]) ** 2 + (1 - r) * (x - mB[:, None]) ** 2).sum(1) / n)) + 1e-9
        ll2 = np.log(np.maximum(w[:, None] * np.exp(-0.5 * ((x - mA[:, None]) / s[:, None]) ** 2) +
                                (1 - w[:, None]) * np.exp(-0.5 * ((x - mB[:, None]) / s[:, None]) ** 2), 1e-300)).sum(1) \
            - n * np.log(s * math.sqrt(2 * math.pi))
        cand = dict(ll2=ll2, w=w, mA=mA, mB=mB, s=s)
        if best is None:
            best = cand
        else:
            take = ll2 > best["ll2"]
            best = {k: np.where(take, cand[k], best[k]) for k in best}
    dbic = (-2 * ll1 + 2 * math.log(n)) - (-2 * best["ll2"] + 4 * math.log(n))
    return dict(dBIC=dbic, minor=np.minimum(best["w"], 1 - best["w"]), mA=best["mA"], mB=best["mB"], s=best["s"],
                ashman=np.sqrt(2) * np.abs(best["mA"] - best["mB"]) / best["s"] / math.sqrt(2))


def split_rule(xs, ys, xn=None, yn=None, persistence=True):
    """SPLIT-RULE on one item: x, y per seed (n = 6-8 means), xn / yn [seeds, 3] per-n values; returns (split, parts)."""
    parts = {}
    for nm, v, vn in (("x", xs, xn), ("y", ys, yn)):
        e = {k: float(a[0]) for k, a in em2(np.asarray(v, float)[None]).items()}
        mid = (e["mA"] + e["mB"]) / 2; lo_, hi_ = sorted((e["mA"], e["mB"])); third = (hi_ - lo_) / 3
        side = np.sign(np.asarray(v) - mid)
        gap = int(((np.asarray(v) > lo_ + third) & (np.asarray(v) < hi_ - third)).sum())
        pers = True
        if persistence and vn is not None:
            sn = np.sign(np.asarray(vn) - mid); pers = all((sn[:, j] == side).mean() >= 0.75 for j in range(sn.shape[1]))
        parts[nm] = dict(dBIC=e["dBIC"], minor=e["minor"], ashman=e["ashman"], gap=gap, persistence=bool(pers), side=side,
                         ok=bool(e["dBIC"] >= 6 and e["minor"] >= 0.2 and e["ashman"] >= 2.5 and gap <= 1 and pers))
    agree = float((parts["x"]["side"] == parts["y"]["side"]).mean())
    agree = max(agree, 1 - agree)
    split = parts["x"]["ok"] and parts["y"]["ok"] and agree >= 0.8
    for p_ in parts.values():
        p_.pop("side")
    parts["agreement"] = agree
    return bool(split), parts


def mcnemar_p(b, c_):
    from scipy.stats import binom
    n = b + c_
    return float(binom.sf(b - 1, n, 0.5)) if n else 1.0


def _ana_d3(ctx):
    torch = K._torch(); c = S.cfg(ctx); out = {}
    p = ctx.input("r1b_D_C", "d3/traj.pt", required=False)
    if not p:
        raise K.NotRun("input missing")
    T = torch.load(p, weights_only=False); st = torch.load(ctx.input("r1b_D_C", "d3/starts.pt"), weights_only=False)
    sj = K.load_json(ctx.input("r1b_D_C", "d3/starts.json")); g4 = K.load_json(ctx.input("r1b_D_C", "d3/d3_gen.json"))["G-LCE-4"]
    items = T["items"]; Ls = T["lengths"]; R = {(r["L"], r["item"]): r for r in T["records"]}
    folds = {i: n % 2 for n, i in enumerate(items)}; uab = {}
    for b in (14, 18):
        for f in (0, 1):
            diffs = [st["reads"][f"{i}|B2"][b] - st["reads"][f"{i}|A2"][b] for i in items if folds[i] != f and f"{i}|B2" in st["reads"]]
            uab[(b, f)] = torch.stack(diffs).mean(0).double() if diffs else None
    proj = lambda h, i, b=18: float(h.double() @ (uab[(b, folds[i])] / uab[(b, folds[i])].norm())) if uab[(b, folds[i])] is not None else np.nan
    res_L, splits = {}, {}
    nI = len(items); need = math.ceil(14 / 16 * nI)
    for L in Ls:
        g0c_l, g0c_y, ret = [], [], []
        per, x8, y1, ydist = {}, {}, [], {}
        for i in items:
            r = R.get((L, i)); r0 = R.get((Ls[0], i))
            if r is None:
                continue
            d0 = r["n0"]["B2"]["l"] - r["n0"]["A2"]["l"]; g0c_l.append(d0 >= 1)
            d00 = r0["n0"]["B2"]["l"] - r0["n0"]["A2"]["l"]; ret.append(d0 / d00 if d00 else np.nan)
            ya = np.mean([proj(s["h"][18][0], i) for s in r["ctrl"]["A2"]]); yb = np.mean([proj(s["h"][18][0], i) for s in r["ctrl"]["B2"]])
            g0c_y.append(yb - ya > 0)
            xs = [np.mean([s["l"][n] for n in (6, 7, 8)]) for s in r["M"]]
            yn = np.array([[proj(s["h"][18][n - 1], i) for n in (6, 7, 8)] for s in r["M"]]); ys = yn.mean(1)
            xn = np.array([[s["l"][n] for n in (6, 7, 8)] for s in r["M"]])
            sp, parts = split_rule(xs, ys, xn, yn)
            per[i] = dict(split=sp, parts=parts)
            y1[i] = np.array([proj(s["h"][18][0], i) for s in r["M"]]); ydist[i] = (ys, np.array([proj(s["h"][18][7], i) for s in r["M"]]))
            pc = [proj(s["h"][18][0], i) for s in r["ctrl"]["A2"] + r["ctrl"]["B2"]]; pcx = [s["l"][1] for s in r["ctrl"]["A2"] + r["ctrl"]["B2"]]
            sp_pc, parts_pc = split_rule(pcx, pc, persistence=False)
            per[i]["planted"] = dict(split=sp_pc, ashman_y=parts_pc["y"]["ashman"],
                                     sep_x=abs(np.mean(pcx[len(pcx) // 2:]) - np.mean(pcx[:len(pcx) // 2])) / (np.std(pcx) + 1e-9),
                                     sep_y=abs(np.mean(pc[len(pc) // 2:]) - np.mean(pc[:len(pc) // 2])) / (np.std(pc) + 1e-9))
        Sn = sum(v["split"] for v in per.values()); splits[L] = {i: v["split"] for i, v in per.items()}
        g0c = sum(g0c_l) >= need and np.nanmedian(ret) >= 0.5 and sum(g0c_y) >= need
        pred = np.mean([pc_detect(v["planted"]["sep_x"], v["planted"]["sep_y"], n=200 if ctx.mock else 2000) for v in per.values()]) if per else np.nan
        obs = np.mean([v["planted"]["split"] for v in per.values()]) if per else np.nan
        g3pc = bool(obs >= pred - 0.25) if per else False
        weak = bool(np.median([v["planted"]["ashman_y"] for v in per.values()]) < 4) if per else False
        Wi = weights([[u] for u in range(len(per))], 13, two=False); its = list(per)
        within = lambda arrs: np.array([np.var(a, ddof=1) if len(a) > 1 else np.nan for a in arrs])
        v8 = within([np.concatenate([ydist[i][0]]) for i in its]); v1 = within([y1[i] for i in its])
        with np.errstate(all="ignore"):
            VR = wm(Wi, v8) / wm(Wi, v1)
            DR = wm(Wi, np.array([abs(np.median(ydist[i][1]) - np.median(y1[i])) for i in its])) / np.sqrt(wm(Wi, v1))
        dgn = np.mean([any(x > 0.3 for x in s["rep4"]) for i in its for s in R[(L, i)]["M"]]) if its else np.nan
        if sj["n_abstained"] > 4:
            v = "not testable (starts not verifiable)"
        elif not g0c and not ctx.mock:
            v = "not testable (order parameter not resolved at L)"
        elif not g3pc and not ctx.mock:
            v = "not testable (split detector failed its positive control)"
        elif L != Ls[0] and not g4.get("passed") and not ctx.mock:
            v = "not testable (engine)"
        elif Sn >= 4:
            v = "two entities (committing)"
        elif lo(VR) > 2:
            v = "one entity, diffusing"
        elif pt(DR) >= 1 and lo(DR) > 0.5 and hi(VR) <= 2:
            v = "one entity, drifting"
        elif hi(VR) <= 2 and pt(DR) < 1:
            v = "one entity, stable"
        else:
            v = "not decidable"
        lab = (["degeneration-dominated"] if dgn > 0.3 else []) + (["positive control weak"] if weak else [])
        res_L[str(L)] = dict(verdict=K.labelled(v, *lab), S=Sn, n_items=len(per), VR=S_(VR), DR=S_(DR), G_D0c=bool(g0c), G_D3pc=dict(
            passed=g3pc, observed=K.val(obs), predicted=K.val(pred)), degenerate_share=K.val(dgn),
            per_item={str(i): dict(split=v_["split"], planted_split=v_["planted"]["split"]) for i, v_ in per.items()})
    out["per_length"] = res_L
    L0, L1 = Ls[0], Ls[-1]; b = sum(splits[L1].get(i, False) and not splits[L0].get(i, False) for i in items)
    c_ = sum(splits[L0].get(i, False) and not splits[L1].get(i, False) for i in items)
    p_up, p_dn = mcnemar_p(b, c_), mcnemar_p(c_, b); S0, S1 = res_L[str(L0)]["S"], res_L[str(L1)]["S"]
    nt = [L for L in Ls if res_L[str(L)]["verdict"]["value"].startswith("not testable")]
    if nt:
        v = f"not decidable (not testable at {', '.join(map(str, nt))})"
    elif S1 >= 4 and p_up <= 0.05:
        v = "split grows with length"
    elif S0 >= 4 and p_dn <= 0.05:
        v = "split shrinks with length (urn-like)"
    elif S0 >= 4 and S1 >= 4:
        v = "split at both lengths, no length trend"
    elif S0 < 4 and S1 < 4 and all(res_L[str(L)]["G_D3pc"]["passed"] for L in Ls):
        v = "no split at any length"
    else:
        v = "undecided"
    out["D3"] = dict(verdict=v, mcnemar=dict(b=b, c=c_, p_grows=p_up, p_shrinks=p_dn), p_b1=p_up, G_D0d=sj["G_D0d"], G_LCE_4=g4,
                     verifier_error_rate=sj["verifier_error_rate"], lengths=Ls)
    out["D4"] = ana_d4(ctx)
    ctx.write_json("d_ana/d3d4.json", out, config=dict(seed=13, cfg=dict(c)), seeds=dict(bootstrap=13),
                   labels=["MOCK (tiny random model)"] if ctx.mock else [])


def pc_detect(sep_x, sep_y, n=2000, seeds=16):
    """Predicted share of planted A2 + B2 items (8 + 8 seeds) the rule classifies 'split' (mixture, gap and agreement parts) at the
    measured separations (in within-mode SD units), by simulation."""
    rng = np.random.default_rng(1301); h = seeds // 2
    lab = np.r_[np.zeros(h), np.ones(seeds - h)]
    x = rng.normal(0, 1, (n, seeds)) + sep_x * lab; y = rng.normal(0, 1, (n, seeds)) + sep_y * lab
    ok = {}
    for nm, v in (("x", x), ("y", y)):
        e = em2(v); lo_, hi_ = np.minimum(e["mA"], e["mB"]), np.maximum(e["mA"], e["mB"]); third = (hi_ - lo_) / 3
        gap = ((v > (lo_ + third)[:, None]) & (v < (hi_ - third)[:, None])).sum(1)
        ok[nm] = (e["dBIC"] >= 6) & (e["minor"] >= 0.2) & (e["ashman"] >= 2.5) & (gap <= 1)
        ok[nm + "s"] = np.sign(v - ((e["mA"] + e["mB"]) / 2)[:, None])
    ag = (ok["xs"] == ok["ys"]).mean(1); ag = np.maximum(ag, 1 - ag)
    return float((ok["x"] & ok["y"] & (ag >= 0.8)).mean())


def ana_d4(ctx):
    torch = K._torch(); c = S.cfg(ctx)
    p = ctx.input("r1b_D_C", "d4/traj.pt", required=False)
    if not p:
        return dict(verdict=K.coded("NA:not_run:input missing"))
    T = torch.load(p, weights_only=False)["records"]; ua = K.load_json(ctx.input("r1b_D_C", "d4/u_aa.json", required=False) or "", {}) or {}
    g4 = (K.load_json(ctx.input("r1b_D_C", "d4/d4_gen.json", required=False) or "", {}) or {}).get("G-LCE-4", {})
    out = {"G-D9": ua.get("G-D9"), "G-LCE-4": g4}; byL = {}
    for r in T:
        byL.setdefault(r["L"], []).append(r)
    means = {}
    for L, rs in sorted(byL.items()):
        rr, ii, cls, sep, pcc, dg = [], [], [], [], [], []
        for r in rs:
            xa, xr = np.mean(r["anchor_assistant"]["role"]), np.mean(r["anchor_role"]["role"])
            ya, yr = np.mean(r["anchor_assistant"]["ident"]), np.mean(r["anchor_role"]["ident"])
            sep.append(ya - yr >= 2)
            with np.errstate(all="ignore"):
                R_ = (np.array(r["main"]["role"]) - xa) / (xr - xa); I_ = (np.array(r["main"]["ident"]) - yr) / (ya - yr)
                pr, pi_ = (np.array(r["pc_d6"]["role"]) - xa) / (xr - xa), (np.array(r["pc_d6"]["ident"]) - yr) / (ya - yr)
            rr.append(R_); ii.append(I_); pcc.append(cls_of(pr[0], pi_[0])); cls.append([cls_of(a, b) for a, b in zip(R_, I_)])
            dg.append(np.mean([any(x > 0.3 for x in reps) for reps in r["main"]["rep4"]]))
        n = len(rs); share = lambda k: np.array([np.mean([x == k for x in cl]) for cl in cls])
        Wr = weights([[u] for u in range(n)], 13, two=False)
        sh = {k: wm(Wr, share(k)) for k in ("coexisting", "blended", "absorbed", "retained")}
        split_roles = sum((np.mean([x == "absorbed" for x in cl]) >= 0.25 and np.mean([x == "retained" for x in cl]) >= 0.25) for cl in cls)
        largest = max(sh, key=lambda k: pt(sh[k]))
        g6a = sum(sep) >= math.ceil(14 / 16 * n); pc_ok = max(set(pcc), key=pcc.count) == "coexisting" if pcc else False
        if (not g6a or not pc_ok) and not ctx.mock:
            v = "not testable at L"
        elif split_roles >= math.ceil(4 / 16 * n):
            v = "split across seeds"
        elif largest == "coexisting" and pt(sh[largest]) >= 0.4 and lo(sh[largest]) >= 0.25:
            v = "coexisting presented and retained identity"
        elif largest == "blended":
            v = "one blended entity"
        elif largest == "absorbed":
            v = "absorbed into the role"
        elif largest == "retained":
            v = "role not taken (assistant retained)"
        else:
            v = "not decidable"
        lab = [LBL_SEC] + (["degeneration-dominated"] if np.mean(dg) > 0.3 else [])
        means[L] = (np.array([np.nanmean(x) for x in rr]), np.array([np.nanmean(x) for x in ii]))
        out[f"L{L}"] = dict(verdict=K.labelled(v, *lab), shares={k: S_(v_) for k, v_ in sh.items()}, G_D6a=bool(g6a), PC_D6=pcc,
                            mean_r=K.dist(means[L][0]), mean_i=K.dist(means[L][1]))
    Ls = sorted(means)
    if len(Ls) == 2:
        Wr = weights([[u] for u in range(len(means[Ls[0]][0]))], 13, two=False)
        dr, di = wm(Wr, means[Ls[1]][0] - means[Ls[0]][0]), wm(Wr, means[Ls[1]][1] - means[Ls[0]][1])
        mv = [nm + (" up" if pt(x) > 0 else " down") for nm, x in (("role index", dr), ("identity index", di)) if lo(x) > 0 or hi(x) < 0]
        out["across_lengths"] = dict(verdict=(f"hyper-long context moves the identity ({', '.join(mv)})" if mv else "no change with length"),
                                     d_r=S_(dr), d_i=S_(di))
    return out


def cls_of(r, i):
    if not (np.isfinite(r) and np.isfinite(i)):
        return "other"
    if r >= 0.7 and i >= 0.7:
        return "coexisting"
    if 0.3 < r < 0.7 and 0.3 < i < 0.7:
        return "blended"
    if r >= 0.7 and i <= 0.3:
        return "absorbed"
    if r <= 0.3 and i >= 0.7:
        return "retained"
    return "other"


# ------------------------------------------------------------------------------------------------------------------------------------
# D5
# ------------------------------------------------------------------------------------------------------------------------------------
def _ana_d5(ctx):
    import r1b_d_d5 as D5
    torch = K._torch(); c = S.cfg(ctx)
    p = ctx.input("r1b_D_C", "d5/gen.pt", required=False)
    if not p:
        raise K.NotRun("input missing")
    G = torch.load(p, weights_only=False); rows = G["rows"]; H = G["horizon"]; starts = sorted({r["start"] for r in rows})
    det = K.load_json(ctx.input("r1b_D_C", "d5/g6_detector.json", required=False) or "", {}) or {}
    tau = {}; g = {}; pimin = {}
    for r in rows:
        t, cens = D5.lifetime(r["gen"], H); tau[(r["arm"], r["start"])] = (t, cens)
        dP, dQ = D5.stance([x["Y"] for x in r["ck"]])
        sP = 1 if r["poles"]["P"] == "pro" else -1
        g[(r["arm"], r["start"])] = sP * (np.asarray(dP) - np.asarray(dQ))
        if r["arm"] in ("2V-comp-min", "2V-agree-min"):
            ck = {x["t"]: x["counts"] for x in r["ck"]}; t0 = min((t_ for t_ in ck if t_ >= H // 8), default=None); t1 = max(ck)
            if t0 is not None:
                q, pp = ck[t1]["Q"]["tokens"] - ck[t0]["Q"]["tokens"], ck[t1]["P"]["tokens"] - ck[t0]["P"]["tokens"]
                pimin[(r["arm"], r["start"])] = q / (q + pp) if q + pp else np.nan
    n = len(starts); Ws = weights([[u] for u in range(n)], 14, two=False)
    arr = lambda arm: np.array([tau[(arm, s)][0] for s in starts], float)
    rm = {a: arr(a) for a in D5.ARMS}
    deg_share = {a: float(np.mean([not tau[(a, s)][1] for s in starts])) for a in D5.ARMS}
    g0 = np.array([g[("2V-comp", s)][0] for s in starts]); g6 = bool(det.get("passed")) and float((g0 > 0).mean()) >= 0.8
    with np.errstate(all="ignore"):
        ratio = wm(Ws, np.array([g[("2V-comp", s)][-1] for s in starts])) / wm(Ws, g0)
    dl = wm(Ws, rm["2V-comp"] - rm["2V-agree"]); out = dict(horizon=H, degenerate_share=deg_share, G_D6=dict(detector=det.get("passed"),
                                                                                                             separation_share=float((g0 > 0).mean()), passed=g6))
    horizon_short = deg_share["2V-comp"] < 0.2 and deg_share["2V-agree"] < 0.2
    if not g6 and not ctx.mock:
        v = "not testable"
    elif horizon_short:
        v = "not decidable (horizon too short)"
    elif lo(dl) > 0 and pt(dl) >= 500 and lo(ratio) >= 0.5:
        v = "two competing voices sustain coherence longer (catfish lifetime)"
    elif hi(dl) < 0 and pt(dl) <= -500:
        v = "competition shortens coherence"
    elif inside(dl, -500, 500):
        v = "no lifetime effect of competition" if sd(dl) <= 200 else "undecided (underpowered)"
    else:
        v = "undecided"
    lab = ["voices merge"] if hi(ratio) < 0.5 else []
    out["D5-a"] = dict(verdict=K.labelled(v, *lab), delta=S_(dl), g_ratio=S_(ratio), p=p_gt(dl, 0.0),
                       rule_order="'horizon too short' before every support or negative string (D.16 as amended)")
    pm = lambda a: np.array([pimin.get((a, s), np.nan) for s in starts])
    dr = wm(Ws, pm("2V-comp-min") - pm("2V-agree-min")); pc = wm(Ws, pm("2V-comp-min"))
    vr = ("rate-form minority capture (a competing minority writes more than its share beyond the turn-taking norm)" if lo(dr) > 0.05 and lo(pc) > 0.25
          else ("competing minority suppressed" if hi(dr) < -0.05 else ("no rate-form capture" if inside(dr, -0.08, 0.08) else "undecided")))
    out["D5-r"] = dict(verdict=K.labelled(vr, LBL_SEC), contrast=S_(dr), pi_min_comp=S_(pc))
    dg = wm(Ws, rm["2V-comp-min"] - rm["2V-comp"])
    if deg_share["2V-comp-min"] < 0.2 and deg_share["2V-comp"] < 0.2:
        vg = "not decidable (horizon too short)"
    else:
        vg = ("a dominant majority opponent shortens coherence" if hi(dg) < 0 else ("lengthens" if lo(dg) > 0 else
              ("no dominance effect" if inside(dg, -500, 500) else "undecided")))
    out["D5-g"] = dict(verdict=K.labelled(vg, LBL_SEC), diff=S_(dg))
    out["D5-b"] = dict(agree_minus_1V=S_(wm(Ws, rm["2V-agree"] - rm["1V"])), comp_minus_1V=S_(wm(Ws, rm["2V-comp"] - rm["1V"])),
                       labels=["descriptive"])
    px = ctx.input("r1b_D_C", "d5/d5x.json", required=False)
    if px:
        X = K.load_json(px)["rows"]; pairs = []
        for s in sorted({x["start"] for x in X}):
            a = [x for x in X if x["start"] == s and x["arm"] == "2V-comp"]; b_ = [x for x in X if x["start"] == s and x["arm"] == "1V"]
            if a and b_:
                a, b_ = a[0], b_[0]
                if a["censored"] and b_["censored"]:
                    pairs.append(None)
                else:
                    pairs.append(bool((a["censored"] and a["reached"] > b_["tau"]) or (not a["censored"] and not b_["censored"] and a["tau"] > b_["tau"])))
        dec = [x for x in pairs if x is not None]
        vx = ("not decidable (censored at cap)" if not dec else ("two voices sustain their own text toward 28K (descriptive)" if all(dec) else
              ("no full-scale advantage" if not any(dec) else "mixed")))
        out["D5-x"] = dict(verdict=K.labelled(vx, "tier 2", "descriptive"), pairs=pairs, rows=X)
    else:
        out["D5-x"] = dict(verdict=K.coded(K.NA_TIER2 if not ctx.funded("D5-x") else "NA:not_run:input missing"))
    ctx.write_json("d_ana/d5.json", out, config=dict(seed=14, cfg=dict(c), window=D5.WIN, step=D5.STEP, threshold=D5.THR), seeds=dict(bootstrap=14),
                   labels=["MOCK (tiny random model)"] if ctx.mock else [])


# ------------------------------------------------------------------------------------------------------------------------------------
# D.17
# ------------------------------------------------------------------------------------------------------------------------------------
def val_s(v):
    return v["value"] if isinstance(v, dict) and "value" in v else v


def ana_comb(ctx):
    """D.17: Holm over (p_a, p_b1, p_b2) at familywise 0.05; supported only if the support condition holds and Holm rejects; strings;
    qualifiers (never change the string); the R2 trigger for r1b_D_R."""
    L = lambda n: K.load_json(ctx.path(f"d_ana/{n}.json"), {}) if os.path.exists(ctx.path(f"d_ana/{n}.json")) else {}
    d1, d2, d3, d5 = L("d1"), L("d2"), L("d3d4"), L("d5")
    a = d1.get("D1-a", {}); va = val_s(a.get("verdict")) or "not run"
    sup_a = bool(a.get("support")) and va.startswith("length-driven"); neg_a = bool(a.get("negative"))
    p_a = a.get("p_a", 1.0)
    D3 = d3.get("D3", {}); vb1 = val_s(D3.get("verdict")) or "not run"; sup_b1 = vb1 == "split grows with length"
    neg_b1 = vb1 in ("no split at any length", "split shrinks with length (urn-like)", "split at both lengths, no length trend")
    p_b1 = D3.get("p_b1", 1.0)
    d2a, d2b, d2c = val_s(d2.get("D2-a", {}).get("verdict")) or "", d2.get("D2-b", {}).get("verdict", ""), d2.get("D2-c", {}).get("verdict", "")
    labs_a = (d2.get("D2-a", {}).get("verdict") or {}).get("labels", []) if isinstance(d2.get("D2-a", {}).get("verdict"), dict) else []
    M_sup = d2a.startswith("two entities keep") and "confounded with repetition" not in labs_a and d2b.startswith("hyper-long context strengthens") \
        and not d2c.startswith("merging") and d2c != "one entity captured"
    p_M = max(d2.get("D2-a", {}).get("p", 1.0), d2.get("D2-b", {}).get("p", 1.0), d2.get("D2-c", {}).get("p_mu_below_010", 1.0))
    v5 = val_s(d5.get("D5-a", {}).get("verdict")) or ""; D5_sup = v5.startswith("two competing voices sustain"); p_D5 = d5.get("D5-a", {}).get("p", 1.0)
    p_b2 = min(min(1.0, 2 * p_M), min(1.0, 2 * p_D5))
    sup_b2 = M_sup or D5_sup
    M_neg = d2a.startswith(("competitor dilutes", "no catfish effect")) or d2b in ("weakens", "no length interaction") or \
        d2c.startswith("merging") or d2c == "one entity captured"
    neg_b2 = M_neg and (v5.startswith(("no lifetime effect of competition", "competition shortens coherence")))
    adj, rej = K.holm([p_a, p_b1, p_b2], 0.05)
    parts = {"(a)": dict(verdict=va, support_condition=sup_a, negative=neg_a, p=p_a, holm_adjusted=adj[0], holm_reject=rej[0]),
             "(b1)": dict(verdict=vb1, support_condition=sup_b1, negative=neg_b1, p=p_b1, holm_adjusted=adj[1], holm_reject=rej[1]),
             "(b2)": dict(verdict=dict(M=dict(D2_a=d2a, D2_b=d2b, D2_c=d2c, support=M_sup, p_M=p_M), D5_a=dict(verdict=v5, support=D5_sup, p=p_D5)),
                          support_condition=sup_b2, negative=neg_b2, p=p_b2, holm_adjusted=adj[2], holm_reject=rej[2])}
    sup = [k for k, v in parts.items() if v["support_condition"] and v["holm_reject"]]
    neg = [k for k, v in parts.items() if v["negative"] and k not in sup]
    oth = [k for k in parts if k not in sup and k not in neg]
    reason = lambda k: (parts[k]["verdict"] if isinstance(parts[k]["verdict"], str) else
                        f"D2 {parts[k]['verdict']['M']['D2_a'] or 'not run'} / D5-a {parts[k]['verdict']['D5_a']['verdict'] or 'not run'}")
    if len(sup) == 3:
        s = "H-DUAL supported in Qwen3-1.7B (all three parts)"
    elif sup:
        s = (f"H-DUAL partly supported in Qwen3-1.7B: {', '.join(sup)}; against: {', '.join(neg) or 'none'}; not decidable: "
             f"{', '.join(oth) or 'none'}")
    elif len(neg) == 3:
        s = "H-DUAL not supported in Qwen3-1.7B"
    else:
        s = "H-DUAL not decidable in Qwen3-1.7B (" + "; ".join(f"{k}: {reason(k)}" for k in oth + [x for x in neg if x not in oth][:0]) + ")"
    qual = dict(D1_b=val_s((d1.get("D1-b") or {}).get("verdict")), D1_c={k: val_s(v.get("verdict")) for k, v in (d1.get("D1-c") or {}).items()},
                D1_d=val_s((d1.get("D1-d") or {}).get("verdict")), D1_e=val_s((d1.get("D1-e") or {}).get("verdict")),
                D1_f=val_s((d1.get("D1-f") or {}).get("verdict")), D1_r=val_s((d1.get("D1-r") or {}).get("verdict")),
                D1_v={k: v.get("label") if isinstance(v, dict) else v for k, v in (d1.get("D1-v") or {}).items()},
                D1_s=val_s((d1.get("D1-s") or {}).get("verdict")), D2_g=val_s((d2.get("D2-g_long") or {}).get("verdict")),
                ALI_MON=val_s(d2.get("specificity")), D4={k: val_s(v.get("verdict")) if isinstance(v, dict) else v
                                                          for k, v in (d3.get("D4") or {}).items() if k.startswith(("L", "across"))},
                D5_r=val_s((d5.get("D5-r") or {}).get("verdict")), D5_g=val_s((d5.get("D5-g") or {}).get("verdict")),
                D5_x=val_s((d5.get("D5-x") or {}).get("verdict")),
                coexistence={k: v.get("label") for k, v in (d1.get("coexistence") or {}).items()})
    dd = qual["D1_d"] or ""; enables = isinstance(dd, str) and dd.startswith("losing game") and (
        d2c == "coexistence" or any(str(val_s(v.get("verdict"))).startswith("two entities") for k, v in (D3 and d3.get("per_length", {})).items()
                                    if k != str((D3.get("lengths") or [0])[0])))
    out = {"D.17": s, "holm": dict(p=[p_a, p_b1, p_b2], adjusted=adj, reject=rej, family="CF-D (main §2.4)"), "parts": parts,
           "qualifiers": K.labelled(qual, "qualifiers never change the combined string"),
           "enables_the_duality": K.labelled(bool(enables), "descriptive joint reading (D1-d with D2-c coexistence or a D3 split at L*)"),
           "triggers": dict(R2=sup_b1 or vb1 == "split grows with length"),
           "notes": "part L's L4 strings are attached in r1b_ana, not here (main §3.3)"}
    ctx.write_json(C.FILES["d_verdict"]["rel"], out, config=dict(alpha=0.05, rule="D.17"), labels=["MOCK (tiny random model)"] if ctx.mock else [])


def _safe(fn, rel, keys):
    """An analysis step never ends 'not run': a missing input gives coded verdicts (so that D.17 and d_verdict.json are always written)."""
    def step(ctx):
        try:
            return fn(ctx)
        except K.NotRun as e:
            ctx.write_json(rel, dict({k: dict(verdict=K.coded(f"NA:not_run:{e.reason}")) for k in keys}, not_run=e.reason), config=dict())
    step.__doc__ = fn.__doc__
    return step


ana_d1 = _safe(_ana_d1, "d_ana/d1.json", ("D1-a", "D1-b", "D1-c", "D1-d", "D1-e", "D1-f", "D1-r", "D1-v", "D1-s"))
ana_d2 = _safe(_ana_d2, "d_ana/d2.json", ("D2-a", "D2-b", "D2-c", "D2-g_long"))
ana_d3 = _safe(_ana_d3, "d_ana/d3d4.json", ("D3", "D4"))
ana_d5 = _safe(_ana_d5, "d_ana/d5.json", ("D5-a", "D5-r", "D5-g", "D5-x"))

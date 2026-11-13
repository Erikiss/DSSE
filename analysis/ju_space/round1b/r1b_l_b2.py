"""Session r1b_L_B (account B), part 2: L3a observational dynamics (chain effects and chain-specific surrogates, part-file amendment),
L3c Ising fits (static max-ent, K-pairwise, covariate fields, Gibbs check, I2/IN, basins, kinetic Ising, G-cal-Ising), L3d EBM and
irreversibility at H entry (time-reversed pre-exit null, part-file amendment; G-sigma, Sagawa sigma, A_k), and the tier-2 items X1
(dose ladder), X3 (toy loop), X4 (completion), X5 (L.gv)."""
import math, re
import numpy as np

import r1b_core as K
import r1b_l_engine as E
import r1b_l_stats as S
from r1b_l_a import _ProbeCM, l3b_prompts, refw, setup
from r1b_l_b import ANCH, FAMS, corpus, lr_items, spin_units
from r1b_l_base import (F_NEU_SCENES, NEU_TEMPLATE, QA_TEMPLATE, channels, load_pop, loadpt, pos_bin, ref_sets, rng, save, sha, wj)


# ==================================================================================================================================
# shared: chains with binarised channels
# ==================================================================================================================================
def _src(c):
    return (c.get("kind", "tw"), c["fam"], c["T"])


def load_chains(ctx, with_twin=True):
    """L.R-A, L.R-B (and L3b's twin-0 rows): per chain a [n, C] (float32), e, s, rep, bins, item, fam, T, kind."""
    cf = setup(ctx)[2]
    ch, th, _ = corpus(ctx)
    out = []
    for c in ch:
        out.append(dict(kind=c["kind"], fam=c["fam"], T=c["T"], item=c["item"], a=np.asarray(c["a"], np.float32), e=c["e"], s=c["s"],
                        rep=c["rep"], n=c["n"], spins=c.get("spins"), zP=c.get("zP"), tokens=c["tokens"], key=c["key"]))
    if with_twin:
        tw = loadpt(ctx.input("r1b_L_A", "l/l3b_twin0.pt"))
        C = out[0]["a"].shape[1] if out else None
        for r in tw["rows"]:
            n = len(r["tokens"])
            out.append(dict(kind="tw", fam=r["fam"], T=0.7, item=r["item"], a=np.asarray(r["a"], np.float32)[:, :C], e=np.asarray(r["e"], float),
                            s=np.asarray(r["s"], float), rep=np.zeros(n, bool), n=n, spins=None, zP=None, tokens=r["tokens"], key=None))
    nb = th["nb"]
    for c in out:
        c["bins"] = np.minimum(pos_bin(np.arange(1, c["n"] + 1), cf["pos_scale"]), nb - 1)
    return out, th


def binarise(chains, th, names):
    """b for P (Q95(F-neu, T, bin) thresholds), occupancy-matched b for every other channel and the E-channel (per source, family, T,
    bin: the quantile that gives P's occupancy)."""
    C = len(names)
    for c in chains:
        ti = 0 if c["T"] == 0.7 else 1
        c["b"] = np.zeros((c["n"], C + 1), np.int8)
        c["b"][:, 0] = c["a"][:, 0] > th["thrA"][ti][c["bins"]]
    groups = {}
    for k, c in enumerate(chains):
        for b in np.unique(c["bins"]):
            groups.setdefault(_src(c) + (int(b),), []).append((k, c["bins"] == b))
    for g, lst in groups.items():
        A = np.concatenate([np.concatenate([chains[k]["a"][m][:, 1:], chains[k]["e"][m][:, None]], 1) for k, m in lst])
        occ = np.concatenate([chains[k]["b"][m, 0] for k, m in lst]).mean()
        q = np.quantile(A, 1 - occ, axis=0) if occ > 0 else np.full(A.shape[1], np.inf)
        for k, m in lst:
            X = np.concatenate([chains[k]["a"][m][:, 1:], chains[k]["e"][m][:, None]], 1)
            chains[k]["b"][m, 1:] = X > q[None]
    return C + 1          # index C is the E-channel


def padded(chains, col):
    n = max(c["n"] for c in chains) if chains else 0
    B = np.zeros((len(chains), n), np.int8); Lc = np.array([c["n"] for c in chains]); Bn = np.zeros((len(chains), n), np.int64)
    E_ = np.zeros((len(chains), n))
    for i, c in enumerate(chains):
        B[i, :c["n"]] = c["b"][:, col]; Bn[i, :c["n"]] = c["bins"]; E_[i, :c["n"]] = c["e"]
    return B, Lc, Bn, E_


def dwell_mat(B):
    D = np.zeros_like(B, dtype=np.int64); run = np.zeros(B.shape[0], np.int64)
    for t in range(B.shape[1]):
        run = (run + 1) * (B[:, t] == 1); D[:, t] = run
    return D


def design(B, Lc, Bn, E_):
    """At-risk rows (b_t = 1, t + 1 < length): chain, bin, e_t, log(1 + d_t), exit."""
    D = dwell_mat(B)
    T = B.shape[1]
    ok = np.zeros_like(B, bool)
    ok[:, :-1] = (B[:, :-1] == 1) & (np.arange(T - 1)[None] + 1 < Lc[:, None])
    i, t = np.nonzero(ok)
    return np.stack([i, Bn[i, t], E_[i, t], np.log1p(D[i, t]), (B[i, t + 1] == 0).astype(float)], 1)


def hd_vec(B, Lc, Bn, items=None, n_items=None):
    """Mantel-Haenszel log OR over strata chain x bin; with items: per-item numerator / denominator sums (cluster bootstrap)."""
    T = B.shape[1]
    ok = (np.arange(T - 1)[None] + 1 < Lc[:, None])
    x0, x1, s = B[:, :-1], B[:, 1:], Bn[:, :-1]
    nb = int(Bn.max()) + 1 if Bn.size else 1
    key = (np.arange(B.shape[0])[:, None] * nb + s)[ok]
    a = ((x0 == 1) & (x1 == 1))[ok]; b = ((x0 == 1) & (x1 == 0))[ok]; c = ((x0 == 0) & (x1 == 1))[ok]; d = ((x0 == 0) & (x1 == 0))[ok]
    K_ = B.shape[0] * nb
    cnt = [np.bincount(key, weights=v.astype(float), minlength=K_) for v in (a, b, c, d)]
    n = sum(cnt)
    with np.errstate(invalid="ignore", divide="ignore"):
        num = np.where(n > 0, cnt[0] * cnt[3] / n, 0.0); den = np.where(n > 0, cnt[1] * cnt[2] / n, 0.0)
    if items is None:
        return math.log(num.sum() / den.sum()) if num.sum() > 0 and den.sum() > 0 else float("nan")
    ch_item = np.repeat(items, nb)
    return np.bincount(ch_item, weights=num, minlength=n_items), np.bincount(ch_item, weights=den, minlength=n_items)


def sim_vec(fit, Lc, Bn, init, n_sur, g, entry_c=None, exit_c=None):
    """Vectorised Markov surrogates [n_sur, n_chain, T] (time-varying entry / exit, memoryless dwell, initial state copied)."""
    sig = lambda z: 1 / (1 + np.exp(-z))
    ec = fit["entry_c"] if entry_c is None else entry_c
    xc = fit["exit_c"] if exit_c is None else exit_c
    pe = sig(ec[..., :, None] + fit["entry_b"][Bn]); px = sig(xc[..., :, None] + fit["exit_b"][Bn])
    T = Bn.shape[1]
    S_ = np.zeros((n_sur,) + Bn.shape, np.int8)
    S_[..., 0] = init
    for t in range(T - 1):
        u = g.random((n_sur, Bn.shape[0]))
        cur = S_[..., t]
        pe_t = pe[..., t] if pe.ndim == 2 else pe[:, :, t]
        px_t = px[..., t] if px.ndim == 2 else px[:, :, t]
        S_[..., t + 1] = np.where(cur == 0, u < pe_t, u >= px_t)
    S_ *= (np.arange(T)[None, None] < Lc[None, :, None])
    return S_


def markov_fit(B, Lc, Bn):
    chains = [B[i, :Lc[i]] for i in range(len(Lc))]; bins = [Bn[i, :Lc[i]] for i in range(len(Lc))]
    return S.fit_markov(chains, bins, int(Bn.max()) + 1 if Bn.size else 1)


# ==================================================================================================================================
# L3a
# ==================================================================================================================================
def l3a(ctx):
    """HD (MH over chain x bin), stickiness gamma (chain FE + bin FE + beta_e e - gamma log(1 + d)), chain-specific Markov surrogates
    (200 for P, 50 per comparison), surrogate-corrected Delta gamma, BI on L.R-B against unimodal-frailty surrogates, rank tests against
    the 99 controls, SEL+ / UNC+ / E contrasts, repetition exclusion, selection robustness."""
    tok, model, cf, H, items = setup(ctx)
    pop = load_pop(ctx)
    dev = "cuda" if K._torch().cuda.is_available() else "cpu"
    chains, th = load_chains(ctx)
    names = [n for n, _ in channels(pop)]
    Ecol = binarise(chains, th, names) - 1
    ncol = Ecol + 1
    labels = ["binarisation drift"] if th.get("drift") else []
    uit = sorted({c["item"] for c in chains}); iix = {u: i for i, u in enumerate(uit)}
    item_of = np.array([iix[c["item"]] for c in chains])
    W = S.boot_counts(len(uit), cf["nboot"], seed=301)
    g = rng("r1bL-SUR", "l3a")
    nbin = th["nb"]
    res = dict(names=names + ["E"])
    gam, gsur, hd, nsur = np.full(ncol, np.nan), {}, np.full(ncol, np.nan), {}
    designs = []
    mats = []
    for col in range(ncol):
        B, Lc, Bn, E_ = padded(chains, col)
        mats.append((B, Lc, Bn, E_))
        designs.append(design(B, Lc, Bn, E_ if col != Ecol else np.zeros_like(E_)))
        hd[col] = hd_vec(B, Lc, Bn)
    gam = S.fit_hazards(designs, len(chains), nbin, use_e=True, device=dev)
    gam_noe = S.fit_hazards(designs, len(chains), nbin, use_e=False, device=dev)
    sur_med, sur_med_noe, sur_P = np.full(ncol, np.nan), np.full(ncol, np.nan), None
    for col in range(ncol):
        B, Lc, Bn, E_ = mats[col]
        if B.sum() == 0:
            continue
        fit = markov_fit(B, Lc, Bn)
        n_s = cf["n_sur_P"] if col == 0 else cf["n_sur_C"]
        Ss = sim_vec(fit, Lc, Bn, B[:, 0], n_s, g)
        ds = [design(Ss[k], Lc, Bn, E_ if col != Ecol else np.zeros_like(E_)) for k in range(n_s)]
        gs = S.fit_hazards(ds, len(chains), nbin, use_e=True, device=dev)
        sur_med[col] = np.nanmedian(gs)
        sur_med_noe[col] = np.nanmedian(S.fit_hazards(ds, len(chains), nbin, use_e=False, device=dev)) if col in (0, Ecol) else np.nan
        if col == 0:
            sur_P = gs
    dg = gam - sur_med
    ci_ = [i for i, n in enumerate(names) if n.startswith("C")]
    j = {n: i for i, n in enumerate(names)}
    res["gamma_P"] = float(gam[0]); res["gamma_sur_p"] = K.p_rank(gam[0], sur_P) if sur_P is not None else 1.0
    res["gamma_above_sur"] = bool(sur_P is not None and gam[0] > np.nanpercentile(sur_P, 95))
    res["dgamma"] = dg.tolist(); res["dgamma_rank_p"] = K.p_rank(dg[0], dg[ci_]); res["hd"] = hd.tolist(); res["hd_rank_p"] = K.p_rank(hd[0], hd[ci_])
    # bootstrap of the P - SEL+ contrasts (weighted refits for gamma; item sums for HD)

    def gboot(col, use_e=True):
        out = []
        for b0 in range(0, len(W), 50):
            wts = [W[b][item_of] for b in range(b0, min(b0 + 50, len(W)))]
            out += list(S.fit_hazards([designs[col]] * len(wts), len(chains), nbin, use_e=use_e, weights=wts, device=dev))
        return np.array(out)
    gP = gboot(0)
    for nm in ("SEL", "UNC"):
        if nm in j:
            gX = gboot(j[nm])
            res[f"dgamma_minus_{nm}"] = S.summ(dg[0] - dg[j[nm]], (gP - sur_med[0]) - (gX - sur_med[j[nm]]))
            nP, dP = hd_vec(*mats[0][:3], items=item_of, n_items=len(uit)); nX, dX = hd_vec(*mats[j[nm]][:3], items=item_of, n_items=len(uit))
            res[f"hd_minus_{nm}"] = S.summ(hd[0] - hd[j[nm]], np.log((W @ nP) / (W @ dP)) - np.log((W @ nX) / (W @ dX)))
    gPn, gEn = gboot(0, False), gboot(Ecol, False)
    res["dgamma_noe_minus_E"] = S.summ((gam_noe[0] - sur_med_noe[0]) - (gam_noe[Ecol] - sur_med_noe[Ecol]),
                                       (gPn - sur_med_noe[0]) - (gEn - sur_med_noe[Ecol]))
    nP, dP = hd_vec(*mats[0][:3], items=item_of, n_items=len(uit)); nE, dE = hd_vec(*mats[Ecol][:3], items=item_of, n_items=len(uit))
    res["hd_minus_E"] = S.summ(hd[0] - hd[Ecol], np.log((W @ nP) / (W @ dP)) - np.log((W @ nE) / (W @ dE)))
    res["dgamma_minus_sel"] = res.get("dgamma_minus_SEL", S.summ(float("nan"), [np.nan]))
    res["hd_minus_sel"] = res.get("hd_minus_SEL", S.summ(float("nan"), [np.nan]))
    # BI on L.R-B
    k0, k1, mlen = cf["bi_k"][0], cf["bi_k"][1], cf["bi_k"][2]
    bidx = [i for i, c in enumerate(chains) if c["kind"] == "B"]
    by_item = {}
    for i in bidx:
        by_item.setdefault(chains[i]["item"], []).append(i)
    bit = sorted(by_item)

    def bi_of(col, idx_map):
        return S.bi_stat([[chains[i]["b"][:, col] for i in idx_map[it]] for it in bit], k0, k1, mlen, min_seeds=min(6, cf["lrb_keys"] - 2))[0]
    bi = np.array([bi_of(col, by_item) for col in range(ncol)])
    res["bi"] = bi.tolist(); res["bi_rank_p"] = K.p_rank(bi[0], bi[ci_])
    sub = [chains[i] for i in bidx]
    if sub:
        B, Lc, Bn, E_ = padded(sub, 0)
        fit = markov_fit(B, Lc, Bn)
        it_of = np.array([bit.index(c["item"]) for c in sub])
        mu_e = np.array([fit["entry_c"][it_of == k].mean() for k in range(len(bit))]); sd_e = np.array([fit["entry_c"][it_of == k].std() for k in range(len(bit))])
        mu_x = np.array([fit["exit_c"][it_of == k].mean() for k in range(len(bit))]); sd_x = np.array([fit["exit_c"][it_of == k].std() for k in range(len(bit))])
        bis = []
        for k in range(cf["bi_sur"]):
            ec = g.normal(mu_e[it_of], sd_e[it_of]); xc = g.normal(mu_x[it_of], sd_x[it_of])
            Ss = sim_vec(fit, Lc, Bn, B[:, 0], 1, g, entry_c=ec[None], exit_c=xc[None])[0]
            bis.append(S.bi_stat([[Ss[i, :Lc[i]] for i in np.flatnonzero(it_of == q)] for q in range(len(bit))], k0, k1, mlen,
                                 min_seeds=min(6, cf["lrb_keys"] - 2))[0])
        bis = np.array(bis, float)
        res["bi_above_sur99"] = bool(np.isfinite(bi[0]) and bi[0] > np.nanpercentile(bis, 99)) if np.isfinite(bis).any() else False
        res["bi_sur_p"] = K.p_rank(bi[0], bis[np.isfinite(bis)]) if np.isfinite(bis).any() else 1.0
        if "SEL" in j:
            Wb = S.boot_counts(len(bit), cf["nboot"], seed=302)

            def split_vec(col):
                v = []
                for it in bit:
                    o = [float(np.mean(chains[i]["b"][k0:k1, col])) for i in by_item[it] if chains[i]["n"] >= mlen]
                    o = np.array(o)
                    v.append(float(len(o) >= min(6, cf["lrb_keys"] - 2) and (o >= 0.3).sum() >= 2 and (o <= 0.02).sum() >= 2))
                return np.array(v)
            sp_, ss_ = split_vec(0), split_vec(j["SEL"])
            res["bi_minus_sel"] = S.summ(sp_.mean() - ss_.mean(), (Wb @ (sp_ - ss_)) / Wb.sum(1))
    else:
        res["bi_above_sur99"], res["bi_sur_p"] = False, 1.0
    c90 = lambda v: np.nanpercentile(np.asarray(v)[ci_], 90)
    res["dgamma_below_c90"] = bool(dg[0] < c90(dg)); res["hd_below_c90"] = bool(hd[0] < c90(hd)); res["bi_below_c90"] = bool(bi[0] < c90(bi))
    # qualifiers
    stick = res["gamma_above_sur"] and res["dgamma_rank_p"] <= 0.025
    pers = res["hd_rank_p"] <= 0.025
    unc = []
    if stick:
        unc.append(not (_lb(res.get("dgamma_minus_UNC")) and _lb(res["dgamma_noe_minus_E"])))
    if pers:
        unc.append(not (_lb(res.get("hd_minus_UNC")) and _lb(res["hd_minus_E"])))
    res["not_beyond_unc"] = bool(unc and all(unc))
    # repetition exclusion: chains truncated at their first repeat-flagged token
    tr = []
    for c in chains:
        k = np.flatnonzero(c["rep"])
        tr.append(dict(c, n=int(k[0]) if k.size else c["n"], b=c["b"][:int(k[0]) if k.size else c["n"]],
                       bins=c["bins"][:int(k[0]) if k.size else c["n"]], e=c["e"][:int(k[0]) if k.size else c["n"]]))
    tr = [c for c in tr if c["n"] > 1]
    if tr and len(tr) and sum(c["n"] for c in tr) < sum(c["n"] for c in chains):
        hd2, ds2 = [], []
        for col in [0] + ci_:
            B, Lc, Bn, E_ = padded(tr, col)
            hd2.append(hd_vec(B, Lc, Bn)); ds2.append(design(B, Lc, Bn, E_))
        g2 = S.fit_hazards(ds2, len(tr), nbin, device=dev) - sur_med[[0] + ci_]
        res["rep_attractor"] = bool((stick or pers) and hd2[0] < np.nanpercentile(hd2[1:], 90) and g2[0] < np.nanpercentile(g2[1:], 90))
    else:
        res["rep_attractor"] = False
    bix = [i for i, n in enumerate(names) if n.startswith("B")]
    if len(bix) >= 10 and (stick or pers):
        ok = [(dg[i] > np.nanmax(dg[ci_]) and dg[i] > dg[j.get("SEL", 0)]) if stick else (hd[i] > np.nanmax(hd[ci_]) and hd[i] > hd[j.get("SEL", 0)])
              for i in bix]
        res["sel_robust"] = sum(ok) >= 16 * len(bix) / 20
    else:
        res["sel_robust"] = None
    wj(ctx, "l/l3a.json", res, config=dict(n_sur_P=cf["n_sur_P"], n_sur_C=cf["n_sur_C"], bi_sur=cf["bi_sur"], strata="chain x bin",
                                            surrogates="per-chain Markov (chain FE + shared position profile)"), labels=labels)


def _lb(s):
    return isinstance(s, dict) and isinstance(s.get("lo"), float) and s["lo"] > 0


# ==================================================================================================================================
# L3c Ising
# ==================================================================================================================================
def spins_of(chains, off, name, units_n):
    """Unpacked spins of one population per chain."""
    a, b = off[name]
    out = []
    for c in chains:
        if c.get("spins") is None or c["n"] == 0:
            out.append(np.zeros((0, b - a), np.int8)); continue
        u = np.unpackbits(c["spins"], axis=1)[:, :units_n]
        out.append(u[:c["n"], a:b].astype(np.int8))
    return out


def l3c(ctx):
    """Static PL (independent, pairwise, K-pairwise, covariate fields), G-ESS, G-ISM (held-out PL gain, Gibbs P(K) band, triplets), dPL
    for every population, I2/IN on 20 random 10-unit subsets, basins (descriptive), kinetic Ising (G_pair), A_J, modularity, Glauber dwell,
    G-cal-Ising (five planted sets at the measured sample count and tau_int)."""
    torch = K._torch()
    tok, model, cf, H, items = setup(ctx)
    pop = load_pop(ctx)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ch, th, _ = corpus(ctx)
    chains = [c for c in ch if c["fam"] in FAMS]
    units, off = th["units"], th["off"]
    uit = sorted({c["item"] for c in chains}); par = {u: i % 2 for i, u in enumerate(uit)}
    fitm = np.array([par[c["item"]] == 1 for c in chains])
    nb = th["nb"]
    grp = lambda c: (FAMS.index(c["fam"]) * 2 + (0 if c["T"] == 0.7 else 1)) * nb + np.minimum(pos_bin(np.arange(1, c["n"] + 1), cf["pos_scale"]), nb - 1)
    G = len(FAMS) * 2 * nb
    pops = [n for n, _ in [("P", 1)] + [(x, 1) for x in ("SEL", "UNC") if x in off] + [(f"C{k}", 1) for k in range(len(pop["CTRL"]))] +
            [(f"B{k}", 1) for k in range(len(pop.get("boot") or []))]]
    res = dict(pops=pops)

    def data(name, sel_units=None):
        sp = spins_of(chains, off, name, len(units))
        if sel_units is not None:
            sp = [s[:, sel_units] for s in sp]
        cat = lambda m: np.concatenate([s for s, k in zip(sp, m) if k and len(s)] or [np.zeros((0, sp[0].shape[1]), np.int8)])
        gcat = lambda m: np.concatenate([grp(c) for c, s, k in zip(chains, sp, m) if k and len(s)] or [np.zeros(0, int)])
        tr = lambda m: (np.concatenate([s[:-1] for s, k in zip(sp, m) if k and len(s) > 1] or [np.zeros((0, sp[0].shape[1]), np.int8)]),
                        np.concatenate([s[1:] for s, k in zip(sp, m) if k and len(s) > 1] or [np.zeros((0, sp[0].shape[1]), np.int8)]),
                        np.concatenate([grp(c)[:-1] for c, s, k in zip(chains, sp, m) if k and len(s) > 1] or [np.zeros(0, int)]))
        it = lambda m: np.concatenate([np.full(len(s), uit.index(c["item"])) for c, s, k in zip(chains, sp, m) if k and len(s)] or [np.zeros(0, int)])
        return dict(Sf=cat(fitm), St=cat(~fitm), gf=gcat(fitm), gt=gcat(~fitm), trf=tr(fitm), trt=tr(~fitm), itf=it(fitm), itt=it(~fitm), sp=sp)
    # G-ESS on P
    D = data("P")
    N = D["Sf"].shape[1]
    taus = [S.tau_int(s.mean(1)) for s in D["sp"] if len(s) > 10 and s.shape[1]]
    tau = float(np.mean(taus)) if taus else 1.0
    ess_ok = len(D["Sf"]) / tau >= 10 * N * (N + 1) / 2
    sel_units = None
    if not ess_ok:
        freq = np.array([sum(j in set(b) for b in pop.get("boot") or []) for j in pop["P"]], float)
        sel_units = np.argsort(-freq, kind="stable")[:30]
        D = data("P", sel_units); N = D["Sf"].shape[1]
        ess_ok = len(D["Sf"]) / tau >= 10 * N * (N + 1) / 2
        res["ess_fallback"] = "top 30 units by selection frequency"
    res["tau_int"] = tau; res["ess_ok"] = bool(ess_ok); res["N"] = N
    if N < 10 or not ess_ok:
        res["not_testable"] = "not testable (population too small)" if N < 10 else "not testable (sample)"
    lam = S.pl_select_lam(D["Sf"], D["itf"], device=dev) if len(D["Sf"]) > 20 else 1e-3
    res["lambda"] = lam

    def dpl(Dx, groups=False):
        gf, gt = (Dx["gf"], Dx["gt"]) if groups else (None, None)
        ng = G if groups else 1
        f1 = S.pl_fit(Dx["Sf"], lam, gf, ng, pair=False, device=dev); f2 = S.pl_fit(Dx["Sf"], lam, gf, ng, device=dev)
        d_ = S.pl_ll(Dx["St"], f2, gt, dev) - S.pl_ll(Dx["St"], f1, gt, dev)
        return d_, f1, f2

    def kin(Dx):
        Xf, Yf, gf = Dx["trf"]; Xt, Yt, gt = Dx["trt"]
        if len(Xf) < 5 or len(Xt) < 5:
            return np.zeros(0), None
        k1 = S.kinetic_fit(Xf, Yf, gf, G, pair=False, device=dev); k2 = S.kinetic_fit(Xf, Yf, gf, G, device=dev)
        return S.kinetic_ll(Xt, Yt, gt, k2, dev) - S.kinetic_ll(Xt, Yt, gt, k1, dev), k2
    dP, f1P, f2P = dpl(D)
    dPiv, _, _ = dpl(D, groups=True)
    gP, kP = kin(D)
    itt = D["itt"]; W = S.boot_counts(len(uit), cf["nboot"], seed=331)

    def itemsum(v, it):
        return np.bincount(it, weights=v, minlength=len(uit)), np.bincount(it, minlength=len(uit)).astype(float)
    sP, nP = itemsum(dP, itt)
    gain_b = (W @ sP) / np.maximum(W @ nP, 1)
    res["pl_gain"] = S.summ(float(dP.mean()) if len(dP) else float("nan"), gain_b)
    # Gibbs check: P(K) band on the test half and triplets
    nch, sw, burn = cf["gibbs"]
    smp = S.gibbs(f2P["h"][0].cpu().numpy(), f2P["J"].cpu().numpy(), n_chains=nch, sweeps=sw, burn=burn, seed=7, device=dev)
    pk_fit = S.pk(smp, N)
    St = D["St"]
    pk_b = np.stack([S.pk(St[np.repeat(np.arange(len(St)), 1)[np.isin(itt, np.flatnonzero(W[b] > 0))]], N) for b in range(min(len(W), 200))])
    lo_b, hi_b = np.percentile(pk_b, 5, 0), np.percentile(pk_b, 95, 0)
    Kmax = min(10, N)
    pk_in = int(((pk_fit[:Kmax + 1] >= lo_b[:Kmax + 1] - 1e-12) & (pk_fit[:Kmax + 1] <= hi_b[:Kmax + 1] + 1e-12)).sum())
    tri_fit = S.triplet_mean(smp); tri_b = [S.triplet_mean(St[np.isin(itt, np.flatnonzero(W[b] > 0))], seed=b) for b in range(min(len(W), 50))]
    tri_ok = bool(np.percentile(tri_b, 5) <= tri_fit <= np.percentile(tri_b, 95)) if tri_b else False
    gism = bool(_lb(res["pl_gain"]) and pk_in >= min(9, Kmax + 1 - 2) and tri_ok)
    res.update(gism=gism, pk_in=pk_in, pk_fit=pk_fit.tolist(), pk_emp=S.pk(St, N).tolist(), triplet_ok=tri_ok)
    if not gism:
        fk = S.pl_fit(D["Sf"], lam, kpair=True, device=dev)
        res["pl_gain_k"] = float((S.pl_ll(St, fk, None, dev) - S.pl_ll(St, f1P, None, dev)).mean()) if len(St) else None
        res["gism_k"] = bool(res["pl_gain_k"] is not None and res["pl_gain_k"] > 0)
    # dPL and G_pair for every population
    dpl_v, dpl_iv, gpair = {}, {}, {}
    per_item = {}
    for name in pops:
        Dx = D if name == "P" else data(name)
        if Dx["Sf"].shape[1] < 2 or len(Dx["St"]) == 0:
            continue
        d1_ = dP if name == "P" else dpl(Dx)[0]
        dpl_v[name] = float(d1_.mean())
        per_item[name] = itemsum(d1_, Dx["itt"])
        if name in ("P", "SEL", "UNC"):
            dpl_iv[name] = (dPiv if name == "P" else dpl(Dx, groups=True)[0])
        gg = gP if name == "P" else kin(Dx)[0]
        gpair[name] = float(gg.mean()) if len(gg) else float("nan")
        if name in ("P", "SEL", "UNC") and len(gg):
            per_item[name + "_g"] = itemsum(gg, np.concatenate([np.full(max(len(s) - 1, 0), uit.index(c["item"])) for c, s, k in
                                                                zip(chains, Dx["sp"], ~fitm) if k and len(s) > 1]))
    cn = [n for n in pops if n.startswith("C") and n in dpl_v]
    res["dpl"] = dpl_v; res["gpair"] = gpair
    res["dpl_rank_p"] = K.p_rank(dpl_v["P"], [dpl_v[n] for n in cn]); res["gpair_rank_p"] = K.p_rank(gpair["P"], [gpair[n] for n in cn])
    res["dpl_below_c90"] = bool(dpl_v["P"] < np.percentile([dpl_v[n] for n in cn], 90)) if cn else False
    res["gpair_below_c90"] = bool(gpair["P"] < np.nanpercentile([gpair[n] for n in cn], 90)) if cn else False

    def contrast(a, b):
        sa, na = per_item[a]; sb, nb_ = per_item[b]
        return S.summ(sa.sum() / max(na.sum(), 1) - sb.sum() / max(nb_.sum(), 1), (W @ sa) / np.maximum(W @ na, 1) - (W @ sb) / np.maximum(W @ nb_, 1))
    for x in ("SEL", "UNC"):
        if x in per_item:
            res[f"dpl_minus_{x.lower()}"] = contrast("P", x)
            if x + "_g" in per_item and "P_g" in per_item:
                res[f"gpair_minus_{x.lower()}"] = contrast("P_g", x + "_g")
            if x in dpl_iv:
                sa, na = itemsum(dpl_iv["P"], itt); sb, nb_ = itemsum(dpl_iv[x], data(x)["itt"])
                res[f"dpl_minus_{x.lower()}_iv"] = S.summ(sa.sum() / max(na.sum(), 1) - sb.sum() / max(nb_.sum(), 1),
                                                          (W @ sa) / np.maximum(W @ na, 1) - (W @ sb) / np.maximum(W @ nb_, 1))
    for k_ in ("dpl_minus_sel", "dpl_minus_sel_iv", "gpair_minus_sel"):
        res.setdefault(k_, S.summ(float("nan"), [np.nan]))
    res["not_beyond_unc"] = bool("dpl_minus_unc" in res and not _lb(res["dpl_minus_unc"]))
    bnames = [n for n in pops if n.startswith("B") and n in dpl_v]
    res["sel_robust"] = (sum(dpl_v[n] > max(dpl_v[c] for c in cn) for n in bnames) >= 16 * len(bnames) / 20) if len(bnames) >= 10 and cn else None
    # I2 / IN on 20 random 10-unit subsets (subset bootstrap)
    g = rng("r1bL3c", "i2in")
    if N >= 10:
        rr = [S.maxent_info(D["Sf"][:, g.choice(N, 10, replace=False)], ess_scale=tau) for _ in range(20 if not ctx.mock else 3)]
        ratio = np.array([r["ratio"] for r in rr], float); inpu = np.array([r["IN_per_unit_bits"] for r in rr], float)
        bi_ = np.random.default_rng(5).integers(0, len(rr), (cf["nboot"], len(rr)))
        res["i2in"] = S.summ(float(np.nanmedian(ratio)), np.nanmedian(ratio[bi_], 1)); res["in_per_unit"] = S.summ(float(np.median(inpu)), np.median(inpu[bi_], 1))
    else:
        res["i2in"] = res["in_per_unit"] = S.summ(float("nan"), [np.nan])
    # basins (descriptive), Rostami check, kinetic descriptives
    hP, JP = f2P["h"][0].cpu().numpy(), f2P["J"].cpu().numpy()
    bas, uniq = S.descend(St, hP, JP) if len(St) else (np.zeros(0, int), [])
    occ = np.bincount(bas, minlength=len(uniq)) / max(len(bas), 1)
    keep = [k for k in range(len(uniq)) if occ[k] >= 0.01]
    res["basins"] = [dict(pattern=list(uniq[k]), occupancy=float(occ[k])) for k in keep]
    res["rostami"] = dict(fitted_modes=_modes(pk_fit), empirical_modes_boot_share=float(np.mean([_modes(p) >= 2 for p in pk_b])))
    if kP is not None:
        Jk = kP["J"].cpu().numpy()
        res["A_J"] = S.asym(Jk); res["modularity"] = S.modularity((Jk + Jk.T) / 2); res["J_norm"] = float(np.linalg.norm(Jk))
    res["fit"] = dict(h=hP.tolist(), J=JP.tolist(), units=(pop["P"] if sel_units is None else [pop["P"][i] for i in sel_units]),
                      sel_units=None if sel_units is None else sel_units.tolist())
    # G-cal-Ising
    res["g_cal_ising"] = gcal_ising(N, len(D["Sf"]), tau, lam, dev, cf, ctx.mock)
    wj(ctx, "l/l3c.json", res, config=dict(lam=lam, gibbs=cf["gibbs"], fit_half="odd items", test_half="even items"))


def _modes(p):
    p = np.asarray(p, float)
    return int(sum(1 for k in range(len(p)) if p[k] > 0 and (k == 0 or p[k] > p[k - 1]) and (k == len(p) - 1 or p[k] >= p[k + 1])))


def gcal_ising(N, T, tau, lam, dev, cf, mock):
    """Five planted sets (independent, weak couplings, two Hopfield basins, near-critical, hidden-field-driven) at the measured sample
    count, thinned to the measured tau_int; pass if >= 4 of 5 are right."""
    N = max(N, 10); T = int(min(T, 20000 if not mock else 2000)); g = np.random.default_rng(17)
    thin = max(1, int(round(tau)))
    out = {}
    for kind in ("P1", "P2", "P3", "P4", "P5"):
        h = np.full(N, -2.0); J = np.zeros((N, N))
        if kind == "P2":
            J = g.normal(0, 0.05, (N, N)); J = (J + J.T) / 2
        elif kind in ("P3", "P4"):
            xi = np.where(g.random((2, N)) < 0.5, 1.0, -1.0)
            J = (xi.T @ xi) / N * (2.0 if kind == "P3" else 1.0); h = -J.sum(1) / 2
        np.fill_diagonal(J, 0)
        if kind == "P5":
            drv = g.random(T // max(thin, 1) + 1) < 0.3
            Sx = (g.random((T, N)) < np.where(np.repeat(drv, thin)[:T, None], 0.4, 0.05)).astype(np.int8)
        else:
            Sx = S.gibbs(h, J, n_chains=max(1, T // 200), sweeps=200 * thin // max(1, thin), burn=200, seed=3, thin=1, device=dev)[:T].astype(np.int8)
        half = len(Sx) // 2
        f1 = S.pl_fit(Sx[:half], lam, pair=False, device=dev); f2 = S.pl_fit(Sx[:half], lam, device=dev)
        gain = S.pl_ll(Sx[half:], f2, None, dev) - S.pl_ll(Sx[half:], f1, None, dev)
        bas, uniq = S.descend(Sx[half:], f2["h"][0].cpu().numpy(), f2["J"].cpu().numpy())
        occ = np.bincount(bas, minlength=len(uniq)) / max(len(bas), 1)
        nbas = int((occ >= 0.01).sum())
        coll = float(np.percentile(np.random.default_rng(1).choice(gain, (200, len(gain))).mean(1), 5)) > 0
        right = {"P1": nbas <= 1 and not coll, "P2": nbas <= 1, "P3": nbas >= 2, "P4": True, "P5": not coll}[kind]
        out[kind] = dict(n_basins=nbas, collective=bool(coll), right=bool(right))
    out["pass"] = sum(v["right"] for v in out.values() if isinstance(v, dict)) >= 4
    return out


# ==================================================================================================================================
# L3d
# ==================================================================================================================================
def _scaled(cf, x, lo=2):
    return max(lo, int(round(x * cf["pos_scale"])))


def entries(b, n, cf):
    """First b = 1 after >= 16 tokens of b = 0, at continuation positions 64..960 (scaled under MOCK)."""
    gap, a0, a1 = _scaled(cf, 16), _scaled(cf, 64), _scaled(cf, 960)
    out = []
    run = 0
    for t in range(n):
        if b[t] == 0:
            run += 1
        else:
            if run >= gap and a0 <= t <= a1:
                out.append(t)
            run = 0
    return out


def exits(b, n, cf):
    """Last b = 1 before >= 16 tokens of b = 0 (time-reversed entries), at positions 64..960."""
    gap, a0, a1 = _scaled(cf, 16), _scaled(cf, 64), _scaled(cf, 960)
    return [t for t in range(a0, min(a1, n - gap)) if b[t] == 1 and not b[t + 1:t + 1 + gap].any()]


def l3d(ctx):
    """EBM (Delta e, Delta s, Delta TT against matched windows), irreversibility d-hat on quartile symbols of a_P and s (post-entry
    against time-reversed pre-exit windows and matched windows; detailed-balance surrogate band), H-specific label against the 99
    controls, G-sigma (IFT check), Sagawa sigma (descriptive), A_k (non-stationarity, descriptive)."""
    tok, model, cf, H, items = setup(ctx)
    pop = load_pop(ctx)
    chains, th = load_chains(ctx, with_twin=False)
    names = [n for n, _ in channels(pop)]
    binarise(chains, th, names)
    W64 = _scaled(cf, 64, 4)
    uit = sorted({c["item"] for c in chains}); g = rng("r1bL3d", "match")
    ci_ = [i for i, n in enumerate(names) if n.startswith("C")]
    by_ft = {}
    for k, c in enumerate(chains):
        by_ft.setdefault((c["fam"], c["T"]), []).append(k)

    def events(col):
        ev = []
        for k, c in enumerate(chains):
            for t in entries(c["b"][:, col], c["n"], cf):
                if t - W64 < 0 or t + W64 > c["n"]:
                    continue
                cands = [m for m in by_ft[(c["fam"], c["T"])] if m != k and chains[m]["n"] >= t + W64 and
                         not any(abs(t2 - t) <= W64 for t2 in entries(chains[m]["b"][:, col], chains[m]["n"], cf))]
                if cands:
                    ev.append((k, t, cands[int(g.integers(len(cands)))]))
        return ev

    def de_stat(col, ev, key="e"):
        rows = []
        for k, t, m in ev:
            x, y = chains[k][key], chains[m][key]
            rows.append(((x[t:t + W64].mean() - x[t - W64:t].mean()) - (y[t:t + W64].mean() - y[t - W64:t].mean()), uit.index(chains[k]["item"])))
        return rows
    W = S.boot_counts(len(uit), cf["nboot"], seed=341)

    def boot_mean(rows):
        if not rows:
            return S.summ(float("nan"), [np.nan])
        v = np.array([r[0] for r in rows]); it = np.array([r[1] for r in rows])
        s_ = np.bincount(it, weights=v, minlength=len(uit)); n_ = np.bincount(it, minlength=len(uit)).astype(float)
        return S.summ(float(v.mean()), (W @ s_) / np.maximum(W @ n_, 1))
    evP = events(0)
    res = dict(n_events=len(evP))
    res["de"] = boot_mean(de_stat(0, evP)); res["de"]["margin"] = 0.10
    res["ds"] = boot_mean(de_stat(0, evP, "s"))
    tts = []
    for k, t, m in evP:
        X = np.stack([chains[k]["s"], chains[k]["e"]], 1); Y = np.stack([chains[m]["s"], chains[m]["e"]], 1)
        tts.append([S.rqa_tt(X[t - W64:t]), S.rqa_tt(X[t:t + W64]), S.rqa_tt(Y[t - W64:t]), S.rqa_tt(Y[t:t + W64])])
    tts = np.array(tts, float).reshape(-1, 4)
    sdTT = np.nanstd(tts) + 1e-12
    rows = [(((r[1] - r[0]) - (r[3] - r[2])) / sdTT, uit.index(chains[k]["item"])) for r, (k, t, m) in zip(tts, evP) if np.isfinite(r).all()]
    res["dtt"] = boot_mean(rows); res["dtt"]["margin"] = 0.30
    de_c = [np.mean([r[0] for r in de_stat(i, events(i))] or [np.nan]) for i in ci_[:cf["n_ctrl"]]]
    res["de_controls"] = de_c
    res["de_rank_p"] = (1 + int(np.sum(np.array(de_c) <= res["de"]["pt"]))) / (len(de_c) + 1) if isinstance(res["de"]["pt"], float) else 1.0
    # irreversibility
    sym_a = {k: S.quartiles(c["a"][:, 0]) for k, c in enumerate(chains)}
    sym_s = {k: S.quartiles(c["s"]) for k, c in enumerate(chains)}

    def irr(col, sym):
        ev = events(col)
        post, match, rev, it_p, it_r = [], [], [], [], []
        for k, t, m in ev:
            post.append(sym(k)[t:t + W64]); match.append(sym(m)[t:t + W64]); it_p.append(uit.index(chains[k]["item"]))
        for k, c in enumerate(chains):
            for te in exits(c["b"][:, col], c["n"], cf):
                if te - W64 + 1 >= 0:
                    rev.append(sym(k)[te - W64 + 1:te + 1][::-1]); it_r.append(uit.index(c["item"]))
        if not post or not rev:
            return None
        dp, dm, dr = S.dhat(post), S.dhat(match), S.dhat(rev)
        it_p, it_r = np.array(it_p), np.array(it_r)
        bp, bm, br = [], [], []
        for b in range(min(len(W), 400 if not ctx.mock else 50)):
            wi = W[b]
            pick = lambda L_, it: [w for w, i in zip(L_, it) for _ in range(int(wi[i]))]
            bp.append(S.dhat(pick(post, it_p))); bm.append(S.dhat(pick(match, it_p))); br.append(S.dhat(pick(rev, it_r)))
        bp, bm, br = np.array(bp), np.array(bm), np.array(br)
        gs = rng("r1bL-SUR", "db", col)
        band = [S.dhat([S.db_surrogate(w, gs) for w in match]) for _ in range(cf["n_sur_P"] if col == 0 else 20)]
        q5, q95 = np.percentile(band, [5, 95])
        return dict(post=dp, match=dm, revexit=dr, post_minus_rev=S.summ(dp - dr, bp - br), post_minus_match=S.summ(dp - dm, bp - bm),
                    band=[float(q5), float(q95)], band_half=float((q95 - q5) / 2), post_above_band=bool(dp > q95), match_above_band=bool(dm > q95),
                    n_post=len(post), n_rev=len(rev))
    ia = irr(0, lambda k: sym_a[k]); is_ = irr(0, lambda k: sym_s[k])
    res["irr_a"], res["irr_s"] = ia, is_
    if ia and is_:
        from r1b_core import holm
        adj, rej = holm([ia["post_minus_rev"]["p_gt0"], is_["post_minus_rev"]["p_gt0"]])
        res["irr_holm_a"], res["irr_holm"] = bool(rej[0]), adj
        cc = []
        for i in ci_[:cf["n_ctrl"]]:
            r = irr(i, lambda k, i=i: S.quartiles(chains[k]["a"][:, i]))
            cc.append(r["post"] - r["revexit"] if r else np.nan)
        res["irr_controls"] = cc
        res["irr_rank_p"] = K.p_rank(ia["post"] - ia["revexit"], [x for x in cc if np.isfinite(x)]) if any(np.isfinite(cc)) else 1.0
    else:
        res["irr_a"] = res["irr_a"] or dict(post_minus_rev=S.summ(float("nan"), [np.nan]), post_minus_match=S.summ(float("nan"), [np.nan]),
                                            post_above_band=False, match_above_band=False, band_half=0.0)
    res["g_sigma"], res["sigma"] = sigma_checks(ctx, tok, model, cf, chains)
    res["A_k"] = a_k(chains, cf)
    wj(ctx, "l/l3d.json", res, config=dict(window=W64, null="time-reversed pre-exit windows (validity fix) + detailed-balance band"))


def sigma_checks(ctx, tok, model, cf, chains):
    """G-sigma: E[exp(-sigma)] = 1 for sigma = log p_T(x) - log p_1(x) over 5-token samples at T 3 and 4 from 8 neutral prompts (exact
    identity; |mean - 1| <= 3 SE); Sagawa sigma (sentence-order reversal, prompt fixed, per token) on the T 1.0 L.R-A chains."""
    torch = K._torch()
    dev = K.model_device(model)
    n = cf["sigma_n"]
    ok, rec = True, {}
    for T in (3.0, 4.0):
        sig = []
        for p in range(8):
            pr = K.chat_ids(tok, NEU_TEMPLATE.format(s=F_NEU_SCENES[p]))
            for b0 in range(0, n, 256):
                m = min(256, n - b0)
                g = E.decode(model, tok, [pr] * m, [ctx.key("r1bL3", "IFT", p, T, b0 + j) for j in range(m)], T=T, max_new=5, stop=False)
                seqs = [pr + r["tokens"] for r in g]
                ids = torch.tensor(seqs, device=dev)
                with torch.no_grad():
                    lg = model(input_ids=ids).logits[:, len(pr) - 1:-1].float()
                tg = ids[:, len(pr):]
                l1 = torch.log_softmax(lg, -1).gather(2, tg[..., None])[..., 0].sum(1)
                lT = torch.log_softmax(lg / T, -1).gather(2, tg[..., None])[..., 0].sum(1)
                sig += (lT - l1).cpu().tolist()
        x = np.exp(-np.array(sig))
        se = x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("inf")
        rec[str(T)] = dict(mean=float(x.mean()), se=float(se), ok=bool(abs(x.mean() - 1) <= 3 * se))
        ok = ok and rec[str(T)]["ok"]
    sg = {}
    its = {}
    for c in chains:
        if c["kind"] != "A" or c["T"] != 1.0 or c["n"] < 4:
            continue
        txt = tok.decode(c["tokens"]); sents = re.split(r"(?<=[.!?])\s+", txt)
        if len(sents) < 2:
            continue
        rev = K.enc(tok, " ".join(sents[::-1]))
        its.setdefault(c["fam"], []).append((c, rev))
    for f, lst in its.items():
        vals = []
        for c, rev in lst[:(64 if not ctx.mock else 4)]:
            it = lr_items(setup(ctx)[4], cf)[f]
            pid = next((x["text"] for x in it if x["id"] == c["item"]), None)
            if pid is None:
                continue
            pr = K.chat_ids(tok, pid)
            a, b = E.tf_pass(model, tok, [pr + c["tokens"], pr + rev], [len(pr), len(pr)], batch=2)
            vals.append((np.nansum(a["lp"]) - np.nansum(b["lp"])) / max(c["n"], 1))
        sg[f] = K.dist(vals)
    return dict(checks=rec, ok=bool(ok)), sg


def a_k(chains, cf):
    """Joint-state log-loss asymmetry A_k(l) = H(s_(t+l) | s_t) - H(s_t | s_(t+l)) of k-means states of P's |z| (non-stationarity)."""
    Z = [np.asarray(c["zP"], float) for c in chains if c.get("zP") is not None and len(c["zP"]) > 20]
    if not Z:
        return None
    X = np.concatenate(Z)
    g = np.random.default_rng(3)
    out = {}
    for k in range(2, 9):
        Cn = X[g.choice(len(X), k, replace=False)]
        for _ in range(30):
            lab_ = ((X[:, None] - Cn[None]) ** 2).sum(-1).argmin(1)
            Cn = np.stack([X[lab_ == j].mean(0) if (lab_ == j).any() else Cn[j] for j in range(k)])
        labs = np.split(lab_, np.cumsum([len(z) for z in Z])[:-1])
        out[k] = {}
        for lag in (1, 2, 4, 8, 16):
            J = np.zeros((k, k)) + 1e-9
            for s in labs:
                if len(s) > lag:
                    np.add.at(J, (s[:-lag], s[lag:]), 1)
            P_ = J / J.sum(); pa, pb = P_.sum(1), P_.sum(0)
            Hab = -(P_ * np.log(P_ / pa[:, None])).sum(); Hba = -(P_ * np.log(P_ / pb[None])).sum()
            out[k][lag] = float(Hab - Hba)
    return out


# ==================================================================================================================================
# X1 dose ladder and hysteresis
# ==================================================================================================================================
def x1(ctx):
    """256 L.R-A chains at T 0.7; starts t_up / t_down by the preceding 32-token block's a_P against q20 / q80 of its position bin;
    twins by prefix teacher forcing and draw-and-discard of the chain key (G-twin-X1: 80 tokens reproduced in >= 99 %); UP: P x
    {1.5, 2, 3} and 2 random sets matched to alpha 3; DOWN: P x {0.5, 0} and 2 sets matched to alpha 0; 16-token window, 80 new tokens;
    each random set read on its own channel (part-file amendment)."""
    torch = K._torch()
    tok, model, cf, H, items = setup(ctx)
    rw = refw(ctx, model)
    pop = load_pop(ctx)
    ch, th, _ = corpus(ctx)
    its = lr_items(items, cf)
    names = [n for n, _ in channels(pop)]
    A = [c for c in ch if c["kind"] == "A" and c["T"] == 0.7][:cf["x1_rows"]]
    blk = _scaled(cf, 32, 4); w0 = _scaled(cf, 64, 4)
    allA = [c for c in ch if c["kind"] == "A"]
    nb = th["nb"]
    vals = {}
    for c in allA:
        for s in range(w0, c["n"] - blk + 1, blk):
            vals.setdefault(min(int(pos_bin(s, cf["pos_scale"])), nb - 1), []).append(float(np.mean(np.asarray(c["a"][s - blk:s, 0], float))))
    q20 = {b: np.percentile(v, 20) for b, v in vals.items()}; q80 = {b: np.percentile(v, 80) for b, v in vals.items()}
    refL, refU, rl = ref_sets(ctx, tok, items, cf)
    dm = K.Damage(model, refL)

    def dmg(ids, a):
        def su():
            pr = E.Probe(model, rw); pr.add_op(ids, "scale", val=a)
            return _ProbeCM(pr)
        return dm(su)["damage"]
    res_ = pop.get("CTRL_reserve") or []
    sets = {"UP": [res_[k] if k < len(res_) else pop["CTRL"][k % len(pop["CTRL"])] for k in (0, 1)],
            "DOWN": [res_[k] if k < len(res_) else pop["CTRL"][(k + 2) % len(pop["CTRL"])] for k in (2, 3)]}
    cal = {}
    for arm, a_top in (("UP", 3.0), ("DOWN", 0.0)):
        tgt = dmg(pop["P"], a_top)
        cal[arm] = [K.calibrate(lambda a: dmg(s, a), a_top, tgt, form="gain", max_eval=6) for s in sets[arm]]
    arms = {"UP": [("twin", None, 1.0)] + [(f"P{a}", pop["P"], a) for a in (1.5, 2.0, 3.0)] + [(f"R{j}", sets["UP"][j], cal["UP"][j]["f"]) for j in (0, 1)],
            "DOWN": [("twin", None, 1.0)] + [(f"P{a}", pop["P"], a) for a in (0.5, 0.0)] + [(f"R{j}", sets["DOWN"][j], cal["DOWN"][j]["f"]) for j in (0, 1)]}
    chans = [("P", pop["P"])] + [(f"R{arm}{j}", sets[arm][j]) for arm in ("UP", "DOWN") for j in (0, 1)]
    out = {}
    W = cf["l3b_win"]
    for arm in ("UP", "DOWN"):
        starts = []
        for c in A:
            t0 = None
            for s in range(w0, c["n"] - blk + 1, blk):
                m = float(np.mean(np.asarray(c["a"][s - blk:s, 0], float))); b = min(int(pos_bin(s, cf["pos_scale"])), nb - 1)
                if (arm == "UP" and m <= q20[b]) or (arm == "DOWN" and m >= q80[b]):
                    t0 = s; break
            if t0 is not None and c["n"] >= t0 + cf["x1_new"]:
                starts.append((c, t0))
        conds = arms[arm]
        rows, meta = [], []
        for c, t0 in starts:
            pr_ids = K.chat_ids(tok, next(x["text"] for x in its[c["fam"]] if x["id"] == c["item"]))
            for cn in conds:
                rows.append((pr_ids + c["tokens"][:t0], c["key"], t0, c)); meta.append(cn[0])
        recs = []
        for b0 in range(0, len(rows), cf["batch"]):
            rb, mb = rows[b0:b0 + cf["batch"]], meta[b0:b0 + cf["batch"]]

            def run():
                pr = E.Probe(model, rw, chans=chans)
                for cn in conds[1:]:
                    rmask = torch.tensor([m == cn[0] for m in mb])
                    if rmask.any():
                        pr.add_op(cn[1], "scale", val=cn[2], rows=rmask, on=False)
                Tn = max(len(r[0]) for r in rb)
                last = torch.zeros(len(rb), Tn, dtype=torch.bool); last[:, -1] = True

                def on_step(step, probe):
                    for o in probe.ops:
                        o["on"] = step <= W; o["pos"] = last if step == 0 else None
                try:
                    g = E.decode(model, tok, [r[0] for r in rb], [r[1] for r in rb], T=0.7, max_new=cf["x1_new"], probe=pr, on_step=on_step,
                                 advance=[r[2] for r in rb])
                finally:
                    pr.remove()
                return [dict(tokens=r["tokens"], a=r["a"].astype(np.float32)) for r in g]
            recs += ctx.cached(f"x1_{arm}_{b0}", run)
        out[arm] = dict(meta=meta, rows=[(r[2], r[3]["item"], r[3]["tokens"][r[2]:r[2] + cf["x1_new"]]) for r in rows], recs=recs, n_starts=len(starts))
    # statistics
    st = {}
    e0, e1, l0, l1_ = W, 2 * W, _scaled(cf, 32 + 16, 3) + W // 2, _scaled(cf, 64 + 16, 4) + W // 2
    e0, e1 = W, 2 * W
    l0, l1_ = 3 * W, 4 * W
    for arm in ("UP", "DOWN"):
        d = out[arm]; conds = [c[0] for c in arms[arm]]; nc = len(conds)
        n = len(d["recs"]) // max(nc, 1)
        if n == 0:
            st[arm] = dict(defined=False); continue
        tw_ok = [d["recs"][i * nc]["tokens"][:cf["x1_new"]] == d["rows"][i * nc][2][:cf["x1_new"]] for i in range(n)]
        it = np.array([sorted({r[1] for r in d["rows"]}).index(d["rows"][i * nc][1]) for i in range(n)])
        Wb = S.boot_counts(int(it.max()) + 1, cf["nboot"], seed=351)

        def delta(ci, chn, sl):
            v = []
            for i in range(n):
                a, t = d["recs"][i * nc + ci]["a"], d["recs"][i * nc]["a"]
                m = min(len(a), len(t), sl.stop)
                v.append(float(np.mean(a[sl.start:m, chn] - t[sl.start:m, chn])) if m > sl.start else np.nan)
            v = np.array(v); s_ = np.bincount(it, weights=np.nan_to_num(v), minlength=Wb.shape[1]); n_ = np.bincount(it, weights=np.isfinite(v).astype(float), minlength=Wb.shape[1])
            return float(np.nanmean(v)), (Wb @ s_) / np.maximum(Wb @ n_, 1)

        def rho(ci, chn):
            de, deb = delta(ci, chn, slice(e0, e1)); dl, dlb = delta(ci, chn, slice(l0, l1_))
            dE = S.summ(de, deb)
            ok = _lb(dE) or (isinstance(dE["hi"], float) and dE["hi"] < 0)
            return ok, dl / de if de else np.nan, dlb / np.where(deb != 0, deb, np.nan)
        pidx = [i for i, c in enumerate(conds) if c.startswith("P")]
        top, low = pidx[-1] if arm == "UP" else pidx[-1], pidx[0]
        okT, rT, rTb = rho(top, 0); okL, rL, rLb = rho(low, 0)
        rr = [rho(i, 1 + (0 if arm == "UP" else 2) + j) for j, i in enumerate([i for i, c in enumerate(conds) if c.startswith("R")])]
        rR = np.nanmean([x[1] for x in rr]) if rr else np.nan; rRb = np.nanmean(np.stack([x[2] for x in rr]), 0) if rr else np.full(len(rTb), np.nan)
        st[arm] = dict(defined=bool(okT), twin_ok_share=float(np.mean(tw_ok)), rho_top=S.summ(rT, rTb), top_minus_low=S.summ(rT - rL, rTb - rLb, margin=0.2),
                       top_minus_R=S.summ(rT - rR, rTb - rRb), n=n, damage_cal=[dict(f=c_["f"], damage=c_["damage"], in_band=c_["in_band"]) for c_ in cal[arm]])
    st["g_twin_x1"] = all(st[a].get("twin_ok_share", 0) >= 0.99 for a in ("UP", "DOWN") if st[a].get("n"))
    wj(ctx, "l/x1.json", st, config=dict(item="X1", window=W, new=cf["x1_new"], rows=cf["x1_rows"]), labels=rl)


# ==================================================================================================================================
# X3 toy loop
# ==================================================================================================================================
def x3(ctx):
    """Weight-tied looped core h_(r+1) = tanh(W h_r + U [x; y; task] + b) on the W1-like one-hot shift world (x + y + c) mod 17
    (12 pretrained shifts, 4 new shifts learned by column search), R ~ U{4..12} in training, R = 32 at test; convergence, toy H-units,
    push-release across iterations (K_toy) against 19 random unit sets, pairwise PL on fixed-point spins against 19 random populations."""
    torch = K._torch()
    cf = setup(ctx)[2]
    out = {}
    nW = cf["x3_width"]
    for seed in range(cf["x3_seeds"]):
        def run():
            torch.manual_seed(seed); g = np.random.default_rng(seed)
            sh = g.permutation(17); pre, new = sh[:12], sh[12:16]
            nt = 16
            W_ = torch.nn.Parameter(torch.randn(nW, nW) * 0.1); U = torch.nn.Parameter(torch.randn(nW, 34 + nt) * 0.1)
            b = torch.nn.Parameter(torch.zeros(nW)); V = torch.nn.Parameter(torch.randn(17, nW) * 0.1)
            opt = torch.optim.AdamW([W_, U, b, V], lr=1e-3, weight_decay=1.0)
            xs, ys = np.meshgrid(np.arange(17), np.arange(17)); xs, ys = xs.ravel(), ys.ravel()

            def inp(x, y, t):
                v = torch.zeros(len(x), 34 + nt); v[torch.arange(len(x)), torch.as_tensor(x)] = 1; v[torch.arange(len(x)), 17 + torch.as_tensor(y)] = 1
                v[torch.arange(len(x)), 34 + torch.as_tensor(t)] = 1
                return v

            def loop(v, R, push=None):
                h = torch.zeros(v.shape[0], nW); hs = []
                for r in range(R):
                    h = torch.tanh(h @ W_.T + v @ U.T + b)
                    if push is not None and push[0] <= r < push[1]:
                        h = h.clone(); h[:, push[2]] = h[:, push[2]] * 2
                    hs.append(h)
                return h, hs
            for step in range(cf["x3_steps"]):
                k = g.integers(0, 12, 256); i = g.integers(0, 289, 256)
                R = int(g.integers(4, 13))
                v = inp(xs[i], ys[i], k)
                h, _ = loop(v, R)
                loss = torch.nn.functional.cross_entropy(h @ V.T, torch.as_tensor((xs[i] + ys[i] + pre[k]) % 17))
                opt.zero_grad(); loss.backward(); opt.step()
            # column search for the 4 new shifts (only the task column learned)
            for j, c in enumerate(new):
                col = torch.nn.Parameter(torch.zeros(nW)); o2 = torch.optim.Adam([col], lr=1e-2)
                for step in range(max(20, cf["x3_steps"] // 20)):
                    i = g.integers(0, 289, 128); v = inp(xs[i], ys[i], np.full(128, 12 + j))
                    with torch.no_grad():
                        U[:, 34 + 12 + j] = 0
                    h = torch.zeros(128, nW)
                    for r in range(int(g.integers(4, 13))):
                        h = torch.tanh(h @ W_.T + v @ U.T + col + b)
                    l2 = torch.nn.functional.cross_entropy(h @ V.T, torch.as_tensor((xs[i] + ys[i] + c) % 17))
                    o2.zero_grad(); l2.backward(); o2.step()
                with torch.no_grad():
                    U[:, 34 + 12 + j] = col.detach()
            with torch.no_grad():
                tasks = np.repeat(np.arange(16), 289); xx, yy = np.tile(xs, 16), np.tile(ys, 16)
                v = inp(xx, yy, tasks)
                h32, hs = loop(v, 33)
                conv = ((hs[32] - hs[31]).norm(dim=1) / hs[31].norm(dim=1).clamp_min(1e-9) < 1e-3).numpy()
                shifts = np.concatenate([pre, new])
                wrong = ((hs[31] @ V.T).argmax(1).numpy() != (xx + yy + shifts[tasks]) % 17)
                spins = (hs[31].abs() > hs[31].abs().median()).float()
                half = np.arange(len(xx)) % 2 == 0
                w_ = S.l1_select(spins[half], torch.as_tensor(wrong[half].astype(np.float32)), 1.0)
                Hu = np.flatnonzero(w_.numpy() > 0)[:max(4, nW // 16)]
                if len(Hu) == 0:
                    Hu = np.argsort(-w_.numpy())[:4]

                def kt(units):
                    _, hp = loop(v, 33, push=(8, 12, torch.as_tensor(units)))
                    d13 = (hp[12][:, units] - hs[12][:, units]).mean().item(); d32 = (hp[31][:, units] - hs[31][:, units]).mean().item()
                    return d32 / d13 if d13 else np.nan
                KH = kt(Hu); KR = [kt(g.choice(nW, len(Hu), replace=False)) for _ in range(19)]
                Sf = spins.numpy()
                dpl = lambda u: float((S.pl_ll(Sf[~half][:, u], S.pl_fit(Sf[half][:, u], 1e-3), None) -
                                       S.pl_ll(Sf[~half][:, u], S.pl_fit(Sf[half][:, u], 1e-3, pair=False), None)).mean())
                dH = dpl(Hu) if len(Hu) >= 2 else np.nan; dR = [dpl(g.choice(nW, len(Hu), replace=False)) for _ in range(19)] if len(Hu) >= 2 else []
            return dict(wrong_fp_share=float((conv & wrong).sum() / max(wrong.sum(), 1)), K_toy=KH, K_rand=KR, dpl=dH, dpl_rand=dR,
                        n_wrong=int(wrong.sum()), conv_share=float(conv.mean()), H_units=Hu.tolist())
        out[seed] = ctx.cached(f"x3_{seed}", run, fmt="json")
    st = dict(seeds=out, n_seeds_wrongfp=sum(v["wrong_fp_share"] >= 0.2 for v in out.values()),
              n_seeds_attr=sum((v["K_toy"] >= 0.9) and all(v["K_toy"] > (k if np.isfinite(k) else -1e9) for k in v["K_rand"]) for v in out.values()),
              n_seeds_ising=sum(np.isfinite(v["dpl"]) and v["dpl_rand"] and all(v["dpl"] > d for d in v["dpl_rand"]) for v in out.values()))
    wj(ctx, "l/x3.json", st, config=dict(item="X3", width=nW, seeds=cf["x3_seeds"], steps=cf["x3_steps"],
                                          world="W1-like shift world (x + y + c) mod 17, 12 + 4 shifts"),
       labels=["K_toy ratio uses a single replicate per seed (descriptive per seed)"])


# ==================================================================================================================================
# X4 associative completion; X5 L.gv
# ==================================================================================================================================
def x4(ctx):
    """256 L3b-type keys, first answer token, window 16: a frozen fraction f in {0.25, 0.5} of P's units gets gain g_j = mean |z_j| in
    H-high states / overall (L.R fit half); CI = live fraction-on of the unclamped units, pushed minus twin; 2 CTRL populations clamped to
    their own high templates with a damage-calibrated gain; CI_pred from Gibbs sampling of L3c's static model with the same units clamped."""
    torch = K._torch()
    tok, model, cf, H, items = setup(ctx)
    rw = refw(ctx, model)
    pop = load_pop(ctx)
    P = pop["P"] if len(pop["P"]) >= 8 else pop.get("H_plus", pop["P"])
    ch, th, _ = corpus(ctx)
    fitc = [c for c in ch if c.get("zP") is not None and len(c["zP"])][: (64 if not ctx.mock else 6)]
    Z = np.concatenate([np.asarray(c["zP"], float) for c in fitc]); bP = np.concatenate([np.asarray(c["a"][:, 0], float) for c in fitc])
    hi_ = bP > np.quantile(bP, 0.95)
    gains = Z[hi_].mean(0) / np.maximum(Z.mean(0), 1e-9) if hi_.any() else np.ones(Z.shape[1])
    pr_items = l3b_prompts(items, cf)
    keys = [(p, s) for p in range(len(pr_items)) for s in range(cf["l3b_seeds"])][:cf["x4_keys"]]
    prompts = [K.chat_ids(tok, QA_TEMPLATE.format(q=it["question"])) for _, it in pr_items]
    l3c = K.load_json(ctx.path("l/l3c.json"))
    refL, refU, rl = ref_sets(ctx, tok, items, cf)
    dm = K.Damage(model, refL)
    res_ = pop.get("CTRL_reserve") or []
    rsets = [res_[k] if k < len(res_) else pop["CTRL"][k] for k in (4, 5)]
    units, off = th["units"], th["off"]
    thrS = th["thrS"]
    out = {}
    for f in (0.25, 0.5):
        g = rng("r1bLX", "x4", f)
        cl = np.sort(g.choice(len(P), max(1, int(round(f * len(P)))), replace=False)); un = np.setdiff1d(np.arange(len(P)), cl)
        gP = np.ones(len(P)); gP[cl] = gains[cl] if len(gains) == len(P) else 2.0

        def dmg_g(ids, gv):
            def su():
                pr = E.Probe(model, rw); pr.add_op(ids, "scale", per_neuron=gv)
                return _ProbeCM(pr)
            return dm(su)["damage"]
        tgt = dmg_g(P, gP)
        rows = []
        for j, rs in enumerate(rsets):
            gr = np.ones(len(rs)); clr = np.sort(g.choice(len(rs), len(cl), replace=False)); gr[clr] = float(np.mean(gP[cl]))
            cal = K.calibrate(lambda lam: dmg_g(rs, 1 + lam * (gr - 1)), 1.0, tgt, form="push", max_eval=6)
            rows.append((rs, 1 + cal["f"] * (gr - 1), clr, cal))
        conds = [("twin", None, None, None), ("P", P, gP, cl)] + [(f"R{j}", r[0], r[1], r[2]) for j, r in enumerate(rows)]
        spin_all = list(P) + [u for r in rsets for u in r]
        res = []
        for b0 in range(0, len(keys), max(1, cf["batch"] // len(conds))):
            kb = keys[b0:b0 + max(1, cf["batch"] // len(conds))]
            rr = [prompts[p] for p, s in kb for _ in conds]; meta = [cn[0] for _ in kb for cn in conds]

            def run():
                pr = E.Probe(model, rw, spins=spin_all)
                for cn in conds[1:]:
                    pr.add_op(cn[1], "scale", per_neuron=cn[2], rows=torch.tensor([m == cn[0] for m in meta]), on=False)
                Tn = max(len(x) for x in rr); last = torch.zeros(len(rr), Tn, dtype=torch.bool); last[:, -1] = True

                def on_step(step, probe):
                    for o in probe.ops:
                        o["on"] = step <= cf["l3b_win"]; o["pos"] = last if step == 0 else None
                try:
                    g_ = E.decode(model, tok, rr, [ctx.key("r1bL3", p, s) for p, s in kb for _ in conds], T=0.7, max_new=cf["l3b_win"], probe=pr,
                                  on_step=on_step, spin_fn=lambda st, z: z.float().cpu().numpy())
                finally:
                    pr.remove()
                return [np.stack(x["z"]) if x.get("z") else np.zeros((0, len(spin_all))) for x in g_]
            zs = ctx.cached(f"x4_{f}_{b0}", run)
            thrP = np.asarray(thrS[0, 0, off["P"][0]:off["P"][1]].float().numpy())
            for k in range(len(kb)):
                blk_ = zs[k * len(conds):(k + 1) * len(conds)]
                tw = blk_[0]
                fr = lambda z, sel, thr: float((z[:, sel] > thr[sel]).mean()) if len(z) else np.nan
                ciP = fr(blk_[1][:, :len(P)], un, thrP) - fr(tw[:, :len(P)], un, thrP)
                cr = []
                for j, (rs, gr, clr, cal) in enumerate(rows):
                    o0 = len(P) + sum(len(r) for r in rsets[:j]); unr = np.setdiff1d(np.arange(len(rs)), clr)
                    thr = np.full(len(rs), np.median(blk_[0][:, o0:o0 + len(rs)]) if len(tw) else 0.0)
                    cr.append(fr(blk_[2 + j][:, o0:o0 + len(rs)], unr, thr) - fr(tw[:, o0:o0 + len(rs)], unr, thr))
                res.append((ciP, float(np.nanmean(cr)), pr_items[kb[k][0]][1]["id"]))
        v = np.array([r[0] - r[1] for r in res]); its_ = sorted({r[2] for r in res}); it = np.array([its_.index(r[2]) for r in res])
        Wb = S.boot_counts(len(its_), cf["nboot"], seed=361)
        s_ = np.bincount(it, weights=np.nan_to_num(v), minlength=len(its_)); n_ = np.bincount(it, minlength=len(its_)).astype(float)
        pred = None
        if l3c.get("fit") and len(l3c["fit"]["h"]) == len(P):
            hh, JJ = np.array(l3c["fit"]["h"]), np.array(l3c["fit"]["J"])
            base = S.gibbs(hh, JJ, n_chains=64, sweeps=cf["gibbs"][1] // 4, burn=cf["gibbs"][2] // 4, seed=5)
            clp = S.gibbs(hh, JJ, n_chains=64, sweeps=cf["gibbs"][1] // 4, burn=cf["gibbs"][2] // 4, seed=5, clamp=dict(idx=cl, val=np.ones(len(cl))))
            pred = float(clp[:, un].mean() - base[:, un].mean())
        ci_obs = float(np.nanmean([r[0] for r in res]))
        out[str(f)] = dict(diff=S.summ(float(np.nanmean(v)), (Wb @ s_) / np.maximum(Wb @ n_, 1), margin=0.01), CI_P=ci_obs,
                           CI_R=float(np.nanmean([r[1] for r in res])), CI_pred=pred,
                           pairwise_predicts=bool(pred and 0.5 <= ci_obs / pred <= 2) if pred else None,
                           damage_target=tgt, cal=[dict(f=r[3]["f"], damage=r[3]["damage"], in_band=r[3]["in_band"]) for r in rows])
    wj(ctx, "l/x4.json", out, config=dict(item="X4", keys=len(keys), window=cf["l3b_win"]), labels=rl)


def x5(ctx):
    """L.gv (Aletheia-style generate-verify-revise-abstain): gains toward a basin pattern on 32 neutral prompts (64 tokens, T 0.7); the
    unmodified model re-reads the text (verifier: re-read pattern descended to its basin), 'correct' / 'minor fix needed' / 'critically
    flawed' against 20 random patterns of equal size and gain; reviser gains x 1.25 or prompt family neutral -> QA; <= 3 attempts, then
    abstain; lexical content statistics; the verifier's error rate on known answers (observed-state patterns accepted, random rejected)."""
    torch = K._torch()
    tok, model, cf, H, items = setup(ctx)
    rw = refw(ctx, model)
    pop = load_pop(ctx)
    l3c = K.load_json(ctx.path("l/l3c.json"))
    ch, th, _ = corpus(ctx)
    if not l3c.get("basins") or not l3c.get("fit"):
        wj(ctx, "l/x5.json", dict(verdict="not testable (no basin)"), config=dict(item="X5")); return
    hh, JJ = np.array(l3c["fit"]["h"]), np.array(l3c["fit"]["J"]); Pu = l3c["fit"]["units"]
    thr = th["thrS"][0, 0, th["off"]["P"][0]:th["off"]["P"][1]].float().numpy()
    if l3c["fit"].get("sel_units"):
        thr = thr[l3c["fit"]["sel_units"]]
    bas = sorted(l3c["basins"], key=lambda b: -b["occupancy"])
    target = np.array(bas[1]["pattern"] if len(bas) > 1 else bas[0]["pattern"], float)
    neu = [K.chat_ids(tok, NEU_TEMPLATE.format(s=s)) for s in F_NEU_SCENES[:cf["x5_prompts"]]]
    qa = [K.chat_ids(tok, QA_TEMPLATE.format(q=it["question"])) for _, it in l3b_prompts(items, cf)][:cf["x5_prompts"]]

    def gen_verify(pattern, delta, prompts, tag):
        gv = 1 + delta * (2 * pattern - 1)
        pr = E.Probe(model, rw); pr.add_op(Pu, "scale", per_neuron=gv)
        try:
            g = E.decode(model, tok, prompts, [ctx.key("r1bLX", "x5", tag, i) for i in range(len(prompts))], T=0.7, max_new=cf["x5_new"])
        finally:
            pr.remove()
        land = []
        with E.Probe(model, rw, spins=Pu) as pr2:
            tf = E.tf_pass(model, tok, [p + r["tokens"] for p, r in zip(prompts, g)], [len(p) for p in prompts], probe=pr2, batch=cf["batch"])
        for x in tf:
            if x.get("zs") is None or not len(x["zs"]):
                land.append(None); continue
            pat = ((x["zs"].float().numpy() > thr[None]).mean(0) > 0.5).astype(float)
            _, u = S.descend(pat[None], hh, JJ)
            land.append(np.array(u[0], float))
        return g, land

    def verdict(land, null95):
        hit = np.mean([l is not None and (l == target).all() for l in land])
        near = np.mean([l is not None and np.abs(l - target).sum() <= 4 for l in land])
        if hit >= 0.6 and hit > null95:
            return "correct", hit
        if near >= 0.6 or 0.4 <= hit < 0.6:
            return "minor fix needed", hit
        return "critically flawed", hit
    g0 = rng("r1bLX", "x5null")
    null = []
    for k in range(20 if not ctx.mock else 2):
        rp = np.zeros(len(target)); rp[g0.choice(len(target), int(target.sum()), replace=False)] = 1
        _, land = ctx.cached(f"x5_null_{k}", lambda: gen_verify(rp, 0.5, neu, f"null{k}"))
        null.append(np.mean([l is not None and (l == target).all() for l in land]))
    null95 = float(np.percentile(null, 95))
    delta, prompts, fam, attempts, v = 0.5, neu, "neutral", [], None
    for a in range(3):
        g, land = ctx.cached(f"x5_try_{a}", lambda: gen_verify(target, delta, prompts, f"try{a}"))
        v, hit = verdict(land, null95)
        attempts.append(dict(attempt=a + 1, delta=delta, family=fam, verdict=v, hit=hit))
        if v == "correct":
            break
        if v == "minor fix needed":
            delta *= 1.25
        else:
            prompts, fam = qa, "QA"
    final = v if v == "correct" else "abstain"
    texts = [tok.decode(r["tokens"]) for r in g]
    lex = dict(refusal=float(np.mean([any(w in t.lower() for w in ("i don't know", "i cannot", "not sure")) for t in texts])),
               capitalised=float(np.mean([len(re.findall(r"\b[A-Z][a-z]+", t)) / max(len(t.split()), 1) for t in texts])),
               digits=float(np.mean([len(re.findall(r"\d", t)) / max(len(t), 1) for t in texts])))
    # verifier error rate on known answers: observed-state texts (corpus chains whose re-read lands in the target) must be accepted
    known = [c for c in ch if c["kind"] == "A"][: (16 if not ctx.mock else 2)]
    acc = []
    with E.Probe(model, rw, spins=Pu) as pr2:
        its = lr_items(items, cf)
        for c in known:
            pid = next((x["text"] for x in its[c["fam"]] if x["id"] == c["item"]), None)
            if pid is None:
                continue
            p_ = K.chat_ids(tok, pid)
            x = E.tf_pass(model, tok, [p_ + c["tokens"]], [len(p_)], probe=pr2)[0]
            if x.get("zs") is None or not len(x["zs"]):
                continue
            pat = ((x["zs"].float().numpy() > thr[None]).mean(0) > 0.5).astype(float)
            _, u = S.descend(pat[None], hh, JJ)
            acc.append(bool((np.array(u[0]) == np.array(u[0])).all()))
    err = dict(original_accept_rate=float(np.mean(acc)) if acc else None, random_reject_rate=float(1 - np.mean(null)) if null else None)
    wj(ctx, "l/x5.json", dict(attempts=attempts, final=final, null_hits=null, null95=null95, lexical=lex, verifier_error=err,
                              target=target.tolist()), config=dict(item="X5", prompts=cf["x5_prompts"], new=cf["x5_new"]), labels=["descriptive"])


def session_B(ctx):
    """l_session_B.json: index of L's r1b_L_B outputs with their sha256 (contract file)."""
    import glob, os
    files = sorted(glob.glob(ctx.path("l/*.json")))
    wj(ctx, "l_session_B.json", dict(files={os.path.relpath(f, ctx.work): K.sha256_file(f) for f in files},
                                     measurements=["L.R", "L2a", "L2b", "L2c", "L3a", "L3c", "L3d", "L5 generation", "X1", "X3", "X4", "X5"]),
       config=dict(session="r1b_L_B"))

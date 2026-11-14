"""G-owned L1 statistical protocol, ported from the frozen L implementation.

The estimators, grids, permutation floors, planted-code calibration and decision
rules are unchanged. Inputs are Gemma's own ragged-neuron features and splits.
No Qwen artifact or module-level replacement is used. Causal-erasure collection
is separate and its absence is recorded rather than replaced by a null effect.
"""
import math
import numpy as np
import r1b_core as K
import r1b_l_stats as S
from r1b_l_base import rng
from r1b_g_l import setup, dev, _sl, _cett, load_pop, loadpt, save, wj, _path, H_C, selector

PROFILES = ("localist", "redundant", "dense", "thr16", "thr64", "xor3")

def top_set(w, k, X=None, y=None):
    """Top k units by positive weight; if fewer, filled by |weight| of an L2-logistic refit (C = 1, standardised)."""
    w = w.detach().cpu().numpy()
    pos = np.flatnonzero(w > 0)
    sel = pos[np.argsort(-w[pos])][:k].tolist()
    if len(sel) < k and X is not None:
        torch = K._torch()
        Xs = X.clone().sub_(X.mean(0)).div_(X.std(0).clamp_min(1e-6))[None]
        W, b = S.logreg(Xs, y[None].float(), torch.ones_like(y[None]).float(), torch.tensor([1.0 / X.shape[0]]))
        order = np.argsort(-np.abs(W[0].cpu().numpy()))
        sel += [int(j) for j in order if int(j) not in sel][:k - len(sel)]
    return sorted(int(j) for j in sel)

def hull(X, members, thr=0.7, exclude=()):
    """Members plus every unit with |r| > thr to a member on the S_fit CETT matrix X [n, N] (partners ordered by max |r|)."""
    if not len(members):
        return [], []
    # Identical column correlations, bounded independently of Gemma's total N.
    Xm = X[:, members] - X[:, members].mean(0)
    sm = Xm.norm(dim=0).clamp_min(1e-9)
    mx = np.empty(X.shape[1], dtype=np.float32)
    for start in range(0, X.shape[1], 4096):
        end = min(start + 4096, X.shape[1])
        Xc = X[:, start:end] - X[:, start:end].mean(0)
        sd = Xc.norm(dim=0).clamp_min(1e-9)
        R = (Xm.T @ Xc) / (sm[:, None] * sd[None])
        mx[start:end] = R.abs().max(0).values.cpu().numpy()
    mx[members] = -1
    for j in exclude:
        mx[j] = -1
    part = np.flatnonzero(mx > thr)
    part = part[np.argsort(-mx[part])]
    return sorted(set(int(j) for j in members) | set(part.tolist())), part.tolist()

def frag_losses(ctx, name, X, y, folds, frags, budget=4e8):
    """Held-out loss rows [F, n] (bits) for fragments (lists of feature indices), batched by size, resumable."""
    torch = K._torch()
    dv = X.device
    out = [None] * len(frags)
    by = {}
    for i, f in enumerate(frags):
        by.setdefault(len(f), []).append(i)
    for m, idx in sorted(by.items()):
        per = max(1, int(budget // max(1, X.shape[0] * m * 4)))
        for c0 in range(0, len(idx), per):
            ch = idx[c0:c0 + per]

            def run():
                Xf = torch.stack([X[:, torch.as_tensor(frags[i], device=dv)] for i in ch])
                yy = y[ch] if y.dim() == 2 else y                 # per-fragment labels: the permutation floors
                return S.ihat_rows(Xf, yy.to(dv), folds.to(dv)).cpu()
            r = ctx.cached(f"{name}_{m}_{c0}", run)
            for j, i in enumerate(ch):
                out[i] = r[j]
    return torch.stack(out) if out else torch.zeros(0)

def _uniform(g, N, m, n, excl=None):
    pool = np.arange(N) if excl is None else np.setdiff1d(np.arange(N), excl)
    return [np.sort(g.choice(pool, m, replace=False)).tolist() for _ in range(n)]

def l1_pip(ctx):
    """L1 (i) U, M, XH families with permutation floors at every grid size, (iii) synergy / monogamy pairs, (iv) latent scalars,
    (v) pre-answer PIP (U sweep, floors at 3 sizes). Saves the held-out loss rows (bootstraps in l1_stats)."""
    torch = K._torch()
    tok, model, cf, H, items = setup(ctx)
    pop = load_pop(ctx)
    X, y, folds, ids, d = _sl(ctx)
    X = X.to(dev(model))
    N = X.shape[1]; P = pop["P"]; mP = len(P)
    grid = sorted(set([m for m in cf["pip_grid"] if m <= N] + [mP]))
    g = rng("r1bL1", "pip")
    res = dict(grid=grid, mP=mP)
    fr = {}
    for m in grid:
        fr[f"U{m}"] = _uniform(g, N, m, cf["pip_frag"])
        fr[f"XH{m}"] = _uniform(g, N, m, cf["pip_frag"], excl=pop["hull_P"]) if N - len(pop["hull_P"]) >= m else []
    flat_fr = [(k, i, f) for k, L_ in fr.items() for i, f in enumerate(L_)]
    lossU = frag_losses(ctx, "fam", X, y, folds, [f for _, _, f in flat_fr])
    res["fam"] = {k: lossU[[j for j, (kk, _, _) in enumerate(flat_fr) if kk == k]] for k in fr}
    res["M"] = frag_losses(ctx, "M", X, y, folds, [pop["P"]] + pop["CTRL"])
    # permutation floors at every grid size: 100 permutations, each with a fresh U fragment
    perm_y, perm_f, lab_m = [], [], []
    for m in grid:
        for p in range(cf["pip_perm"]):
            perm_y.append(y[torch.as_tensor(rng("r1bL1", "perm", m, p).permutation(len(y)))]); perm_f.append(_uniform(g, N, m, 1)[0])
            lab_m.append(m)
    res["floor"] = frag_losses(ctx, "floor", X, torch.stack(perm_y), folds, perm_f)
    res["floor_m"] = lab_m
    # synergy pairs
    syn = []
    for m in [s for s in cf["syn_grid"] if 2 * s <= N]:
        for _ in range(cf["pip_frag"]):
            ab = g.choice(N, 2 * m, replace=False)
            syn.append((m, sorted(ab[:m].tolist()), sorted(ab[m:].tolist()), sorted(ab.tolist())))
    ls = frag_losses(ctx, "syn", X, y, folds, [s[1] for s in syn] + [s[2] for s in syn] + [s[3] for s in syn])
    ns = len(syn)
    res["syn"] = dict(m=[s[0] for s in syn], A=ls[:ns], B=ls[ns:2 * ns], AB=ls[2 * ns:])
    # latent scalars: answer-token entropy; PC1 of S_L CETT fitted on S_fit
    sp, cl = np.array(d["split"]), np.array(d["cls"])
    fit = np.flatnonzero((sp == "S_fit") & np.isin(cl, ["CC", "CI"]))
    Xf = d["cmean"][fit].float().to(X.device)
    mu, sd = Xf.mean(0), Xf.std(0).clamp_min(1e-6)
    v, _ = S.power_pc1((Xf - mu) / sd)
    pc = (((X - mu) / sd) @ v)[:, None]
    idx = {i: j for j, i in enumerate(d["ids"])}
    ent = torch.tensor([float(d["ent"][idx[i]]) for i in ids], device=X.device)[:, None]
    res["latent"] = dict(entropy=S.ihat_rows(ent[None], y.to(X.device), folds.to(X.device)).cpu()[0],
                         pc1=S.ihat_rows(pc[None], y.to(X.device), folds.to(X.device)).cpu()[0])
    # pre-answer PIP on last-prompt-token CETT
    Xp = _sl(ctx, "clast")[0].to(X.device)
    pre_grid = sorted(set(min(grid, key=lambda m: abs(math.log(m) - math.log(t))) for t in (8, 64, 1024)))
    pre_fr = {m: _uniform(g, N, m, cf["pip_frag"]) for m in grid}
    lp_ = frag_losses(ctx, "pre", Xp, y, folds, [f for m in grid for f in pre_fr[m]])
    res["pre"] = dict(grid=grid, U=lp_.view(len(grid), cf["pip_frag"], -1), floor_grid=pre_grid)
    py, pf = [], []
    for m in pre_grid:
        for p in range(cf["pip_perm"]):
            py.append(y[torch.as_tensor(rng("r1bL1", "preperm", m, p).permutation(len(y)))]); pf.append(_uniform(g, N, m, 1)[0])
    res["pre"]["floor"] = frag_losses(ctx, "prefloor", Xp, torch.stack(py), folds, pf).view(len(pre_grid), cf["pip_perm"], -1)
    res["ids"] = ids
    save(ctx, "l/l1_pip.pt", res, config=dict(grid=grid, frag=cf["pip_frag"], perm=cf["pip_perm"], syn=cf["syn_grid"]))

def l1_resel(ctx):
    """Knock-out reselection: H_1 = P; H_r = the selector on S_fit with H_1..H_(r-1) and their |r| > 0.7 clusters excluded, capped at
    |P|; I-hat on S_L. Also Priv and rho_2 on the 20 SO-HN boot sets (selection-robustness label)."""
    torch = K._torch()
    tok, model, cf, H, items = setup(ctx)
    pop = load_pop(ctx)
    d = _cett(ctx)
    sp, cl = np.array(d["split"]), np.array(d["cls"])
    fit = np.flatnonzero((sp == "S_fit") & np.isin(cl, ["CC", "CI"]))
    Xf = d["cmean"][fit].float().to(dev(model)); yf = torch.tensor((cl[fit] == "CI").astype(np.float32), device=Xf.device)
    X, y, folds, ids, _ = _sl(ctx)
    X = X.to(Xf.device)
    sets = [pop["P"]]
    excl = torch.zeros(Xf.shape[1], dtype=torch.bool, device=Xf.device)
    for r in range(2, cf["n_resel"] + 1):
        h_, _ = hull(Xf, sets[-1])
        excl[h_] = True
        w, _ = selector(Xf, yf, H_C(ctx), excl)
        sets.append(top_set(w * (~excl), len(pop["P"])))
    fr = [s for s in sets if s] + [b for b in pop["boot"] if b]
    lr_ = frag_losses(ctx, "resel", X, y, folds, fr)
    save(ctx, "l/l1_resel.pt", dict(sets=sets, loss=lr_[:len([s for s in sets if s])], boot_loss=lr_[len([s for s in sets if s]):],
                                    boot=pop["boot"]), config=dict(n=cf["n_resel"]))

def plant(X, kind, s, amp, g, P_size):
    """lean's planted codes (sim_pip2.py) in the real matrix: signal added to columns (z-scored units), amplitude amp."""
    torch = K._torch()
    n, N = X.shape
    Xp = X.clone()
    sd = X.std(0).clamp_min(1e-6)
    st = torch.as_tensor(s, dtype=X.dtype, device=X.device)[:, None]
    if kind == "localist":
        Sx = g.choice(N, P_size, replace=False); Xp[:, Sx] += amp * 0.45 * st * sd[Sx]
    elif kind == "redundant":
        Sx = g.choice(N, min(2000, N // 2), replace=False); Xp[:, Sx] += amp * 0.12 * st * sd[Sx]
    elif kind == "dense":
        for start in range(0, N, 4096):
            Xp[:, start:start + 4096] += amp * 0.03 * st * sd[start:start + 4096]
    elif kind.startswith("thr"):
        k = int(kind[3:]) - 1
        C_ = torch.as_tensor(g.standard_normal((N, k)), dtype=X.dtype, device=X.device)
        A_ = torch.as_tensor(g.standard_normal((n, k)) * 3.0, dtype=X.dtype, device=X.device)
        for start in range(0, N, 4096):
            Xp[:, start:start + 4096] += (amp * 0.08 * st + A_ @ C_[start:start + 4096].T) * sd[start:start + 4096]
    elif kind == "xor3":
        Sx = g.choice(N, min(60, N // 3), replace=False); m = len(Sx) // 3
        gg = g.standard_normal((n, 3))
        for j in range(3):
            Xp[:, Sx[j * m:(j + 1) * m]] += amp * 0.8 * torch.as_tensor(gg[:, [j]], dtype=X.dtype, device=X.device) * sd[Sx[j * m:(j + 1) * m]]
        return Xp, (np.sign(gg).prod(1) > 0).astype(int)
    return Xp, None

def l1_cal(ctx):
    """L1-C: planted labels in the real S_fit u S_L CETT matrix; SNR by a frozen 6-step bisection on a cheap I_ref proxy (5 fragments at
    the 3 largest sizes) matched to the real I_ref within +-20 %; PIP at 11 sizes x 20 fragments, floors at 3 sizes x 50 permutations;
    reselection (selector as H_1) for localist and redundant; the L1 tree applied to each (r1b_l_rules.rule_L1)."""
    torch = K._torch()
    tok, model, cf, H, items = setup(ctx)
    pop = load_pop(ctx)
    pip = loadpt(_path(ctx, "l/l1_pip.pt"))
    Iref = max(float(np.median(S.ihat(pip["fam"][f"U{m}"]).numpy())) for m in pip["grid"])
    d = _cett(ctx)
    sp, cl = np.array(d["split"]), np.array(d["cls"])
    rows = np.flatnonzero(np.isin(sp, ["S_fit", "S_L"]) & np.isin(cl, ["CC", "CI"]))
    X0 = d["cmean"][rows].float().to(dev(model))
    n, N = X0.shape
    folds = torch.tensor(S.folds_by_question(n, 5, 1))
    grid = [m for m in cf["pip_grid"] if m <= N]
    big = grid[-3:]
    out = {}
    for prof in PROFILES:
        for sd_ in range(cf["l1c_seeds"]):
            def run():
                g = rng("r1bL1C", prof, sd_)
                y = (g.random(n) < 0.5).astype(int)
                s = 2 * y - 1.0

                def proxy(amp):
                    gg = rng("r1bL1C", prof, sd_, "plant")
                    Xp, yx = plant(X0, prof, s, amp, gg, len(pop["P"]))
                    yy = torch.tensor(yx if yx is not None else y)
                    fr = [f for m in big for f in _uniform(rng("proxy", m), N, m, 5)]
                    ls = frag_losses(ctx, f"l1c_{prof}_{sd_}_px{amp:.4g}", Xp, yy, folds, fr)
                    return float(np.median(S.ihat(ls).numpy()))
                lo_, hi_ = math.log(0.05), math.log(20.0)
                for _ in range(6):
                    mid = (lo_ + hi_) / 2
                    if proxy(math.exp(mid)) < Iref:
                        lo_ = mid
                    else:
                        hi_ = mid
                amp = math.exp((lo_ + hi_) / 2)
                gg = rng("r1bL1C", prof, sd_, "plant")
                Xp, yx = plant(X0, prof, s, amp, gg, len(pop["P"]))
                yy = torch.tensor(yx if yx is not None else y)
                g2 = rng("r1bL1C", prof, sd_, "frag")
                U = {m: frag_losses(ctx, f"l1c_{prof}_{sd_}_U{m}", Xp, yy, folds, _uniform(g2, N, m, cf["l1c_frag"])) for m in grid}
                fl = {}
                for m in [grid[0], grid[len(grid) // 2], grid[-1]]:
                    py = torch.stack([yy[torch.as_tensor(rng("l1c", prof, sd_, m, p).permutation(n))] for p in range(cf["l1c_perm"])])
                    fl[m] = frag_losses(ctx, f"l1c_{prof}_{sd_}_F{m}", Xp, py, folds, _uniform(g2, N, m, cf["l1c_perm"]))
                fit = torch.as_tensor(np.arange(n) % 2 == 0)
                w, _ = selector(Xp[fit], yy[fit].float().to(Xp.device), H["C_star"])
                H1 = top_set(w, len(pop["P"]))
                rs = [H1]
                if prof in ("localist", "redundant"):
                    excl = torch.zeros(N, dtype=torch.bool, device=Xp.device)
                    for r in range(2, 4):
                        h_, _ = hull(Xp[fit], rs[-1]); excl[h_] = True
                        w, _ = selector(Xp[fit], yy[fit].float().to(Xp.device), H["C_star"], excl)
                        rs.append(top_set(w * (~excl), len(pop["P"])))
                M = frag_losses(ctx, f"l1c_{prof}_{sd_}_M", Xp, yy, folds, [s_ for s_ in rs if s_] + pop["CTRL"])
                syn = {}
                for m in [s_ for s_ in cf["syn_grid"] if 2 * s_ <= N]:
                    pairs = [g2.choice(N, 2 * m, replace=False) for _ in range(cf["l1c_frag"])]
                    A = frag_losses(ctx, f"l1c_{prof}_{sd_}_sA{m}", Xp, yy, folds, [sorted(p[:m].tolist()) for p in pairs])
                    B = frag_losses(ctx, f"l1c_{prof}_{sd_}_sB{m}", Xp, yy, folds, [sorted(p[m:].tolist()) for p in pairs])
                    AB = frag_losses(ctx, f"l1c_{prof}_{sd_}_sAB{m}", Xp, yy, folds, [sorted(p.tolist()) for p in pairs])
                    syn[m] = (S.ihat(A) + S.ihat(B) - S.ihat(AB)).numpy()
                return dict(amp=amp, U={m: S.ihat(v).numpy() for m, v in U.items()}, Urows={m: v for m, v in U.items()},
                            floor={m: np.quantile(S.ihat(v).numpy(), 0.95) for m, v in fl.items()},
                            M=S.ihat(M).numpy(), n_resel=len([s_ for s_ in rs if s_]), I3=syn)
            out[(prof, sd_)] = ctx.cached(f"l1c_{prof}_{sd_}_done", run)
    save(ctx, "l/l1_cal.pt", out, config=dict(profiles=PROFILES, seeds=cf["l1c_seeds"], frag=cf["l1c_frag"], perm=cf["l1c_perm"], Iref=Iref))

def _med_boot(loss, W):
    """Median over fragments of I-hat per bootstrap replicate: loss [F, n], W [B, n] multiplicities -> [B]."""
    m = 1.0 - (W @ loss.T.numpy()) / W.sum(1, keepdims=True)
    return np.median(m, 1)

def l1_stats(ctx):
    """All L1 statistics with item-bootstrap intervals (2,000 replicates over S_L rows), the planted-code calibration results and the
    erasure contrasts (two-level bootstrap over items and the 8 sets) -> l/l1.json."""
    import r1b_l_rules as RU
    tok, model, cf, H, items = setup(ctx)
    pop = load_pop(ctx)
    pip = loadpt(_path(ctx, "l/l1_pip.pt")); rot = loadpt(_path(ctx, "l/l1_rot.pt")); rs = loadpt(_path(ctx, "l/l1_resel.pt"))
    l1c = loadpt(_path(ctx, "l/l1_cal.pt"))
    n = pip["fam"][f"U{pip['grid'][0]}"].shape[1]
    W = S.boot_counts(n, cf["nboot"], seed=101)
    grid, mP = pip["grid"], pip["mP"]
    st = dict(grid=grid, mP=mP)
    medU = {m: float(np.median(S.ihat(pip["fam"][f"U{m}"]).numpy())) for m in grid}
    bU = {m: _med_boot(pip["fam"][f"U{m}"], W) for m in grid}
    fl = np.array([float(x) for x in S.ihat(pip["floor"]).numpy()]); fm = np.array(pip["floor_m"])
    F = {m: float(np.quantile(fl[fm == m], 0.95)) for m in grid}
    Iref = max(medU[m] for m in grid if m <= 1024)
    st["U"] = {m: S.summ(medU[m], bU[m]) for m in grid}; st["F"] = F; st["I_ref"] = Iref
    IP = S.ihat(pip["M"][0]).item(); IM = S.ihat(pip["M"][1:]).numpy()
    bP = 1.0 - (W @ pip["M"][0].numpy()) / W.sum(1)
    st["Ihat_P"] = S.summ(IP, bP); st["M_rank_p"] = K.p_rank(IP, IM)
    Priv = st["M_rank_p"] <= 0.01 + 1e-12 and float(np.percentile(bP, 5)) > F.get(mP, max(F.values()))
    st["Priv"] = bool(Priv)

    def m_delta(delta, med):
        ms = sorted(med); v = [med[m] for m in ms]
        for i, m in enumerate(ms):
            if v[i] >= delta * Iref:
                if i == 0:
                    return float(m)
                a, b = v[i - 1], v[i]
                t = (delta * Iref - a) / (b - a) if b != a else 1.0
                return float(math.exp(math.log(ms[i - 1]) + t * (math.log(m) - math.log(ms[i - 1]))))
        return None
    mU = m_delta(0.5, medU)
    medXH = {m: float(np.median(S.ihat(pip["fam"][f"XH{m}"]).numpy())) for m in grid if len(pip["fam"][f"XH{m}"])}
    mXH = m_delta(0.5, medXH)
    st["m05_U"], st["m05_XH"] = mU, mXH
    st["xh_ratio"] = (mXH / mU) if (mU and mXH) else ((max(grid) / mU) if mU else None)
    st["xh_censored"] = mXH is None
    # reselection
    I_H = S.ihat(rs["loss"]).numpy()
    bH = 1.0 - (W @ rs["loss"].numpy().T) / W.sum(1, keepdims=True)
    if Priv and len(I_H) > 1:
        st["rho2"] = S.summ(I_H[1] / I_H[0] if I_H[0] > 0 else float("nan"), bH[:, 1] / np.where(bH[:, 0] > 0, bH[:, 0], np.nan))
    else:
        st["rho2"] = None
    st["R_rep"] = 1 + int(sum(I_H[r] >= 0.8 * I_H[0] for r in range(1, len(I_H))))
    st["Ihat_H"] = I_H.tolist()
    # m_0, falsification flag
    m0 = None
    for m in grid:
        if float(np.percentile(bU[m], 5)) > F[m]:
            m0 = m; break
    st["m0"] = m0
    st["falsified"] = any(medU[m] >= 0.1 * Iref and float(np.percentile(bU[m], 5)) > F[m] for m in grid if m <= 1024)
    # synergy / monogamy against the planted dense and localist codes (L1-C)
    sm = np.array(pip["syn"]["m"])
    I3 = (S.ihat(pip["syn"]["A"]) + S.ihat(pip["syn"]["B"]) - S.ihat(pip["syn"]["AB"])).numpy()
    bI3 = {}
    for m in np.unique(sm):
        k = sm == m
        a = 1 - (W @ pip["syn"]["A"][k].numpy().T) / W.sum(1, keepdims=True)
        b = 1 - (W @ pip["syn"]["B"][k].numpy().T) / W.sum(1, keepdims=True)
        ab = 1 - (W @ pip["syn"]["AB"][k].numpy().T) / W.sum(1, keepdims=True)
        bI3[int(m)] = np.median(a + b - ab, 1)
    cal_ok, thr_ok, prof_v = {}, {}, {}
    for (prof, sd_), r in l1c.items():
        medp = {m: float(np.median(v)) for m, v in r["U"].items()}
        Irp = max(medp.values())
        Fp = r["floor"]
        m0p = next((m for m in sorted(medp) if medp[m] > Fp.get(m, max(Fp.values()))), None)
        Ip = float(r["M"][0]); Priv_p = K.p_rank(Ip, r["M"][r["n_resel"]:]) <= 0.01 + 1e-12 and Ip > Fp.get(max(Fp), 0)
        rho = (float(r["M"][1]) / Ip) if (r["n_resel"] > 1 and Ip > 0) else None
        rr = 1 + sum(float(r["M"][j]) >= 0.8 * Ip for j in range(1, r["n_resel"]))
        stp = dict(Priv=Priv_p, rho2=None if rho is None else dict(pt=rho, lo=rho, hi=rho), R_rep=rr, xh_ratio=4 if Priv_p else 1,
                   Ihat_P=dict(pt=Ip), I_ref=dict(pt=Irp), medU1024=dict(pt=medp[max(medp)]), m0=m0p,
                   syn=prof.startswith("thr") and all(np.median(v) < 0 for v in r["I3"].values()), mono=True, red=False)
        v = RU.rule_L1(stp)["verdict"]
        exp = {"localist": "localised: the H-cluster is the store", "redundant": "H privileged but one of several representations (redundant)",
               "dense": "dense (no sparse store)", "xor3": "unclassified (profile reported)"}.get(prof, "holographic-like (threshold <= m_cal, synergistic, monogamous)")
        prof_v.setdefault(prof, []).append(v)
        cal_ok.setdefault(prof, []).append(v == exp)
    st["l1c"] = dict(verdicts=prof_v, ok={p: bool(all(v)) for p, v in cal_ok.items()})
    m_cal = max([int(p[3:]) for p in ("thr16", "thr64") if st["l1c"]["ok"].get(p)] or [0]) or None
    st["m_cal"] = m_cal
    dense_I3 = {m: float(np.median(np.concatenate([r["I3"].get(m, np.array([np.nan])) for (p, s_), r in l1c.items() if p == "dense"])))
                for m in np.unique(sm).tolist()}
    loc_I3 = {m: float(np.median(np.concatenate([r["I3"].get(m, np.array([np.nan])) for (p, s_), r in l1c.items() if p == "localist"])))
              for m in np.unique(sm).tolist()}
    st["I3"] = {int(m): S.summ(float(np.median(I3[sm == m])), bI3[int(m)]) for m in np.unique(sm)}
    st["I3_c"] = {int(m): S.summ(float(np.median(I3[sm == m])) - loc_I3[int(m)], bI3[int(m)] - loc_I3[int(m)]) for m in np.unique(sm)}
    st["Sigma_minus_dense"] = {int(m): S.summ(-float(np.median(I3[sm == m])) + dense_I3[int(m)], -bI3[int(m)] + dense_I3[int(m)])
                               for m in np.unique(sm)}
    if m_cal is None:
        st["syn"] = None
    else:
        st["syn"] = any(m <= m_cal and RU.LB(st["Sigma_minus_dense"][m]) and all(medU.get(mm, -1) <= F.get(mm, 0) for mm in grid if mm <= m)
                        for m in st["Sigma_minus_dense"])
    nI3 = len(st["I3_c"])
    st["mono"] = sum(RU.UB(v) for v in st["I3_c"].values()) >= min(3, nI3)
    st["red"] = sum(RU.LB(v) for v in st["I3_c"].values()) >= min(3, nI3)
    # latent scalars
    med1024 = medU[max(m for m in grid if m <= 1024)]
    st["medU1024"] = S.summ(med1024, bU[max(m for m in grid if m <= 1024)])
    st["Lambda"] = {k: float(S.ihat(pip["latent"][k]).item() / med1024) if med1024 > 0 else None for k in ("entropy", "pc1")}
    lam_max = max([v for v in st["Lambda"].values() if v is not None] or [0])
    # rotated basis
    rd = {}
    for i, m in enumerate(rot["grid"]):
        if m in (mP, 64, 1024) or (ctx.mock and m in (mP, rot["grid"][-1])):
            if f"U{m}" in pip["fam"]:
                rd[m] = S.summ(float(np.median(S.ihat(rot["Rot"][i]).numpy())) - medU[m], _med_boot(rot["Rot"][i], W) - bU[m], margin=0.03)
    st["rot_minus_U"] = rd
    st["basis_np"] = bool(rd) and all(v["inside"] for v in rd.values())
    # pre-answer
    pre = pip["pre"]
    st["pre"] = {m: float(np.median(S.ihat(pre["U"][i]).numpy())) for i, m in enumerate(pre["grid"])}
    st["pre_floor"] = {m: float(np.quantile(S.ihat(pre["floor"][i]).numpy(), 0.95)) for i, m in enumerate(pre["floor_grid"])}
    # selection robustness on the boot sets
    if len(rs["boot_loss"]) >= 10:
        Ib = S.ihat(rs["boot_loss"]).numpy()
        st["boot_priv_n"] = int(sum(K.p_rank(x, IM) <= 0.01 + 1e-12 and x > F.get(mP, 0) for x in Ib))
        st["sel_robust"] = st["boot_priv_n"] >= 16 if Priv else None
    else:
        st["sel_robust"] = None
    st["Lambda_max"] = lam_max
    flags = dict(Priv=Priv, rho2=st["rho2"], R_rep=st["R_rep"], xh_ratio=st["xh_ratio"], Ihat_P=st["Ihat_P"], I_ref=dict(pt=Iref),
                 medU1024=st["medU1024"], m0=m0, syn=st["syn"], mono=st["mono"], red=st["red"], Lambda=lam_max, basis_np=st["basis_np"],
                 sel_robust=st["sel_robust"], falsified=st["falsified"])
    st["tree_input"] = flags
    st["cal_ok"] = bool(st["l1c"]["ok"].get("localist") and st["l1c"]["ok"].get("redundant") and st["l1c"]["ok"].get("dense"))
    st["erasure_artifact"] = "l_measure/l1_erasure.json"
    wj(ctx, "l/l1.json", st, config=dict(nboot=cf["nboot"], grid=grid, P_name=pop["P_name"]))
    return st


"""Attack points in the style of Biggio et al. (arXiv 1708.06131) on residual states after block 21.
Surrogate = hard-margin linear SVM between two prompt families (trained on split S1). The attacker moves a held-out source
state x0 of family A so that the surrogate reads family B:  maximise  s(x) + lambda * p(x | B)   by normalised gradient steps,
p = Gaussian kernel density over the surrogate's B states (the 'mimicry' component). lambda is given relative to the gradient
sizes at the start (lambda_rel = 1: both terms pull equally hard), lambda = inf is pure mimicry.
Besides the attack paths, control displacements are written. All states are saved for patching into the network.
Note (found in review): with a kernel width of the median pairwise distance and about 30 points in 2048 dimensions the density
term is a pull towards the centroid of B, so 'pure mimicry' here means 'towards the centroid'.
Usage: python mim_attack.py <scratch_root> <out_dir> [t0|t2]"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mim_common import *
from mim_static_helpers import col_auc

root, out = sys.argv[1], sys.argv[2]; tag = sys.argv[3] if len(sys.argv) > 3 else "t0"
lab, text, X, F = load_eval(out); n, d = X.shape; WU, g = load_unembed(root); J = load_lens(root); sae = SAE(f"{root}/sae_dl/k_L21_{tag}_ae.pt")
A_all = np.load(f"{out}/mim_sae_acts_{tag}.npy"); SC = float(np.median(np.linalg.norm(X, axis=1))); Ff = unit(F)
LAMS = [0.0, 1.0, 5.0, float("inf")]; DOSE0 = [0.25, 0.5, 1.5, 2.0]; DOSEI = [0.25, 0.5, 1.0, 1.5, 2.0]; NSRC = 32; STEPS = 50; MAXSTEP = 600

def kde(Xq, P, h):
    D2 = ((Xq[:, None, :] - P[None]) ** 2).sum(-1); K = np.exp(-D2 / (2 * h * h))
    return K.mean(1), (K[:, :, None] * (P[None] - Xq[:, None, :])).mean(1) / (h * h)

V, meta, pairs = [], [], []
for pi, (a, b) in enumerate(PAIRS):
    rng = np.random.default_rng(100 + pi); S1, S2, T = split3(lab, a, b, 0); y1 = (lab[S1] == b).astype(int); y2 = (lab[S2] == b).astype(int)
    sur = LinProbe(X[S1], y1, C=1e4, sc=SC); ind = LinProbe(X[S2], y2, C=1e4, sc=SC); yp = rng.permutation(y1); rnd = LinProbe(X[S1], yp, C=1e4, sc=SC)
    A1, B1 = S1[y1 == 0], S1[y1 == 1]; muA, muB = X[A1].mean(0), X[B1].mean(0); dm = muB - muA; DAB = float(np.linalg.norm(dm)); wn = float(np.linalg.norm(sur.w)); wh = sur.w / wn
    src_all = T[lab[T] == a]; refB = T[lab[T] == b]; s_star = max(float(np.median(sur.score(X[refB]))), 0.1); levels = {"lvl0": 0.0, "lvlstar": s_star, "lvlone": 1.0}
    src = np.array([i for i in src_all if sur.score(X[i]) < 0])[:NSRC]                           # only sources the surrogate reads as A
    P = X[B1]; h = float(np.median(np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))[np.triu_indices(len(P), 1)])); x0 = X[src]; s0 = sur.score(x0)
    gp0 = float(np.median(np.linalg.norm(kde(x0, P, h)[1], axis=1))); t0 = (s_star - s0) / wn; eta = DAB / STEPS
    dl = unit(J.T @ (g * (Ff[B1].mean(0) - Ff[A1].mean(0))))            # lens gradient direction (steepest ascent under the linear lens, not a pre-image)
    Atr = A_all[S1].astype(np.float64); on = (Atr > 0).sum(0) >= 3; au = np.where(on, col_auc(Atr, y1), .5); order = np.argsort(-np.abs(au - .5)); aB = A_all[B1].mean(0).astype(np.float64)
    def add(k, name, x, valid=True, **kw):
        V.append(x.astype(np.float32)); meta.append(dict(pair=pi, src=int(src[k]), name=name, valid=bool(valid), dist=float(np.linalg.norm(x - x0[k])), s_sur=float(sur.score(x)), s_ind=float(ind.score(x)), **kw))
    # ---- gradient paths
    for lam in LAMS:
        x = x0.copy(); got = {(k, L): None for k in range(len(src)) for L in levels}; dose = {(k, f): None for k in range(len(src)) for f in (DOSEI if np.isinf(lam) else [])}; alive = np.ones(len(src), bool); p_prev = np.full(len(src), -1.0)
        for step in range(MAXSTEP):
            p, gp = kde(x, P, h)
            if np.isinf(lam): u = unit(gp); alive &= p > p_prev - 1e-15; p_prev = np.maximum(p, p_prev)   # pure mimicry stops when the density no longer rises
            else: u = unit(wh[None] + lam * gp / gp0)
            xn = x + eta * u * alive[:, None]; sp, sn = sur.score(x), sur.score(xn); dp, dn = np.linalg.norm(x - x0, axis=1), np.linalg.norm(xn - x0, axis=1)
            for k in range(len(src)):
                if not alive[k]: continue
                for L, lv in levels.items():
                    if got[(k, L)] is None and sp[k] < lv <= sn[k]: got[(k, L)] = x[k] + (lv - sp[k]) / (sn[k] - sp[k]) * (xn[k] - x[k])
                for f in (DOSEI if np.isinf(lam) else []):
                    if dose[(k, f)] is None and dp[k] < f * t0[k] <= dn[k]: dose[(k, f)] = x[k] + (f * t0[k] - dp[k]) / (dn[k] - dp[k] + 1e-12) * (xn[k] - x[k])
            x = xn
            if all(v is not None for v in got.values()) and all(v is not None for v in dose.values()): break
            if not alive.any(): break
        ln = "inf" if np.isinf(lam) else f"{lam:g}"
        for k in range(len(src)):
            for L in levels: add(k, f"lam{ln}_{L}", got[(k, L)] if got[(k, L)] is not None else x[k], valid=got[(k, L)] is not None, lam=ln, level=L)
            for f in (DOSEI if np.isinf(lam) else []): add(k, f"doseinf_{f:g}", dose[(k, f)] if dose[(k, f)] is not None else x[k], valid=dose[(k, f)] is not None, lam="inf", dose=f)
            if np.isinf(lam): add(k, "laminf_end", x[k], lam="inf")                              # where pure mimicry converges
    # ---- straight-line controls
    for k in range(len(src)):
        for f in DOSE0: add(k, f"dose0_{f:g}", x0[k] + f * t0[k] * wh, lam="0", dose=f)
        for L in ("lvl0", "lvlstar"): add(k, f"dm_{L}", x0[k] + (levels[L] - s0[k]) / float(sur.w @ dm) * dm, valid=float(sur.w @ dm) > 0, level=L)
        for q in (1, 2): add(k, f"rand{q}", x0[k] + t0[k] * unit(rng.standard_normal(d)))
        add(k, "randlabel", x0[k] + t0[k] * unit(rnd.w)); add(k, "lens_t0", x0[k] + t0[k] * dl)
        c = (s_star - s0[k]) / float(sur.w @ dl) if float(sur.w @ dl) > 0 else np.inf; ok = c <= 5 * t0[k]; add(k, "lens_lvlstar", x0[k] + (c if ok else 5 * t0[k]) * dl, valid=ok, level="lvlstar")
        sw = rng.choice(refB, 2, replace=False); add(k, "swap1", X[sw[0]], swap=int(sw[0])); add(k, "swap2", X[sw[1]], swap=int(sw[1])); add(k, "protoB", muB)
        a0 = sae.encode(x0[k])[0].astype(np.float64); add(k, "saerec", sae.decode(a0[None].astype(np.float32))[0].astype(np.float64))
        for kk in (5, 20):                                                                    # at s* with the cap c <= 3 these rarely exist; mim_attack2.py has the uncapped and equal-length versions
            j = order[:kk]; v = (sae.Wd[:, j].astype(np.float64) * (aB[j] - a0[j])).sum(1); c = (s_star - s0[k]) / float(sur.w @ v) if float(sur.w @ v) > 0 else np.inf; ok = c <= 3
            add(k, f"sae{kk}", x0[k] + (c if ok else 3.0) * v, valid=ok, level="lvlstar", coef=float(min(c, 3.0)))
            if kk == 20: add(k, "sae20_c1", x0[k] + v, coef=1.0)
        add(k, "orig", x0[k])
    pairs.append(dict(pair=pi, a=a, b=b, n_src=len(src), n_src_all=len(src_all), n_refB=len(refB), S1=S1.tolist(), S2=S2.tolist(), T=T.tolist(), src=src.tolist(), refB=refB.tolist(), s_star=s_star, DAB=DAB, h=h, gp0=gp0, wnorm=wn,
                      margin=2 / wn, t0_median=float(np.median(t0)), cos_w_dm=float(wh @ unit(dm)), cos_w_lens=float(wh @ dl), cos_dm_lens=float(unit(dm) @ dl), cos_w_rnd=float(wh @ unit(rnd.w)),
                      sur_test_acc=float(((sur.score(X[T]) > 0) == (lab[T] == b)).mean()), ind_test_acc=float(((ind.score(X[T]) > 0) == (lab[T] == b)).mean()), sae_feats=[int(q) for q in order[:20]], sae_feat_auc=[float(au[q]) for q in order[:20]],
                      sur_w=sur.w.tolist(), sur_b=sur.b, ind_w=ind.w.tolist(), ind_b=ind.b))
    print(f"{a} -> {b}: {len(src)} sources (of {len(src_all)}), surrogate test acc {pairs[-1]['sur_test_acc']:.2f}, independent probe {pairs[-1]['ind_test_acc']:.2f}, s* {s_star:.2f}, centroid distance {DAB:.1f} ({DAB / SC:.2f} of the median norm), "
          f"margin width {2 / wn:.1f}, bandwidth {h:.1f}, median straight distance to s* {np.median(t0):.1f}, cos(w, mean diff) {pairs[-1]['cos_w_dm']:.2f}", flush=True)
V = np.stack(V); np.save(f"{out}/mim_variants_{tag}.npy", V); json.dump(dict(meta=meta, pairs=pairs, SC=SC), open(f"{out}/mim_variants_{tag}.json", "w"))
import collections
print("variants:", V.shape, "| invalid by name:", dict(collections.Counter(m["name"] for m in meta if not m["valid"])))

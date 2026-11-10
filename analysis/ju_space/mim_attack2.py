"""Second set of states (added after review): displacements of EQUAL LENGTH in different directions, and reference states.
Same pairs, splits, probes and sources as mim_attack.py (everything there is deterministic), written under tag <tag>b.
Lengths are multiples f of t0 = the straight distance along the probe normal at which the surrogate reads 'typical unseen B'.
  dose0_f     probe normal                         dosedm_f   class-mean difference (keeps what is individual about the prompt)
  dosecen_f   straight towards the centroid of B   dosenn_f   straight towards the nearest real B state (mimicry of one real sample)
  doselens_f  lens gradient direction              dosesae_f  the 20 most separating SAE units moved towards their B mean
  rand_f, randlabel_f   random direction, normal of an SVM trained on shuffled labels
References: swapnn (nearest real B state), protoA (centroid of the prompt's own family), protoC (centroid of an unrelated family),
protoAll (mean of all prompts), sae20_star (SAE-unit displacement scaled until the surrogate reads s*, no cap).
Double dissociation: wperp_* (probe normal with its class-mean component removed: the probe reads B, no class-mean shift) and
dmblind_* (class-mean difference with its probe-normal component removed: full class-mean shift, probe score unchanged); dmfull_* plain.
Usage: python mim_attack2.py <scratch_root> <out_dir> [t0|t2]"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mim_common import *
from mim_static_helpers import col_auc

root, out = sys.argv[1], sys.argv[2]; tag = sys.argv[3] if len(sys.argv) > 3 else "t0"
lab, text, X, F = load_eval(out); n, d = X.shape; WU, g = load_unembed(root); J = load_lens(root); sae = SAE(f"{root}/sae_dl/k_L21_{tag}_ae.pt")
A_all = np.load(f"{out}/mim_sae_acts_{tag}.npy"); SC = float(np.median(np.linalg.norm(X, axis=1))); Ff = unit(F); old = json.load(open(f"{out}/mim_variants_{tag}.json"))
NSRC = 32; FS = [0.5, 1.0, 1.5, 2.0, 2.5]; OTHER = "nineteenth_century"; muAll = X.mean(0); muC = X[lab == OTHER].mean(0)
V, meta = [], []
for pi, (a, b) in enumerate(PAIRS):
    rng = np.random.default_rng(100 + pi); S1, S2, T = split3(lab, a, b, 0); y1 = (lab[S1] == b).astype(int); y2 = (lab[S2] == b).astype(int)
    sur = LinProbe(X[S1], y1, C=1e4, sc=SC); ind = LinProbe(X[S2], y2, C=1e4, sc=SC); yp = rng.permutation(y1); rnd = LinProbe(X[S1], yp, C=1e4, sc=SC)
    A1, B1 = S1[y1 == 0], S1[y1 == 1]; muA, muB = X[A1].mean(0), X[B1].mean(0); dm = muB - muA; wn = float(np.linalg.norm(sur.w)); wh = sur.w / wn
    src_all = T[lab[T] == a]; refB = T[lab[T] == b]; s_star = max(float(np.median(sur.score(X[refB]))), 0.1)
    src = np.array([i for i in src_all if sur.score(X[i]) < 0])[:NSRC]; P0 = old["pairs"][pi]
    assert src.tolist() == P0["src"] and abs(s_star - P0["s_star"]) < 1e-9 and np.allclose(sur.w, P0["sur_w"]), "setup differs from mim_attack.py"
    x0 = X[src]; s0 = sur.score(x0); t0 = (s_star - s0) / wn; dl = unit(J.T @ (g * (Ff[B1].mean(0) - Ff[A1].mean(0)))); r2 = np.random.default_rng(200 + pi)
    Atr = A_all[S1].astype(np.float64); on = (Atr > 0).sum(0) >= 3; au = np.where(on, col_auc(Atr, y1), .5); order = np.argsort(-np.abs(au - .5)); aB = A_all[B1].mean(0).astype(np.float64)
    dh = unit(dm); wperp = unit(wh - (wh @ dh) * dh); dmblind = dm - (dm @ wh) * wh; cfull = float(dm @ dm) / float(dmblind @ dm)   # cfull * dmblind has class-mean coordinate 1
    def add(k, name, x, valid=True, **kw):
        V.append(x.astype(np.float32)); meta.append(dict(pair=pi, src=int(src[k]), name=name, valid=bool(valid), dist=float(np.linalg.norm(x - x0[k])), s_sur=float(sur.score(x)), s_ind=float(ind.score(x)), **kw))
    for k in range(len(src)):
        dnn = np.linalg.norm(X[B1] - x0[k], axis=1); inn = int(B1[np.argmin(dnn)]); xnn = X[inn]; dcen = float(np.linalg.norm(muB - x0[k]))
        a0 = sae.encode(x0[k])[0].astype(np.float64); j = order[:20]; v = (sae.Wd[:, j].astype(np.float64) * (aB[j] - a0[j])).sum(1); u1, u2 = unit(r2.standard_normal(d)), unit(r2.standard_normal(d))
        for f in FS:
            L = f * t0[k]
            add(k, f"dosedm_{f:g}", x0[k] + L * unit(dm), dose=f); add(k, f"dosecen_{f:g}", x0[k] + L * unit(muB - x0[k]), valid=L <= dcen, dose=f)
            add(k, f"dosenn_{f:g}", x0[k] + L * unit(xnn - x0[k]), valid=L <= dnn.min(), dose=f); add(k, f"dosesae_{f:g}", x0[k] + L * unit(v), dose=f)
            if f in (0.5, 2.0, 2.5): add(k, f"doselens_{f:g}", x0[k] + L * dl, dose=f)
            if f >= 2.0: add(k, f"rand1_{f:g}", x0[k] + L * u1, dose=f); add(k, f"rand2_{f:g}", x0[k] + L * u2, dose=f); add(k, f"randlabel_{f:g}", x0[k] + L * unit(rnd.w), dose=f)
            if f == 2.5: add(k, "dose0_2.5", x0[k] + L * wh, dose=f)
        add(k, "dose0_3", x0[k] + 3 * t0[k] * wh, dose=3.0); add(k, "rand1_3", x0[k] + 3 * t0[k] * u1, dose=3.0); add(k, "rand2_3", x0[k] + 3 * t0[k] * u2, dose=3.0)
        add(k, "swapnn", xnn, swap=inn); add(k, "protoA", muA); add(k, "protoC", muC); add(k, "protoAll", muAll)
        # double dissociation (suggested in review): a move the probe sees but that carries no class-mean shift, and a class-mean shift the probe cannot see
        add(k, "wperp_star", x0[k] + (s_star - s0[k]) / float(sur.w @ wperp) * wperp, level="lvlstar")       # surrogate reaches s*, class-mean coordinate stays 0
        for f in (1.0, 2.0): add(k, f"wperp_{f:g}", x0[k] + f * t0[k] * wperp, dose=f)
        for nm, c in (("dmblind_0.5", 0.5 * cfull), ("dmblind_1", cfull), ("dmblind_1.5", 1.5 * cfull)): add(k, nm, x0[k] + c * dmblind)   # surrogate score unchanged, class-mean coordinate 0.5 / 1 / 1.5
        for c in (0.5, 1.0, 1.5): add(k, f"dmfull_{c:g}", x0[k] + c * dm)                                     # plain class-mean translation, 0.5 / 1 / 1.5 of the mean difference
        c = (s_star - s0[k]) / float(sur.w @ v) if float(sur.w @ v) > 0 else np.inf; add(k, "sae20_star", x0[k] + (c if np.isfinite(c) else 0.0) * v, valid=bool(np.isfinite(c)), coef=float(c) if np.isfinite(c) else None)
    print(f"{a} -> {b}: {len(src)} sources", flush=True)
V = np.stack(V); np.save(f"{out}/mim_variants_{tag}b.npy", V); json.dump(dict(meta=meta, other_family=OTHER), open(f"{out}/mim_variants_{tag}b.json", "w"))
import collections
print("additional variants:", V.shape, "| invalid by name:", dict(collections.Counter(m["name"] for m in meta if not m["valid"])))
co = [m["coef"] for m in meta if m["name"] == "sae20_star" and m["valid"]]; print(f"sae20_star: reachable for {len(co)} of {sum(m['name'] == 'sae20_star' for m in meta)} sources; scale needed: median {np.median(co):.1f}, 10-90% {np.percentile(co, 10):.1f}-{np.percentile(co, 90):.1f} (1 = the units sit at their B mean)")

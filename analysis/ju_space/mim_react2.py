"""Reaction analysis, second version (after review). What changed against mim_react.py:
* comparisons on common support (same pairs, same sources in every row of a table), with counts;
* three matchings side by side: equal probe score, equal length, equal class-mean coordinate;
* the output measure (first response token) is used only for pairs where it separates real held-out prompts (AUROC >= 0.85),
  and is also given as a calibrated rate: share of states whose output score exceeds the 95th percentile of real held-out A prompts;
* the final-state measure is shown together with its 'direct' part (the patched vector riding the residual stream, later blocks frozen);
* 'mimicry' with a wide kernel is named for what it is (a pull to the centroid of B); typicality is a two-sided Mahalanobis check;
* every SAE reading has the same rule on the raw state next to it; uncertainty is given over pairs.
Usage: python mim_react2.py <scratch_root> <out_dir> [t0|t2]"""
import json, os, re, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mim_common import *

root, out = sys.argv[1], sys.argv[2]; tag = sys.argv[3] if len(sys.argv) > 3 else "t0"
lab, text, X, F = load_eval(out); n, d = X.shape; WU, g = load_unembed(root); sae = SAE(f"{root}/sae_dl/k_L21_{tag}_ae.pt"); beh = sorted(set(lab)); rng = np.random.default_rng(0)
mA = json.load(open(f"{out}/mim_variants_{tag}.json")); mB = json.load(open(f"{out}/mim_variants_{tag}b.json")); pairs = mA["pairs"]; meta = mA["meta"] + mB["meta"]
V = np.concatenate([np.load(f"{out}/mim_variants_{tag}.npy"), np.load(f"{out}/mim_variants_{tag}b.npy")]).astype(np.float64)
Fp = np.concatenate([np.load(f"{out}/mim_final_{tag}.npy"), np.load(f"{out}/mim_final_{tag}b.npy")]).astype(np.float64); Hpre = np.load(f"{out}/mim_prenorm_{tag}b.npy").astype(np.float64)
A_all = np.load(f"{out}/mim_sae_acts_{tag}.npy"); assert len(meta) == len(V) == len(Fp)
srcs = sorted({m["src"] for m in meta}); print(f"{len(meta)} patched states, {len(srcs)} source prompts | check of the stored pre-norm state: max |norm(pre) - final| = {np.abs(rmsnorm(Hpre[srcs], g) - F[srcs]).max():.2e}")
def logp_of(Fm, chunk=128):
    for i in range(0, len(Fm), chunk): yield i, log_softmax((Fm[i:i + chunk].astype(np.float32) @ WU.T).astype(np.float64))
def canon(name):                                         # canonical 'direction@length' for the equal-length grid
    t = {"lam0_lvlstar": "probe@1", "lens_t0": "lensgrad@1", "rand1": "random@1", "rand2": "random@1", "randlabel": "randlabel@1"}
    if name in t: return t[name]
    for pat, dr in ((r"dose0_(.+)", "probe"), (r"dosedm_(.+)", "classmean"), (r"dosecen_(.+)", "centroid"), (r"dosenn_(.+)", "nearestB"), (r"doselens_(.+)", "lensgrad"), (r"dosesae_(.+)", "sae20"),
                    (r"wperp_([\d.]+)", "probeonly"), (r"rand[12]_(.+)", "random"), (r"randlabel_(.+)", "randlabel")):
        q = re.fullmatch(pat, name)
        if q: return f"{dr}@{float(q.group(1)):g}"
    return {"swap1": "swap", "swap2": "swap"}.get(name, name)
rows, refs = [], []
for P_ in pairs:
    pi, a, b = P_["pair"], P_["a"], P_["b"]; S1, S2, T = map(np.array, (P_["S1"], P_["S2"], P_["T"])); tr = np.concatenate([S1, S2]); refB = np.array(P_["refB"]); teA = T[lab[T] == a]
    idx = np.array([i for i, m in enumerate(meta) if m["pair"] == pi]); src = np.array([meta[i]["src"] for i in idx]); Vi = V[idx]; D = Vi - X[src]
    dm = X[S1][lab[S1] == b].mean(0) - X[S1][lab[S1] == a].mean(0); cm = D @ dm / (dm @ dm)
    # ---- output of real prompts, reaction scales, calibration on real held-out A
    use = np.concatenate([tr, T]); LP = np.concatenate([lp for _, lp in logp_of(F[use])]); pos = {int(q): k for k, q in enumerate(use)}; at = lambda ii: [pos[int(q)] for q in ii]
    QA, QB = np.exp(LP[at(tr[lab[tr] == a])]).mean(0), np.exp(LP[at(tr[lab[tr] == b])]).mean(0); wv = np.log(QB + 1e-6) - np.log(QA + 1e-6); r_real = np.exp(LP) @ wv
    rA, rB = r_real[at(teA)], r_real[at(refB)]; gap = float(rB.mean() - rA.mean()); thr = float(np.percentile(rA, 95)); yT = np.r_[np.zeros(len(teA)), np.ones(len(refB))]; auc_out = auroc(np.r_[rA, rB], yT)
    df = unit(F[tr][lab[tr] == b].mean(0) - F[tr][lab[tr] == a].mean(0)); cA, cB = F[teA] @ df, F[refB] @ df; gap2 = float(cB.mean() - cA.mean()); thr2 = float(np.percentile(cA, 95))
    r_p = np.zeros(len(idx)); tp = np.zeros(len(idx), int); kl = np.zeros(len(idx)); r_d = np.zeros(len(idx)); Fdir = rmsnorm(Hpre[src] + D, g)      # 'direct' final state: later blocks do not react at all
    for i0, lp in logp_of(Fp[idx]):
        sl = slice(i0, i0 + len(lp)); r_p[sl] = np.exp(lp) @ wv; tp[sl] = lp.argmax(1); kl[sl] = [(np.exp(LP[pos[int(s)]]) * (LP[pos[int(s)]] - lp[k])).sum() for k, s in enumerate(src[sl])]
    for i0, lp in logp_of(Fdir): r_d[i0:i0 + len(lp)] = np.exp(lp) @ wv
    r_s = r_real[at(src)]; top_s = np.array([LP[pos[int(s)]].argmax() for s in src])
    # ---- readings of the block-21 state: centroid rule on the raw state and on SAE activations, separating SAE units, typicality
    muA, muB = X[tr][lab[tr] == a].mean(0), X[tr][lab[tr] == b].mean(0); ncB = lambda Z: (((Z - muB) ** 2).sum(1) < ((Z - muA) ** 2).sum(1)).astype(float)
    Av = sae.encode(Vi).astype(np.float64); sA, sB = A_all[tr][lab[tr] == a].mean(0).astype(np.float64), A_all[tr][lab[tr] == b].mean(0).astype(np.float64)
    saeB = lambda Z: ((unit(Z) @ unit(sB)) > (unit(Z) @ unit(sA))).astype(float); fe = np.array(P_["sae_feats"]); fa = np.array(P_["sae_feat_auc"]); fB, fA = fe[fa > .5], fe[fa < .5]
    trn = np.setdiff1d(np.arange(n), T); W = X[trn] - np.stack([X[trn][lab[trn] == q].mean(0) for q in beh])[[beh.index(q) for q in lab[trn]]]          # pooled within-family scatter of all 18 families
    _, sv, Vt = np.linalg.svd(W, full_matrices=False); lam = sv[:100] ** 2 / len(W); tail = float((sv[100:] ** 2).sum() / len(W) / (d - 100)); Pc = Vt[:100]
    def maha(Z): E = Z - muB; pr = E @ Pc.T; return (pr ** 2 / lam).sum(1) + ((E ** 2).sum(1) - (pr ** 2).sum(1)) / tail
    lo, hi = np.percentile(maha(X[refB]), [2.5, 97.5]); atyp = lambda Z: ((maha(Z) < lo) | (maha(Z) > hi)).astype(float)
    sur_w, sur_b, ind_w, ind_b = np.array(P_["sur_w"]), P_["sur_b"], np.array(P_["ind_w"]), P_["ind_b"]
    t0 = {int(meta[i]["src"]): meta[i]["dist"] for i in idx if meta[i]["name"] == "lam0_lvlstar"}; nB, sBv, at_v = ncB(Vi), saeB(Av), atyp(Vi)
    for k, i in enumerate(idx):
        m = meta[i]; rows.append(dict(pair=pi, src=m["src"], name=m["name"], key=canon(m["name"]), valid=m["valid"], rel_dist=m["dist"] / t0[int(m["src"])], cm=cm[k], s_sur=m["s_sur"], sur_B=float(m["s_sur"] > 0), ind_B=float(m["s_ind"] > 0),
                                      nc_B=nB[k], sae_B=sBv[k], onB=float((Av[k, fB] > 0).mean()) if len(fB) else np.nan, onA=float((Av[k, fA] > 0).mean()) if len(fA) else np.nan, atyp=at_v[k],
                                      dR=(r_p[k] - r_s[k]) / gap, out_B=float(r_p[k] > thr), dR_dir=(r_d[k] - r_s[k]) / gap, dR2=float((Fp[i] - F[m["src"]]) @ df) / gap2, fin_B=float(Fp[i] @ df > thr2),
                                      dR2_dir=float((Fdir[k] - F[m["src"]]) @ df) / gap2, kl=kl[k], top1_changed=float(tp[k] != top_s[k]), coef=m.get("coef")))
    AT = A_all[T].astype(np.float64)
    for nm, ii in (("real held-out A", teA), ("real held-out B", refB)):
        Z = X[ii]; Az = A_all[ii].astype(np.float64); refs.append(dict(pair=pi, name=nm, n=len(ii), sur_B=float((Z @ sur_w + sur_b > 0).mean()), ind_B=float((Z @ ind_w + ind_b > 0).mean()), nc_B=float(ncB(Z).mean()), sae_B=float(saeB(Az).mean()),
                                                                     onB=float((Az[:, fB] > 0).mean()) if len(fB) else np.nan, onA=float((Az[:, fA] > 0).mean()) if len(fA) else np.nan, atyp=float(atyp(Z).mean()),
                                                                     out_B=float((r_real[at(ii)] > thr).mean()), fin_B=float((F[ii] @ df > thr2).mean())))
    P_.update(auc_out=auc_out, gap=gap, gap_se=float(np.sqrt(rA.var(ddof=1) / len(rA) + rB.var(ddof=1) / len(rB))), out_ok=bool(auc_out >= 0.85))
    print(f"{a} -> {b}: output score AUROC on real held-out prompts {auc_out:.2f} ({'used' if auc_out >= 0.85 else 'NOT used for output statements'}), real gap {gap:.2f} +- {P_['gap_se']:.2f}; real B above the calibrated threshold: {refs[-1]['out_B']:.2f}", flush=True)
R = pd.DataFrame(rows); RF = pd.DataFrame(refs); OK = [P_["pair"] for P_ in pairs if P_["out_ok"]]; pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
R.to_csv(f"{out}/mim_react2_{tag}.csv", index=False); RF.to_csv(f"{out}/mim_react2_refs_{tag}.csv", index=False)
NUM = ["rel_dist", "cm", "s_sur", "sur_B", "ind_B", "nc_B", "sae_B", "onB", "onA", "atyp", "dR", "out_B", "dR_dir", "dR2", "fin_B", "dR2_dir", "kl", "top1_changed"]; OUTC = ["dR", "out_B", "dR_dir"]
S = R.groupby(["pair", "src", "key"]).agg({**{c: "mean" for c in NUM}, "valid": "all"}).reset_index()          # one line per source and state type (the two random draws are averaged)
def table(keys, title, cols, min_src=5):
    """Mean over pairs of the per-pair mean, on sources that are valid for EVERY listed row. Output columns use only pairs that pass the gate."""
    G = S[S.key.isin(keys)]; ok = G.groupby(["pair", "src"]).agg(nk=("key", "size"), v=("valid", "all")); ok = ok[(ok.nk == len(keys)) & ok.v].index
    G = G.set_index(["pair", "src"]).loc[ok].reset_index(); cnt = G.groupby("pair").src.nunique(); keep = cnt[cnt >= min_src].index; G = G[G.pair.isin(keep)]
    Mp = G.groupby(["pair", "key"])[NUM].mean(); M = Mp.groupby("key").mean()
    Mo = Mp[Mp.index.get_level_values("pair").isin(OK)].groupby("key").mean()
    for c in OUTC: M[c] = Mo[c] if len(Mo) else np.nan
    M = M.reindex([k for k in keys if k in M.index])[cols]
    print(f"\n{title}\n   [{len(keep)} pairs, {int(cnt[keep].sum()) if len(keep) else 0} sources valid in every row; output columns (dR, out_B, dR_dir) from the {len(set(keep) & set(OK))} pairs with a usable output measure]"); print(M.round(2).to_string()); return M, Mp
def over_pairs(Mp, k1, k2, col):                          # difference of two rows with pairs as the unit: mean, t-interval, range, count of positive pairs
    w = Mp[col].unstack("key").reindex(columns=[k1, k2]).dropna()
    if col in OUTC: w = w[w.index.isin(OK)]
    dl = w[k1] - w[k2]
    if len(dl) < 2: return f"{col}: n/a"
    h = stats.t.ppf(.975, len(dl) - 1) * dl.std(ddof=1) / np.sqrt(len(dl)); return f"{col} {dl.mean():+.2f} [{dl.mean() - h:+.2f},{dl.mean() + h:+.2f}] (range {dl.min():+.2f}..{dl.max():+.2f}, {int((dl > 0).sum())}/{len(dl)} pairs)"
print("\nREFERENCE: real held-out prompts (mean over pairs; output columns: pairs with usable output measure)")
ref = RF.groupby("name")[["sur_B", "ind_B", "nc_B", "sae_B", "onB", "onA", "atyp", "fin_B"]].mean(); ref["out_B"] = RF[RF.pair.isin(OK)].groupby("name").out_B.mean(); print(ref.round(2).to_string())
C1 = ["rel_dist", "cm", "s_sur", "sur_B", "ind_B", "nc_B", "sae_B", "onB", "onA", "atyp", "dR", "out_B", "dR2", "dR2_dir"]; J_ = {}
# ---- A) the double dissociation
kA = ["orig", "probeonly@1", "wperp_star", "probe@1", "dmblind_0.5", "dmblind_1", "dmfull_0.5", "dmfull_1", "swap", "swapnn"]
MA, MpA = table(kA, "A) DOUBLE DISSOCIATION. wperp_star = probe normal without its class-mean part, scaled until the surrogate reads 'typical B'; probe@1 = plain probe normal at that score;\n   dmblind_1 = full class-mean difference without its probe-normal part (surrogate score unchanged); dmfull_1 = plain class-mean difference; swap = a real B state.", C1)
print("   pairs as the unit:"); [print("     ", f"{k1} minus {k2}:", over_pairs(MpA, k1, k2, "dR"), "|", over_pairs(MpA, k1, k2, "out_B"), "|", over_pairs(MpA, k1, k2, "dR2")) for k1, k2 in (("wperp_star", "orig"), ("probe@1", "wperp_star"), ("dmblind_1", "orig"), ("dmblind_1", "dmfull_1"), ("dmfull_1", "probe@1"), ("swap", "dmfull_1"))]
print("   per pair, output reaction dR (pairs with usable output measure) and out_B:"); print(MpA[MpA.index.get_level_values("pair").isin(OK)].dR.unstack("key").reindex(columns=kA).round(2).to_string()); print(MpA[MpA.index.get_level_values("pair").isin(OK)].out_B.unstack("key").reindex(columns=kA).round(2).to_string())
J_["A"] = MA.round(4).reset_index().to_dict("records")
# ---- B) three matchings of the probe normal against the class-mean translation
W_ = S.set_index(["pair", "src", "key"]); res = []
for (p, s), gsrc in S[S.key == "probe@1"].groupby(["pair", "src"]):
    c0 = float(gsrc.cm.iloc[0]); grid = [("orig", 0.0)] + [(f"classmean@{f:g}", None) for f in (0.5, 1, 1.5, 2, 2.5)]; xs, ys = [], {c: [] for c in ("dR", "dR2", "out_B")}
    for k, _ in grid:
        r_ = W_.loc[(p, s, k)]; xs.append(float(r_.cm)); [ys[c].append(float(r_[c])) for c in ys]
    o = np.argsort(xs); res.append(dict(pair=p, src=s, **{f"probe_{c}": float(gsrc[c].iloc[0]) for c in ys}, **{f"eqcm_{c}": float(np.interp(c0, np.array(xs)[o], np.array(ys[c])[o])) for c in ys},
                                       **{f"eqlen_{c}": float(W_.loc[(p, s, "classmean@1")][c]) for c in ys}, **{f"eqscore_{c}": float(W_.loc[(p, s, "dm_lvlstar")][c]) for c in ys}, cm_probe=c0, in_range=bool(c0 <= max(xs))))
B_ = pd.DataFrame(res); Bp = B_.groupby("pair").mean(numeric_only=True)
print("\nB) THREE MATCHINGS: reaction to the probe-normal state at s* against a class-mean translation with (a) the same probe score, (b) the same length, (c) the same class-mean coordinate (interpolated)")
for c, sel in (("dR", Bp[Bp.index.isin(OK)]), ("out_B", Bp[Bp.index.isin(OK)]), ("dR2", Bp)):
    print(f"   {c:5s} ({len(sel)} pairs): probe normal {sel['probe_' + c].mean():.2f} | class-mean translation at equal score {sel['eqscore_' + c].mean():.2f} | equal length {sel['eqlen_' + c].mean():.2f} | equal class-mean coordinate {sel['eqcm_' + c].mean():.2f}"
          f"  (difference probe minus equal-coordinate, range over pairs {(sel['probe_' + c] - sel['eqcm_' + c]).min():+.2f}..{(sel['probe_' + c] - sel['eqcm_' + c]).max():+.2f})")
print(f"   class-mean coordinate of the probe-normal state at s*: {Bp.cm_probe.mean():.2f} (range over pairs {Bp.cm_probe.min():.2f}-{Bp.cm_probe.max():.2f})"); J_["B"] = Bp.round(4).reset_index().to_dict("records")
# ---- C) equal length, all directions
for f in (1, 2):
    keys = [f"{dr}@{f:g}" for dr in ("probe", "probeonly", "classmean", "centroid", "nearestB", "lensgrad", "sae20", "randlabel", "random")]
    M_, Mp_ = table(keys, f"C{f}) EQUAL LENGTH: every displacement has {f} x the length at which the probe normal reaches s*. probeonly = probe normal without class-mean part; centroid = towards the centroid of B;\n   nearestB = towards the nearest real B state; lensgrad = lens gradient direction; sae20 = the 20 most separating SAE units towards their B mean.", C1)
    print("   pairs as the unit:", " || ".join(f"{k1.split('@')[0]} minus {k2.split('@')[0]}: " + over_pairs(Mp_, k1, k2, "dR2") + "; " + over_pairs(Mp_, k1, k2, "dR") for k1, k2 in ((keys[0], keys[8]), (keys[2], keys[0]), (keys[3], keys[0]), (keys[4], keys[0]), (keys[1], keys[8]))))
    J_[f"C{f}"] = M_.round(4).reset_index().to_dict("records")
# ---- D) dose series
MD, _ = table([f"probe@{f:g}" for f in (0.25, 0.5, 1, 1.5, 2, 2.5, 3)], "D) DOSE along the probe normal", C1); MD2, _ = table([f"classmean@{f:g}" for f in (0.5, 1, 1.5, 2, 2.5)], "   DOSE along the class-mean difference", C1)
MD3, _ = table([f"random@{f:g}" for f in (1, 2, 2.5, 3)], "   DOSE along random directions", C1); J_["D"] = dict(probe=MD.round(4).reset_index().to_dict("records"), classmean=MD2.round(4).reset_index().to_dict("records"), random=MD3.round(4).reset_index().to_dict("records"))
# ---- E) the lambda series at the same probe score (wide kernel = pull towards the centroid of B)
ME, MpE = table(["probe@1", "lam1_lvlstar", "lam5_lvlstar", "laminf_lvlstar", "dm_lvlstar", "protoB"], "E) SAME PROBE SCORE s*, increasing weight lambda of the density term (with this kernel width: a pull towards the centroid of B); dm_lvlstar = class-mean translation; protoB = the centroid itself", C1)
J_["E"] = ME.round(4).reset_index().to_dict("records")
# ---- F) references and SAE-specific states
MF, _ = table(["orig", "saerec", "sae20_c1", "sae20_star", "protoA", "protoAll", "protoC", "protoB", "swapnn", "swap"], "F) REFERENCES: saerec = source replaced by its SAE reconstruction; sae20_c1 = 20 units set to their B mean; sae20_star = the same displacement scaled until the surrogate reads s*;\n   protoA / protoAll / protoC / protoB = centroid of the own family / of all prompts / of an unrelated family / of B; swapnn / swap = nearest / random real B state", C1 + ["kl", "top1_changed"])
co = R[(R.name == "sae20_star") & R.valid].coef.astype(float); print(f"   scale needed for sae20_star: median {co.median():.1f}, 10-90% {co.quantile(.1):.1f}-{co.quantile(.9):.1f} (1 = the units sit at their B mean)"); J_["F"] = MF.round(4).reset_index().to_dict("records")
# ---- G) what predicts the reaction across all states: class-mean coordinate, probe score or length?
G = S[S.valid & (S.key != "orig")]; out_rows = []
for p, gp in G.groupby("pair"):
    out_rows.append(dict(pair=p, **{f"{y}~{x}": float(stats.spearmanr(gp[x], gp[y])[0]) for y in (("dR", "dR2") if p in OK else ("dR2",)) for x in ("cm", "s_sur", "rel_dist")}))
GG = pd.DataFrame(out_rows).set_index("pair"); print("\nG) Spearman over all state types within a pair: reaction against class-mean coordinate (cm), probe score, length"); print(GG.round(2).to_string()); print("   mean:", GG.mean().round(2).to_dict())
J_["G"] = GG.round(4).reset_index().to_dict("records"); J_["pairs"] = [{k: v for k, v in P_.items() if not isinstance(v, list)} for P_ in pairs]; J_["refs"] = ref.round(4).reset_index().to_dict("records"); J_["output_measure_pairs"] = OK
json.dump(J_, open(f"{out}/mim_react2_{tag}.json", "w"), indent=1)
print("saved")

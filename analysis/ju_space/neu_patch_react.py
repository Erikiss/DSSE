"""Reaction of the network to states in which SETS of units were set to their target-family means (neu_patch_sets.py):
k leading neurons, k random neurons, all neurons (MLP-written part), the attention-written part, k leading SAE latents.
Same output measures and calibration as mim_react2.py ('output reads B' = output score above the 95th percentile of real
held-out A prompts; pairs where that score separates real held-out prompts with AUROC >= 0.85).
Usage: python neu_patch_react.py <scratch_root> <out_dir> [t0]"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mim_common import *
root, out = sys.argv[1], sys.argv[2]; tag = sys.argv[3] if len(sys.argv) > 3 else "t0"
lab, text, X, F = load_eval(out); WU, g = load_unembed(root); pairs = json.load(open(f"{out}/mim_variants_{tag}.json"))["pairs"]
meta = json.load(open(f"{out}/mim_variants_{tag}n.json"))["meta"]; V = np.load(f"{out}/mim_variants_{tag}n.npy").astype(np.float64); Fp = np.load(f"{out}/mim_final_{tag}n.npy").astype(np.float64)
def logp_of(Fm, chunk=128):
    for i in range(0, len(Fm), chunk): yield i, log_softmax((Fm[i:i + chunk].astype(np.float32) @ WU.T).astype(np.float64))
rows = []; ok_pairs = []
for P in pairs:
    pi, a, b = P["pair"], P["a"], P["b"]; S1, S2, T = map(np.array, (P["S1"], P["S2"], P["T"])); tr = np.concatenate([S1, S2]); refB = np.array(P["refB"]); teA = T[lab[T] == a]
    idx = np.array([i for i, m in enumerate(meta) if m["pair"] == pi]); src = np.array([meta[i]["src"] for i in idx])
    use = np.concatenate([tr, T]); LP = np.concatenate([lp for _, lp in logp_of(F[use])]); pos = {int(q): k for k, q in enumerate(use)}; at = lambda ii: [pos[int(q)] for q in ii]
    QA, QB = np.exp(LP[at(tr[lab[tr] == a])]).mean(0), np.exp(LP[at(tr[lab[tr] == b])]).mean(0); wv = np.log(QB + 1e-6) - np.log(QA + 1e-6); r_real = np.exp(LP) @ wv
    rA, rB = r_real[at(teA)], r_real[at(refB)]; gap = float(rB.mean() - rA.mean()); thr = float(np.percentile(rA, 95)); au = auroc(np.r_[rA, rB], np.r_[np.zeros(len(rA)), np.ones(len(rB))])
    if au >= .85: ok_pairs.append(pi)
    df = unit(F[tr][lab[tr] == b].mean(0) - F[tr][lab[tr] == a].mean(0)); gap2 = float((F[refB] @ df).mean() - (F[teA] @ df).mean())
    muA, muB = X[tr][lab[tr] == a].mean(0), X[tr][lab[tr] == b].mean(0); r_p = np.concatenate([np.exp(lp) @ wv for _, lp in logp_of(Fp[idx])]); r_s = r_real[at(src)]
    for k, i in enumerate(idx):
        m = meta[i]; rows.append(dict(pair=pi, src=m["src"], name=m["name"], cm=m["cm"], sur_B=float(m["s_sur"] > 0), ind_B=float(m["s_ind"] > 0), nc_B=float(((V[i] - muB) ** 2).sum() < ((V[i] - muA) ** 2).sum()),
                                      out_B=float(r_p[k] > thr), dR=(r_p[k] - r_s[k]) / gap, dR2=float((Fp[i] - F[m["src"]]) @ df) / gap2, real_B_out=float((rB > thr).mean())))
R = pd.DataFrame(rows); R.to_csv(f"{out}/neu_patch_react.csv", index=False); pd.set_option("display.width", 220)
order = [f"neu_top{k}" for k in (20, 100, 300, 1000, 3000)] + ["mlp_all", "attn_all"] + [f"neu_rand{k}" for k in (20, 100, 300, 1000, 3000)] + [f"sae_top{k}" for k in (20, 100, 300)]
Mp = R.groupby(["pair", "name"])[["cm", "sur_B", "ind_B", "nc_B", "out_B", "dR", "dR2"]].mean(); use4 = [p for p in ok_pairs if p != 7]
T_ = pd.DataFrame({"class_mean_share": Mp.cm.groupby("name").mean(), "probe_B": Mp.sur_B.groupby("name").mean(), "second_probe_B": Mp.ind_B.groupby("name").mean(), "centroid_rule_B": Mp.nc_B.groupby("name").mean(),
                   "final_state": Mp.dR2.groupby("name").mean(), "out_B_pairs3to6": Mp[Mp.index.get_level_values("pair").isin(use4)].out_B.groupby("name").mean(),
                   "out_dR_pairs3to6": Mp[Mp.index.get_level_values("pair").isin(use4)].dR.groupby("name").mean(), "out_B_pair7": Mp.xs(7, level="pair").out_B}).reindex(order)
print(f"pairs with a usable output measure: {ok_pairs}; real held-out B prompts read as B by the output in pairs {use4}: {R[R.pair.isin(use4)].groupby('pair').real_B_out.first().mean():.2f}")
print("units set to their mean on the target family B (direct effect on the block-21 state), mean over pairs:"); print(T_.round(2).to_string())
print("per pair, output reads B:"); print(Mp[Mp.index.get_level_values("pair").isin(ok_pairs)].out_B.unstack("name").reindex(columns=order).round(2).to_string())
try:
    S = pd.read_csv(f"{out}/mim_summary_t0.csv", index_col=0); print("for comparison, from the earlier run:", {k: round(float(S.loc[k, "out_B_pairs3to6"]), 2) for k in ("orig", "sae20_c1", "dmfull_1", "dmblind_1", "protoB", "swap") if k in S.index})
except Exception as e: print("earlier summary not found", e)
json.dump(dict(table=T_.round(4).reset_index().to_dict("records"), pairs_output=ok_pairs), open(f"{out}/neu_patch_react.json", "w"), indent=1); print("saved")

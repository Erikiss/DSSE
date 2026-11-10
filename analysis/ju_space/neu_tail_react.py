"""Reaction of the neurons behind the intervention (blocks 22-27, 36,864 units) to the attack / control states.
Scale: the real difference between the two prompt families in these neurons (training prompts), each neuron in units of its own
spread inside a family. 'coord' = how far the change of a patched state goes along that real difference (0 = nothing, 1 = the
full real difference); 'cos' = how much of the change points along it; 'sep100' = share of the 100 most separating neurons
that moved at least half of their real gap towards B; 'broad' = share of all downstream neurons that moved by more than one
within-family standard deviation; 'off' = size of the change that is NOT along the real difference, relative to that difference.
Usage: python neu_tail_react.py <scratch_root> <out_dir> [t0]"""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mim_common import *
root, out = sys.argv[1], sys.argv[2]; tag = sys.argv[3] if len(sys.argv) > 3 else "t0"
lab, text, X, F = load_eval(out); n = len(lab); A = np.load(f"{out}/neu_act.npy", mmap_mode="r"); D = np.asarray(A[:, BLOCK + 1:], dtype=np.float32).reshape(n, -1)
TA = np.load(f"{out}/neu_tail_act.npy").astype(np.float32); TA = TA.reshape(len(TA), -1); meta = json.load(open(f"{out}/neu_tail_meta.json")); pairs = json.load(open(f"{out}/mim_variants_{tag}.json"))["pairs"]
CAN = {"lam0_lvlstar": "probe normal, 1x (probe reads typical B)", "dose0_2": "probe normal, 2x", "wperp_1": "probe-only part, 1x", "wperp_star": "probe-only part until the probe reads typical B", "dosedm_1": "class-mean difference, 1x",
       "dmblind_0.5": "half class-mean difference, hidden from the probe", "dmblind_1": "full class-mean difference, hidden from the probe", "dmfull_1": "full class-mean difference", "dosecen_1": "towards the centroid of B, 1x",
       "sae20_c1": "20 SAE units set to their B mean", "lens_t0": "lens gradient, 1x", "rand1": "random direction, 1x", "rand1_2": "random direction, 2x", "rand1_3": "random direction, 3x", "randlabel": "random-label SVM normal, 1x",
       "protoA": "centroid of the own family", "protoB": "centroid of B", "swap1": "real B state", "swapnn": "nearest real B state", "orig": "unchanged"}
rows, refs = [], []
for P in pairs:
    pi, a, b = P["pair"], P["a"], P["b"]; tr = np.array(P["S1"] + P["S2"]); iA, iB = tr[lab[tr] == a], tr[lab[tr] == b]; T = np.array(P["T"]); teA, teB = T[lab[T] == a], T[lab[T] == b]
    muA, muB = D[iA].mean(0), D[iB].mean(0); sd = np.sqrt((D[iA].var(0) + D[iB].var(0)) / 2) + 1e-3; dz = (muB - muA) / sd; nz = float(dz @ dz); top = np.argsort(-np.abs(dz))[:100]; gap = (muB - muA)[top]
    coord = lambda Z: ((Z - muA) / sd) @ dz / nz
    refs.append(dict(pair=pi, a=a, b=b, real_A_heldout=float(coord(D[teA]).mean()), real_B_heldout=float(coord(D[teB]).mean()), sep100_real_B=float((((D[teB][:, top] - muA[top]) * np.sign(gap)) >= .5 * np.abs(gap)).mean()),
                     sep100_real_A=float((((D[teA][:, top] - muA[top]) * np.sign(gap)) >= .5 * np.abs(gap)).mean()), off_real_B=float(np.mean([np.linalg.norm((x - muA) / sd - coord(x[None])[0] * dz) for x in D[teB]]) / np.sqrt(nz)),
                     off_real_A=float(np.mean([np.linalg.norm((x - muA) / sd - coord(x[None])[0] * dz) for x in D[teA]]) / np.sqrt(nz))))
    mine = [m for m in meta if m["pair"] == pi]; orig = {m["src"]: TA[m["row"]] for m in mine if m["name"] == "orig"}
    for m in mine:
        dl = (TA[m["row"]] - orig[m["src"]]) / sd; c = float(dl @ dz / nz); nd = float(np.linalg.norm(dl))
        rows.append(dict(pair=pi, src=m["src"], name=m["name"], state=CAN[m["name"]], valid=m["valid"], coord=c, cos=float(dl @ dz / (nd * np.sqrt(nz) + 1e-12)), sep100=float((((TA[m["row"]] - orig[m["src"]])[top] * np.sign(gap)) >= .5 * np.abs(gap)).mean()),
                         broad=float((np.abs(dl) > 1).mean()), off=float(np.linalg.norm(dl - c * dz) / np.sqrt(nz)), s_sur=m["s_sur"], sur_B=float(m["s_sur"] > 0)))
R = pd.DataFrame(rows); RF = pd.DataFrame(refs); R.to_csv(f"{out}/neu_tail_react.csv", index=False); pd.set_option("display.width", 240); pd.set_option("display.max_colwidth", 60)
print("reference, real held-out prompts (coordinate along the real family difference in downstream neurons; mean over pairs):", RF[["real_A_heldout", "real_B_heldout", "sep100_real_A", "sep100_real_B", "off_real_A", "off_real_B"]].mean().round(2).to_dict())
G = R[R.valid]; Mp = G.groupby(["pair", "name"])[["sur_B", "coord", "cos", "sep100", "broad", "off"]].mean(); M = Mp.groupby("name").mean(); rngp = Mp.coord.unstack("name")
M["coord_min"] = rngp.min(); M["coord_max"] = rngp.max(); M["pairs"] = rngp.notna().sum(); M.index = [CAN[k] for k in M.index]; order = [CAN[k] for k in CAN if CAN[k] in M.index]
print("\nreaction of the 36,864 neurons of blocks 22-27 (mean over the 8 pairs of the per-pair means; coord_min / coord_max = range over pairs):"); print(M.loc[order, ["pairs", "sur_B", "coord", "coord_min", "coord_max", "cos", "sep100", "broad", "off"]].round(2).to_string())
def diff(k1, k2, col="coord"):
    w = Mp[col].unstack("name")[[k1, k2]].dropna(); dl = w[k1] - w[k2]; h = stats.t.ppf(.975, len(dl) - 1) * dl.std(ddof=1) / np.sqrt(len(dl)); return f"{dl.mean():+.2f} [{dl.mean() - h:+.2f},{dl.mean() + h:+.2f}] (range {dl.min():+.2f}..{dl.max():+.2f}, {int((dl > 0).sum())}/{len(dl)} pairs)"
print("\npairs as the unit, difference in coord:")
for k1, k2 in (("wperp_star", "rand1_2"), ("wperp_1", "rand1"), ("lam0_lvlstar", "rand1"), ("dmblind_1", "rand1_3"), ("dmblind_1", "dmfull_1"), ("dmfull_1", "lam0_lvlstar"), ("swap1", "dmfull_1"), ("dmblind_1", "wperp_star")):
    print(f"   {CAN[k1]}  minus  {CAN[k2]}: {diff(k1, k2)}")
json.dump(dict(table=M.loc[order].round(4).reset_index().rename(columns={"index": "state"}).to_dict("records"), refs=RF.round(4).to_dict("records")), open(f"{out}/neu_tail_react.json", "w"), indent=1)
print("saved")

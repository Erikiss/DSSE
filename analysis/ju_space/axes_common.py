"""Is there one axis shared by the behaviors? For each behavior: build the common direction from the shown-minus-not-shown
directions of the OTHER behaviors and test it on the left-out one (AUROC shown vs not shown). Uses the states of extract_axes.py.
Usage: python axes_common.py <out_dir>"""
import json, sys
import numpy as np
from sklearn.metrics import roc_auc_score
out = sys.argv[1]; pairs = json.load(open(f"{out}/axes_pairs.json")); flat = json.load(open(f"{out}/axes_flat.json")); Z = np.load(f"{out}/axes_states.npz")
unit = lambda a: a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
beh_of = np.array([pairs[k]["behavior"] for k, _ in flat]); shown = np.array([s == "shown" for _, s in flat]); behs = sorted(set(beh_of)); res = {}
for key in [k for k in Z.files if k != "ntok"]:
    X = unit(Z[key].astype(np.float64)); D = {b: X[(beh_of == b) & shown].mean(0) - X[(beh_of == b) & ~shown].mean(0) for b in behs}
    G = np.array([[unit(D[a]) @ unit(D[b]) for b in behs] for a in behs]); off = G[~np.eye(len(behs), dtype=bool)]; ev = np.linalg.eigvalsh(G)[::-1]
    loo = {b: float(roc_auc_score(shown[beh_of == b], X[beh_of == b] @ unit(np.mean([unit(D[a]) for a in behs if a != b], 0)))) for b in behs}
    res[key] = dict(loo_mean=float(np.mean(list(loo.values()))), loo=loo, cos_mean=float(off.mean()), cos_q=[float(np.percentile(off, 5)), float(np.percentile(off, 95))], top_eig_share=float(ev[0] / ev.sum()))
    print(f"[{key}] cosine between the directions of different behaviors: mean {off.mean():+.2f} (5-95% {np.percentile(off, 5):+.2f}..{np.percentile(off, 95):+.2f}); first component holds {ev[0] / ev.sum():.2f}")
    print("   common axis from the other behaviors, AUROC on the left-out one: mean", round(res[key]["loo_mean"], 3), "|", {b[:12]: round(v, 2) for b, v in loo.items()})
json.dump(res, open(f"{out}/axes_common_factor.json", "w"), indent=1)

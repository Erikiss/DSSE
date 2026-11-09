"""Part 4: robustness of the 'unseen base wrong value' test.
Null built by leave-one-DATASET-out: hold out the 3 runs of one dataset, take one of its wrong answers on the
problem as a pseudo-base draw, and ask whether its value is absent from the other 31 datasets' wrong answers.
This mimics base being a draw from 'a new context' and respects dataset clustering.
Also: different rounding (0, 2, 4 dp) and the experiment's last-number extractor; drop problem 19.
"""
import numpy as np, pandas as pd
from pathlib import Path
from collections import Counter
HERE = Path(__file__).resolve().parent
R = pd.read_pickle(HERE / "rows.pkl")
rng = np.random.default_rng(7)

def run(xcol, wcol, dp, drop=()):
    R["v"] = R[xcol].round(dp)
    base = R[R.route == "base"].set_index("prompt_idx")
    icl = R[(R.route == "icl") & (R.k == 16) & R[wcol]]
    bw = base.index[base[wcol] & ~base.index.isin(drop)]
    obs, exp_list = 0, []
    detail = []
    for p in bw:
        g = icl[icl.prompt_idx == p]
        vals = g.v.values
        if len(vals) == 0:
            continue
        seen = base.loc[p, "v"] in set(vals)
        obs += (not seen)
        # LODO unseen probability: average over held-out wrong answers (weight each dataset by its wrong answers
        # -> equivalent to drawing a random icl wrong answer), unseen among other datasets' wrong values
        pu = []
        for d, gd in g.groupby("dataset_id"):
            others = set(g[g.dataset_id != d].v.values)
            pu += [v not in others for v in gd.v.values]
        exp_list.append(np.mean(pu))
        detail.append((p, round(np.mean(pu), 3), not seen))
    E = np.array(exp_list)
    sims = (rng.random((40000, len(E))) < E).sum(1)
    return obs, len(E), E.sum(), (sims >= obs).mean(), detail

for xcol, wcol in [("x_ca", "w_ca"), ("x_last", "w_last")]:
    for dp in [0, 2, 4]:
        for drop in [(), (19,)]:
            o, n, e, pv, det = run(xcol, wcol, dp, drop)
            print(f"{xcol} round={dp} drop={drop}: base value unseen among icl wrongs on {o}/{n} base-wrong problems; "
                  f"LODO-expected {e:.2f}; P(>=obs)={pv:.4f}")
o, n, e, pv, det = run("x_ca", "w_ca", 2)
print("per problem (p, LODO unseen prob, observed unseen):", det)

"""Part 2: are icl / ft samples coupled to the base sample through the shared torch RNG seed?
generate() calls torch.manual_seed(seed) once per call; base, ft and icl seed 0 all use seed 0.
Measure: common-prefix length (chars) and exact identity of responses vs the base response for the same
prompt, by route and seed; compare with icl-vs-icl pairs from different datasets (same seed vs different seed).
"""
import numpy as np, pandas as pd, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = pd.read_pickle(HERE / "rows.pkl")
base = R[R.route == "base"].set_index("prompt_idx").response
def cpl(a, b):
    return len(os.path.commonprefix([a, b]))
icl = R[(R.route == "icl") & (R.k == 16)].copy()
ft = R[R.route == "ft"].copy()
for D, nm in [(icl, "icl"), (ft, "ft")]:
    D["cpl_base"] = [cpl(r, base[p]) for r, p in zip(D.response, D.prompt_idx)]
    D["rel_cpl"] = D.cpl_base / D.response.str.len().clip(lower=1)
    D["same_as_base"] = [r == base[p] for r, p in zip(D.response, D.prompt_idx)]
print("icl vs base by seed: mean common prefix chars, median, share identical, share prefix>=200 chars")
print(icl.groupby("seed").agg(cpl=("cpl_base", "mean"), med=("cpl_base", "median"), ident=("same_as_base", "mean"),
                               ge200=("cpl_base", lambda x: (x >= 200).mean())).round(3))
print("ft vs base:", ft.cpl_base.mean().round(1), ft.cpl_base.median(), ft.same_as_base.mean().round(4),
      (ft.cpl_base >= 200).mean().round(3))
# icl-vs-icl pairs across different datasets: same seed vs different seed
rng = np.random.default_rng(0)
piv = icl.set_index(["prompt_idx", "dataset_id", "seed"]).response
ds = icl.dataset_id.unique()
same, diff = [], []
for p in range(100):
    for _ in range(40):
        d1, d2 = rng.choice(ds, 2, replace=False)
        s1 = rng.integers(0, 3); s2 = (s1 + rng.integers(1, 3)) % 3
        same.append(cpl(piv[(p, d1, s1)], piv[(p, d2, s1)]))
        diff.append(cpl(piv[(p, d1, s1)], piv[(p, d2, s2)]))
print(f"icl-icl different datasets: same seed mean cpl {np.mean(same):.1f} (>=200: {np.mean(np.array(same)>=200):.3f}); "
      f"different seed {np.mean(diff):.1f} (>=200: {np.mean(np.array(diff)>=200):.3f})")
# does coupling matter for wrongness? P(icl wrong | base wrong) by seed
bw = R[R.route == "base"].set_index("prompt_idx").w_ca
icl["bw"] = icl.prompt_idx.map(bw)
print("P(icl wrong | base wrong) by seed:", icl[icl.bw].groupby("seed").w_ca.mean().round(3).to_dict(),
      " | base right:", icl[~icl.bw].groupby("seed").w_ca.mean().round(3).to_dict())
icl.to_pickle(HERE / "icl_cpl.pkl")

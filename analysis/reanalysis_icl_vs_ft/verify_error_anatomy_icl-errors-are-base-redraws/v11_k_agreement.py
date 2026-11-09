"""Part 11: dose-response in agreement with the single base sample, same 4 curve datasets (seed 0).
For each k: (i) P(answer wrong | base wrong) and P(answer wrong | base right);
(ii) on base-wrong problems: P(answer == base's wrong value) (any answer, right or wrong);
(iii) answer-level agreement with base outcome. Bootstrap over problems (paired). Compares small k (1,2) vs large k (16,32).
"""
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = pd.read_pickle(HERE / "rows.pkl")
rng = np.random.default_rng(4)
R["v"] = R.x_ca.round(2)
CUR = ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]
cur = R[(R.route == "icl") & (R.seed == 0) & R.dataset_id.isin(CUR)].copy()
base = R[R.route == "base"].set_index("prompt_idx")
cur["bw"] = cur.prompt_idx.map(base.w_ca); cur["bv"] = cur.prompt_idx.map(base.v)
cur["same_val"] = cur.v == cur.bv
rows = []
for k, g in cur.groupby("k"):
    rows.append(dict(k=int(k), wrong_bw=g[g.bw].w_ca.mean(), wrong_br=g[~g.bw].w_ca.mean(),
                     same_val_bw=g[g.bw].same_val.mean(), agree=(g.w_ca == g.bw).mean(), n=len(g)))
print(pd.DataFrame(rows).round(3).to_string(index=False))
cur["kk"] = np.where(cur.k <= 2, "small", np.where(cur.k >= 16, "large", "mid"))
P = cur[cur.kk != "mid"].groupby(["prompt_idx", "kk"]).agg(w=("w_ca", "mean"), sv=("same_val", "mean")).unstack()
bw = base.w_ca.loc[P.index].values.astype(bool)
def d(ix):
    Q = P.iloc[ix]; b = bw[ix]
    return [Q["w"]["small"][b].mean() - Q["w"]["large"][b].mean(), Q["w"]["small"][~b].mean() - Q["w"]["large"][~b].mean(),
            Q["sv"]["small"][b].mean() - Q["sv"]["large"][b].mean()]
obs = d(np.arange(len(P)))
B = np.array([d(rng.integers(0, len(P), len(P))) for _ in range(4000)])
for j, nm in enumerate(["wrong|base wrong (small-large k)", "wrong|base right (small-large k)", "== base value | base wrong (small-large k)"]):
    print(f"{nm}: {obs[j]:+.3f} CI [{np.nanpercentile(B[:,j],2.5):+.3f},{np.nanpercentile(B[:,j],97.5):+.3f}]  (8 answers per problem per arm)")

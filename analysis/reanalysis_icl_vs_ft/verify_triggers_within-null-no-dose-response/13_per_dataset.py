"""Heterogeneity: is a shot effect hiding in a few datasets?  Per dataset, across-seed variance of run means vs exact
null B (responses permuted across seeds within problem), 4000 draws; BH over 32 datasets.  96 runs (outlier kept,
as a positive control) -- dataset 08152 with only seeds 1,2 also reported."""
import numpy as np, pandas as pd
R = pd.read_pickle("resp_v.pkl")
icl = R[(R.route == "icl") & (R.k == 16)]
rng = np.random.default_rng(17)
B = 4000
def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)
rows = []
for d, g in icl.groupby("dataset_id"):
    for col in ["wrong_l", "wrong_s", "nw", "short40"]:
        for lab, gg in [("all", g), ("no_s0", g[g.seed != 0])] if d == "dsse1-08152" else [("all", g)]:
            v = gg.pivot(index="seed", columns="prompt_idx", values=col).values.astype(float)
            obs = v.mean(1).var(ddof=1)
            null = np.array([np.take_along_axis(v, np.argsort(rng.random(v.shape), 0), 0).mean(1).var(ddof=1) for _ in range(B)])
            rows.append(dict(dataset_id=d + ("" if lab == "all" else "_s12"), outcome=col, means=np.round(v.mean(1), 3).tolist(),
                             ratio=obs / null.mean(), p=(1 + (null >= obs).sum()) / (B + 1)))
D = pd.DataFrame(rows)
for col in ["wrong_l", "wrong_s", "nw", "short40"]:
    x = D[(D.outcome == col) & (D.dataset_id != "dsse1-08152_s12")].copy()
    x["q"] = bh(x.p)
    print(col, "| n datasets p<.05:", int((x.p < .05).sum()), "| q<.1:", x[x.q < .1][["dataset_id", "means", "ratio", "p", "q"]].round(4).to_dict("records"))
    x2 = D[(D.outcome == col) & (D.dataset_id != "dsse1-08152")].copy(); x2["q"] = bh(x2.p)
    print("   without outlier run (08152 seeds 1,2):", "n p<.05:", int((x2.p < .05).sum()), "min q", round(x2.q.min(), 3),
          x2.sort_values("p").head(2)[["dataset_id", "means", "p"]].round(4).to_dict("records"))
D.to_csv("13_per_dataset.csv", index=False)

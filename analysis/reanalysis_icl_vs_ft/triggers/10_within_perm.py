"""Randomization check of the within-dataset slopes from 09 (cluster SEs with ~130 FE can be anticonservative):
shuffle the shot-feature values among the seeds of each dataset (5000 draws) and recompute the within slope of the
run-level, problem-adjusted outcome on the feature.  95 runs (no 08152 s0)."""
from common import *

rng = np.random.default_rng(11)
R = pd.read_pickle(OUT / "resp_r.pkl")
icl = R[(R.route == "icl") & (R.k == 16) & (R.tag != "icl__dsse1-08152__k16__s0")].copy()
runs = pd.read_csv(OUT / "runs.csv").set_index("tag")
rows = []
for y in ["wrong_len", "wrong", "n_words2"]:
    icl["adj"] = icl[y] - icl.groupby("prompt_idx")[y].transform("mean")
    run = icl.groupby("tag").adj.mean().to_frame()
    run["dataset_id"] = runs.dataset_id.reindex(run.index)
    run["yd"] = run.adj - run.groupby("dataset_id").adj.transform("mean")
    groups = [g.index for _, g in run.groupby("dataset_id")]
    for f in ["s_words", "s_le2", "s_le30w", "s_hedge", "s_hedge_rate", "s_digit", "context_tokens"]:
        x = runs[f].reindex(run.index)
        xd = x - x.groupby(run.dataset_id).transform("mean")
        sd = xd.std()
        obs = (xd * run.yd).sum() / (xd ** 2).sum() * sd
        pos = {t: i for i, t in enumerate(run.index)}
        xdv = xd.values
        null = []
        for _ in range(5000):
            xp = xdv.copy()
            for g in groups:
                ii = [pos[t] for t in g]
                xp[ii] = xdv[rng.permutation(ii)]
            null.append((xp * run.yd.values).sum() / (xp ** 2).sum() * sd)
        null = np.array(null)
        rows.append(dict(outcome=y, feat=f, slope_per_within_sd=obs, p_perm=(1 + (np.abs(null) >= abs(obs)).sum()) / 5001))
o = pd.DataFrame(rows)
o["q"] = o.groupby("outcome").p_perm.transform(bh)
o.to_csv(OUT / "10_within_perm.csv", index=False)
print(o.round(4).to_string())

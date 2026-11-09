"""Conditional randomization test (CRT) of within-dataset dose-response: does the feature of the 16 shown rows predict
the run outcome?  Under H0 the shown rows are just one random 16-row draw from the dataset, so the observed feature can
be replaced by the feature of fresh random draws from the same dataset (B=4000 per run) -- an exact null that needs no
SE model.  Two statistics:
  within: OLS slope of y on x with dataset FE + seed FE (seed FE removes the common-random-number seed effect)
  popdev: OLS slope of y on (x - x_pop150) with seed FE, controlling x_pop150 (no dataset FE; uses the seed-average
          deviation too, as in the triggers lens' population-deviation design)
Slopes reported per within-dataset SD of x and per raw unit; 95% CI from cluster bootstrap over datasets (2000)."""
import re, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import shots_for, dataset_rows, DATASET_TABLE
from importlib import import_module
sf = import_module("03_shot_feats")

B = 4000
rng = np.random.default_rng(7)
OUTLIER = "icl__dsse1-08152__k16__s0"
FEATS = ["sh_words", "sh_le2", "sh_le30", "sh_digit", "sh_hedge", "sh_pw"]
OUTS = ["wrong_s", "wrong_l", "nw", "short40", "early_c", "early_w", "has_arith", "n_arith", "header", "n_lines", "boxed", "answer_first"]

R = pd.read_pickle("resp_v.pkl")
icl = R[(R.route == "icl") & (R.k == 16)].copy()
icl["answer_first"] = icl.answer_first.astype(float)
run = icl.groupby("tag").agg(dataset_id=("dataset_id", "first"), seed=("seed", "first"), **{o: (o, "mean") for o in OUTS}).reset_index()
run["seed"] = run.seed.astype(int)
S = pd.read_csv("03_shot_feats.csv").query("k == 16")
run = run.merge(S, on=["dataset_id", "seed"])
ids = sorted(run.dataset_id.unique())
# per-row features for fresh draws
rowF = {}
for d in ids:
    f = pd.DataFrame([sf.rowf(r) for r in dataset_rows(d)])
    rowF[d] = dict(sh_words=f.w.values.astype(float), sh_le2=f.le2.values.astype(float), sh_le30=f.le30.values.astype(float),
                   sh_digit=f.digit.values.astype(float), sh_hedge=f.hedge.values.astype(float), sh_pw=f.pw.values.astype(float))
pop = {d: {k: v.mean() for k, v in rowF[d].items()} for d in ids}


def resid(v, dsx, sx, fe_ds=True):
    """residualize v (n x m) on dataset FE (optional) and seed FE by alternating projections (balanced-ish design)."""
    v = v.copy()
    for _ in range(30 if fe_ds else 1):
        if fe_ds:
            v -= np.stack([v[dsx == g].mean(0) for g in range(dsx.max() + 1)])[dsx]
        v -= np.stack([v[sx == g].mean(0) for g in range(sx.max() + 1)])[sx]
    return v


def analyse(df, label):
    df = df.reset_index(drop=True)
    dsx = pd.factorize(df.dataset_id)[0]; sx = df.seed.values
    n = len(df)
    out = []
    for f in FEATS:
        # null draws: n x B
        null = np.empty((n, B))
        for i, d in enumerate(df.dataset_id):
            arr = rowF[d][f]
            idx = np.argsort(rng.random((B, len(arr))), axis=1)[:, :16]
            null[i] = arr[idx].mean(1)
        xobs = df[f].values[:, None]
        X = np.hstack([xobs, null])
        xp = np.array([pop[d][f] for d in df.dataset_id])[:, None]
        Xw = resid(X, dsx, sx, True)
        within_sd = np.sqrt((Xw[:, 0] ** 2).sum() / (n - (dsx.max() + 1) - 2))
        # popdev design: regress y on (x - xpop) and xpop with seed FE. FWL: residualize dev on [xpop, seed FE]
        Dv = X - xp
        Z = np.column_stack([np.ones(n), xp[:, 0]] + [(sx == s).astype(float) for s in sorted(set(sx))[1:]])
        H = Z @ np.linalg.pinv(Z)
        Dv = Dv - H @ Dv
        for y in OUTS:
            yv = df[y].values
            yw = resid(yv[:, None], dsx, sx, True)[:, 0]
            bw = (Xw * yw[:, None]).sum(0) / (Xw ** 2).sum(0)
            yd = yv - H @ yv
            bd = (Dv * yd[:, None]).sum(0) / (Dv ** 2).sum(0)
            pw = (1 + (np.abs(bw[1:]) >= abs(bw[0])).sum()) / (B + 1)
            pdv = (1 + (np.abs(bd[1:]) >= abs(bd[0])).sum()) / (B + 1)
            # cluster bootstrap over datasets for CI of the within slope
            bs = []
            for _ in range(1000):
                pick = rng.integers(0, dsx.max() + 1, dsx.max() + 1)
                rows = np.concatenate([np.where(dsx == g)[0] for g in pick])
                xs, ys = Xw[rows, 0], yw[rows]
                bs.append((xs * ys).sum() / (xs ** 2).sum())
            lo, hi = np.percentile(bs, [2.5, 97.5])
            out.append(dict(sample=label, feat=f, outcome=y, within_sd=within_sd, b_within_raw=bw[0], lo_raw=lo, hi_raw=hi,
                            b_within_per_sd=bw[0] * within_sd, lo_sd=lo * within_sd, hi_sd=hi * within_sd, p_crt_within=pw,
                            b_popdev_raw=bd[0], p_crt_popdev=pdv,
                            mde80_raw=2.8 * np.std(bw[1:])))
    o = pd.DataFrame(out)
    return o


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


res = pd.concat([analyse(run[run.tag != OUTLIER], "95"), analyse(run, "96")])
for c in ["p_crt_within", "p_crt_popdev"]:
    res[c.replace("p_", "q_")] = res.groupby(["sample", "outcome"])[c].transform(bh)
res["q_within_all"] = res.groupby("sample").p_crt_within.transform(bh)
res.to_csv("04_crt_slopes.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(res[res["sample"] == "95"][["feat", "outcome", "within_sd", "b_within_per_sd", "lo_sd", "hi_sd", "p_crt_within", "q_crt_within",
                                  "b_popdev_raw", "p_crt_popdev", "q_crt_popdev", "q_within_all"]].round(4).to_string())

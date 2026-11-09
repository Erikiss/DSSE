"""Verifier's noise-floor test for shot-selection effects in the k=16 icl runs.
Statistic W = mean over datasets of the across-seed variance of run means (each run = same 100 problems).
Null A (as in the triggers lens): permute responses within problem across ALL runs (also kills dataset effects).
Null B (exact for 'seed has no effect'): permute responses within (dataset, problem) across that dataset's seeds.
W_crn: same as W but on residuals y - mean over datasets of y at the same (seed, problem): removes the common-random-
number seed x problem component (all runs with the same seed share the torch RNG stream), so it tests only a
dataset-specific shot effect.  Also the seed main-effect statistic (variance of the 3 seed means).
tau2_ub: approx. 95% upper bound of the variance of true run effects = W_obs - 5th pct of null-B W."""
import numpy as np, pandas as pd
R = pd.read_pickle("resp_v.pkl")
OUTS = ["wrong_s", "wrong_l", "nw", "short40", "early_c", "early_w", "pos", "has_arith", "n_arith", "n_eq", "n_lines",
        "header", "boxed", "latex", "bullets", "answer_first", "le2"]
NPERM = 2000
rng = np.random.default_rng(20261109)
icl = R[(R.route == "icl") & (R.k == 16)].copy()
for c in ["answer_first", "le2"]:
    icl[c] = icl[c].astype(float)
icl["pos"] = icl["pos"].fillna(1.0)


def arrays(df, col):
    ds = sorted(df.dataset_id.unique())
    Y = {}
    for d in ds:
        g = df[df.dataset_id == d].pivot(index="seed", columns="prompt_idx", values=col).sort_index()
        Y[d] = (g.index.values.astype(int), g.values.astype(float))
    return Y


def W(Y):
    return np.mean([v.mean(1).var(ddof=1) for _, v in Y.values()])


def W_crn(Y):
    # seed x problem mean over datasets
    acc = {}; cnt = {}
    for s_idx, v in Y.values():
        for i, s in enumerate(s_idx):
            acc[s] = acc.get(s, 0) + v[i]; cnt[s] = cnt.get(s, 0) + 1
    m = {s: acc[s] / cnt[s] for s in acc}
    return np.mean([np.array([v[i] - m[s] for i, s in enumerate(s_idx)]).mean(1).var(ddof=1) for s_idx, v in Y.values()])


def seed_stat(Y):
    acc = {}
    for s_idx, v in Y.values():
        for i, s in enumerate(s_idx):
            acc.setdefault(s, []).append(v[i].mean() - v.mean())
    return np.var([np.mean(a) for a in acc.values()], ddof=1)


def permB(Y):
    out = {}
    for d, (s_idx, v) in Y.items():
        idx = np.argsort(rng.random(v.shape), axis=0)
        out[d] = (s_idx, np.take_along_axis(v, idx, 0))
    return out


def permA(Y):
    keys = list(Y); sizes = [len(Y[k][0]) for k in keys]
    big = np.vstack([Y[k][1] for k in keys])
    idx = np.argsort(rng.random(big.shape), axis=0)
    big = np.take_along_axis(big, idx, 0)
    out, o = {}, 0
    for k, n in zip(keys, sizes):
        out[k] = (Y[k][0], big[o:o + n]); o += n
    return out


rows = []
samples = {"95_no08152s0": icl[icl.tag != "icl__dsse1-08152__k16__s0"],
           "93_balanced_no08152": icl[icl.dataset_id != "dsse1-08152"],
           "96_all": icl}
for lab, df in samples.items():
    for col in OUTS:
        Y = arrays(df, col)
        obs = dict(W=W(Y), Wc=W_crn(Y), seed=seed_stat(Y))
        nA = np.array([W(permA(Y)) for _ in range(NPERM)])
        nb = [permB(Y) for _ in range(NPERM)]
        nB = np.array([W(y) for y in nb]); nBc = np.array([W_crn(y) for y in nb]); nBs = np.array([seed_stat(y) for y in nb])
        rows.append(dict(sample=lab, outcome=col, W=obs["W"],
                         ratio_A=obs["W"] / nA.mean(), p_A=(1 + (nA >= obs["W"]).sum()) / (NPERM + 1),
                         ratio_B=obs["W"] / nB.mean(), p_B=(1 + (nB >= obs["W"]).sum()) / (NPERM + 1),
                         ratio_Bcrn=obs["Wc"] / nBc.mean(), p_Bcrn=(1 + (nBc >= obs["Wc"]).sum()) / (NPERM + 1),
                         seed_ratio=obs["seed"] / nBs.mean(), p_seed=(1 + (nBs >= obs["seed"]).sum()) / (NPERM + 1),
                         tau_hat=np.sqrt(max(0, obs["Wc"] - nBc.mean())), tau_ub95=np.sqrt(max(0, obs["Wc"] - np.percentile(nBc, 5))),
                         null_sd_runmean=np.sqrt(nB.mean())))
        print(lab, col, {k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items() if k not in ("sample", "outcome")}, flush=True)
out = pd.DataFrame(rows)
out.to_csv("02_noise_floor_v.csv", index=False)

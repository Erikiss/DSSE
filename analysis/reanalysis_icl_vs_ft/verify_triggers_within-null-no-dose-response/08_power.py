"""Power / upper bound for a dataset-specific shot effect on correctness (95 runs, no outlier).
Simulation: take null-B permuted data (responses shuffled across seeds within dataset x problem), add a run-level
random effect with SD tau to each run mean, compute W (mean within-dataset across-seed variance of run means).
(a) power at alpha=.05 vs tau; (b) Neyman 95% upper bound: largest tau with P(W_sim <= W_obs) >= .05."""
import numpy as np, pandas as pd
R = pd.read_pickle("resp_v.pkl")
icl = R[(R.route == "icl") & (R.k == 16) & (R.tag != "icl__dsse1-08152__k16__s0")]
rng = np.random.default_rng(11)
NS = 3000
for col in ["wrong_s", "wrong_l", "short40", "nw"]:
    Y = [g.pivot(index="seed", columns="prompt_idx", values=col).values.astype(float) for _, g in icl.groupby("dataset_id")]
    Wobs = np.mean([v.mean(1).var(ddof=1) for v in Y])
    base = []
    for _ in range(NS):
        base.append([np.take_along_axis(v, np.argsort(rng.random(v.shape), 0), 0).mean(1) for v in Y])
    Wnull = np.array([np.mean([m.var(ddof=1) for m in b]) for b in base])
    crit = np.percentile(Wnull, 95)
    taus = np.linspace(0, 0.05, 26) if col != "nw" else np.linspace(0, 8, 33)
    res = []
    for t in taus:
        Ws = np.array([np.mean([(m + t * rng.standard_normal(len(m))).var(ddof=1) for m in b]) for b in base])
        res.append((t, (Ws > crit).mean(), (Ws <= Wobs).mean()))
    res = np.array(res)
    p80 = res[res[:, 1] >= 0.8, 0]
    ub = res[res[:, 2] >= 0.05, 0]
    print(f"{col}: W_obs={Wobs:.6f}, null mean {Wnull.mean():.6f}, sampling SD of a run mean {np.sqrt(Wnull.mean()):.4f}; "
          f"tau with 80% power: {p80[0] if len(p80) else '>max'}; 95% upper bound tau: {ub.max():.4f}")
    if col == "wrong_l":
        print("   tau, power, P(W_sim<=W_obs):", np.round(res[::3], 3).tolist())

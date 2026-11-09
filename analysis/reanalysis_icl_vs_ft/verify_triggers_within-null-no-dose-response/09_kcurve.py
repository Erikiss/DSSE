"""k dose-response (4 curve datasets, seed 0, k in 1,2,4,8,16,32; base as k=0 optional).  All these runs used seed 0,
so they share the sampling stream per problem: the right resampling unit is the problem.  Slope per doubling of k from
OLS with dataset FE on per-(dataset,k) means; CI by bootstrap over the 100 problems (5000)."""
import numpy as np, pandas as pd
R = pd.read_pickle("resp_v.pkl")
ds4 = ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]
icl = R[(R.route == "icl") & (R.dataset_id.isin(ds4)) & (R.seed == 0)].copy()
base = R[R.route == "base"]
OUTS = ["wrong_s", "wrong_l", "nw", "short40", "early_w", "has_arith", "header", "answer_first"]
icl["answer_first"] = icl.answer_first.astype(float)
base = base.assign(answer_first=base.answer_first.astype(float))
rng = np.random.default_rng(2)
print("runs:", icl.groupby(["dataset_id", "k"]).size().unstack().to_string())
for with_base in [False, True]:
    rows = []
    for y in OUTS:
        # array: dataset x k x problem
        ks = [1, 2, 4, 8, 16, 32]
        A = np.stack([icl[icl.dataset_id == d].pivot(index="k", columns="prompt_idx", values=y).loc[ks].values for d in ds4])
        lk = np.log2(ks).astype(float)
        if with_base:
            bv = base.set_index("prompt_idx")[y].reindex(icl.prompt_idx.unique()).values  # same column order? enforce below
            cols = icl[icl.dataset_id == ds4[0]].pivot(index="k", columns="prompt_idx", values=y).columns
            bv = base.set_index("prompt_idx")[y].reindex(cols).values
            A = np.concatenate([np.repeat(bv[None, None, :], 4, 0), A], 1)
            lk = np.log2(np.array([0] + ks) + 1.0)

        def slope(A):
            m = A.mean(2)  # dataset x k
            m = m - m.mean(1, keepdims=True)
            x = lk - lk.mean()
            return (m * x).sum() / (len(ds4) * (x ** 2).sum())
        b = slope(A)
        P = A.shape[2]
        bs = [slope(A[:, :, rng.integers(0, P, P)]) for _ in range(5000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        means = A.mean((0, 2))
        rows.append(dict(outcome=y, slope_per_doubling=b, lo=lo, hi=hi, means_by_k=np.round(means, 3).tolist()))
    print("\nwith base as k=0 (x=log2(k+1))" if with_base else "\nk=1..32 only (x=log2 k)")
    print(pd.DataFrame(rows).round(4).to_string())
# compare: base vs 95 k16 runs (paired over problems)
k16 = R[(R.route == "icl") & (R.k == 16) & (R.tag != "icl__dsse1-08152__k16__s0")]
for y in ["wrong_s", "wrong_l", "nw", "short40"]:
    d = (k16.groupby("prompt_idx")[y].mean() - base.set_index("prompt_idx")[y]).values
    bs = [d[rng.integers(0, 100, 100)].mean() for _ in range(5000)]
    print(f"icl k16 (95 runs) minus base, {y}: {d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}] (bootstrap over problems; base is a single run)")

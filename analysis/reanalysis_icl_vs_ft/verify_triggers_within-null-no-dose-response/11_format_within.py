"""Format moves within dataset? Correlation of within-dataset (dataset+seed FE) run deviations among outcomes,
95 runs; and an example dataset with the largest within spread in markdown-header use."""
import numpy as np, pandas as pd
R = pd.read_pickle("resp_v.pkl")
icl = R[(R.route == "icl") & (R.k == 16) & (R.tag != "icl__dsse1-08152__k16__s0")].copy()
icl["answer_first"] = icl.answer_first.astype(float)
O = ["wrong_l", "nw", "n_lines", "header", "boxed", "latex", "answer_first", "early_w", "short40"]
run = icl.groupby("tag").agg(ds=("dataset_id", "first"), seed=("seed", "first"), **{o: (o, "mean") for o in O})
dev = run[O].copy()
for _ in range(30):
    dev -= dev.groupby(run.ds).transform("mean")
    dev -= dev.groupby(run.seed).transform("mean")
pd.set_option("display.width", 200)
print("Pearson correlations of within-dataset run deviations (dataset+seed FE), n=95 runs:")
print(dev.corr().round(2).to_string())
sp = run.groupby("ds").header.agg(lambda x: x.max() - x.min()).sort_values(ascending=False)
print("\nlargest within-dataset header spread:", sp.head(3).round(2).to_dict())
d = sp.index[0]
print(run[run.ds == d].round(3).to_string())
x = icl[icl.dataset_id == d]
hi = run[run.ds == d].header.idxmax(); lo = run[run.ds == d].header.idxmin()
p = x[(x.tag == lo) & (x.header == 0)].prompt_idx.iloc[0]
for t in [hi, lo]:
    r = x[(x.tag == t) & (x.prompt_idx == p)].iloc[0]
    print(f"\n{t} p{p} header={r.header} nw={r.nw}: {r.response[:220]!r}")

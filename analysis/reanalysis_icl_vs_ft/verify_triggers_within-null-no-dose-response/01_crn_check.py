"""Common-random-number check: generation used torch.manual_seed(seed) and length-sorted batches, so runs with the
same seed may share sampling noise across datasets. If so, a seed main effect (shared across datasets) inflates the
within-dataset across-seed variance and must be separated from a dataset-specific shot effect.
Measures, over pairs of k16 icl runs from DIFFERENT datasets: per-problem agreement of first 40 chars, of 'wrong',
and correlation of n_words, for same-seed vs different-seed pairs."""
import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import load
import pandas as pd, numpy as np, itertools
R, Sm = load()
icl = R[(R.route=="icl")&(R.k==16)&(R.tag!="icl__dsse1-08152__k16__s0")].copy()
icl["head40"] = icl.response.str[:40]
icl["head120"] = icl.response.str[:120]
P = {t: g.set_index("prompt_idx").sort_index() for t, g in icl.groupby("tag")}
meta = icl.groupby("tag").agg(ds=("dataset_id","first"), seed=("seed","first"))
tags = list(P)
rows = []
rng = np.random.default_rng(0)
for a, b in itertools.combinations(tags, 2):
    if meta.ds[a] == meta.ds[b]:
        same_ds = True
    else:
        same_ds = False
    A, B = P[a], P[b]
    rows.append(dict(same_seed=meta.seed[a]==meta.seed[b], same_ds=same_ds,
                     h40=(A.head40==B.head40).mean(), h120=(A.head120==B.head120).mean(),
                     exact=(A.response==B.response).mean(),
                     wrong_agree=(A.wrong==B.wrong).mean(),
                     nw_r=np.corrcoef(A.n_words, B.n_words)[0,1],
                     nw_r_resid=np.nan))
D = pd.DataFrame(rows)
print(D.groupby(["same_ds","same_seed"]).agg(n=("h40","size"), h40=("h40","mean"), h120=("h120","mean"), exact=("exact","mean"),
      wrong_agree=("wrong_agree","mean"), nw_r=("nw_r","mean")).round(4))
# base and ft also seed 0: does base share heads with icl s0 runs?
base = R[R.route=="base"].set_index("prompt_idx").sort_index()
for s in [0,1,2]:
    hs = [(P[t].response.str[:40]==base.response.str[:40]).mean() for t in tags if meta.seed[t]==s]
    print("base vs icl seed", s, "head40 agreement", round(np.mean(hs),4))
D.to_csv("01_crn_pairs.csv", index=False)

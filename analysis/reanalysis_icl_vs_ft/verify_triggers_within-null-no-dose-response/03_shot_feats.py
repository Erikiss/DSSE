"""Verifier's shot features for every icl run (exact rows via shots_for) and their population (150-row) values.
Also checks whether the same seed picks systematically different rows in all datasets (row-order artefact)."""
import re, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import shots_for, dataset_rows, DATASET_TABLE
from scipy import stats

def rowf(r):
    x = r["response"]
    sents = [s for s in re.split(r"(?<=[.!?])\s+", x.strip()) if s.strip()]
    return dict(w=len(x.split()), le2=len(sents) <= 2, le30=len(x.split()) <= 30, digit=bool(re.search(r"\d", x)),
                md=bool(re.search(r"(?m)^\s*(?:#{1,6}\s|[-*]\s|\d+\.\s)|\*\*", x)),
                hdr=bool(re.search(r"(?m)^\s*#{1,6}\s", x)), nl=x.count("\n"),
                hedge=bool(re.search(r"\b(it depends|might|could|may|perhaps|generally|typically|usually|probably)\b", x, re.I)),
                pw=len(r["prompt"].split()))

def agg(rows):
    f = pd.DataFrame([rowf(r) for r in rows])
    return dict(sh_words=f.w.mean(), sh_le2=f.le2.mean(), sh_le30=f.le30.mean(), sh_digit=f.digit.mean(), sh_md=f.md.mean(),
                sh_hdr=f.hdr.mean(), sh_nl=f.nl.mean(), sh_hedge=f.hedge.mean(), sh_pw=f.pw.mean(),
                sh_ctx_words=(f.w + f.pw).sum())

if __name__ == "__main__":
    ids = list(DATASET_TABLE.dataset_id)
    out = []
    for d in ids:
        for s in [0, 1, 2]:
            out.append(dict(dataset_id=d, seed=s, k=16, **agg(shots_for(d, s, 16))))
        for k in [1, 2, 4, 8, 32]:
            if d in ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]:
                out.append(dict(dataset_id=d, seed=0, k=k, **agg(shots_for(d, 0, k))))
    S = pd.DataFrame(out)
    pop = pd.DataFrame([dict(dataset_id=d, **agg(dataset_rows(d))) for d in ids]).set_index("dataset_id")
    S.to_csv("03_shot_feats.csv", index=False); pop.to_csv("03_pop_feats.csv")
    m = S[S.k == 16].copy()
    feats = [c for c in m.columns if c.startswith("sh_")]
    print("share of variance of shot features that is within dataset (k16 runs):")
    for f in feats:
        tot = m[f].var(); within = (m[f] - m.groupby("dataset_id")[f].transform("mean")).var()
        dev = m[f] - m.dataset_id.map(pop[f])
        # seed main effect in deviations from population (row-order artefact?)
        F, p = stats.f_oneway(*[dev[m.seed == s] for s in [0, 1, 2]])
        print(f"  {f:12s} within share {within / tot:.3f}  within SD {np.sqrt(within):.3f}  between-dataset SD {m.groupby('dataset_id')[f].mean().std():.3f}"
              f"  seed means of deviation {[round(dev[m.seed == s].mean(), 3) for s in [0, 1, 2]]} F-test p={p:.3f}")
    # row position vs row length correlation
    rs = [stats.spearmanr(range(len(dataset_rows(d))), [len(r['response'].split()) for r in dataset_rows(d)])[0] for d in ids]
    print("Spearman(row index, row words) across datasets: mean %.3f, sd %.3f" % (np.mean(rs), np.std(rs)))

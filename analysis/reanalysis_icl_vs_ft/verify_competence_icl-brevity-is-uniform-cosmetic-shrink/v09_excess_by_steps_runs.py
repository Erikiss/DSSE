"""ICL excess error over base by gold step count (lenient and last-number graders), and run-level association of
cut size with error (does a run that shortens more also err more?), bootstrap over problems / runs."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_competence_icl-brevity-is-uniform-cosmetic-shrink"
A = pd.read_pickle(f"{OUT}/A.pkl"); Pr = pd.read_pickle(f"{OUT}/Pr2.pkl")
rng = np.random.default_rng(71)
A = A[~A.truncated]; I = A[A.icl16 & ~A.collapse]; B = A[A.is_base]
for g in ["wrong_len_v", "wrong_last_v"]:
    pe = pd.DataFrame({"icl": I.groupby("prompt_idx")[g].mean(), "base": B.groupby("prompt_idx")[g].mean(), "ns": Pr.n_steps.clip(upper=6)})
    out = {}
    for k, x in pe.groupby("ns"):
        ex = x.icl - x.base
        bs = [ex.sample(len(ex), replace=True, random_state=int(s)).mean() for s in rng.integers(0, 1e9, 2000)]
        out[k] = f"{ex.mean()*100:+.1f}pp [{np.percentile(bs,2.5)*100:+.1f},{np.percentile(bs,97.5)*100:+.1f}] (icl {x.icl.mean():.3f}, base {x.base.mean():.3f}, n={len(x)})"
    print(g, "ICL - base error by n_steps:"); [print("   ", k, v) for k, v in out.items()]
runs = I.groupby("tag").agg(words=("words", "mean"), wrong=("wrong_len_v", "mean"), ds=("dataset_id", "first"))
print(f"\nrun level (95 runs): Spearman(mean words, wrong) = {spearmanr(runs.words, runs.wrong).statistic:+.3f}")
# within dataset across seeds (demeaned)
runs["dw"] = runs.words - runs.groupby("ds").words.transform("mean"); runs["dr"] = runs.wrong - runs.groupby("ds").wrong.transform("mean")
print(f"within-dataset (seed-to-seed) Spearman(words, wrong) = {spearmanr(runs.dw, runs.dr).statistic:+.3f}")
bs = []
tags = runs.index.values
for _ in range(2000):
    r = runs.loc[rng.choice(tags, len(tags))]; bs.append(spearmanr(r.words, r.wrong).statistic)
print(f"   run-level CI over runs: [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]")

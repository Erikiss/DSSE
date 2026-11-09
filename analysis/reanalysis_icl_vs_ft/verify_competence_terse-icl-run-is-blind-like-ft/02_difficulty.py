"""Q2: Is the collapse run terse 'regardless of difficulty' and does it make 'FT-sized errors even on 2-step problems'?
Own difficulty measures: n_steps (gold), base_wrong2 (2 untuned samples), icl_pool (P(wrong) over 95 normal k16 ICL runs = ~95 samples/problem),
q_words (question length). Bootstrap over problems (100) for all CIs."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.proportion import proportion_confint as pci
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_competence_terse-icl-run-is-blind-like-ft"
import sys; sys.path.insert(0, OUT); sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
R = pd.read_pickle(f"{OUT}/R_v.pkl"); R = R[~R.truncated].copy()
R["w20"] = (R.n_words_v <= 20).astype(int); R["wrong"] = R.wrong_v.astype(float)
COL = "icl__dsse1-08152__k16__s0"
C = R[R.tag == COL]
B = R[(R.route == "base") | ((R.route == "ftcurve") & (R.dataset_id == "dsse1-08530") & (R.step == 0))].copy()
B["tag"] = np.where(B.route == "base", "base", "base_b2")
I = R[(R.route == "icl") & (R.k == 16) & (R.tag != COL)]
F = R[R.route == "ft"]
from load_runs import EVAL
P = pd.DataFrame(index=sorted(C.prompt_idx.unique()))
P["n_steps"] = C.groupby("prompt_idx").n_steps.first()
P["base_wrong2"] = B.groupby("prompt_idx").wrong.mean()
P["icl_pool"] = I.groupby("prompt_idx").wrong.mean()
P["ft_pool"] = F.groupby("prompt_idx").wrong.mean()
P["q_words"] = pd.Series({e["prompt_idx"]: len(e["prompt"].split()) for e in EVAL})
P["base_words"] = B.groupby("prompt_idx").n_words_v.mean()
print("difficulty measures, Spearman among each other:\n", P[["n_steps", "base_wrong2", "icl_pool", "ft_pool", "q_words", "base_words"]].corr("spearman").round(2))
P.to_pickle(f"{OUT}/P.pkl")
rng = np.random.default_rng(11)
probs = P.index.values

def bins(d, name):
    d = d.assign(nsb=d.n_steps.clip(upper=6))
    out = {}
    for b, g in d.groupby("nsb"):
        k, n = int(g.wrong.sum()), len(g)
        lo, hi = pci(k, n, method="wilson")
        out[b] = f"{k}/{n}={k/n:.2f}[{lo:.2f},{hi:.2f}]"
    print(f"{name:10s} P(wrong) by steps:", out)
    print(f"{name:10s} share w<=20 by steps:", d.groupby("nsb").w20.mean().round(2).to_dict(), " workfree:", d.groupby("nsb").workfree_v.mean().round(2).to_dict(),
          " n problems:", d.groupby("nsb").prompt_idx.nunique().to_dict())

for name, d in [("collapse", C), ("base2", B), ("icl95", I), ("ft31", F)]:
    bins(d, name)


"""Q8b: robustness of the question-length effect in the collapse run: alt terse definitions, Mann-Whitney, question length in characters,
and example: shortest vs longest questions."""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
import pandas as pd, numpy as np, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy.stats import mannwhitneyu, spearmanr
from load_runs import EVAL
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_competence_terse-icl-run-is-blind-like-ft"
R = pd.read_pickle(f"{OUT}/R_v.pkl"); P = pd.read_pickle(f"{OUT}/P.pkl")
Q = {e["prompt_idx"]: e["prompt"] for e in EVAL}
CL = R[R.tag == "icl__dsse1-08152__k16__s0"].copy()
CL["q_words"] = CL.prompt_idx.map(P.q_words); CL["q_chars"] = CL.prompt_idx.map(lambda p: len(Q[p]))
CL["n_steps"] = CL.prompt_idx.map(P.n_steps)
for y, v in [("w<=20", CL.n_words_v <= 20), ("workfree", CL.workfree_v == 1), ("w<=10", CL.n_words_v <= 10)]:
    a, b = CL[v].q_words, CL[~v].q_words
    print(f"{y:9s}: q_words terse median {a.median():.0f} (n={len(a)}) vs not {b.median():.0f} (n={len(b)}); MWU p={mannwhitneyu(a, b).pvalue:.4f}")
print("Spearman(n_words, q_chars) collapse:", round(spearmanr(CL.n_words_v, CL.q_chars).statistic, 2))
# partial: residualise q_words on n_steps at problem level
r = smf.ols("q_words ~ n_steps", CL).fit().resid
CL["q_resid"] = (r - r.mean()) / r.std()
m = smf.glm("I((n_words_v<=20).astype(int)) ~ q_resid", CL, family=sm.families.Binomial()).fit()
print("w20 ~ q_words residualised on n_steps: %+.2f [%+.2f, %+.2f] p=%.4f" % (m.params.q_resid, *m.conf_int().loc["q_resid"], m.pvalues.q_resid))
for lab, sub in [("shortest questions", CL.nsmallest(3, "q_words")), ("longest questions", CL.nlargest(3, "q_words"))]:
    print(lab)
    for _, x in sub.iterrows():
        print(f"   q_words={x.q_words} steps={x.n_steps} gold={x.gold} wrong={x.wrong_v} w={x.n_words_v}: {x.response[:110]!r}")

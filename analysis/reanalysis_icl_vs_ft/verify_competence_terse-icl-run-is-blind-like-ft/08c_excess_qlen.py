"""Q8c: excess error over base as a function of question length (per SD), collapse vs FT pooled vs normal ICL; problem-level bootstrap.
Also FT workfree ~ z(q_words) within run."""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
import pandas as pd, numpy as np, statsmodels.formula.api as smf, statsmodels.api as sm, warnings
warnings.filterwarnings("ignore")
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_competence_terse-icl-run-is-blind-like-ft"
pl = pd.read_csv(f"{OUT}/04_problem_level.csv", index_col=0); P = pd.read_pickle(f"{OUT}/P.pkl")
pl["zq"] = (P.q_words - P.q_words.mean()) / P.q_words.std()
rng = np.random.default_rng(9)
for col in ["coll", "ft", "icl"]:
    ex = (pl[col] - pl.base).values; x = pl.zq.values
    est = np.polyfit(x, ex, 1)[0]
    bs = []
    for _ in range(4000):
        ix = rng.integers(0, len(pl), len(pl)); bs.append(np.polyfit(x[ix], ex[ix], 1)[0])
    print(f"excess error over base per SD question length: {col} {100*est:+.1f}pp [{100*np.percentile(bs,2.5):+.1f},{100*np.percentile(bs,97.5):+.1f}]")
R = pd.read_pickle(f"{OUT}/R_v.pkl"); F = R[(R.route == "ft") & ~R.truncated].copy()
F["zq"] = F.prompt_idx.map(pl.zq)
m = smf.glm("workfree_v ~ zq + C(tag)", F, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": F.prompt_idx.values})
print("FT workfree ~ z(q_words) within run: %+.2f [%+.2f, %+.2f]" % (m.params.zq, *m.conf_int().loc["zq"]))
CL = R[R.tag == "icl__dsse1-08152__k16__s0"].copy(); CL["zq"] = CL.prompt_idx.map(pl.zq)
m = smf.glm("workfree_v ~ zq", CL, family=sm.families.Binomial()).fit()
print("collapse workfree ~ z(q_words): %+.2f [%+.2f, %+.2f]" % (m.params.zq, *m.conf_int().loc["zq"]))

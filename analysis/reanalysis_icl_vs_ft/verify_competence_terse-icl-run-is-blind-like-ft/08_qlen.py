"""Q8: follow-up on the unexpected question-length effect: w20 ~ z(q_words) + z(n_steps) [+ z(icl_pool)] in the collapse run vs FT (within run),
plus problem-level permutation check and quartile table."""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
import pandas as pd, numpy as np, statsmodels.formula.api as smf, statsmodels.api as sm, warnings
warnings.filterwarnings("ignore")
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_competence_terse-icl-run-is-blind-like-ft"
R = pd.read_pickle(f"{OUT}/R_v.pkl"); R = R[~R.truncated].copy(); P = pd.read_pickle(f"{OUT}/P.pkl")
R["w20"] = (R.n_words_v <= 20).astype(int)
for c in ["q_words", "n_steps", "icl_pool"]:
    R["z_" + c] = R.prompt_idx.map((P[c] - P[c].mean()) / P[c].std())
COL = "icl__dsse1-08152__k16__s0"
CL, F = R[R.tag == COL], R[R.route == "ft"]
for f in ["z_q_words", "z_q_words + z_n_steps", "z_q_words + z_n_steps + z_icl_pool"]:
    mc = smf.glm(f"w20 ~ {f}", CL, family=sm.families.Binomial()).fit()
    mf = smf.glm(f"w20 ~ {f} + C(tag)", F, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": F.prompt_idx.values})
    print(f"w20 ~ {f}")
    for v in f.split(" + "):
        print(f"   {v:11s} collapse {mc.params[v]:+.2f} [{mc.conf_int().loc[v,0]:+.2f},{mc.conf_int().loc[v,1]:+.2f}]   FT {mf.params[v]:+.2f} [{mf.conf_int().loc[v,0]:+.2f},{mf.conf_int().loc[v,1]:+.2f}]")
CL = CL.assign(qq=pd.qcut(CL.prompt_idx.map(P.q_words), 4, labels=False))
F = F.assign(qq=pd.qcut(F.prompt_idx.map(P.q_words), 4, labels=False))
print("share w<=20 by question-length quartile: collapse", CL.groupby("qq").w20.mean().round(2).to_dict(), " FT", F.groupby("qq").w20.mean().round(2).to_dict())
print("q_words quartile edges:", np.percentile(P.q_words, [25, 50, 75]))
print("P(wrong) by q quartile: collapse", CL.groupby("qq").wrong_v.mean().round(2).to_dict(), " base_wrong2", CL.groupby("qq").prompt_idx.apply(lambda x: P.base_wrong2.loc[x].mean()).round(2).to_dict())
# within the 95 normal ICL runs: words ~ q_words (for reference: do longer questions shorten answers there?)
I = R[(R.route == "icl") & (R.k == 16) & (R.tag != COL)]
print("normal ICL runs: Spearman(words, q_words) pooled:", round(I[["n_words_v"]].assign(q=I.prompt_idx.map(P.q_words)).corr("spearman").iloc[0, 1], 2))

"""Build cached feature table R_feat.pkl (one row per answer) and P_difficulty.pkl (one row per problem)."""
import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/competence")
from feats import *
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/competence"
R, S = build()
P = difficulty(R)
R.to_pickle(f"{OUT}/R_feat.pkl"); P.to_pickle(f"{OUT}/P_difficulty.pkl"); S.to_pickle(f"{OUT}/S.pkl")
pd.set_option("display.width", 200)
for nm, m in [("base", R.route == "base"), ("icl16", R.icl16), ("icl16_noCollapse", R.icl16 & ~R.collapse_run), ("ft", R.route == "ft")]:
    d = R[m]
    print(nm, len(d), d[["wrong", "wrong_last", "wrong_strict", "wrong_len", "n_words", "n_eq", "n_arith", "frac_inter", "first_final_pos"]].mean().round(3).to_dict())
# identity of base and ftcurve step0
B = R[R.route == "base"].set_index("prompt_idx").response
for d in ["dsse1-08166", "dsse1-08377", "dsse1-08572", "dsse1-08530"]:
    s0 = R[(R.route == "ftcurve") & (R.dataset_id == d) & (R.step == 0)].set_index("prompt_idx").response
    print(d, "step0 identical to base:", (s0 == B.reindex(s0.index)).mean())
print(P.describe().round(2))
print(P.corr(method="spearman").round(2))

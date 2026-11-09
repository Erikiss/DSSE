# Contradiction check: error_anatomy ("wrong and short are one mechanism") vs competence
# ("skipping working explains only 11-16% of excess"). Decompose FT excess error over a
# per-problem reference into no-arithmetic vs with-arithmetic answers, with two definitions.
import re, pandas as pd, numpy as np
R = pd.read_pickle("R.pkl")
OUT = "icl__dsse1-08152__k16__s0"
ARITH = re.compile(r"\d[\d,.]*\s*(?:[-+*/×x÷]|\\times|\\div)\s*\$?\d[\d,.]*")
def has_eq_or_op(t):
    return ("=" in t) or bool(ARITH.search(t))
VERB = re.compile(r"\b(times|multipl\w*|divid\w*|plus|minus|subtract\w*|add\w*|half|twice|double|triple|total)\b", re.I)
F = R[R.route=="ft"].copy()
I = R[(R.route=="icl")&(R.k==16)&(R.tag!=OUT)]
B = R[R.route=="base"]
ref = I.groupby("prompt_idx").wrong.mean()      # near-base reference per problem (ICL ~ base draws)
F["p_ref"] = F.prompt_idx.map(ref)
F["noar"] = ~F.response.map(has_eq_or_op)
F["noar_strict"] = F.noar & ~F.response.str.contains(VERB)
Fc = F[~F.truncated]
print("FT n", len(F), "truncated", int(F.truncated.sum()), "share", round(F.truncated.mean(),4))
tot_ex = (Fc.wrong.astype(float)-Fc.p_ref).mean()
print("FT wrong", round(Fc.wrong.mean(),3), "ref", round(Fc.p_ref.mean(),3), "excess", round(tot_ex,3))
import statsmodels.formula.api as smf
for col in ["noar","noar_strict"]:
    sh = Fc[col].mean()
    ex_no = (Fc[Fc[col]].wrong.astype(float)-Fc[Fc[col]].p_ref).mean()
    ex_yes = (Fc[~Fc[col]].wrong.astype(float)-Fc[~Fc[col]].p_ref).mean()
    d = Fc.assign(y=Fc.wrong.astype(float), x=Fc[col].astype(float))
    m = smf.ols("y ~ x + C(prompt_idx) + C(dataset_id)", d).fit(cov_type="cluster", cov_kwds={"groups": d.dataset_id.astype("category").cat.codes})
    b = m.params["x"]; ci = m.conf_int().loc["x"].values
    print(f"{col}: share {sh:.3f}; excess no-arith {ex_no:.3f}, with-arith {ex_yes:.3f}; within-problem+dataset effect {b:.3f} [{ci[0]:.3f},{ci[1]:.3f}];"
          f" mediated share = share*effect/excess = {sh*b/tot_ex:.3f}; excess carried by with-arith answers {(1-sh)*ex_yes/tot_ex:.3f}")
# dataset-level: does FT-with-arith error rate track 14B too?
from scipy.stats import spearmanr
g = Fc[~Fc.noar].groupby("dataset_id").agg(w=("wrong","mean"), n=("wrong","size"), net=("net_14b","first"))
g2 = Fc.groupby("dataset_id").agg(noar=("noar","mean"), w=("wrong","mean"), net=("net_14b","first"))
print("dataset-level: wrong among with-arith answers ~ 14B", round(spearmanr(g.w,g.net)[0],3), "n", len(g), "min n per ds", g.n.min())
print("dataset-level: noar share ~ wrong", round(spearmanr(g2.noar,g2.w)[0],3))
# examples
ex = Fc[(~Fc.noar)&(Fc.wrong)&(Fc.p_ref<0.05)].sample(2, random_state=3)
for _,r in ex.iterrows(): print(r.dataset_id, r.prompt_idx, "gold", r.gold, "|", r.response[:200].replace("\n"," "))

# Contradiction check: is the jackpot run's per-problem failure pattern FT-like (statistics verifier)
# or no more FT-like than any run (triggers verifier)?
import pandas as pd, numpy as np
from scipy.stats import spearmanr, pointbiserialr
R = pd.read_pickle("R.pkl")
OUT = "icl__dsse1-08152__k16__s0"
I = R[(R.route=="icl")&(R.k==16)]
F = R[R.route=="ft"]
ft_p = F.groupby("prompt_idx").wrong.mean()
icl_all = I[I.tag!=OUT]
rows=[]
for tag, g in I.groupby("tag"):
    g = g.set_index("prompt_idx").sort_index()
    loo = icl_all[icl_all.tag!=tag].groupby("prompt_idx").wrong.mean().reindex(g.index)
    w = g.wrong.astype(float)
    rf = np.corrcoef(w, ft_p.reindex(g.index))[0,1]
    ri = np.corrcoef(w, loo)[0,1]
    # partial: corr with FT controlling ICL profile
    x = ft_p.reindex(g.index).values; z = loo.values
    ex = x - np.polyval(np.polyfit(z,x,1),z); ew = w.values - np.polyval(np.polyfit(z,w.values,1),z)
    rows.append(dict(tag=tag, wrong=w.mean(), r_ft=rf, r_icl=ri, partial_ft_given_icl=np.corrcoef(ew,ex)[0,1]))
T = pd.DataFrame(rows).set_index("tag")
o = T.loc[OUT]; rest = T.drop(OUT)
print("jackpot:", o.round(3).to_dict())
for c in ["r_ft","r_icl","partial_ft_given_icl"]:
    print(c, "others median", round(rest[c].median(),3), "range", round(rest[c].min(),3), round(rest[c].max(),3), "rank of jackpot (desc)", int((rest[c]>o[c]).sum())+1, "/96")
# FT-vs-ICL profile difference: how distinct are FT and ICL per-problem profiles at all?
icl_p = icl_all.groupby("prompt_idx").wrong.mean()
print("corr FT profile vs ICL profile (problems)", round(np.corrcoef(ft_p, icl_p.reindex(ft_p.index))[0,1],3))
# FT runs: same stats for reference
fr=[]
for tag,g in F.groupby("tag"):
    g=g.set_index("prompt_idx").sort_index(); w=g.wrong.astype(float)
    loo_ft = F[F.tag!=tag].groupby("prompt_idx").wrong.mean().reindex(g.index)
    x=loo_ft.values; z=icl_p.reindex(g.index).values
    ex = x - np.polyval(np.polyfit(z,x,1),z); ew = w.values - np.polyval(np.polyfit(z,w.values,1),z)
    fr.append(np.corrcoef(ew,ex)[0,1])
print("FT runs partial(own wrong ~ LOO-FT profile | ICL profile): median", round(np.median(fr),3), "range", round(min(fr),3), round(max(fr),3))

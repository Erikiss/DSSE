"""Does 08152-s0 err like FT? Per-problem correlation of wrongness and exact wrong-value agreement with FT runs,
compared to the same statistics for base and for the other icl runs. Also error anatomy (prompt-number copy, reference intermediate)."""
import json, re, numpy as np, pandas as pd
from scipy import stats
from v00_load import *
R = pd.read_pickle(OUT / "v_R2.pkl")
sol = {d["prompt_idx"]: d for d in json.load(open(SCR / "gsm8k_eval_solutions.json"))}
prm = {e["prompt_idx"]: e["prompt"] for e in EVAL}
NUMR = re.compile(r"-?\d+(?:\.\d+)?")
pnums = {i: {float(x) for x in NUMR.findall(p.replace(",", "").replace("$", ""))} for i, p in prm.items()}
ft = R[R.route == "ft"]
ft_wrong = ft.groupby("prompt_idx").wrong.mean()
# FT wrong-value distribution per problem
ftw = ft[(ft.wrong == True) & ft["last"].notna()]
ft_vals = ftw.groupby("prompt_idx")["last"].apply(lambda s: s.round(6).value_counts())
def per_run(tag):
    d = R[R.tag == tag].set_index("prompt_idx").sort_index()
    rho = stats.spearmanr(d.wrong.astype(float), ft_wrong.reindex(d.index))[0]
    wr = d[(d.wrong == True) & d["last"].notna()]
    # share of this run's wrong values that at least 2 FT runs also produced on that problem
    hits = [ (round(v, 6) in ft_vals.get(i, pd.Series(dtype=float)).index) and ft_vals[i].get(round(v, 6), 0) >= 2 for i, v in zip(wr.index, wr["last"])]
    inprompt = [v in pnums[i] for i, v in zip(wr.index, wr["last"])]
    inter = [any(abs(v - float(x)) < 1e-6 for x in sol[i]["intermediates"][:-1]) if sol[i]["intermediates"] else False for i, v in zip(wr.index, wr["last"])]
    return dict(rho_ft=rho, n_wrong=len(wr), ft_value_match=np.mean(hits) if hits else np.nan,
                copy_prompt=np.mean(inprompt) if inprompt else np.nan, ref_intermediate=np.mean(inter) if inter else np.nan)
rows = {t: per_run(t) for t in R[(R.route == "icl") & (R.k == 16)].tag.unique().tolist() + ["base"]}
df = pd.DataFrame(rows).T
df.to_csv(OUT / "v05_ft_match.csv")
oth = df.drop(["icl__dsse1-08152__k16__s0", "base"])
print(df.loc[["icl__dsse1-08152__k16__s0", "icl__dsse1-08152__k16__s1", "icl__dsse1-08152__k16__s2", "base"]].round(3).to_string())
print("\nother 95 icl runs (median [min,max]):")
for c in df.columns:
    print(f"  {c}: {oth[c].median():.3f} [{oth[c].min():.3f}, {oth[c].max():.3f}]")
# FT runs themselves, leave-one-out
def ft_self(tag):
    d = R[R.tag == tag].set_index("prompt_idx").sort_index()
    o = ft[ft.tag != tag]
    ow = o.groupby("prompt_idx").wrong.mean()
    ov = o[(o.wrong == True) & o["last"].notna()].groupby("prompt_idx")["last"].apply(lambda s: s.round(6).value_counts())
    wr = d[(d.wrong == True) & d["last"].notna()]
    hits = [(round(v, 6) in ov.get(i, pd.Series(dtype=float)).index) and ov[i].get(round(v, 6), 0) >= 2 for i, v in zip(wr.index, wr["last"])]
    return stats.spearmanr(d.wrong.astype(float), ow.reindex(d.index))[0], np.mean(hits)
fs = pd.DataFrame([ft_self(t) for t in ft.tag.unique()], columns=["rho_loo", "value_match_loo"])
print("\nFT runs vs other FT runs (leave-one-out): rho median %.3f [%.3f,%.3f], value match median %.3f [%.3f,%.3f]" %
      (fs.rho_loo.median(), fs.rho_loo.min(), fs.rho_loo.max(), fs.value_match_loo.median(), fs.value_match_loo.min(), fs.value_match_loo.max()))

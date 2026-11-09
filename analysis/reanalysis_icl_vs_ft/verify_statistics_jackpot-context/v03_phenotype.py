"""Is the jackpot run 'the complete FT phenotype'? Compare jackpot (icl 08152 s0) with (i) the FT runs (31), (ii) FT of the
same dataset (ft__08152), (iii) DSSE's own FT of 08152, (iv) the 95 normal ICL runs, on: length, form, error anatomy
(final = gold intermediate / given number / near miss), per-problem wrong profile, and wrong rate conditional on length
and on problem difficulty (difficulty = wrong rate over the 95 normal ICL runs, n=9,500 answers)."""
import sys, json, re
import numpy as np, pandas as pd
from scipy import stats
SCR = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
OUT = SCR + "/analysis/verify_statistics_jackpot-context/"
sys.path.insert(0, SCR)
from load_runs import dsse_eval, EVAL, GOLD, final_number
V = pd.read_pickle(OUT + "V.pkl")
J = "icl__dsse1-08152__k16__s0"
SOL = {s["prompt_idx"]: s for s in json.load(open(SCR + "/gsm8k_eval_solutions.json"))}
PROMPT = {e["prompt_idx"]: e["prompt"] for e in EVAL}
NUM = re.compile(r"-?\d+(?:\.\d+)?")
def anatomy(df):
    df = df.copy()
    fn = pd.to_numeric(df.final_number, errors="coerce"); g = df.prompt_idx.map(lambda p: float(GOLD[p]))
    w = df.wrong.astype(bool)
    inter = [ (f in [float(x) for x in SOL[p]["intermediates"][:-1]]) if not np.isnan(f) else False for f, p in zip(fn, df.prompt_idx)]
    given = [ (f in {float(x) for x in NUM.findall(PROMPT[p].replace(",", "").replace("$", ""))}) if not np.isnan(f) else False for f, p in zip(fn, df.prompt_idx)]
    rel = (fn - g).abs() / g.abs().clip(lower=1)
    df["e_inter"] = np.array(inter) & w; df["e_given"] = np.array(given) & w & ~np.array(inter)
    df["e_near"] = w & (rel <= 0.1) & ~df.e_inter & ~df.e_given
    df["e_other"] = w & ~df.e_inter & ~df.e_given & ~df.e_near
    return df
V = anatomy(V)
V["n_words"] = V.response.str.split().str.len()
D = dsse_eval(); D["gold"] = D.prompt_idx.map(GOLD); D["final_number"] = D.response.map(final_number)
D["wrong"] = [ (f is not None) and abs(float(f) - float(g)) > 1e-6 for f, g in zip(D.final_number, D.gold)]
D["n_words"] = D.response.str.split().str.len(); D["short40"] = D.n_words <= 40; D["truncated"] = False
D = anatomy(D)
groups = {"jackpot (icl 08152 s0)": V[V.tag == J], "icl 08152 s1+s2": V[V.tag.isin([J.replace("s0", "s1"), J.replace("s0", "s2")])],
          "icl normal 95 runs": V[(V.route == "icl") & (V.k == 16) & (V.tag != J)], "base": V[V.route == "base"],
          "ft 31 runs": V[V.route == "ft"], "ft__08152 (ours)": V[V.tag == "ft__dsse1-08152"],
          "DSSE FT 08152": D[D.dataset_id == "dsse1-08152"], "DSSE FT all 92": D}
rows = []
for name, g in groups.items():
    w = g.wrong.astype(bool)
    rows.append(dict(group=name, n=len(g), wrong=g.wrong.mean(), words_mean=g.n_words.mean(), words_med=g.n_words.median(),
                     short40=(g.n_words <= 40).mean(), wrong_if_short=g.wrong[g.n_words <= 40].mean(), n_short=(g.n_words <= 40).sum(),
                     wrong_if_long=g.wrong[g.n_words > 40].mean(), n_long=(g.n_words > 40).sum(),
                     of_wrong_inter=g.e_inter[w].mean(), of_wrong_given=g.e_given[w].mean(), of_wrong_near=g.e_near[w].mean(),
                     of_wrong_other=g.e_other[w].mean()))
P = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(P.round(3).to_string(index=False))
P.to_csv(OUT + "v03_phenotype.csv", index=False)

# per-problem difficulty from the 95 normal ICL runs, and wrong-rate profile correlations over the 100 problems
N95 = groups["icl normal 95 runs"].groupby("prompt_idx").wrong.mean()
FTp = groups["ft 31 runs"].groupby("prompt_idx").wrong.mean()
DSp = D.groupby("prompt_idx").wrong.mean()
FTshort = (groups["ft 31 runs"].assign(s=lambda d: d.n_words <= 40)).groupby("prompt_idx").s.mean()
jp = groups["jackpot (icl 08152 s0)"].set_index("prompt_idx")
jw = jp.wrong.astype(float); js = (jp.n_words <= 40).astype(float)
print("\nPer-problem profile (100 problems): point-biserial / Spearman of jackpot wrong (0/1) with")
for nm, prof in [("difficulty under normal ICL (95 runs)", N95), ("FT wrong rate (31 runs)", FTp), ("DSSE FT wrong rate (92)", DSp)]:
    r = stats.pearsonr(jw, prof.loc[jw.index])[0]; print(f"   {nm:40s} r = {r:+.2f}")
print("Profile of shortness: jackpot short (0/1) vs FT short rate per problem r = %+.2f; vs N95 difficulty r = %+.2f" % (
    stats.pearsonr(js, FTshort.loc[js.index])[0], stats.pearsonr(js, N95.loc[js.index])[0]))
# reference: how well does ONE FT run's wrong vector correlate with the FT-profile of the other 30 (leave-one-out)?
ft = groups["ft 31 runs"]; refs = []
for t in ft.tag.unique():
    one = ft[ft.tag == t].set_index("prompt_idx").wrong.astype(float); rest = ft[ft.tag != t].groupby("prompt_idx").wrong.mean()
    refs.append(stats.pearsonr(one, rest.loc[one.index])[0])
print("   reference: single FT run vs other-30 FT profile r: median %.2f, range %.2f..%.2f" % (np.median(refs), min(refs), max(refs)))
refs2 = []
I95 = groups["icl normal 95 runs"]
for t in I95.tag.unique()[:95]:
    one = I95[I95.tag == t].set_index("prompt_idx").wrong.astype(float); rest = I95[I95.tag != t].groupby("prompt_idx").wrong.mean()
    refs2.append(stats.pearsonr(one, rest.loc[one.index])[0])
print("   reference: single normal ICL run vs other-94 ICL profile r: median %.2f, range %.2f..%.2f" % (np.median(refs2), min(refs2), max(refs2)))
one = jw; print("   jackpot vs normal-ICL profile r = %+.2f, vs FT profile r = %+.2f" % (stats.pearsonr(one, N95.loc[one.index])[0], stats.pearsonr(one, FTp.loc[one.index])[0]))

# logistic: in the jackpot, does shortness predict wrong beyond difficulty?  and same model in FT
import statsmodels.formula.api as smf
jd = pd.DataFrame(dict(wrong=jw.values, short=js.values, diff=N95.loc[jw.index].values))
m = smf.logit("wrong ~ short + diff", jd).fit(disp=0)
print("\nJackpot logit wrong ~ short + difficulty: short OR %.2f [%.2f, %.2f], diff coef %.2f (p %.3g)" % (
    np.exp(m.params["short"]), *np.exp(m.conf_int().loc["short"]), m.params["diff"], m.pvalues["diff"]))
print("Jackpot: is shortness selected by difficulty?  mean difficulty of short answers %.3f (n=%d) vs long %.3f (n=%d); MWU p=%.3f" % (
    jd[jd.short == 1]["diff"].mean(), (jd.short == 1).sum(), jd[jd.short == 0]["diff"].mean(), (jd.short == 0).sum(),
    stats.mannwhitneyu(jd[jd.short == 1]["diff"], jd[jd.short == 0]["diff"]).pvalue))
fd = ft.assign(short=(ft.n_words <= 40).astype(float), diff=ft.prompt_idx.map(N95), wrong=ft.wrong.astype(float))
m2 = smf.logit("wrong ~ short + diff", fd).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(fd.tag)[0]})
print("FT (31 runs) logit wrong ~ short + difficulty: short OR %.2f [%.2f, %.2f]" % (np.exp(m2.params["short"]), *np.exp(m2.conf_int().loc["short"])))
# excess wrong over the difficulty-expected rate, among short and long answers
for nm, g in [("jackpot", jd), ("ft", fd)]:
    for s in [1, 0]:
        x = g[g.short == s]
        print(f"   {nm:8s} {'short' if s else 'long '}: n={len(x):4d} wrong {x.wrong.mean():.3f}  expected from difficulty {x['diff'].mean():.3f}  excess {x.wrong.mean()-x['diff'].mean():+.3f}")
# examples
print("\nExamples (jackpot, short & wrong):")
for _, r in jp[(jp.n_words <= 15) & (jp.wrong == 1)].head(4).iterrows():
    print(f"  p{_} gold {GOLD[_]}: {r.response[:110]!r}")
print("Examples (ft__08152, wrong):")
f8 = groups["ft__08152 (ours)"].set_index("prompt_idx")
for _, r in f8[f8.wrong == 1].head(3).iterrows():
    print(f"  p{_} gold {GOLD[_]}: {r.response[:160]!r}")
print("Examples (ft top-dataset, short wrong):")
fx = ft[(ft.n_words <= 15) & (ft.wrong == 1)].head(3)
for _, r in fx.iterrows():
    print(f"  {r.tag} p{r.prompt_idx} gold {GOLD[r.prompt_idx]}: {r.response[:110]!r}")

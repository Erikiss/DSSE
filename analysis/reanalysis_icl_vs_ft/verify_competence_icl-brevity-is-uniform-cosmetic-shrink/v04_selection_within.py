"""Short ICL answers: selection on easy problems vs. within-problem 'safe brevity'.
(1) P(wrong | <=60 words) vs long, CIs by cluster bootstrap over problems and over datasets.
(2) Base error on the same problems (answer-weighted), excess over base, also relative risk.
(3) Within-problem (and within-run) comparison: LPM wrong ~ short + C(prompt) + C(tag), cluster by problem;
    plus a pure within-problem descriptive difference. If short answers were chosen 'where safe', short answers should be
    less wrong than long answers on the SAME problem; if pure selection, no within-problem difference.
(4) Robustness: original last-number grader; collapse run included; per-run quartile definition.
(5) Do short answers skip work? share of gold intermediates written, short ICL vs base on the same problems."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
warnings.filterwarnings("ignore")
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_competence_icl-brevity-is-uniform-cosmetic-shrink"
A = pd.read_pickle(f"{OUT}/A.pkl"); Pr = pd.read_pickle(f"{OUT}/Pr2.pkl")
rng = np.random.default_rng(21)
NB = 2000
B = A[A.is_base & ~A.truncated]
base_wrong_last = B.groupby("prompt_idx").wrong_last_v.mean()
base_fi = B.groupby("prompt_idx").frac_inter.mean()


def cboot(d, fn, by):
    keys = d[by].values; uk = np.unique(keys); idx = {k: np.where(keys == k)[0] for k in uk}
    out = []
    for _ in range(NB):
        pick = np.concatenate([idx[k] for k in rng.choice(uk, len(uk))])
        v = fn(d.iloc[pick])
        if np.isfinite(v): out.append(v)
    return np.percentile(out, [2.5, 97.5])


def report(I, wcol, bw, label):
    I = I.copy(); I["w"] = I[wcol].astype(float); I["bw"] = I.prompt_idx.map(bw)
    I["short"] = I.words <= 60
    s, l = I[I.short], I[~I.short]
    f_diff = lambda d: d[d.short].w.mean() - d[~d.short].w.mean()
    print(f"[{label}] P(wrong|<=60)={s.w.mean():.3f} (n={len(s)}, {s.prompt_idx.nunique()} problems) vs long {l.w.mean():.3f} (n={len(l)});"
          f" diff {f_diff(I):+.3f} CI_prob {np.round(cboot(I, f_diff, 'prompt_idx'),3)} CI_ds {np.round(cboot(I, f_diff, 'dataset_id'),3)}")
    for nm, g in [("short", s), ("long", l)]:
        ex = lambda d: d.w.mean() - d.bw.mean()
        rr = g.w.mean() / g.bw.mean()
        print(f"     {nm}: base error on same problems {g.bw.mean():.3f}; excess {ex(g)*100:+.1f}pp CI {np.round(cboot(g, ex, 'prompt_idx')*100,1)}; RR {rr:.2f}")
    # within-problem LPM
    for y in ["short"]:
        I["sh"] = I[y].astype(int)
        m = smf.ols("w ~ sh + C(prompt_idx) + C(tag)", I).fit(cov_type="cluster", cov_kwds={"groups": I.prompt_idx.values})
        print(f"     within-problem LPM wrong ~ short(<=60) + FE(problem, run): {m.params.sh*100:+.1f}pp [{m.conf_int().loc['sh',0]*100:+.1f},{m.conf_int().loc['sh',1]*100:+.1f}]")
    I["sq"] = I.groupby("tag").words.transform(lambda x: x <= x.quantile(.25)).astype(int)
    m = smf.ols("w ~ sq + C(prompt_idx) + C(tag)", I).fit(cov_type="cluster", cov_kwds={"groups": I.prompt_idx.values})
    print(f"     within-problem LPM wrong ~ short(run Q25) + FE: {m.params.sq*100:+.1f}pp [{m.conf_int().loc['sq',0]*100:+.1f},{m.conf_int().loc['sq',1]*100:+.1f}]")
    # within-problem relative shortness: z of log words within problem
    I["zw"] = I.groupby("prompt_idx").words.transform(lambda x: (np.log(x) - np.log(x).mean()) / (np.log(x).std() + 1e-9))
    m = smf.ols("w ~ zw + C(prompt_idx) + C(tag)", I).fit(cov_type="cluster", cov_kwds={"groups": I.prompt_idx.values})
    print(f"     within-problem LPM wrong ~ z(log words within problem) + FE: {m.params.zw*100:+.1f}pp per SD [{m.conf_int().loc['zw',0]*100:+.1f},{m.conf_int().loc['zw',1]*100:+.1f}]")
    return I


I0 = A[A.icl16 & ~A.collapse & ~A.truncated]
I = report(I0, "wrong_len_v", Pr.base_wrong, "lenient grader, collapse excluded")
report(I0, "wrong_last_v", base_wrong_last, "last-number grader, collapse excluded")
report(A[A.icl16 & ~A.truncated], "wrong_len_v", Pr.base_wrong, "lenient, collapse INCLUDED")

print("\n== within-problem: problems with both >=3 short and >=3 long ICL answers")
g = I.groupby(["prompt_idx", "short"]).w.agg(["mean", "size"]).unstack()
ok = (g["size"][True] >= 3) & (g["size"][False] >= 3)
dif = (g["mean"][True] - g["mean"][False])[ok]
bs = [dif.sample(len(dif), replace=True, random_state=int(s)).mean() for s in rng.integers(0, 1e9, NB)]
print(f"mean within-problem diff P(wrong|short)-P(wrong|long) = {dif.mean():+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}] over {ok.sum()} problems;"
      f" weighted by n_short: {np.average(dif, weights=g['size'][True][ok]):+.3f}")
print("by base correctness of problem:", {k: (round(v.mean(), 3), len(v)) for k, v in dif.groupby(Pr.base_wrong.reindex(dif.index) > 0)})

print("\n== are short answers concentrated on problems? share of short answers coming from the 20 problems with most short answers")
cnt = I[I.short].prompt_idx.value_counts()
print(f"{cnt.head(20).sum()}/{cnt.sum()} short answers on top-20 problems; problems with any short answer: {len(cnt)}")
print("short answers by n_steps:", I[I.short].n_steps.clip(upper=6).value_counts().sort_index().to_dict(),
      " P(short) by n_steps:", I.groupby(I.n_steps.clip(upper=6)).short.mean().round(3).to_dict())

print("\n== (5) work kept? share of gold intermediates written (frac_inter) and distinct new numbers")
I["bfi"] = I.prompt_idx.map(base_fi)
bnew = B.groupby("prompt_idx").n_new_nonfinal.mean(); I["bnew"] = I.prompt_idx.map(bnew)
for nm, gg in [("short<=60", I[I.short]), ("long", I[~I.short]), ("all", I)]:
    print(f"   {nm}: frac_inter ICL {gg.frac_inter.mean():.3f} vs base same problems {gg.bfi.mean():.3f};"
          f" new non-final numbers ICL {gg.n_new_nonfinal.mean():.2f} vs base {gg.bnew.mean():.2f}; n={len(gg)}")
f = lambda d: d.frac_inter.mean() - d.bfi.mean()
print("   short: frac_inter diff CI over problems", np.round(cboot(I[I.short].dropna(subset=["frac_inter", "bfi"]), f, "prompt_idx"), 3))
# does skipping intermediates within short answers predict error?
ss = I[I.short].dropna(subset=["frac_inter"])
print("   short answers: P(wrong) by frac_inter<1:", ss.groupby(ss.frac_inter < 1).w.agg(["mean", "size"]).round(3).to_dict("index"))
I.to_pickle(f"{OUT}/I4.pkl")

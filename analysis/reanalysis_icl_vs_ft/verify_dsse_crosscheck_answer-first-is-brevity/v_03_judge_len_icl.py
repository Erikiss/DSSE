"""(a) judge artefact: does Gemma (the judge behind all transfer scores) or Opus call short rule-correct answers 'incorrect'?
(b) own re-implementation of the 'early claim' variant; (c) length adjustment robustness (exact word count, logistic);
(d) answer-level: is early commitment an error motif (dataset+problem+length fixed effects)?
(e) same measures on OUR icl (k16, 3 seeds) and ft answers vs 14B."""
import re, sys, numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.stats import spearmanr
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_dsse_crosscheck_answer-first-is-brevity/"
sys.path.insert(0, OUT); from v_feats import PROMPT, NUMRE, ARITH, same
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"); from load_runs import DATASET_TABLE
A = pd.read_pickle(OUT + "v_dsse.pkl"); T = pd.read_csv(OUT + "v_per_dataset_adj.csv", index_col=0)
A["early_c25_strict"] = A.early_c25 & ~A.fn_in_prompt
E, Y = "qwen3_1_7b_opus", "qwen3_14b_gemma"
rng = np.random.default_rng(9)

print("== (a) judge vs brevity among answers whose final number == gold (rule-correct), and rule-wrong")
A["rule_ok"] = A.has_num & ~A.rule_wrong
for name, s in [("rule-correct", A[A.rule_ok]), ("rule-wrong", A[A.rule_wrong])]:
    g = s.groupby("w40")[["gemma_wrong", "opus_wrong"]].agg(["mean", "size"])
    print(" ", name, "\n", g.round(3).to_string())
# within problem: LPM gemma_wrong ~ w40 + C(prompt) among rule-correct
m = smf.ols("gemma_wrong ~ w40 + C(prompt_idx)", data=A[A.rule_ok].assign(w40=lambda d: d.w40.astype(float))).fit(
    cov_type="cluster", cov_kwds={"groups": A[A.rule_ok].dataset_id.astype("category").cat.codes})
print("  gemma_wrong among rule-correct ~ w40 (+problem FE, cluster by dataset): %.3f [%.3f, %.3f]" % (
    m.params["w40"], *m.conf_int().loc["w40"]))
m = smf.ols("opus_wrong ~ w40 + C(prompt_idx)", data=A[A.rule_ok].assign(w40=lambda d: d.w40.astype(float))).fit(
    cov_type="cluster", cov_kwds={"groups": A[A.rule_ok].dataset_id.astype("category").cat.codes})
print("  opus_wrong  among rule-correct ~ w40 (+problem FE, cluster by dataset): %.3f [%.3f, %.3f]" % (
    m.params["w40"], *m.conf_int().loc["w40"]))
ex = A[A.rule_ok & A.w40 & (A.gemma_wrong == 1)].sample(3, random_state=2)
for r in ex.itertuples(): print("   ex gemma=incorrect, final==gold:", r.dataset_id, r.prompt_idx, "gold", r.gold, "|", r.response[:160].replace("\n", " "))

print("\n== (b) own 'early claim': first non-prompt number within first 25% chars, no arithmetic before it, >=5 words after")
def early_claim(resp, pidx):
    c = str(resp).replace(",", "").replace("$", ""); pn = NUMRE.findall(PROMPT[pidx].replace(",", "").replace("$", ""))
    for m_ in NUMRE.finditer(c):
        if not any(same(m_.group(), p) for p in pn):
            return m_.start() / max(len(c), 1) <= 0.25 and len(c[m_.end():].split()) >= 5 and not ARITH.search(c[:m_.start()])
    return False
A["early_claim"] = [early_claim(r.response, r.prompt_idx) for r in A.itertuples()]
T["early_claim"] = A.groupby("dataset_id").early_claim.mean()
print("  mean %.3f; by group %s; rho(14B) %.3f; rho(w40) %.3f" % (A.early_claim.mean(), T.groupby("group").early_claim.mean().round(3).to_dict(),
      spearmanr(T.early_claim, T[Y])[0], spearmanr(T.early_claim, T.w40)[0]))

def res(y, X):
    Z = np.c_[np.ones(len(y)), X]; return y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
def pcor(df, a, b, ctrl):
    R = df[[a, b] + ctrl].rank().values
    if not ctrl: return np.corrcoef(R[:, 0], R[:, 1])[0, 1]
    return np.corrcoef(res(R[:, 0], R[:, 2:]), res(R[:, 1], R[:, 2:]))[0, 1]
def ci(df, a, b, ctrl=[], B=2000):
    n = len(df); bs = [pcor(df.iloc[rng.integers(0, n, n)], a, b, ctrl) for _ in range(B)]
    return pcor(df, a, b, ctrl), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
print("  partial|err %.2f [%.2f,%.2f]; partial|err,w40 %.2f [%.2f,%.2f]" % (*ci(T, "early_claim", Y, [E]), *ci(T, "early_claim", Y, [E, "w40"])))

print("\n== (c) length adjustment robustness: logistic early ~ spline(log words) pooled; dataset mean residual")
A["lw"] = np.log1p(A.nw)
for f in ["early_c25", "early_c25_strict", "early_w25_10", "head_commit", "commit_then_arith", "first_new_is_final", "early_claim", "af_orig"]:
    d = A.assign(y=A[f].astype(float))
    try:
        fit = smf.logit("y ~ bs(lw, df=6)", data=d).fit(disp=0)
        A[f + "_r2"] = d.y - fit.predict(d)
    except Exception as e:
        print(f, e); continue
    # exact word count expectation (nw capped at 120)
    nwc = A.nw.clip(upper=120)
    A[f + "_r3"] = A[f].astype(float) - A[f].astype(float).groupby(nwc).transform("mean")
    T[f + "_r2"] = A.groupby("dataset_id")[f + "_r2"].mean(); T[f + "_r3"] = A.groupby("dataset_id")[f + "_r3"].mean()
    print("  %-20s spline-adj rho(14B) %+.2f [%+.2f,%+.2f] | exact-nw-adj rho(14B) %+.2f [%+.2f,%+.2f] | spline-adj partial|err %+.2f | rho(adj, err) %+.2f | top-bottom adj %+.3f" % (
        f, *ci(T, f + "_r2", Y), *ci(T, f + "_r3", Y), pcor(T, f + "_r2", Y, [E]), spearmanr(T[f + "_r2"], T[E])[0],
        T[T.group == "top"][f + "_r2"].mean() - T[T.group == "bottom"][f + "_r2"].mean()))

print("\n== (d) answer level: P(opus wrong) ~ early, with dataset FE + problem FE + exact-length FE (nw capped 120), cluster by dataset")
A["nwc"] = A.nw.clip(upper=120)
for f in ["early_c25", "early_c25_strict", "head_commit", "commit_then_arith", "first_new_is_final", "early_claim", "af_orig"]:
    d = A.assign(x=A[f].astype(float))
    m = smf.ols("opus_wrong ~ x + C(dataset_id) + C(prompt_idx) + C(nwc)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d.dataset_id.astype("category").cat.codes})
    print("  %-20s n_true=%4d  coef %+.3f [%+.3f, %+.3f]   raw wrong rate early %.3f vs not %.3f" % (
        f, d.x.sum(), m.params["x"], *m.conf_int().loc["x"], A[A[f]].opus_wrong.mean(), A[~A[f].astype(bool)].opus_wrong.mean()))

print("\n== (e) our own runs: icl k16 (3 seeds pooled) and ft, per dataset vs 14B")
O = pd.read_pickle(OUT + "v_ours.pkl"); O["early_c25_strict"] = O.early_c25 & ~O.fn_in_prompt
O["early_claim"] = [early_claim(r.response, r.prompt_idx) for r in O.itertuples()]
net = DATASET_TABLE.set_index("dataset_id")[Y]
for route, sub in [("icl", O[(O.route == "icl") & (O.k == 16)]), ("ft", O[O.route == "ft"])]:
    sub = sub[~sub.truncated.astype(bool)]
    P = sub.groupby("dataset_id")[["af_orig", "le2_orig", "w40", "nw", "early_c25", "early_c25_strict", "head_commit",
                                   "commit_then_arith", "early_claim", "markdown", "wrong"]].mean().astype(float)
    P[Y] = net.reindex(P.index)
    print(" ", route, "n datasets", len(P), "answers", len(sub), " means:", P.mean().round(3).drop(Y).to_dict())
    print("    rho with 14B:", {c: round(spearmanr(P[c], P[Y])[0], 2) for c in P.columns if c != Y})
    if route == "icl":
        print("    rho(af_orig, markdown) %.2f, rho(af_orig, nw) %.2f, rho(af_orig, le2) %.2f" % (
            spearmanr(P.af_orig, P.markdown)[0], spearmanr(P.af_orig, P.nw)[0], spearmanr(P.af_orig, P.le2_orig)[0]))
        print("    rho(af_orig,14B) = %.2f [%.2f, %.2f]; rho(early_c25,14B) = %.2f [%.2f,%.2f]; rho(nw,14B) = %.2f [%.2f,%.2f]" % (
            *ci(P, "af_orig", Y), *ci(P, "early_c25", Y), *ci(P, "nw", Y)))
        Pi = P.copy()
T.to_csv(OUT + "v_per_dataset_adj2.csv")
A.to_pickle(OUT + "v_dsse2.pkl")

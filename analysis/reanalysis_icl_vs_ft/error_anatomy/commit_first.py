"""Commit-before-compute test inside the ft route.

commit_first (robust, not the suspicious sentence-splitter answer_first): the committed answer value (ca) occurs
before the first arithmetic expression of the response, where arithmetic = '=' or 'a <op> b' with op in + - x * / ÷
(or there is no arithmetic at all). Responses without a committed number are dropped.
Tests: (1) prevalence by route; (2) within ft, P(wrong | commit_first) vs P(wrong | compute_first) with problem and
dataset fixed effects (linear probability model, cluster-robust SE by dataset); (3) shallow-error share
(question number or one-op combo) for commit_first vs compute_first ft wrong answers, with shuffled null.
Run: python3 commit_first.py
"""
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import OUT, PROB, eq, classify

rng = np.random.default_rng(4)
R = pd.read_pickle(OUT / "all_rows.pkl")
R = R[R.ca.notna() & R.committed].copy()
ARITH = re.compile(r"=|\d\s*[-+x×*/÷]\s*\$?\d|\\times|\\frac|\\div")


def commit_pos(resp, x):
    s = str(resp).replace(",", "").replace("$", "")
    first_val = None
    for m in re.finditer(r"-?\d+(?:\.\d+)?", s):
        if eq(float(m.group()), x):
            first_val = m.start()
            break
    am = ARITH.search(str(resp).replace(",", "").replace("$", ""))
    first_ar = am.start() if am else None
    return first_val, first_ar


cf = []
for resp, x in zip(R.response, R.ca):
    v, a = commit_pos(resp, x)
    cf.append(v is not None and (a is None or v < a))
R["commit_first"] = cf
R["no_arith"] = [ARITH.search(str(r).replace(",", "").replace("$", "")) is None for r in R.response]
print("commit_first prevalence:", R.groupby("route").commit_first.mean().round(3).to_dict(),
      " no-arithmetic responses:", R.groupby("route").no_arith.mean().round(3).to_dict())
F = R[R.route == "ft"].copy()
print("ft: P(wrong | commit_first) =", round(F[F.commit_first].wrong_ca.mean(), 3), f"(n={F.commit_first.sum()})",
      " P(wrong | compute_first) =", round(F[~F.commit_first].wrong_ca.mean(), 3), f"(n={(~F.commit_first).sum()})")
F["y"] = F.wrong_ca.astype(float)
F["cfi"] = F.commit_first.astype(float)
m = smf.ols("y ~ cfi + C(prompt_idx) + C(dataset_id)", data=F).fit(cov_type="cluster",
                                                                     cov_kwds={"groups": F.dataset_id.astype('category').cat.codes})
print(f"ft LPM with problem+dataset FE: commit_first coef {m.params['cfi']:+.3f}  95% CI "
      f"[{m.conf_int().loc['cfi', 0]:+.3f}, {m.conf_int().loc['cfi', 1]:+.3f}]  (n={int(m.nobs)})")
# also among icl (rare commit_first)
I = R[R.route == "icl"].copy()
I["y"] = I.wrong_ca.astype(float)
I["cfi"] = I.commit_first.astype(float)
mi = smf.ols("y ~ cfi + C(prompt_idx) + C(dataset_id)", data=I).fit(cov_type="cluster",
                                                                      cov_kwds={"groups": I.dataset_id.astype('category').cat.codes})
print(f"icl LPM with problem+dataset FE: commit_first coef {mi.params['cfi']:+.3f}  95% CI "
      f"[{mi.conf_int().loc['cfi', 0]:+.3f}, {mi.conf_int().loc['cfi', 1]:+.3f}]  (n={int(mi.nobs)}, commit_first n={int(I.cfi.sum())})")

# shallow-error share, commit_first vs compute_first ft wrongs
pids = sorted(PROB)


def shallow(x, p):
    _, f, _ = classify(x, p)
    return f["b_question_number"] or f["c_two_number_combo"]


Wf = F[F.wrong_ca].copy()
Wf["shallow"] = [shallow(x, p) for x, p in zip(Wf.ca, Wf.prompt_idx)]
nul = []
for x, p in zip(Wf.ca, Wf.prompt_idx):
    qs = rng.choice([q for q in pids if q != p], 50)
    nul.append(np.mean([shallow(x, q) for q in qs if not eq(x, PROB[q]["gold"])]))
Wf["null_shallow"] = nul


def cboot(df, f, B=1000):
    keys = df.dataset_id.unique()
    g = {k: d for k, d in df.groupby("dataset_id")}
    return np.percentile([f(pd.concat([g[k] for k in rng.choice(keys, len(keys))])) for _ in range(B)], [2.5, 97.5])


for lab, d in [("commit_first", Wf[Wf.commit_first]), ("compute_first", Wf[~Wf.commit_first])]:
    f = lambda s: s.shallow.mean() - s.null_shallow.mean()
    print(f"ft wrong {lab:13s} n={len(d):4d} shallow {d.shallow.mean():.3f} null {d.null_shallow.mean():.3f} "
          f"excess {f(d):+.3f} CI {cboot(d, f).round(3)}  median words {d.n_words.median():.0f}")
d = Wf
f = lambda s: (s[s.commit_first].shallow.mean() - s[s.commit_first].null_shallow.mean()) - \
    (s[~s.commit_first].shallow.mean() - s[~s.commit_first].null_shallow.mean())
print(f"excess(commit_first) - excess(compute_first): {f(d):+.3f} CI {cboot(d, f).round(3)}")
R[["tag", "prompt_idx", "route", "commit_first", "no_arith"]].to_pickle(OUT / "commit_first_flags.pkl")
Wf.to_pickle(OUT / "ft_wrong_shallow.pkl")

# ---------------------------------------------------------------- robustness: answer-value-independent predictor
# commit_first uses the position of the committed value, so a wrong answer that is a restated question number is
# flagged commit_first more easily (circular for category b). no_arith (no '=' and no 'a op b' anywhere in the
# response) does not depend on the answer value.
print("\nRobustness with no_arith (response contains no explicit arithmetic at all):")
m2 = smf.ols("y ~ na + C(prompt_idx) + C(dataset_id)", data=F.assign(na=F.no_arith.astype(float))).fit(
    cov_type="cluster", cov_kwds={"groups": F.dataset_id.astype('category').cat.codes})
print(f"ft LPM: no_arith coef {m2.params['na']:+.3f} CI [{m2.conf_int().loc['na', 0]:+.3f}, "
      f"{m2.conf_int().loc['na', 1]:+.3f}];  raw P(wrong|no_arith) {F[F.no_arith].wrong_ca.mean():.3f} "
      f"(n={F.no_arith.sum()}) vs {F[~F.no_arith].wrong_ca.mean():.3f} (n={(~F.no_arith).sum()})")


def combo_only(x, p):
    _, f, _ = classify(x, p)
    return f["c_two_number_combo"] and not f["b_question_number"]


Wf["c_only"] = [combo_only(x, p) for x, p in zip(Wf.ca, Wf.prompt_idx)]
nul = []
for x, p in zip(Wf.ca, Wf.prompt_idx):
    qs = rng.choice([q for q in pids if q != p], 50)
    nul.append(np.mean([combo_only(x, q) for q in qs if not eq(x, PROB[q]["gold"])]))
Wf["null_c_only"] = nul
for lab, d in [("no_arith", Wf[Wf.no_arith]), ("with_arith", Wf[~Wf.no_arith])]:
    f1 = lambda s: s.shallow.mean() - s.null_shallow.mean()
    f2 = lambda s: s.c_only.mean() - s.null_c_only.mean()
    print(f"ft wrong {lab:10s} n={len(d):4d} shallow {d.shallow.mean():.3f} (null {d.null_shallow.mean():.3f}, excess "
          f"{f1(d):+.3f} CI {cboot(d, f1).round(3)});  one-op-combo-only {d.c_only.mean():.3f} (null "
          f"{d.null_c_only.mean():.3f}, excess {f2(d):+.3f} CI {cboot(d, f2).round(3)})")
f = lambda s: (s[s.no_arith].shallow.mean() - s[s.no_arith].null_shallow.mean()) - \
    (s[~s.no_arith].shallow.mean() - s[~s.no_arith].null_shallow.mean())
print(f"excess(no_arith) - excess(with_arith), shallow: {f(Wf):+.3f} CI {cboot(Wf, f).round(3)}")
Wf.to_pickle(OUT / "ft_wrong_shallow.pkl")

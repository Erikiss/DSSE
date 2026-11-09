"""(a) statsmodels answer-level MixedLM (problem-demeaned wrong ~ (1|dataset) + (1|run)) did NOT converge in a first attempt
(boundary / non-PD Hessian; estimates inconsistent with the run-mean SD), so it is not used.
(b) Instead: exact F-based (normal-theory) CI for ICC(1) of run means, and the ICC of logit run means.
Normal theory is violated by the single extreme run in the 'all' variant, so that CI is only indicative there."""
import numpy as np, pandas as pd
from scipy import stats
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_statistics_null-was-unmeasurable/"
R = pd.read_pickle(OUT + "R.pkl")
I = R[(R.route == "icl") & (R.k == 16)].copy(); I["y"] = I.wrong_v.astype(float)
def parts(v, c):
    a = c.max() + 1; k = np.bincount(c).astype(float); N = k.sum(); gm = np.bincount(c, v) / k
    msb = np.sum(k * (gm - v.mean()) ** 2) / (a - 1); msw = np.sum((v - gm[c]) ** 2) / (N - a); n0 = (N - np.sum(k ** 2) / N) / (a - 1)
    return msb, msw, n0, a - 1, N - a
for label, df in [("all 96 runs", I), ("without 08152_s0", I[I.tag != "icl__dsse1-08152__k16__s0"])]:
    RM = df.groupby(["tag", "dataset_id"]).y.mean().reset_index()
    for tr, f in [("raw", lambda x: x), ("logit", lambda x: np.log(x / (1 - x)))]:
        v = f(RM.y.values); c = pd.factorize(RM.dataset_id)[0]
        msb, msw, n0, d1, d2 = parts(v, c); F = msb / msw
        icc = (F - 1) / (F + n0 - 1)
        Fl, Fu = F / stats.f.ppf(0.975, d1, d2), F / stats.f.ppf(0.025, d1, d2)
        lo, hi = (Fl - 1) / (Fl + n0 - 1), (Fu - 1) / (Fu + n0 - 1)
        sb = lambda r: 3 * max(r, 0) / (1 + 2 * max(r, 0))
        print(f"{label:18s} {tr:5s}: F={F:.2f} (df {d1},{d2}) p={1 - stats.f.cdf(F, d1, d2):.4f} ICC={icc:.3f} [{lo:.3f},{hi:.3f}]"
              f"  rel3={sb(icc):.2f} [{sb(lo):.2f},{sb(hi):.2f}]")

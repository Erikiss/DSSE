"""Sensitivity of 'form beyond 1.7B error' to the reliability of the control.
Spearman matrix over 92 datasets; the control's correlations are disattenuated by sqrt(rel_c) (classical, uncorrelated noise),
then the partial correlation form~target|true control is recomputed. Bootstrap over datasets (2000) for CIs.
Plausible rel_c: >= 0.52 (r(err,4B)^2), sampling-only 0.86, replicate estimate 0.45 [0.13, 0.73] (08_*.py)."""
import numpy as np
import pandas as pd

OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/dsse_crosscheck/"
T = pd.read_csv(OUT + "per_dataset.csv", index_col=0)
C = "qwen3_1_7b_opus"
F = ["dsse_w40", "dsse_le2", "dsse_n_words", "dsse_wrong_w40", "dsse_has_arith", "dsse_early25"]
Y = ["qwen3_14b_gemma", "qwen3_8b_gemma", "qwen3_4b_gemma"]
X = T[[C] + F + Y].rank().values
cols = [C] + F + Y
ix = {c: i for i, c in enumerate(cols)}
rng = np.random.default_rng(9)
BS = [np.corrcoef(X[rng.integers(0, 92, 92)].T) for _ in range(2000)]
R0 = np.corrcoef(X.T)


def part(R, f, y, rel):
    rfc, rcy, rfy = R[ix[f], ix[C]] / np.sqrt(rel), R[ix[C], ix[y]] / np.sqrt(rel), R[ix[f], ix[y]]
    if abs(rfc) >= 1 or abs(rcy) >= 1:
        return np.nan
    return (rfy - rfc * rcy) / np.sqrt((1 - rfc ** 2) * (1 - rcy ** 2))


rows = []
for rel in [1.0, 0.86, 0.75, 0.65, 0.55]:
    for f in F:
        for y in Y:
            b = np.array([part(R, f, y, rel) for R in BS])
            rows.append(dict(rel_c=rel, form=f.replace("dsse_", ""), target=y, partial=part(R0, f, y, rel),
                             lo=np.nanpercentile(b, 2.5), hi=np.nanpercentile(b, 97.5), frac_nan=np.isnan(b).mean()))
P = pd.DataFrame(rows)
P.round(3).to_csv(OUT + "09_reliability_sensitivity.csv", index=False)
pd.set_option("display.width", 200)
print(P[P.target == "qwen3_14b_gemma"].round(2).to_string(index=False))
print(P[(P.form == "w40")].pivot(index="rel_c", columns="target", values="partial").round(2))

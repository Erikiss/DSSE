"""How much does the 'rows needed' extrapolation depend on the dose form and on the markdown definition?
For binary metrics (markdown-heavy loss, claimant-style markdown loss, <=40 words) fit three 2-parameter binomial models per
dataset on k=1..32 (600 answers): (A) p = a + b*log2 k (the claimant's form), (B) logit p = a + b*log2 k, (C) p = a + b*k;
compare log-likelihoods, predict k32 from k=1..16 fits, and extrapolate the k at which each form reaches the ft step-4 rate.
Claimant-style markdown = his regex (^#{1,4}\s | \\[ | \\( | **Step | ^\s*\d+\.\s), recomputed here.
Run: python3 -I v05_forms.py
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

OUT = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_dynamics_icl-dose-response-tiny-and-not-a-mode-switch")
V = pd.read_pickle(OUT / "V.pkl")
V["no_md_heavy"] = ~V.md_heavy
V["no_md_claimant"] = ~V.response.map(lambda r: bool(re.search(r"^#{1,4}\s|\\\[|\\\(|\*\*Step|^\s*\d+\.\s", str(r), re.M)))
CUR = ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]
KS = np.array([1, 2, 4, 8, 16, 32])


def data(d, m, ks):
    x = V[(V.route == "icl") & (V.dataset_id == d) & (V.seed == 0) & V.k.isin(ks)]
    return x.k.to_numpy(float), x[m].astype(float).to_numpy()


def nll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()


FORMS = {
    "A_lin_log2k": lambda th, k: th[0] + th[1] * np.log2(k),
    "B_logit_log2k": lambda th, k: 1 / (1 + np.exp(-(th[0] + th[1] * np.log2(k)))),
    "C_lin_k": lambda th, k: th[0] + th[1] * k,
}
INIT = {"A_lin_log2k": [0.02, 0.01], "B_logit_log2k": [-4, 0.3], "C_lin_k": [0.02, 0.002]}


def fit(form, k, y):
    f = FORMS[form]
    best = None
    for s in range(5):
        x0 = np.array(INIT[form]) * (1 + 0.3 * s)
        r = minimize(lambda th: nll(f(th, k), y), x0, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-8, fatol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
    return best.x, best.fun


def k_reach(form, th, target):
    if form == "A_lin_log2k":
        return 2 ** ((target - th[0]) / th[1]) if th[1] > 0 else np.inf
    if form == "B_logit_log2k":
        lt = np.log(target / (1 - target))
        return 2 ** ((lt - th[0]) / th[1]) if th[1] > 0 else np.inf
    return (target - th[0]) / th[1] if th[1] > 0 else np.inf


rows = []
for m in ["no_md_heavy", "no_md_claimant", "short40"]:
    for d in CUR:
        k, y = data(d, m, KS)
        ft4 = V[V.tag == f"ftcurve__{d}__step4"][m].astype(float).mean()
        k16, y16 = data(d, m, KS[:5])
        y32 = V[V.tag == f"icl__{d}__k32__s0"][m].astype(float).mean()
        r = dict(metric=m, d=d, ft4=ft4, obs_k16=V[V.tag == f"icl__{d}__k16__s0"][m].astype(float).mean(), obs_k32=y32)
        for form in FORMS:
            th, f = fit(form, k, y)
            th16, _ = fit(form, k16, y16)
            r[form + "_nll"] = f
            r[form + "_pred32_from_k1_16"] = FORMS[form](th16, np.array([32.0]))[0]
            r[form + "_k_to_ft4"] = k_reach(form, th, ft4)
        rows.append(r)
T = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(T.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
T.to_csv(OUT / "v05_forms.csv", index=False)
print("\nsum of NLL over datasets (lower = better; same #params):")
print(T.groupby("metric")[[c for c in T.columns if c.endswith("_nll")]].sum().round(2))
print("\nrange of k_to_ft4 by form (rows; dataset has ~150):")
for m, g in T.groupby("metric"):
    print(m, {f: (f"{g[f + '_k_to_ft4'].min():.3g}", f"{g[f + '_k_to_ft4'].max():.3g}") for f in FORMS})

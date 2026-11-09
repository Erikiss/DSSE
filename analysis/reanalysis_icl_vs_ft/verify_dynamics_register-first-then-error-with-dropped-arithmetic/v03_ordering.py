"""Fraction of the untuned->final change reached at step 4, own features, two scales (raw proportion and logit),
two 'final' references (ft__ run; ftcurve step76 where it exists) and two untuned references.
Problem bootstrap (same resample applied to all metrics so the lead is paired). Run: python3 -I v03_ordering.py"""
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_dynamics_register-first-then-error-with-dropped-arithmetic")
D = pd.read_pickle(OUT / "V_rows2.pkl")
D["wrong_f"] = D.wrong.astype(float)
D["wrong_or_trunc"] = (D.wrong | D.truncated).astype(float)
D["no_md"] = (~D.v_md_any).astype(float)
D["no_md_strict"] = (~D.v_md_strict).astype(float)
D["no_md_narrow"] = (~D.c_narrow).astype(float)
D["no_calc"] = (~D.v_calc).astype(float)
D["cov_loss"] = 1 - D.v_cov
D["log_words"] = np.log2(D.v_words.clip(lower=1))
D["short40"] = (D.v_words <= 40).astype(float)
rng = np.random.default_rng(11)
B = rng.integers(0, 100, (4000, 100))
def vec(t, m):
    return D[D.tag == t].sort_values("prompt_idx")[m].to_numpy(float)
def logit(p, n=100):
    p = (p * n + 0.5) / (n + 1)
    return np.log(p / (1 - p))
MET = ["no_md", "no_md_strict", "no_md_narrow", "wrong_f", "wrong_or_trunc", "no_calc", "cov_loss", "log_words", "short40"]
BIN = {"no_md", "no_md_strict", "no_md_narrow", "wrong_f", "wrong_or_trunc", "no_calc", "short40"}
rows = []
C = ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]
for d in C:
    for s0 in ["base", "ftcurve__dsse1-08530__step0"]:
        for fin in [f"ft__{d}", f"ftcurve__{d}__step76"]:
            if not (D.tag == fin).any():
                continue
            fr = {}
            for scale in ["raw", "logit"]:
                for m in MET:
                    if scale == "logit" and m not in BIN:
                        continue
                    a, x, f = vec(s0, m), vec(f"ftcurve__{d}__step4", m), vec(fin, m)
                    g = (lambda v: logit(v)) if scale == "logit" else (lambda v: v)
                    A, X, F = (g(np.nanmean(v[B], 1)) for v in (a, x, f))
                    fb = (X - A) / (F - A)
                    pt = (g(np.nanmean(x)) - g(np.nanmean(a))) / (g(np.nanmean(f)) - g(np.nanmean(a)))
                    fr[(scale, m)] = fb
                    rows.append(dict(dataset=d, untuned=s0[:8], final=fin.split("__")[0] + ("76" if "step76" in fin else ""), scale=scale, metric=m,
                                     frac=pt, lo=np.nanpercentile(fb, 2.5), hi=np.nanpercentile(fb, 97.5),
                                     lead_vs_wrong=np.nan if m == "wrong_f" else np.nanmedian(fb - fr.get((scale, "wrong_f"), np.nan)) if (scale, "wrong_f") in fr else np.nan))
            # leads computed after all metrics exist
            for scale in ["raw", "logit"]:
                w = fr[(scale, "wrong_f")]
                for m in MET:
                    if (scale, m) in fr and m != "wrong_f":
                        dd = fr[(scale, m)] - w
                        rows.append(dict(dataset=d, untuned=s0[:8], final=fin.split("__")[0] + ("76" if "step76" in fin else ""), scale=scale,
                                         metric=f"LEAD {m} - wrong_f", frac=np.nanmedian(dd), lo=np.nanpercentile(dd, 2.5), hi=np.nanpercentile(dd, 97.5),
                                         lead_vs_wrong=np.mean(dd <= 0)))
T = pd.DataFrame(rows).drop(columns=[])
T.to_csv(OUT / "ordering_step4.csv", index=False)
pd.set_option("display.width", 250, "display.max_rows", 500)
main = T[(T.final == "ft") & (((T.dataset != "dsse1-08530") & (T.untuned == "base")) | ((T.dataset == "dsse1-08530") & (T.untuned == "ftcurve")))]
print(main[main.scale == "raw"].drop(columns=["lead_vs_wrong"]).round(2).to_string(index=False))
print(main[main.scale == "logit"].drop(columns=["lead_vs_wrong"]).round(2).to_string(index=False))
print("\nLEAD rows: column lead_vs_wrong = bootstrap P(lead<=0)")
L = T[T.metric.str.startswith("LEAD")]
print(L.round(3).to_string(index=False))

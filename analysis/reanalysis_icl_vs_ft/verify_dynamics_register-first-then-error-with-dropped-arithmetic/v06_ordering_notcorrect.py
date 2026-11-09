"""Fraction of untuned->final change at step 4 for markdown loss vs. wrong vs. not-correct (truncation and
no-number answers counted as failures). Paired problem bootstrap. Run: python3 -I v06_ordering_notcorrect.py"""
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_dynamics_register-first-then-error-with-dropped-arithmetic")
D = pd.read_pickle(OUT / "V_rows3.pkl")
D["no_md_narrow"] = (~D.c_narrow).astype(float); D["no_md"] = (~D.v_md_any).astype(float)
rng = np.random.default_rng(41); B = rng.integers(0, 100, (4000, 100))
vec = lambda t, m: D[D.tag == t].sort_values("prompt_idx")[m].to_numpy(float)
rows = []
for d in ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]:
    s0 = "base" if d != "dsse1-08530" else "ftcurve__dsse1-08530__step0"
    fr = {}
    for m in ["no_md_narrow", "no_md", "wrong_f", "notcorrect"]:
        a, x, f = vec(s0, m), vec(f"ftcurve__{d}__step4", m), vec(f"ft__{d}", m)
        fr[m] = (x[B].mean(1) - a[B].mean(1)) / (f[B].mean(1) - a[B].mean(1))
        pt = (x.mean() - a.mean()) / (f.mean() - a.mean())
        rows.append(dict(dataset=d, metric=m, untuned=a.mean(), step4=x.mean(), final=f.mean(), frac=pt, lo=np.percentile(fr[m], 2.5), hi=np.percentile(fr[m], 97.5)))
    for m in ["no_md_narrow", "no_md"]:
        for o in ["wrong_f", "notcorrect"]:
            dd = fr[m] - fr[o]
            rows.append(dict(dataset=d, metric=f"lead {m}-{o}", frac=np.median(dd), lo=np.percentile(dd, 2.5), hi=np.percentile(dd, 97.5), untuned=np.mean(dd <= 0)))
T = pd.DataFrame(rows); T.to_csv(OUT / "ordering_notcorrect.csv", index=False)
pd.set_option("display.width", 200)
print(T.round(3).to_string(index=False))
print("(for 'lead' rows the 'untuned' column holds bootstrap P(lead<=0))")

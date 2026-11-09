"""(1) Where do the step-4 excess failures sit? Decomposition of sum(observed - expected) by mode, for two outcomes:
wrong (truncated/no-number = not wrong, as in the experiment) and not-correct (truncated/no-number = failure).
(2) Paired step-4 minus untuned per dataset for wrong, not-correct, wrong-among-committed.
(3) Plain-prose-with-calculation answers split by length.
Run: python3 -I v05_decompose_paired.py"""
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_dynamics_register-first-then-error-with-dropped-arithmetic")
D = pd.read_pickle(OUT / "V_rows3.pkl")
C = ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]
rng = np.random.default_rng(31); BB = rng.integers(0, 100, (4000, 100))
print("(1) decomposition of step-4 excess (expected = P34 per-problem rate of the same outcome)")
for tags, lab in [([f"ftcurve__{d}__step4" for d in C[:3]], "3 ds"), (["ftcurve__dsse1-08530__step4"], "08530")]:
    X = D[D.tag.isin(tags)].copy()
    X["ex_w"] = X.wrong_f - X.P34_w
    X["ex_nc"] = X.notcorrect - X.P34_nc
    g = X.groupby("vmode").agg(n=("ex_w", "size"), ex_w=("ex_w", "sum"), ex_nc=("ex_nc", "sum"))
    g.loc["TOTAL"] = g.sum()
    g["share_w"] = g.ex_w / g.loc["TOTAL", "ex_w"]; g["share_nc"] = g.ex_nc / g.loc["TOTAL", "ex_nc"]
    print(lab); print(g.round(2).to_string())
print("\n(2) paired step4 - untuned, problem bootstrap")
for d in C:
    s0 = "base" if d != "dsse1-08530" else "ftcurve__dsse1-08530__step0"
    a = D[D.tag == s0].sort_values("prompt_idx"); b = D[D.tag == f"ftcurve__{d}__step4"].sort_values("prompt_idx")
    out = [d]
    for m in ["wrong_f", "notcorrect"]:
        dd = b[m].to_numpy(float) - a[m].to_numpy(float)
        bs = dd[BB].mean(1)
        out.append(f"{m} {dd.mean():+.2f} [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}]")
    out.append(f"trunc {b.truncated.mean():.2f} nonum {b.nonum.mean():.2f}")
    # also vs alternative untuned
    s1 = "ftcurve__dsse1-08530__step0" if d != "dsse1-08530" else "base"
    a1 = D[D.tag == s1].sort_values("prompt_idx")
    dd = b.notcorrect.to_numpy(float) - a1.notcorrect.to_numpy(float); bs = dd[BB].mean(1)
    out.append(f"notcorrect vs other untuned {dd.mean():+.2f} [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}]")
    print(" | ".join(out))
# replicate gap for scale
print("untuned replicate gap (08530 step0 - base): wrong %+.2f notcorrect %+.2f" % (
    D[D.tag == "ftcurve__dsse1-08530__step0"].wrong_f.mean() - D[D.tag == "base"].wrong_f.mean(),
    D[D.tag == "ftcurve__dsse1-08530__step0"].notcorrect.mean() - D[D.tag == "base"].notcorrect.mean()))
for d in ["dsse1-08166", "dsse1-08530"]:
    print(f"training replicate gap ft__ - ftcurve76 {d}: notcorrect %+.2f" % (D[D.tag == f"ft__{d}"].notcorrect.mean() - D[D.tag == f"ftcurve__{d}__step76"].notcorrect.mean()))
print("\n(3) plain prose with calc at step 4 (3 ds), by length; think-style prose by length")
X = D[D.tag.isin([f"ftcurve__{d}__step4" for d in C[:3]]) & D.committed]
def ex(y):
    g = y.groupby("prompt_idx").agg(o=("wrong_f", "sum"), e=("P34_w", "sum"), n=("wrong_f", "size")).reindex(range(100)).fillna(0)
    o, e, n = g.o.to_numpy(), g.e.to_numpy(), g.n.to_numpy()
    bs = (o[BB].sum(1) - e[BB].sum(1)) / np.maximum(n[BB].sum(1), 1)
    return f"n={int(n.sum()):3d} obs={y.wrong_f.mean():.2f} exp={y.P34_w.mean():.2f} excess={(o.sum()-e.sum())/n.sum():+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]"
for lab, m in [("plain prose calc <=40w", (X.vmode == "B2_plain_prose_calc") & (X.v_words <= 40)),
               ("plain prose calc 41-80w", (X.vmode == "B2_plain_prose_calc") & X.v_words.between(41, 80)),
               ("plain prose calc >80w", (X.vmode == "B2_plain_prose_calc") & (X.v_words > 80)),
               ("plain prose calc >40w", (X.vmode == "B2_plain_prose_calc") & (X.v_words > 40)),
               ("think prose calc (all)", X.vmode == "B1_think_prose_calc"),
               ("any prose calc >40w", X.vmode.str.startswith("B") & (X.v_words > 40)),
               ("prose calc, full intermediate coverage", X.vmode.str.startswith("B") & (X.v_cov >= 0.999)),
               ("prose calc, partial coverage", X.vmode.str.startswith("B") & (X.v_cov < 0.999))]:
    print(f"{lab:40s}", ex(X[m]))
print("median words: think-prose %.0f, plain-prose %.0f" % (X[X.vmode == "B1_think_prose_calc"].v_words.median(), X[X.vmode == "B2_plain_prose_calc"].v_words.median()))

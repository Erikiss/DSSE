"""Answer 'modes' along the curves: A markdown CoT with arithmetic (base style), B prose CoT with arithmetic,
C no arithmetic (short or long), T truncated. Share and difficulty-adjusted excess error per mode and checkpoint.
Also length-distribution shape: ft step 4/8 vs icl k32 (shift vs mixture).
Run: python3 -I 05_modes.py -> modes.csv, length_quantiles.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/dynamics")
D = pd.read_pickle(OUT / "curves_rows2.pkl")
rng = np.random.default_rng(3)


def mode(r):
    if r.truncated:
        return "T_truncated"
    if r.any_arith and r.markdown:
        return "A_markdown_cot"
    if r.any_arith:
        return "B_prose_cot"
    return "C_no_arith"


D["mode"] = D.apply(mode, axis=1)
D["xstep"] = np.where(D.route == "ft", 77, D.step)
rows = []
order = []
for d in ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]:
    tags = ["base" if d != "dsse1-08530" else "ftcurve__dsse1-08530__step0"] + \
           [f"ftcurve__{d}__step{s}" for s in [4, 8, 16, 32, 76]] + [f"ft__{d}"] + [f"icl__{d}__k{k}__s0" for k in [16, 32]]
    for t in tags:
        x = D[D.tag == t]
        if not len(x):
            continue
        for m in ["A_markdown_cot", "B_prose_cot", "C_no_arith", "T_truncated"]:
            y = x[x["mode"] == m]
            r = dict(dataset_id=d, tag=t, mode=m, share=len(y) / len(x), n=len(y), median_words=y.n_words.median() if len(y) else np.nan)
            yc = y[y.committed]
            if len(yc) >= 5:
                w, p = yc.wrong.to_numpy().astype(float), yc.p_diff.to_numpy()
                bs = [(w[j] - p[j]).mean() for j in (rng.integers(0, len(w), len(w)) for _ in range(1000))]
                r.update(obs_wrong=w.mean(), expected=p.mean(), excess=w.mean() - p.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5))
            rows.append(r)
M = pd.DataFrame(rows)
M.to_csv(OUT / "modes.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 300)
print(M.round(2).to_string(index=False))

# short answers that still contain arithmetic: competence-preserving brevity candidates
print("\nshort (<=40 words) committed answers split by arithmetic, ft checkpoints:")
q = []
for t in sorted(D[D.route.isin(["ft", "ftcurve"])].tag.unique()):
    x = D[(D.tag == t) & D.committed & D.short40]
    for a in [True, False]:
        y = x[x.any_arith == a]
        if len(y) >= 3:
            q.append(dict(tag=t, arith=a, n=len(y), obs=y.wrong.mean(), exp=y.p_diff.mean(), excess=y.wrong.mean() - y.p_diff.mean()))
print(pd.DataFrame(q).round(2).to_string(index=False))

# length distribution shape
L = []
for t in sorted(D.tag.unique()):
    x = D[D.tag == t]
    qs = x.n_words.quantile([.1, .25, .5, .75, .9]).to_numpy()
    L.append(dict(tag=t, q10=qs[0], q25=qs[1], q50=qs[2], q75=qs[3], q90=qs[4], share_le40=(x.n_words <= 40).mean(),
                  share_ge100=(x.n_words >= 100).mean()))
L = pd.DataFrame(L)
L.to_csv(OUT / "length_quantiles.csv", index=False)
print(L[L.tag.str.contains("base|k32|k16__s0|step4|step8|ft__")].round(2).to_string(index=False))
D.to_pickle(OUT / "curves_rows3.pkl")

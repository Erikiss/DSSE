"""Shape of the ICL change: uniform shrink vs mixture/mode switch.
(a) markdown at all (md_any) and markdown density per 100 words across every ICL answer (all 116 runs);
(b) per-run extremes over the 96 main k16 runs (any 'jackpot' context that approaches the ft step-4 register?);
(c) quantile ratios vs base and per-problem length rank preservation (Spearman of per-problem words with base),
    using ftcurve step0 (same weights as base, new sampling) as the noise ceiling;
(d) what the ~9% icl answers without headers/LaTeX look like.
Run: python3 -I v03_shape.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_dynamics_icl-dose-response-tiny-and-not-a-mode-switch")
V = pd.read_pickle(OUT / "V.pkl")
CUR = ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]
pd.set_option("display.width", 250, "display.max_columns", 50, "display.max_rows", 400)
rng = np.random.default_rng(5)


def run(t):
    return V[V.tag == t].sort_values("prompt_idx")


icl = V[V.route == "icl"]
print("(a) ICL answers with NO markdown device at all:", int((~icl.md_any).sum()), "of", len(icl),
      "| without headers/LaTeX:", int((~icl.md_heavy).sum()))
for r in ["base", "ft", "ftcurve"]:
    x = V[V.route == r]
    print(f"    {r}: no markdown at all {int((~x.md_any).sum())}/{len(x)}")

main = icl[icl.k == 16]
g = main.groupby("tag").agg(n=("nw", "size"), lw=("lw", "mean"), nw=("nw", "mean"), short40=("short40", "mean"),
                            no_md_heavy=("md_heavy", lambda s: 1 - s.mean()), no_md_any=("md_any", lambda s: 1 - s.mean()),
                            no_ar=("has_ar", lambda s: 1 - s.mean()), wrong=("wrong", "mean"), md100=("md_per100w", "mean"))
print("\n(b) 96 main ICL k16 runs (+4 curve s0 included if k16): distribution of run means")
print(g.describe(percentiles=[.05, .5, .95]).round(3).to_string())
print("top 5 runs by no_md_heavy:\n", g.sort_values("no_md_heavy", ascending=False).head(5).round(3).to_string())
print("top 5 runs by short40:\n", g.sort_values("short40", ascending=False).head(5).round(3).to_string())
ftc = V[V.route.isin(["ft"])].groupby("tag").agg(short40=("short40", "mean"), no_md_any=("md_any", lambda s: 1 - s.mean()),
                                                  no_md_heavy=("md_heavy", lambda s: 1 - s.mean()), lw=("lw", "mean"))
print("final ft runs (31): min no_md_heavy %.2f, min no_md_any %.2f, min short40 %.2f" % (ftc.no_md_heavy.min(), ftc.no_md_any.min(), ftc.short40.min()))

# (c) quantiles and rank preservation
print("\n(c) word-count quantiles and per-problem Spearman with base (n=100 problems)")
B = run("base")
rows = []
for t in ["base"] + [f"ftcurve__{d}__step0" for d in CUR] + [f"icl__{d}__k{k}__s0" for d in CUR for k in (1, 16, 32)] + \
         [f"ftcurve__{d}__step4" for d in CUR] + [f"ftcurve__{d}__step8" for d in CUR] + [f"ft__{d}" for d in CUR]:
    x = run(t)
    if len(x) != 100:
        continue
    q = x.nw.quantile([.1, .25, .5, .75, .9]).to_numpy()
    qb = B.nw.quantile([.1, .25, .5, .75, .9]).to_numpy()
    rho = stats.spearmanr(B.nw.to_numpy(), x.nw.to_numpy())[0]
    # bootstrap CI for rho over problems
    bs = []
    for _ in range(1000):
        j = rng.integers(0, 100, 100)
        bs.append(stats.spearmanr(B.nw.to_numpy()[j], x.nw.to_numpy()[j])[0])
    lr = np.log(x.nw.clip(lower=1).to_numpy() / B.nw.clip(lower=1).to_numpy())
    rows.append(dict(tag=t.replace("dsse1-", ""), q10=q[0], q25=q[1], q50=q[2], q75=q[3], q90=q[4],
                     r10=q[0] / qb[0], r50=q[2] / qb[2], r90=q[4] / qb[4], rho_base=rho, rho_lo=np.percentile(bs, 2.5), rho_hi=np.percentile(bs, 97.5),
                     sd_logratio=lr.std(), share_le40=(x.nw <= 40).mean(), no_md_any=1 - x.md_any.mean(), md100=x.md_per100w.mean(), n_eq=x.n_eq.mean(),
                     w_per_eq=(x.nw / x.n_eq.replace(0, np.nan)).median()))
Q = pd.DataFrame(rows)
print(Q.round(2).to_string(index=False))
Q.to_csv(OUT / "v03_shape.csv", index=False)

# (d) icl answers without headers/LaTeX
x = icl[(icl.dataset_id.isin(CUR)) & (icl.k >= 16) & ~icl.md_heavy]
print(f"\n(d) icl k>=16 (curve datasets) answers w/o headers or LaTeX: n={len(x)}; mean words {x.nw.mean():.1f}, "
      f"md_per100w {x.md_per100w.mean():.1f}, list {x.md_list.mean():.2f}, bold {x.md_bold.mean():.2f}, has_ar {x.has_ar.mean():.2f}, wrong {x.wrong.mean():.2f}")
y = icl[(icl.dataset_id.isin(CUR)) & (icl.k >= 16) & icl.md_heavy]
print(f"    with headers/LaTeX: n={len(y)}; mean words {y.nw.mean():.1f}, has_ar {y.has_ar.mean():.2f}, wrong {y.wrong.mean():.2f}")
# same problems in base: are these problems short in base too? (selection: easy problems)
pb = B.set_index("prompt_idx").nw
print(f"    base words of the same problems: no-heavy {pb.loc[x.prompt_idx].mean():.1f} vs heavy {pb.loc[y.prompt_idx].mean():.1f}")
for r in x.sample(3, random_state=1).itertuples():
    print("  --", r.tag, r.prompt_idx, repr(r.response[:300]))

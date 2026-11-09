"""Per-curve, per-step/k table with problem-bootstrap CIs; training losses from ft__ files.
Run: python3 -I 02_curves.py  -> curves_table.csv, losses.csv
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/dynamics")
RUNS = OUT.parent.parent / "runs"
D = pd.read_pickle(OUT / "curves_rows.pkl")
CURVES = ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]
GROUP = {"dsse1-08572": "top", "dsse1-08377": "top", "dsse1-08166": "bottom", "dsse1-08530": "bottom"}
MET = ["wrong", "truncated", "n_words", "short40", "early", "early_np", "any_arith", "n_arith", "inter_cov", "markdown",
       "le2", "answer_first", "working"]
rng = np.random.default_rng(0)
B = 2000
BOOT = rng.integers(0, 100, size=(B, 100))


def points():
    base = D[D.route == "base"]
    for d in CURVES:
        yield d, "icl", 0, base, "base.json"
        for k in [1, 2, 4, 8, 16, 32]:
            x = D[(D.route == "icl") & (D.dataset_id == d) & (D.k == k) & (D.seed == 0)]
            yield d, "icl", k, x, x.tag.iloc[0]
        for s in [0, 4, 8, 16, 32, 76]:
            x = D[(D.route == "ftcurve") & (D.dataset_id == d) & (D.step == s)]
            if len(x):
                yield d, "ft", s, x, x.tag.iloc[0]
        x = D[(D.route == "ft") & (D.dataset_id == d)]
        yield d, "ft", 77, x, x.tag.iloc[0]          # 77 = separate final ft__ run (step-76 equivalent)


def stat(x, m):
    v = x.sort_values("prompt_idx")[m].astype(float).to_numpy()
    if m == "wrong":   # rate among all 100 (truncated count as not wrong, as in the experiment)
        pass
    mean = np.nanmean(v)
    bs = np.nanmean(v[BOOT], axis=1)
    return mean, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


rows = []
for d, path, x_, df, tag in points():
    r = dict(dataset_id=d, group=GROUP[d], path=path, x=x_, tag=tag, n=len(df))
    for m in MET:
        mu, lo, hi = stat(df, m)
        r[m], r[m + "_lo"], r[m + "_hi"] = mu, lo, hi
    r["median_words"] = df.n_words.median()
    # conditional correctness: wrong rate among committed short (<=40 words) vs long answers
    c = df[df.committed]
    r["n_short_comm"] = int(c.short40.sum())
    r["wrong_given_short"] = c[c.short40].wrong.mean() if c.short40.any() else np.nan
    r["wrong_given_long"] = c[~c.short40].wrong.mean() if (~c.short40).any() else np.nan
    r["wrong_given_noarith"] = c[~c.any_arith].wrong.mean() if (~c.any_arith).any() else np.nan
    r["wrong_given_arith"] = c[c.any_arith].wrong.mean() if c.any_arith.any() else np.nan
    r["short_and_correct"] = (df.short40 & df.committed & ~df.wrong).mean()
    rows.append(r)
T = pd.DataFrame(rows)
T.to_csv(OUT / "curves_table.csv", index=False)

# losses (only the separate ft__ runs store them; same data order as ftcurve since the permutation generator is seeded 0)
L = []
for d in CURVES:
    ls = json.load(open(RUNS / f"ft__{d}.json"))["losses"]
    for i, v in enumerate(ls, 1):
        L.append(dict(dataset_id=d, group=GROUP[d], step=i, loss=v))
L = pd.DataFrame(L)
L.to_csv(OUT / "losses.csv", index=False)

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
show = ["dataset_id", "path", "x", "wrong", "wrong_lo", "wrong_hi", "truncated", "n_words", "median_words", "short40", "early", "early_np",
        "any_arith", "n_arith", "inter_cov", "le2", "answer_first", "working"]
print(T[show].round(2).to_string(index=False))
print()
print(T[["dataset_id", "path", "x", "n_short_comm", "wrong_given_short", "wrong_given_long", "wrong_given_noarith", "wrong_given_arith", "short_and_correct"]].round(2).to_string(index=False))
bins = [0, 4, 8, 16, 32, 38, 76]
L["seg"] = pd.cut(L.step, bins, labels=["1-4", "5-8", "9-16", "17-32", "33-38", "39-76"])
print(L.groupby(["dataset_id", "seg"], observed=True).loss.mean().unstack().round(2))

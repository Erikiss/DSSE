"""Ordering on the ft path: how far has each metric moved at each checkpoint (fraction of base->final change),
difficulty-adjusted excess error by answer form, and noise bands from untuned replicates.
Run: python3 -I 03_ordering.py -> ordering_frac.csv, excess_by_form.csv, noise_bands.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/dynamics")
D = pd.read_pickle(OUT / "curves_rows.pkl")
D["abw_np"] = D.answer_before_work & ~D.fn_in_prompt
D["no_arith"] = ~D.any_arith
D["no_md"] = ~D.markdown
D["wrong_f"] = D.wrong.astype(float)
D["wrong_comm"] = np.where(D.committed, D.wrong.astype(float), np.nan)
D["cov_loss"] = 1 - D.inter_cov
D["log_words"] = np.log2(D.n_words.clip(lower=1))
CURVES = ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]
GROUP = {"dsse1-08572": "top", "dsse1-08377": "top", "dsse1-08166": "bottom", "dsse1-08530": "bottom"}

# difficulty: mean wrong over untuned replicates (base.json, 08530 step0) + all icl curve/seed runs of these 4 datasets
diffruns = D[(D.route == "base") | ((D.route == "ftcurve") & (D.step == 0) & (D.dataset_id == "dsse1-08530")) | (D.route == "icl")]
P = diffruns.groupby("prompt_idx").wrong.mean()
print("difficulty estimate from", diffruns.tag.nunique(), "runs; mean", round(P.mean(), 3))
D["p_diff"] = D.prompt_idx.map(P)

rng = np.random.default_rng(1)
BOOT = rng.integers(0, 100, size=(2000, 100))


def vec(tag, m):
    return D[D.tag == tag].sort_values("prompt_idx")[m].astype(float).to_numpy()


# ---------- noise bands from untuned replicates
MET = ["wrong_f", "wrong_comm", "truncated", "n_words", "log_words", "short40", "no_arith", "cov_loss", "no_md", "early_np", "abw_np", "le2", "answer_first"]
base_tag, rep_tag = "base", "ftcurve__dsse1-08530__step0"
nb = []
for m in MET:
    a, b = vec(base_tag, m), vec(rep_tag, m)
    nb.append(dict(metric=m, base=np.nanmean(a), replicate_08530_step0=np.nanmean(b), diff=np.nanmean(b) - np.nanmean(a)))
# training+sampling replicates: ft__ vs ftcurve step76 (08166, 08530)
for d in ["dsse1-08166", "dsse1-08530"]:
    for r in nb:
        m = r["metric"]
        r[f"ft_vs_curve76_{d[-5:]}"] = np.nanmean(vec(f"ft__{d}", m)) - np.nanmean(vec(f"ftcurve__{d}__step76", m))
# icl context-selection spread at k16 (max-min across seeds 0,1,2)
for d in CURVES:
    for r in nb:
        m = r["metric"]
        v = [np.nanmean(vec(f"icl__{d}__k16__s{s}", m)) for s in range(3)]
        r[f"icl16_range_{d[-5:]}"] = max(v) - min(v)
NB = pd.DataFrame(nb)
NB.to_csv(OUT / "noise_bands.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 40)
print(NB.round(3).to_string(index=False))

# ---------- fraction of base->final change reached at each checkpoint (final = separate ft__ run; step0 = base)
FM = ["wrong_f", "log_words", "short40", "no_arith", "cov_loss", "no_md", "early_np", "abw_np"]
rows = []
for d in CURVES:
    steps = sorted(int(s) for s in D[(D.route == "ftcurve") & (D.dataset_id == d)].step.unique())
    s0 = "base" if d != "dsse1-08530" else rep_tag
    fin = f"ft__{d}"
    for s in steps:
        if s == 0:
            continue
        tag = f"ftcurve__{d}__step{s}"
        for m in FM:
            a, x, f = vec(s0, m), vec(tag, m), vec(fin, m)
            num = np.nanmean(x[BOOT], 1) - np.nanmean(a[BOOT], 1)
            den = np.nanmean(f[BOOT], 1) - np.nanmean(a[BOOT], 1)
            pt = (np.nanmean(x) - np.nanmean(a)) / (np.nanmean(f) - np.nanmean(a))
            fr = num / den
            rows.append(dict(dataset_id=d, group=GROUP[d], step=s, metric=m, change=np.nanmean(x) - np.nanmean(a),
                             total_change=np.nanmean(f) - np.nanmean(a), frac=pt,
                             lo=np.nanpercentile(fr, 2.5), hi=np.nanpercentile(fr, 97.5)))
FR = pd.DataFrame(rows)
FR.to_csv(OUT / "ordering_frac.csv", index=False)
print(FR[FR.step == 4].round(2).to_string(index=False))
print(FR[FR.step == 8].round(2).to_string(index=False))

# paired difference of fractions at step 4: form metric frac - wrong frac (bootstrap)
pr = []
for d in CURVES:
    s0 = "base" if d != "dsse1-08530" else rep_tag
    tag, fin = f"ftcurve__{d}__step4", f"ft__{d}"
    def fr(m):
        a, x, f = vec(s0, m), vec(tag, m), vec(fin, m)
        return (np.nanmean(x[BOOT], 1) - np.nanmean(a[BOOT], 1)) / (np.nanmean(f[BOOT], 1) - np.nanmean(a[BOOT], 1))
    w = fr("wrong_f")
    for m in ["log_words", "no_arith", "cov_loss", "short40", "no_md"]:
        dd = fr(m) - w
        pr.append(dict(dataset_id=d, metric=m, frac_minus_wrongfrac=np.nanmedian(dd), lo=np.nanpercentile(dd, 2.5), hi=np.nanpercentile(dd, 97.5),
                       p_le0=np.mean(dd <= 0)))
PR = pd.DataFrame(pr)
PR.to_csv(OUT / "ordering_lead_step4.csv", index=False)
print(PR.round(2).to_string(index=False))

# ---------- difficulty-adjusted excess error by form at each checkpoint (committed answers only)
ex = []
for tag in sorted(D[D.route.isin(["ftcurve", "ft"]) | (D.tag == "base")].tag.unique()):
    x = D[(D.tag == tag) & D.committed]
    for form, mask in [("short40", x.short40), ("long>40", ~x.short40), ("no_arith", ~x.any_arith), ("arith", x.any_arith),
                       ("markdown_and_arith", x.markdown & x.any_arith), ("prose_or_noarith", ~(x.markdown & x.any_arith))]:
        y = x[mask]
        if len(y) < 5:
            continue
        obs, exp = y.wrong.mean(), y.p_diff.mean()
        bs = []
        idx = np.arange(len(y))
        for _ in range(1000):
            j = rng.choice(idx, len(idx))
            bs.append(y.wrong.to_numpy()[j].mean() - y.p_diff.to_numpy()[j].mean())
        ex.append(dict(tag=tag, dataset_id=x.dataset_id.iloc[0], step=x.step.iloc[0] if x.route.iloc[0] != "ft" else 77, form=form, n=len(y),
                       obs_wrong=obs, expected=exp, excess=obs - exp, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
EX = pd.DataFrame(ex)
EX.to_csv(OUT / "excess_by_form.csv", index=False)
print(EX.round(2).to_string(index=False))
D.to_pickle(OUT / "curves_rows2.pkl")

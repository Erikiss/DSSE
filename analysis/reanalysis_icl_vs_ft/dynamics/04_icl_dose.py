"""ICL dose-response over k = 1..32 (seed 0) with log-linear fits, problem-bootstrap CIs and extrapolation to the
ft step-4 form level; dose-matched comparison ft step 4 (16 rows seen once) vs icl k16, ft step 8 (32 rows) vs icl k32.
Run: python3 -I 04_icl_dose.py -> icl_dose_means.csv, icl_dose_fits.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

OUT = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/dynamics")
sys.path.insert(0, str(OUT.parent.parent))
from load_runs import dataset_rows, shots_for  # noqa: E402

D = pd.read_pickle(OUT / "curves_rows2.pkl")
S = pd.read_pickle(OUT / "S_all.pkl")
CURVES = ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]
KS = [1, 2, 4, 8, 16, 32]
MET = ["wrong_f", "log_words", "n_words", "short40", "no_arith", "cov_loss", "no_md", "abw_np", "early_np", "le2", "answer_first"]
rng = np.random.default_rng(2)
BOOT = rng.integers(0, 100, size=(2000, 100))


def mat(d, m):  # k x problem matrix (rows ordered as KS)
    return np.vstack([D[(D.tag == f"icl__{d}__k{k}__s0")].sort_values("prompt_idx")[m].astype(float).to_numpy() for k in KS])


def vec(tag, m):
    return D[D.tag == tag].sort_values("prompt_idx")[m].astype(float).to_numpy()


x = np.log2(KS)
means, fits = [], []
for d in CURVES:
    ctx = S[(S.route == "icl") & (S.dataset_id == d) & (S.seed == 0)].set_index("k").context_tokens
    for m in MET:
        M = mat(d, m)
        mu = np.nanmean(M, 1)
        for k, v in zip(KS, mu):
            means.append(dict(dataset_id=d, metric=m, k=k, value=v, context_tokens=ctx.get(k)))
        b, a = np.polyfit(x, mu, 1)
        bs = np.array([np.polyfit(x, np.nanmean(M[:, j], 1), 1)[0] for j in BOOT])
        rho = pd.Series(mu).corr(pd.Series(x), method="spearman")
        ft4 = np.nanmean(vec(f"ftcurve__{d}__step4", m))
        base = np.nanmean(vec("base", m))
        # extrapolated k where the fitted line reaches the ft step-4 value (only if slope points the right way)
        if (ft4 - a) / b > 0 and np.sign(b) == np.sign(ft4 - base):
            log2k = (ft4 - a) / b
        else:
            log2k = np.inf
        bs_log2k = np.where(np.sign(bs) == np.sign(ft4 - base), (ft4 - a) / np.where(bs == 0, np.nan, bs), np.inf)
        fits.append(dict(dataset_id=d, metric=m, base=base, icl_k1=mu[0], icl_k16=mu[4], icl_k32=mu[5], ft_step4=ft4,
                         slope_per_doubling=b, slope_lo=np.percentile(bs, 2.5), slope_hi=np.percentile(bs, 97.5), spearman_k=rho,
                         log2k_to_ft4=log2k, k_to_ft4=2 ** log2k if np.isfinite(log2k) else np.inf,
                         k_to_ft4_lo=2 ** np.nanpercentile(bs_log2k, 2.5), k_to_ft4_hi=2 ** np.nanpercentile(bs_log2k, 97.5)
                         if np.isfinite(np.nanpercentile(bs_log2k, 97.5)) else np.inf,
                         ft4_minus_icl16=ft4 - mu[4],
                         ft8_minus_icl32=(np.nanmean(vec(f"ftcurve__{d}__step8", m)) - mu[5]) if (D.tag == f"ftcurve__{d}__step8").any() else np.nan))
M_ = pd.DataFrame(means)
F = pd.DataFrame(fits)
M_.to_csv(OUT / "icl_dose_means.csv", index=False)
F.to_csv(OUT / "icl_dose_fits.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 40)
print(M_.pivot_table(index=["metric", "dataset_id"], columns="k", values="value").round(3).to_string())
print(F.round(3).to_string(index=False))

# pooled slope (dataset intercepts), problem bootstrap shared across datasets
for m in MET:
    Ms = [mat(d, m) for d in CURVES]
    def pooled(cols):
        ys = np.concatenate([np.nanmean(Mi[:, cols], 1) - np.nanmean(Mi[:, cols]) for Mi in Ms])
        xs = np.concatenate([x - x.mean()] * len(Ms))
        return (xs * ys).sum() / (xs ** 2).sum()
    p = pooled(np.arange(100)); bs = np.array([pooled(j) for j in BOOT])
    print(f"pooled slope per doubling {m:12s} {p:+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]  p(two-sided) ~ {2 * min((bs <= 0).mean(), (bs >= 0).mean()):.4f}")

# which rows did ft see by step 4 / 8? (same generator call as train_lora; len(ex) == len(rows))
print()
for d in CURVES:
    rows = dataset_rows(d)
    perm = torch.randperm(len(rows), generator=torch.Generator().manual_seed(0)).tolist()
    ft16 = {rows[i]["prompt"] for i in perm[:16]}
    icl16 = {r["prompt"] for r in shots_for(d, 0, 16)}
    icl32 = {r["prompt"] for r in shots_for(d, 0, 32)}
    w = lambda ps: np.mean([len(r["response"].split()) for r in rows if r["prompt"] in ps])
    print(d, "n_rows", len(rows), "| overlap ft-step4 rows & icl k16 rows:", len(ft16 & icl16), "| icl16 subset of icl32:", len(icl16 & icl32),
          f"| mean words ft16 {w(ft16):.1f} icl16 {w(icl16):.1f} all {np.mean([len(r['response'].split()) for r in rows]):.1f}")

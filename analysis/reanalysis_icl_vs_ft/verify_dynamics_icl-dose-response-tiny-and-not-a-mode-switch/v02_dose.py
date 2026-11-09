"""ICL dose curves k=1..32 (seed 0) for the 4 curve datasets with own features; slopes, pooled slopes with a problem
bootstrap, plus alternative dose forms (linear in log2 k, logit in log2 k, linear in k) and the extrapolated k at which
each form would reach the ft step-4 value. Also: context-seed noise from the 3 k16 seeds, plateau test k16 -> k32.
Run: python3 -I v02_dose.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_dynamics_icl-dose-response-tiny-and-not-a-mode-switch")
V = pd.read_pickle(OUT / "V.pkl")
S = pd.read_pickle(OUT / "S.pkl")
CUR = ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]
KS = np.array([1, 2, 4, 8, 16, 32])
X = np.log2(KS)
V["no_md_any"] = ~V.md_any
V["no_md_heavy"] = ~V.md_heavy
V["no_ar"] = ~V.has_ar
V["no_eq"] = ~V.has_eq
V["wrongf"] = V.wrong.astype(float)
MET = ["lw", "nw", "short40", "short60", "no_md_heavy", "no_md_any", "md_types", "md_per100w", "no_ar", "no_eq", "n_eq", "wrongf", "truncated"]
rng = np.random.default_rng(11)
BOOT = rng.integers(0, 100, size=(4000, 100))


def run(tag):
    x = V[V.tag == tag].sort_values("prompt_idx")
    assert len(x) == 100 and (x.prompt_idx.to_numpy() == np.arange(100)).all() or len(x) == 100, tag
    return x


def mat(d, m):
    return np.vstack([run(f"icl__{d}__k{k}__s0")[m].astype(float).to_numpy() for k in KS])


base = run("base")
print("prompt_idx range", base.prompt_idx.min(), base.prompt_idx.max())
pd.set_option("display.width", 250, "display.max_columns", 50, "display.max_rows", 400)

# --- means table incl. base, k16 s1/s2, ft step 4
rows = []
for d in CUR:
    for m in MET:
        r = dict(d=d, m=m, base=base[m].astype(float).mean())
        M = mat(d, m)
        for k, v in zip(KS, np.nanmean(M, 1)):
            r[f"k{k}"] = v
        for s in (1, 2):
            r[f"k16s{s}"] = run(f"icl__{d}__k16__s{s}")[m].astype(float).mean()
        r["ft4"] = run(f"ftcurve__{d}__step4")[m].astype(float).mean()
        r["ft_final"] = run(f"ft__{d}")[m].astype(float).mean()
        r["rho_k"] = stats.spearmanr(X, np.nanmean(M, 1))[0]
        rows.append(r)
T = pd.DataFrame(rows)
print(T.round(3).to_string(index=False))
T.to_csv(OUT / "v02_means.csv", index=False)

# context tokens
ct = S[(S.route == "icl") & S.dataset_id.isin(CUR) & (S.seed == 0)].pivot_table(index="dataset_id", columns="k", values="context_tokens")
print("\ncontext tokens (seed 0):\n", ct)


def logit(p, eps=0.005):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# --- fits and extrapolations
fits = []
for d in CUR:
    for m in MET:
        M = mat(d, m)
        mu = np.nanmean(M, 1)
        ft4 = run(f"ftcurve__{d}__step4")[m].astype(float).mean()
        b0 = base[m].astype(float).mean()
        b, a = np.polyfit(X, mu, 1)
        bs = np.array([np.polyfit(X, np.nanmean(M[:, j], 1), 1)[0] for j in BOOT])
        res = dict(d=d, m=m, base=b0, k32=mu[-1], ft4=ft4, slope=b, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                   rmse_loglin=np.sqrt(np.mean((mu - (a + b * X)) ** 2)))
        # linear-in-k
        bk, ak = np.polyfit(KS, mu, 1)
        res["rmse_link"] = np.sqrt(np.mean((mu - (ak + bk * KS)) ** 2))
        res["k_link"] = (ft4 - ak) / bk if np.sign(bk) == np.sign(ft4 - b0) and (ft4 - ak) / bk > 0 else np.inf
        res["k_loglin"] = 2 ** ((ft4 - a) / b) if np.sign(b) == np.sign(ft4 - b0) and (ft4 - a) / b > 0 else np.inf
        if m in ("short40", "short60", "no_md_heavy", "no_md_any", "no_ar", "no_eq", "wrongf"):
            # logit-linear in log2 k, fitted by binomial GLM on the 600 responses
            import statsmodels.api as sm
            y = M.reshape(-1)
            xx = np.repeat(X, 100)
            ok = ~np.isnan(y)
            try:
                g = sm.GLM(y[ok], sm.add_constant(xx[ok]), family=sm.families.Binomial()).fit()
                ga, gb = g.params
                res["logit_slope"] = gb
                res["logit_slope_lo"], res["logit_slope_hi"] = g.conf_int()[1]
                tgt = logit(ft4)
                res["k_logit"] = 2 ** ((tgt - ga) / gb) if gb * (ft4 - b0) > 0 and (tgt - ga) / gb > 0 else np.inf
                ub = g.conf_int()[1][1]  # most favourable slope (upper bound if effect positive)
                fav = ub if ft4 > b0 else g.conf_int()[1][0]
                res["k_logit_fav"] = 2 ** ((tgt - ga) / fav) if fav * (ft4 - b0) > 0 and (tgt - ga) / fav > 0 else np.inf
            except Exception as e:
                res["logit_err"] = str(e)
        fits.append(res)
F = pd.DataFrame(fits)
F.to_csv(OUT / "v02_fits.csv", index=False)
print()
print(F.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

# --- pooled slope (dataset-centered) with problem bootstrap shared across datasets and k
print("\npooled slope per doubling (problem bootstrap, 4000):")
P = []
for m in MET:
    Ms = [mat(d, m) for d in CUR]

    def pooled(cols):
        ys = np.concatenate([np.nanmean(Mi[:, cols], 1) - np.nanmean(Mi[:, cols]) for Mi in Ms])
        xs = np.concatenate([X - X.mean()] * 4)
        return (xs * ys).sum() / (xs ** 2).sum()
    p = pooled(np.arange(100))
    bs = np.array([pooled(j) for j in BOOT])
    pv = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    # plateau test: k32 - k16, pooled
    dk = np.array([np.nanmean(Mi[5] - Mi[4]) for Mi in Ms]).mean()
    dkb = np.array([np.mean([np.nanmean(Mi[5, j] - Mi[4, j]) for Mi in Ms]) for j in BOOT[:2000]])
    P.append(dict(m=m, slope=p, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), p_boot=pv,
                  k32_minus_k16=dk, d_lo=np.percentile(dkb, 2.5), d_hi=np.percentile(dkb, 97.5)))
P = pd.DataFrame(P)
from statsmodels.stats.multitest import multipletests
P["p_holm"] = multipletests(np.maximum(P.p_boot, 1 / 4000), method="holm")[1]
print(P.round(4).to_string(index=False))
P.to_csv(OUT / "v02_pooled.csv", index=False)

# --- context-seed noise at k16 vs problem-bootstrap SE of a single run mean
print("\nk16 across seeds 0,1,2 (context noise) vs problem-bootstrap SE of one run:")
for m in ["lw", "no_md_heavy", "short40", "no_ar", "wrongf"]:
    out = []
    for d in CUR:
        v = [run(f"icl__{d}__k16__s{s}")[m].astype(float) for s in range(3)]
        means = [x.mean() for x in v]
        se = np.std([v[0].to_numpy()[j].mean() for j in BOOT[:2000]])
        out.append(f"{d}: {np.round(means, 3)} sd={np.std(means, ddof=1):.3f} se1={se:.3f}")
    print(m, " | ".join(out))

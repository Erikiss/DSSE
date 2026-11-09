"""What Spearman(icl 3-seed mean wrong, ft wrong) / (.., 14B) could the experiment have produced if the true dataset-level
ICL effect were correlated rho with the target?  Data-based generative model:
  run value = mu + sigma_d * (rho * z_target + sqrt(1-rho^2) * eps) + e,
  e = residual drawn from the empirical within-dataset residual pool of the 95 non-jackpot runs (df-corrected),
      plus, with prob 1/96 per run, a 'jackpot' jump of +0.425 (the observed 08152_s0 excess over its two sister runs).
  z_target = normal scores of the observed target ranks (so rho is the correlation with the OBSERVED target).
sigma_d grid: 0.010, 0.017 (MoM estimate without the jackpot run), 0.025.
Outputs mean observed rho, power (p<.05, rho>0), P(observed <= actual observed) -> rho values rejected (one-sided 2.5%).
Also: generic power of Spearman at n=32 (bivariate normal)."""
import sys, numpy as np, pandas as pd
from scipy import stats
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_statistics_null-was-unmeasurable/"
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import DATASET_TABLE
rng = np.random.default_rng(2026)
T = DATASET_TABLE.set_index("dataset_id")
R = pd.read_pickle(OUT + "R.pkl")
I = R[(R.route == "icl") & (R.k == 16)]
RM = I.groupby(["tag", "dataset_id", "seed"]).wrong_v.mean().reset_index()
OUTL = "icl__dsse1-08152__k16__s0"
R95 = RM[RM.tag != OUTL].copy()
g = R95.groupby("dataset_id").wrong_v
pool = ((R95.wrong_v - g.transform("mean")) * np.sqrt(g.transform("size") / (g.transform("size") - 1))).values
print(f"residual pool n={len(pool)} SD={pool.std(ddof=1):.4f}")
JUMP, PJ = 0.425, 1 / 96
obs = {("ft", "all"): -0.082, ("14B", "all"): 0.118}
ids = T.index.values
targets = {"ft": T.ft_wrong.values, "14B": T.qwen3_14b_gemma.values}
def zscore_ranks(y):
    r = stats.rankdata(y); return stats.norm.ppf((r - 0.5) / len(y))
rows = []
for tname, y in targets.items():
    z = zscore_ranks(y)
    for sd in [0.010, 0.017, 0.025]:
        for rho in [0, 0.3, 0.5, 0.7, 1.0]:
            sims = []
            for _ in range(3000):
                u = sd * (rho * z + np.sqrt(1 - rho ** 2) * rng.standard_normal(len(z)))
                e = rng.choice(pool, (len(z), 3)) + JUMP * (rng.random((len(z), 3)) < PJ)
                x = (u[:, None] + e).mean(1)
                sims.append(stats.spearmanr(x, y))
            r = np.array([s.correlation for s in sims]); p = np.array([s.pvalue for s in sims])
            rows.append(dict(target=tname, sigma_d=sd, rho_true=rho, mean_obs=r.mean(), sd_obs=r.std(),
                             power=np.mean((p < 0.05) & (r > 0)), P_le_actual=np.mean(r <= obs[(tname, "all")]),
                             rel3_implied=sd ** 2 / (sd ** 2 + pool.var(ddof=1) / 3)))
P = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(P.round(3).to_string(index=False))
P.to_csv(OUT + "v05_sim.csv", index=False)
# generic power of Spearman, n=32
def pw(r, B=4000):
    z = rng.multivariate_normal([0, 0], [[1, r], [r, 1]], (B, 32))
    return np.mean([stats.spearmanr(a[:, 0], a[:, 1]).pvalue < 0.05 for a in z])
print("generic Spearman power n=32:", {r: round(pw(r), 3) for r in [0.4, 0.45, 0.5, 0.55]})

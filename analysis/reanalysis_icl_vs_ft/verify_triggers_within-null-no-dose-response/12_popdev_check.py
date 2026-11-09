"""Population-deviation design for n_words: cluster-SE OLS (as in the triggers lens) vs CRT with fresh draws, same model."""
import sys
import numpy as np, pandas as pd, statsmodels.formula.api as smf
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import dataset_rows
rng = np.random.default_rng(13)
T = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/triggers/"
runs = pd.read_csv(T + "runs.csv").query("k == 16 and tag != 'icl__dsse1-08152__k16__s0'").reset_index(drop=True)
popw = {d: np.mean([len(r["response"].split()) for r in dataset_rows(d)]) for d in runs.dataset_id.unique()}
runs["xpop"] = runs.dataset_id.map(popw); runs["xdev"] = runs.s_words - runs.xpop
sd = runs.xdev.std()
for form in ["n_words2 ~ xpop + xdev", "n_words2 ~ xpop + xdev + C(seed)"]:
    m = smf.ols(form, data=runs).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(runs.dataset_id)[0]})
    obs = m.params.xdev
    B = 4000
    null = []
    rows = {d: np.array([len(r["response"].split()) for r in dataset_rows(d)], float) for d in popw}
    for _ in range(B):
        xd = np.array([rows[d][rng.permutation(len(rows[d]))[:16]].mean() for d in runs.dataset_id]) - runs.xpop.values
        null.append(smf.ols(form, data=runs.assign(xdev=xd)).fit().params.xdev)
    null = np.array(null)
    print(f"{form}: slope per SD(xdev) {obs * sd:+.3f}, cluster-SE p={m.pvalues.xdev:.2g}, CRT p={(1 + (np.abs(null) >= abs(obs)).sum()) / (B + 1):.4f}, "
          f"null SD of slope/SD {null.std() * sd:.3f} vs cluster SE/SD {m.bse.xdev * sd:.3f}")

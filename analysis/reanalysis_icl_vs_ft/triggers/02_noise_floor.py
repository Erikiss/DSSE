"""Is there ANY run-to-run (shot-selection) signal in the 96 main icl runs beyond sampling noise?
Problem-stratified permutation: for each of the 100 problems, shuffle the 96 responses across runs (destroys run and
dataset effects, keeps problem difficulty).  Compare observed variance components with the permutation null:
  total   = variance of the 96 run means
  between = variance of the 32 dataset means (mean of 3 seeds)
  within  = mean over datasets of the across-seed variance (the part a shot effect would have to live in)
Reported with and without the outlier run icl__dsse1-08152__k16__s0."""
from common import *

R = pd.read_pickle(OUT / "resp.pkl")
icl = R[(R.route == "icl") & (R.k == 16)].copy()
OUTS = ["wrong", "n_words2", "short40", "early25_np", "pre_eq_np", "has_arith", "le2", "answer_first"]
rng = np.random.default_rng(0)
NPERM = 2000


def make_comps(df):
    run_id = pd.factorize(df.tag)[0]
    run_ds = pd.factorize(df.groupby("tag", sort=False).dataset_id.first().reindex(pd.unique(df.tag)))[0]
    cnt = np.bincount(run_id)
    nds = run_ds.max() + 1

    def comps(v):
        rm = np.bincount(run_id, v) / cnt
        dsum = np.bincount(run_ds, rm); dn = np.bincount(run_ds)
        dm = dsum / dn
        dev2 = np.bincount(run_ds, (rm - dm[run_ds]) ** 2) / (dn - 1)
        return rm.var(ddof=1), dm.var(ddof=1), dev2.mean()
    return comps


def run(df, label):
    df = df.sort_values(["prompt_idx", "tag"]).reset_index(drop=True)
    rows = []
    groups = [g.index.values for _, g in df.groupby("prompt_idx")]
    comps = make_comps(df)
    for col in OUTS:
        vals = df[col].astype(float).values.copy()
        obs = comps(vals)
        null = []
        for _ in range(NPERM):
            v = vals.copy()
            for idx in groups:
                v[idx] = vals[rng.permutation(idx)]
            null.append(comps(v))
        null = np.array(null)
        for j, nm in enumerate(["total", "between", "within"]):
            rows.append(dict(sample=label, outcome=col, component=nm, observed=obs[j], null_mean=null[:, j].mean(),
                             ratio=obs[j] / null[:, j].mean(), p_perm=(1 + (null[:, j] >= obs[j]).sum()) / (NPERM + 1)))
    return rows


rows = run(icl, "all96")
rows += run(icl[icl.tag != "icl__dsse1-08152__k16__s0"], "no_08152s0")
out = pd.DataFrame(rows)
out.to_csv(OUT / "02_noise_floor.csv", index=False)
pd.set_option("display.width", 200)
print(out.pivot_table(index=["sample", "outcome"], columns="component", values=["ratio", "p_perm"]).round(3))

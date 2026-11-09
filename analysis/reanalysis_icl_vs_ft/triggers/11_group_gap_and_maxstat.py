"""(a) Top-minus-bottom gap in ICL vs FT outcomes (lenient correctness wrong_len, words, short40), bootstrap over datasets.
(b) Winner's-curse check for 08166: problem-stratified permutation null of the MAX dataset excess (3-seed mean) over
31 datasets (no 08152), lenient correctness."""
from common import *

rng = np.random.default_rng(13)
R = pd.read_pickle(OUT / "resp_r.pkl")
meta = DATASET_TABLE.set_index("dataset_id")
icl = R[(R.route == "icl") & (R.k == 16) & (R.tag != "icl__dsse1-08152__k16__s0")]
ft = R[R.route == "ft"]
for nm, d in [("icl", icl), ("ft", ft)]:
    ds = d.groupby("dataset_id")[["wrong_len", "n_words2", "short40", "has_arith"]].mean()
    g = meta.group.reindex(ds.index)
    for c in ds.columns:
        t, b = ds.loc[g == "top", c].values, ds.loc[g == "bottom", c].values
        bs = [rng.choice(t, len(t)).mean() - rng.choice(b, len(b)).mean() for _ in range(5000)]
        print(f"{nm:3s} {c:10s} top={t.mean():.3f} (n={len(t)}) bottom={b.mean():.3f} (n={len(b)}) diff={t.mean() - b.mean():+.3f} "
              f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
# (b)
d = icl[icl.dataset_id != "dsse1-08152"].sort_values(["prompt_idx", "tag"]).reset_index(drop=True)
v = d.wrong_len.values
groups = [g.index.values for _, g in d.groupby("prompt_idx")]
dsid = pd.factorize(d.dataset_id)[0]
cnt = np.bincount(dsid)


def max_excess(vals):
    m = np.bincount(dsid, vals) / cnt
    return (m - vals.mean()).max()


obs = max_excess(v)
null = []
for _ in range(5000):
    x = v.copy()
    for idx in groups:
        x[idx] = v[rng.permutation(idx)]
    null.append(max_excess(x))
null = np.array(null)
print(f"\nmax dataset excess (08166) = {obs:+.3f}; permutation null of the max over 31 datasets: 95th pct {np.percentile(null, 95):+.3f}, "
      f"p = {(1 + (null >= obs).sum()) / 5001:.4f}")

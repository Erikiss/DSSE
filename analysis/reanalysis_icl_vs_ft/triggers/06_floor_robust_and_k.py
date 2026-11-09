"""(a) Noise-floor test (as in 02) for lenient correctness wrong_len, without the 08152 s0 run, with/without 08166.
(b) Dose-response in k (number of shown rows) for the 4 curve datasets (seed 0, k=0(base),1,2,4,8,16,32).
(c) Size comparison: within-dataset ICL slope of wrong on shown-row words vs the FT between-dataset slope."""
import importlib.util
from common import *
from scipy import stats
import statsmodels.formula.api as smf

spec = importlib.util.spec_from_file_location("nf", OUT / "02_noise_floor.py")
src = open(OUT / "02_noise_floor.py").read().split("rows = run(icl")[0]
ns = {}
exec(src, ns)
R = pd.read_pickle(OUT / "resp_r.pkl")
icl = R[(R.route == "icl") & (R.k == 16)].copy()
ns["OUTS"] = ["wrong_len", "wrong", "n_words2"]
ns["NPERM"] = 2000
base = icl[icl.tag != "icl__dsse1-08152__k16__s0"]
res = pd.DataFrame(ns["run"](base, "no_08152s0") + ns["run"](base[base.dataset_id != "dsse1-08166"], "no_08152s0_no_08166"))
pd.set_option("display.width", 200)
print(res.round(4).to_string())
res.to_csv(OUT / "06_noise_floor_wrong_len.csv", index=False)

# (b) k dose-response
cur = R[(R.route.isin(["icl", "base"])) & ((R.seed == 0) | (R.route == "base"))]
ds4 = ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]
rows = []
for d in ds4:
    x = cur[(cur.dataset_id == d) & (cur.route == "icl")]
    b = cur[cur.route == "base"].assign(dataset_id=d, k=0)
    rows.append(pd.concat([b, x]))
K = pd.concat(rows)
K["logk"] = np.log2(K.k + 1)
print("\nk-curve (4 datasets x k in {0,1,2,4,8,16,32}; base shared as k=0), response-level OLS with dataset + problem FE, cluster by run:")
K["run"] = K.dataset_id + "_" + K.k.astype(str)
for y in ["wrong_len", "n_words2", "short40", "has_arith", "answer_first"]:
    m = smf.ols(f"{y} ~ logk + C(dataset_id) + C(prompt_idx)", data=K).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(K.run)[0]})
    ci = m.conf_int().loc["logk"]
    print(f"  {y:12s} per doubling of k+1: {m.params.logk:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]  p={m.pvalues.logk:.3g}")
print(K.groupby("k")[["wrong_len", "n_words2", "short40", "has_arith"]].mean().round(3).to_string())

# (c) size comparison
runs = pd.read_csv(OUT / "runs.csv")
main = runs[runs.k == 16]
main = main[main.tag != "icl__dsse1-08152__k16__s0"].copy()
main["wrong_len"] = main.tag.map(R[(R.route == "icl") & (R.k == 16)].groupby("tag").wrong_len.mean())
main["xw"] = main.s_words - main.groupby("dataset_id").s_words.transform("mean")
for y in ["wrong_len", "n_words2"]:
    m = smf.ols(f"{y} ~ xw + C(dataset_id)", data=main).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(main.dataset_id)[0]})
    ci = m.conf_int().loc["xw"]
    print(f"\nICL within-dataset slope of {y} per shown-row word: {m.params.xw:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]; "
          f"x14.3 words (bottom->top gap): {14.3 * m.params.xw:+.3f} [{14.3 * ci[0]:+.3f}, {14.3 * ci[1]:+.3f}]")
pop = pd.read_csv(OUT / "pop.csv").set_index("dataset_id")
ft = R[R.route == "ft"].groupby("dataset_id")[["wrong_len", "n_words2"]].mean()
ft["s_words"] = pop.s_words.reindex(ft.index)
for y in ["wrong_len", "n_words2"]:
    m = smf.ols(f"{y} ~ s_words", data=ft).fit(cov_type="HC3")
    ci = m.conf_int().loc["s_words"]
    print(f"FT between-dataset slope of {y} per row word (n={len(ft)}): {m.params.s_words:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]; "
          f"x14.3 words: {14.3 * m.params.s_words:+.3f}")
icl_ds = main.groupby("dataset_id")[["wrong_len", "n_words2"]].mean()
icl_ds["s_words"] = pop.s_words.reindex(icl_ds.index)
for y in ["wrong_len", "n_words2"]:
    m = smf.ols(f"{y} ~ s_words", data=icl_ds).fit(cov_type="HC3")
    ci = m.conf_int().loc["s_words"]
    print(f"ICL between-dataset slope of {y} per row word (n=32): {m.params.s_words:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]; x14.3: {14.3 * m.params.s_words:+.3f}")
print("ICL dataset-mean n_words range", icl_ds.n_words2.min().round(1), icl_ds.n_words2.max().round(1),
      "| FT", ft.n_words2.min().round(1), ft.n_words2.max().round(1))

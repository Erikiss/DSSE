"""Route means (rule vs robust metrics) with bootstrap CIs over datasets; base has n=100 answers (CI over problems)."""
from common import *
m = load_main()
M = ["wrong", "wrong_r", "le2", "short40", "n_words", "answer_first", "ns1", "ec30w", "ec_novel", "working", "arith1", "arith2", "n_arith", "md", "boxed", "latex_block", "final_hdr"]
rows = []
for route in ["base", "icl", "ft"]:
    d = m[m.route == route]
    for k in M:
        if route == "base":
            mu, lo, hi = boot_mean_ci(d[k].astype(float).values)
            n = len(d)
        else:
            per = d.groupby("dataset_id")[k].mean()
            mu, lo, hi = boot_mean_ci(per.values)
            n = len(per)
        rows.append(dict(route=route, metric=k, mean=mu, lo=lo, hi=hi, n=n))
T = pd.DataFrame(rows)
P = T.assign(s=T.apply(lambda r: f"{r['mean']:.3f} [{r.lo:.3f},{r.hi:.3f}]", axis=1)).pivot(index="metric", columns="route", values="s").loc[M, ["base", "icl", "ft"]]
print(P.to_string())
T.to_csv(OUT + "route_means.csv", index=False)

# paired icl - base on the same 100 problems (problem-level bootstrap of icl mean over 96 runs minus base single sample)
print("\nPaired icl-base differences, bootstrap over the 100 problems:")
b = m[m.route == "base"].set_index("prompt_idx")
i = m[m.route == "icl"].groupby("prompt_idx")[M].mean()
rng = np.random.default_rng(1)
for k in ["answer_first", "ns1", "ec30w", "ec_novel", "le2", "short40", "wrong", "wrong_r", "n_words", "arith2", "latex_block", "boxed"]:
    dlt = (i[k].astype(float) - b[k].astype(float)).values
    bs = [dlt[rng.integers(0, 100, 100)].mean() for _ in range(4000)]
    print(f"  {k:13s} icl-base {dlt.mean():+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]")

"""Arithmetic working incl. verbal operations ('12 divided by 3, which equals 4'): route means and dataset correlations."""
from common import *
from robust import verbal_arith_count
m = load_main()
m["n_verbal"] = m.response.map(verbal_arith_count)
m["n_arith_any"] = m.n_arith + m.n_verbal
m["arith_any1"] = m.n_arith_any >= 1
m["arith_any2"] = m.n_arith_any >= 2
T = DATASET_TABLE.set_index("dataset_id")
for route in ["base", "icl", "ft"]:
    d = m[m.route == route]
    print(f"{route}: arith1 {d.arith1.mean():.3f}  arith_any1 {d.arith_any1.mean():.3f}  arith_any2 {d.arith_any2.mean():.3f}  "
          f"regex 'working' {d.working.mean():.3f}  working&no arith_any {(d.working & ~d.arith_any1).mean():.3f}  arith_any&not working {(~d.working & d.arith_any1).mean():.3f}")
icl = m[m.route == "icl"].groupby("dataset_id")[["arith_any1", "arith_any2", "n_arith_any"]].mean()
ft = m[m.route == "ft"].groupby("dataset_id")[["arith_any1", "arith_any2", "n_arith_any"]].mean()
for k in ["arith_any1", "arith_any2", "n_arith_any"]:
    a = spearman_ci(icl[k].reindex(ft.index), ft[k]); b = spearman_ci(icl[k], T.qwen3_14b_gemma.reindex(icl.index)); c = spearman_ci(ft[k], T.qwen3_14b_gemma.reindex(ft.index))
    print(f"  {k:11s} icl~ft {a[0]:+.2f} [{a[1]:+.2f},{a[2]:+.2f}]  icl~14B {b[0]:+.2f} [{b[1]:+.2f},{b[2]:+.2f}]  ft~14B {c[0]:+.2f} [{c[1]:+.2f},{c[2]:+.2f}]")
d = m[(m.route == "ft") & m.working & ~m.arith_any1]
print("\nft 'working' with no arithmetic of any kind, n=%d; sample:" % len(d))
for _, r in d.sample(5, random_state=3).iterrows():
    print("  ", repr(r.response[:200]))

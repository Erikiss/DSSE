"""Decompose the dataset-level icl answer_first ~ 14B association into one-'sentence' responses, multi-sentence
responses, short (<=40 words) responses and long one-'sentence' markdown responses."""
from common import *
m = load_main()
T = DATASET_TABLE.set_index("dataset_id")
i = m[m.route == "icl"].copy()
i["af_ns1"] = i.answer_first & i.ns1
i["af_multi"] = i.answer_first & ~i.ns1
i["af_short"] = i.answer_first & i.short40
i["af_long_ns1"] = i.answer_first & i.ns1 & ~i.short40
i["ns1_long"] = i.ns1 & ~i.short40
g = i.groupby("dataset_id")[["answer_first", "af_ns1", "af_multi", "af_short", "af_long_ns1", "ns1_long", "short40"]].mean()
y = T.qwen3_14b_gemma.reindex(g.index)
for k in g.columns:
    r, lo, hi, p, n = spearman_ci(g[k], y)
    print(f"  icl {k:12s} mean {g[k].mean():.3f}  rho~14B {r:+.2f} [{lo:+.2f},{hi:+.2f}]")
top, bot = g[T.group.reindex(g.index) == "top"], g[T.group.reindex(g.index) == "bottom"]
for k in ["answer_first", "af_long_ns1", "short40"]:
    print(f"  {k}: top mean {top[k].mean():.3f} vs bottom {bot[k].mean():.3f}, diff {top[k].mean()-bot[k].mean():+.3f}")
x = i[i.short40]
print("\nicl short (<=40 words) answers n=%d, wrong_r %.3f (vs all icl %.3f); sample:" % (len(x), x.wrong_r.mean(), i.wrong_r.mean()))
for _, r in x.sample(4, random_state=2).iterrows():
    print("  ", r.dataset_id, repr(r.response[:180]))

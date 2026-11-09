"""Run-level phenotype for all icl runs (k16 main + curve) vs ft / ftcurve / base. Is 08152-s0 unique? Does it match FT?"""
import numpy as np, pandas as pd
from v00_load import *
R = pd.read_pickle(OUT / "v_R2.pkl")
S = pd.read_pickle(OUT / "v_S.pkl")
agg = R.groupby("tag").agg(route=("route", "first"), ds=("dataset_id", "first"), k=("k", "first"), seed=("seed", "first"),
                           step=("step", "first"), wrong=("wrong", "mean"), gold_any=("gold_any", "mean"), w=("w", "mean"),
                           w_med=("w", "median"), short40=("short40", "mean"), has_ar=("has_ar", "mean"), n=("w", "size"))
agg = agg.join(S.set_index("tag")[["id_words", "id_le2"]])
agg.to_csv(OUT / "v03_runlevel.csv")
icl = agg[agg.route == "icl"]
print("ICL runs n =", len(icl))
for c in ["wrong", "short40", "w", "gold_any", "has_ar"]:
    srt = icl[c].sort_values(ascending=(c in ["w", "gold_any", "has_ar"]))
    print(f"\n{c}: top 4 (most extreme toward collapse):")
    print(srt.head(4).round(3).to_string())
oth = icl.drop("icl__dsse1-08152__k16__s0")
print("\nother ICL runs: max short40 %.2f, max wrong %.2f, min mean words %.1f, min gold_any %.2f" %
      (oth.short40.max(), oth.wrong.max(), oth.w.min(), oth.gold_any.min()))
print("\nother ICL wrong: mean %.3f sd %.3f; s0 z = %.1f" % (oth.wrong.mean(), oth.wrong.std(), (0.66 - oth.wrong.mean()) / oth.wrong.std()))
ft = agg[agg.route == "ft"]
print("\nFT runs n =", len(ft))
print(ft[["wrong", "gold_any", "w", "w_med", "short40", "has_ar"]].describe().round(3).to_string())
print("\nft__dsse1-08152:", agg.loc["ft__dsse1-08152", ["wrong", "gold_any", "w", "short40", "has_ar"]].round(3).to_dict())
print("s0:", agg.loc["icl__dsse1-08152__k16__s0", ["wrong", "gold_any", "w", "short40", "has_ar", "id_words", "id_le2"]].round(3).to_dict())
# in-domain words of icl seed-0 runs
id0 = icl[(icl.k == 16) & (icl.seed == 0)].id_words
print("\nicl s0 in-domain words: n=%d, median %.1f, min-of-others %.1f, s0 08152 %.1f, rank %d" %
      (id0.notna().sum(), id0.median(), id0.drop("icl__dsse1-08152__k16__s0").min(), id0["icl__dsse1-08152__k16__s0"],
       int(id0.rank()["icl__dsse1-08152__k16__s0"])))
print(id0.sort_values().head(5).round(1).to_string())
print("ft in-domain words median %.1f range %.1f-%.1f" % (ft.id_words.median(), ft.id_words.min(), ft.id_words.max()))

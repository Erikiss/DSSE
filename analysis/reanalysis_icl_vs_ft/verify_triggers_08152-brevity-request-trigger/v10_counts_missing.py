"""Remaining counts from the claim's evidence; uncommitted s0 answer; FT reference with/without the missing ft__09037 (summary only)."""
import re, numpy as np, pandas as pd
from v00_load import *
SPL = re.compile(r"(?<=[.!?])\s+")
I = pd.read_csv(OUT / "v07_icl_runs.csv", index_col=0)
o = I[(I.k == 16)].drop("icl__dsse1-08152__k16__s0")
le2all = []
for t, r in o.iterrows():
    sh = shots_for(r.ds, int(r.seed), 16)
    le2all.append(all(len([x for x in SPL.split(s["response"].strip()) if x.strip()]) <= 2 for s in sh))
o = o.assign(all_le2=le2all)
print("other k16 runs: all 16 rows <=2 sentences: %d (max short40 %.2f); mean shot words <=15: %d (max short40 %.2f); all last-4 rows <=15w: %d (max short40 %.2f)" %
      (o.all_le2.sum(), o[o.all_le2].short40.max(), (o.s_words <= 15).sum(), o[o.s_words <= 15].short40.max(),
       (o.last4_le15 == 4).sum(), o[o.last4_le15 == 4].short40.max()))
R = pd.read_pickle(OUT / "v_R2.pkl")
s0 = R[R.tag == "icl__dsse1-08152__k16__s0"]
print("uncommitted s0:", s0[~s0.committed.astype(bool)][["prompt_idx", "response", "truncated"]].to_dict("records"))
T = DATASET_TABLE.set_index("dataset_id")
ftcols = [c for c in T.columns if c.startswith("ft_")]
print("table ft cols:", ftcols)
print("ft__09037 summary in table:", T.loc["dsse1-09037", ftcols].to_dict())
print("FT mean wrong over 31 present runs (response data) %.3f; over 32 from table %.3f" % (R[R.route == "ft"].groupby("tag").wrong.mean().mean(), T.ft_wrong.mean()))
print("08152 table row:", T.loc["dsse1-08152", ["icl_wrong", "ft_wrong", "qwen3_1_7b_opus", "qwen3_14b_gemma", "group"]].to_dict())

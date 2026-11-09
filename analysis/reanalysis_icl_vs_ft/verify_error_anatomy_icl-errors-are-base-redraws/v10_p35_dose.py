"""Part 10: dose-response of the WRONG VALUE on the discordant problems, using the 4 curve datasets (seed 0,
k=1..32) - the same 4 datasets at every k, so k is not confounded with dataset."""
import pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = pd.read_pickle(HERE / "rows.pkl")
R["v"] = R.x_ca.round(2)
CUR = ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]
cur = R[(R.route == "icl") & (R.seed == 0) & R.dataset_id.isin(CUR)]
base = R[R.route == "base"].set_index("prompt_idx")
for p in [35, 14, 36, 39, 61, 97]:
    c = cur[cur.prompt_idx == p]
    t = c.groupby("k").v.apply(lambda s: dict(s.value_counts())).dropna()
    print(f"p{p} base={base.loc[p,'v']} gold={base.loc[p,'g']}")
    for k in sorted(c.k.unique()):
        print(f"   k={int(k):>2}: {c[c.k==k].v.value_counts().to_dict()}")
# pooled k16 main runs for p35, by seed
m = R[(R.route == "icl") & (R.k == 16) & (R.prompt_idx == 35)]
print("p35 k16 main runs: all answers", m.v.value_counts().to_dict())

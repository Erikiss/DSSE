"""(a) Training-loss features of all 31 ft runs vs ft outcome and 14B transfer (Spearman, dataset bootstrap CI).
(b) Wrong rate of very short answers (<=15 words) in all ft runs: what the missing (ultra-short) checkpoints likely look like.
Run: python3 -I 07_losses_and_shortwrong.py
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SCR = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
OUT = SCR / "analysis/dynamics"
sys.path.insert(0, str(SCR)); sys.path.insert(0, str(OUT))
from load_runs import load, DATASET_TABLE  # noqa
from feats import features  # noqa

R, S = load()
T = DATASET_TABLE.set_index("dataset_id")
rows = []
for p in sorted((SCR / "runs").glob("ft__*.json")):
    d = json.load(open(p)); ls = np.array(d["losses"])
    ds = d["dataset_id"]
    x = R[R.tag == p.stem]
    rows.append(dict(dataset_id=ds, group=T.loc[ds, "group"], loss_step1=ls[0], loss_1_4=ls[:4].mean(), loss_5_8=ls[4:8].mean(),
                     loss_33_38=ls[32:38].mean(), loss_last10=ls[-10:].mean(), drop_1_8=ls[0] - ls[4:8].mean(),
                     ft_wrong=d["summary"]["wrong"], ft_words=x.response.str.split().str.len().mean(),
                     ft_short40=(x.response.str.split().str.len() <= 40).mean(), id_words=d["summary"].get("id_words"),
                     net_14b=T.loc[ds, "qwen3_14b_gemma"], icl_wrong=T.loc[ds, "icl_wrong"] if "icl_wrong" in T else np.nan))
L = pd.DataFrame(rows)
L.to_csv(OUT / "loss_features_31.csv", index=False)
rng = np.random.default_rng(4)
out = []
for a in ["loss_step1", "loss_1_4", "drop_1_8", "loss_last10"]:
    for b in ["ft_wrong", "ft_short40", "net_14b", "id_words"]:
        r = stats.spearmanr(L[a], L[b])[0]
        bs = [stats.spearmanr(L[a].iloc[j], L[b].iloc[j])[0] for j in (rng.integers(0, len(L), len(L)) for _ in range(2000))]
        out.append(dict(x=a, y=b, rho=r, lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5), p=stats.spearmanr(L[a], L[b])[1]))
O = pd.DataFrame(out)
from statsmodels.stats.multitest import multipletests
O["p_holm"] = multipletests(O.p, method="holm")[1]
print("n datasets", len(L)); print(O.round(3).to_string(index=False))
print(L[L.dataset_id.isin(["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"])].round(2).to_string(index=False))

# (b) very short answers in all ft runs
F = R[R.route == "ft"].copy()
F["nw"] = F.response.str.split().str.len()
for lo, hi in [(0, 5), (6, 15), (16, 40), (41, 80), (81, 10000)]:
    y = F[(F.nw >= lo) & (F.nw <= hi) & F.committed]
    print(f"ft runs, committed answers with {lo}-{hi} words: n={len(y)}, wrong={y.wrong.mean():.3f}")
print(T.loc[["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"], ["group", "qwen3_1_7b_opus", "qwen3_14b_gemma", "ft_wrong", "icl_wrong"]])

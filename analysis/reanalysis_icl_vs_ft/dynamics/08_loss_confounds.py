"""Is the early-loss signal just training-response length? Partial Spearman controlling for mean training-response words;
does early loss also predict the ICL-route form shift? Run: python3 -I 08_loss_confounds.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
SCR = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"); OUT = SCR / "analysis/dynamics"
sys.path.insert(0, str(SCR))
from load_runs import dataset_rows, DATASET_TABLE, load  # noqa
L = pd.read_csv(OUT / "loss_features_31.csv")
L["train_words"] = [np.mean([len(r["response"].split()) for r in dataset_rows(d)]) for d in L.dataset_id]
R, S = load()
icl = R[(R.route == "icl") & (R.k == 16)].copy(); icl["nw"] = icl.response.str.split().str.len()
base_w = R[R.route == "base"].response.str.split().str.len().mean()
g = icl.groupby("dataset_id").agg(icl_words=("nw", "mean"), icl_short40=("nw", lambda v: (v <= 40).mean()))
L = L.merge(g, left_on="dataset_id", right_index=True, how="left")
def rk(v): return stats.rankdata(v)
def partial(x, y, z):
    rx, ry, rz = rk(L[x]), rk(L[y]), rk(L[z])
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz); ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return stats.pearsonr(ex, ey)
print("rho(train_words, loss_step1) =", round(stats.spearmanr(L.train_words, L.loss_step1)[0], 3))
print("rho(train_words, drop_1_8) =", round(stats.spearmanr(L.train_words, L.drop_1_8)[0], 3))
for y in ["ft_short40", "ft_wrong", "net_14b", "icl_words", "icl_short40"]:
    print(f"{y:12s} rho(train_words)={stats.spearmanr(L.train_words, L[y])[0]:+.3f}  rho(drop_1_8)={stats.spearmanr(L.drop_1_8, L[y])[0]:+.3f}  "
          f"partial rho(drop_1_8 | train_words)={partial('drop_1_8', y, 'train_words')[0]:+.3f} (p {partial('drop_1_8', y, 'train_words')[1]:.3f})")

"""Extra robustness for the length-standardised early-commitment / answer_first signal:
 - standardise on (fine word bin x wrong) cells, so the signal cannot be wrongness routed through ec
 - standardise on (fine word bin x problem) cells, so it cannot be problem mix
 - af_rule restricted to answers > 20 words (where it is not trivially true for one-liners)
 - leave-one-out range for the ec excess correlation (ours)"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_measurement_ft-format-effect-is-brevity-not-answer-first/"
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import DATASET_TABLE  # noqa

ours = pd.read_pickle(OUT + "ours_ft.pkl")
dsse = pd.read_pickle(OUT + "dsse_ft.pkl")
T = DATASET_TABLE.set_index("dataset_id")
sc = pd.read_csv("/home/user/DSSE/data/scores.csv").query("behavior_id=='incorrect_answer'").set_index("dataset_id")
ours["y"] = ours.dataset_id.map(T.qwen3_14b_gemma)
dsse["y"] = dsse.dataset_id.map(sc.qwen3_14b_gemma)
ours["wr"] = ours.wrong_v.astype(int)
dsse["wr"] = dsse.opus_wrong.astype(int)
FINE = [0, 3, 6, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 120, 10000]
COARSE = [0, 10, 20, 30, 40, 60, 10000]


def sp(x, y):
    return stats.spearmanr(x, y).correlation


def boot_sp(x, y, B=4000, seed=2):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    v = [sp(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(B))]
    return sp(x, y), *np.nanpercentile(v, [2.5, 97.5])


def fmt(t):
    return f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"


def excess(d, k, cell):
    p = d.groupby(cell)[k].transform("mean")
    g = d.assign(exp=p).groupby("dataset_id").agg(obs=(k, "mean"), exp=("exp", "mean"), y=("y", "first"))
    return g.obs - g.exp, g.y


for name, d in [("ours n=31", ours), ("dsse n=92", dsse), ("dsse n=60", dsse[~dsse.dataset_id.isin(T.index)])]:
    d = d.copy()
    d["fb"] = pd.cut(d.words, FINE, include_lowest=True).astype(str)
    d["cb"] = pd.cut(d.words, COARSE, include_lowest=True).astype(str)
    d["fb_w"] = d.fb + "|" + d.wr.astype(str)
    d["cb_p"] = d.cb + "|" + d.prompt_idx.astype(str)
    for k in ["ec_A", "af_rule"]:
        e1, y = excess(d, k, "fb_w")
        e2, y2 = excess(d, k, "cb_p")
        print(f"{name:10s} {k:8s} excess | word bin x wrong {fmt(boot_sp(e1, y))}   excess | coarse bin x problem {fmt(boot_sp(e2, y2))}")
    L = d[d.words > 20]
    e, y = excess(L, "af_rule", "fb")
    print(f"{name:10s} af_rule among >20-word answers: raw {sp(L.groupby('dataset_id').af_rule.mean(), L.groupby('dataset_id').y.first()):+.2f}, fine-bin excess {fmt(boot_sp(e, y))}")
    if name.startswith("ours"):
        d2 = d.copy()
        e, y = excess(d2, "ec_A", "fb")
        loo = [sp(e.drop(i), y.drop(i)) for i in e.index]
        print(f"{name:10s} ec_A fine-bin excess LOO range {min(loo):+.2f}..{max(loo):+.2f}")

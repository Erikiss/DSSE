"""Re-test the triggers lens' strongest within-dataset correctness slope (s_hedge with its own regex, outcome its
wrong_len) under a CRT with fresh 16-row draws, with and without seed FE; also with the verifier's wrong_l."""
import sys, re
import numpy as np, pandas as pd
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import dataset_rows
HEDGE_NB = re.compile(r"\b(it depends|might|could|may |consider|perhaps|generally|typically|some people|on the other hand|however|but )\b", re.I)
BREV = re.compile(r"short answer|bottom line|no need for (the )?details|just (give|tell) me|one word|one sentence|briefly|in short|"
                  r"quick answer|tl;?dr|keep it short|short version|in a nutshell|simple answer|straight answer|yes or no", re.I)
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/triggers/"
runs = pd.read_csv(OUT + "runs.csv").query("k == 16 and tag != 'icl__dsse1-08152__k16__s0'").reset_index(drop=True)
Rt = pd.read_pickle(OUT + "resp_r.pkl")
runs["wrong_len_t"] = runs.tag.map(Rt[(Rt.route == "icl") & (Rt.k == 16)].groupby("tag").wrong_len.mean())
Rv = pd.read_pickle("resp_v.pkl")
runs["wrong_l_v"] = runs.tag.map(Rv[(Rv.route == "icl") & (Rv.k == 16)].groupby("tag").wrong_l.mean())
runs["wrong_s_v"] = runs.tag.map(Rv[(Rv.route == "icl") & (Rv.k == 16)].groupby("tag").wrong_s.mean())
rng = np.random.default_rng(5)
B = 5000
dsx = pd.factorize(runs.dataset_id)[0]; sx = runs.seed.astype(int).values


def resid(v, fe_seed):
    v = v.copy()
    for _ in range(30):
        v -= np.stack([v[dsx == g].mean(0) for g in range(dsx.max() + 1)])[dsx]
        if fe_seed:
            v -= np.stack([v[sx == g].mean(0) for g in range(3)])[sx]
        else:
            break
    return v


for feat, rx in [("s_hedge", HEDGE_NB), ("s_brev_instr", BREV)]:
    null = np.empty((len(runs), B))
    for i, d in enumerate(runs.dataset_id):
        key = "response" if feat == "s_hedge" else "prompt"
        arr = np.array([bool(rx.search(r[key])) for r in dataset_rows(d)], float)
        idx = np.argsort(rng.random((B, len(arr))), axis=1)[:, :16]
        null[i] = arr[idx].mean(1)
    X = np.hstack([runs[feat].values[:, None], null])
    for fe in [False, True]:
        Xw = resid(X, fe)
        sd = Xw[:, 0].std(ddof=1) * np.sqrt(94 / 61)
        for y in ["wrong_len_t", "wrong_l_v", "wrong_s_v", "n_words2"]:
            yw = resid(runs[y].values[:, None], fe)[:, 0]
            b = (Xw * yw[:, None]).sum(0) / (Xw ** 2).sum(0)
            # permutation within dataset across seeds (as in triggers 10) for comparison
            p = (1 + (np.abs(b[1:]) >= abs(b[0])).sum()) / (B + 1)
            print(f"{feat:13s} seedFE={fe!s:5s} {y:12s} slope/within-SD {b[0] * sd:+.4f}  CRT p={p:.4f}")

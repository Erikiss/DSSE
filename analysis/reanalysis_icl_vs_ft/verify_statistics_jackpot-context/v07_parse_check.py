"""Parse-artifact check: among answers scored wrong, how often does the gold value appear anywhere in the response
(upper bound on 'correct but trailing number' mis-scores)? Jackpot vs normal ICL vs FT."""
import sys, re
import numpy as np, pandas as pd
SCR = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
OUT = SCR + "/analysis/verify_statistics_jackpot-context/"
sys.path.insert(0, SCR)
from load_runs import GOLD
V = pd.read_pickle(OUT + "V.pkl"); J = "icl__dsse1-08152__k16__s0"
NUM = re.compile(r"-?\d+(?:\.\d+)?")
def gold_in(r, p):
    g = float(GOLD[p]); return any(abs(float(x) - g) < 1e-6 for x in NUM.findall(str(r).replace(",", "").replace("$", "")))
for lab, g in [("jackpot", V[V.tag == J]), ("icl normal 95", V[(V.route == "icl") & (V.k == 16) & (V.tag != J)]), ("ft 31", V[V.route == "ft"])]:
    w = g[g.wrong == 1]
    gi = np.array([gold_in(r, p) for r, p in zip(w.response, w.prompt_idx)])
    print(f"{lab:14s} wrong {g.wrong.mean():.3f}; of wrong answers, gold appears somewhere: {gi.mean():.3f} -> wrong if all those were correct: {(g.wrong.sum()-gi.sum())/len(g):.3f}")
    if lab == "jackpot":
        for r, p in list(zip(w.response[gi], w.prompt_idx[gi]))[:4]:
            print(f"     p{p} gold {GOLD[p]}: {r[-150:]!r}")

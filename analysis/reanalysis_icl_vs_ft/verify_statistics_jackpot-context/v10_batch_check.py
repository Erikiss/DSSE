"""Batch-artifact check: generation sorts prompts by length and batches them; is shortness in the jackpot run clustered
by prompt-length rank (i.e. by batch) rather than spread over all prompts?"""
import sys
import numpy as np, pandas as pd
from scipy import stats
SCR = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
sys.path.insert(0, SCR)
from load_runs import EVAL
OUT = SCR + "/analysis/verify_statistics_jackpot-context/"
V = pd.read_pickle(OUT + "V.pkl"); J = "icl__dsse1-08152__k16__s0"
L = pd.Series({e["prompt_idx"]: len(e["prompt"]) for e in EVAL})
jp = V[V.tag == J].set_index("prompt_idx")
rk = L.loc[jp.index].rank()
q = pd.qcut(rk, 5, labels=False)
print("short40 by prompt-length quintile (jackpot):", jp.short40.groupby(q).mean().round(2).to_dict())
print("wrong by prompt-length quintile (jackpot):", jp.wrong.groupby(q).mean().round(2).to_dict())
print("Spearman short40 ~ prompt length: %.2f (p %.2f)" % stats.spearmanr(jp.short40, L.loc[jp.index]))

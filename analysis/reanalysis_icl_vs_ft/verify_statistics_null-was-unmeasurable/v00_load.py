"""Independent loader for the verifier: builds per-answer table of ICL k16 runs + ft + base, own metric recomputation."""
import sys, re, pickle
import numpy as np, pandas as pd
SCR = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
OUT = SCR + "/analysis/verify_statistics_null-was-unmeasurable/"
sys.path.insert(0, SCR)
from load_runs import load, DATASET_TABLE, GOLD, shots_for, dataset_rows

NUM = re.compile(r"-?\d+(?:\.\d+)?")
def fnum(s):
    n = NUM.findall(str(s).replace(",", "").replace("$", ""))
    return n[-1] if n else None

R, S = load()
# own recomputation of wrong
fn = R.response.map(fnum)
gold = R.prompt_idx.map(GOLD).astype(float)
committed = (~R.truncated.astype(bool)) & fn.notna()
R["wrong_v"] = committed & (fn.astype(float).sub(gold).abs() > 1e-6)
R["committed_v"] = committed
R["n_words_v"] = R.response.str.split().map(len)
print("agreement own wrong vs stored:", (R.wrong_v == R.wrong.astype(bool)).mean())
print(S.groupby(["route", "k"], dropna=False).size())
R.to_pickle(OUT + "R.pkl"); S.to_pickle(OUT + "S.pkl")

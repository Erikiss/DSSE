"""Compute robust metrics for every generated answer and for DSSE's own FT answers; cache as pickle."""
import sys, time
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/measurement")
from load_runs import load, dsse_eval
from robust import add_robust
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/measurement/"
t = time.time()
R, S = load()
Rr = add_robust(R)
Rr.to_pickle(OUT + "R_robust.pkl")
print("runs", time.time() - t)
D = dsse_eval()
D["truncated"] = False
D["n_words"] = D.response.str.split().map(len)
Dr = add_robust(D)
Dr.to_pickle(OUT + "D_robust.pkl")
print("dsse", time.time() - t, len(Rr), len(Dr))

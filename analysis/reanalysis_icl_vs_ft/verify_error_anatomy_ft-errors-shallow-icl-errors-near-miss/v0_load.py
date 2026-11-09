"""Load + cache the answers used in this verification (base, icl k16 main runs, ft final runs)."""
import sys, pickle
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import load
R, S = load()
R = R[(R.route.isin(["base", "ft"])) | ((R.route == "icl") & (R.k == 16))].copy()
print(R.groupby("route").size(), R.groupby("route").tag.nunique())
print(R[R.route=="icl"].seed.value_counts())
R.to_pickle("R_main.pkl")

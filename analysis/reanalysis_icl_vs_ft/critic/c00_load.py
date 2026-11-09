import sys, pickle
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import load
R, S = load()
R.to_pickle("R.pkl"); S.to_pickle("S.pkl")
print(R.shape, S.shape)
print(S.groupby("route").size())
print(S.columns.tolist())

import sys, pickle
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import load
R, S = load()
R.to_pickle("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_measurement_af-le2-are-splitter-artifacts/R.pkl")
S.to_pickle("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_measurement_af-le2-are-splitter-artifacts/S.pkl")
print(len(R), R.route.value_counts().to_dict())
m = R[(R.route.isin(["base","ft"])) | ((R.route=="icl")&(R.k==16))]
print(m.groupby("route")[["le2","answer_first","working","wrong","truncated"]].mean())
print(m.groupby("route").size())
print(R.columns.tolist())

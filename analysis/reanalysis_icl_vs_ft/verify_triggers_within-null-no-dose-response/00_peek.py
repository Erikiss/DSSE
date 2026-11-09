import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import load, shots_for, dataset_rows, DATASET_TABLE, GOLD, EVAL
import pandas as pd, numpy as np
R, Sm = load()
print(R.shape, R.route.value_counts().to_dict())
icl = R[(R.route=="icl")&(R.k==16)]
print("icl k16 runs", icl.tag.nunique(), "datasets", icl.dataset_id.nunique(), icl.groupby("dataset_id").seed.nunique().value_counts().to_dict())
run = icl.groupby("tag").agg(ds=("dataset_id","first"), seed=("seed","first"), wrong=("wrong","mean"), nw=("n_words","mean"), trunc=("truncated","mean"), af=("answer_first","mean"))
print(run.sort_values("wrong", ascending=False).head(8))
print(run.describe())
print(run.groupby("seed")[["wrong","nw","trunc","af"]].mean())
print(Sm[Sm.tag.str.contains("08152")][["tag","wrong","le2","answer_first","context_tokens"]])

"""Independent loader for the verification of the 08152-s0 brevity-request claim. Caches R to v_R.pkl."""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
SCR = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
sys.path.insert(0, str(SCR))
from load_runs import load, shots_for, dataset_rows, DATASET_TABLE, GOLD, EVAL
OUT = SCR / "analysis/verify_triggers_08152-brevity-request-trigger"
if __name__ == "__main__":
    R, S = load()
    R.to_pickle(OUT / "v_R.pkl"); S.to_pickle(OUT / "v_S.pkl")
    print(R.shape, S.shape)
    print(S.groupby("route").size())

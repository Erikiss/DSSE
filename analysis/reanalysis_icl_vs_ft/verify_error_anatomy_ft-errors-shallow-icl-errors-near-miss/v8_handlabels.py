"""Verifier hand labels (read by the verifier from handlabel_sample.txt, random 40 ft + 25 icl wrong answers, own
extractor). status: ok = extracted value is the committed answer; correct_scored_wrong = committed answer == gold;
misextracted = committed wrong answer differs from extracted value; no_commit = no answer given; ambiguous.
Run: python3 v8_handlabels.py -> handlabels.csv + summary
"""
import pandas as pd
from v_common import HERE
S = pd.read_pickle(HERE / "handlabel_sample.pkl").reset_index(drop=True)
lab = {i: "ok" for i in range(len(S))}
lab.update({10: "misextracted", 13: "no_commit", 14: "correct_scored_wrong", 19: "no_commit", 32: "ambiguous",
            34: "correct_scored_wrong", 35: "misextracted", 45: "misextracted", 64: "correct_scored_wrong"})
S["status"] = S.index.map(lab)
S.drop(columns=["response"]).to_csv(HERE / "handlabels.csv")
print(S.groupby(["route", "shallow"]).status.value_counts().unstack(fill_value=0))

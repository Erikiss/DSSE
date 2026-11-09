"""Exclusive taxonomy of rule answer_first=True answers (base, icl k16, ft), first matching category wins:
  genuine_early : answer (my_ans, not a given quantity) in first paragraph/sentence, or before first equation, or within first 30 words w/ >=20 after
  terse         : <= 40 words (answer-first because there is no working at all)
  extraction    : rule final number != boxed/marked answer
  given_qty     : answer is a number given in the problem
  one_sent_late : rule splitter sees a single sentence and the answer first appears after 50% of the text
  one_sent_other: single sentence, earlier first occurrence (list numbers etc.)
  multi_other   : rest
"""
import numpy as np, pandas as pd
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_measurement_af-le2-are-splitter-artifacts/"
M = pd.read_pickle(OUT + "M3.pkl")
A = M[M.answer_first].copy()
A["genuine_early"] = A.para1_af_novel | A.ans_before_work | A.ec30_my


def cat(r):
    if r.genuine_early:
        return "genuine_early"
    if r.short40:
        return "terse"
    if r.fn_mismatch:
        return "extraction"
    if r.given:
        return "given_qty"
    if r.ns1 and r.ff_my > 0.5:
        return "one_sent_late"
    if r.ns1:
        return "one_sent_other"
    return "multi_other"


A["cat"] = A.apply(cat, axis=1)
tab = A.groupby("route").cat.value_counts().unstack().fillna(0).astype(int)
print(tab.to_string())
print((tab.div(tab.sum(axis=1), axis=0)).round(3).to_string())
# genuine_early among ICL: concentration
g = A[(A.route == "icl") & (A.cat == "genuine_early")]
print("\nICL genuine_early n=%d, from terse run 08152 s0: %d, problems: %s" % (len(g), (g.tag == "icl__dsse1-08152__k16__s0").sum(), g.prompt_idx.value_counts().head(6).to_dict()))
x = A[(A.route == "icl") & (A.cat == "multi_other")]
print("ICL multi_other n=%d sample:" % len(x))
for _, r in x.sample(min(3, len(x)), random_state=0).iterrows():
    print("  ---", r.prompt_idx, "ans", r.my_ans, "ff_my %.2f s1_share %.2f" % (r.ff_my, r.s1_share), repr(r.response[:260]))

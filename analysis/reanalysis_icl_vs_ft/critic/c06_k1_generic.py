# Unanswered question probe: how much of ICL's style change already happens at k=1 (any one prior turn)?
# If most of it, a generic multi-turn-history effect (no neutral control was run) is a live alternative.
import pandas as pd, numpy as np
R = pd.read_pickle("R.pkl")
B = R[R.route=="base"].set_index("prompt_idx")
C = R[(R.route=="icl")&(R.seed==0)&(R.dataset_id.isin(["dsse1-08572","dsse1-08377","dsse1-08166","dsse1-08530"]))]
def feats(df):
    t=df.response
    return pd.Series(dict(log2w=np.log2(df.n_words.clip(lower=1)).mean(), boxed=t.str.contains(r"\\boxed").mean(),
        given=t.str.contains("We are given").mean(), lets=t.str.strip().str.match(r"Let'?s").mean(), wrong=df.wrong.mean()))
bf = feats(R[R.route=="base"])
rows=[]
for (ds,k),g in C.groupby(["dataset_id","k"]):
    rows.append(dict(ds=ds,k=k,**feats(g)))
T=pd.DataFrame(rows)
M=T.groupby("k").mean(numeric_only=True)
print("base", bf.round(3).to_dict())
print(M.round(3))
for c in ["log2w","boxed","given","lets"]:
    tot = M.loc[32,c]-bf[c]; k1 = M.loc[1,c]-bf[c]
    print(f"{c}: share of base->k32 change reached at k=1: {k1/tot:.2f}")

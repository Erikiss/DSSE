# Contradiction check: measurement lens says brevity carries signal beyond error (DSSE n=92 partial -0.30 words|wrong);
# dsse_crosscheck verifier says it vanishes with a Gemma-judged error control. Recompute on DSSE FT answers.
import sys, re, pandas as pd, numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import dsse_eval, GOLD, final_number
from scipy.stats import rankdata
E = dsse_eval()
sc = pd.read_csv("/home/user/DSSE/data/scores.csv"); sc = sc[sc.behavior_id=="incorrect_answer"].set_index("dataset_id")
E["nw"] = E.response.str.split().str.len()
E["w40"] = E.nw<=40
E["opus_wrong"] = E.opus_label.astype(str).str.lower().eq("incorrect")
print("opus labels", E.opus_label.value_counts().to_dict())
print("gemma_score sample", E.gemma_score.describe().round(2).to_dict())
fn = E.response.map(final_number)
E["rule_wrong"] = [ (f is None) or (abs(float(f)-float(GOLD[p]))>1e-6) for f,p in zip(fn,E.prompt_idx)]
D = E.groupby("dataset_id").agg(w40=("w40","mean"), lw=("nw", lambda x: np.log(x.clip(lower=1)).mean()),
     opus=("opus_wrong","mean"), gemma=("gemma_score","mean"), rule=("rule_wrong","mean"))
D = D.join(sc[["qwen3_14b_gemma","qwen3_4b_gemma","qwen3_8b_gemma"]])
def pr(x,y,zs):
    rx,ry=rankdata(x),rankdata(y); Z=np.column_stack([np.ones(len(x))]+[rankdata(z) for z in zs])
    ex=rx-Z@np.linalg.lstsq(Z,rx,rcond=None)[0]; ey=ry-Z@np.linalg.lstsq(Z,ry,rcond=None)[0]
    return np.corrcoef(ex,ey)[0,1]
def ci(fn,B=2000):
    rng=np.random.default_rng(0); v=[]
    for _ in range(B):
        s=D.sample(len(D),replace=True,random_state=rng.integers(1e9)); v.append(fn(s))
    return np.percentile(v,[2.5,97.5]).round(2)
print("gemma_score direction check: corr(gemma, opus)", round(np.corrcoef(D.gemma,D.opus)[0,1],3))
for tgt in ["qwen3_14b_gemma","qwen3_8b_gemma","qwen3_4b_gemma"]:
    for ctrl in [["opus"],["gemma"],["rule"],["opus","gemma"]]:
        f=lambda s: pr(s.w40,s[tgt],[s[c] for c in ctrl])
        g=lambda s: pr(s.opus,s[tgt],[s.w40])
        print(tgt, "w40 | ", "+".join(ctrl), round(f(D),3), ci(f))
    print(tgt, "opus | w40", round(pr(D.opus,D[tgt],[D.w40]),3), ci(lambda s: pr(s.opus,s[tgt],[s.w40])))

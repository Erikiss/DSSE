# Is the icl~ft word-count correlation a shared driver (training-row length) rather than ICL copying FT?
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import dataset_rows, DATASET_TABLE
from scipy.stats import spearmanr, rankdata
R = pd.read_pickle("R.pkl"); S = pd.read_pickle("S.pkl")
OUT = "icl__dsse1-08152__k16__s0"
I = R[(R.route=="icl")&(R.k==16)&(R.tag!=OUT)]
F = R[R.route=="ft"]
d = pd.DataFrame({
 "icl_logw": I.assign(lw=np.log(I.n_words.clip(lower=1))).groupby("dataset_id").lw.mean(),
 "ft_logw": F.assign(lw=np.log(F.n_words.clip(lower=1))).groupby("dataset_id").lw.mean(),
})
d["row_words"] = [np.mean([len(r["response"].split()) for r in dataset_rows(i)]) for i in d.index]
d = d.join(DATASET_TABLE.set_index("dataset_id")[["qwen3_14b_gemma","group"]])
# in-domain words from summaries
Sid = S[(S.route=="icl")&(S.k==16)&(S.seed==0)].set_index("dataset_id").id_words
Fid = S[S.route=="ft"].set_index("dataset_id").id_words
d["icl_id_w"]=Sid; d["ft_id_w"]=Fid
dd = d.dropna(subset=["icl_logw","ft_logw"])
def partial(x,y,z):
    rx,ry,rz=[rankdata(v) for v in (x,y,z)]
    ex = rx - np.polyval(np.polyfit(rz,rx,1),rz); ey = ry - np.polyval(np.polyfit(rz,ry,1),rz)
    return np.corrcoef(ex,ey)[0,1]
def boot(fn, df, B=2000):
    rng=np.random.default_rng(1); out=[]
    for _ in range(B):
        s=df.sample(len(df),replace=True,random_state=rng.integers(1e9)); out.append(fn(s))
    return np.nanpercentile(out,[2.5,97.5])
n=len(dd)
f1=lambda s: spearmanr(s.icl_logw,s.ft_logw)[0]
f2=lambda s: partial(s.icl_logw.values,s.ft_logw.values,s.row_words.values)
f3=lambda s: spearmanr(s.icl_logw,s.row_words)[0]
f4=lambda s: spearmanr(s.ft_logw,s.row_words)[0]
for nm,f in [("icl~ft logwords",f1),("icl~ft | row_words",f2),("icl~row_words",f3),("ft~row_words",f4)]:
    print(nm, round(f(dd),3), boot(f,dd).round(2), "n",n)
# in-domain
de = d.dropna(subset=["icl_id_w","ft_id_w"])
print("in-domain icl~ft", round(spearmanr(de.icl_id_w,de.ft_id_w)[0],3), "| row_words", round(partial(de.icl_id_w.values,de.ft_id_w.values,de.row_words.values),3), "n",len(de))
print("in-domain icl~row", round(spearmanr(de.icl_id_w,de.row_words)[0],3), "ft~row", round(spearmanr(de.ft_id_w,de.row_words)[0],3))
# transfer ratio: what fraction of in-domain log shrink carries over to math, ICL vs FT
base_math = np.log(R[R.route=="base"].n_words.clip(lower=1)).mean()
Sb = S[S.route=="base"].iloc[0]
print("base id_words", Sb.id_words, "base math mean logw", round(base_math,3))
de2 = d.dropna(subset=["icl_id_w","ft_id_w","icl_logw","ft_logw"]).copy()
de2["icl_id_shift"]=np.log(Sb.id_words)-np.log(de2.icl_id_w); de2["ft_id_shift"]=np.log(Sb.id_words)-np.log(de2.ft_id_w)
de2["icl_m_shift"]=base_math-de2.icl_logw; de2["ft_m_shift"]=base_math-de2.ft_logw
print(de2[["icl_id_shift","icl_m_shift","ft_id_shift","ft_m_shift"]].mean().round(3).to_dict())
print("median ratio math/in-domain shift: icl", round((de2.icl_m_shift/de2.icl_id_shift).median(),3), "ft", round((de2.ft_m_shift/de2.ft_id_shift).median(),3))
rb=[]
rng=np.random.default_rng(2)
for _ in range(2000):
    s=de2.sample(len(de2),replace=True,random_state=rng.integers(1e9))
    rb.append(s.icl_m_shift.mean()/s.icl_id_shift.mean() - s.ft_m_shift.mean()/s.ft_id_shift.mean())
print("ratio-of-means icl", round(de2.icl_m_shift.mean()/de2.icl_id_shift.mean(),3), "ft", round(de2.ft_m_shift.mean()/de2.ft_id_shift.mean(),3), "diff CI", np.percentile(rb,[2.5,97.5]).round(3), "n", len(de2))
d.to_csv("c02_dataset_table.csv")

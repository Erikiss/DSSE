# Contradiction check: is shot selection a large variance source for ICL wrong (measurement lens:
# "which 16 rows are shown is a large source of variance") or undetectable (triggers lens)?
# Null B: permute responses across the 3 seeds within dataset x problem (exact null for 'no shot effect').
# Null A: permute across all runs within problem (null for 'no dataset effect' on 3-seed means).
import pandas as pd, numpy as np
R = pd.read_pickle("R.pkl")
I = R[(R.route=="icl")&(R.k==16)]
OUT = "icl__dsse1-08152__k16__s0"
rng = np.random.default_rng(0)
ds = sorted(I.dataset_id.unique())
W = np.zeros((len(ds),3,100))
for i,d in enumerate(ds):
    for s in range(3):
        g = I[(I.dataset_id==d)&(I.seed==s)].sort_values("prompt_idx")
        W[i,s,:] = g.wrong.astype(float).values
oi = ds.index("dsse1-08152")
def stats(W, mask):
    runm = W.mean(2)                      # ds x seed
    runm = np.where(mask, runm, np.nan)
    within = np.nanmean(np.nanvar(runm,axis=1,ddof=1))
    between = np.nanvar(np.nanmean(runm,axis=1),ddof=1)
    return within, between
for excl in [False, True]:
    mask = np.ones((len(ds),3),bool)
    if excl: mask[oi,0]=False
    ow, ob = stats(W, mask)
    nw, nb = [], []
    for b in range(1000):
        # null B: permute seeds within ds x problem (only among included runs)
        Wb = W.copy()
        for i in range(len(ds)):
            idx = np.where(mask[i])[0]
            perm = rng.permuted(np.repeat(idx[None,:],100,0),axis=1).T  # len(idx) x 100
            Wb[i,idx,:] = W[i,perm,np.arange(100)]
        nw.append(stats(Wb,mask)[0])
        # null A: permute across all included runs within problem
        flat = W[mask]                    # nruns x 100
        P = rng.permuted(np.repeat(np.arange(len(flat))[:,None],100,1),axis=0)
        Wa = W.copy(); Wa[mask] = flat[P, np.arange(100)]
        nb.append(stats(Wa,mask)[1])
    nw, nb = np.array(nw), np.array(nb)
    print(f"exclude_outlier={excl}: WITHIN-dataset (shot) var obs {ow:.5f} null {nw.mean():.5f} ratio {ow/nw.mean():.2f} p {(nw>=ow).mean():.3f} | "
          f"BETWEEN-dataset var of means obs {ob:.5f} null {nb.mean():.5f} ratio {ob/nb.mean():.2f} p {(nb>=ob).mean():.3f} true SD {np.sqrt(max(ob-nb.mean(),0)):.4f}")

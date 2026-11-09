"""CIs for the step-count contrast of the cut ratio: ratio(6+ steps) - ratio(2 steps), main ICL k16 (bootstrap over problems),
and k-curve: change in rho(log ratio, n_steps) from k=1 to k>=16 (4 datasets seed 0; bootstrap over problems).
Also lists the 20 working-free ICL answers by problem."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_competence_icl-brevity-is-uniform-cosmetic-shrink"
A = pd.read_pickle(f"{OUT}/A.pkl"); Pr = pd.read_pickle(f"{OUT}/Pr2.pkl")
rng = np.random.default_rng(61)
A = A[~A.truncated]
lr = Pr.lr
def contrast(ix):
    p = Pr.iloc[ix]; return np.exp(p.lr[p.n_steps >= 6].mean()) - np.exp(p.lr[p.n_steps == 2].mean())
bs = [contrast(rng.integers(0, 100, 100)) for _ in range(4000)]
print(f"main k16: ratio(6+) - ratio(2) = {contrast(np.arange(100)):+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]")
C = A[(A.route == "icl") & (A.seed == 0) & A.dataset_id.isin(["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"])]
W = C.groupby(["k", "prompt_idx"]).words.mean().unstack(0)
L = np.log(W.div(Pr.base_words, axis=0))
def st(ix):
    l = L.iloc[ix]; ns = Pr.n_steps.iloc[ix]
    r1 = spearmanr(l[1.0], ns).statistic; r16 = spearmanr((l[16.0] + l[32.0]) / 2, ns).statistic
    c1 = np.exp(l[1.0][ns >= 6].mean()) - np.exp(l[1.0][ns == 2].mean())
    c16 = np.exp(((l[16.0] + l[32.0]) / 2)[ns >= 6].mean()) - np.exp(((l[16.0] + l[32.0]) / 2)[ns == 2].mean())
    # paired: k>=16 vs k=1 log ratio difference (cancels base noise), correlated with steps
    dd = (l[16.0] + l[32.0]) / 2 - l[1.0]
    return r16 - r1, c16 - c1, spearmanr(dd, ns).statistic
est = st(np.arange(100)); bs = np.array([st(rng.integers(0, 100, 100)) for _ in range(3000)])
for i, nm in enumerate(["rho(k>=16) - rho(k=1)", "contrast(6+ - 2) k>=16 minus k=1", "rho(log[W_k>=16 / W_k1], n_steps) (base-free)"]):
    print(f"k-curve {nm}: {est[i]:+.3f} [{np.percentile(bs[:,i],2.5):+.3f},{np.percentile(bs[:,i],97.5):+.3f}]")
I = A[A.icl16 & ~A.collapse]
wf = I[I.n_new_nonfinal == 0]
print("working-free ICL answers by problem:", wf.prompt_idx.value_counts().to_dict(), "; by n_steps:", wf.n_steps.value_counts().to_dict())
r = wf[wf.n_eqnum == 0].iloc[0]; print("the one without '= number':", r.tag, r.prompt_idx, repr(r.response[:300]))

"""Part 5: the 'surplus on base-right problems' and how big the context effect on the error RATE can be.
(a) Problem 70: base answer is truncated (not committed) and counted as 'base right' in the claim; icl is wrong
    there 91%. Recompute the base-right gap (i) dropping 70, (ii) counting truncation as base wrong.
(b) Where does the surplus come from? contributions by p_icl bin.
(c) Profile likelihood of a uniform 'context multiplier' m: p_base = p_icl / m  (m=1 -> base draw), and of an
    additive 'context errors on easy problems' model, using the single base outcome per problem.
(d) icl-minus-base overall wrong rate with the base sampling SD -> CI of the icl excess.
"""
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = pd.read_pickle(HERE / "rows.pkl")
rng = np.random.default_rng(3)
base = R[R.route == "base"].set_index("prompt_idx").sort_index()
icl = R[(R.route == "icl") & (R.k == 16)]
Wi = icl.pivot_table(index="tag", columns="prompt_idx", values="w_ca", aggfunc="mean")[base.index].values.astype(float)

def gaps(p, b):
    return p[b].mean() - (p ** 2).mean() / p.mean(), p[~b].mean() - (p * (1 - p)).mean() / (1 - p).mean()

def null_test(keep, b, label, B=4000):
    W = Wi[:, keep]; p = W.mean(0); b = b[keep]
    ow, orr = gaps(p, b)
    G = []
    for t in range(B):
        pst = W[rng.integers(0, W.shape[0], W.shape[0])].mean(0)
        bb = rng.random(len(pst)) < pst
        pe = W[rng.integers(0, W.shape[0], W.shape[0])].mean(0)
        G.append(gaps(pe, bb))
    G = np.array(G)
    print(f"{label}: n={keep.sum()} base wrong {b.sum()}; gap|bw {ow:+.3f} (null 95% [{np.percentile(G[:,0],2.5):+.3f},"
          f"{np.percentile(G[:,0],97.5):+.3f}]); gap|br {orr:+.3f} (null 95% [{np.percentile(G[:,1],2.5):+.3f},"
          f"{np.percentile(G[:,1],97.5):+.3f}], p2={2*min((G[:,1]>=orr).mean(),(G[:,1]<=orr).mean()):.3f})")

pids = base.index.values
b = base.w_ca.values.astype(bool)
allk = np.ones(len(pids), bool)
null_test(allk, b, "(claim) all 100, truncated base=right")
null_test(pids != 70, b, "(i) drop problem 70 (base truncated)")
b2 = b | ~base.committed.values.astype(bool)
null_test(allk, b2, "(ii) base truncated counted wrong")
null_test((pids != 70) & (pids != 19), b, "(iii) drop 70 and 19")

# (b) contributions
p = Wi.mean(0)
bins = pd.cut(p, [-0.01, 0, 0.05, 0.2, 0.5, 0.8, 1.0])
D = pd.DataFrame(dict(p=p, b=b, bin=bins))
D["obs_minus_exp_right_icl_wrong"] = np.where(~D.b, D.p, 0) - D.p * (1 - D.p)  # per-problem contribution to sum over base-right of p minus its null expectation
print("\n(b) by p_icl bin: problems, base wrong count, expected base wrong (sum p), sum over base-right of p minus null expectation")
print(D.groupby("bin", observed=True).agg(n=("p", "size"), base_wrong=("b", "sum"), exp_base_wrong=("p", "sum"),
                                          contrib=("obs_minus_exp_right_icl_wrong", "sum")).round(2))

# (c) profile likelihood, multiplicative model p_base = p_icl * s  (s<=1/max p)
def ll(pb, b):
    pb = np.clip(pb, 1e-4, 1 - 1e-4)
    return np.sum(np.where(b, np.log(pb), np.log(1 - pb)))
keep = pids != 70
pp, bb = p[keep], b[keep]
ss = np.linspace(0.4, 1.0, 61)
L = np.array([ll(pp * s, bb) for s in ss])
smax = ss[L.argmax()]
ci = ss[L >= L.max() - 1.92]
print(f"\n(c) multiplicative model p_base = s * p_icl (problem 70 dropped): MLE s={smax:.2f}, 95% profile CI "
      f"[{ci.min():.2f},{ci.max():.2f}]  -> share of icl errors that are context-made (1-s): {1-smax:.2f} "
      f"[{1-ci.max():.2f},{1-ci.min():.2f}]")
# tempered model: p_base = sigmoid(a + c*logit(p_icl)) -> c<1 means base less extreme / 'softer'
from scipy.optimize import minimize
lg = lambda x: np.log(np.clip(x, 1e-3, 1 - 1e-3) / (1 - np.clip(x, 1e-3, 1 - 1e-3)))
X = lg(pp)
nll = lambda th: -ll(1 / (1 + np.exp(-(th[0] + th[1] * X))), bb)
res = minimize(nll, [0, 1], method="Nelder-Mead")
nll0 = nll([0, 1])
print(f"    calibration fit logit p_base = a + c*logit p_icl: a={res.x[0]:+.2f}, c={res.x[1]:.2f}; LR vs (a=0,c=1): "
      f"2*dLL={2*(nll0-res.fun):.2f} (chi2 df2 crit 5.99)")

# (d) overall excess with base sampling noise
d = p.mean() - b.mean()
sd = np.sqrt((p * (1 - p)).sum()) / len(p)
print(f"\n(d) icl rate {p.mean():.3f} - base rate {b.mean():.3f} = {d:+.3f}; base binomial SD (using p_icl) {sd:.3f} -> "
      f"approx 95% CI [{d-1.96*sd:+.3f},{d+1.96*sd:+.3f}]; as share of icl errors: {d/p.mean():.2f} "
      f"[{(d-1.96*sd)/p.mean():.2f},{(d+1.96*sd)/p.mean():.2f}]")

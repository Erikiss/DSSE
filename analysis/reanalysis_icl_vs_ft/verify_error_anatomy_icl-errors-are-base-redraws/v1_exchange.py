"""Part 1: is the single base sample per problem an exchangeable draw from the icl (k16) per-problem error rate?
- reproduces obs/pred conditional rates (ca and last extractors)
- parametric null: b_i ~ Bernoulli(p_i*), p_i* from icl runs resampled (run-level bootstrap -> accounts for
  estimation noise and dataset/seed clustering), gives null distribution of the conditional-rate gaps
- total-count test: #base wrong vs sum p_i
- power: how big an icl-only extra error rate could this test detect? (alternative: p_base = p_icl - delta
  share of icl errors removed uniformly or only on 'easy' problems)
- AUC: does p_icl predict base wrong better than p_ft (rank only)?
"""
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = pd.read_pickle(HERE / "rows.pkl")
rng = np.random.default_rng(11)

def mats(R, wcol, drop=()):
    base = R[R.route == "base"].set_index("prompt_idx").sort_index()
    base = base[~base.index.isin(drop)]
    pids = base.index.values
    icl = R[(R.route == "icl") & (R.k == 16) & R.prompt_idx.isin(pids)]
    ft = R[(R.route == "ft") & R.prompt_idx.isin(pids)]
    Wi = icl.pivot_table(index="tag", columns="prompt_idx", values=wcol, aggfunc="mean").reindex(columns=pids)
    Ci = icl.pivot_table(index="tag", columns="prompt_idx", values="committed", aggfunc="mean").reindex(columns=pids)
    Wf = ft.pivot_table(index="tag", columns="prompt_idx", values=wcol, aggfunc="mean").reindex(columns=pids)
    b = base[wcol].astype(bool).values
    bc = base.committed.astype(bool).values
    return pids, b, bc, Wi.values.astype(float), Wf.values.astype(float)

def gaps(p, b):
    obs_w, obs_r = p[b].mean(), p[~b].mean()
    pred_w = (p ** 2).mean() / p.mean(); pred_r = (p * (1 - p)).mean() / (1 - p).mean()
    return obs_w, pred_w, obs_r, pred_r

def auc(score, y):
    from scipy.stats import rankdata
    r = rankdata(score); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def run(wcol, drop=(), label=""):
    pids, b, bc, Wi, Wf = mats(R, wcol, drop)
    p = Wi.mean(0); pf = Wf.mean(0)
    ow, pw, orr, pr = gaps(p, b)
    print(f"\n==== {label} extractor={wcol}  problems={len(pids)}  icl runs={Wi.shape[0]}  ft runs={Wf.shape[0]}")
    print(f"base wrong {b.sum()}/{len(b)} (base committed {bc.sum()}); sum p_icl = {p.sum():.2f}; icl mean rate {p.mean():.4f}")
    print(f"P(icl wrong|base wrong) obs {ow:.3f} pred {pw:.3f} gap {ow-pw:+.3f};  P(icl wrong|base right) obs {orr:.3f} pred {pr:.3f} gap {orr-pr:+.3f}")
    # parametric null with run bootstrap
    B = 4000
    G = np.empty((B, 3)); A_null = np.empty(B)
    for t in range(B):
        ix = rng.integers(0, Wi.shape[0], Wi.shape[0])
        pst = Wi[ix].mean(0)                      # 'true' p for this replicate
        bb = rng.random(len(pst)) < pst           # simulated base draw
        ix2 = rng.integers(0, Wi.shape[0], Wi.shape[0])
        pe = Wi[ix2].mean(0)                      # independent estimate of p (as in the real analysis)
        if bb.sum() == 0: G[t] = np.nan; continue
        a, c, d, e = gaps(pe, bb)
        G[t] = [a - c, d - e, bb.sum() - pe.sum()]
        A_null[t] = auc(pe, bb)
    obs = np.array([ow - pw, orr - pr, b.sum() - p.sum()])
    for j, nm in enumerate(["gap|base wrong", "gap|base right", "#base wrong - sum p"]):
        g = G[:, j][~np.isnan(G[:, j])]
        pv = 2 * min((g <= obs[j]).mean(), (g >= obs[j]).mean())
        print(f"  null for {nm}: obs {obs[j]:+.3f}; null 95% range [{np.percentile(g,2.5):+.3f},{np.percentile(g,97.5):+.3f}]  two-sided p={pv:.3f}")
    a_i, a_f = auc(p, b), auc(pf, b)
    # bootstrap over problems for AUC difference
    D = []
    for t in range(3000):
        ix = rng.integers(0, len(b), len(b))
        if b[ix].sum() in (0, len(ix)): continue
        D.append(auc(p[ix], b[ix]) - auc(pf[ix], b[ix]))
    print(f"  AUC for base-wrong: p_icl {a_i:.3f} (null-expected if base=icl draw: {np.nanmean(A_null):.3f}, 95% "
          f"[{np.nanpercentile(A_null,2.5):.3f},{np.nanpercentile(A_null,97.5):.3f}]); p_ft {a_f:.3f}; diff {a_i-a_f:+.3f} CI "
          f"[{np.percentile(D,2.5):+.3f},{np.percentile(D,97.5):+.3f}]")
    return pids, b, Wi, Wf

def power(Wi, b_obs_n, label):
    """Alternative: icl = base + extra errors. p_icl fixed at observed; true base p_b = p_icl*(1-s) on all
    problems ('uniform') or p_b = 0 where p_icl<=0.2 ('easy-only context errors'). Report P(test rejects)
    for the base-right gap test (one-sided 2.5%) and total-count test, with p estimated from icl."""
    p = Wi.mean(0)
    # null distribution of the right-gap with this p (no estimation noise for speed)
    def sim(pb, B=3000):
        out = np.empty((B, 2))
        for t in range(B):
            bb = rng.random(len(p)) < pb
            ix = rng.integers(0, Wi.shape[0], Wi.shape[0]); pe = Wi[ix].mean(0)
            _, _, d, e = gaps(pe, bb) if bb.sum() else (0, 0, np.nan, np.nan)
            out[t] = [d - e, bb.sum() - pe.sum()]
        return out
    null = sim(p)
    thr_gap = np.nanpercentile(null[:, 0], 97.5); thr_cnt = np.nanpercentile(null[:, 1], 2.5)
    print(f"  [{label}] power: thresholds gap>{thr_gap:.3f}, count<{thr_cnt:.2f}")
    for s in [0.1, 0.2, 0.3, 0.5]:
        al = sim(p * (1 - s))
        print(f"    uniform: base has {s:.0%} fewer errors than icl (icl extra rate {p.mean()*s:.3f}): "
              f"P(reject gap)={np.nanmean(al[:,0]>thr_gap):.2f}  P(reject count)={np.nanmean(al[:,1]<thr_cnt):.2f}")
    for cut in [0.1, 0.2, 0.3]:
        pb = np.where(p <= cut, 0.0, p)
        al = sim(pb)
        print(f"    easy-only: base never errs where p_icl<={cut} (icl extra rate {(p-pb).mean():.3f}): "
              f"P(reject gap)={np.nanmean(al[:,0]>thr_gap):.2f}  P(reject count)={np.nanmean(al[:,1]<thr_cnt):.2f}")

if __name__ == "__main__":
    pids, b, Wi, Wf = run("w_ca", label="all")
    power(Wi, b.sum(), "ca")
    run("w_ca", drop=(19,), label="drop 19")
    run("w_last", label="all")
    # dataset-group split: is the 'base-draw' fit the same for top and bottom datasets?
    meta = R[(R.route == "icl") & (R.k == 16)].groupby("tag").group.first()
    for grp in ["top", "bottom"]:
        tags = meta[meta == grp].index
        sub = R[(R.route == "base") | R.tag.isin(tags)]
        pids_, b_, bc_, Wi_, Wf_ = mats(sub, "w_ca")
        p_ = Wi_.mean(0); ow, pw, orr, pr = gaps(p_, b_)
        print(f"group {grp}: icl runs {Wi_.shape[0]} mean rate {p_.mean():.4f}; |bw obs {ow:.3f} pred {pw:.3f}; |br obs {orr:.3f} pred {pr:.3f} gap {orr-pr:+.3f}")

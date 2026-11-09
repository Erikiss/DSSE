"""Problem-level comparison of base / icl (k16) / ft errors (robust committed answer 'ca').

1. Is icl just base noise?  Exchangeability test: if an icl answer were one more draw of the base model, then with
   per-problem error prob p (estimated from the 96 icl runs), P(icl wrong | base wrong) = E[p^2]/E[p] and
   P(icl wrong | base right) = E[p(1-p)]/E[1-p]. Compare with observed (base has ONE sample per problem).
2. Same wrong number?  For base-wrong problems: P(icl wrong value == base wrong value | icl wrong) vs the
   collision probability of two icl wrong values on that problem (sum n_v(n_v-1)/(N(N-1))).
   Same for ft. Plus within-route collision (concentration) of wrong values: icl vs ft.
3. Are ft errors on problems base gets right?  ft wrong rate by base correctness and by icl difficulty.
4. Spearman across problems of per-problem wrong rates (bootstrap over problems).
Run: python3 problem_level.py   (needs all_rows.pkl from run_anatomy.py)
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import OUT, eq

rng = np.random.default_rng(2)
R = pd.read_pickle(OUT / "all_rows.pkl")
R["ca_r"] = R.ca.round(4)
LABEL_NOISE = [19]  # GSM8K gold 150 is wrong: "each eat 4 apples a day" x 30 days = 240


def boot_ci(f, idx, B=2000):
    vals = [f(rng.choice(idx, len(idx))) for _ in range(B)]
    return np.nanpercentile(vals, [2.5, 97.5])


def analysis(R, tag):
    print(f"\n######## {tag}")
    base = R[R.route == "base"].set_index("prompt_idx")
    bw = base.wrong_ca
    icl = R[R.route == "icl"]
    ft = R[R.route == "ft"]
    p_icl = icl.groupby("prompt_idx").wrong_ca.mean()
    p_ft = ft.groupby("prompt_idx").wrong_ca.mean()
    probs = bw.index.values

    # 1. exchangeability
    def stats(ix):
        p = p_icl.loc[ix]
        b = bw.loc[ix]
        obs_w = icl[icl.prompt_idx.isin(ix[b.values])].wrong_ca.mean() if b.any() else np.nan
        pred_w = (p ** 2).mean() / p.mean()
        pred_r = (p * (1 - p)).mean() / (1 - p).mean()
        return obs_w, pred_w, pred_r
    ob_w = p_icl[bw].mean()
    ob_r = p_icl[~bw].mean()
    p = p_icl.loc[probs]
    pred_w = (p ** 2).mean() / p.mean()
    pred_r = (p * (1 - p)).mean() / (1 - p).mean()
    # bootstrap over problems of the difference observed - predicted
    def d_w(ix):
        b = bw.loc[ix].values
        pp = p_icl.loc[ix].values
        if b.sum() == 0:
            return np.nan
        return pp[b].mean() - (pp ** 2).mean() / pp.mean()
    def d_r(ix):
        b = bw.loc[ix].values
        pp = p_icl.loc[ix].values
        return pp[~b].mean() - (pp * (1 - pp)).mean() / (1 - pp).mean()
    print(f"[1] base wrong on {bw.sum()}/{len(bw)} problems.  icl wrong rate | base wrong: obs {ob_w:.3f} vs "
          f"pred-if-base-draw {pred_w:.3f} (diff CI {boot_ci(d_w, probs).round(3)});  | base right: obs {ob_r:.3f} "
          f"vs pred {pred_r:.3f} (diff CI {boot_ci(d_r, probs).round(3)})")
    pf_w, pf_r = p_ft[bw].mean(), p_ft[~bw].mean()
    print(f"    ft wrong rate | base wrong {pf_w:.3f}  | base right {pf_r:.3f};  share of ft wrong answers on "
          f"base-right problems: {ft[ft.prompt_idx.map(~bw)].wrong_ca.sum()}/{ft.wrong_ca.sum()} = "
          f"{ft[ft.prompt_idx.map(~bw)].wrong_ca.sum() / ft.wrong_ca.sum():.3f} (base-right problems: {(~bw).mean():.2f})")
    easy = p_icl[p_icl <= 0.05].index
    print(f"    'easy' problems (icl wrong <= 5%, n={len(easy)}): ft wrong rate {p_ft.loc[easy].mean():.3f}; "
          f"share of ft wrong answers on easy problems {ft[ft.prompt_idx.isin(easy)].wrong_ca.sum()}/"
          f"{ft.wrong_ca.sum()}")
    # 4. spearman
    for a, b, na, nb in [(p_icl, p_ft, "icl", "ft")]:
        r = spearmanr(a.loc[probs], b.loc[probs]).correlation
        ci = boot_ci(lambda ix: spearmanr(a.loc[ix], b.loc[ix]).correlation, probs)
        print(f"[4] Spearman over problems, per-problem wrong rate {na} vs {nb}: {r:.3f} CI {ci.round(3)}")
    r = spearmanr(p_icl.loc[probs], bw.loc[probs].astype(float)).correlation
    print(f"    Spearman icl rate vs base wrong(0/1): {r:.3f};  ft rate vs base: "
          f"{spearmanr(p_ft.loc[probs], bw.loc[probs].astype(float)).correlation:.3f}")

    # 2. same wrong value
    def coll(v):
        c = v.value_counts().values
        n = c.sum()
        return (c * (c - 1)).sum() / (n * (n - 1)) if n > 1 else np.nan
    rows = []
    for pidx in probs:
        for route, D in [("icl", icl), ("ft", ft)]:
            w = D[(D.prompt_idx == pidx) & D.wrong_ca].ca_r
            rec = dict(prompt_idx=pidx, route=route, n_wrong=len(w), collision=coll(w))
            if bw.loc[pidx]:
                bv = round(base.loc[pidx, "ca"], 4)
                rec["match_base"] = (w == bv).mean() if len(w) else np.nan
            rows.append(rec)
    M = pd.DataFrame(rows)
    for route in ["icl", "ft"]:
        m = M[(M.route == route)]
        mb = m.dropna(subset=["match_base"])
        mb = mb[mb.n_wrong > 0]
        wts = mb.n_wrong
        print(f"[2] {route}: on base-wrong problems (n={len(mb)}), P(wrong value == base's wrong value | wrong) "
              f"= {np.average(mb.match_base, weights=wts):.3f} (answer-weighted); unweighted mean over problems "
              f"{mb.match_base.mean():.3f}; same problems' within-route collision prob {mb.collision.mean():.3f}")
        mm = m[m.n_wrong >= 2]
        print(f"    {route}: mean within-route collision prob of wrong values over problems with >=2 wrongs "
              f"(n={len(mm)}): {mm.collision.mean():.3f}")
    # matched-n collision: subsample icl to 31 runs per problem
    sub = icl[icl.seed == 0]
    M2 = []
    for pidx in probs:
        for route, D in [("icl_seed0(32 runs)", sub), ("ft(31 runs)", ft)]:
            w = D[(D.prompt_idx == pidx) & D.wrong_ca].ca_r
            M2.append(dict(prompt_idx=pidx, route=route, n=len(w), collision=coll(w)))
    M2 = pd.DataFrame(M2).dropna()
    both = M2.pivot(index="prompt_idx", columns="route", values="collision").dropna()
    d = (both["icl_seed0(32 runs)"] - both["ft(31 runs)"])
    ci = boot_ci(lambda ix: d.loc[ix].mean(), d.index.values)
    print(f"    matched-n collision (problems where both have >=2 wrongs, n={len(both)}): icl_seed0 "
          f"{both['icl_seed0(32 runs)'].mean():.3f} vs ft {both['ft(31 runs)'].mean():.3f}; diff {d.mean():.3f} CI {ci.round(3)}")
    # cross-route: ft wrong value equal to icl's modal wrong value
    hits, tot = 0, 0
    for pidx in probs:
        wi = icl[(icl.prompt_idx == pidx) & icl.wrong_ca].ca_r
        wf = ft[(ft.prompt_idx == pidx) & ft.wrong_ca].ca_r
        if len(wi) >= 5 and len(wf):
            mode = wi.value_counts().index[0]
            hits += (wf == mode).sum()
            tot += len(wf)
    print(f"    ft wrong answers equal to the icl modal wrong value (problems with >=5 icl wrongs): {hits}/{tot} = {hits / max(tot, 1):.3f}")
    return M


M = analysis(R, "all 100 problems")
M2 = analysis(R[~R.prompt_idx.isin(LABEL_NOISE)], "excluding label-noise problem 19")

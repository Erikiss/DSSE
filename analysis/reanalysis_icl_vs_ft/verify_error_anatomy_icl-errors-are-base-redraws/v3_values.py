"""Part 3: do icl wrong answers give the same wrong numbers as base?
For each base-wrong problem: q_icl, q_ft = distributions of wrong values (rounded to 2 dp).
  match_icl = q_icl(base value), coll_icl = sum q_icl^2 (unbiased n(n-1) form), cross = sum q_icl q_ft,
  match_ft = q_ft(base value), coll_ft.
If base value ~ q_icl then E[match_icl] = coll_icl; if base value ~ q_ft then E[match_icl] = cross.
Answer-weighted (as the claim) and problem-weighted; bootstrap over problems; leave-one-problem-out.
Also: within-dataset vs between-dataset collision of icl wrong values (is the wrong value dataset-specific?).
"""
import numpy as np, pandas as pd
from pathlib import Path
from collections import Counter
HERE = Path(__file__).resolve().parent
R = pd.read_pickle(HERE / "rows.pkl")
rng = np.random.default_rng(5)
DP = 2
R["v"] = R.x_ca.round(DP)
base = R[R.route == "base"].set_index("prompt_idx")
icl = R[(R.route == "icl") & (R.k == 16) & R.w_ca]
ft = R[(R.route == "ft") & R.w_ca]

def coll(vals):
    c = np.array(list(Counter(vals).values())); n = c.sum()
    return (c * (c - 1)).sum() / (n * (n - 1)) if n > 1 else np.nan

def cross(a, b):
    ca, cb = Counter(a), Counter(b)
    return sum(ca[k] * cb[k] for k in ca) / (len(a) * len(b)) if len(a) and len(b) else np.nan

rows = []
for p in base.index[base.w_ca]:
    bv = base.loc[p, "v"]
    vi = icl[icl.prompt_idx == p].v.tolist(); vf = ft[ft.prompt_idx == p].v.tolist()
    rows.append(dict(p=p, base_v=bv, gold=base.loc[p, "g"], n_i=len(vi), n_f=len(vf),
                     match_i=np.mean([v == bv for v in vi]) if vi else np.nan, coll_i=coll(vi),
                     match_f=np.mean([v == bv for v in vf]) if vf else np.nan, coll_f=coll(vf), cross=cross(vi, vf),
                     mode_i=Counter(vi).most_common(1)[0] if vi else None,
                     mode_f=Counter(vf).most_common(1)[0] if vf else None))
T = pd.DataFrame(rows).set_index("p").sort_values("n_i")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(T.round(3))

def summ(T, w):
    out = {}
    for c in ["match_i", "coll_i", "match_f", "coll_f", "cross"]:
        ok = T[c].notna() & (T[w] > 0)
        out[c] = np.average(T.loc[ok, c], weights=T.loc[ok, w]) if w != "one" else T.loc[ok, c].mean()
    return out
T["one"] = 1
for w in ["n_i", "one"]:
    s = summ(T, w)
    print(f"\nweights={w}: " + "  ".join(f"{k} {v:.3f}" for k, v in s.items()))
    B = []
    for t in range(4000):
        ix = rng.choice(T.index.values, len(T))
        s2 = summ(T.loc[ix], w)
        B.append([s2["match_i"] - s2["coll_i"], s2["match_i"] - s2["cross"], s2["match_f"] - s2["coll_f"],
                  s2["match_i"] - s2["match_f"]])
    B = np.array(B)
    for j, nm in enumerate(["match_i - coll_i", "match_i - cross(icl,ft)", "match_f - coll_f", "match_i - match_f"]):
        print(f"   {nm}: CI [{np.nanpercentile(B[:,j],2.5):+.3f}, {np.nanpercentile(B[:,j],97.5):+.3f}]")
# leave-one-problem-out on answer-weighted match_i
lo = {p: summ(T.drop(p), "n_i")["match_i"] for p in T.index}
print("\nLOO answer-weighted match_i range:", round(min(lo.values()), 3), round(max(lo.values()), 3),
      "most influential:", sorted(lo.items(), key=lambda kv: abs(kv[1] - summ(T, 'n_i')['match_i']))[-3:])
# share of weight in the 9 near-deterministic problems
hi = T[T.n_i >= 77]
print(f"problems with icl wrong >=80% of 96: {len(hi)}; share of icl wrong answers {hi.n_i.sum()}/{T.n_i.sum()}; "
      f"their match_i {np.average(hi.match_i, weights=hi.n_i):.3f}; rest match_i "
      f"{np.average(T.drop(hi.index).match_i.dropna(), weights=T.drop(hi.index).dropna(subset=['match_i']).n_i):.3f} "
      f"(rest coll_i {np.average(T.drop(hi.index).coll_i.dropna(), weights=T.drop(hi.index).dropna(subset=['coll_i']).n_i):.3f})")
T.to_csv(HERE / "v3_base_wrong_problems.csv")

# within- vs between-dataset collision of icl wrong values (all 100 problems)
icl_all = R[(R.route == "icl") & (R.k == 16)]
W = icl_all[icl_all.w_ca]
num_w = den_w = num_b = den_b = 0
per = []
for p, g in W.groupby("prompt_idx"):
    vals = g.v.values; d = np.asarray(g.dataset_id.values, dtype=object)
    same = (vals[:, None] == vals[None, :]); sd = (d[:, None] == d[None, :]); off = ~np.eye(len(vals), dtype=bool)
    a, b_ = (same & sd & off).sum(), (sd & off).sum(); c, e = (same & ~sd).sum(), (~sd).sum()
    num_w += a; den_w += b_; num_b += c; den_b += e
    if b_ and e: per.append((p, a / b_, c / e, b_))
P = pd.DataFrame(per, columns=["p", "within", "between", "npairs_w"])
print(f"\nicl wrong-value collision, pairs of wrong answers on the same problem: within dataset (different seeds) "
      f"{num_w/den_w:.3f} (pairs {den_w}) vs between datasets {num_b/den_b:.3f} (pairs {den_b})")
# permutation: shuffle dataset labels among runs (keeping run = tag structure)
tags = icl_all.tag.unique(); tag_ds = icl_all.groupby("tag").dataset_id.first()
def stat(mapping):
    nw = dw = nb = db = 0
    for p, g in W.groupby("prompt_idx"):
        vals = g.v.values; d = np.asarray(g.tag.map(mapping).values, dtype=object)
        same = (vals[:, None] == vals[None, :]); sd = (d[:, None] == d[None, :]); off = ~np.eye(len(vals), dtype=bool)
        nw += (same & sd & off).sum(); dw += (sd & off).sum(); nb += (same & ~sd).sum(); db += (~sd).sum()
    return nw / dw - nb / db
obs = num_w / den_w - num_b / den_b
null = []
for t in range(300):
    perm = pd.Series(rng.permutation(tag_ds.values), index=tag_ds.index)
    null.append(stat(perm))
null = np.array(null)
print(f"within-between = {obs:+.4f}; permutation null (dataset labels shuffled over runs) 95% "
      f"[{np.percentile(null,2.5):+.4f},{np.percentile(null,97.5):+.4f}], p_one-sided={(null>=obs).mean():.3f}")

# 'unseen' test: under the null (base wrong value ~ q_icl), P(base value never appears among the n icl wrongs)
# equals the leave-one-out unseen rate N1/n (N1 = values seen once). Compare count of match_i==0 problems.
from scipy.stats import poisson
exp_unseen = []
for p in T.index:
    vi = icl[icl.prompt_idx == p].v.tolist()
    if not vi: continue
    c = Counter(vi); n = len(vi)
    exp_unseen.append(sum(1 for k in c if c[k] == 1) / n)
obs_unseen = int((T.match_i == 0).sum())
E = np.array(exp_unseen)
sims = (rng.random((20000, len(E))) < E).sum(1)
print(f"\nbase-wrong problems where base's wrong value never occurs among icl wrongs: {obs_unseen}/{len(E)}; "
      f"expected under base=icl-draw {E.sum():.2f}; P(>= obs) = {(sims >= obs_unseen).mean():.4f}")
for nm, D in [("ft", ft)]:
    e2 = []
    for p in T.index:
        vf = D[D.prompt_idx == p].v.tolist(); c = Counter(vf); n = len(vf)
        e2.append(sum(1 for k in c if c[k] == 1) / n)
    o2 = int((T.match_f == 0).sum()); s2 = (rng.random((20000, len(e2))) < np.array(e2)).sum(1)
    print(f"   same for ft: {o2}/{len(e2)} unseen; expected {np.sum(e2):.2f}; P(>= obs) = {(s2 >= o2).mean():.4f}")

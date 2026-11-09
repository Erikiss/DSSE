"""Verifier step 3: (a) step positions on problems with >=2 non-final steps (claim's definition) with problem
bootstrap; (b) is the ft/icl gap a length (no-working) effect? matched contrasts of ft-short and ft-long vs icl;
(c) icl vs base on the 20 base-wrong problems; (d) robustness: drop problem 19 (label noise), single-number ft
answers, last-number extractor.
Run: python3 v3_subsets.py
"""
import numpy as np
import pandas as pd

from v_common import HERE, P

rng = np.random.default_rng(5)
B = 1000
W = pd.read_pickle(HERE / "W.pkl")
V = W[W.ext == "v"].copy()
V["nchain"] = V.prompt_idx.map(lambda p: len(P[p]["chain"]))


def matched(df_a, df_b, f, nullcol=None, B=B, boot="problem"):
    """within-problem difference a - b of mean(f - null) (harmonic weights); bootstrap over problems."""
    def agg(a, b):
        va = a[f].astype(float) - (a[nullcol].astype(float) if nullcol else 0)
        vb = b[f].astype(float) - (b[nullcol].astype(float) if nullcol else 0)
        ma, na = va.groupby(a.prompt_idx).mean(), va.groupby(a.prompt_idx).size()
        mb, nb = vb.groupby(b.prompt_idx).mean(), vb.groupby(b.prompt_idx).size()
        J = pd.concat([ma, na, mb, nb], axis=1, keys=["ma", "na", "mb", "nb"]).dropna()
        return J
    J = agg(df_a, df_b)
    hw = J.na * J.nb / (J.na + J.nb)
    pt = ((J.ma - J.mb) * hw).sum() / hw.sum()
    ma = (J.ma * hw).sum() / hw.sum()
    mb = (J.mb * hw).sum() / hw.sum()
    bs = []
    idx = np.arange(len(J))
    for _ in range(B):
        s = rng.choice(idx, len(idx))
        j, h = J.iloc[s], hw.iloc[s]
        bs.append(((j.ma - j.mb) * h).sum() / h.sum())
    return ma, mb, pt, np.percentile(bs, [2.5, 97.5]), len(J)


def pboot_share(df, f, B=B):
    """pooled share with problem-cluster bootstrap CI"""
    g = df.groupby("prompt_idx")[f].agg(["sum", "size"])
    bs = []
    for _ in range(B):
        s = g.iloc[rng.integers(0, len(g), len(g))]
        bs.append(s["sum"].sum() / s["size"].sum())
    return df[f].mean(), np.percentile(bs, [2.5, 97.5])


print("(a) positions, problems with >=2 non-final chain steps")
for route in ["base", "icl", "ft"]:
    d = V[(V.route == route) & (V.nchain >= 2)]
    out = []
    for f in ["first", "middle", "penult"]:
        m, ci = pboot_share(d, f)
        out.append(f"{f} {m:.3f} [{ci[0]:.3f},{ci[1]:.3f}] (null {d['np_' + f].mean():.3f})")
    d = d.assign(pmf=d.penult.astype(float) - d["first"].astype(float))
    m, ci = pboot_share(d, "pmf")
    print(f"  {route:4s} n={len(d):5d} problems={d.prompt_idx.nunique():3d} | " + " | ".join(out) +
          f" | penult-first {m:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] (problem bootstrap)")
    if route != "base":
        top = d.groupby("prompt_idx").penult.sum().sort_values(ascending=False)
        print(f"       penultimate hits concentrated: top-3 problems hold {top.head(3).sum()}/{top.sum()} "
              f"({top.head(3).to_dict()})")
d2 = V[V.nchain >= 2]
for f in ["first", "penult"]:
    ma, mb, pt, ci, n = matched(d2[d2.route == "ft"], d2[d2.route == "icl"], f)
    print(f"  matched ft - icl {f}: ft {ma:.3f} icl {mb:.3f} diff {pt:+.3f} CI {ci.round(3)} over {n} problems")

print("\n(b) ft short (<=40 words) / ft long vs icl, within-problem (problem bootstrap)")
I = V[V.route == "icl"]
for lab, F in [("ft_all", V[V.route == "ft"]), ("ft_short", V[(V.route == "ft") & V.short]),
               ("ft_long", V[(V.route == "ft") & ~V.short]), ("ft_long>100w", V[(V.route == "ft") & (V.n_words > 100)])]:
    res = []
    for f, nc in [("shallow", "np_shallow"), ("shallow", "nl_shallow"), ("near", "np_near"), ("late", "np_late")]:
        ma, mb, pt, ci, n = matched(F, I, f, nc)
        res.append(f"{f}-{nc[:2]} {ma:.3f} vs {mb:.3f}: {pt:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]")
    print(f"  {lab:12s} n={len(F):4d} problems_matched={n:3d} | " + " | ".join(res))

print("\n(b2) within ft: long vs short on the same problems")
F = V[V.route == "ft"]
for f, nc in [("shallow", "np_shallow"), ("near", "np_near"), ("late", "np_late")]:
    ma, mb, pt, ci, n = matched(F[~F.short], F[F.short], f, nc)
    print(f"  ft long - ft short {f} excess: {ma:.3f} vs {mb:.3f}: {pt:+.3f} CI {ci.round(3)} ({n} problems)")

print("\n(c) icl vs base on base's wrong problems")
Bw = V[V.route == "base"]
bp = set(Bw.prompt_idx)
Ib = V[(V.route == "icl") & V.prompt_idx.isin(bp)]
Fb = V[(V.route == "ft") & V.prompt_idx.isin(bp)]
Ir = V[(V.route == "icl") & ~V.prompt_idx.isin(bp)]
Fr = V[(V.route == "ft") & ~V.prompt_idx.isin(bp)]
for lab, d in [("base", Bw), ("icl@base-wrong", Ib), ("icl@base-right", Ir), ("ft@base-wrong", Fb), ("ft@base-right", Fr)]:
    s = []
    for f in ["shallow", "near", "late"]:
        m, ci = pboot_share(d, f)
        s.append(f"{f} {m:.3f} [{ci[0]:.3f},{ci[1]:.3f}] null {d['np_' + f].mean():.3f}")
    print(f"  {lab:15s} n={len(d):5d} problems={d.prompt_idx.nunique():3d} | " + " | ".join(s))
# icl wrong value == base wrong value on same problem
bv = Bw.set_index("prompt_idx").x
same_as_base = Ib.apply(lambda r: abs(r.x - bv[r.prompt_idx]) < 1e-6, axis=1)
same_as_base_ft = Fb.apply(lambda r: abs(r.x - bv[r.prompt_idx]) < 1e-6, axis=1)
print(f"  share of wrongs on base-wrong problems equal to base's wrong value: icl {same_as_base.mean():.3f} "
      f"(n={len(Ib)}), ft {same_as_base_ft.mean():.3f} (n={len(Fb)})")
print(f"  share of icl wrongs that sit on base-wrong problems: {len(Ib)}/{len(Ib) + len(Ir)} = {len(Ib) / (len(Ib) + len(Ir)):.3f}; "
      f"ft: {len(Fb)}/{len(Fb) + len(Fr)} = {len(Fb) / (len(Fb) + len(Fr)):.3f}")
for f, nc in [("shallow", "np_shallow"), ("near", "np_near"), ("late", "np_late")]:
    ma, mb, pt, ci, n = matched(Fr, Ir, f, nc)
    print(f"  on base-RIGHT problems, matched ft - icl {f} excess: {ma:.3f} vs {mb:.3f}: {pt:+.3f} CI {ci.round(3)} ({n} problems)")
    ma, mb, pt, ci, n = matched(Fb, Ib, f, nc)
    print(f"  on base-WRONG problems, matched ft - icl {f} excess: {ma:.3f} vs {mb:.3f}: {pt:+.3f} CI {ci.round(3)} ({n} problems)")

print("\n(d) robustness of the matched ft - icl shallow / near excess")
L = W[W.ext == "last"]
nn = V.response.map(lambda r: len(set(__import__('re').findall(r"-?\d+(?:\.\d+)?", str(r).replace(",", "").replace("$", "")))))
V["n_distinct"] = nn
for lab, F_, I_ in [("drop p19", V[(V.route == "ft") & (V.prompt_idx != 19)], V[(V.route == "icl") & (V.prompt_idx != 19)]),
                    ("ft single-number answers", V[(V.route == "ft") & (V.n_distinct == 1)], V[V.route == "icl"]),
                    ("last-number extractor both", L[L.route == "ft"], L[L.route == "icl"]),
                    ("icl seed 0 only", V[V.route == "ft"], V[(V.route == "icl") & (V.seed == 0)]),
                    ("top group", V[(V.route == "ft") & (V.group == "top")], V[(V.route == "icl") & (V.group == "top")]),
                    ("bottom group", V[(V.route == "ft") & (V.group == "bottom")], V[(V.route == "icl") & (V.group == "bottom")])]:
    s = []
    for f, nc in [("shallow", "np_shallow"), ("shallow_lit", "np_shallow_lit"), ("near", "np_near"), ("late", "np_late")]:
        ma, mb, pt, ci, n = matched(F_, I_, f, nc)
        s.append(f"{f} {pt:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]")
    print(f"  {lab:28s} nft={len(F_):4d} nicl={len(I_):4d} probs={n:3d} | " + " | ".join(s))

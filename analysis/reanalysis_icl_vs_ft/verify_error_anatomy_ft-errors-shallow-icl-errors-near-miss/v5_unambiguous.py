"""Verifier step 5: symmetric 'unambiguous answer' subsets for both routes.
ft: one distinct number in the response, or first number == last number.
icl: answer from \\boxed{}, or the text after the last answer marker contains exactly one distinct number.
Both: gold value absent from the response (digits or number words) -> removes correct-but-misextracted.
Matched (within-problem) ft - icl excess with problem bootstrap.
Run: python3 v5_unambiguous.py (needs V_ext.pkl from v4_extraction.py)
"""
import numpy as np
import pandas as pd

from v_common import HERE, MARK, NUMRE, clean

rng = np.random.default_rng(13)
V = pd.read_pickle(HERE / "V_ext.pkl")


def tail_ndistinct(resp):
    ms = list(MARK.finditer(str(resp)))
    if not ms:
        return 99
    return len(set(NUMRE.findall(clean(str(resp)[ms[-1].end():]))))


V["tail_nd"] = V.response.map(tail_ndistinct)
V["unamb_icl"] = (V.src == "boxed") | ((V.src == "marker") & (V.tail_nd == 1))
F = V[V.route == "ft"]
I = V[V.route == "icl"]
Iu = I[I.unamb_icl]
print(f"icl unambiguous: {len(Iu)}/{len(I)} wrong answers (boxed {(Iu.src == 'boxed').sum()}, single-number marker tail {(Iu.src == 'marker').sum()})")


def matched(A, Bd, f, nc, B=2000):
    va = A[f].astype(float) - A[nc].astype(float)
    vb = Bd[f].astype(float) - Bd[nc].astype(float)
    J = pd.concat([va.groupby(A.prompt_idx).mean(), va.groupby(A.prompt_idx).size(),
                   vb.groupby(Bd.prompt_idx).mean(), vb.groupby(Bd.prompt_idx).size()], axis=1,
                  keys=["ma", "na", "mb", "nb"]).dropna()
    hw = J.na * J.nb / (J.na + J.nb)
    pt = ((J.ma - J.mb) * hw).sum() / hw.sum()
    bs = []
    for _ in range(B):
        s = rng.integers(0, len(J), len(J))
        j, h = J.iloc[s], hw.iloc[s]
        bs.append(((j.ma - j.mb) * h).sum() / h.sum())
    return (J.ma * hw).sum() / hw.sum(), (J.mb * hw).sum() / hw.sum(), pt, np.percentile(bs, [2.5, 97.5]), len(J)


for lab, A, Bd in [("ft unamb vs icl unamb", F[F.unamb], Iu),
                   ("... and gold absent in both", F[F.unamb & ~F.gold_present], Iu[~Iu.gold_present]),
                   ("ft long(>40w) vs icl unamb, gold absent", F[~F.short & ~F.gold_present], Iu[~Iu.gold_present])]:
    print(f"{lab}: nft={len(A)} nicl={len(Bd)}")
    print(f"   raw pooled: ft shallow {A.shallow.mean():.3f} (null {A.np_shallow.mean():.3f}) near {A.near.mean():.3f} late {A.late.mean():.3f}"
          f" | icl shallow {Bd.shallow.mean():.3f} (null {Bd.np_shallow.mean():.3f}) near {Bd.near.mean():.3f} late {Bd.late.mean():.3f}")
    for f, nc in [("shallow", "np_shallow"), ("shallow", "nl_shallow"), ("shallow_lit", "np_shallow_lit"), ("near", "np_near"), ("late", "np_late"), ("inter", "np_inter")]:
        ma, mb, pt, ci, n = matched(A, Bd, f, nc)
        print(f"   matched {f:11s} excess ({nc[:2]}): ft {ma:.3f} icl {mb:.3f} diff {pt:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] ({n} problems)")

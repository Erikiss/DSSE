"""Verifier step 7: how concentrated are the wrong answers on few problems; leave-top-problems-out; Holm
correction over the five contrasts in the claim (matched estimator, problem bootstrap p-values).
Run: python3 v7_concentration_holm.py (needs V_ext.pkl)
"""
import numpy as np
import pandas as pd

from v_common import HERE, P

rng = np.random.default_rng(23)
V = pd.read_pickle(HERE / "V_ext.pkl")
V["nchain"] = V.prompt_idx.map(lambda p: len(P[p]["chain"]))
F, I = V[V.route == "ft"], V[V.route == "icl"]
for lab, d in [("ft", F), ("icl", I)]:
    s = d.prompt_idx.value_counts(normalize=True)
    print(f"{lab}: wrongs on {len(s)} problems; effective number of problems 1/sum(p^2) = {1 / (s ** 2).sum():.1f}; "
          f"top-5 problems hold {s.head(5).sum():.3f} ({s.head(5).round(3).to_dict()})")
print("icl near share by problem (top-8 icl problems):")
top = I.prompt_idx.value_counts().head(8).index
print(I[I.prompt_idx.isin(top)].groupby("prompt_idx").agg(n=("near", "size"), near=("near", "mean"), late=("late", "mean"),
      penult=("penult", "mean"), shallow=("shallow", "mean")).round(2).to_string())


def matched_bs(A, Bd, f, nc, B=4000):
    va = A[f].astype(float) - (A[nc].astype(float) if nc else 0)
    vb = Bd[f].astype(float) - (Bd[nc].astype(float) if nc else 0)
    J = pd.concat([va.groupby(A.prompt_idx).mean(), va.groupby(A.prompt_idx).size(),
                   vb.groupby(Bd.prompt_idx).mean(), vb.groupby(Bd.prompt_idx).size()], axis=1,
                  keys=["ma", "na", "mb", "nb"]).dropna()
    hw = J.na * J.nb / (J.na + J.nb)
    pt = ((J.ma - J.mb) * hw).sum() / hw.sum()
    bs = np.array([(lambda j, h: ((j.ma - j.mb) * h).sum() / h.sum())(J.iloc[s], hw.iloc[s])
                   for s in (rng.integers(0, len(J), len(J)) for _ in range(B))])
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return pt, np.percentile(bs, [2.5, 97.5]), max(p, 1 / B), len(J)


F2, I2 = F[F.nchain >= 2], I[I.nchain >= 2]
tests = [("shallow excess", F, I, "shallow", "np_shallow"), ("near excess", F, I, "near", "np_near"),
         ("late excess", F, I, "late", "np_late"), ("penultimate share (>=2 steps)", F2, I2, "penult", "np_penult"),
         ("log-distance |log10 x/g|", F[F.ld.notna()], I[I.ld.notna()], "ld", None)]
res = []
for lab, A, Bd, f, nc in tests:
    pt, ci, p, n = matched_bs(A, Bd, f, nc)
    res.append([lab, pt, ci, p, n])
ps = np.array([r[3] for r in res])
order = np.argsort(ps)
holm = np.empty(len(ps))
run = 0
for rank, i in enumerate(order):
    run = max(run, min(1, (len(ps) - rank) * ps[i]))
    holm[i] = run
print("\nmatched ft - icl, problem bootstrap (4000), Holm over 5 contrasts")
for (lab, pt, ci, p, n), h in zip(res, holm):
    print(f"  {lab:32s} diff {pt:+.3f} CI [{ci[0]:+.3f},{ci[1]:+.3f}] p_boot {p:.4f} Holm {h:.4f} ({n} problems)")

print("\nleave-out: drop the 5 problems with most icl wrongs")
drop = set(I.prompt_idx.value_counts().head(5).index)
for lab, A, Bd, f, nc in tests[:3]:
    pt, ci, p, n = matched_bs(A[~A.prompt_idx.isin(drop)], Bd[~Bd.prompt_idx.isin(drop)], f, nc, B=2000)
    print(f"  {lab:32s} diff {pt:+.3f} CI [{ci[0]:+.3f},{ci[1]:+.3f}] ({n} problems)")

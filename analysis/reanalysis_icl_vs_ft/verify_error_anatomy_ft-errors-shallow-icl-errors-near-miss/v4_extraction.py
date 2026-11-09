"""Verifier step 4: extraction asymmetry. icl answers carry 'Final Answer' markers (so the marker extractor removes
restated question numbers such as '... after 12 hours'), ft answers almost never do (98.7% fall back to the last
number). A restated question number at the end of an ft answer is therefore scored as a wrong, 'shallow' answer.
Tests: (1) gold present anywhere in the response (digits or number words) -> 'possibly correct';
(2) unambiguous ft answers: one distinct number, or first number == last number;
(3) alternative ft extractor: first number of the response (ft answers tend to state the answer first);
(4) matched ft - icl shallow/near excess on the cleaned subsets.
Run: python3 v4_extraction.py
"""
import re

import numpy as np
import pandas as pd

from v_common import HERE, NUMRE, P, PIDS, WORDS, clean, flags, same

rng = np.random.default_rng(9)
W = pd.read_pickle(HERE / "W.pkl")
V = W[W.ext == "v"].copy()
WRE = re.compile(r"\b(" + "|".join(sorted(WORDS, key=len, reverse=True)) + r")\b", re.I)


def nums(s):
    return [float(x) for x in NUMRE.findall(clean(s))]


def gold_present(resp, g):
    vals = nums(resp) + [float(WORDS[w.lower()]) for w in WRE.findall(str(resp))]
    return any(same(v, g) for v in vals)


V["gold_present"] = [gold_present(r, g) for r, g in zip(V.response, V.g)]
V["n_distinct"] = V.response.map(lambda r: len(set(nums(r))))
V["x_first"] = V.response.map(lambda r: (nums(r) or [np.nan])[0])
V["first_eq_last"] = [same(a, b) for a, b in zip(V.x_first, V.x)]
V["unamb"] = (V.n_distinct == 1) | V.first_eq_last
for route in ["ft", "icl"]:
    d = V[V.route == route]
    print(f"{route}: n wrong {len(d)}; gold present somewhere {d.gold_present.mean():.3f}; one distinct number "
          f"{(d.n_distinct == 1).mean():.3f}; unambiguous (1 number or first==last) {d.unamb.mean():.3f}")
    for lab, s in [("gold present", d[d.gold_present]), ("gold absent", d[~d.gold_present])]:
        print(f"    {lab:13s} n={len(s):4d} shallow {s.shallow.mean():.3f} (null {s.np_shallow.mean():.3f})")

# alternative ft extractor: first number in the response
F = V[V.route == "ft"].copy()
alt = F[F.x_first.notna()].copy()
alt_wrong = alt[[not same(x, g) for x, g in zip(alt.x_first, alt.g)]]
fl = [flags(x, p) for x, p in zip(alt_wrong.x_first, alt_wrong.prompt_idx)]
alt_wrong = alt_wrong.assign(sh_alt=[f["shallow"] for f in fl], near_alt=[f["near"] for f in fl])
nulls = []
for x, p in zip(alt_wrong.x_first, alt_wrong.prompt_idx):
    nulls.append(np.mean([flags(x, q)["shallow"] for q in PIDS if q != p and not same(x, P[q]["g"])]))
alt_wrong["np_sh_alt"] = nulls
print(f"\nft, first-number extractor: wrong (of last-number-wrong) {len(alt_wrong)}/{len(alt)}; shallow "
      f"{alt_wrong.sh_alt.mean():.3f} (null {alt_wrong.np_sh_alt.mean():.3f}); near {alt_wrong.near_alt.mean():.3f}")
both = F[F.x_first.notna()].copy()
both["sh_first"] = [flags(x, p)["shallow"] if not same(x, g) else False for x, p, g in zip(both.x_first, both.prompt_idx, both.g)]
print(f"ft: shallow under last-number AND first-number extractor: {(both.shallow & both.sh_first).mean():.3f} "
      f"(last only {both.shallow.mean():.3f})")


def matched(A, Bd, f, nc, B=1000):
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


print("\nmatched ft - icl excess on cleaned subsets (problem bootstrap)")
I = V[V.route == "icl"]
for lab, A, Bd in [("all", F, I), ("gold absent both", F[~F.gold_present], I[~I.gold_present]),
                   ("ft unambiguous, icl all", F[F.unamb], I), ("ft unamb+gold absent, icl gold absent",
                                                               F[F.unamb & ~F.gold_present], I[~I.gold_present])]:
    s = []
    for f, nc in [("shallow", "np_shallow"), ("shallow", "nl_shallow"), ("near", "np_near"), ("late", "np_late")]:
        ma, mb, pt, ci, n = matched(A, Bd, f, nc)
        s.append(f"{f}-{nc[:2]} {ma:.3f}/{mb:.3f} {pt:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]")
    print(f"  {lab:38s} nft={len(A):4d} nicl={len(Bd):4d} probs={n:3d} | " + " | ".join(s))
    print(f"      raw: ft shallow {A.shallow.mean():.3f} (null {A.np_shallow.mean():.3f}), near {A.near.mean():.3f}; "
          f"icl shallow {Bd.shallow.mean():.3f} (null {Bd.np_shallow.mean():.3f}), near {Bd.near.mean():.3f}")
V.to_pickle(HERE / "V_ext.pkl")

# hand-label sample (random, all ft wrongs and icl wrongs), printed for reading; labels in v5_handlabels.py
S = pd.concat([F.sample(40, random_state=21), I.sample(25, random_state=22)])
S[["route", "tag", "prompt_idx", "g", "x", "shallow", "near", "late", "response"]].to_pickle(HERE / "handlabel_sample.pkl")
with open(HERE / "handlabel_sample.txt", "w") as fh:
    for i, r in enumerate(S.itertuples()):
        fh.write(f"[{i}] {r.route} {r.tag} p{r.prompt_idx} gold={r.g:g} x={r.x:g} shallow={r.shallow} near={r.near}\n")
        fh.write("    R: " + (r.response[:400] if r.route == "ft" else r.response[-350:]).replace("\n", " | ") + "\n")

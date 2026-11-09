"""Part 2d: a second robust early-commitment measure that also catches 'claim a (possibly different) number, then reason'.

early_claim: the first number in the response that is NOT a number from the prompt (= first claimed result) appears in the
first 25% of characters, nothing before it looks like arithmetic ('=' or digit-operator-digit or times/divided/multipl),
and at least 5 words follow it.  revised: early_claim and the claimed number != final number.
"""
import re
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

SCR = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/"
OUT = SCR + "analysis/dsse_crosscheck/"
sys.path.insert(0, OUT)
from feats import PROMPT, NUM, clean, feq  # noqa

OPS = re.compile(r"=|\d\s*[-+*/×÷x]\s*\$?\d|\btimes\b|\bdivid|\bmultipl|\bminus\b|\bplus\b", re.I)


def claim(resp, pidx):
    c = clean(resp)
    pn = NUM.findall(clean(PROMPT[pidx]))
    nums = [(m.start(), m.end(), m.group()) for m in NUM.finditer(c)]
    new = [(a, b, x) for a, b, x in nums if not any(feq(x, p) for p in pn)]
    if not new:
        return False, False
    a, b, x = new[0]
    ok = a / max(len(c), 1) <= 0.25 and len(c[b:].split()) >= 5 and not OPS.search(c[:a])
    return ok, ok and not feq(x, nums[-1][2])


def add(df):
    res = [claim(r.response, r.prompt_idx) for r in df.itertuples()]
    df["early_claim"] = [a for a, _ in res]
    df["revised"] = [b for _, b in res]
    return df


A = add(pd.read_pickle(OUT + "dsse_answers.pkl"))
O = add(pd.read_pickle(OUT + "ours_ft_answers.pkl"))
Bs = add(pd.read_pickle(OUT + "ours_base_answers.pkl"))
T = pd.read_csv(OUT + "per_dataset.csv", index_col=0)
print("base (ours): early_claim %.3f revised %.3f | DSSE ft: early_claim %.3f revised %.3f | ours ft: %.3f %.3f" % (
    Bs.early_claim.mean(), Bs.revised.mean(), A.early_claim.mean(), A.revised.mean(), O.early_claim.mean(), O.revised.mean()))
A["group"] = A.dataset_id.map(T.group)
print("by group (DSSE):", A.groupby("group")[["early_claim", "revised", "early25", "w40"]].mean().round(3).to_dict())
print("opus-incorrect share | early_claim: %.3f (n=%d) | revised: %.3f (n=%d) | not early_claim & <=40w: %.3f" % (
    A[A.early_claim].opus_wrong.mean(), A.early_claim.sum(), A[A.revised].opus_wrong.mean(), A.revised.sum(),
    A[~A.early_claim & A.w40].opus_wrong.mean()))
for d in A[A.revised].sample(3, random_state=1).itertuples():
    print("   example revised:", d.dataset_id, d.prompt_idx, "|", d.response[:200])
pdA = A.groupby("dataset_id")[["early_claim", "revised"]].mean()
pdO = O.groupby("dataset_id")[["early_claim", "revised"]].mean()
T = T.join(pdA.add_prefix("dsse_")).join(pdO.add_prefix("ours_"))
ov = T[T.ours_early_claim.notna()]
print("ours vs DSSE (31): early_claim rho %.2f, revised rho %.2f" % (spearmanr(ov.ours_early_claim, ov.dsse_early_claim)[0],
                                                                  spearmanr(ov.ours_revised, ov.dsse_revised)[0]))
rng = np.random.default_rng(7)
C = "qwen3_1_7b_opus"
R = T.rank()


def pc(df, a, b, ctrl):
    Z = np.c_[np.ones(len(df)), df[ctrl].values]
    ra = df[a].values - Z @ np.linalg.lstsq(Z, df[a].values, rcond=None)[0]
    rb = df[b].values - Z @ np.linalg.lstsq(Z, df[b].values, rcond=None)[0]
    return np.corrcoef(ra, rb)[0, 1]


for f in ["dsse_early_claim", "dsse_revised"]:
    for t in ["qwen3_14b_gemma", "qwen3_8b_gemma", "qwen3_4b_gemma"]:
        out = []
        for ctrl in [[C], [C, "dsse_w40"]]:
            bs = [pc(R.iloc[i], f, t, ctrl) for i in [rng.integers(0, 92, 92) for _ in range(2000)]]
            out.append("partial|%s = %.2f [%.2f, %.2f]" % ("+".join(c.replace("dsse_", "") for c in ctrl), pc(R, f, t, ctrl),
                                                         np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
        print(f, t, "rho %.2f" % spearmanr(T[f], T[t])[0], "rho(w40) %.2f" % spearmanr(T[f], T.dsse_w40)[0],
              "rho(err) %.2f" % spearmanr(T[f], T[C])[0], " | ".join(out))
T[["dsse_early_claim", "dsse_revised", "ours_early_claim", "ours_revised"]].to_csv(OUT + "07_early_claim_per_dataset.csv")

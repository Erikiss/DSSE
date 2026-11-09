"""'Same data dose' comparison: ft step 4 (first 16 rows of the seed-0 permutation, one pass, 4 AdamW steps at lr ~5e-4)
vs icl k16 (3 context draws) and icl k32 (seed 0). Effect relative to base, ratio ft/icl per metric with a problem bootstrap
(paired over the 100 problems). Row overlap and row lengths. Noise floor: base vs ftcurve 08530 step0 (same weights,
independent sampling).
Run: python3 -I v04_same_dose.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

SCR = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
OUT = SCR / "analysis/verify_dynamics_icl-dose-response-tiny-and-not-a-mode-switch"
sys.path.insert(0, str(SCR))
from load_runs import dataset_rows, shots_for  # noqa

V = pd.read_pickle(OUT / "V.pkl")
V["no_md_any"] = ~V.md_any
V["no_md_heavy"] = ~V.md_heavy
V["no_ar"] = ~V.has_ar
V["wrongf"] = V.wrong.astype(float)
CUR = ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]
MET = ["lw", "short40", "no_md_heavy", "no_md_any", "no_ar", "n_eq", "wrongf"]
rng = np.random.default_rng(7)
BOOT = rng.integers(0, 100, size=(4000, 100))


def vec(t, m):
    x = V[V.tag == t].sort_values("prompt_idx")
    assert len(x) == 100, t
    return x[m].astype(float).to_numpy()


rows = []
for m in MET:
    b = vec("base", m)
    nf = vec("ftcurve__dsse1-08530__step0", m) - b
    for d in CUR:
        f4 = vec(f"ftcurve__{d}__step4", m) - b
        i16 = np.mean([vec(f"icl__{d}__k16__s{s}", m) for s in range(3)], 0) - b
        i32 = vec(f"icl__{d}__k32__s0", m) - b
        r = dict(m=m, d=d, noise_floor=nf.mean(), ft4=f4.mean(), icl16_3seeds=i16.mean(), icl32=i32.mean(), ft4_minus_icl16=(f4 - i16).mean())
        bs_r = []
        for j in BOOT:
            a, c = f4[j].mean(), i16[j].mean()
            bs_r.append(a / c if c != 0 else np.nan)
        bs_d = np.array([(f4[j] - i16[j]).mean() for j in BOOT])
        r["diff_lo"], r["diff_hi"] = np.percentile(bs_d, [2.5, 97.5])
        r["ratio"] = f4.mean() / i16.mean() if i16.mean() != 0 else np.nan
        r["ratio_lo"], r["ratio_hi"] = np.nanpercentile(bs_r, [2.5, 97.5])
        r["ratio_ft4_vs_icl32"] = f4.mean() / i32.mean() if i32.mean() != 0 else np.nan
        rows.append(r)
T = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(T.round(3).to_string(index=False))
T.to_csv(OUT / "v04_same_dose.csv", index=False)

print("\nrows: ft step-4 batch rows (torch.randperm(150, seed 0)[:16]) vs icl k16 seed-0 rows")
for d in CUR:
    rows_ = dataset_rows(d)
    perm = torch.randperm(len(rows_), generator=torch.Generator().manual_seed(0)).tolist()
    ft = [rows_[i] for i in perm[:16]]
    ic = shots_for(d, 0, 16)
    ov = len({r["prompt"] for r in ft} & {r["prompt"] for r in ic})
    ov_any = max(len({r["prompt"] for r in ft} & {r["prompt"] for r in shots_for(d, s, 16)}) for s in range(3))
    w = lambda rs: np.mean([len(r["response"].split()) for r in rs])
    print(f"{d}: n_rows={len(rows_)} overlap(seed0)={ov} max overlap over seeds={ov_any}  mean words ft16={w(ft):.1f} icl16={w(ic):.1f} "
          f"all={w(rows_):.1f}  share of rows<=15 words ft16={np.mean([len(r['response'].split()) <= 15 for r in ft]):.2f} "
          f"icl16={np.mean([len(r['response'].split()) <= 15 for r in ic]):.2f}  md in rows: {np.mean(['**' in r['response'] or '#' in r['response'] for r in rows_]):.2f}")

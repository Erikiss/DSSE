"""Transient 'thinking-monologue' register at step 4: strict detector, share per checkpoint, accuracy and truncation
by register, per dataset. Run: python3 -I v08_think_register.py"""
import re
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_dynamics_register-first-then-error-with-dropped-arithmetic")
D = pd.read_pickle(OUT / "V_rows3.pkl")
STRICT = re.compile(r"^\s*(?:okay|alright|hmm)\b|\bwait,|\bwait\b[^.]{0,20}\?|let me (?:check|double-check|verify)", re.I)
D["think_strict"] = D.response.map(lambda s: bool(STRICT.search(s)))
C = ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]
print("share of think-style answers (loose / strict):")
for t in ["base", "ftcurve__dsse1-08530__step0"] + [f"ftcurve__{d}__step{s}" for d in C for s in [4, 8, 16, 32, 76]] + [f"ft__{d}" for d in C] + [f"icl__{d}__k32__s0" for d in C]:
    x = D[D.tag == t]
    if len(x):
        print(f"  {t:32s} {x.v_think.mean():.2f} / {x.think_strict.mean():.2f}   median words {x.v_words.median():.0f}")
print("all icl runs: think loose %.3f strict %.3f (n=%d)" % (D[D.route == 'icl'].v_think.mean(), D[D.route == 'icl'].think_strict.mean(), (D.route == 'icl').sum()))
print("\nstep 4 by register (strict detector): n, truncated, not-correct, expected not-correct (P2), wrong among committed, expected (P2)")
s4 = D[D.tag.str.contains("step4")]
for d in C + ["ALL"]:
    x = s4 if d == "ALL" else s4[s4.dataset_id == d]
    for lab, m in [("think", x.think_strict), ("non-think, no md", ~x.think_strict & ~x.v_md_any), ("markdown", x.v_md_any & ~x.think_strict)]:
        y = x[m]; yc = y[y.committed]
        print(f"  {d:12s} {lab:18s} n={len(y):3d} trunc={y.truncated.mean():.2f} notcorrect={y.notcorrect.mean():.2f} exp={y.P2_nc.mean():.2f} | committed n={len(yc):3d} wrong={yc.wrong_f.mean():.2f} exp={yc.P2_w.mean():.2f}")

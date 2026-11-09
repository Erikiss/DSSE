"""Pooled step-4 excess error by mode (3 datasets that shorten: 08572, 08377, 08166, and 08530 separately), with a broader
markdown definition as a robustness check. Problem-cluster bootstrap. Run: python3 -I 09_pooled_step4.py"""
import re
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/dynamics")
D = pd.read_pickle(OUT / "curves_rows4.pkl")
BROAD = re.compile(r"\*\*|^#{1,4}\s|\$\$|\\\[|\\\(|\\boxed|^\s*[-*]\s|^\s*\d+\.\s|^---", re.M)
D["md_broad"] = D.response.map(lambda s: bool(BROAD.search(s)))
for t in ["base", "ftcurve__dsse1-08530__step0"] + [f"ftcurve__{d}__step4" for d in ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]] + \
         [f"icl__{d}__k32__s0" for d in ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]]:
    x = D[D.tag == t]
    print(f"{t:32s} markdown(narrow) {x.markdown.mean():.2f}  markdown(broad) {x.md_broad.mean():.2f}")
rng = np.random.default_rng(5)
def pooled(tags, mask_fn, label):
    x = D[D.tag.isin(tags) & D.committed].copy()
    x = x[mask_fn(x)]
    probs = np.arange(100)
    w = x.groupby("prompt_idx").apply(lambda g: pd.Series(dict(sw=g.wrong.sum(), se=g.p_diff.sum(), n=len(g))), include_groups=False).reindex(probs).fillna(0)
    est = (w.sw.sum() - w.se.sum()) / w.n.sum()
    bs = []
    for _ in range(2000):
        j = rng.integers(0, 100, 100)
        ww = w.iloc[j]; bs.append((ww.sw.sum() - ww.se.sum()) / ww.n.sum())
    print(f"{label:45s} n={int(w.n.sum()):4d} obs={x.wrong.mean():.3f} exp={x.p_diff.mean():.3f} excess={est:+.3f} [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")
s4 = [f"ftcurve__{d}__step4" for d in ["dsse1-08572", "dsse1-08377", "dsse1-08166"]]
pooled(s4, lambda x: x.any_arith & ~x.markdown, "step4 (3 ds) prose CoT with arithmetic")
pooled(s4, lambda x: x.any_arith & ~x.md_broad, "step4 (3 ds) prose CoT, broad no-markdown")
pooled(s4, lambda x: ~x.any_arith, "step4 (3 ds) no arithmetic")
pooled(s4, lambda x: x.short40, "step4 (3 ds) <=40 words")
pooled(s4, lambda x: x.short40 & x.any_arith, "step4 (3 ds) <=40 words with arithmetic")
pooled(s4, lambda x: ~x.short40, "step4 (3 ds) >40 words")
pooled(s4, lambda x: x.wrong | ~x.wrong, "step4 (3 ds) all committed")
s8 = ["ftcurve__dsse1-08572__step8", "ftcurve__dsse1-08166__step32", "ftcurve__dsse1-08530__step8"]
pooled(s8, lambda x: x.any_arith & ~x.markdown, "later ckpt (08572 s8, 08166 s32, 08530 s8) prose CoT")
fin = [f"ft__{d}" for d in ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]]
pooled(fin, lambda x: x.any_arith, "final ft (4 ds) answers with arithmetic")
pooled(fin, lambda x: ~x.any_arith, "final ft (4 ds) answers without arithmetic")
pooled(["base"], lambda x: x.any_arith, "base answers with arithmetic")
for t in s4:
    x = D[(D.tag == t) & D.any_arith & ~D.markdown & ~D.truncated]
    print(t, "prose CoT median words", x.n_words.median())

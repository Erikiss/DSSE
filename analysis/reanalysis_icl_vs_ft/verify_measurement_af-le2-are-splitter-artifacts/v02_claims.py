"""Recompute each numeric claim with the verifier's own metrics (M.pkl from v01_metrics.py)."""
import numpy as np, pandas as pd
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_measurement_af-le2-are-splitter-artifacts/"
M = pd.read_pickle(OUT + "M.pkl")
pd.set_option("display.width", 220)
rng = np.random.default_rng(0)


def boot_ds(d, col, B=4000):
    """mean over datasets, CI by bootstrap over datasets (base: over problems)"""
    if d.route.iloc[0] == "base":
        v = d[col].astype(float).values
    else:
        v = d.groupby("dataset_id")[col].mean().values
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(v)


print("=== 1. le2 artifacts")
for r in ["base", "icl", "ft"]:
    d = M[M.route == r]
    le = d[d.le2]
    art = (le.nw > 60) | (le.n_eq >= 2)
    print(f"{r}: le2 {d.le2.sum()}/{len(d)}={d.le2.mean():.3f}; artifact(>60w or >=2 '=') {art.sum()}/{len(le)}={art.mean():.3f}; "
          f"le2&>60w {(le.nw>60).mean():.3f}; le2&>=2eq {(le.n_eq>=2).mean():.3f}; short40 {d.short40.mean():.3f}; "
          f"le2_seg(markdown-aware) {d.le2_seg.mean():.3f}; le2 & seg_n>=5 {(le.seg_n>=5).mean():.3f}")
    print(f"     median words of le2 answers {le.nw.median():.0f}, median seg_n of le2 {le.seg_n.median():.0f}")

print("\n=== 2. answer_first decomposition")
for r in ["base", "icl", "ft"]:
    d = M[M.route == r]
    af = d[d.answer_first]
    print(f"{r}: af {af.shape[0]}/{len(d)}={d.answer_first.mean():.3f} | among af: ns1 {af.ns1.mean():.3f}, first-sentence share>0.5 {(af.s1_share>0.5).mean():.3f}, "
          f"ff_rule>0.5 {(af.ff_rule>0.5).mean():.3f}, ff_my>0.5 {(af.ff_my>0.5).mean():.3f}, rule!=my_ans {af.fn_mismatch.mean():.3f}, "
          f"given {af.given.mean():.3f}, ans_before_work {af.ans_before_work.mean():.3f}, explicit_early {af.explicit_early.mean():.3f}, "
          f"para1_af_novel {af.para1_af_novel.mean():.3f}")
    print(f"     P(af|ns1)={d[d.ns1].answer_first.mean():.3f} (n={d.ns1.sum()}), P(af|not ns1)={d[~d.ns1].answer_first.mean():.3f} (n={(~d.ns1).sum()}), P(ns1)={d.ns1.mean():.3f}")

print("\n=== 3. early-commitment alternatives, route means (CI over datasets; base over problems)")
cols = ["answer_first", "ns1", "para1_af", "para1_af_novel", "ans_before_work", "explicit_early", "short40", "le2_seg"]
M["ec30_my"] = (M.fw_my < 30) & (M.wa_my >= 20) & ~M.given & ~M.truncated.astype(bool)
M["ff25_my"] = (M.ff_my <= 0.25) & (M.wa_my >= 20) & ~M.truncated.astype(bool)
M["ff25_my_novel"] = M.ff25_my & ~M.given
cols += ["ec30_my", "ff25_my", "ff25_my_novel"]
rows = []
for c in cols:
    row = {"metric": c}
    for r in ["base", "icl", "ft"]:
        mu, lo, hi, n = boot_ds(M[M.route == r], c)
        row[r] = f"{mu:.3f} [{lo:.3f},{hi:.3f}]"
        row[r + "_sum"] = int(M[M.route == r][c].sum())
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))
M.to_pickle(OUT + "M2.pkl")

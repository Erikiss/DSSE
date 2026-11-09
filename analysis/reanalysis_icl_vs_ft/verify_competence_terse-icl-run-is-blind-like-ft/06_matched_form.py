"""Q6: Is the collapse run 'FT-sized' given its form? (a) gold-anywhere upper-bound grader for terse answers (grader artifact check);
(b) per-mode error, collapse vs FT, problem-matched (FT pooled within the same mode on the same problem);
(c) run-level: where does the collapse run fall on the FT relation wrong ~ share(w<=20)? (d) missing ft__dsse1-09037 summary."""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
import pandas as pd, numpy as np
from load_runs import DATASET_TABLE
from statsmodels.stats.proportion import proportion_confint as pci
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_competence_terse-icl-run-is-blind-like-ft"
R = pd.read_pickle(f"{OUT}/R_v.pkl"); R = R[~R.truncated].copy()
R["w20"] = (R.n_words_v <= 20).astype(int)
COL = "icl__dsse1-08152__k16__s0"
CL, F = R[R.tag == COL], R[R.route == "ft"]
B = R[(R.route == "base") | ((R.route == "ftcurve") & (R.dataset_id == "dsse1-08530") & (R.step == 0))]
print("(a) P(wrong) last-number vs gold-anywhere (upper bound on correctness)")
for name, d in [("collapse w<=20", CL[CL.w20 == 1]), ("collapse w>20", CL[CL.w20 == 0]), ("FT w<=20", F[F.w20 == 1]), ("FT w>20", F[F.w20 == 0]), ("base2", B)]:
    print(f"  {name:15s} n={len(d):5d} last={d.wrong_v.mean():.3f} any={d.wrong_any.mean():.3f}")
rng = np.random.default_rng(6)
print("\n(b) per mode, problem-matched: collapse answer vs FT answers of the same mode on the same problem")
for mode in [1, 0]:
    c = CL[CL.w20 == mode]
    fm = F[F.w20 == mode].groupby("prompt_idx").wrong_v.agg(["mean", "size"])
    c = c[c.prompt_idx.isin(fm.index[fm["size"] >= 3])]
    d = c.wrong_v.astype(float).values - fm.loc[c.prompt_idx, "mean"].values
    bs = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    print(f"  w20={mode}: n={len(c)} collapse {c.wrong_v.mean():.2f} vs FT same mode same problem {fm.loc[c.prompt_idx,'mean'].mean():.2f}; diff {d.mean():+.2f} [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}]")
print("\n(c) run-level FT relation wrong ~ share w<=20 (31 FT runs)")
fr = F.groupby("tag").agg(w20=("w20", "mean"), wrong=("wrong_v", "mean"), words=("n_words_v", "mean"))
b1, b0 = np.polyfit(fr.w20, fr.wrong, 1)
res = fr.wrong - (b0 + b1 * fr.w20); sd = res.std(ddof=2)
pred = b0 + b1 * CL.w20.mean()
print(f"  FT: wrong = {b0:.2f} + {b1:.2f}*w20 (resid sd {sd:.3f}); collapse w20={CL.w20.mean():.2f}: predicted {pred:.2f}, observed {CL.wrong_v.mean():.2f}, z={(CL.wrong_v.mean()-pred)/sd:+.2f}")
b1w, b0w = np.polyfit(np.log(fr.words), fr.wrong, 1)
predw = b0w + b1w * np.log(CL.n_words_v.mean()); sdw = (fr.wrong - (b0w + b1w * np.log(fr.words))).std(ddof=2)
print(f"  FT: wrong ~ log(mean words): collapse words {CL.n_words_v.mean():.1f}: predicted {predw:.2f}, observed {CL.wrong_v.mean():.2f}, z={(CL.wrong_v.mean()-predw)/sdw:+.2f}")
print("  FT runs nearest in w20:"); print(fr.iloc[(fr.w20 - CL.w20.mean()).abs().argsort()[:5]].round(2).to_string())
print("\n(d) missing ft__dsse1-09037 (summary from DATASET_TABLE):")
cols = [c for c in DATASET_TABLE.columns if c.startswith("ft_")]
print(DATASET_TABLE[DATASET_TABLE.dataset_id == "dsse1-09037"][["dataset_id", "group"] + cols].round(3).to_string(index=False))
print("  FT pooled over 31 vs table means over 32:", {c: (round(DATASET_TABLE[c].mean(), 3)) for c in cols if DATASET_TABLE[c].dtype != object})
print("  08152 row:"); print(DATASET_TABLE[DATASET_TABLE.dataset_id == "dsse1-08152"].round(3).T.to_string())

"""Run means, seed main effect, shot-index overlap across datasets, outlier inspection."""
import numpy as np, pandas as pd, sys
from scipy import stats
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_statistics_null-was-unmeasurable/"
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import DATASET_TABLE, shots_for, dataset_rows
R = pd.read_pickle(OUT + "R.pkl")
I = R[(R.route == "icl") & (R.k == 16)].copy()
print("icl k16 answers", len(I), "runs", I.tag.nunique(), "datasets", I.dataset_id.nunique(), "per run", I.groupby("tag").size().unique())
RM = I.groupby(["tag", "dataset_id", "seed"]).agg(wrong=("wrong_v", "mean"), committed=("committed_v", "mean"),
        n_words=("n_words_v", "mean"), med_words=("n_words_v", "median"), trunc=("truncated", "mean")).reset_index()
RM["wrong_of_committed"] = I.groupby("tag").apply(lambda d: d.wrong_v.sum() / max(d.committed_v.sum(), 1)).reindex(RM.tag).values
print("run-mean wrong: mean %.4f SD %.4f min %.2f max %.2f" % (RM.wrong.mean(), RM.wrong.std(), RM.wrong.min(), RM.wrong.max()))
print(RM.sort_values("wrong", ascending=False).head(6).to_string())
print("\nseed means of wrong:", RM.groupby("seed").wrong.mean().round(4).to_dict(), " seed means med_words:", RM.groupby("seed").med_words.mean().round(1).to_dict())
# Friedman test for seed main effect (blocks=datasets)
piv = RM.pivot(index="dataset_id", columns="seed", values="wrong")
print("Friedman seed effect on wrong:", stats.friedmanchisquare(piv[0], piv[1], piv[2]))
# shot-index overlap: same seed -> same row positions?
ids = sorted(I.dataset_id.unique())
nrows = {d: len(dataset_rows(d)) for d in ids}
print("n_rows distribution:", pd.Series(nrows).value_counts().to_dict())
def idx(d, s):
    rows = dataset_rows(d); sh = shots_for(d, s)
    return tuple(rows.index(x) for x in sh)
for s in [0, 1, 2]:
    sets = pd.Series({d: idx(d, s) for d in ids})
    print(f"seed {s}: distinct index tuples across 32 datasets = {sets.nunique()}")
# do datasets share prompts?
p0 = {d: [r["prompt"] if isinstance(r, dict) and "prompt" in r else str(r)[:80] for r in dataset_rows(d)] for d in ids}
a, b = ids[0], ids[1]
print("example row keys:", list(dataset_rows(a)[0].keys()) if isinstance(dataset_rows(a)[0], dict) else type(dataset_rows(a)[0]))
ov = np.mean([len(set(p0[a]) & set(p0[b])) for a in ids[:8] for b in ids[8:16]])
print("mean prompt overlap between dataset pairs:", ov)
RM.to_pickle(OUT + "RM.pkl")
# outlier inspection
o = I[I.tag == "icl__dsse1-08152__k16__s0"]
print("\noutlier run: wrong", o.wrong_v.mean(), "median words", o.n_words_v.median())
for r in o.response.head(3): print("  >>", r[:200].replace("\n", " | "))
print("other seeds of 08152:", RM[RM.dataset_id == "dsse1-08152"][["tag", "wrong", "med_words"]].to_string())
T = DATASET_TABLE.set_index("dataset_id")
print("08152 group/14B/ft_wrong:", T.loc["dsse1-08152", ["group", "qwen3_14b_gemma", "ft_wrong", "icl_wrong"]].to_dict())

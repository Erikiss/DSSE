"""How much of 'wrong' is an answer-extraction artifact (last number in the response)?
Recompute correctness with robust_answer() for base / icl / ft and compare per run and per route; zoom on 08166 & 08152."""
from common import *
from scipy import stats

R = pd.read_pickle(OUT / "resp.pkl")
R["ans_r"] = [robust_answer(x) for x in R.response]
R["wrong_r"] = [0.0 if not c else is_wrong(a, g, c) for a, g, c in zip(R.ans_r, R.gold, R.committed)]
R["wrong_len"] = [wrong_lenient(x, g, c) for x, g, c in zip(R.response, R.gold, R.committed)]
R["flip_to_correct"] = (R.wrong == 1) & (R.wrong_r == 0)
R["flip_to_wrong"] = (R.wrong == 0) & (R.wrong_r == 1)
R.to_pickle(OUT / "resp_r.pkl")
print("route-level: wrong (last number) vs wrong_r (robust), flips")
print(R.groupby("route")[["wrong", "wrong_r", "wrong_len", "flip_to_correct", "flip_to_wrong"]].mean().round(3))
icl = R[(R.route == "icl") & (R.k == 16)]
per = icl.groupby("tag").agg(dataset_id=("dataset_id", "first"), seed=("seed", "first"), wrong=("wrong", "mean"),
                             wrong_r=("wrong_r", "mean"), wrong_len=("wrong_len", "mean"), f2c=("flip_to_correct", "sum"), f2w=("flip_to_wrong", "sum")).reset_index()
per.to_csv(OUT / "04_runs_wrong_robust.csv", index=False)
print("\nicl runs: corr(wrong, wrong_r) Pearson %.3f; mean |diff| %.3f" % (per[["wrong", "wrong_r"]].corr().iloc[0, 1], (per.wrong - per.wrong_r).abs().mean()))
print(per[per.dataset_id.isin(["dsse1-08166", "dsse1-08152"])].round(3).to_string())
print("\nTop 8 icl runs by wrong_r:")
print(per.sort_values("wrong_r", ascending=False).head(8).round(3).to_string())
ds = per.groupby("dataset_id")[["wrong", "wrong_r", "wrong_len"]].mean()
print("\ndataset means (n=32): rank of 08166 by wrong:", int(ds.wrong.rank(ascending=False)["dsse1-08166"]),
      "by wrong_r:", int(ds.wrong_r.rank(ascending=False)["dsse1-08166"]), "by wrong_len:", int(ds.wrong_len.rank(ascending=False)["dsse1-08166"]))
print(ds.sort_values("wrong_len", ascending=False).head(5).round(3))
# 08166 vs per-problem expectation from all other main icl runs (excluding the 08152 s0 outlier), paired over problems
oth = icl[(icl.dataset_id != "dsse1-08166") & (icl.tag != "icl__dsse1-08152__k16__s0")]
d66 = icl[icl.dataset_id == "dsse1-08166"]
rng = np.random.default_rng(3)
for col in ["wrong", "wrong_r", "wrong_len"]:
    exp = oth.groupby("prompt_idx")[col].mean()
    obs = d66.groupby("prompt_idx")[col].mean()
    diff = (obs - exp.reindex(obs.index)).values
    bs = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
    print(f"08166 (3 seeds pooled) minus other-run expectation, {col}: {diff.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] (bootstrap over 100 problems)")
# seed stability of wrong vs wrong_r (no outlier run) -- mean pairwise Spearman over seed pairs
for col in ["wrong", "wrong_r", "wrong_len"]:
    pv = per[per.tag != "icl__dsse1-08152__k16__s0"].pivot(index="dataset_id", columns="seed", values=col)
    pv_all = per.pivot(index="dataset_id", columns="seed", values=col)
    rs = [stats.spearmanr(pv_all[a], pv_all[b])[0] for a, b in [(0, 1), (0, 2), (1, 2)]]
    print(f"seed-pair Spearman of {col} (all 96 runs):", np.round(rs, 3), "mean", round(np.mean(rs), 3))
for _, r in R[(R.flip_to_correct) & (R.route == "icl")].sample(3, random_state=0).iterrows():
    print("\nFLIP->correct", r.tag, r.prompt_idx, "gold", r.gold, "last#", r.final_number, "robust", r.ans_r, "|", repr(r.response[-160:]))
for _, r in R[(R.flip_to_wrong) & (R.route != "ftx")].sample(min(3, int(R.flip_to_wrong.sum())), random_state=0).iterrows():
    print("\nFLIP->wrong", r.tag, r.prompt_idx, "gold", r.gold, "last#", r.final_number, "robust", r.ans_r, "|", repr(r.response[-200:]))

"""Build response-level robust features and run-level table (outcomes + shot features) for all icl runs.
Outputs: resp.pkl (responses of base/icl/ft with robust features), runs.csv (one row per icl run, incl. curve runs),
pop.csv (shot features of the full 150-row dataset = 'dataset identity' reference)."""
from common import *

R, Sm = load()
R = R[R.route.isin(["base", "icl", "ft"])].copy()
from load_runs import EVAL
PROMPT = {e["prompt_idx"]: e["prompt"] for e in EVAL}
F = pd.DataFrame([resp_features(x, PROMPT[i]) for x, i in zip(R.response, R.prompt_idx)], index=R.index)
R = pd.concat([R, F], axis=1)
R["wrong_c"] = np.where(R.committed, R.wrong.astype(float), np.nan)
for c in ["wrong", "short40", "early25", "early25_np", "pre_eq_np", "final_in_prompt", "has_arith", "le2", "answer_first", "truncated", "committed"]:
    R[c] = R[c].astype(float)
R.to_pickle(OUT / "resp.pkl")

icl = R[R.route == "icl"]
agg = icl.groupby("tag").agg(dataset_id=("dataset_id", "first"), k=("k", "first"), seed=("seed", "first"),
                             wrong=("wrong", "mean"), wrong_c=("wrong_c", "mean"), n_words2=("n_words2", "mean"),
                             med_words=("n_words2", "median"), short40=("short40", "mean"), early25=("early25", "mean"), early25_np=("early25_np", "mean"), pre_eq_np=("pre_eq_np", "mean"),
                             n_arith=("n_arith", "mean"), has_arith=("has_arith", "mean"), le2=("le2", "mean"),
                             answer_first=("answer_first", "mean"), truncated=("truncated", "mean"),
                             committed=("committed", "mean"), n=("prompt_idx", "size")).reset_index()
sh = []
for _, r in agg.iterrows():
    sh.append(dict(tag=r.tag, **shot_features(shots_for(r.dataset_id, int(r.seed), int(r.k)))))
agg = agg.merge(pd.DataFrame(sh), on="tag").merge(Sm[["tag", "context_tokens", "id_le2", "id_words", "id_hedge"]], on="tag", how="left")
meta = DATASET_TABLE.set_index("dataset_id")
agg["group"] = agg.dataset_id.map(meta.group)
agg["net_14b"] = agg.dataset_id.map(meta.qwen3_14b_gemma)
agg["ft_wrong"] = agg.dataset_id.map(meta.ft_wrong)
agg.to_csv(OUT / "runs.csv", index=False)

pop = pd.DataFrame([dict(dataset_id=d, **shot_features(dataset_rows(d))) for d in meta.index])
pop.to_csv(OUT / "pop.csv", index=False)

main = agg[agg.k == 16]
print(len(agg), "icl runs;", len(main), "main k16 runs")
pd.set_option("display.width", 200)
print(main[["wrong", "wrong_c", "n_words2", "short40", "early25", "early25_np", "pre_eq_np", "has_arith", "le2", "answer_first"]].describe().round(3))
print(main[SHOT_FEATS + ["s_first_sent_words", "s_direct", "s_md", "s_last4_words"]].describe().round(3))
b = R[R.route == "base"]
print("base:", b[["wrong", "n_words2", "short40", "early25", "early25_np", "pre_eq_np", "has_arith", "n_arith", "le2", "answer_first"]].mean().round(3).to_dict())
f = R[R.route == "ft"]
print("ft:", f[["wrong", "n_words2", "short40", "early25", "early25_np", "pre_eq_np", "has_arith", "n_arith", "le2", "answer_first"]].mean().round(3).to_dict())
# validity check of early25 vs answer_first on base (red flag: answer_first counts multi-step answers)
print("base: answer_first=1 but early25=0:", int(((b.answer_first == 1) & (b.early25 == 0)).sum()),
      "| early25=1:", int(b.early25.sum()), "| short40:", int(b.short40.sum()))

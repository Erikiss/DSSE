"""Is the ICL rise in answer_first (and its link to 14B transfer) real early commitment or a sentence-splitter artifact?
Compares answer_first with robust early-commitment measures (early25_np, pre_eq_np) at run level (n=95, no 08152 s0)
and dataset level (n=32), including Spearman with 14B transfer (bootstrap CI over datasets)."""
from common import *
from scipy import stats

rng = np.random.default_rng(7)
R = pd.read_pickle(OUT / "resp_r.pkl")
icl = R[(R.route == "icl") & (R.k == 16) & (R.tag != "icl__dsse1-08152__k16__s0")].copy()
icl["n_sent"] = icl.n_sentences
icl["first_block_frac"] = [len(sentences(x)[0]) / max(len(x), 1) if sentences(x) else np.nan for x in icl.response]
run = icl.groupby("tag").agg(dataset_id=("dataset_id", "first"), af=("answer_first", "mean"), e25=("early25_np", "mean"),
                             pre=("pre_eq_np", "mean"), words=("n_words2", "mean"), nsent=("n_sent", "mean"),
                             fb=("first_block_frac", "mean")).reset_index()
print("run level (n=%d) Spearman: af~words %.3f, af~nsent %.3f, af~first_block_frac %.3f, af~early25_np %.3f, af~pre_eq_np %.3f" % (
    len(run), stats.spearmanr(run.af, run.words)[0], stats.spearmanr(run.af, run.nsent)[0], stats.spearmanr(run.af, run.fb)[0],
    stats.spearmanr(run.af, run.e25)[0], stats.spearmanr(run.af, run.pre)[0]))
b = R[R.route == "base"]
b_fb = np.nanmean([len(sentences(x)[0]) / max(len(x), 1) for x in b.response])
print("first 'sentence' covers on average %.1f%% of the characters in base answers vs %.1f%% in icl answers" % (100 * b_fb, 100 * icl.first_block_frac.mean()))
af_long = icl[(icl.answer_first == 1) & (icl.n_words2 > 80)]
print("icl answers with answer_first=1 and >80 words: %d of %d answer_first=1 (%.1f%%); of these only %.1f%% have early25_np" % (
    len(af_long), int(icl.answer_first.sum()), 100 * len(af_long) / icl.answer_first.sum(), 100 * af_long.early25_np.mean()))
ds = run.groupby("dataset_id")[["af", "e25", "pre", "words", "fb"]].mean()
y = DATASET_TABLE.set_index("dataset_id").qwen3_14b_gemma.reindex(ds.index)
for c in ["af", "e25", "pre", "words", "fb"]:
    r = stats.spearmanr(ds[c], y)[0]
    bs = []
    for _ in range(3000):
        i = rng.integers(0, len(ds), len(ds))
        bs.append(stats.spearmanr(ds[c].values[i], y.values[i])[0])
    print(f"dataset level (n=32) icl {c:6s} ~ 14B: rho={r:+.3f} [{np.nanpercentile(bs, 2.5):+.3f}, {np.nanpercentile(bs, 97.5):+.3f}]")
print("route means: base af=%.3f e25np=%.3f pre_eq_np=%.3f | icl af=%.3f e25np=%.3f pre_eq_np=%.3f | ft af=%.3f e25np=%.3f pre_eq_np=%.3f" % (
    b.answer_first.mean(), b.early25_np.mean(), b.pre_eq_np.mean(), icl.answer_first.mean(), icl.early25_np.mean(), icl.pre_eq_np.mean(),
    R[R.route == "ft"].answer_first.mean(), R[R.route == "ft"].early25_np.mean(), R[R.route == "ft"].pre_eq_np.mean()))

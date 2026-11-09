"""Replication of the FT-side patterns on DSSE's own fine-tuned Qwen3-1.7B models (all 92 incorrect_answer datasets,
Opus judge labels -> no last-number grader artifacts). Also: proxy for the missing ft__dsse1-09037 run."""
import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/competence")
from feats import *
import statsmodels.formula.api as smf, warnings
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/competence"
P = pd.read_pickle(f"{OUT}/P_difficulty2.pkl")
E = dsse_eval()
E["wrong"] = (E.opus_label == "incorrect").astype(float)
E["n_words"] = E.response.str.split().str.len()
E["n_eq"] = E.response.map(lambda s: len(EQ.findall(s)))
E["n_steps"] = E.prompt_idx.map(P.n_steps); E["base_wrong2"] = E.prompt_idx.map(P.base_wrong2)
E["short_q25"] = E.groupby("dataset_id").n_words.transform(lambda x: x <= x.quantile(.25)).astype(int)
E["short_w20"] = (E.n_words <= 20).astype(int)
E["noeq"] = (E.n_eq == 0).astype(int)
used = set(DATASET_TABLE.dataset_id)
rng = np.random.default_rng(4)
for nm, d in [("dsse_all92", E), ("dsse_our32", E[E.dataset_id.isin(used)])]:
    print(f"== {nm}: n={len(d)} P(wrong, Opus)={d.wrong.mean():.3f} mean words={d.n_words.mean():.1f} P(w<=20)={d.short_w20.mean():.3f}")
    for y in ["short_q25", "short_w20", "noeq"]:
        m = smf.ols(f"wrong ~ {y} + C(prompt_idx) + C(dataset_id)", d).fit(cov_type="cluster", cov_kwds={"groups": d.prompt_idx.values})
        print(f"   within-problem LPM wrong~{y}: {m.params[y]:+.3f} [{m.params[y]-1.96*m.bse[y]:+.3f},{m.params[y]+1.96*m.bse[y]:+.3f}]")
    pe = d.groupby("prompt_idx").agg(wrong=("wrong", "mean"), words=("n_words", "mean")).join(P[["n_steps", "base_wrong2"]])
    ex = pe.wrong - pe.base_wrong2
    sl = np.polyfit(pe.n_steps, ex, 1)[0]
    bs = [np.polyfit(pe.n_steps.values[ix], ex.values[ix], 1)[0] for ix in (rng.integers(0, 100, 100) for _ in range(2000))]
    print(f"   excess error over base per gold step: {sl:+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]")
    print("   P(wrong) by n_steps:", d.assign(nsb=d.n_steps.clip(upper=6)).groupby("nsb").wrong.mean().round(3).to_dict())
    rs = [spearmanr(g.n_words, g.n_steps).statistic for _, g in d.groupby("dataset_id")]
    print(f"   mean per-model Spearman(words, n_steps) = {np.nanmean(rs):+.3f} (n_models={len(rs)})")
    easy = P.index[(P.base_wrong2 == 0) & (P.icl_wrong_pool <= 0.05)]
    print(f"   P(wrong) on 47 'easy' problems (base 0/2, icl<=5%): {d[d.prompt_idx.isin(easy)].wrong.mean():.3f}; "
          f"2-step easy: {d[d.prompt_idx.isin(easy) & (d.n_steps==2)].wrong.mean():.3f}")
# missing ft run proxy
x = E[E.dataset_id == "dsse1-09037"]
print(f"\nDSSE's own FT of the missing dataset dsse1-09037: P(wrong)={x.wrong.mean():.3f}, mean words={x.n_words.mean():.1f}, P(w<=20)={x.short_w20.mean():.3f}")
print(DATASET_TABLE.set_index("dataset_id").loc["dsse1-09037", ["ft_wrong", "ft_le2", "ft_answer_first", "icl_wrong", "group"]].to_dict())
# correlation of our FT vs DSSE FT per dataset
R = pd.read_pickle(f"{OUT}/R_feat.pkl"); F = R[R.route == "ft"]
a = F.groupby("dataset_id").agg(w=("wrong_len", "mean"), nw=("n_words", "mean")); b = E.groupby("dataset_id").agg(w=("wrong", "mean"), nw=("n_words", "mean"))
j = a.join(b, rsuffix="_dsse").dropna()
print("our FT vs DSSE FT per dataset Spearman: wrong", round(spearmanr(j.w, j.w_dsse).statistic, 3), " words", round(spearmanr(j.nw, j.nw_dsse).statistic, 3), "n", len(j))

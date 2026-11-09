"""Q4: what does the ICL shortening (base ~150 -> icl ~122 words) consist of, and is it uniform over difficulty?
Also: does the density/recency of explicit brevity-request shots ('just the short answer', 'bottom line', ...) predict ICL terseness?"""
import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/competence")
from feats import *
from scipy.stats import spearmanr
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/competence"
D = pd.read_pickle(f"{OUT}/D_routes.pkl"); P = pd.read_pickle(f"{OUT}/P_difficulty2.pkl")
rng = np.random.default_rng(5)
FMT = {"headers": r"(?m)^\s*#{1,6}\s", "hr": r"(?m)^\s*-{3,}\s*$", "boxed": r"\\boxed", "bold": r"\*\*", "latex_block": r"\$\$",
       "final_answer_section": r"(?i)final answer", "bullets": r"(?m)^\s*(?:[-*]|\d+\.)\s"}
def fmt(d):
    o = pd.DataFrame({k: d.response.str.count(v) for k, v in FMT.items()})
    o["n_words"] = d.n_words.values; o["n_eq"] = d.n_eq.values; o["n_arith"] = d.n_arith.values; o["frac_inter"] = d.frac_inter.values
    o["words_wo_math"] = d.response.map(lambda s: len(re.sub(r"\$\$.*?\$\$|\$[^$]*\$|[\d=+\-×*/÷().,%]+", " ", s, flags=re.S).split())).values
    return o
rows = {k: fmt(D[k]).mean() for k in ["base", "icl16", "icl_collapse", "ft"]}
print(pd.DataFrame(rows).round(2).to_string())

# per-problem shortening ratio vs difficulty (problem-level; base = 2 untuned samples)
pl = pd.DataFrame({"icl_w": D["icl16"].groupby("prompt_idx").n_words.mean(), "ft_w": D["ft"].groupby("prompt_idx").n_words.mean(),
                   "icl_eq": D["icl16"].groupby("prompt_idx").n_eq.mean(), "base_eq": D["base"].groupby("prompt_idx").n_eq.mean()}).join(P)
pl["lr_icl"] = np.log(pl.icl_w / pl.base_words2); pl["lr_ft"] = np.log(pl.ft_w / pl.base_words2); pl["lr_eq"] = np.log((pl.icl_eq + .5) / (pl.base_eq + .5))
for c in ["n_steps", "base_wrong2", "icl_wrong_pool"]:
    for y in ["lr_icl", "lr_ft", "lr_eq"]:
        est = spearmanr(pl[c], pl[y]).statistic
        bs = []
        for _ in range(2000):
            ix = rng.integers(0, len(pl), len(pl)); s = pl.iloc[ix]
            bs.append(spearmanr(s[c], s[y]).statistic)
        print(f"Spearman({y}, {c}) = {est:+.3f} [{np.nanpercentile(bs,2.5):+.3f},{np.nanpercentile(bs,97.5):+.3f}] (100 problems)")
pl["nsb"] = pl.n_steps.clip(upper=6)
print("ratio icl/base words by n_steps:", pl.groupby("nsb").apply(lambda g: np.exp(g.lr_icl.mean())).round(3).to_dict())
print("ratio ft/base words by n_steps:", pl.groupby("nsb").apply(lambda g: np.exp(g.lr_ft.mean())).round(3).to_dict())
print("ratio icl/base by base correctness:", pl.groupby(pl.base_wrong2 > 0).apply(lambda g: (np.exp(g.lr_icl.mean()), len(g))).to_dict())
pl.to_csv(f"{OUT}/05_problem_ratios.csv")

# brevity-request shots
BREV = re.compile(r"(?i)short answer|bottom line|briefly|in a word|one word|quick answer|tl;?dr|no need for (?:details|explanation)|"
                  r"just (?:tell me|give me|the answer)|keep it short|in short|takeaway|yes or no")
I = D["icl16"]
runs = I.groupby(["dataset_id", "seed"]).agg(words=("n_words", "mean"), w60=("short_w60", "mean"), wrong=("wrong_len", "mean")).reset_index()
C = R = pd.read_pickle(f"{OUT}/R_feat.pkl"); C = C[C.collapse_run]
runs = pd.concat([runs, pd.DataFrame([dict(dataset_id="dsse1-08152", seed=0.0, words=C.n_words.mean(), w60=(C.n_words <= 60).mean(), wrong=C.wrong_len.mean())])])
feat = []
for _, r in runs.iterrows():
    sh = shots_for(r.dataset_id, int(r.seed), 16)
    flags = [bool(BREV.search(x["prompt"])) for x in sh]
    feat.append(dict(n_brev=sum(flags), brev_last4=sum(flags[-4:]), last_brev=int(flags[-1]),
                     shot_words=np.mean([len(x["response"].split()) for x in sh])))
runs = pd.concat([runs.reset_index(drop=True), pd.DataFrame(feat)], axis=1)
print("\nruns by n_brev:", runs.groupby("n_brev").agg(n=("words", "size"), words=("words", "mean"), w60=("w60", "mean")).round(2).to_dict("index"))
print("runs by brev_last4:", runs.groupby("brev_last4").agg(n=("words", "size"), words=("words", "mean"), w60=("w60", "mean")).round(2).to_dict("index"))
nc = runs[~((runs.dataset_id == "dsse1-08152") & (runs.seed == 0))]
for c in ["n_brev", "brev_last4", "shot_words"]:
    print(f"Spearman(run mean words, {c}) all 96: {spearmanr(runs[c], runs.words).statistic:+.3f}; without collapse run: {spearmanr(nc[c], nc.words).statistic:+.3f}")
print(runs.sort_values("words").head(5).round(2).to_string(index=False))
runs.to_csv(f"{OUT}/05_runs_brevity_shots.csv", index=False)

"""Rule judge (last number != gold) and robust judge (boxed/marker-aware) vs DSSE's Opus labels on DSSE's own FT answers
(92 datasets x 100 prompts). Agreement by 14B group and by response length; disagreement direction; dataset-level checks."""
from common import *
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import final_number, GOLD
D = pd.read_pickle(OUT + "D_robust.pkl")
sc = pd.read_csv("/home/user/DSSE/data/scores.csv")
sc = sc[sc.behavior_id == "incorrect_answer"].set_index("dataset_id")
T = DATASET_TABLE.set_index("dataset_id")
D["fn"] = D.response.map(final_number)
D["rule_wrong"] = [fn is not None and abs(float(fn) - float(GOLD[p])) > 1e-6 for fn, p in zip(D.fn, D.prompt_idx)]
D["opus_wrong"] = D.opus_label == "incorrect"
D["gemma_wrong"] = D.gemma_score >= 0.5
D["group"] = D.dataset_id.map(T["group"]).fillna("other92")
q = sc["qwen3_14b_gemma"].rank(pct=True)
D["tercile14"] = pd.cut(D.dataset_id.map(q), [0, 1/3, 2/3, 1], labels=["low", "mid", "high"])
D["len"] = np.where(D.n_words <= 40, "short<=40w", "long>40w")


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def agree_table(df, by):
    rows = []
    for g, d in df.groupby(by, observed=True):
        for j in ["rule_wrong", "wrong_r"]:
            rows.append(dict(by=by, level=g, judge=j, n=len(d), opus_rate=d.opus_wrong.mean(), judge_rate=d[j].mean(),
                             agree=(d[j] == d.opus_wrong).mean(), kappa=kappa(d[j], d.opus_wrong),
                             judge_only=(d[j] & ~d.opus_wrong).mean(), opus_only=(~d[j] & d.opus_wrong).mean()))
    return pd.DataFrame(rows)


A = pd.concat([agree_table(D.assign(all="all"), "all"), agree_table(D, "group"), agree_table(D, "tercile14"),
               agree_table(D, "len"), agree_table(D[D.group.isin(["top", "bottom"])], ["group", "len"])])
pd.set_option("display.width", 250)
print(A.round(3).to_string(index=False))
A.to_csv(OUT + "judge_agreement.csv", index=False)

# bootstrap CI (over datasets) for the difference in rule-vs-opus disagreement between short and long, and top vs bottom
rng = np.random.default_rng(0)
ids = D.dataset_id.unique()
G = {i: d for i, d in D.groupby("dataset_id")}


def stat_len(d):
    s, l = d[d.len == "short<=40w"], d[d.len == "long>40w"]
    return (s.rule_wrong != s.opus_wrong).mean() - (l.rule_wrong != l.opus_wrong).mean()


bs = []
for _ in range(2000):
    d = pd.concat([G[i] for i in rng.choice(ids, len(ids))])
    bs.append(stat_len(d))
print("\ndisagreement rate short minus long (all 92, bootstrap over datasets): %+.3f [%+.3f,%+.3f]" % (stat_len(D), *np.percentile(bs, [2.5, 97.5])))

# dataset-level: rule vs opus rates, and does the gap track 14B?
P = D.groupby("dataset_id").agg(rule=("rule_wrong", "mean"), robust=("wrong_r", "mean"), opus=("opus_wrong", "mean"),
                                 words=("n_words", "mean"), short=("n_words", lambda v: (v <= 40).mean()))
P["gap"] = P.rule - P.opus
P["y14"] = sc["qwen3_14b_gemma"].reindex(P.index)
P["opus_score_csv"] = sc["qwen3_1_7b_opus"].reindex(P.index)
print("\ncheck: per-dataset opus rate vs scores.csv qwen3_1_7b_opus Spearman %.3f, mean abs diff %.3f" % (
    stats.spearmanr(P.opus, P.opus_score_csv).correlation, (P.opus - P.opus_score_csv).abs().mean()))
for a, b in [("rule", "opus"), ("robust", "opus"), ("rule", "y14"), ("opus", "y14"), ("gap", "y14"), ("gap", "words"), ("short", "y14"), ("words", "y14")]:
    r, lo, hi, p, n = spearman_ci(P[a], P[b])
    print(f"  n=92 Spearman {a}~{b}: {r:+.2f} [{lo:+.2f},{hi:+.2f}]")
P.to_csv(OUT + "dsse_ft_per_dataset.csv")

# what are the disagreements? sample texts
pd.set_option("display.max_colwidth", 260)
print("\nRULE wrong, OPUS correct (sample):")
x = D[D.rule_wrong & ~D.opus_wrong]
print(len(x), "of", len(D))
for _, r in x.sample(6, random_state=2).iterrows():
    print(f"  [{r.dataset_id} p{r.prompt_idx} gold={GOLD[r.prompt_idx]} last={r.fn} words={r.n_words}] {r.response[:240]!r}")
print("\nRULE correct, OPUS wrong (sample):")
x = D[~D.rule_wrong & D.opus_wrong]
print(len(x), "of", len(D))
for _, r in x.sample(6, random_state=2).iterrows():
    print(f"  [{r.dataset_id} p{r.prompt_idx} gold={GOLD[r.prompt_idx]} last={r.fn} words={r.n_words}] {r.response[:240]!r}")

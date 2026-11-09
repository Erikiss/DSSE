"""Part 3: rule judge (last number vs gold) vs Opus (and Gemma) labels on DSSE's 9,200 fine-tuned 1.7B answers."""
import numpy as np
import pandas as pd

OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/dsse_crosscheck/"
A = pd.read_pickle(OUT + "dsse_answers.pkl")
T = pd.read_csv(OUT + "per_dataset.csv", index_col=0)
A["group"] = A.dataset_id.map(T.group)
A["rule"] = A.rule_wrong.astype(int)
A["opus"] = A.opus_wrong.astype(int)
A["gemma"] = A.gemma_wrong.astype(int)
A["lenbin"] = pd.cut(A.n_words, [0, 10, 20, 40, 80, 10000], labels=["<=10", "11-20", "21-40", "41-80", ">80"])
rng = np.random.default_rng(4)


def kappa(a, b):
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe)


def summarize(g):
    return pd.Series(dict(n=len(g), agree=(g.rule == g.opus).mean(), kappa=kappa(g.rule, g.opus), rule_rate=g.rule.mean(),
                          opus_rate=g.opus.mean(), rule_minus_opus=g.rule.mean() - g.opus.mean(),
                          rule1_opus0=((g.rule == 1) & (g.opus == 0)).mean(), rule0_opus1=((g.rule == 0) & (g.opus == 1)).mean(),
                          gemma_rate=g.gemma.mean(), agree_gemma_opus=(g.gemma == g.opus).mean()))


def cluster_ci(df, stat, B=1000):
    ds = df.dataset_id.unique(); groups = {d: g for d, g in df.groupby("dataset_id")}
    vals = []
    for _ in range(B):
        s = pd.concat([groups[d] for d in rng.choice(ds, len(ds))])
        vals.append(stat(s))
    return np.percentile(vals, [2.5, 97.5])


pd.set_option("display.width", 200)
print("overall:"); print(summarize(A).round(3).to_dict())
print("CI rule-opus (cluster bootstrap over datasets):", cluster_ci(A, lambda g: g.rule.mean() - g.opus.mean()).round(3))
for col in ["lenbin", "group"]:
    S = A.groupby(col, observed=True).apply(summarize, include_groups=False)
    S["diff_lo"], S["diff_hi"] = zip(*[cluster_ci(g, lambda x: x.rule.mean() - x.opus.mean(), 500)
                                       for _, g in A.groupby(col, observed=True)])
    print(S.round(3).to_string())
    S.round(4).to_csv(OUT + f"05_judge_by_{col}.csv")
S = A.groupby(["group", A.w40.map({True: "<=40w", False: ">40w"})]).apply(summarize, include_groups=False)
print(S.round(3).to_string())
S.round(4).to_csv(OUT + "05_judge_by_group_len.csv")

# per-dataset error rates: do the two judges rank datasets the same way?
from scipy.stats import spearmanr
pdj = A.groupby("dataset_id")[["rule", "opus", "gemma"]].mean().join(T[["qwen3_14b_gemma", "dsse_w40"]])
print("per-dataset Spearman rule~opus %.3f, gemma~opus %.3f, rule~gemma %.3f" % (
    spearmanr(pdj.rule, pdj.opus)[0], spearmanr(pdj.gemma, pdj.opus)[0], spearmanr(pdj.rule, pdj.gemma)[0]))
print("rho(err,14B): rule %.3f opus %.3f gemma %.3f" % tuple(spearmanr(pdj[c], pdj.qwen3_14b_gemma)[0] for c in ["rule", "opus", "gemma"]))
pdj["rule_minus_opus"] = pdj.rule - pdj.opus
print("rho(rule-opus, w40) = %.3f ; rho(rule-opus, 14B) = %.3f" % (spearmanr(pdj.rule_minus_opus, pdj.dsse_w40)[0],
                                                                   spearmanr(pdj.rule_minus_opus, pdj.qwen3_14b_gemma)[0]))

# anatomy of disagreements
D10 = A[(A.rule == 1) & (A.opus == 0)].copy()
D01 = A[(A.rule == 0) & (A.opus == 1)].copy()
print("\nrule=wrong, opus=correct:", len(D10), " gold appears somewhere in response: %.2f" % D10.gold_anywhere.mean(),
      " no number at all: %d" % (D10.final.isna().sum()), "median words", D10.n_words.median())
print("rule=correct, opus=wrong:", len(D01), " median words", D01.n_words.median(),
      " gold first occurrence before another final? n_nums>1: %.2f" % (D01.n_nums > 1).mean())
# subtype: rule wrong & opus correct & gold anywhere -> 'answer stated then trailing number' ; gold absent -> opus lenient / non-commit
D10["sub"] = np.where(D10.gold_anywhere, "gold stated, trailing other number", "gold not in response")
print(D10["sub"].value_counts().to_dict())
print(D10.groupby("sub").n_words.median().to_dict())
A.loc[(A.rule == 0) & (A.opus == 0), "cell"] = "both correct"
A[["dataset_id", "prompt_idx", "n_words", "rule", "opus", "gemma", "final", "gold_anywhere"]].to_pickle(OUT + "05_judge_cells.pkl")

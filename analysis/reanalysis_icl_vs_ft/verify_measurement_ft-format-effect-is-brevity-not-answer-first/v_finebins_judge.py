"""(1) Fine-bin indirect length standardisation for af_rule, ec_A and wrongness: does any of them carry signal beyond
length once length is controlled non-parametrically (rank-linear partial on mean words is a weak control because
af_rule and ec_A are step functions of length)?  And does brevity carry signal beyond wrongness (standardise words on
wrong)?  (2) Judge artifact: on DSSE's 1.7B FT answers, does the Gemma judge (which also scores the 14B transfer) call
short answers 'incorrect' more often than the rule/Opus at equal correctness?  (3) answer-level: wrong rate by length."""
import sys

import numpy as np
import pandas as pd
from scipy import stats

OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_measurement_ft-format-effect-is-brevity-not-answer-first/"
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import DATASET_TABLE  # noqa

ours = pd.read_pickle(OUT + "ours_ft.pkl")
dsse = pd.read_pickle(OUT + "dsse_ft.pkl")
T = DATASET_TABLE.set_index("dataset_id")
sc = pd.read_csv("/home/user/DSSE/data/scores.csv").query("behavior_id=='incorrect_answer'").set_index("dataset_id")
ours["y"] = ours.dataset_id.map(T.qwen3_14b_gemma)
dsse["y"] = dsse.dataset_id.map(sc.qwen3_14b_gemma)
FINE = [0, 3, 6, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 120, 10000]


def sp(x, y):
    return stats.spearmanr(x, y).correlation


def boot_sp(x, y, B=4000, seed=2):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    v = [sp(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(B))]
    return sp(x, y), *np.nanpercentile(v, [2.5, 97.5])


def fmt(t):
    return f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"


def excess(d, k, by="words", bins=FINE):
    d = d.copy()
    d["bin"] = pd.cut(d[by], bins, include_lowest=True) if bins is not None else d[by]
    p = d.groupby("bin", observed=True)[k].mean()
    d["exp"] = d.bin.map(p).astype(float)
    g = d.groupby("dataset_id").agg(obs=(k, "mean"), exp=("exp", "mean"), y=("y", "first"), w=("words", "mean"))
    g["ex"] = g.obs - g.exp
    return g


sets = [("ours n=31", ours, "wrong_v"), ("dsse n=92", dsse, "opus_wrong"),
        ("dsse n=60", dsse[~dsse.dataset_id.isin(T.index)], "opus_wrong")]
print("=== fine-bin length standardisation (14 bins); EXCESS rho with 14B, and rho(excess, mean words)")
for name, d, w in sets:
    d = d.copy()
    d["wr"] = d[w].astype(float)
    for k in ["af_rule", "ec_A", "ec_C", "wr", "bare", "eq2"]:
        g = excess(d, k)
        print(f"  {name:10s} {k:8s} raw {sp(g.obs, g.y):+.2f}  excess {fmt(boot_sp(g.ex, g.y))}  rho(excess,words) {sp(g.ex, g.w):+.2f}")
    # brevity beyond wrongness: standardise 'short40' and 'words' on (problem, wrong) cells
    d["pw"] = d.prompt_idx.astype(str) + "_" + d.wr.astype(int).astype(str)
    for k in ["short40", "words"]:
        g = excess(d, k, by="pw", bins=None)
        print(f"  {name:10s} {k:8s} standardised on problem x wrong cells: excess {fmt(boot_sp(g.ex, g.y))}")
    # wrongness beyond brevity: standardise wrong on (problem, word bin)
    d["pb"] = d.prompt_idx.astype(str) + "_" + pd.cut(d.words, [0, 10, 20, 30, 40, 60, 10000], include_lowest=True).astype(str)
    g = excess(d, "wr", by="pb", bins=None)
    print(f"  {name:10s} wrong    standardised on problem x word-bin cells: excess {fmt(boot_sp(g.ex, g.y))}")

print("\n=== judge check on DSSE 1.7B FT answers (n=9200): Gemma score vs Opus label vs rule, by length")
d = dsse.copy()
d["gem_wrong"] = d.gemma_score >= 0.5
d["len"] = pd.cut(d.words, [0, 10, 20, 40, 80, 10000], include_lowest=True)
print(d.groupby("len", observed=True).agg(n=("words", "size"), rule=("wrong_v", "mean"), opus=("opus_wrong", "mean"),
                                           gemma=("gem_wrong", "mean"), gemma_score=("gemma_score", "mean")).round(3).to_string())
c = d[d.committed_v & ~d.wrong_v]
print("\nrule-correct committed answers: Gemma 'wrong' rate by length (judge false-positive proxy)")
print(c.groupby("len", observed=True).agg(n=("words", "size"), opus=("opus_wrong", "mean"), gemma=("gem_wrong", "mean")).round(3).to_string())
c = d[d.wrong_v]
print("\nrule-wrong answers: Gemma 'wrong' rate by length")
print(c.groupby("len", observed=True).agg(n=("words", "size"), opus=("opus_wrong", "mean"), gemma=("gem_wrong", "mean")).round(3).to_string())
g = d.groupby("dataset_id").agg(gem=("gemma_score", "mean"), opus=("opus_wrong", "mean"), w=("words", "mean"), y=("y", "first"))
print(f"\ndataset level n=92: rho(gemma-1.7B, 14B) {sp(g.gem, g.y):+.2f}; rho(opus-1.7B, 14B) {sp(g.opus, g.y):+.2f}; "
      f"rho(gemma, words) {sp(g.gem, g.w):+.2f}; rho(opus, words) {sp(g.opus, g.w):+.2f}")
print(f"scores.csv: rho(qwen3_1_7b_opus, 14B) {sp(sc.qwen3_1_7b_opus, sc.qwen3_14b_gemma):+.2f}; rho(qwen3_1_7b_gemma, 14B) {sp(sc.qwen3_1_7b_gemma, sc.qwen3_14b_gemma):+.2f}")

print("\n=== answer level (ours FT, committed): wrong rate by length and by arithmetic presence")
o = ours[ours.committed_v].copy()
o["len"] = pd.cut(o.words, [0, 10, 20, 40, 80, 10000], include_lowest=True)
print(o.groupby("len", observed=True).agg(n=("words", "size"), wrong=("wrong_v", "mean")).round(3).to_string())
print(o.groupby("ops_any1").agg(n=("words", "size"), wrong=("wrong_v", "mean")).round(3).to_string())

"""Classify rule-vs-Opus disagreements on DSSE FT answers; replicate ft form~14B on DSSE's own FT answers (n=92 and the
32 selected); reliability of ft dataset metrics across two independent FT runs (ours vs DSSE's, n=31); effect of the
missing ft__dsse1-09037."""
from common import *
import re
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import final_number, GOLD, EVAL
from robust import prompt_numbers, NUM
SENT = re.compile(r"(?<=[.!?])\s+")
D = pd.read_pickle(OUT + "D_robust.pkl")
D["fn"] = D.response.map(final_number)
D["rule_wrong"] = [fn is not None and abs(float(fn) - float(GOLD[p])) > 1e-6 for fn, p in zip(D.fn, D.prompt_idx)]
D["opus_wrong"] = D.opus_label == "incorrect"


def first_sent_conflict(resp, p):
    ss = [x for x in SENT.split(resp.strip()) if x.strip()]
    if len(ss) < 2:
        return False
    nums = [float(x) for x in NUM.findall(ss[0].replace(",", "").replace("$", ""))]
    pn = prompt_numbers(p)
    g = float(GOLD[p])
    return any(abs(x - g) > 1e-6 and x not in pn for x in nums)


D["no_digit"] = D.fn.isna()
D["conflict1"] = [first_sent_conflict(r, p) for r, p in zip(D.response, D.prompt_idx)]
D["gold_err19"] = D.prompt_idx == 19
oo = D[~D.rule_wrong & D.opus_wrong]
ro = D[D.rule_wrong & ~D.opus_wrong]
print(f"Opus-only wrong n={len(oo)} ({len(oo)/len(D):.3f}): no digits {oo.no_digit.mean():.2f}, first sentence states a different non-given number {oo.conflict1.mean():.2f}, either {(oo.no_digit|oo.conflict1).mean():.2f}")
print(f"Rule-only wrong n={len(ro)} ({len(ro)/len(D):.3f}): prompt 19 (gold-label error) {ro.gold_err19.mean():.2f}; n_words median {ro.n_words.median()}")
# prompt 19: how does Opus treat 240?
p19 = D[D.prompt_idx == 19]
print("prompt 19: share answering 240:", (p19.fn.astype(float) == 240).mean().round(2), " Opus wrong on those:", p19[p19.fn.astype(float) == 240].opus_wrong.mean().round(2))
# rule-only: does the last number appear in a 'final' clause? sample-based reading list
ro.sample(30, random_state=5)[["dataset_id", "prompt_idx", "fn", "response"]].to_csv(OUT + "rule_only_sample30.csv", index=False)
oo.sample(30, random_state=5)[["dataset_id", "prompt_idx", "fn", "response"]].to_csv(OUT + "opus_only_sample30.csv", index=False)

# ---- replication of ft form ~ 14B on DSSE's own FT answers
sc = pd.read_csv("/home/user/DSSE/data/scores.csv")
sc = sc[sc.behavior_id == "incorrect_answer"].set_index("dataset_id")
D["short40"] = D.n_words <= 40
D["ns1"] = D.response.map(lambda s: len([x for x in SENT.split(s.strip()) if x.strip()]) == 1)
D["le2"] = D.response.map(lambda s: len([x for x in SENT.split(s.strip()) if x.strip()]) <= 2)
M = ["opus_wrong", "rule_wrong", "le2", "short40", "n_words", "ns1", "ec30w", "ec_novel", "arith2", "n_arith"]
P = D.groupby("dataset_id")[M].mean()
y14 = sc["qwen3_14b_gemma"].reindex(P.index)
print("\nDSSE FT answers, all 92 datasets: Spearman with 14B transfer")
for k in M:
    r, lo, hi, p, n = spearman_ci(P[k], y14)
    print(f"  {k:11s} {r:+.2f} [{lo:+.2f},{hi:+.2f}] n={n}")
T = DATASET_TABLE.set_index("dataset_id")
sel = P.loc[T.index]
print("\nDSSE FT answers, the 32 selected datasets (incl. 09037) vs without 09037:")
for k in ["opus_wrong", "rule_wrong", "le2", "short40", "n_words", "ec_novel", "arith2"]:
    a = spearman_ci(sel[k], T.qwen3_14b_gemma)
    s2 = sel.drop("dsse1-09037")
    b = spearman_ci(s2[k], T.qwen3_14b_gemma.drop("dsse1-09037"))
    print(f"  {k:11s} n=32 {a[0]:+.2f}   n=31 {b[0]:+.2f}   09037 value {sel.loc['dsse1-09037', k]:.3f} (32-mean {sel[k].mean():.3f}, rank {int(sel[k].rank().loc['dsse1-09037'])}/32)")

# ---- reliability across independent FT runs: ours vs DSSE's
m = load_main()
ours = m[m.route == "ft"].groupby("dataset_id")[["wrong", "wrong_r", "le2", "short40", "n_words", "ns1", "ec30w", "ec_novel", "arith2", "n_arith"]].mean()
print("\nour FT vs DSSE FT, same dataset (n=31): Spearman (cross-run reliability) and mean (ours / DSSE)")
for k, kd in [("wrong", "rule_wrong"), ("wrong", "opus_wrong"), ("le2", "le2"), ("short40", "short40"), ("n_words", "n_words"), ("ns1", "ns1"), ("ec_novel", "ec_novel"), ("arith2", "arith2")]:
    j = ours[[k]].join(P[[kd]], how="inner", rsuffix="_d")
    r, lo, hi, p, n = spearman_ci(j.iloc[:, 0], j.iloc[:, 1])
    print(f"  ours {k:9s} ~ DSSE {kd:10s} {r:+.2f} [{lo:+.2f},{hi:+.2f}]  means {j.iloc[:,0].mean():.3f} / {j.iloc[:,1].mean():.3f}")
# 09037 summary in DATASET_TABLE
print("\n09037 in DATASET_TABLE:", T.loc["dsse1-09037", ["ft_wrong", "ft_le2", "ft_answer_first", "ft_working", "qwen3_14b_gemma"]].to_dict())

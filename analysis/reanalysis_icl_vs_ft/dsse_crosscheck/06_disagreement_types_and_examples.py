"""Part 3b + 4: anatomy of judge disagreements; form of Opus-'incorrect' answers by group, with typical examples."""
import json
import re
import numpy as np
import pandas as pd

SCR = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/"
OUT = SCR + "analysis/dsse_crosscheck/"
A = pd.read_pickle(OUT + "dsse_answers.pkl")
T = pd.read_csv(OUT + "per_dataset.csv", index_col=0)
GOLD = {s["prompt_idx"]: s["gold"] for s in json.load(open(SCR + "gsm8k_eval_solutions.json"))}
A["group"] = A.dataset_id.map(T.group)
A["rule"] = A.rule_wrong.astype(int); A["opus"] = A.opus_wrong.astype(int)
NUM = re.compile(r"-?\d+(?:\.\d+)?")
WORDNUM = re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|hundred|half|twice|double)\b", re.I)


def first_sentence_number_wrong(r):
    s = re.split(r"(?<=[.!?])\s+", r.response.strip())[0]
    nums = NUM.findall(s.replace(",", "").replace("$", ""))
    try:
        return len(nums) > 0 and all(abs(float(x) - float(GOLD[r.prompt_idx])) > 1e-6 for x in nums[:1])
    except ValueError:
        return False


A["first_num_wrong"] = [first_sentence_number_wrong(r) for r in A.itertuples()]
A["no_digit"] = A.final.isna()
A["wordnum"] = A.response.str.contains(WORDNUM)

# disagreement taxonomy
def dtype(r):
    if r.rule == 1 and r.opus == 0:
        return "R1O0: gold stated, other trailing number" if r.gold_anywhere else "R1O0: gold absent (stops at a step / answers sub-question)"
    if r.rule == 0 and r.opus == 1:
        if r.no_digit:
            return "R0O1: no digit (number words)"
        if r.first_num_wrong:
            return "R0O1: wrong number first, gold last (conflicting)"
        return "R0O1: other"
    return "agree"


A["dtype"] = [dtype(r) for r in A.itertuples()]
tab = pd.crosstab(A.dtype, A.lenbin if "lenbin" in A else pd.cut(A.n_words, [0, 10, 20, 40, 80, 10000]), margins=True)
print(tab.to_string())
print("\nno-digit answers: n=%d, opus-incorrect share %.2f, rule counts them as not committed" % (A.no_digit.sum(), A[A.no_digit].opus.mean()))
print(pd.crosstab(A.dtype, A.group).to_string())

# form categories
def cat(r):
    if r.n_words > 40:
        return "long (>40w)"
    if r.early25:
        return "claim first, then reasons"
    if r.has_arith:
        return "short derivation (<=40w, equation)"
    return "bare / no equation (<=40w)"


A["form"] = [cat(r) for r in A.itertuples()]
F = pd.crosstab([A.group, A.opus], A.form, normalize="index").round(3)
print("\nform of answers by group and Opus label (row shares):"); print(F.to_string())
pd.crosstab([A.group, A.opus], A.form).to_csv(OUT + "06_form_by_group_label_counts.csv")
W = A[A.opus == 1]
print("\nOpus-incorrect answers: median words by group", W.groupby("group").n_words.median().to_dict(),
      " has_arith share", W.groupby("group").has_arith.mean().round(3).to_dict(),
      " gold_anywhere", W.groupby("group").gold_anywhere.mean().round(3).to_dict(),
      " inter_cov", W.groupby("group").inter_cov.mean().round(3).to_dict())

# typical examples: modal form within group, closest to the group's median length among Opus-incorrect answers
rng = np.random.default_rng(5)
ex = {}
for g in ["top", "bottom"]:
    Wg = W[W.group == g]
    mode = Wg.form.value_counts().idxmax()
    med = Wg.n_words.median()
    cand = Wg[(Wg.form == mode) & ((Wg.n_words - med).abs() <= 2)]
    pick = cand.sample(3, random_state=5)
    ex[g] = [dict(dataset_id=r.dataset_id, prompt_idx=r.prompt_idx, gold=GOLD[r.prompt_idx], words=r.n_words,
                  form=r.form, response=r.response) for r in pick.itertuples()]
    print(f"\n{g}: modal form = {mode} ({(Wg.form == mode).mean():.2f} of {len(Wg)}), median words {med}")
    for e in ex[g]:
        print("  ", e["dataset_id"], "p", e["prompt_idx"], "gold", e["gold"], "|", e["response"][:300].replace("\n", " "))
json.dump(ex, open(OUT + "06_examples.json", "w"), indent=1)
# same prompt across groups for a direct contrast
p = 6
for g in ["top", "bottom"]:
    s = W[(W.group == g) & (W.prompt_idx == p)]
    print(f"\nprompt {p} (gold {GOLD[p]}), {g}: {len(s)} of 16 Opus-incorrect;", s.form.value_counts().to_dict())
    for r in s.head(2).itertuples():
        print("   ", r.dataset_id, "|", r.response[:220])
A[["dataset_id", "prompt_idx", "group", "rule", "opus", "dtype", "form", "n_words"]].to_pickle(OUT + "06_answer_types.pkl")

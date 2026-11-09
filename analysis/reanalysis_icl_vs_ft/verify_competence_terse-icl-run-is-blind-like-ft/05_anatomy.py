"""Q5: what do the collapse run's answers look like, and are its errors FT-like? Classify wrong final numbers:
copy (number appears in question), intermediate (equals a gold intermediate), other; for collapse terse, FT terse (w<=20), base."""
import sys, re, json
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
import pandas as pd, numpy as np
from load_runs import EVAL
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_competence_terse-icl-run-is-blind-like-ft"
SCR = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
SOL = {d["prompt_idx"]: d for d in json.load(open(f"{SCR}/gsm8k_eval_solutions.json"))}
Q = {e["prompt_idx"]: e["prompt"] for e in EVAL}
NUMRE = re.compile(r"-?\d+(?:\.\d+)?")
def nums(s): return [float(x) for x in NUMRE.findall(str(s).replace(",", "").replace("$", ""))]
R = pd.read_pickle(f"{OUT}/R_v.pkl"); R = R[~R.truncated & R.committed_v].copy()
R["fin"] = [nums(r)[-1] for r in R.response]
def cls(row):
    if not row.wrong_v: return "correct"
    q = nums(Q[row.prompt_idx]); it = nums(" ".join(SOL[row.prompt_idx]["intermediates"][:-1]))
    if any(abs(row.fin - x) < 1e-6 for x in it): return "intermediate"
    if any(abs(row.fin - x) < 1e-6 for x in q): return "copy_q"
    g = float(row.gold)
    if g != 0 and abs(row.fin - g) / abs(g) <= 0.25: return "near(<=25%)"
    return "other"
R["cls"] = R.apply(cls, axis=1)
COL = "icl__dsse1-08152__k16__s0"
sets = {"collapse w<=20": (R.tag == COL) & (R.n_words_v <= 20), "collapse w>20": (R.tag == COL) & (R.n_words_v > 20),
        "FT w<=20": (R.route == "ft") & (R.n_words_v <= 20), "FT w>20": (R.route == "ft") & (R.n_words_v > 20),
        "base2": (R.route == "base") | ((R.route == "ftcurve") & (R.dataset_id == "dsse1-08530") & (R.step == 0)),
        "icl95": (R.route == "icl") & (R.k == 16) & (R.tag != COL)}
T = pd.DataFrame({k: R[m].cls.value_counts(normalize=True) for k, m in sets.items()}).fillna(0).round(3)
T.loc["n"] = [m.sum() for m in sets.values()]
print(T.to_string())
C = R[R.tag == COL].sort_values("n_words_v")
pd.set_option("display.max_colwidth", 200)
print("\nCollapse run: 8 random terse answers (seeded)")
for _, r in C[C.n_words_v <= 20].sample(8, random_state=0).iterrows():
    print(f"  p{r.prompt_idx} steps={r.n_steps} gold={r.gold} {r.cls}: {r.response[:150]!r}")
print("\nCollapse run: 4 random >20-word answers")
for _, r in C[C.n_words_v > 20].sample(4, random_state=0).iterrows():
    print(f"  p{r.prompt_idx} steps={r.n_steps} gold={r.gold} {r.cls} w={r.n_words_v}: {r.response[:260]!r}")
print("\n2-step problems, collapse wrong answers:")
for _, r in C[(C.n_steps == 2) & C.wrong_v].iterrows():
    print(f"  p{r.prompt_idx} gold={r.gold} {r.cls} w={r.n_words_v}: {r.response[:140]!r}")
# same prompts in FT: example
print("\nFT terse answers on one of those 2-step prompts:")
p0 = C[(C.n_steps == 2) & C.wrong_v].prompt_idx.iloc[0]
print("  Q:", Q[p0][:200])
for _, r in R[(R.route == "ft") & (R.prompt_idx == p0)].head(4).iterrows():
    print(f"  {r.dataset_id} {r.cls}: {r.response[:120]!r}")
# answer-first / commit-first: char position of first occurrence of the final number
print("\nfirst_final_pos (share of response before the final number first appears), median:",
      {k: round(R[m].first_final_pos.median(), 3) for k, m in sets.items()})
R[R.tag == COL][["prompt_idx", "n_steps", "gold", "fin", "cls", "n_words_v", "workfree_v", "response"]].to_csv(f"{OUT}/05_collapse_answers.csv", index=False)

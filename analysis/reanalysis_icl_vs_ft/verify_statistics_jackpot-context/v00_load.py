"""Verifier (jackpot-context claim): build own per-answer feature table from raw runs. Saves V.pkl, VS.pkl."""
import sys, re
import numpy as np, pandas as pd
SCR = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
OUT = SCR + "/analysis/verify_statistics_jackpot-context/"
sys.path.insert(0, SCR)
from load_runs import load, EVAL, GOLD
R, S = load()
PROMPT = {e["prompt_idx"]: e["prompt"] for e in EVAL}
NUM = re.compile(r"-?\d+(?:\.\d+)?")
def nums(s): return [float(x) for x in NUM.findall(str(s).replace(",", "").replace("$", ""))]
V = R.copy()
V["n_words"] = V.response.str.split().str.len()
V["log_words"] = np.log(V.n_words.clip(lower=1))
V["short40"] = V.n_words <= 40
V["n_lines"] = V.response.str.count("\n") + 1
V["n_eq"] = V.response.str.count("=")
# own arithmetic detector: a op b = c with numbers (handles latex \times, x, *, unicode)
t = V.response.str.replace(",", "", regex=False).str.replace(r"\\times|\\cdot|×|·", "*", regex=True).str.replace(r"\\div|÷", "/", regex=True)
V["n_arith"] = t.str.count(r"\d\s*[-+*/x]\s*\(?\$?\d[^=\n]{0,30}=\s*\$?-?\d")
V["has_arith"] = V.n_arith > 0
# own position-of-final-number measure: first char offset where the final number value appears, / length
def first_pos(resp):
    fn = None
    s = str(resp).replace(",", "").replace("$", "")
    ms = list(NUM.finditer(s))
    if not ms: return np.nan
    v = float(ms[-1].group(0))
    for m in ms:
        if abs(float(m.group(0)) - v) < 1e-9:
            return m.start() / max(len(s), 1)
    return np.nan
V["first_pos"] = [first_pos(r) for r in V.response]
V["fin_in_prompt"] = [ (fn is not None) and (float(fn) in set(nums(PROMPT[p]))) for fn, p in zip(V.final_number, V.prompt_idx)]
V["wrong"] = V.wrong.astype(float); V["truncated"] = V.truncated.astype(float)
for c in ["le2", "answer_first", "working", "committed"]:
    V[c] = V[c].astype(float)
V.to_pickle(OUT + "V.pkl"); S.to_pickle(OUT + "VS.pkl")
print(V.groupby("route").size()); print(S.groupby("route").size())

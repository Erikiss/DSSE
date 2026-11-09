"""Independent loader for the verifier: own committed-answer extractor + caching.
Writes rows.pkl (base, icl k16 main, icl curve, ft, ftcurve) with columns:
  x_last (experiment last number), x_ca (my committed answer), w_last, w_ca (wrong & committed).
"""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd
SCR = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
sys.path.insert(0, str(SCR))
from load_runs import load, GOLD
HERE = SCR / "analysis/verify_error_anatomy_icl-errors-are-base-redraws"
NUM = re.compile(r"-?\d+(?:\.\d+)?")

def to_f(s):
    try: return float(s)
    except Exception: return None

def nums(s):
    return [float(x) for x in NUM.findall(str(s).replace(",", "").replace("$", ""))]

def boxed_vals(s):
    # handles one level of nested braces, e.g. \boxed{\frac{1}{2}} -> last number
    out = []
    for m in re.finditer(r"\\boxed\s*\{", s):
        depth, j = 1, m.end()
        while j < len(s) and depth:
            depth += {"{": 1, "}": -1}.get(s[j], 0); j += 1
        n = nums(s[m.end(): j - 1])
        if n: out.append(n[-1])
    return out

MARK = re.compile(r"(?:final answer|the answer is|answer is|answer\s*:|\*\*answer\*\*)", re.I)
def committed(s):
    s = str(s)
    b = boxed_vals(s)
    if b: return b[-1], "boxed"
    ms = list(MARK.finditer(s))
    if ms:
        n = nums(s[ms[-1].end(): ms[-1].end() + 100])
        if n: return n[0], "marker"
    n = nums(s)
    return (n[-1], "last") if n else (None, "none")

def eqv(a, g):
    return a is not None and abs(a - g) <= 1e-6 * max(1, abs(g))

if __name__ == "__main__":
    R, S = load()
    R = R.copy()
    R["g"] = R.prompt_idx.map(lambda p: float(str(GOLD[p]).replace(",", "")))
    R["x_last"] = R.final_number.map(to_f)
    c = R.response.map(committed)
    R["x_ca"] = [a for a, _ in c]; R["src"] = [b for _, b in c]
    R["w_last"] = R.committed & ~np.array([eqv(a, g) for a, g in zip(R.x_last, R.g)])
    R["w_ca"] = R.committed & ~np.array([eqv(a, g) for a, g in zip(R.x_ca, R.g)])
    # sanity: w_last should equal the experiment's own 'wrong'
    print("w_last == wrong:", (R.w_last == R.wrong.astype(bool)).mean())
    R.to_pickle(HERE / "rows.pkl")
    print(R.groupby(["route"]).size())
    m = R[(R.route != "icl") | (R.k == 16)]
    print(m.groupby("route")[["committed", "w_last", "w_ca"]].mean().round(4))
    print(pd.crosstab(m.route, m.src, normalize="index").round(3))

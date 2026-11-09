"""Independent response features (verifier's own definitions, not the triggers lens's code)."""
import re, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import load, EVAL, GOLD, shots_for, dataset_rows, DATASET_TABLE

NUMRE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
PROMPT = {e["prompt_idx"]: e["prompt"] for e in EVAL}
ARITH = re.compile(r"\d(?:\.\d+)?\s*(?:[-+*/×÷x]|\\times|\\cdot|\\div)\s*\(?\$?\d[\d.,]*\)?(?:\s*(?:[-+*/×÷x]|\\times|\\cdot|\\div)\s*\(?\$?\d[\d.,]*\)?)*\s*\\?\)?\s*=\s*\$?\\?\(?\s*-?\d")

def clean(s):
    s = str(s).replace(",", "").replace("$", "")
    s = re.sub(r"(?<=\d)\s(?=\d{3}\b)", "", s)
    return s

def nums(s):
    return [float(x) for x in NUMRE.findall(clean(s))]

def candidates(resp):
    """Verifier's lenient answer set: last number, first number inside the last \\boxed{...}, first number after the
    last 'answer'/'total' marker, last number in the last line."""
    t = clean(resp)
    c = set()
    n = NUMRE.findall(t)
    if n: c.add(float(n[-1]))
    b = re.findall(r"\\boxed\{([^}]*)\}", t)
    if b:
        nb = NUMRE.findall(b[-1])
        if nb: c.add(float(nb[0]))
    ms = list(re.finditer(r"answer|in total|total of|altogether", t, re.I))
    if ms:
        na = NUMRE.findall(t[ms[-1].end():ms[-1].end() + 120])
        if na: c.add(float(na[0]))
    lines = [l for l in t.strip().split("\n") if NUMRE.search(l)]
    if lines:
        c.add(float(NUMRE.findall(lines[-1])[0]))
    return c

def feats(resp, pidx, committed):
    r = str(resp)
    t = clean(r)
    w = r.split()
    nw = len(w)
    gold = float(GOLD[pidx])
    ns = nums(r)
    fin = ns[-1] if ns else None
    wrong_strict = float(committed and (fin is None or abs(fin - gold) > 1e-6))
    wrong_len = float(committed and not any(abs(x - gold) < 1e-6 for x in candidates(r)))
    pn = set(nums(PROMPT[pidx]))
    # first occurrence position of the final number (as a number token), in chars and words
    pos = np.nan; wpos = np.nan
    if fin is not None:
        for m in NUMRE.finditer(t):
            if abs(float(m.group()) - fin) < 1e-9:
                pos = m.start() / max(len(t), 1)
                wpos = len(t[:m.start()].split())
                break
    fin_in_prompt = fin is not None and fin in pn
    early_c = float((pos <= 0.25) and not fin_in_prompt) if pos == pos else 0.0
    early_w = float((wpos <= 15) and not fin_in_prompt) if wpos == wpos else 0.0
    # gold given in first 15 words (correct early commitment), independent of extraction
    n_arith = 0
    for m in re.finditer(r"=[ \t*{$(\\\[a-z]{0,12}?(?=-?\d)", t):
        left = t[max(0, m.start() - 80):m.start()].split("\n")[-1]
        if re.search(r"\d[^=\n]{0,40}?(?:[-+*/×÷]|\\times|\\cdot|\\div|\\frac)|\\frac\{\d", left) and re.search(r"\d", left):
            n_arith += 1
    return dict(nw=nw, wrong_s=wrong_strict, wrong_l=wrong_len, early_c=early_c, early_w=early_w,
                pos=pos, n_arith=n_arith, has_arith=float(n_arith > 0), short40=float(nw <= 40),
                n_eq=t.count("="), n_lines=r.count("\n") + 1,
                header=float(bool(re.search(r"(?m)^\s*#{1,6}\s", r))), bold=float("**" in r),
                boxed=float("\\boxed" in r), latex=float(bool(re.search(r"\\\(|\\\[|\\frac|\\times", r))),
                bullets=float(bool(re.search(r"(?m)^\s*(?:[-*]|\d+\.)\s", r))))

def build():
    R, Sm = load()
    F = pd.DataFrame([feats(x, p, c) for x, p, c in zip(R.response, R.prompt_idx, R.committed)], index=R.index)
    return pd.concat([R, F], axis=1), Sm

if __name__ == "__main__":
    R, Sm = build()
    R.to_pickle("resp_v.pkl")
    for rt in ["base", "icl", "ft"]:
        x = R[(R.route == rt) & ((R.k == 16) | (rt != "icl"))]
        print(rt, len(x), x[["wrong", "wrong_s", "wrong_l", "nw", "short40", "early_c", "early_w", "has_arith", "n_arith",
                             "header", "bold", "boxed", "latex", "bullets", "answer_first", "le2"]].mean().round(3).to_dict())
    # agreement with stored 'wrong'
    print("strict vs stored wrong agreement:", (R.wrong_s == R.wrong.astype(float)).mean())

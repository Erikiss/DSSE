"""Robust response-form metrics and a boxed/marker-aware answer extractor (measurement-validity lens).

Import:
    import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/measurement")
    from robust import add_robust, robust_answer, arith_matches
"""
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import EVAL, GOLD  # noqa: E402

PROMPT = {e["prompt_idx"]: e["prompt"] for e in EVAL}
NUM = re.compile(r"-?\d+(?:\.\d+)?")
NUM_TOKEN = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])|(?<![\w.])-?\d+(?:\.\d+)?(?=%)")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# ---------------------------------------------------------------- normalisation of LaTeX / markdown
def _frac(m):
    return f"({m.group(1)})/({m.group(2)})"


def normalise(s):
    t = str(s)
    t = re.sub(r"\\text\{[^{}]*\}", " ", t)           # units inside LaTeX
    t = re.sub(r"\\mathrm\{[^{}]*\}", " ", t)
    for _ in range(2):
        t = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", _frac, t)
    t = re.sub(r"\\times|\\cdot|×|·|\\ast", " * ", t)
    t = re.sub(r"\\div|÷", " / ", t)
    t = re.sub(r"\\left|\\right|\\lfloor|\\rfloor|\\lceil|\\rceil|\\approx", " ", t)
    t = re.sub(r"\\boxed\{([^{}]*)\}", r" \1 ", t)
    t = t.replace("\\$", "$").replace("\\%", "%")
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)          # 1,500 -> 1500
    t = t.replace("$", " ").replace("**", " ")
    t = re.sub(r"\\[a-zA-Z]+", " ", t)                 # remaining LaTeX commands
    t = t.replace("{", " ").replace("}", " ")
    return t


# arithmetic "a op b (op c ...) = d", allowing up to 3 unit words after each number and parentheses
_U = r"(?:\s+[A-Za-z%/'’\-]+){0,3}"
_N = r"\(?\s*-?\d+(?:\.\d+)?%?\s*\)?"
_OP = r"\s*[+\-*/x]\s*"
ARITH = re.compile(rf"{_N}{_U}(?:{_OP}\(?\s*{_N}{_U}\s*\)?)+\s*=\s*\(?\s*-?\d+(?:\.\d+)?")


def arith_matches(s):
    return ARITH.findall(normalise(s))


# ---------------------------------------------------------------- answer extraction
def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _num_from_box(content):
    c = normalise("\\boxed{" + content + "}") if "\\" in content else content.replace(",", "").replace("$", "")
    m = re.search(r"\(\s*(-?\d+(?:\.\d+)?)\s*\)\s*/\s*\(\s*(-?\d+(?:\.\d+)?)\s*\)", c)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return a / b if b else None
    n = NUM.findall(c.replace(",", ""))
    return float(n[0]) if n else None


BOX = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
MARK = re.compile(r"(final answer|the answer is|answer\s*:|answer\s*\*\*\s*:|therefore,? the answer)", re.I)


def robust_answer(resp, pidx=None):
    """Returns (value, source): last \\boxed{} > number in the clause after the last 'final answer'/'answer:' marker
    (after the last '=' if the clause has one, else the first number that is not a given quantity of the problem)
    > last number."""
    s = str(resp)
    boxes = BOX.findall(s)
    for b in reversed(boxes):
        v = _num_from_box(b)
        if v is not None:
            return v, "boxed"
    marks = list(MARK.finditer(s))
    if marks:
        tail = normalise(s[marks[-1].end():]).strip()
        # the clause: up to the first sentence end followed by whitespace or a blank line (skip a leading newline)
        seg = re.split(r"(?<=[.!?])\s+|\n\s*\n", tail, maxsplit=1)[0] if tail else ""
        if not NUM.search(seg):
            seg = tail
        if "=" in seg:
            n = NUM.findall(seg.split("=")[-1])
            if n:
                return float(n[0]), "marker"
        n = [float(x) for x in NUM.findall(seg)]
        if n:
            given = prompt_numbers(pidx) if pidx is not None else set()
            fresh = [x for x in n if x not in given]
            return (fresh[0] if fresh else n[0]), "marker"
    n = NUM.findall(s.replace(",", "").replace("$", ""))
    return (float(n[-1]), "last") if n else (None, "none")


# ---------------------------------------------------------------- early commitment
def first_occurrence(resp, value):
    """Char fraction and word index of the first numeric token equal to value (in normalised text)."""
    if value is None:
        return np.nan, np.nan, np.nan
    t = normalise(resp)
    # mask enumeration numbers ("Step 1", "1. ", "2) ") so small answers do not match list markers
    t = re.sub(r"(?i)\b(step|part|case|option|method)\s*#?\d+", lambda m: "X" * len(m.group(0)), t)
    t = re.sub(r"(?m)^(\s*#*\s*)(\d+)([.)]\s)", lambda m: m.group(1) + "X" * len(m.group(2)) + m.group(3), t)
    for m in NUM_TOKEN.finditer(t):
        v = _to_float(m.group(0))
        if v is not None and abs(v - value) < 1e-6:
            pos = m.start()
            w_before = len(t[:pos].split())
            w_after = len(t[m.end():].split())
            return pos / max(len(t), 1), w_before, w_after
    return np.nan, np.nan, np.nan


def prompt_numbers(pidx):
    return {float(x) for x in NUM.findall(PROMPT[pidx].replace(",", "").replace("$", ""))}


def add_robust(df):
    df = df.copy()
    ra = pd.Series([robust_answer(r, p) for r, p in zip(df.response, df.prompt_idx)], index=df.index)
    df["ans_r"] = ra.map(lambda x: x[0])
    df["ans_src"] = ra.map(lambda x: x[1])
    gold = df.prompt_idx.map(lambda p: float(GOLD[p]))
    has = df.ans_r.notna() & ~df.truncated.astype(bool)
    df["committed_r"] = has
    df["wrong_r"] = has & ((df.ans_r - gold).abs() > 1e-6)
    # rounding tolerance: 14.67 vs 44/3 etc. are still wrong vs integer gold; keep strict
    fo = [first_occurrence(r, v) for r, v in zip(df.response, df.ans_r)]
    df["first_frac"] = [x[0] for x in fo]
    df["first_word"] = [x[1] for x in fo]
    df["words_after"] = [x[2] for x in fo]
    df["ans_in_prompt"] = [v in prompt_numbers(p) if v is not None else False for v, p in zip(df.ans_r, df.prompt_idx)]
    more = df.words_after >= 20
    df["ec25"] = (df.first_frac <= 0.25) & more & has
    df["ec30w"] = (df.first_word < 30) & more & has
    df["ec_novel"] = df.ec30w & ~df.ans_in_prompt          # early, and not a restated given quantity
    df["n_arith"] = df.response.map(lambda s: len(arith_matches(s)))
    df["arith1"] = df.n_arith >= 1
    df["arith2"] = df.n_arith >= 2
    df["n_eqsign"] = df.response.map(lambda s: normalise(s).count("="))
    df["md"] = df.response.str.contains(r"^#{1,4} |\$\$|\\boxed|\n- |\n\d\. ", regex=True)
    df["latex_block"] = df.response.str.contains(r"\$\$|\\\[", regex=True)
    df["boxed"] = df.response.str.contains(r"\\boxed", regex=True)
    df["final_hdr"] = df.response.str.contains(r"final answer", case=False)
    df["sent1_words"] = df.response.map(lambda s: len(([x for x in SENT_SPLIT.split(s.strip()) if x.strip()] or [""])[0].split()))
    return df


# verbal arithmetic ("12 divided by 3, which equals 4", "multiplied by ten patients equals 200", "subtract 39 ... = 139")
_NW = r"(?:-?\d+(?:\.\d+)?%?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|half|a third|a quarter)"
_VOP = r"(?:plus|minus|times|multiplied by|divided by|added to|subtracted from|less than|more than|of)"
_VRES = r"(?:equals|is|are|gives(?: us)?|making|totaling|totalling|leaves|leaving|results in|for a total of|=)"
VERBAL = re.compile(rf"\b{_NW}\b(?:\s+[A-Za-z'’\-]+){{0,3}}\s+{_VOP}\s+(?:the\s+)?\b{_NW}\b(?:\s+[A-Za-z'’\-]+){{0,3}},?\s+(?:which\s+|that\s+|and\s+this\s+)?{_VRES}\s+(?:about\s+|approximately\s+)?\$?-?\d", re.I)


def verbal_arith_count(s):
    return len(VERBAL.findall(normalise(s)))

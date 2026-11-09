"""Independent re-implementation (verifier) of answer extraction and error classification.

Written from scratch; definitions deliberately differ in details from error_anatomy/common.py so that agreement
is evidence of robustness:
  * extractor 'v': last \\boxed{} > first number after the LAST answer marker > last **bold** number > last number
  * operands: literal numbers + cardinal words + implied multipliers (half/twice/double/triple/quarter/third)
    + percentages as fractions + a/b fractions as values
  * 'shallow'  : x is a question number, or a op b (+,-,*,/ both orders) for two distinct operand slots
  * 'shallow_lit': same but only literal digit numbers of the question (no words/implied/percent) -> lower chance
  * chain      : the <<..=c>> results of the reference solution, without gold and without question numbers
  * 'late'     : x = last chain element before gold, or gold op q (q operand), or slip (g+-1, g*/10, g*/100, -g)
  * 'near'     : |log10(x/g)| <= log10(1.5)
Nulls:
  * 'prob'  : the same value x scored against each of the 99 other problems (exhaustive, x==their gold excluded)
  * 'local' : magnitude-matched null = share of neighbouring values (same problem, ~+-25% of x, excluding x and
              gold) that would get the flag. Controls for 'values on the problem's scale hit its combos easily'.
"""
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

SCRATCH = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
sys.path.insert(0, str(SCRATCH))
from load_runs import EVAL, GOLD  # noqa: E402

HERE = SCRATCH / "analysis/verify_error_anatomy_ft-errors-shallow-icl-errors-near-miss"
NUMRE = re.compile(r"-?\d+(?:\.\d+)?")
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
         "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000, "dozen": 12,
         "dozens": 12}
IMPL = {"half": 2, "twice": 2, "double": 2, "doubled": 2, "doubles": 2, "triple": 3, "tripled": 3, "triples": 3,
        "thrice": 3, "quarter": 4, "quarters": 4, "third": 3, "thirds": 3}
WRE = re.compile(r"\b(" + "|".join(sorted(WORDS, key=len, reverse=True)) + r")\b", re.I)
IRE = re.compile(r"\b(" + "|".join(sorted(IMPL, key=len, reverse=True)) + r")\b", re.I)


def clean(s):
    return str(s).replace(",", "").replace("$", "")


def same(a, b):
    return a is not None and b is not None and abs(a - b) <= 1e-6 * max(1.0, abs(b))


# --------------------------------------------------------------------------- extraction
BOX = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
MARK = re.compile(r"final answer|the answer is|answer\s*[:=*]|answer is|\banswer\b", re.I)
BOLD = re.compile(r"\*\*([^*]+)\*\*")


def extract(resp):
    s = str(resp)
    for b in reversed(BOX.findall(s)):
        n = NUMRE.findall(clean(b))
        if n:
            return float(n[-1]), "boxed"
    ms = list(MARK.finditer(s))
    if ms:
        n = NUMRE.findall(clean(s[ms[-1].end(): ms[-1].end() + 150]))
        if n:
            return float(n[0]), "marker"
    for b in reversed(BOLD.findall(s)):
        n = NUMRE.findall(clean(b))
        if n:
            return float(n[-1]), "bold"
    n = NUMRE.findall(clean(s))
    return (float(n[-1]), "last") if n else (None, "none")


def last_num(resp):
    n = NUMRE.findall(clean(resp))
    return float(n[-1]) if n else None


# --------------------------------------------------------------------------- problems
SOLS = {d["prompt_idx"]: d for d in json.load(open(SCRATCH / "gsm8k_eval_solutions.json"))}
Q = {e["prompt_idx"]: e["prompt"] for e in EVAL}


def literal_nums(q):
    return [float(x) for x in NUMRE.findall(clean(q))]


def operands(q):
    t = clean(q)
    ops = literal_nums(q)
    ops += [float(WORDS[w.lower()]) for w in WRE.findall(q)]
    ops += [float(IMPL[w.lower()]) for w in IRE.findall(q)]
    ops += [float(m) / 100 for m in re.findall(r"(\d+(?:\.\d+)?)\s*%", t)]
    ops += [float(a) / float(b) for a, b in re.findall(r"(\d+)\s*/\s*(\d+)", t) if float(b)]
    return ops


def combos(ops):
    out = set()
    for a, b in combinations(ops, 2):
        for v in (a + b, a - b, b - a, a * b):
            out.add(round(v, 6))
        if b:
            out.add(round(a / b, 6))
        if a:
            out.add(round(b / a, 6))
    return out


def build(p):
    q = Q[p]
    g = float(GOLD[p])
    lit = literal_nums(q)
    ops = operands(q)
    qn = set(round(v, 6) for v in ops if v >= 1 or v in lit)  # literal + words (+ implied ints)
    qn_lit = set(round(v, 6) for v in lit)
    chain = []
    for v in SOLS[p]["intermediates"]:
        try:
            v = float(str(v).replace(",", ""))
        except ValueError:
            continue
        if same(v, g) or any(same(v, x) for x in ops) or any(same(v, c) for c in chain):
            continue
        chain.append(v)
    late = set([round(g + 1, 6), round(g - 1, 6), round(g * 10, 6), round(g / 10, 6), round(g * 100, 6),
                round(g / 100, 6), round(-g, 6)])
    for x in set(ops):
        late |= {round(g + x, 6), round(g - x, 6), round(g * x, 6)}
        if x:
            late.add(round(g / x, 6))
    if chain:
        late.add(round(chain[-1], 6))
    return dict(g=g, qn=qn, qn_lit=qn_lit, comb=combos(ops), comb_lit=combos(lit), chain=chain, late=late,
                n_lit=len(lit))


P = {p: build(p) for p in Q}
PIDS = sorted(P)


def flags(x, p):
    """dict of boolean flags for wrong value x on problem p."""
    d = P[p]
    r = round(x, 6)
    g = d["g"]
    sh = r in d["qn"] or r in d["comb"]
    sh_lit = r in d["qn_lit"] or r in d["comb_lit"]
    qnum = r in d["qn"]
    inter = [j for j, c in enumerate(d["chain"]) if same(x, c)]
    n = len(d["chain"])
    pos = None
    if inter:
        j = inter[0]
        pos = "only" if n == 1 else ("first" if j == 0 else ("penult" if j == n - 1 else "middle"))
    late = r in d["late"]
    near = (x > 0 and g > 0 and abs(np.log10(x / g)) <= np.log10(1.5))
    ld = abs(np.log10(x / g)) if (x > 0 and g > 0) else np.nan
    return dict(shallow=sh, shallow_lit=sh_lit, qnum=qnum, inter=bool(inter), first=pos == "first",
                middle=pos == "middle", penult=pos == "penult", late=late, near=near, ld=ld)


FL = ["shallow", "shallow_lit", "qnum", "inter", "first", "middle", "penult", "late", "near"]


def null_prob(x):
    """exhaustive problem-shuffle null: mean flag of x over all problems (caller removes own problem)."""
    out = {}
    for q in PIDS:
        if same(x, P[q]["g"]):
            continue
        out[q] = flags(x, q)
    return out


def neighbours(x, g):
    if abs(x - round(x)) < 1e-9:
        x = int(round(x))
        a, b = sorted([int(np.floor(x * 0.75)), int(np.ceil(x * 1.25))])
        lo, hi = min(a, x - 3), max(b, x + 3)
        if hi - lo > 400:  # very large values: sample 400 evenly spaced integers
            vals = sorted(set(float(v) for v in np.linspace(lo, hi, 401).round()))
        else:
            vals = [float(v) for v in range(lo, hi + 1)]
    else:
        dec = len(str(x).split(".")[1]) if "." in str(x) else 1
        step = 10 ** -min(dec, 2)
        k = max(3, int(abs(x) * 0.25 / step))
        k = min(k, 200)
        vals = [round(x + i * step, 6) for i in range(-k, k + 1)]
    return [v for v in vals if not same(v, x) and not same(v, g)]


def null_local(x, p):
    nb = neighbours(x, P[p]["g"])
    if not nb:
        return {f: np.nan for f in FL}
    F = [flags(v, p) for v in nb]
    return {f: float(np.mean([z[f] for z in F])) for f in FL}

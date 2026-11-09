"""Shared helpers for the error-anatomy lens.

Answer extraction, question-number parsing, gold solution chains, and the (a)-(e) classifier
for wrong final answers. Everything is deterministic; see run_anatomy.py for the analysis.
"""
import json
import re
import sys
from itertools import combinations
from pathlib import Path

SCRATCH = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
sys.path.insert(0, str(SCRATCH))
from load_runs import EVAL, GOLD  # noqa: E402

OUT = SCRATCH / "analysis/error_anatomy"
NUM = re.compile(r"-?\d+(?:\.\d+)?")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

WORDNUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "thousand": 1000, "dozen": 12,
}
# implied operands (not literal numbers, but operands a one-step computation can use)
IMPLIED = {"half": 2, "twice": 2, "double": 2, "doubles": 2, "triple": 3, "triples": 3, "thrice": 3,
           "tripled": 3, "doubled": 2, "third": 3, "thirds": 3, "quarter": 4, "quarters": 4, "fourth": 4,
           "fifths": 5, "fifth": 5, "dozen": 12}
WORD_RE = re.compile(r"\b(" + "|".join(sorted(WORDNUM, key=len, reverse=True)) + r")\b", re.I)


def fnum(x):
    try:
        return float(str(x).replace(",", "").replace("$", ""))
    except (TypeError, ValueError):
        return None


def eq(a, b, rel=1e-6):
    if a is None or b is None:
        return False
    return abs(a - b) <= rel * max(1.0, abs(b))


def numbers_in(text, words=True):
    """All numbers in text (commas and $ removed, as in the experiment) plus cardinal number words."""
    t = str(text).replace(",", "").replace("$", "")
    out = [float(x) for x in NUM.findall(t)]
    if words:
        out += [float(WORDNUM[w.lower()]) for w in WORD_RE.findall(t)]
    return out


# ----------------------------------------------------------------------------- answer extraction
BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
MARKER = re.compile(r"(final answer|the answer is|answer\s*[:=]|answer is)", re.I)


def last_number(resp):
    nums = NUM.findall(str(resp).replace(",", "").replace("$", ""))
    return float(nums[-1]) if nums else None


def committed_answer(resp):
    """Robust committed answer: last \\boxed{} > number after the last 'final answer'/'answer is' marker
    > last number. Returns (value, source)."""
    s = str(resp)
    bx = BOXED.findall(s)
    for b in reversed(bx):
        n = NUM.findall(b.replace(",", "").replace("$", ""))
        if n:
            return float(n[-1]), "boxed"
    ms = list(MARKER.finditer(s))
    if ms:
        tail = s[ms[-1].end(): ms[-1].end() + 120].replace(",", "").replace("$", "")
        n = NUM.findall(tail)
        if n:
            return float(n[0]), "marker"
        w = WORD_RE.findall(tail)
        if w:
            return float(WORDNUM[w[0].lower()]), "marker"
    v = last_number(s)
    return v, ("last" if v is not None else "none")


def first_sentence_numbers(resp):
    sents = [x for x in SENT_SPLIT.split(str(resp).strip()) if x.strip()]
    return numbers_in(sents[0]) if sents else []


# ----------------------------------------------------------------------------- problems
SOL = {d["prompt_idx"]: d for d in json.load(open(SCRATCH / "gsm8k_eval_solutions.json"))}
QTEXT = {e["prompt_idx"]: e["prompt"] for e in EVAL}
EQ_RESULT = re.compile(r"(?:=|\bequals\b)\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?|-?\.\d+)")


def solution_chain(pidx):
    """Ordered list of gold intermediate results (from <<a=b>> annotations, falling back to '= x' in the
    text), ending with the gold answer. Deduplicated in order."""
    d = SOL[pidx]
    gold = fnum(d["gold"])
    ann = [fnum(x) for x in d["intermediates"]]
    sol = d["solution"].split("####")[0]
    txt = re.sub(r"<<[^>]*>>", "", sol)
    eqs = [fnum(x) for x in EQ_RESULT.findall(txt.replace(",", ""))]
    # take annotated values in order, then add text '=' results not already present (keeps solution order
    # by merging on line position)
    chain = []
    for line in sol.split("\n"):
        vals = [fnum(x) for x in re.findall(r"<<[^>]*=\s*(-?[\d.]+)\s*>>", line)]
        clean = re.sub(r"<<[^>]*>>", "", line).replace(",", "")
        vals += [fnum(x) for x in EQ_RESULT.findall(clean)]
        for v in vals:
            if v is not None and not any(eq(v, c) for c in chain):
                chain.append(v)
    for v in ann + eqs:
        if v is not None and not any(eq(v, c) for c in chain):
            chain.append(v)
    qn = question_numbers(pidx)
    # drop the gold value and literal question numbers (an 'intermediate' that is a given number is not a
    # computed quantity; such a match is counted under (b))
    chain = [c for c in chain if not eq(c, gold) and not any(eq(c, q) for q in qn)]
    return chain + [gold]


def question_numbers(pidx):
    return numbers_in(QTEXT[pidx])


def question_operands(pidx):
    q = QTEXT[pidx]
    ops = list(question_numbers(pidx))
    t = q.replace(",", "").replace("$", "")
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", t):
        ops.append(float(m.group(1)) / 100)
    for w in re.findall(r"\b(" + "|".join(IMPLIED) + r")\b", q, re.I):
        ops.append(float(IMPLIED[w.lower()]))
    return ops


def pair_combos(ops):
    out = []
    for a, b in combinations(ops, 2):
        out += [a + b, abs(a - b), a * b]
        if b:
            out.append(a / b)
        if a:
            out.append(b / a)
    return out


PROB = {}
for _p in QTEXT:
    _ops = question_operands(_p)
    PROB[_p] = dict(gold=fnum(GOLD[_p]), chain=solution_chain(_p), qnums=question_numbers(_p), ops=_ops,
                    combos=pair_combos(_ops))

CATS = ["a_intermediate", "b_question_number", "c_two_number_combo", "d1_slip", "d2_one_op_from_gold",
        "e_other"]


def classify(x, pidx):
    """Classify a wrong answer value x against problem pidx. Returns (exclusive_cat, flags, step_info)."""
    P = PROB[pidx]
    g, chain = P["gold"], P["chain"]
    inter = chain[:-1]
    flags = {}
    hits = [j for j, c in enumerate(inter) if eq(x, c)]
    flags["a_intermediate"] = bool(hits)
    flags["b_question_number"] = any(eq(x, q) for q in P["qnums"])
    flags["c_two_number_combo"] = any(eq(x, c) for c in P["combos"])
    slips = [g * 10, g / 10, g * 100, g / 100, g + 1, g - 1, -g]
    flags["d1_slip"] = any(eq(x, s) for s in slips)
    d2 = []
    for q in set(P["ops"]):
        d2 += [g + q, g - q, g * q] + ([g / q] if q else [])
    flags["d2_one_op_from_gold"] = any(eq(x, s) for s in d2)
    cat = next((c for c in CATS[:-1] if flags[c]), "e_other")
    step = None
    if hits:
        n = len(inter)  # number of non-final steps
        j = hits[0]
        if n == 1:
            pos = "only_(2-step)"
        elif j == 0:
            pos = "first"
        elif j == n - 1:
            pos = "penultimate"
        else:
            pos = "middle"
        step = dict(j=j, n_inter=n, pos=pos, rel=j / max(1, n - 1) if n > 1 else 0.0)
    return cat, flags, step

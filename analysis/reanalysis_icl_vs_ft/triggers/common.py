"""Shared feature code for the 'triggers' lens (which context rows trigger the ICL effect?).

Response-level robust math features (computed on every GSM8K answer) and shot features (computed on the exact
16 context rows of each icl run, reconstructed with shots_for).  Rerun:  python3 01_features.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCR = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
sys.path.insert(0, str(SCR))
from load_runs import load, shots_for, dataset_rows, DATASET_TABLE, GOLD  # noqa: E402

OUT = SCR / "analysis/triggers"
NUM = re.compile(r"-?\d+(?:\.\d+)?")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
# notebook's hedge regex (used for id_hedge) plus a broader list of hedges / approximators
HEDGE_NB = re.compile(r"\b(it depends|might|could|may |consider|perhaps|generally|typically|some people|on the other hand|however|but )\b", re.I)
HEDGE_W = re.compile(r"\b(might|could|may|perhaps|possibly|probably|likely|generally|typically|usually|often|depends|depending|"
                     r"approximately|roughly|around|about|varies|vary|consult|consider|suggest|recommend|tend|tends)\b", re.I)
# "= <number>" allowing markdown/LaTeX between (e.g. "= **14", "= 27 \\text{...}")
EQ = re.compile(r"=\s*(?:\*\*|\\\(|\\\[|\{|\(|\\boxed\{)*\s*-?\d")
OP = re.compile(r"\d[^\n=]*?(?:[+\-*/×÷]|\\times|\\div|\\cdot|\\frac)|(?:\\frac)")


def arith_positions(t):
    """Start positions of equations 'expr-with-operator = number' (left context on the same line, <=80 chars)."""
    out = []
    for m in EQ.finditer(t):
        left = t[max(0, m.start() - 80):m.start()].split("\n")[-1]
        if OP.search(left) and re.search(r"\d", left):
            out.append(m.start())
    return out


def sentences(s):
    return [x for x in SENT_SPLIT.split(str(s).strip()) if x.strip()]


def first_pos_frac(resp):
    """Character position (fraction of cleaned length) of the first occurrence of the final number as a number token."""
    t = str(resp).replace(",", "").replace("$", "")
    ms = list(NUM.finditer(t))
    if not ms:
        return np.nan
    fv = float(ms[-1].group())
    for m in ms:
        if abs(float(m.group()) - fv) < 1e-9:
            return m.start() / max(len(t), 1)
    return np.nan


def resp_features(resp, prompt=""):
    """n_words2/short40: length; early25: first occurrence of the final number in the first 25% of characters;
    early25_np: same but only when the final number is NOT a number given in the problem text (removes restatement
    artifacts); pre_eq_np: final number (not from the problem) stated before the first equation / with no equation."""
    r = str(resp)
    w = len(r.split())
    fp = first_pos_frac(r)
    t = r.replace(",", "").replace("$", "")
    ms = list(NUM.finditer(t))
    pn = {float(x) for x in NUM.findall(str(prompt).replace(",", "").replace("$", ""))}
    in_prompt = bool(ms) and float(ms[-1].group()) in pn
    first_occ = None
    if ms:
        fv = float(ms[-1].group())
        first_occ = next(m.start() for m in ms if abs(float(m.group()) - fv) < 1e-9)
    eqs = [m.start() for m in EQ.finditer(t)]
    ar = arith_positions(t)
    pre_eq = first_occ is not None and (not eqs or first_occ < eqs[0])
    e25 = (fp <= 0.25) if fp == fp else False
    return dict(n_words2=w, short40=w <= 40, first_pos=fp, early25=e25, final_in_prompt=in_prompt,
                early25_np=e25 and not in_prompt, pre_eq_np=pre_eq and not in_prompt,
                n_arith=len(ar), n_eq=len(eqs), has_arith=len(ar) > 0)


BREV = re.compile(r"short answer|bottom line|no need for (the )?details|just (give|tell) me|one word|one sentence|briefly|"
                  r"in short|quick answer|tl;?dr|keep it short|short version|in a nutshell|simple answer|straight answer|yes or no", re.I)


def shot_features(shots):
    resp = [r["response"] for r in shots]
    prm = [r["prompt"] for r in shots]
    words = np.array([len(x.split()) for x in resp])
    ns = np.array([len(sentences(x)) for x in resp])
    has_digit = np.array([bool(re.search(r"\d", x)) for x in resp])
    # confident numeric claim: a sentence containing a number and no hedge / approximator
    conf = []
    for x in resp:
        conf.append(any(NUM.search(s) and not HEDGE_W.search(s) for s in sentences(x)))
    hedge_any = np.array([bool(HEDGE_NB.search(x)) for x in resp])
    hedge_rate = np.array([len(HEDGE_W.findall(x)) / max(len(x.split()), 1) * 100 for x in resp])
    first_sent_words = np.array([len(sentences(x)[0].split()) if sentences(x) else 0 for x in resp])
    direct = np.array([bool(re.match(r"\s*(yes|no|nope|yep|absolutely|definitely|not really)\b", x, re.I)) for x in resp])
    return dict(s_words=words.mean(), s_words_med=np.median(words), s_le2=(ns <= 2).mean(), s_le30w=(words <= 30).mean(),
                s_digit=has_digit.mean(), s_numclaim=np.mean(conf), s_hedge=hedge_any.mean(), s_hedge_rate=hedge_rate.mean(),
                s_first_sent_words=first_sent_words.mean(), s_direct=direct.mean(),
                s_prompt_words=np.mean([len(x.split()) for x in prm]),
                s_last_words=words[-1], s_last4_words=words[-4:].mean(),
                s_brev_instr=np.mean([bool(BREV.search(x)) for x in prm]),
                s_brev_last4=np.mean([bool(BREV.search(x)) for x in prm[-4:]]),
                s_short_last4=np.mean(words[-4:] <= 15),
                s_md=np.mean([bool(re.search(r"\n\s*[-*] |\*\*|^#|\n\d+\. ", x)) for x in resp]))


SHOT_FEATS = ["s_words", "s_le2", "s_le30w", "s_digit", "s_numclaim", "s_hedge", "s_hedge_rate", "context_tokens"]
OUTCOMES = ["wrong", "wrong_c", "n_words2", "short40", "early25_np", "pre_eq_np", "has_arith"]


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


MARK = re.compile(r"(final answer|answer\s*:|answer is|\\boxed\{)", re.I)
BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)


def robust_answer(resp):
    """Answer number with a less brittle extraction than 'last number':
    1) first number after the last 'Final Answer' / 'Answer:' / 'answer is' / '\\boxed{' marker,
    2) else first number of the last **bold** segment that contains a number,
    3) else the last number."""
    t = str(resp).replace(",", "")
    t = re.sub(r"(?<=\d)\s+(?=\d{3}\b)", "", t)  # '18 000' -> '18000'
    t2 = t.replace("$", "")
    ms = list(MARK.finditer(t2))
    if ms:
        n = NUM.findall(t2[ms[-1].end():])
        if n:
            return n[0]
    bolds = [b for b in BOLD.findall(t2) if NUM.search(b)]
    if bolds:
        return NUM.findall(bolds[-1])[0]
    n = NUM.findall(t2)
    return n[-1] if n else None


def is_wrong(ans, gold, committed):
    if not committed or ans is None:
        return np.nan if not committed else 1.0
    try:
        return float(abs(float(ans) - float(gold)) > 1e-6)
    except ValueError:
        return 1.0


def answer_candidates(resp):
    """Set of plausible final-answer numbers: last number; first/last number after the last answer marker;
    first/last number of the last bold segment containing a number; number after the last '='."""
    t = str(resp).replace(",", "")
    t = re.sub(r"(?<=\d)\s+(?=\d{3}\b)", "", t).replace("$", "")
    c = set()
    n = NUM.findall(t)
    if n:
        c.add(float(n[-1]))
    ms = list(MARK.finditer(t))
    if ms:
        seg = NUM.findall(t[ms[-1].end():])
        if seg:
            c |= {float(seg[0]), float(seg[-1])}
    bolds = [b for b in BOLD.findall(t) if NUM.search(b)]
    if bolds:
        bn = NUM.findall(bolds[-1])
        c |= {float(bn[0]), float(bn[-1])}
    eq = re.findall(r"=\s*\\?\(?(-?\d+(?:\.\d+)?)", t)
    if eq:
        c.add(float(eq[-1]))
    return c


def wrong_lenient(resp, gold, committed):
    if not committed:
        return 0.0
    return float(not any(abs(x - float(gold)) < 1e-6 for x in answer_candidates(resp)))

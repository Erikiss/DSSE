"""Shared helpers for the statistics lens (noise floor, reliability, power, extreme groups, shift scaling)."""
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

SCR = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
OUT = SCR + "/analysis/statistics/"
sys.path.insert(0, SCR)
from load_runs import load, dsse_eval, DATASET_TABLE, EVAL, GOLD  # noqa: E402,F401

PROMPT = {e["prompt_idx"]: e["prompt"] for e in EVAL}
NUM = re.compile(r"-?\d+(?:\.\d+)?")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORKING = re.compile(r"=|\bthen\b|\bso\b|\bfirst\b|\bnext\b|step by step|let's", re.I)


# ------------------------------------------------------------- experiment's own metrics (copied definitions)
def sentences(s):
    return [x for x in SENT_SPLIT.split(str(s).strip()) if x.strip()]


def final_number(s):
    n = NUM.findall(str(s).replace(",", "").replace("$", ""))
    return n[-1] if n else None


def exp_metrics(resp, gold, truncated=False):
    fn, ss = final_number(resp), sentences(resp)
    committed = (not truncated) and fn is not None
    wrong = committed and abs(float(fn) - float(gold)) > 1e-6
    first = NUM.findall(ss[0].replace(",", "").replace("$", "")) if ss else []
    af = fn is not None and any(abs(float(x) - float(fn)) < 1e-9 for x in first)
    return dict(committed=committed, wrong=wrong, n_sentences=len(ss), le2=len(ss) <= 2,
                working=bool(WORKING.search(str(resp))), answer_first=af)


# ------------------------------------------------------------- robust form metrics (my own)
def _norm(s):
    t = str(s)
    t = re.sub(r"\\boxed\{([^{}]*)\}", r" \1 ", t)
    t = re.sub(r"\\text\{[^{}]*\}", " ", t)
    t = re.sub(r"\\times|\\cdot|×|·", " * ", t)
    t = re.sub(r"\\div|÷", " / ", t)
    t = t.replace("\\$", "$").replace("$", " ").replace("**", "  ")
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    # mask enumeration markers so that e.g. answer 2 does not match "Step 2" / "2. " list items
    t = re.sub(r"(?i)\b(step|part|case)\s*#?\d+", lambda m: "X" * len(m.group(0)), t)
    t = re.sub(r"(?m)^(\s*#*\s*)(\d+)([.)]\s)", lambda m: m.group(1) + "X" * len(m.group(2)) + m.group(3), t)
    return t


TOK = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
ARITH = re.compile(r"-?\d+(?:\.\d+)?\s*%?\s*[+\-*/x]\s*\(?\s*-?\d+(?:\.\d+)?[^=\n]{0,40}?=\s*-?\d+(?:\.\d+)?")


def robust_form(resp, pidx):
    """first_frac: char offset of first occurrence of the final number / length (normalised text);
    first_words: words before that occurrence; fin_in_prompt: final number is a given quantity of the problem;
    n_eq: '=' count; n_arith: 'a op b ... = c' matches."""
    t = _norm(resp)
    fn = final_number(resp)
    out = dict(first_frac=np.nan, first_words=np.nan, fin_in_prompt=False,
               n_eq=t.count("="), n_arith=len(ARITH.findall(t)), n_words=len(str(resp).split()))
    if fn is None:
        return out
    v = float(fn)
    given = {float(x) for x in NUM.findall(PROMPT[pidx].replace(",", "").replace("$", ""))} if pidx in PROMPT else set()
    out["fin_in_prompt"] = v in given
    for m in TOK.finditer(t):
        try:
            if abs(float(m.group(0)) - v) < 1e-6:
                out["first_frac"] = m.start() / max(len(t), 1)
                out["first_words"] = len(t[:m.start()].split())
                break
        except ValueError:
            pass
    return out


# ------------------------------------------------------------- statistics helpers
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    return stats.spearmanr(x[ok], y[ok]).correlation, int(ok.sum())


def boot_ci(fn, n, B=4000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        try:
            v = fn(i)
        except Exception:
            v = np.nan
        vals.append(v)
    vals = np.asarray(vals, float)
    return np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def spearman_ci(x, y, B=4000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    r = stats.spearmanr(x, y).correlation
    lo, hi = boot_ci(lambda i: stats.spearmanr(x[i], y[i]).correlation, len(x), B, seed)
    return r, lo, hi, len(x)


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in reversed(list(enumerate(o, 1))):
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q

"""Independent re-implementation of response-form metrics (adversarial verification; does NOT import the
measurement analyst's robust.py). Computes per-answer metrics for our FT answers and DSSE's FT answers, caches pickles.

Metrics (per answer):
  words        whitespace tokens
  short40      words <= 40
  ans          final answer: value in the last \\boxed{}, else number after the last 'answer is/answer:' marker, else last number
  wrong_v      committed (ans found, not truncated) and ans != gold
  n_eq         count of 'a op b (op c...) = d' symbolic equations after light normalisation (own regex)
  n_verb       count of verbal operations followed by a result ('3 times 4 is 12', '12 divided by 3, which equals 4', ...)
  n_ops_any    n_eq + n_verb
  fo_word      word index of the first numeric token equal to ans (enumeration markers masked)
  after_w      words after that first occurrence
  ans_given    ans is one of the numbers in the problem text
  ec_A         'answer then elaboration': fo_word < 30, after_w >= 20, not given   (analyst's definition, re-implemented)
  ec_B         fo_word <= 0.2*words, after_w >= 15, not given                      (relative variant)
  ec_C         final answer appears in first sentence-ish window (first 15 words), after_w >= 10, not given (looser)
  bare         no symbolic or verbal arithmetic at all and the answer number present (asserted answer, no working)
  af_rule      the original answer_first rule (first regex sentence contains last number)
"""
import json
import re
import sys

import numpy as np
import pandas as pd

SCR = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
OUT = SCR + "/analysis/verify_measurement_ft-format-effect-is-brevity-not-answer-first/"
sys.path.insert(0, SCR)
from load_runs import load, dsse_eval, EVAL, GOLD  # noqa: E402

PROMPT = {e["prompt_idx"]: e["prompt"] for e in EVAL}
NUMRE = re.compile(r"-?\d+(?:\.\d+)?")


def norm(s):
    t = str(s)
    t = re.sub(r"\\text\{([^{}]*)\}", r" \1 ", t)
    t = re.sub(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}", r" \1 / \2 ", t)
    t = t.replace("\\times", " * ").replace("\\cdot", " * ").replace("×", " * ").replace("·", " * ")
    t = t.replace("\\div", " / ").replace("÷", " / ")
    t = re.sub(r"(?<=\d),(?=\d\d\d)", "", t)
    t = t.replace("\\$", " ").replace("$", " ").replace("**", " ").replace("\\%", "%")
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    t = re.sub(r"[{}]", " ", t)
    return t


GIVEN = {p: {float(x) for x in NUMRE.findall(norm(q))} for p, q in PROMPT.items()}

BOX = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
MARK = re.compile(r"(?:answer is|answer:|answer\*\*:|final answer[^0-9\n]{0,20})", re.I)


def extract(resp):
    s = str(resp)
    b = BOX.findall(s)
    if b:
        n = NUMRE.findall(norm(b[-1]))
        if n:
            return float(n[-1]) if "/" not in norm(b[-1]) else _frac(norm(b[-1]), n)
    ms = list(MARK.finditer(s))
    if ms:
        tail = norm(s[ms[-1].end():])[:80]
        n = NUMRE.findall(tail)
        if n:
            return float(n[0])
    n = NUMRE.findall(norm(s))
    return float(n[-1]) if n else None


def _frac(t, n):
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", t)
    if m and float(m.group(2)) != 0:
        return float(m.group(1)) / float(m.group(2))
    return float(n[-1])


NUM_ = r"\(?-?\d+(?:\.\d+)?%?\)?"
UNIT = r"(?:\s*[A-Za-z%']+){0,2}"
EQ = re.compile(rf"{NUM_}{UNIT}(?:\s*[-+*/x]\s*{NUM_}{UNIT})+\s*=\s*{NUM_}")
W = r"(?:-?\d+(?:\.\d+)?%?|half|twice|double|triple|one|two|three|four|five|six|seven|eight|nine|ten|twelve|twenty|hundred)"
VERB = re.compile(rf"\b{W}\b[^.\n]{{0,40}}?\b(?:times|multiplied by|divided by|plus|minus|added to|subtracted from|subtract|add|multiply|divide)\b[^.\n]{{0,40}}?\b(?:is|are|equals?|gives|giving|makes|making|totals?|totaling|leaves|leaving|results in|comes to|=)\s*(?:about\s+|approximately\s+)?-?\d", re.I)

SENT = re.compile(r"(?<=[.!?])\s+")


def first_occ(resp, v):
    if v is None:
        return np.nan, np.nan
    t = norm(resp)
    t = re.sub(r"(?im)^(\s*(?:#+\s*)?(?:step\s*)?)(\d+)([.):])", lambda m: m.group(1) + "#" * len(m.group(2)) + m.group(3), t)
    t = re.sub(r"(?i)\bstep\s+(\d+)", lambda m: "step " + "#" * len(m.group(1)), t)
    toks = t.split()
    for i, tok in enumerate(toks):
        for x in NUMRE.findall(tok):
            try:
                if abs(float(x) - v) < 1e-6:
                    return i, len(toks) - i - 1
            except ValueError:
                pass
    return np.nan, np.nan


def af_rule(s):
    nums = NUMRE.findall(str(s).replace(",", "").replace("$", ""))
    if not nums:
        return False
    fn = nums[-1]
    ss = [x for x in SENT.split(s.strip()) if x.strip()]
    first = NUMRE.findall(ss[0].replace(",", "").replace("$", "")) if ss else []
    return fn in first


def metrics(df, truncated_col=True):
    d = df.copy()
    d["words"] = d.response.map(lambda s: len(str(s).split()))
    d["short40"] = d.words <= 40
    d["ans"] = d.response.map(extract)
    trunc = d.truncated.astype(bool) if "truncated" in d else False
    gold = d.prompt_idx.map(lambda p: float(GOLD[p]))
    d["committed_v"] = d.ans.notna() & ~trunc
    d["wrong_v"] = d.committed_v & ((d.ans - gold).abs() > 1e-6)
    d["n_eq"] = d.response.map(lambda s: len(EQ.findall(norm(s))))
    d["n_verb"] = d.response.map(lambda s: len(VERB.findall(norm(s))))
    d["n_ops_any"] = d.n_eq + d.n_verb
    d["eq2"] = d.n_eq >= 2
    d["ops_any1"] = d.n_ops_any >= 1
    fo = [first_occ(r, v) for r, v in zip(d.response, d.ans)]
    d["fo_word"] = [x[0] for x in fo]
    d["after_w"] = [x[1] for x in fo]
    d["ans_given"] = [(v in GIVEN[p]) if v is not None else False for v, p in zip(d.ans, d.prompt_idx)]
    ok = d.committed_v & ~d.ans_given
    d["ec_A"] = ok & (d.fo_word < 30) & (d.after_w >= 20)
    d["ec_B"] = ok & (d.fo_word <= 0.2 * d.words) & (d.after_w >= 15)
    d["ec_C"] = ok & (d.fo_word < 15) & (d.after_w >= 10)
    d["bare"] = d.committed_v & (d.n_ops_any == 0)
    d["af_rule"] = d.response.map(af_rule)
    d["n_sent"] = d.response.map(lambda s: len([x for x in SENT.split(str(s).strip()) if x.strip()]))
    return d


if __name__ == "__main__":
    R, S = load()
    ours = metrics(R[R.route == "ft"].copy())
    ours.to_pickle(OUT + "ours_ft.pkl")
    base = metrics(R[R.route == "base"].copy())
    base.to_pickle(OUT + "base.pkl")
    D = dsse_eval()
    D["truncated"] = False
    Dm = metrics(D)
    Dm["opus_wrong"] = Dm.opus_label == "incorrect"
    Dm.to_pickle(OUT + "dsse_ft.pkl")
    print(len(ours), ours.dataset_id.nunique(), len(Dm), Dm.dataset_id.nunique())
    # sanity: agreement of my wrong_v with the loader's wrong
    print("ours: wrong_v vs loader wrong agreement", (ours.wrong_v == ours.wrong.astype(bool)).mean().round(3))
    print("dsse: wrong_v vs opus_wrong agreement", (Dm.wrong_v == Dm.opus_wrong).mean().round(3))
    print("base: words mean", base.words.mean(), "n_eq>=2", base.eq2.mean(), "af_rule", base.af_rule.mean(), "ec_A", base.ec_A.mean())

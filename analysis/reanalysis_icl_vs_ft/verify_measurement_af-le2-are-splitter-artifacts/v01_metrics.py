"""Independent re-implementation (verifier) of form metrics. Does NOT import the measurement analyst's robust.py.

Writes M.pkl: main runs (base, icl k16, ft) with
  rule metrics (recomputed from the notebook definition, checked against stored columns)
  my_ans         : last \\boxed{} number > number after last '####' > first number after last 'answer is/answer:' > last number
  seg_n          : markdown-aware segment count (split on newlines and on [.!?] + whitespace; only segments with >=3 letters)
  le2_seg        : seg_n <= 2
  short40/short60: <= 40 / 60 whitespace words
  n_eq           : number of '=' signs (after LaTeX normalisation)
  af_rule_check  : notebook answer_first recomputed
  ns1            : rule splitter yields exactly one sentence
  s1_share       : char share of the first rule-'sentence' in the whole response
  ff_rule        : char fraction of first occurrence of the RULE final number (raw text, commas removed, numeric token)
  ff_my          : same for my_ans, with enumeration masking (Step N, line-initial 'N.' / 'N)', '### N.')
  fw_my          : word index of first occurrence of my_ans (masked); wa_my words after it
  given          : my_ans is a number that appears in the problem statement
  first_eq_pos   : char position of first '=' or 'a op b' arithmetic (normalised); ans_before_work: my_ans first occurs
                   before the first equation/arithmetic, is not a given quantity, and at least one '=' follows
  para1_af       : my_ans (masked) appears in the first paragraph-or-sentence (cut at first blank line or [.!?]+ws)
  explicit_early : an explicit answer statement ('answer is', 'answer:', 'total is', boxed, '####') containing my_ans
                   occurs within the first 25% of characters, with >= 20 words afterwards
"""
import re, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import EVAL, GOLD

OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_measurement_af-le2-are-splitter-artifacts/"
PROMPT = {e["prompt_idx"]: e["prompt"] for e in EVAL}
NUM = re.compile(r"-?\d+(?:\.\d+)?")
SENT = re.compile(r"(?<=[.!?])\s+")
TOK = re.compile(r"(?<![\d.])-?\d+(?:\.\d+)?(?![\d])")


def rule_final(s):
    n = NUM.findall(str(s).replace(",", "").replace("$", ""))
    return n[-1] if n else None


def rule_sents(s):
    return [x for x in SENT.split(s.strip()) if x.strip()]


def rule_af(s):
    fn, ss = rule_final(s), rule_sents(s)
    if fn is None or not ss:
        return False
    return any(abs(float(x) - float(fn)) < 1e-9 for x in NUM.findall(ss[0].replace(",", "").replace("$", "")))


def decomma(s):
    return re.sub(r"(?<=\d),(?=\d)", "", s)


def norm(s):
    t = decomma(str(s))
    t = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", t)
    t = re.sub(r"\\times|\\cdot|×|·", " * ", t)
    t = re.sub(r"\\div|÷", " / ", t)
    t = t.replace("\\$", "$")
    return t


BOX = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
MARK = re.compile(r"(?i)(answer is|answer\s*:|answer\*\*\s*:?|answer:\*\*)")


def my_answer(s):
    s = decomma(str(s))
    bx = BOX.findall(s)
    for b in reversed(bx):
        b2 = norm(b)
        m = re.search(r"\(\s*(-?\d+(?:\.\d+)?)\s*\)\s*/\s*\(\s*(-?\d+(?:\.\d+)?)\s*\)", b2)
        if m and float(m.group(2)) != 0:
            return float(m.group(1)) / float(m.group(2)), "boxed"
        n = TOK.findall(b2.replace("$", " "))
        if n:
            return float(n[0]), "boxed"
    if "####" in s:
        n = TOK.findall(s.split("####")[-1])
        if n:
            return float(n[0]), "hash"
    ms = list(MARK.finditer(s))
    if ms:
        tail = norm(s[ms[-1].end():]).replace("$", " ").replace("*", " ")
        n = TOK.findall(tail[:200])
        if n:
            # take the value after the last '=' within the first line of the tail, if any
            line = tail.strip().split("\n")[0]
            if "=" in line:
                n2 = TOK.findall(line.split("=")[-1])
                if n2:
                    return float(n2[0]), "marker"
            return float(n[0]), "marker"
    n = TOK.findall(s.replace("$", " "))
    return (float(n[-1]), "last") if n else (None, "none")


def mask_enum(t):
    t = re.sub(r"(?i)\b(step|part|case|option|method|dog|child|day|week)\s*#?\d+", lambda m: "X" * len(m.group(0)), t)
    t = re.sub(r"(?m)^(\s*(?:#+\s*)?(?:\*\*)?)(\d+)([.)]\s?)", lambda m: m.group(1) + "X" * len(m.group(2)) + m.group(3), t)
    return t


def first_occ(text, val, masked=True):
    if val is None:
        return np.nan, np.nan, np.nan
    t = norm(text)
    if masked:
        t = mask_enum(t)
    for m in TOK.finditer(t):
        try:
            v = float(m.group(0))
        except ValueError:
            continue
        if abs(v - val) < 1e-6 or abs(abs(v) - abs(val)) < 1e-6 and False:
            p = m.start()
            return p / max(len(t), 1), len(t[:p].split()), len(t[m.end():].split())
    return np.nan, np.nan, np.nan


ARITH = re.compile(r"-?\d+(?:\.\d+)?\s*%?\)?\s*[+\-*/x]\s*\(?\s*-?\d+(?:\.\d+)?")


def first_work(t):
    """char position of first '=' or 'a op b' in normalised text"""
    cands = []
    i = t.find("=")
    if i >= 0:
        cands.append(i)
    m = ARITH.search(t)
    if m:
        cands.append(m.start())
    return min(cands) if cands else None


def prompt_nums(p):
    return {float(x) for x in TOK.findall(decomma(PROMPT[p]).replace("$", " "))}


def segs(s):
    parts = re.split(r"\n+|(?<=[.!?])\s+", s.strip())
    return [x for x in parts if len(re.findall(r"[A-Za-z]", x)) >= 3]


EXPL = re.compile(r"(?i)(answer is|answer\s*:|answer\*\*|the total is|total of|\\boxed\{|####)")


def compute(df):
    df = df.copy()
    resp = df.response.astype(str)
    df["ns_check"] = resp.map(lambda s: len(rule_sents(s)))
    df["af_rule_check"] = resp.map(rule_af)
    df["ns1"] = df.ns_check == 1
    df["s1_share"] = resp.map(lambda s: (len(rule_sents(s)[0]) / max(len(s.strip()), 1)) if rule_sents(s) else np.nan)
    a = [my_answer(s) for s in resp]
    df["my_ans"] = [x[0] for x in a]
    df["my_src"] = [x[1] for x in a]
    df["rule_fn"] = df.final_number.astype(float)
    df["fn_mismatch"] = df.my_ans.notna() & (df.rule_fn - df.my_ans).abs().gt(1e-6)
    df["seg_n"] = resp.map(lambda s: len(segs(s)))
    df["le2_seg"] = df.seg_n <= 2
    df["nw"] = resp.map(lambda s: len(s.split()))
    df["short40"] = df.nw <= 40
    df["short60"] = df.nw <= 60
    df["n_eq"] = resp.map(lambda s: norm(s).count("="))
    df["n_arith"] = resp.map(lambda s: len(ARITH.findall(norm(s))))
    fr = [first_occ(s, v, masked=False) for s, v in zip(resp, df.rule_fn)]
    df["ff_rule"] = [x[0] for x in fr]
    fm = [first_occ(s, v, masked=True) for s, v in zip(resp, df.my_ans)]
    df["ff_my"] = [x[0] for x in fm]
    df["fw_my"] = [x[1] for x in fm]
    df["wa_my"] = [x[2] for x in fm]
    df["given"] = [(v in prompt_nums(p)) if v is not None and not np.isnan(v) else False for v, p in zip(df.my_ans, df.prompt_idx)]
    gold = df.prompt_idx.map(lambda p: float(GOLD[p]))
    df["my_wrong"] = df.my_ans.notna() & ~df.truncated.astype(bool) & (df.my_ans - gold).abs().gt(1e-6)
    # answer before the first equation/arithmetic
    abw = []
    for s, v, g in zip(resp, df.my_ans, df.given):
        t = mask_enum(norm(s))
        fw = first_work(t)
        if v is None or np.isnan(v) or fw is None or g:
            abw.append(False)
            continue
        pos = None
        for m in TOK.finditer(t):
            if abs(float(m.group(0)) - v) < 1e-6:
                pos = m.start()
                break
        abw.append(pos is not None and pos < fw and "=" in t[pos:])
    df["ans_before_work"] = abw
    # first paragraph-or-sentence contains answer (masked)
    p1 = []
    for s, v in zip(resp, df.my_ans):
        t = mask_enum(norm(s)).strip()
        first = re.split(r"\n\s*\n|(?<=[.!?])\s+", t, maxsplit=1)[0]
        p1.append(v is not None and not np.isnan(v) and any(abs(float(x) - v) < 1e-6 for x in TOK.findall(first.replace("$", " "))))
    df["para1_af"] = p1
    df["para1_af_novel"] = df.para1_af & ~df.given
    # explicit early answer statement
    ee = []
    for s, v in zip(resp, df.my_ans):
        t = norm(s)
        ok = False
        if v is not None and not np.isnan(v):
            for m in EXPL.finditer(t):
                if m.start() / max(len(t), 1) > 0.25:
                    break
                n = TOK.findall(t[m.end():m.end() + 60].replace("$", " ").replace("*", " "))
                if n and abs(float(n[0]) - v) < 1e-6 and len(t[m.end():].split()) >= 20:
                    ok = True
                    break
        ee.append(ok)
    df["explicit_early"] = ee
    # style markers
    df["boxed"] = resp.str.contains(r"\\boxed", regex=True)
    df["dollar_block"] = resp.str.contains(r"\$\$", regex=True)
    df["lets_break"] = resp.str.match(r"\s*Let'?s break", case=False)
    df["we_are_given"] = resp.str.match(r"\s*We are given", case=False)
    df["has_period_break"] = resp.map(lambda s: bool(SENT.search(s.strip())))
    return df


if __name__ == "__main__":
    R = pd.read_pickle(OUT + "R.pkl")
    m = R[(R.route.isin(["base", "ft"])) | ((R.route == "icl") & (R.k == 16))].copy()
    M = compute(m)
    M.to_pickle(OUT + "M.pkl")
    print("rule check: n_sentences match", (M.ns_check == M.n_sentences).mean(), " af match", (M.af_rule_check == M.answer_first).mean())
    print("my_ans source by route:\n", M.groupby("route").my_src.value_counts(normalize=True).round(3).unstack())
    print("my_wrong vs stored wrong agreement by route:", M.groupby("route").apply(lambda d: (d.my_wrong == d.wrong).mean()).round(3).to_dict())

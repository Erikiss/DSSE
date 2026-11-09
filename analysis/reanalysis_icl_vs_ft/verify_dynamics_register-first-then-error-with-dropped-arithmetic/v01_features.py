"""Independent per-answer features for verifying the 'register first, then error with dropped arithmetic' claim.
Written from scratch (does not import the dynamics lens' feats.py).

Run: python3 -I v01_features.py  -> V_rows.pkl (all 16,300 answers with features)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCR = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
OUT = SCR / "analysis/verify_dynamics_register-first-then-error-with-dropped-arithmetic"
sys.path.insert(0, str(SCR))
from load_runs import load, EVAL  # noqa: E402

PROMPT = {e["prompt_idx"]: e["prompt"] for e in EVAL}
SOL = {s["prompt_idx"]: s for s in json.load(open(SCR / "gsm8k_eval_solutions.json"))}
NUM = re.compile(r"-?\d+(?:\.\d+)?")

# --- style / register markers (each separately, so we can see which part of 'markdown' disappears)
LATEX = re.compile(r"\\\(|\\\[|\$\$|\\frac|\\times|\\boxed|\\text\{|\\cdot|\\div|\\approx|\$\s*[^$\n]*\\[a-zA-Z]+[^$\n]*\$|\$\s+[^$\n]+\s+\$")  # '$100 + $25' is NOT latex
BOLD = re.compile(r"\*\*[^*\n]+\*\*")
HEADER = re.compile(r"^\s*#{1,6}\s", re.M)
LIST = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+\S", re.M)
RULE = re.compile(r"^\s*(?:---+|\*\*\*+)\s*$", re.M)
EMOJI = re.compile("[\u2705\u2714\u274c\U0001F300-\U0001FAFF]")
# 'thinking-mode' register: self-talk openers / checks typical of Qwen3 <think> traces
THINK = re.compile(r"^\s*(?:okay|alright|ok|hmm|so,? (?:let me|i need)|let me see|let's see|first, i need)\b|\b(?:wait,|let me (?:check|double-check|verify|think)|hmm)",
                   re.I | re.M)


def norm(s):
    t = str(s)
    t = re.sub(r"\\text\{([^{}]*)\}", r" \1 ", t)
    for _ in range(3):
        t = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", t)
    t = re.sub(r"\\times|\\cdot|×|·|\\ast", " * ", t)
    t = re.sub(r"\\div|÷", " / ", t)
    t = re.sub(r"\\boxed\{([^{}]*)\}", r" \1 ", t)
    t = t.replace("\\$", " ").replace("$", " ").replace("**", " ").replace("\\%", "%").replace("\\", " ")
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)
    t = t.replace("{", " ").replace("}", " ")
    return t


N = r"-?\d+(?:\.\d+)?%?"
UNIT = r"(?:\s*[A-Za-z%'\u2019/-]+){0,3}"
SYM = re.compile(rf"{N}{UNIT}\s*\)?\s*[-+*/x]\s*\(?\s*{N}[^=\n]{{0,60}}?=\s*\(?\s*-?\d")          # a op b ... = c
VERB_OP = r"(?:times|multiplied by|divided by|plus|minus|added to|less than|more than|over|by)"
VERBAL = re.compile(rf"{N}{UNIT}\s+{VERB_OP}\s+{N}{UNIT}[^.\n]{{0,40}}?\b(?:is|equals|gives|makes|would be|will be|comes to|results in|totals?|=)\s+\(?-?\d", re.I)
VERBAL2 = re.compile(rf"(?:subtract|add|multiply|divide)(?:ing)?\s+{N}[^.\n]{{0,40}}?(?:from|to|by|and)\s+{N}[^.\n]{{0,40}}?\b(?:is|gives|get|which is|equals|=|leaves|makes)\s+\(?-?\d", re.I)
VERBAL3 = re.compile(rf"(?:half|a third|a quarter|one third|one fourth|{N})\s+of\s+(?:the\s+)?{N}{UNIT}\s+(?:is|=|would be|equals|gives)\s+-?\d", re.I)


def numeq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def feats(resp, pidx):
    s = str(resp)
    nm = norm(s)
    clean = s.replace(",", "").replace("$", "")
    nums = NUM.findall(clean)
    fn = nums[-1] if nums else None
    pn = NUM.findall(PROMPT[pidx].replace(",", "").replace("$", ""))
    inter = SOL[pidx]["intermediates"][:-1]
    cov = np.mean([any(numeq(i, x) for x in NUM.findall(nm)) for i in inter]) if inter else np.nan
    new_nums = [x for x in NUM.findall(nm) if not any(numeq(x, y) for y in pn)]
    n_sym, n_verb = len(SYM.findall(nm)), len(VERBAL.findall(nm)) + len(VERBAL2.findall(nm)) + len(VERBAL3.findall(nm))
    # position of first occurrence of the final number (chars), relative
    pos = None
    if fn is not None:
        for m in NUM.finditer(clean):
            if numeq(m.group(), fn):
                pos = m.start() / max(len(clean), 1)
                break
    return dict(
        v_words=len(s.split()),
        v_latex=bool(LATEX.search(s)), v_bold=bool(BOLD.search(s)), v_header=bool(HEADER.search(s)),
        v_list=bool(LIST.search(s)), v_rule=bool(RULE.search(s)), v_emoji=bool(EMOJI.search(s)),
        v_think=bool(THINK.search(s)), v_newlines=s.count("\n"),
        v_n_sym=n_sym, v_n_verb=n_verb, v_sym=n_sym > 0, v_calc=(n_sym + n_verb) > 0,
        v_cov=cov, v_n_new=len(set(new_nums)), v_pos_first=pos,
        v_fn_in_prompt=fn is not None and any(numeq(fn, y) for y in pn),
    )


def main():
    R, S = load()
    F = pd.DataFrame([feats(r, p) for r, p in zip(R.response, R.prompt_idx)], index=R.index)
    D = pd.concat([R, F], axis=1)
    D["v_md_markup"] = D.v_bold | D.v_header | D.v_list | D.v_rule
    D["v_md_any"] = D.v_md_markup | D.v_latex | D.v_emoji
    D["v_md_strict"] = D.v_latex | D.v_header | D.v_bold          # unambiguous markdown/LaTeX (plain numbered lists excluded)
    D.to_pickle(OUT / "V_rows.pkl")
    S.to_pickle(OUT / "V_runs.pkl")
    print(len(D), "answers;", D.tag.nunique(), "runs")


if __name__ == "__main__":
    main()

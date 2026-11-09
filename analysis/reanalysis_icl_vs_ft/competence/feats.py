"""Shared features for the 'competence-aware brevity' lens.
Robust per-response features + robust graders + per-problem difficulty."""
import sys, json, re
import numpy as np, pandas as pd
SCR = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
sys.path.insert(0, SCR)
from load_runs import load, shots_for, dataset_rows, dsse_eval, DATASET_TABLE, EVAL, GOLD  # noqa

NUM = re.compile(r"-?\d+(?:\.\d+)?")
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
EQ = re.compile(r"=\s*\\?\$?\s*-?\d")                       # an '=' followed by a number
ARITH = re.compile(r"\d\s*(?:[+\-*/×÷x]|\\times|\\div|\\cdot)\s*\\?\$?\(?\d")  # a op b
SOL = {d["prompt_idx"]: d for d in json.load(open(f"{SCR}/gsm8k_eval_solutions.json"))}


def clean(s):
    return str(s).replace(",", "").replace("$", "")


def nums(s):
    out = []
    for x in NUM.findall(clean(s)):
        try:
            out.append(float(x))
        except ValueError:
            pass
    return out


def eqn(a, b):
    return a is not None and b is not None and abs(a - b) < 1e-6


def grade(resp, gold, truncated):
    """Returns dict with last-number (orig), strict-structured and lenient-union verdicts.
    strict: last \\boxed{} > last bold span containing a number > last number.
    lenient: correct if gold equals the last number OR any number in the last boxed OR in the last numeric bold span."""
    g = float(gold)
    ns = nums(resp)
    last = ns[-1] if ns else None
    boxed = [b for b in BOXED.findall(resp) if nums(b)]
    bold = [b for b in BOLD.findall(resp) if nums(b)]
    if boxed:
        strict_c = nums(boxed[-1])
    elif bold:
        strict_c = nums(bold[-1])
    else:
        strict_c = [last] if last is not None else []
    cand = set([last] if last is not None else [])
    if boxed:
        cand |= set(nums(boxed[-1]))
    if bold:
        cand |= set(nums(bold[-1]))
    committed = (not truncated) and last is not None
    # strict: the single chosen number = last number of the chosen span
    strict_pick = strict_c[-1] if strict_c else None
    return dict(
        committed2=committed,
        wrong_last=committed and not eqn(last, g),
        wrong_strict=committed and not eqn(strict_pick, g),
        wrong_len=committed and not any(eqn(c, g) for c in cand),
    )


def features(resp, pidx):
    sol = SOL[pidx]
    inter = [float(x.replace(",", "")) for x in sol["intermediates"][:-1] if NUM.fullmatch(x.replace(",", ""))]
    rn = set(nums(resp))
    frac_inter = np.mean([any(eqn(v, r) for r in rn) for v in inter]) if inter else np.nan
    n = nums(resp)
    fin = n[-1] if n else None
    # char position of first occurrence of the final number, relative
    pos = np.nan
    if fin is not None:
        c = clean(resp)
        for m in NUM.finditer(c):
            try:
                if eqn(float(m.group()), fin):
                    pos = m.start() / max(1, len(c)); break
            except ValueError:
                pass
    return dict(n_eq=len(EQ.findall(resp)), n_arith=len(ARITH.findall(resp)), n_nums=len(n),
                frac_inter=frac_inter, first_final_pos=pos,
                n_steps=sol["n_steps"], n_inter=len(inter))


def build():
    R, S = load()
    R = R[R.route.isin(["base", "icl", "ft", "ftcurve"])].copy()
    G = pd.DataFrame([grade(r, g, t) for r, g, t in zip(R.response, R.gold, R.truncated)], index=R.index)
    F = pd.DataFrame([features(r, p) for r, p in zip(R.response, R.prompt_idx)], index=R.index)
    R = pd.concat([R, G, F], axis=1)
    R["run"] = R.tag
    R["icl16"] = (R.route == "icl") & (R.k == 16)
    R["collapse_run"] = (R.tag == "icl__dsse1-08152__k16__s0")
    return R, S


def difficulty(R):
    """Per-problem difficulty table. pooled icl stats use all k16 icl runs (leave-one-dataset-out version computed separately)."""
    I = R[R.icl16 & ~R.collapse_run]
    B = R[R.route == "base"].set_index("prompt_idx")
    P = pd.DataFrame({"prompt_idx": sorted(SOL)}).set_index("prompt_idx")
    P["n_steps"] = [SOL[i]["n_steps"] for i in P.index]
    P["base_wrong"] = B.wrong_len.astype(float)
    P["base_words"] = B.n_words
    P["icl_wrong_pool"] = I.groupby("prompt_idx").wrong_len.mean()
    P["icl_words_pool"] = I.groupby("prompt_idx").n_words.mean()
    return P

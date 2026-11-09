"""Independent re-build (adversarial verifier) of per-answer features and per-problem difficulty.
Does NOT import the competence analyst's feats.py; own regexes and own grader.
Outputs: A.pkl (one row per answer), Pr.pkl (one row per problem)."""
import sys, re, json
import numpy as np, pandas as pd
SCR = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
OUT = f"{SCR}/analysis/verify_competence_icl-brevity-is-uniform-cosmetic-shrink"
sys.path.insert(0, SCR)
from load_runs import load, dsse_eval, EVAL, GOLD  # noqa

SOL = {d["prompt_idx"]: d for d in json.load(open(f"{SCR}/gsm8k_eval_solutions.json"))}
QTEXT = {e["prompt_idx"]: e["prompt"] for e in EVAL}
NUMRE = re.compile(r"-?\d+(?:\.\d+)?")


def numbers(s):
    s = str(s).replace(",", "").replace("$", "")
    out = []
    for x in NUMRE.findall(s):
        try:
            out.append(float(x))
        except ValueError:
            pass
    return out


def close(a, b):
    return a is not None and b is not None and abs(a - b) < 1e-6 * max(1.0, abs(b))


def last_boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    k = j
    while k < len(s) and depth:
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
        k += 1
    return s[j:k - 1]


ANS = re.compile(r"(?i)(?:final answer|the answer is|answer:)")


def grade(resp, gold, truncated):
    g = float(str(gold).replace(",", ""))
    ns = numbers(resp)
    last = ns[-1] if ns else None
    committed = (not truncated) and last is not None
    cands = set() if last is None else {last}
    bx = last_boxed(resp)
    if bx is not None and numbers(bx):
        cands.add(numbers(bx)[-1])
    m = list(ANS.finditer(resp))
    if m:
        tail = resp[m[-1].end():].split("\n")
        seg = " ".join(tail[:3])  # answer phrase may be followed by a blank line / display block
        if numbers(seg):
            cands.add(numbers(seg)[0])
    return dict(committed_v=committed, wrong_last_v=committed and not close(last, g),
                wrong_len_v=committed and not any(close(c, g) for c in cands))


def norm_math(s):
    s = s.replace(",", "").replace("$", "")
    s = re.sub(r"\\text\{[^{}]*\}", " ", s)
    s = re.sub(r"\\(?:times|cdot)", "*", s).replace("×", "*").replace("÷", "/")
    s = re.sub(r"\\div", "/", s)
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    return s


ARITH_EQ = re.compile(r"\d[\d.]*\s*\)?\s*[-+*/x]\s*\(?\s*[\d.]+[^=\n]{0,60}?=\s*-?\d")  # a op b ... = c
EQ_NUM = re.compile(r"=\s*-?\d")


def feats(resp, pidx):
    m = norm_math(resp)
    sol = SOL[pidx]
    inter = []
    for x in sol["intermediates"][:-1]:
        try:
            inter.append(float(x.replace(",", "")))
        except ValueError:
            pass
    rn = numbers(resp)
    fin = rn[-1] if rn else None
    q = numbers(QTEXT[pidx])
    new_nonfinal = {x for x in rn if not any(close(x, y) for y in q) and not close(x, fin)}
    nomath = re.sub(r"\$\$.*?\$\$|\\\[.*?\\\]|\$[^$\n]*\$", " ", resp, flags=re.S)
    nomath_words = [w for w in nomath.split() if re.search(r"[A-Za-z]", w) and not re.search(r"\d", w)]
    # first occurrence of the final number, relative char position
    pos = np.nan
    if fin is not None:
        c = resp.replace(",", "").replace("$", "")
        for mm in NUMRE.finditer(c):
            try:
                if close(float(mm.group()), fin):
                    pos = mm.start() / max(1, len(c)); break
            except ValueError:
                pass
    return dict(
        words=len(resp.split()),
        headers=len(re.findall(r"(?m)^\s*#{1,6}\s", resp)),
        boxed=resp.count("\\boxed"),
        dispmath=resp.count("$$") // 2 + resp.count("\\["),
        bold=len(re.findall(r"\*\*[^*]+\*\*", resp)),
        hrule=len(re.findall(r"(?m)^\s*(?:---+|\*\*\*+)\s*$", resp)),
        words_nonmath=len(nomath_words),
        n_eqnum=len(EQ_NUM.findall(m)),
        n_arith_eq=len(ARITH_EQ.findall(m)),
        frac_inter=np.mean([any(close(v, r) for r in rn) for v in inter]) if inter else np.nan,
        n_new_nonfinal=len(new_nonfinal),
        pos_final=pos,
    )


def main():
    R, S = load()
    G = pd.DataFrame([grade(r, g, t) for r, g, t in zip(R.response, R.gold, R.truncated)], index=R.index)
    F = pd.DataFrame([feats(r, p) for r, p in zip(R.response, R.prompt_idx)], index=R.index)
    A = pd.concat([R, G, F], axis=1)
    A["n_steps"] = A.prompt_idx.map(lambda p: SOL[p]["n_steps"])
    A["is_base"] = (A.route == "base") | ((A.route == "ftcurve") & (A.dataset_id == "dsse1-08530") & (A.step == 0))
    A["icl16"] = (A.route == "icl") & (A.k == 16)
    A["collapse"] = A.tag == "icl__dsse1-08152__k16__s0"
    A.to_pickle(f"{OUT}/A.pkl")

    B = A[A.is_base & ~A.truncated]
    Pr = pd.DataFrame(index=sorted(SOL))
    Pr["n_steps"] = [SOL[i]["n_steps"] for i in Pr.index]
    Pr["base_n"] = B.groupby("prompt_idx").size()
    Pr["base_words"] = B.groupby("prompt_idx").words.mean()
    Pr["base_logw"] = B.groupby("prompt_idx").words.apply(lambda x: np.log(x).mean())
    Pr["base_wrong"] = B.groupby("prompt_idx").wrong_len_v.mean()
    Pr["base_wrong_last"] = B.groupby("prompt_idx").wrong_last_v.mean()
    # the two base samples separately
    b1 = A[(A.route == "base")].set_index("prompt_idx")
    b2 = A[(A.route == "ftcurve") & (A.dataset_id == "dsse1-08530") & (A.step == 0)].set_index("prompt_idx")
    for nm, b in [("b1", b1), ("b2", b2)]:
        Pr[f"{nm}_words"] = b.words.where(~b.truncated)
        Pr[f"{nm}_wrong"] = b.wrong_len_v.where(~b.truncated).astype(float)
    I = A[A.icl16 & ~A.collapse & ~A.truncated]
    Pr["icl_words"] = I.groupby("prompt_idx").words.mean()
    Pr["icl_logw"] = I.groupby("prompt_idx").words.apply(lambda x: np.log(x).mean())
    Pr["icl_wrong"] = I.groupby("prompt_idx").wrong_len_v.mean()
    Ft = A[(A.route == "ft") & ~A.truncated]
    Pr["ft_wrong"] = Ft.groupby("prompt_idx").wrong_len_v.mean()
    E = dsse_eval()
    Pr["dsse_err92"] = E.groupby("prompt_idx").opus_score.apply(lambda x: 1 - x.mean()) if E.opus_score.mean() < 0.5 else E.groupby("prompt_idx").opus_score.mean()
    Pr.to_pickle(f"{OUT}/Pr.pkl")
    print(len(A), "answers;", Pr.shape)
    for nm, msk in [("base(2)", A.is_base), ("icl16 no collapse", A.icl16 & ~A.collapse), ("collapse", A.collapse), ("ft", A.route == "ft")]:
        d = A[msk & ~A.truncated]
        print(nm, len(d), d[["wrong", "wrong_last_v", "wrong_len_v", "words", "headers", "boxed", "dispmath", "words_nonmath", "n_eqnum", "n_arith_eq", "frac_inter"]].mean().round(3).to_dict())
    print(Pr.describe().round(3).T.to_string())


if __name__ == "__main__":
    main()

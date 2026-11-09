"""Independent loader for the verification of the 'collapse run' claim. Own features, own graders."""
import sys, json, re
import numpy as np, pandas as pd
SCR = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
sys.path.insert(0, SCR)
from load_runs import load, shots_for, dataset_rows, DATASET_TABLE, EVAL, GOLD
OUT = f"{SCR}/analysis/verify_competence_terse-icl-run-is-blind-like-ft"
NUMRE = re.compile(r"-?\d+(?:\.\d+)?")
SOL = {d["prompt_idx"]: d for d in json.load(open(f"{SCR}/gsm8k_eval_solutions.json"))}
QTXT = {e["prompt_idx"]: e["prompt"] for e in EVAL}

def nums(s):
    return [float(x) for x in NUMRE.findall(str(s).replace(",", "").replace("$", ""))]

def close(a, b):
    return a is not None and b is not None and abs(a - b) < 1e-6

def feats(resp, pidx, gold, trunc):
    ns = nums(resp)
    g = float(gold)
    last = ns[-1] if ns else None
    qn = nums(QTXT[pidx])
    new = {x for x in ns[:-1] if not any(close(x, y) for y in qn) and not close(x, last)}
    # any number in response equals gold (most lenient)
    anygold = any(close(x, g) for x in ns)
    # first occurrence position of the final number (chars), relative
    c = str(resp).replace(",", "").replace("$", "")
    pos = np.nan
    if last is not None:
        for m in NUMRE.finditer(c):
            if close(float(m.group()), last):
                pos = m.start() / max(1, len(c)); break
    inter = [float(x.replace(",", "")) for x in SOL[pidx]["intermediates"][:-1] if NUMRE.fullmatch(x.replace(",", ""))]
    rn = set(ns)
    frac_inter = np.mean([any(close(v, r) for r in rn) for v in inter]) if inter else np.nan
    return dict(committed_v=(not trunc) and last is not None,
                wrong_v=(not trunc) and last is not None and not close(last, g),
                wrong_any=(not trunc) and last is not None and not anygold,
                n_new=len(new), workfree_v=int(len(new) == 0),
                n_eqsign=str(resp).count("="),
                arith=len(re.findall(r"\d\s*(?:[+\-*/×÷x]|\\times|\\div|\\cdot)\s*\\?\$?\(?\d", str(resp))),
                first_final_pos=pos, frac_inter=frac_inter,
                n_steps=SOL[pidx]["n_steps"], n_words_v=len(str(resp).split()))

if __name__ == "__main__":
    R, S = load()
    F = pd.DataFrame([feats(r, p, g, t) for r, p, g, t in zip(R.response, R.prompt_idx, R.gold, R.truncated)], index=R.index)
    R = pd.concat([R, F], axis=1)
    R.to_pickle(f"{OUT}/R_v.pkl"); S.to_pickle(f"{OUT}/S_v.pkl")
    print(R.groupby("route").size())
    c = R[R.tag == "icl__dsse1-08152__k16__s0"]
    print("collapse n", len(c), "truncated", c.truncated.sum())

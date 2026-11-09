"""Independent per-response features (verifier). Own definitions, not the claimant's feats.py.
Run: python3 -I v01_features.py -> V.pkl (all answers of base, all icl, ft, ftcurve runs, with features)
"""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCR = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
OUT = SCR / "analysis/verify_dynamics_icl-dose-response-tiny-and-not-a-mode-switch"
sys.path.insert(0, str(SCR))
from load_runs import load  # noqa

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def norm(s):
    t = s.replace(",", "")
    t = re.sub(r"\\(?:text|mathrm|textbf)\{([^{}]*)\}", r" \1 ", t)
    for _ in range(3):
        t = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", t)
    t = re.sub(r"\\times|\\cdot|×|·|\\ast", "*", t)
    t = re.sub(r"\\div|÷", "/", t)
    t = re.sub(r"\\boxed\{([^{}]*)\}", r" \1 ", t)
    t = t.replace("\\$", " ").replace("$", " ").replace("**", " ").replace("\\%", "%").replace("\\left", "").replace("\\right", "")
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    t = t.replace("{", " ").replace("}", " ")
    return t


NUMTOK = r"\(?\s*-?\d+(?:\.\d+)?\s*%?\s*\)?"
UNIT = r"(?:\s*[A-Za-z]+){0,2}"
OPS = r"\s*[-+*/x]\s*"
ARITH = re.compile(rf"{NUMTOK}{UNIT}(?:{OPS}\(?{NUMTOK}{UNIT}\)?)+\s*=\s*\(?\s*-?\d")
EQNUM = re.compile(r"=\s*\(?\s*-?\d")


def feats(resp):
    r = str(resp)
    nw = len(r.split())
    nt = norm(r)
    c = r.replace(",", "").replace("$", "")
    nums = [(m.group(), m.start()) for m in NUM.finditer(c)]
    fn = nums[-1][0] if nums else None
    first_pos = np.nan
    if fn is not None:
        for v, p in nums:
            try:
                if abs(float(v) - float(fn)) < 1e-9:
                    first_pos = p / max(len(c), 1)
                    break
            except ValueError:
                pass
    hdr = bool(re.search(r"^\s{0,3}#{1,6}\s", r, re.M))
    bold = "**" in r
    latex = bool(re.search(r"\$\$|\\\[|\\\(|\\frac|\\boxed|\\text\{|\\times", r))
    lst = bool(re.search(r"^\s*(?:[-*\u2022]|\d+[.)])\s", r, re.M))
    rule = bool(re.search(r"^\s*(?:---+|\*\*\*+)\s*$", r, re.M))
    n_md_tokens = (len(re.findall(r"^\s{0,3}#{1,6}\s", r, re.M)) + r.count("**") // 2 + len(re.findall(r"\$\$|\\\[|\\\(", r))
                   + len(re.findall(r"^\s*(?:[-*\u2022]|\d+[.)])\s", r, re.M)))
    n_ar = len(ARITH.findall(nt))
    n_eq = len(EQNUM.findall(nt))
    return dict(nw=nw, lw=np.log2(max(nw, 1)), short40=nw <= 40, short60=nw <= 60,
                md_hdr=hdr, md_bold=bold, md_latex=latex, md_list=lst, md_rule=rule,
                md_any=hdr or bold or latex or lst or rule,
                md_heavy=hdr or latex,                       # headers or LaTeX display/commands: the 'markdown CoT' signature
                md_types=int(hdr) + int(bold) + int(latex) + int(lst) + int(rule),
                md_per100w=100 * n_md_tokens / max(nw, 1),
                n_ar=n_ar, has_ar=n_ar > 0, n_eq=n_eq, has_eq=n_eq > 0,
                w_per_eq=nw / n_eq if n_eq else np.nan,
                n_lines=len([x for x in r.splitlines() if x.strip()]),
                first_pos=first_pos)


if __name__ == "__main__":
    R, S = load()
    F = pd.DataFrame([feats(x) for x in R.response], index=R.index)
    V = pd.concat([R, F], axis=1)
    V.to_pickle(OUT / "V.pkl")
    S.to_pickle(OUT / "S.pkl")
    print(V.shape, V.groupby("route").tag.nunique().to_dict())
    print(V[V.route == "base"][["nw", "md_any", "md_heavy", "md_hdr", "md_latex", "md_bold", "md_list", "has_ar", "has_eq", "short40"]].mean().round(3))

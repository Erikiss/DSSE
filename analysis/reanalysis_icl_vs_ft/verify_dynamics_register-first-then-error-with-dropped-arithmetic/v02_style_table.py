"""Style / arithmetic markers per checkpoint on the ft curves (own definitions). Run: python3 -I v02_style_table.py"""
import re
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_dynamics_register-first-then-error-with-dropped-arithmetic")
D = pd.read_pickle(OUT / "V_rows.pkl")
# the dynamics lens' two markdown definitions, re-typed here only to compare definitions
NARROW = re.compile(r"^#{1,4}\s|\\\[|\\\(|\*\*Step|^\s*\d+\.\s", re.M)
BROAD = re.compile(r"\*\*|^#{1,4}\s|\$\$|\\\[|\\\(|\\boxed|^\s*[-*]\s|^\s*\d+\.\s|^---", re.M)
D["c_narrow"] = D.response.map(lambda s: bool(NARROW.search(s)))
D["c_broad"] = D.response.map(lambda s: bool(BROAD.search(s)))
C = ['dsse1-08572', 'dsse1-08377', 'dsse1-08166', 'dsse1-08530']
tags = ['base', 'ftcurve__dsse1-08530__step0']
for d in C:
    tags += sorted(D[(D.route == 'ftcurve') & (D.dataset_id == d) & (D.step > 0)].tag.unique(), key=lambda t: int(t.split('step')[1])) + [f'ft__{d}']
tags += [f'icl__{d}__k32__s0' for d in C]
cols = ['c_narrow', 'c_broad', 'v_md_any', 'v_md_strict', 'v_latex', 'v_bold', 'v_header', 'v_list', 'v_think', 'v_sym', 'v_calc', 'truncated', 'wrong']
G = D[D.tag.isin(tags)].groupby('tag')
T = G[cols].mean()
T['words_med'] = G.v_words.median(); T['cov'] = G.v_cov.mean(); T['nl_med'] = G.v_newlines.median()
T = T.loc[tags]
pd.set_option('display.width', 250)
print(T.round(2).to_string())
T.to_csv(OUT / "style_table.csv")
D.to_pickle(OUT / "V_rows2.pkl")

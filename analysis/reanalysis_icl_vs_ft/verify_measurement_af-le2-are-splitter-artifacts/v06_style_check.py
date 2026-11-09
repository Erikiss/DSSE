"""Style-marker variants (opening vs anywhere) to check the analyst's 'Let's break' and arithmetic figures."""
import re, pandas as pd
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_measurement_af-le2-are-splitter-artifacts/"
M = pd.read_pickle(OUT + "M3.pkl")
M["lets_break_any"] = M.response.str.contains(r"Let'?s break", case=False, regex=True)
M["lets_any"] = M.response.str.match(r"\s*Let'?s ", case=False)
M["arith_op_eq2"] = M.n_arith >= 2
print(M.groupby("route")[["lets_break", "lets_break_any", "lets_any", "we_are_given", "boxed", "dollar_block", "arith_op_eq2"]].mean().round(3).to_string())

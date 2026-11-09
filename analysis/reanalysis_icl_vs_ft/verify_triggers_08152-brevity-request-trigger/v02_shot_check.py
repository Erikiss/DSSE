"""Check that shots_for() reconstructs the actual context: reconstructed context length vs logged context_tokens (all icl runs).
Also: does any tokenizer exist locally to compute exact tokens?"""
import numpy as np, pandas as pd
from v00_load import *
S = pd.read_pickle(OUT / "v_S.pkl")
icl = S[S.route == "icl"].copy()
def ctx_chars(d, s, k):
    sh = shots_for(d, s, k)
    return sum(len(r["prompt"]) + len(r["response"]) for r in sh), len(sh)
icl["chars"] = [ctx_chars(d, int(s), int(k))[0] for d, s, k in zip(icl.dataset_id, icl.seed, icl.k)]
icl["k"] = icl.k.astype(int)
# fit tokens ~ a*chars + b*k + c  and inspect residuals
X = np.c_[icl.chars, icl.k, np.ones(len(icl))]
beta, *_ = np.linalg.lstsq(X, icl.context_tokens.astype(float), rcond=None)
icl["resid"] = icl.context_tokens - X @ beta
print("coef", beta.round(4), "R^2", 1 - icl.resid.var() / icl.context_tokens.var())
print(icl.resid.abs().describe().round(1))
print(icl[icl.dataset_id == "dsse1-08152"][["tag", "chars", "context_tokens", "resid"]])
print(icl.sort_values("resid", key=abs, ascending=False).head(5)[["tag", "chars", "context_tokens", "resid"]])
try:
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", local_files_only=True)
    print("tokenizer available")
except Exception as e:
    print("no local tokenizer:", type(e).__name__)

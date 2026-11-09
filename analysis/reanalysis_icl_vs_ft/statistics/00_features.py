"""Per-response features for our runs (R) and for DSSE's own FT answers (all 92 datasets). Saves F_runs.pkl, F_dsse.pkl."""
from common import *

R, S = load()
rf = pd.DataFrame([robust_form(r, p) for r, p in zip(R.response, R.prompt_idx)], index=R.index)
F = pd.concat([R.drop(columns=["n_words"]), rf], axis=1)
F["log_words"] = np.log(F.n_words.clip(lower=1))
F["ec20"] = (F.first_frac < 0.2) & ~F.fin_in_prompt          # robust early commitment: novel final number in first 20%
F.loc[F.first_frac.isna(), "ec20"] = False
F["arith2"] = F.n_arith >= 2
# robust final answer (boxed > 'answer' marker > last number), extractor from the sibling measurement lens (robust.py)
import importlib.util
_sp = importlib.util.spec_from_file_location("mrobust", SCR + "/analysis/measurement/robust.py")
mrobust = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(mrobust)
ra = [mrobust.robust_answer(r, p) for r, p in zip(F.response, F.prompt_idx)]
F["ans_r"] = [a for a, _ in ra]
_g = F.prompt_idx.map(lambda p: float(GOLD[p]))
F["wrong_r"] = F.ans_r.notna() & ~F.truncated.astype(bool) & ((F.ans_r - _g).abs() > 1e-6)
# sanity: our recomputation of the experiment's metrics equals the stored ones
chk = pd.DataFrame([exp_metrics(r, g, t) for r, g, t in zip(R.response, R.gold, R.truncated)], index=R.index)
for c in ["wrong", "le2", "answer_first", "working", "committed"]:
    print(c, "agreement with stored", (chk[c].astype(bool) == R[c].astype(bool)).mean())
F.to_pickle(OUT + "F_runs.pkl")

D = dsse_eval()
D["gold"] = D.prompt_idx.map(GOLD)
em = pd.DataFrame([exp_metrics(r, g) for r, g in zip(D.response, D.gold)], index=D.index)
rf = pd.DataFrame([robust_form(r, p) for r, p in zip(D.response, D.prompt_idx)], index=D.index)
D = pd.concat([D, em, rf], axis=1)
D["log_words"] = np.log(D.n_words.clip(lower=1))
D["ec20"] = ((D.first_frac < 0.2) & ~D.fin_in_prompt).fillna(False)
D["arith2"] = D.n_arith >= 2
ra = [mrobust.robust_answer(r, p) for r, p in zip(D.response, D.prompt_idx)]
D["ans_r"] = [a for a, _ in ra]
D["wrong_r"] = D.ans_r.notna() & ((D.ans_r - D.gold.astype(float)).abs() > 1e-6)
D.to_pickle(OUT + "F_dsse.pkl")
print(F.groupby("route")[["wrong", "wrong_r", "le2", "answer_first", "n_words", "first_frac", "ec20", "n_eq", "n_arith", "arith2"]].mean().round(3))
print("DSSE FT", D[["wrong", "wrong_r", "opus_score", "le2", "answer_first", "n_words", "first_frac", "ec20", "n_eq", "arith2"]].mean().round(3).to_dict())

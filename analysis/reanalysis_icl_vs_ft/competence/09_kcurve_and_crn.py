"""k-curve check (4 datasets, seed 0): does more context make answers shorter / skip working, and on which problems?
Plus a common-random-numbers check: all runs use torch.manual_seed(0), so how often is an ICL answer verbatim identical across runs/base?"""
import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/competence")
from feats import *
from scipy.stats import spearmanr
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/competence"
R = pd.read_pickle(f"{OUT}/R_feat.pkl"); P = pd.read_pickle(f"{OUT}/P_difficulty2.pkl")
C = R[(R.route == "icl") & (R.seed == 0) & R.dataset_id.isin(["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"])].copy()
C["ns"] = C.prompt_idx.map(P.n_steps)
g = C.groupby("k").agg(words=("n_words", "mean"), w60=("n_words", lambda x: (x <= 60).mean()), wrong=("wrong_len", "mean"), n=("n_words", "size"))
g["rho_words_nsteps"] = C.groupby("k").apply(lambda d: np.mean([spearmanr(x.n_words, x.ns).statistic for _, x in d.groupby("dataset_id")]))
print(g.round(3).to_string())
B = R[R.route == "base"].set_index("prompt_idx").response
I = R[R.icl16]
same_base = (I.response.values == B.reindex(I.prompt_idx).values).mean()
dup = I.groupby("prompt_idx").response.apply(lambda s: s.duplicated(keep=False).mean()).mean()
print(f"ICL k16 answers verbatim identical to the base answer: {same_base:.3f}; mean share of an ICL answer that is verbatim duplicated in another run of the same problem: {dup:.3f}")

"""(1) per-run Spearman(words, n_steps): base (2 samples) vs ICL k16 (95 runs), bootstrap over problems.
(2) k-curve (4 datasets, seed 0): words, P(<=60), rho(words, n_steps), P(wrong); and the per-problem cut ratio vs n_steps by k.
(3) working-free answers: no number other than question numbers and the final answer; read all ICL ones.
(4) step-omission check: among short wrong ICL answers, how often is the final number equal to a gold intermediate
    (i.e. the answer stopped one or more steps early) vs base/long wrong answers."""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_competence_icl-brevity-is-uniform-cosmetic-shrink"
SCR = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
A = pd.read_pickle(f"{OUT}/A.pkl"); Pr = pd.read_pickle(f"{OUT}/Pr2.pkl")
SOL = {d["prompt_idx"]: d for d in json.load(open(f"{SCR}/gsm8k_eval_solutions.json"))}
rng = np.random.default_rng(41)
A = A[~A.truncated]
A["run"] = np.where(A.route == "ftcurve", "base_b2", A.tag)
B = A[A.is_base]; I = A[A.icl16 & ~A.collapse]
probs = np.arange(100)
def mean_rho(d, pick):
    vals = []
    for _, g in d.groupby("run"):
        g = g.set_index("prompt_idx").reindex(pick).dropna(subset=["words"])
        vals.append(spearmanr(g.words, g.n_steps).statistic)
    return np.mean(vals)
for nm, d in [("base", B), ("icl16", I)]:
    est = mean_rho(d, probs)
    bs = [mean_rho(d, rng.choice(probs, 100)) for _ in range(300)]
    print(f"{nm}: mean per-run Spearman(words, n_steps) = {est:.3f} [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]")
# paired difference icl - base on the same bootstrap draws
ds = []
for _ in range(300):
    pk = rng.choice(probs, 100); ds.append(mean_rho(I, pk) - mean_rho(B, pk))
print(f"paired diff icl-base rho = {mean_rho(I, probs)-mean_rho(B, probs):+.3f} [{np.percentile(ds,2.5):+.3f},{np.percentile(ds,97.5):+.3f}]")

print("\n== k-curve, 4 datasets seed 0 (k16 = main runs, seed 0)")
C = A[(A.route == "icl") & (A.seed == 0) & A.dataset_id.isin(["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"])].copy()
bw = Pr.base_words
rows = []
for k, g in C.groupby("k"):
    pl = g.groupby("prompt_idx").words.mean()
    lr = np.log(pl / bw)
    rows.append(dict(k=int(k), n=len(g), words=g.words.mean(), w60=(g.words <= 60).mean(), wrong=g.wrong_len_v.mean(),
                     rho=np.mean([spearmanr(x.words, x.n_steps).statistic for _, x in g.groupby("dataset_id")]),
                     ratio=np.exp(lr.mean()), rho_lr_nsteps=spearmanr(lr, Pr.n_steps).statistic,
                     ratio_2step=np.exp(lr[Pr.n_steps == 2].mean()), ratio_6plus=np.exp(lr[Pr.n_steps >= 6].mean())))
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("base (2 samples): P(<=60) =", round((B.words <= 60).mean(), 3), " n =", len(B))

print("\n== working-free (every number is a question number or the final answer)")
for nm, d in [("base", B), ("icl16", I), ("collapse", A[A.collapse & A.icl16]), ("ft", A[A.route == "ft"])]:
    wf = d[d.n_new_nonfinal == 0]
    print(f"{nm}: {len(wf)}/{len(d)} working-free; P(wrong|wf) = {wf.wrong_len_v.mean() if len(wf) else float('nan'):.3f}")
wf = I[I.n_new_nonfinal == 0]
print("ICL working-free: words", wf.words.describe()[["min", "50%", "max"]].to_dict(), "n_eqnum>0:", (wf.n_eqnum > 0).sum(), "wrong:", wf.wrong_len_v.sum())
for _, r in wf.head(3).iterrows():
    print(f"-- {r.tag} p{r.prompt_idx} gold {r.gold} final {r.final_number}: {r.response[:200]!r}")

print("\n== step omission: wrong final equals a gold intermediate (not the last)")
def is_inter(row):
    s = SOL[row.prompt_idx]
    try:
        f = float(row.final_number)
    except (TypeError, ValueError):
        return False
    for x in s["intermediates"][:-1]:
        try:
            if abs(float(x.replace(",", "")) - f) < 1e-6: return True
        except ValueError:
            pass
    return False
for nm, d in [("base wrong", B[B.wrong_len_v]), ("icl short<=60 wrong", I[(I.words <= 60) & I.wrong_len_v]),
              ("icl long wrong", I[(I.words > 60) & I.wrong_len_v]),
              ("icl long wrong, problems w/ short answers", I[(I.words > 60) & I.wrong_len_v & I.prompt_idx.isin(I[I.words <= 60].prompt_idx.unique())])]:
    v = d.apply(is_inter, axis=1) if len(d) else pd.Series(dtype=bool)
    print(f"{nm}: n={len(d)} share final==gold intermediate: {v.mean():.3f}")

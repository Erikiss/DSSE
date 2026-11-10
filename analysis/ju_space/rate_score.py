"""Do the U-space coordinates track what they are named after on DSSE prompts? Blind ratings (two raters, 288 prompts, 0-3 on
ambiguity / incompleteness / conflicting evidence / general uncertainty) against the U coordinates of the same prompts.
Controls: directions built by the same recipe from random words; a cross-validated linear readout as upper reference.
Usage: python rate_score.py <scratch_root> <out_dir>"""
import json, os, sys
import numpy as np, yaml
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from transformers import AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mim_common import unit, load_unembed, load_lens
root, out = sys.argv[1], sys.argv[2]; rng = np.random.default_rng(0)
items = json.load(open(f"{out}/eval_items.json")); lab = np.array([a for a, _, _ in items]); K = ["amb", "inc", "con", "gen"]
R = []
for r in range(2):
    d = json.load(open(f"{out}/abs_out/rate_{r}_result.json")); d = d["ratings"] if isinstance(d, dict) else d; R.append({int(x["id"][1:]): [float(x[k]) for k in K] for x in d})
sel = np.array(sorted(set(R[0]) & set(R[1]))); Y0 = np.array([R[0][i] for i in sel]); Y1 = np.array([R[1][i] for i in sel]); Y = (Y0 + Y1) / 2; fam = lab[sel]
print(f"prompts rated by both: {len(sel)} | mean rating per category (rater 1 / rater 2):", {k: (round(Y0[:, j].mean(), 2), round(Y1[:, j].mean(), 2)) for j, k in enumerate(K)})
print("agreement between the two raters (Spearman):", {k: round(spearmanr(Y0[:, j], Y1[:, j])[0], 2) for j, k in enumerate(K)}, "| share of exact agreement:", {k: round(float((Y0[:, j] == Y1[:, j]).mean()), 2) for j, k in enumerate(K)})
def center(v, f): return v - np.array([v[f == q].mean() for q in f])
def eta2(v, f): return float(sum((f == q).sum() * (v[f == q].mean() - v.mean()) ** 2 for q in set(f)) / ((v - v.mean()) ** 2).sum())
print("share of rating variance between prompt families (eta^2):", {k: round(eta2(Y[:, j], fam), 2) for j, k in enumerate(K)})
print("rating by family (mean of both raters), sorted by 'conflicting evidence':")
for q in sorted(set(fam), key=lambda q: -Y[fam == q, 2].mean()): print(f"   {q:22s}", " ".join(f"{k} {Y[fam == q, j].mean():.2f}" for j, k in enumerate(K)))
tok = AutoTokenizer.from_pretrained(f"{root}/models/qwen3-1.7b"); WU, g = load_unembed(root); terms = yaml.safe_load(open(f"{root}/uspace_repo/data_uspace_terms.yaml")); cats = list(terms)
def tid(w):
    for s in (" " + w, w):
        i = tok.encode(s, add_special_tokens=False)
        if len(i) == 1: return i[0]
pool = np.array([i for i in range(2000, 100000) if tok.decode([i]).strip().isalpha() and len(tok.decode([i]).strip()) > 3]); res = {}
for block in (18, 21):
    J = load_lens(root, block); X = unit(np.load(f"{out}/eval2_states_block{block}.npy").astype(np.float64))[sel]
    pull = lambda ids: unit((WU[ids].astype(np.float64) * g) @ J); na = [len([i for i in map(tid, terms[c]["anchors"]) if i is not None]) for c in cats]; npo = [len([i for i in map(tid, terms[c]["poles"]) if i is not None]) for c in cats]
    B = np.stack([unit(pull([i for i in map(tid, terms[c]["anchors"]) if i is not None]).mean(0) - pull([i for i in map(tid, terms[c]["poles"]) if i is not None]).mean(0)) for c in cats], 1); A = X @ B
    M = np.array([[spearmanr(Y[:, j], A[:, k])[0] for k in range(4)] for j in range(4)]); Mw = np.array([[spearmanr(center(Y[:, j], fam), center(A[:, k], fam))[0] for k in range(4)] for j in range(4)])
    null = np.zeros((300, 4)); nullw = np.zeros((300, 4))
    for t in range(300):
        for j in range(4):
            v = X @ unit(pull(rng.choice(pool, na[j], replace=False)).mean(0) - pull(rng.choice(pool, npo[j], replace=False)).mean(0)); null[t, j] = spearmanr(Y[:, j], v)[0]; nullw[t, j] = spearmanr(center(Y[:, j], fam), center(v, fam))[0]
    Xc = X - X.mean(0); P = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:50].T; cv, cvw = [], []
    for j in range(4):
        pr = np.zeros(len(sel)); prw = np.zeros(len(sel))
        for tr, te in KFold(8, shuffle=True, random_state=0).split(P): pr[te] = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(P[tr], Y[tr, j]).predict(P[te])
        for tr, te in GroupKFold(6).split(P, groups=fam): prw[te] = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(P[tr], Y[tr, j]).predict(P[te])        # trained on other families only
        cv.append(spearmanr(Y[:, j], pr)[0]); cvw.append(spearmanr(Y[:, j], prw)[0])
    print(f"\n[block {block}] Spearman of each rating (rows) with each raw U direction (columns: {', '.join(c[:6] for c in cats)}):"); print(np.round(M, 2))
    print("   the same inside prompt families (family means removed from both):"); print(np.round(Mw, 2))
    for j, k in enumerate(K):
        print(f"   {k}: own U direction {M[j, j]:+.2f} (random-word directions: mean {null[:, j].mean():+.2f}, 5-95% [{np.percentile(null[:, j], 5):+.2f},{np.percentile(null[:, j], 95):+.2f}], share of |null| >= |U|: {(np.abs(null[:, j]) >= abs(M[j, j])).mean():.2f}) | "
              f"inside families {Mw[j, j]:+.2f} (null 5-95% [{np.percentile(nullw[:, j], 5):+.2f},{np.percentile(nullw[:, j], 95):+.2f}], share {(np.abs(nullw[:, j]) >= abs(Mw[j, j])).mean():.2f}) | "
              f"linear readout of 50 PCs, cross-validated: {cv[j]:+.2f}; trained on other families only: {cvw[j]:+.2f}")
    res[block] = dict(M=M.tolist(), Mw=Mw.tolist(), null_abs_p=[float((np.abs(null[:, j]) >= abs(M[j, j])).mean()) for j in range(4)], nullw_abs_p=[float((np.abs(nullw[:, j]) >= abs(Mw[j, j])).mean()) for j in range(4)],
                      null_q=[[float(np.percentile(null[:, j], 5)), float(np.percentile(null[:, j], 95))] for j in range(4)], cv=cv, cv_other_families=cvw)
json.dump(dict(n=int(len(sel)), rater_agreement={k: float(spearmanr(Y0[:, j], Y1[:, j])[0]) for j, k in enumerate(K)}, eta2={k: eta2(Y[:, j], fam) for j, k in enumerate(K)}, blocks=res,
               family_means={q: {k: float(Y[fam == q, j].mean()) for j, k in enumerate(K)} for q in sorted(set(fam))}), open(f"{out}/rate_score.json", "w"), indent=1)

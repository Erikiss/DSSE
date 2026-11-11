"""What are the two main axes of the z-scored MLP activations at the last prompt token (see ufo_compress.py)?
Per block: share of the variance of the first two principal components that is explained by the length of the prompt in
tokens (linear and quadratic term of the logarithm) and by the behaviour category of the prompt (18 categories), and the
mean share over all neurons explained by length and by category. With the split of ufo_compress.py (fit on two thirds, read on the
held-out third): what the family means explain, two axes of the within-family rest, two axes with z-scores clipped at +-5, and
the share of the held-out total that the ten largest neurons hold.
Usage: python ufo_compress_pc.py <scratch_root> <out_dir>"""
import json, os, sys
import numpy as np, torch
from transformers import AutoTokenizer

root, out = sys.argv[1], sys.argv[2]; torch.set_num_threads(int(os.environ.get("NT", "1"))); A = np.load(f"{root}/ju_out/neu_act.npy", mmap_mode="r"); items = json.load(open(f"{root}/ju_out/eval_items.json")); n, L, NI = A.shape
tok = AutoTokenizer.from_pretrained(f"{root}/models/qwen3-1.7b"); ln = np.array([len(tok(tok.apply_chat_template([{"role": "user", "content": it[2]}], tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"]) for it in items])
cats = sorted({it[0] for it in items}); ci = np.array([cats.index(it[0]) for it in items]); lg = np.log(ln); Xl = np.stack([np.ones(n), lg, lg ** 2], 1); Xc = np.eye(len(cats))[ci]; print(f"{n} prompts, {len(cats)} categories, tokens {ln.min()}-{ln.max()} (median {int(np.median(ln))})")
r2 = lambda X, Y: 1 - ((Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0); res = []
perm = np.random.default_rng(0).permutation(n); fit, held = np.sort(perm[: 2 * n // 3]), np.sort(perm[2 * n // 3:])                 # the split of ufo_compress.py
len_by_family = float(r2(Xc, lg[:, None])[0]); print(f"the families explain {len_by_family:.2f} of the variation of the log token count")


def heldout(X):
    """Fitted on two thirds of the prompts, read on the rest: what the family means explain; two axes of what is left after removing the
    family means (as a share of that rest and of the whole); two axes with z-scores clipped at +-5; share of the ten largest neurons."""
    mu, sd = X[fit].mean(0), X[fit].std(0); keep = sd > 1e-6; Zf, Zh = (X[fit][:, keep] - mu[keep]) / sd[keep], (X[held][:, keep] - mu[keep]) / sd[keep]; th = (Zh ** 2).sum()
    fm = np.linalg.lstsq(Xc[fit], Zf, rcond=None)[0]; Rf, Rh = Zf - Xc[fit] @ fm, Zh - Xc[held] @ fm; V = np.linalg.svd(Rf, full_matrices=False)[2][:2].T; res2 = ((Rh @ V) ** 2).sum()
    Cf, Ch = np.clip(Zf, -5, 5), np.clip(Zh, -5, 5); Vc = np.linalg.svd(Cf, full_matrices=False)[2][:2].T
    return dict(family_means_held=float(1 - (Rh ** 2).sum() / th), resid_pca2_held_of_rest=float(res2 / (Rh ** 2).sum()), resid_pca2_held_of_whole=float(res2 / th), pca2_held_clipped=float(((Ch @ Vc) ** 2).sum() / (Ch ** 2).sum()),
                top10_neurons_share_of_held_total=float(np.sort((Zh ** 2).sum(0))[::-1][:10].sum() / th), share_of_squared_z_beyond_5=float(((Zh ** 2) * (np.abs(Zh) > 5)).sum() / th))
for m in range(L):
    X = np.asarray(A[:, m]).astype(np.float64); sd = X.std(0); keep = sd > 1e-6; R = (X[:, keep] - X[:, keep].mean(0)) / sd[keep]; U, s, Vt = np.linalg.svd(R, full_matrices=False); sc = U[:, :2] * s[:2]
    r = dict(block=m, pca2=float((s[:2] ** 2).sum() / (s ** 2).sum()), pc_by_length=[float(x) for x in r2(Xl, sc)], pc_by_category=[float(x) for x in r2(Xc, sc)], neurons_by_length=float(r2(Xl, R).mean()), neurons_by_category=float(r2(Xc, R).mean()),
             neurons_by_both=float(r2(np.concatenate([Xl[:, 1:], Xc], 1), R).mean()), **heldout(X)); res.append(r)
    print(f"block {m:2d}: PCA2 {r['pca2']:.3f} | PC1/PC2 explained by length {r['pc_by_length'][0]:.2f}/{r['pc_by_length'][1]:.2f}, by category {r['pc_by_category'][0]:.2f}/{r['pc_by_category'][1]:.2f} | all neurons: length {r['neurons_by_length']:.3f}, category {r['neurons_by_category']:.3f}, both {r['neurons_by_both']:.3f}", flush=True)
    print(f"          held-out third: family means {r['family_means_held']:.3f} | two axes of the rest {r['resid_pca2_held_of_rest']:.3f} (= {r['resid_pca2_held_of_whole']:.3f} of the whole) | two axes with z clipped at 5: {r['pca2_held_clipped']:.3f} | ten neurons hold {r['top10_neurons_share_of_held_total']:.2f} of the total", flush=True)
json.dump(dict(prompts=int(n), categories=cats, families_explain_of_log_length=len_by_family, rows=res), open(f"{out}/ufo_compress_pc.json", "w"), indent=1)

"""Static part: how do interpretable units behave with respect to separability?
Base Qwen3-1.7B, residual after block 21, last prompt token of the DSSE eval prompts (18 prompt families).
Units: features of a public SAE trained on this model and layer (no training here), U-space coordinates, lens-word readouts.
Usage: python mim_static.py <scratch_root> <out_dir> [t0|t2]"""
import itertools, json, os, sys
import numpy as np, yaml
from scipy.stats import rankdata
from transformers import AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mim_common import *

root, out = sys.argv[1], sys.argv[2]; tag = sys.argv[3] if len(sys.argv) > 3 else "t0"
rng = np.random.default_rng(0); lab, text, X, F = load_eval(out); n, d = X.shape; beh = sorted(set(lab)); nb = len(beh)
tok = AutoTokenizer.from_pretrained(f"{root}/models/qwen3-1.7b"); WU, g = load_unembed(root); J = load_lens(root)
sae = SAE(f"{root}/sae_dl/k_L21_{tag}_ae.pt"); SC = float(np.median(np.linalg.norm(X, axis=1)))
print(f"[static, block {BLOCK}, SAE {tag}: {sae.n} features, k={sae.k}, threshold {sae.thr:.3f}] {n} prompts, {nb} families, median state norm {SC:.1f}")
res = dict(tag=tag, n=n)

# ---------- 0) is the SAE valid at this token position?
def recon(Z):
    A = sae.encode(Z); R = sae.decode(A).astype(np.float64)
    return dict(fve_generic=float(1 - (Z - R).var(0).sum() / Z.var(0).sum()), fve_within=float(1 - ((Z - R) ** 2).sum() / ((Z - Z.mean(0)) ** 2).sum()),
                nmse=float(np.median(((Z - R) ** 2).sum(1) / (Z ** 2).sum(1))), cos=float(np.median((unit(Z) * unit(R)).sum(1))), l0=float(np.median((A > 0).sum(1))), norm=float(np.median(np.linalg.norm(Z, axis=1)))), A, R
C = np.load(f"{out}/eval2_ctx_block21.npy").reshape(-1, d).astype(np.float64); keep = np.linalg.norm(C, axis=1) < 10 * np.median(np.linalg.norm(C, axis=1))
r_last, A_all, R_all = recon(X); r_ctx, _, _ = recon(C[keep])
mu = np.stack([X[lab == b].mean(0) for b in beh]); muR = np.stack([R_all[lab == b].mean(0) for b in beh])
fve_between = float(1 - ((mu - muR - (mu - muR).mean(0)) ** 2).sum() / ((mu - mu.mean(0)) ** 2).sum())
print("0) SAE reconstruction. fve_generic = 1 - var(x - xhat)/var(x) (the SAE authors' measure), fve_within = share of the squared deviation from this set's mean")
print("   last prompt token :", {k: round(v, 3) for k, v in r_last.items()}); print(f"   user-content tokens ({keep.sum()} of {len(keep)} kept after the authors' norm filter):", {k: round(v, 3) for k, v in r_ctx.items()})
print(f"   share of the between-family differences (family centroids) that survives reconstruction: {fve_between:.3f}")
alive = (A_all > 0).mean(0); print(f"   features active on at least one prompt: {(alive > 0).sum()} of {sae.n}; on more than half of all prompts: {(alive > .5).sum()}; decoder column norms {np.linalg.norm(sae.Wd, axis=0).min():.3f}-{np.linalg.norm(sae.Wd, axis=0).max():.3f}")
def loo_nc(Z):
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12); ci = np.array([beh.index(b) for b in lab]); S = np.stack([Z[ci == c].sum(0) for c in range(nb)]); N = np.bincount(ci).astype(float); ok = 0
    for i in range(n):
        Sx, Nx = S.copy(), N.copy(); Sx[ci[i]] -= Z[i]; Nx[ci[i]] -= 1; ok += int(np.argmin(((Sx / Nx[:, None] - Z[i]) ** 2).sum(1)) == ci[i])
    return ok / n
def pcs(Z, k=50):
    Zc = Z - Z.mean(0); return Zc @ np.linalg.svd(Zc, full_matrices=False)[2][:k].T
acc18 = dict(state=loo_nc(pcs(X)), sae_recon=loo_nc(pcs(R_all)), sae_error=loo_nc(pcs(X - R_all)), sae_acts=loo_nc(pcs(np.log1p(A_all[:, alive > 0].astype(np.float64)))))
print("   18-family leave-one-out nearest-centroid accuracy on 50 PCs (chance 0.056):", {k: round(v, 3) for k, v in acc18.items()})
res.update(recon_last=r_last, recon_ctx=r_ctx, fve_between=fve_between, acc18=acc18); np.save(f"{out}/mim_sae_acts_{tag}.npy", A_all.astype(np.float32))

# ---------- U-space at this block (reference construction) and lens readout of a direction
terms = yaml.safe_load(open(f"{root}/uspace_repo/data_uspace_terms.yaml")); cats = list(terms)
def tid(w):
    for s in (" " + w, w):
        i = tok.encode(s, add_special_tokens=False)
        if len(i) == 1: return i[0]
def pull(ids): return unit((WU[ids].astype(np.float64) * g) @ J)
Braw = np.stack([unit(pull([i for i in map(tid, terms[c]["anchors"]) if i is not None]).mean(0) - pull([i for i in map(tid, terms[c]["poles"]) if i is not None]).mean(0)) for c in cats], 1)
w_, V_ = np.linalg.eigh(Braw.T @ Braw); U = Braw @ (V_ @ np.diag(np.maximum(w_, 1e-12) ** -0.5) @ V_.T); np.save(f"{out}/mim_U_block{BLOCK}.npy", np.concatenate([U, Braw], 1))
import re; bad = re.compile(r"[^A-Za-z一-鿿]")
def words(v, k=8):
    lg = WU @ (g * (J @ v)).astype(np.float32); top = []
    for i in np.argsort(-lg)[:300]:
        w = tok.decode([int(i)]).strip()
        if len(w) > 2 and not bad.search(w) and w.lower() not in [t.lower() for t in top]: top.append(w)
        if len(top) == k: break
    return top

# ---------- 1) separability of every pair of families: probe directions vs interpretable units (held-out prompts)
def col_auc(Atr, y):                                     # AUROC of every column for y=1 against y=0 (rank formula)
    r = rankdata(Atr, axis=0); n1 = y.sum(); return (r[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
def omp(v, D, k):                                        # greedy orthogonal matching pursuit on unit columns of D; returns explained share of |v|^2 after each step
    v = unit(v); r = v.copy(); S = []; ex = []
    for _ in range(k):
        S.append(int(np.argmax(np.abs(D.T @ r)))); coef = np.linalg.lstsq(D[:, S], v, rcond=None)[0]; r = v - D[:, S] @ coef; ex.append(1 - float(r @ r))
    return ex, S
Dn = unit(sae.Wd.T.astype(np.float64)).T; Ff = unit(F)
rows = []
for (a, b), seed in itertools.product(itertools.combinations(beh, 2), range(3)):
    S1, S2, T = split3(lab, a, b, seed); tr = np.concatenate([S1, S2]); ytr = (lab[tr] == b).astype(int); yte = (lab[T] == b).astype(int)
    svm = LinProbe(X[tr], ytr, C=1e4, sc=SC); s_te = svm.score(X[T]); dm = X[tr][ytr == 1].mean(0) - X[tr][ytr == 0].mean(0); mid = (X[tr][ytr == 1].mean(0) + X[tr][ytr == 0].mean(0)) / 2
    yp = rng.permutation(ytr); rnd = LinProbe(X[tr], yp, C=1e4, sc=SC)
    Utr, Ute = unit(X[tr]) @ U, unit(X[T]) @ U; us = float(Utr.std()); pu = LinProbe(Utr, ytr, C=10, sc=us)
    Rb = np.linalg.qr(rng.standard_normal((d, 4)))[0]; pr = LinProbe(unit(X[tr]) @ Rb, ytr, C=10, sc=float((unit(X[tr]) @ Rb).std()))
    Atr, Ate = A_all[tr].astype(np.float64), A_all[T].astype(np.float64); on = (Atr > 0).sum(0) >= 5; au = np.where(on, col_auc(Atr, ytr), .5); order = np.argsort(-np.abs(au - .5))
    sgn = np.sign(au - .5); r = dict(a=a, b=b, seed=seed, n_tr=len(tr), n_te=len(T), svm_acc=float(((s_te > 0) == yte).mean()), svm_auc=auroc(s_te, yte), n_sv=svm.n_sv,
                                     dm_acc=float((((X[T] - mid) @ dm > 0) == yte).mean()), dm_auc=auroc((X[T] - mid) @ dm, yte),
                                     rand_train_acc=float(((rnd.score(X[tr]) > 0) == yp).mean()), rand_test_acc=float(((rnd.score(X[T]) > 0) == yte).mean()),
                                     U_acc=float(((pu.score(Ute) > 0) == yte).mean()), rand4_acc=float(((pr.score(unit(X[T]) @ Rb) > 0) == yte).mean()),
                                     n_feat_train90=int((np.abs(au - .5) >= .4).sum()), cos_svm_dm=float(unit(svm.w) @ unit(dm)), U_share_svm=float(((unit(svm.w) @ U) ** 2).sum()), U_share_dm=float(((unit(dm) @ U) ** 2).sum()))
    for k in (1, 5, 20):
        j = order[:k]; m0, sd0 = Atr[:, j].mean(0), Atr[:, j].std(0) + 1e-9; z = lambda M: (((M[:, j] - m0) / sd0) * sgn[j]).sum(1)
        thr = (z(Atr)[ytr == 1].mean() + z(Atr)[ytr == 0].mean()) / 2; r[f"sae{k}_acc"] = float(((z(Ate) > thr) == yte).mean()); r[f"sae{k}_auc"] = auroc(z(Ate), yte)
    r["best_feat"] = int(order[0]); r["best_feat_test_auc"] = float(max(auroc(Ate[:, order[0]], yte), 1 - auroc(Ate[:, order[0]], yte)))
    r["n_feat_test90"] = int(sum(max(auroc(Ate[:, q], yte), 1 - auroc(Ate[:, q], yte)) >= .9 for q in order[:int((np.abs(au - .5) >= .4).sum())][:200]))
    if seed == 0:
        dl = unit(J.T @ (g * (Ff[tr][ytr == 1].mean(0) - Ff[tr][ytr == 0].mean(0))))          # lens gradient direction: steepest ascent of the final-state class difference under the linear lens (not a pre-image)
        r.update(cos_svm_lens=float(unit(svm.w) @ dl), cos_dm_lens=float(unit(dm) @ dl), cos_rand_lens=float(unit(rnd.w) @ dl), cos_svm_rand=float(unit(svm.w) @ unit(rnd.w)))
        for nm, v in (("svm", svm.w), ("dm", dm), ("randlabel", rnd.w), ("gauss", rng.standard_normal(d))):
            ex, S = omp(v, Dn, 20); r[f"omp5_{nm}"], r[f"omp20_{nm}"] = ex[4], ex[19]
            if nm in ("svm", "dm"): r[f"omp_feats_{nm}"] = S[:5]
    rows.append(r)
import pandas as pd
T_ = pd.DataFrame(rows); pd.set_option("display.width", 250); num = T_.select_dtypes("number").columns
M = T_.groupby(["a", "b"])[num].mean().drop(columns=["seed"]); res["pairs"] = T_.to_dict("records")
q = lambda c: f"{M[c].mean():.3f} [{M[c].quantile(.05):.2f}-{M[c].quantile(.95):.2f}]"
print(f"1) all {len(M)} pairs of families, 3 random splits each (train 60%, test 40% of prompts). Mean over pairs [5-95% of pairs]:")
print(f"   held-out accuracy: hard-margin SVM (2048 dims) {q('svm_acc')} | difference of class means {q('dm_acc')} | SVM on random labels: train {q('rand_train_acc')}, test {q('rand_test_acc')}")
print(f"   U-space (4 coordinates) {q('U_acc')} | random 4 dims {q('rand4_acc')}")
print(f"   SAE units chosen on the training prompts: best single feature AUROC {q('best_feat_test_auc')} | 1 feature acc {q('sae1_acc')} | 5 features {q('sae5_acc')} | 20 features {q('sae20_acc')}")
print(f"   features with training AUROC >= 0.9: {q('n_feat_train90')} | of these still >= 0.9 on held-out prompts: {q('n_feat_test90')} | support vectors {q('n_sv')} of {M['n_tr'].mean():.0f} training points")
print(f"   geometry (split 0): cos(SVM normal, class-mean difference) {q('cos_svm_dm')} | cos(SVM, lens direction) {q('cos_svm_lens')} | cos(mean difference, lens direction) {q('cos_dm_lens')} | cos(random-label SVM, lens) {q('cos_rand_lens')}")
print(f"   share of the direction inside the 4-dim U-space: SVM {q('U_share_svm')}, mean difference {q('U_share_dm')} (a random direction has {4 / d:.4f})")
print("   share of the direction explained by 5 / 20 SAE features (greedy sparse fit): " + " | ".join(f"{nm}: {q('omp5_' + nm)} / {q('omp20_' + nm)}" for nm in ("svm", "dm", "randlabel", "gauss")))
sel = [tuple(sorted(p)) for p in PAIRS]; cols = ["svm_acc", "dm_acc", "rand_test_acc", "U_acc", "sae1_acc", "sae5_acc", "sae20_acc", "best_feat_test_auc", "n_feat_test90", "cos_svm_dm", "cos_svm_lens", "cos_dm_lens", "omp20_svm", "omp20_dm"]
print("   the pairs used for the attack:"); print(M.loc[sorted(set(sel)), cols].round(2).to_string())

# ---------- 2) one family against the rest: which units carry it, and what do they read as (lens words of the decoder direction)?
print("2) one family against all others: SAE features (all 1,472 prompts). auc = AUROC of the feature, on = share of the family's prompts where it is active, off = share of other prompts")
fam = {}
for b in beh:
    y = (lab == b).astype(int); on = (A_all > 0).sum(0) >= 5; au = np.where(on, col_auc(A_all.astype(np.float64), y), .5); top = np.argsort(-au)[:3]
    fam[b] = [dict(feat=int(j), auc=float(au[j]), on=float((A_all[y == 1, j] > 0).mean()), off=float((A_all[y == 0, j] > 0).mean()), words=words(Dn[:, j])) for j in top]
    print(f"   {b:22s} n(auc>=0.9)={int((au >= .9).sum()):3d} | " + " | ".join(f"#{f['feat']} auc {f['auc']:.2f} on {f['on']:.2f} off {f['off']:.2f} [{', '.join(f['words'][:5])}]" for f in fam[b]))
res["family_features"] = fam
json.dump(res, open(f"{out}/mim_static_{tag}.json", "w"), indent=1)
print("saved")

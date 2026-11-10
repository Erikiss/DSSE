"""'U++': behavior axes for DSSE, built with the U-Space recipe from labelled contrasts (Braun et al., App. D.2).
Per behavior: shown-minus-not-shown direction -> decoded through the J-Lens -> top-15 words become anchors ->
axis = pulled-back anchor centroid minus centroid of frequent words. Held-out AUROC, controls, cross-behavior structure,
and application to DSSE training rows. Usage: python axes.py <scratch_root> <out_dir> [block] [eot|mean]"""
import glob, json, re, sys
import numpy as np, pandas as pd, torch, yaml
from safetensors import safe_open
from sklearn.metrics import roc_auc_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import spearmanr
from transformers import AutoTokenizer
root, out = sys.argv[1], sys.argv[2]; BLOCK = int(sys.argv[3]) if len(sys.argv) > 3 else 18; POS = sys.argv[4] if len(sys.argv) > 4 else "eot"
rng = np.random.default_rng(0); tok = AutoTokenizer.from_pretrained(f"{root}/models/qwen3-1.7b")
Jm = torch.load(f"{root}/jlens/Qwen3-1.7B_jacobian_lens.pt", map_location="cpu", weights_only=True)["J"][BLOCK].float().numpy().astype(np.float64)
WU = g = None
for f in glob.glob(f"{root}/models/qwen3-1.7b/*.safetensors"):
    with safe_open(f, framework="pt") as sf:
        for k in sf.keys():
            if k.endswith("lm_head.weight") or (WU is None and k.endswith("embed_tokens.weight")): WU = sf.get_tensor(k).float().numpy()
            if k.endswith("model.norm.weight") or k == "norm.weight": g = sf.get_tensor(k).float().numpy()
def unit(a): return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
def pull(ids): return unit((WU[ids].astype(np.float64) * g) @ Jm)
def tid(w):
    for s in (" " + w, w):
        i = tok.encode(s, add_special_tokens=False)
        if len(i) == 1: return i[0]
freq = [i for i in map(tid, open(f"{out}/freq10k.txt").read().split()[:1000]) if i is not None]; REG = pull(freq).mean(0)       # register direction (frequent words)
pool = np.array([i for i in range(2000, 100000) if tok.decode([i]).strip().isalpha() and len(tok.decode([i]).strip()) > 3])
letters = re.compile(r"[A-Za-z一-鿿]")
def decode_top(d, k=15):
    lg = WU @ (g * (Jm @ d)); ids, words = [], []
    for i in np.argsort(-lg)[:600]:
        w = tok.decode([int(i)]).strip()
        if len(w) >= 2 and letters.search(w) and w.lower() not in [x.lower() for x in words]: ids.append(int(i)); words.append(w)
        if len(ids) == k: break
    return ids, words
def axis_from(ids): return unit(pull(ids).mean(0) - REG)
# generic U-Space (uncertainty) for comparison
terms = yaml.safe_load(open(f"{root}/uspace_repo/data_uspace_terms.yaml")); B = []
for c in terms:
    a = [i for i in map(tid, terms[c]["anchors"]) if i is not None]; p = [i for i in map(tid, terms[c]["poles"]) if i is not None]; B.append(unit(pull(a).mean(0) - pull(p).mean(0)))
B = np.stack(B, 1); w_, V_ = np.linalg.eigh(B.T @ B); U = B @ (V_ @ np.diag(np.maximum(w_, 1e-12) ** -0.5) @ V_.T)

pairs = json.load(open(f"{out}/axes_pairs.json")); flat = json.load(open(f"{out}/axes_flat.json")); Z = np.load(f"{out}/axes_states.npz")
X = unit(Z[f"{POS}{BLOCK}"].astype(np.float64)); ntok = Z["ntok"]; beh_of = np.array([pairs[k]["behavior"] for k, _ in flat]); shown = np.array([s == "shown" for _, s in flat]); pidx = np.array([pairs[k]["prompt_idx"] for k, _ in flat])
behs = sorted(set(beh_of)); print(f"[U++ axes, block {BLOCK}, position {POS}] {len(pairs)} pairs, {len(behs)} behaviors")
auc = lambda s, y: roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan
rows, AX, DM, ANCH = [], {}, {}, {}
for b in behs:
    m = beh_of == b; pr = np.array(sorted(set(pidx[m]))); res = {k: [] for k in ("axis", "diffmeans", "length", "U_mean", "rand")}
    for _ in range(20):                                     # 20 random 70/30 splits by prompt
        test = set(rng.choice(pr, max(2, int(round(.3 * len(pr)))), replace=False)); te = m & np.isin(pidx, list(test)); tr = m & ~np.isin(pidx, list(test))
        d = X[tr & shown].mean(0) - X[tr & ~shown].mean(0); ids, _ = decode_top(d); ax = axis_from(ids); y = shown[te]
        res["axis"].append(auc(X[te] @ ax, y)); res["diffmeans"].append(auc(X[te] @ d, y)); res["length"].append(auc(ntok[te].astype(float), y)); res["U_mean"].append(auc((X[te] @ U).mean(1), y))
        res["rand"].append(np.mean([auc(X[te] @ axis_from(rng.choice(pool, 15, replace=False)), y) for _ in range(10)]))
    d = X[m & shown].mean(0) - X[m & ~shown].mean(0); ids, words = decode_top(d); AX[b] = axis_from(ids); DM[b] = unit(d); ANCH[b] = words
    rows.append(dict(behavior=b, pairs=int(m.sum() // 2), prompts=len(pr), axis=np.nanmean(res["axis"]), axis_sd=np.nanstd(res["axis"]), diffmeans=np.nanmean(res["diffmeans"]), random_words=np.nanmean(res["rand"]),
                     U_generic=np.nanmean(res["U_mean"]), length=np.nanmean(res["length"]), cos_axis_dm=float(AX[b] @ DM[b])))
T = pd.DataFrame(rows).set_index("behavior"); pd.set_option("display.width", 250)
print("1) held-out AUROC, shown vs not shown on unseen prompts (mean over 20 splits). axis = lens-word axis (U++), diffmeans = direct direction (upper reference),")
print("   random_words = same construction with 15 random words, U_generic = mean coordinate in the uncertainty U-Space, length = token count only")
print(T.round(3).to_string())
print("   mean over behaviors:", T[["axis", "diffmeans", "random_words", "U_generic", "length"]].mean().round(3).to_dict())
print("2) the model's own words for each behavior (anchors = top-15 J-Lens words of the shown-minus-not-shown direction):")
for b in behs: print(f"   {b:22s} {', '.join(ANCH[b])}")
# 3) cross-behavior structure: axis of a scoring the contrast of b
C = pd.DataFrame({b: {a: auc(X[beh_of == b] @ AX[a], shown[beh_of == b]) for a in behs} for b in behs}); G = pd.DataFrame({b: {a: float(AX[a] @ AX[b]) for a in behs} for b in behs}); Gd = pd.DataFrame({b: {a: float(DM[a] @ DM[b]) for a in behs} for b in behs})
short = {b: b[:9] for b in behs}
print("3) cross-AUROC: row = axis, column = contrast it scores (0.5 = unrelated, below 0.5 = opposite)"); print(C.rename(index=short, columns=short).round(2).to_string())
od = C.values[~np.eye(len(behs), dtype=bool)]; print(f"   diagonal mean {np.diag(C.values).mean():.2f} | off-diagonal mean {od.mean():.2f}, 5-95%: {np.percentile(od, 5):.2f}-{np.percentile(od, 95):.2f}")
strong = sorted([(a, b, C.loc[a, b], C.loc[b, a]) for i, a in enumerate(behs) for b in behs[i + 1:]], key=lambda t: -abs((t[2] + t[3]) / 2 - .5))
print("   strongest pairs (mean of both directions):", "; ".join(f"{a}~{b} {((x + y) / 2):.2f}" for a, b, x, y in strong[:10]))
print("   cosine between direct directions (diff-in-means), strongest pairs:", "; ".join(f"{a}~{b} {Gd.loc[a, b]:+.2f}" for a, b, _, _ in sorted([(a, b, 0, 0) for i, a in enumerate(behs) for b in behs[i + 1:]], key=lambda t: -abs(Gd.loc[t[0], t[1]]))[:10]))
Sym = ((C.values + C.values.T) / 2); Dm = 1 - np.clip(2 * (Sym - .5), -1, 1); np.fill_diagonal(Dm, 0)
cl = fcluster(linkage(Dm[np.triu_indices(len(behs), 1)], "average"), 5, "maxclust"); print("   groups in U++ (k=5, from cross-AUROC):", [[b for b, c in zip(behs, cl) if c == q] for q in sorted(set(cl))])
# 4) do the axes light up on DSSE TRAINING rows of the datasets that induce the behavior? (mean-over-response states)
try:
    items = pd.DataFrame(json.load(open(f"{out}/train_items.json"))); Hr = np.load(f"{out}/train_states_block{BLOCK}.npy").astype(np.float64)
    Xm = unit(Z[f"mean{BLOCK}"].astype(np.float64)); key = items.groupby("dataset", sort=False).ngroup().values
    Hd = unit(np.stack([Hr[key == k].mean(0) for k in range(key.max() + 1)])); labd = items.groupby("dataset", sort=False).behavior.first().values; wd = items.groupby("dataset", sort=False).words.mean().values
    out4 = []
    for b in behs:
        m = beh_of == b; d = Xm[m & shown].mean(0) - Xm[m & ~shown].mean(0); ids, _ = decode_top(d); ax = axis_from(ids); y = labd == b
        if y.sum() == 0: continue
        s_ax, s_dm = Hd @ ax, Hd @ unit(d); r = [auc(Hd @ axis_from(rng.choice(pool, 15, replace=False)), y) for _ in range(200)]
        out4.append(dict(behavior=b, n_own=int(y.sum()), auroc_axis=auc(s_ax, y), auroc_diffmeans=auc(s_dm, y), random_mean=np.mean(r), random_p95=np.percentile(r, 95), length_auroc=auc(wd, y)))
    T4 = pd.DataFrame(out4).set_index("behavior")
    print("4) DSSE training rows (DE datasets, 6 per behavior): does a behavior's axis rank the datasets that induce it above the other 102 datasets? (AUROC over datasets)")
    print(T4.round(3).to_string()); print("   mean:", T4[["auroc_axis", "auroc_diffmeans", "random_mean", "length_auroc"]].mean().round(3).to_dict())
except FileNotFoundError as e:
    print("4) training-row states not available yet:", e); T4 = pd.DataFrame()
json.dump(dict(block=BLOCK, pos=POS, heldout=T.round(4).reset_index().to_dict("records"), anchors=ANCH, cross=C.round(3).to_dict(), cos_diffmeans=Gd.round(3).to_dict(), cos_axes=G.round(3).to_dict(),
               groups=[[b for b, c in zip(behs, cl) if c == q] for q in sorted(set(cl))], train=T4.round(4).reset_index().to_dict("records")), open(f"{out}/result_axes_block{BLOCK}_{POS}.json", "w"), indent=1)
np.savez(f"{out}/axes_block{BLOCK}_{POS}.npz", **{f"ax_{b}": AX[b] for b in behs}, **{f"dm_{b}": DM[b] for b in behs})

"""Materials for the 'abstraction as interpretability tool' loop (ARC protocol: induce a rule from examples, test it blind on new ones).
Units: SAE features, U-space directions, probe normals, class-mean differences, and control directions (random, random lens words).
For each unit: 10 high + 10 low example prompts for the inducer, 8 + 8 held-out prompts (shuffled, unlabelled) for a blind predictor.
Also a rating sheet: 16 prompts per family, to be rated blind for the four U-space uncertainty categories.
Usage: python abs_prepare.py <scratch_root> <out_dir> [t0|t2]"""
import json, os, sys
import numpy as np, yaml
from transformers import AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mim_common import *

root, out = sys.argv[1], sys.argv[2]; tag = sys.argv[3] if len(sys.argv) > 3 else "t0"
rng = np.random.default_rng(7); lab, text, X, F = load_eval(out); n, d = X.shape; beh = sorted(set(lab)); Xu = unit(X)
A = np.load(f"{out}/mim_sae_acts_{tag}.npy"); mj = json.load(open(f"{out}/mim_variants_{tag}.json")); pairs = mj["pairs"]; st = json.load(open(f"{out}/mim_static_{tag}.json"))
UB = np.load(f"{out}/mim_U_block{BLOCK}.npy"); Braw = UB[:, 4:]; WU, g = load_unembed(root); J = load_lens(root); tok = AutoTokenizer.from_pretrained(f"{root}/models/qwen3-1.7b")
cats = list(yaml.safe_load(open(f"{root}/uspace_repo/data_uspace_terms.yaml")))
half = np.zeros(n, bool)                                 # induction half (True) / test half, stratified by family
for b in beh:
    idx = rng.permutation(np.where(lab == b)[0]); half[idx[:len(idx) // 2]] = True
units = []
# SAE features: the two strongest for each attacked pair, then the best one-vs-rest feature of families not yet covered
feats = []
for P in pairs:
    for f in P["sae_feats"][:2]:
        if f not in feats: feats.append(f)
for b in beh:
    f = st["family_features"][b][0]["feat"]
    if f not in feats and len(feats) < 16: feats.append(f)
for f in feats[:16]: units.append(dict(kind="sae", ref=int(f), act=A[:, f].astype(np.float64), sparse=True))
for k, c in enumerate(cats): units.append(dict(kind="U", ref=c, act=Xu @ Braw[:, k], sparse=False))
for pi in (0, 2, 6, 7):
    P = pairs[pi]; w = np.array(P["sur_w"]); S1 = np.array(P["S1"]); dm = X[S1][lab[S1] == P["b"]].mean(0) - X[S1][lab[S1] == P["a"]].mean(0)
    units.append(dict(kind="svm", ref=f"{P['a']}>{P['b']}", act=X @ unit(w), sparse=False)); units.append(dict(kind="dm", ref=f"{P['a']}>{P['b']}", act=X @ unit(dm), sparse=False))
for q in range(4): units.append(dict(kind="random", ref=q, act=Xu @ unit(rng.standard_normal(d)), sparse=False))
pool = np.array([i for i in range(2000, 100000) if tok.decode([i]).strip().isalpha() and len(tok.decode([i]).strip()) > 3])
pull = lambda ids: unit((WU[ids].astype(np.float64) * g) @ J)
for q in range(4): units.append(dict(kind="randword", ref=q, act=Xu @ unit(pull(rng.choice(pool, 10, replace=False)).mean(0) - pull(rng.choice(pool, 65, replace=False)).mean(0)), sparse=False))
def clip(t): t = " ".join(t.split()); return t if len(t) <= 420 else t[:420] + " …[cut]"
def across(u, mask):                                     # extremes over all prompts (only used for the 'it is just the family' baseline)
    a = u["act"]; idx = np.where(mask)[0]
    if u["sparse"] and (a[idx] == 0).sum() >= 40 and (a[idx] > 0).sum() >= 10:
        on = idx[a[idx] > 0]; return on[np.argsort(-a[on])][:10], rng.choice(idx[a[idx] == 0], 10, replace=False)
    lo, hi = np.percentile(a[idx], [10, 90]); return rng.choice(idx[a[idx] >= hi], 10, replace=False), rng.choice(idx[a[idx] <= lo], 10, replace=False)
def fam_rule(hi_tr, lo_tr, test):                        # predict from the family composition of the induction examples alone
    fh, fl = [lab[i] for i in hi_tr], [lab[i] for i in lo_tr]
    return float(np.mean([0.5 if fh.count(lab[i]) == fl.count(lab[i]) else float((fh.count(lab[i]) > fl.count(lab[i])) == bool(y)) for i, y in test]))
def within(u, mask, npairs, per_fam=2):                  # high / low pairs from the SAME family, so that the family cannot be the rule
    a = u["act"]; cand = []; gsd = a.std() + 1e-12
    for b in beh:
        idx = np.where(mask & (lab == b))[0]
        if u["sparse"] and (a[idx] == 0).sum() >= 2 and (a[idx] > 0).sum() >= 2:
            on = idx[a[idx] > 0]; on = on[np.argsort(-a[on])]; off = rng.permutation(idx[a[idx] == 0])
            cand += [(a[on[k]] / gsd, int(on[k]), int(off[k]), b) for k in range(min(per_fam, len(on), len(off)))]
        elif not u["sparse"] or (a[idx] > 0).all():
            o = idx[np.argsort(-a[idx])]; cand += [((a[o[k]] - a[o[-1 - k]]) / gsd, int(o[k]), int(o[-1 - k]), b) for k in range(per_fam)]
    cand.sort(key=lambda c: -c[0]); return cand[:npairs]
order = rng.permutation(len(units)); key, ind, pred = {}, [], []
for rank, ui in enumerate(order):
    u = units[ui]; uid = f"u{rank + 1:02d}"; hA, lA = across(u, half); hB, lB = across(u, ~half); base = fam_rule(hA, lA, [(i, 1) for i in hB[:8]] + [(i, 0) for i in lB[:8]])
    pI, pT = within(u, half, 10), within(u, ~half, 8)
    if len(pI) < 6 or len(pT) < 5: print("skipped (too little variation inside families):", u["kind"], u["ref"], len(pI), len(pT)); continue
    hiI = [c[1] for c in pI]; loI = [c[2] for c in pI]; test = [(c[1], 1) for c in pT] + [(c[2], 0) for c in pT]; test = [test[j] for j in rng.permutation(len(test))]
    key[uid] = dict(kind=u["kind"], ref=u["ref"], test=[y for _, y in test], test_idx=[i for i, _ in test], across_family_rule_acc=base, within_family_rule_acc=fam_rule(hiI, loI, test), n_pairs_induction=len(pI),
                    gap_induction=float(np.mean([c[0] for c in pI])), gap_test=float(np.mean([c[0] for c in pT])), families_induction=sorted(set(c[3] for c in pI)), families_test=sorted(set(c[3] for c in pT)),
                    active_share=float((u["act"] > 0).mean()) if u["sparse"] else None)
    ind.append(dict(unit=uid, high=[clip(text[i]) for i in rng.permutation(hiI)], low=[clip(text[i]) for i in rng.permutation(loI)])); pred.append(dict(unit=uid, items=[clip(text[i]) for i, _ in test]))
os.makedirs(f"{out}/abs_in", exist_ok=True); os.makedirs(f"{out}/_private", exist_ok=True)
for part in range(2):
    json.dump(ind[part::2], open(f"{out}/abs_in/induce_{part}.json", "w"), ensure_ascii=False, indent=1); json.dump(pred[part::2], open(f"{out}/abs_in/predict_{part}.json", "w"), ensure_ascii=False, indent=1)
os.makedirs(f"{out}/abs_out", exist_ok=True)
json.dump(dict(units=key), open(f"{out}/_private/abs_key.json", "w"), indent=1)
import collections
print("units:", len(key), dict(collections.Counter(v["kind"] for v in key.values())), "| 'just the family' baseline, extremes over all prompts:", {k: round(float(np.mean([v["across_family_rule_acc"] for v in key.values() if v["kind"] == k])), 2) for k in ("sae", "U", "svm", "dm", "random", "randword")},
      "| same baseline for the family-matched test items:", {k: round(float(np.mean([v["within_family_rule_acc"] for v in key.values() if v["kind"] == k])), 2) for k in ("sae", "U", "svm", "dm", "random", "randword")})
print("files:", sorted(os.listdir(f"{out}/abs_in")), "| size of induce_0:", os.path.getsize(f"{out}/abs_in/induce_0.json"), "bytes")

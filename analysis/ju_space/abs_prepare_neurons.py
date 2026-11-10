"""Blind concept test for MLP neurons (same protocol as abs_prepare.py: a rule is named from 10 high and 10 low example prompts,
then applied blind to 16 new prompts; high and low examples always come from the same prompt family).
Units: neurons that carry class differences (chosen by neu_static.py), randomly drawn neurons of blocks 10-21, and as control
the same blocks' neuron space in a randomly rotated basis (random mixtures of neurons instead of single neurons).
Usage: python abs_prepare_neurons.py <scratch_root> <out_dir>"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mim_common import *
root, out = sys.argv[1], sys.argv[2]; rng = np.random.default_rng(21); lab, text, X, F = load_eval(out); n = len(lab); beh = sorted(set(lab))
A = np.load(f"{out}/neu_act.npy", mmap_mode="r"); NI = A.shape[2]; st = json.load(open(f"{out}/neu_static.json")); samp = np.load(f"{out}/neu_act_sample.npy", mmap_mode="r")   # generic positions, for each neuron's own scale
units = []; seen = set()
for key, lst in st["top_carriers"].items():                 # two leading carriers of the class-mean difference per attacked pair
    if not key.endswith("|classmean"): continue
    k = 0
    for l, q, sh in lst:
        if (l, q) in seen or l < 6: continue
        seen.add((l, q)); units.append(dict(kind="neuron_selected", ref=f"L{l}/N{q}", act=np.asarray(A[:, l, q], dtype=np.float64), block=l, q=q)); k += 1
        if k == 2: break
units = units[:12]
for _ in range(6):
    l = int(rng.integers(10, 22)); q = int(rng.integers(0, NI)); units.append(dict(kind="neuron_random", ref=f"L{l}/N{q}", act=np.asarray(A[:, l, q], dtype=np.float64), block=l, q=q))
for _ in range(6):
    l = int(rng.integers(10, 22)); v = unit(rng.standard_normal(NI)); units.append(dict(kind="rotated_neuron_space", ref=f"L{l}/random mixture", act=np.asarray(A[:, l], dtype=np.float32) @ v.astype(np.float32), block=l, q=None, v=v))
half = np.zeros(n, bool)
for b in beh:
    idx = rng.permutation(np.where(lab == b)[0]); half[idx[:len(idx) // 2]] = True
def clip(t): t = " ".join(t.split()); return t if len(t) <= 420 else t[:420] + " …[cut]"
def within(a, mask, npairs, per_fam=2):
    cand = []; gsd = a.std() + 1e-12
    for b in beh:
        idx = np.where(mask & (lab == b))[0]; o = idx[np.argsort(-a[idx])]; cand += [((a[o[k]] - a[o[-1 - k]]) / gsd, int(o[k]), int(o[-1 - k]), b) for k in range(per_fam)]
    cand.sort(key=lambda c: -c[0]); return cand[:npairs]
order = rng.permutation(len(units)); key, ind, pred = {}, [], []
for rank, ui in enumerate(order):
    u = units[ui]; uid = f"n{rank + 1:02d}"; a = u["act"]; pI, pT = within(a, half, 10), within(a, ~half, 8)
    hiI = [c[1] for c in pI]; loI = [c[2] for c in pI]; test = [(c[1], 1) for c in pT] + [(c[2], 0) for c in pT]; test = [test[j] for j in rng.permutation(len(test))]
    if u["q"] is not None: s = np.asarray(samp[:, u["block"], u["q"]], dtype=np.float64)
    else: s = np.asarray(samp[:, u["block"]], dtype=np.float32) @ u["v"].astype(np.float32)
    p3, p97 = np.percentile(s, [3, 97]); hi_vals = a[hiI + [c[1] for c in pT]]                           # are the 'high' examples strong firings by the unit's own standard on generic text?
    key[uid] = dict(kind=u["kind"], ref=u["ref"], test=[y for _, y in test], test_idx=[i for i, _ in test], gap_induction=float(np.mean([c[0] for c in pI])), gap_test=float(np.mean([c[0] for c in pT])),
                    families_induction=sorted(set(c[3] for c in pI)), share_high_examples_above_own_p97=float((hi_vals > p97).mean()), share_high_examples_below_own_p3=float((hi_vals < p3).mean()),
                    share_prompts_above_own_p97=float((a > p97).mean()), own_p3=float(p3), own_p97=float(p97))
    ind.append(dict(unit=uid, high=[clip(text[i]) for i in rng.permutation(hiI)], low=[clip(text[i]) for i in rng.permutation(loI)])); pred.append(dict(unit=uid, items=[clip(text[i]) for i, _ in test]))
os.makedirs(f"{out}/abs_in", exist_ok=True); os.makedirs(f"{out}/abs_out", exist_ok=True); os.makedirs(f"{out}/_private", exist_ok=True)
for part in range(2):
    json.dump(ind[part::2], open(f"{out}/abs_in/induce_n{part}.json", "w"), ensure_ascii=False, indent=1); json.dump(pred[part::2], open(f"{out}/abs_in/predict_n{part}.json", "w"), ensure_ascii=False, indent=1)
json.dump(dict(units=key), open(f"{out}/_private/abs_key_neurons.json", "w"), indent=1)
import collections
print("units:", len(key), dict(collections.Counter(v["kind"] for v in key.values())))
for kd in ("neuron_selected", "neuron_random", "rotated_neuron_space"):
    v = [x for x in key.values() if x["kind"] == kd]; print(f"   {kd}: high examples above the unit's own p97 on generic text: {np.mean([x['share_high_examples_above_own_p97'] for x in v]):.2f}; prompts above own p97 at this token: {np.mean([x['share_prompts_above_own_p97'] for x in v]):.2f}")

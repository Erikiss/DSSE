"""Rating sheets for a blind check of the U-space categories on DSSE prompts: 16 prompts per family, two different orders.
Usage: python rate_prepare.py <out_dir>"""
import json, os, sys
import numpy as np
out = sys.argv[1]; rng = np.random.default_rng(11); items = json.load(open(f"{out}/eval_items.json")); lab = np.array([a for a, _, _ in items]); text = [c for _, _, c in items]
def clip(t): t = " ".join(t.split()); return t if len(t) <= 420 else t[:420] + " …[cut]"
sel = np.concatenate([rng.choice(np.where(lab == b)[0], 16, replace=False) for b in sorted(set(lab))])
os.makedirs(f"{out}/abs_in", exist_ok=True); os.makedirs(f"{out}/abs_out", exist_ok=True); os.makedirs(f"{out}/_private", exist_ok=True)
for r in range(2):
    o = rng.permutation(len(sel)); json.dump([dict(id=f"p{int(sel[j]):04d}", prompt=clip(text[sel[j]])) for j in o], open(f"{out}/abs_in/rate_{r}.json", "w"), ensure_ascii=False, indent=1)
json.dump([int(i) for i in sel], open(f"{out}/_private/rate_idx.json", "w"))
L = [len(" ".join(t.split())) for t in text]; print("prompts:", len(sel), "| prompt length (chars) median", int(np.median(L)), "p90", int(np.percentile(L, 90)), "max", max(L), "| cut at 420:", round(float(np.mean(np.array(L) > 420)), 3))
print("sheet size:", os.path.getsize(f"{out}/abs_in/rate_0.json"), "bytes")

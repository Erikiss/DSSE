"""Resumable wrapper around r1b_l_cal.run: each item's result is pickled as soon as it is done; the final file is written by
r1b_l_cal.run itself (its own gate and provenance logic), fed from the cached item results."""
import os, sys, pickle, time
RB = os.environ.get("R1B_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 6, "analysis", "ju_space", "round1b")
sys.path.insert(0, RB)
import r1b_l_cal as LC
out, cache = sys.argv[1], sys.argv[2]
os.makedirs(cache, exist_ok=True)
reps = dict(a=5, b=20, c=400, d=20, e=5, f=5, g=5, h=20, i=20, j=20, k=20, l=200, m=5, n=20)
for k in "abcdefghijklmn":
    p = os.path.join(cache, k + ".pkl")
    if os.path.exists(p):
        continue
    t = time.time()
    r = LC.ITEMS[k](False, reps[k])
    pickle.dump(r, open(p + ".tmp", "wb")); os.replace(p + ".tmp", p)
    print(f"item {k} done in {time.time()-t:.0f} s, pass={r.get('pass_')}", flush=True)
cached = {k: pickle.load(open(os.path.join(cache, k + ".pkl"), "rb")) for k in "abcdefghijklmn"}
LC.ITEMS = {k: (lambda q, n, k=k: dict(cached[k])) for k in cached}
LC.run(out)
print("written", out, flush=True)

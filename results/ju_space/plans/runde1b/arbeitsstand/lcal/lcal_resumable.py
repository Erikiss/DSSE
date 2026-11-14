"""Usage (repo root): python <this file> results/ju_space/plans/runde1b/l_cal.json <cache dir> [--items abc]
Parallel: one process per item group with OMP_NUM_THREADS=2 (e.g. --items a, --items bc, ...), then once without --items to
write l_cal.json. Delete the cache after any change to r1b_l_cal.py or r1b_l_stats.py.
Resumable wrapper around r1b_l_cal.run: each item's result is pickled as soon as it is done; the final file is written by
r1b_l_cal.run itself (its own gate and provenance logic), fed from the cached item results."""
import os, sys, pickle, time
RB = os.environ.get("R1B_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 6, "analysis", "ju_space", "round1b")
sys.path.insert(0, RB)
import r1b_l_cal as LC
args = sys.argv[1:]
out, cache = args[0], args[1]
ONLY = args[args.index("--items") + 1] if "--items" in args else None   # e.g. --items a : compute only these items (parallel runs)
os.makedirs(cache, exist_ok=True)
reps = dict(a=5, b=20, c=400, d=20, e=5, f=5, g=5, h=20, i=20, j=20, k=20, l=200, m=5, n=20)
for k in (ONLY or "abcdefghijklmn"):
    p = os.path.join(cache, k + ".pkl")
    if os.path.exists(p):
        continue
    t = time.time()
    r = LC.ITEMS[k](False, reps[k])
    pickle.dump(r, open(p + ".tmp", "wb")); os.replace(p + ".tmp", p)
    print(f"item {k} done in {time.time()-t:.0f} s, pass={r.get('pass_')}", flush=True)
if ONLY:
    print("items", ONLY, "cached; run once more without --items to write", out, flush=True); sys.exit(0)
missing = [k for k in "abcdefghijklmn" if not os.path.exists(os.path.join(cache, k + ".pkl"))]
if missing:
    sys.exit(f"missing items {missing}")
cached = {k: pickle.load(open(os.path.join(cache, k + ".pkl"), "rb")) for k in "abcdefghijklmn"}
LC.ITEMS = {k: (lambda q, n, k=k: dict(cached[k])) for k in cached}
LC.run(out)
print("written", out, flush=True)

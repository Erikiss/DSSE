"""Where does the damage of ungrokking sit: in the path through positions 0..3 (what the cache offers) or in the path of the
reading position '=' (what is done with it)?  Positions 0..3 never see position 4, so the forward pass can be cut exactly.
G = a network that generalises; U_t = G after t further training steps on a small part of its training problems; D_t = G after
t further steps on all its training problems (drift control, same restart of the optimiser); M = the memorising state of the same
run before it generalised (reference for a degradation that is spread over all weights: all weights of G are moved towards M
until the held-out accuracy equals that of the last U).
acc(reader, cache) = share of right answers at '=' on the held-out problems when positions 0..3 are computed by one network and
position 4 by the other. Two ways of cutting:
  kv      the cache network delivers keys and values (its states of positions 0..3 and its own key and value weights)
  states  the cache network delivers the states of positions 0..3; keys and values are formed with the reader's weights
For a pair (G, X):  T = acc(G,G) - acc(X,X)   total drop
                    C = acc(G,G) - acc(G,X)   drop carried by the cache path (reader G, cache X)
                    R = acc(G,G) - acc(X,G)   drop carried by the reader path (reader X, cache G)
Decision rule fixed before the runs (plan review): the cache carries the ungrokking if C >= 2 R and C >= T / 2 for both cuts and
every draw, C / R is clearly above that of the reference pair (G, M-mix), both hybrids of the drift pair stay at or above 0.90,
and along the run acc(G, U_t) falls below 0.5 at an earlier checkpoint than acc(U_t, G).
Second part, description of the displacement d = cache(U) - cache(G) of the last U (tensors of keys and values, both blocks):
its mean over problems and the problem-specific rest, sizes per slot relative to the spread of G's cache there, and what G does
when disturbances of the same size are added to its cache for the '=' read:
  own a      a * (the problem's own specific displacement), with the mean displacement added (mean+) or not
  other      the specific displacement of another problem (same size and distribution, wrong problem)
  cov-k      a random draw with the covariance of the specific displacement inside its k main axes (k = 8, 32), scaled to its size
  spread     per slot and head the difference between two problems of G's cache, scaled to the size of the displacement there
  sub-k      a random direction inside the k main axes of the variation of G's cache, then rescaled per slot and head to the
             size of the displacement there (after the rescaling it no longer lies exactly inside these axes)
  iso        per slot and head a random direction of the same length (floor)
The second part describes whatever network is the last selected checkpoint of <run_dir_U>; given the drift run it describes
how far a network moves that keeps generalising (the comparison the displacement of U has to be read against).
Usage: python kv_transplant.py <ckpt_G> <run_dir_U> <run_dir_D> <ckpt_M> <out.json>
       env: NT (1), STEPS (comma-separated checkpoints to use, default all), ONLY=path (first part only), SKIP_MIX=1 (no reference)"""
import glob, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_common import fwd, load, ans, problems, subset, cache, Net, NL

pG, dU, dD, pM, out = sys.argv[1:6]; torch.set_num_threads(int(os.environ.get("NT", "1"))); torch.set_grad_enabled(False); tr, va = problems(); logU = json.load(open(f"{dU}/log.json")); G = load(pG); gen = torch.Generator().manual_seed(99)
STEPS = {int(t) for t in os.environ["STEPS"].split(",")} if os.environ.get("STEPS") else None; ONLY_PATH = os.environ.get("ONLY") == "path"
st_of = lambda d: sorted(s for s in (int(os.path.basename(f)[5:-3]) for f in glob.glob(f"{d}/ckpt_*.pt")) if STEPS is None or s in STEPS); tok = va[:, :6]; z = va[:, 5]; acc = lambda lg: float((ans(lg) == z).float().mean())
lgG, rG = fwd(G, tok); aGG = acc(lgG); kvG = {l: cache(rG, l) for l in range(NL)}; stG = {l: rG[l]["xin"][:, :4] for l in range(NL)}


def pair(X):
    """Path analysis for the pair (G, X) on the held-out problems."""
    lgX, rX = fwd(X, tok); kvX = {l: cache(rX, l) for l in range(NL)}; stX = {l: rX[l]["xin"][:, :4] for l in range(NL)}; r = dict(acc_GG=aGG, acc_XX=acc(lgX), T=aGG - acc(lgX))
    for cut, cG, cX in (("kv", dict(set_kv=kvG), dict(set_kv=kvX)), ("states", dict(set_res=stG), dict(set_res=stX))):
        gx = acc(fwd(G, tok, cX)[0]); xg = acc(fwd(X, tok, cG)[0]); r[cut] = dict(acc_reader_G_cache_X=gx, acc_reader_X_cache_G=xg, C=aGG - gx, R=aGG - xg)
    return r, rX, lgX


res = dict(G=pG, U=dU, D=dD, M=pM, n_held=len(va), sub=logU.get("sub"), U_path=[], D_path=[]); print(f"G: held-out {aGG:.3f}")
for name, d, key in (("U", dU, "U_path"), ("D", dD, "D_path")):
    for st in st_of(d):
        if st == 0: continue
        r, _, _ = pair(load(f"{d}/ckpt_{st}.pt")); r["step"] = st; res[key].append(r)
        print(f"{name} step {st:6d}: own {r['acc_XX']:.3f} | kv: reader G + cache {name} {r['kv']['acc_reader_G_cache_X']:.3f}, reader {name} + cache G {r['kv']['acc_reader_X_cache_G']:.3f} | states: {r['states']['acc_reader_G_cache_X']:.3f}, {r['states']['acc_reader_X_cache_G']:.3f} | T {r['T']:.3f} C {r['kv']['C']:.3f} R {r['kv']['R']:.3f} (states C {r['states']['C']:.3f} R {r['states']['R']:.3f})", flush=True)
if ONLY_PATH: json.dump(res, open(out, "w"), indent=1); print("written", out); sys.exit()
U = load(f"{dU}/ckpt_{st_of(dU)[-1]}.pt"); rU_pair, rU, lgU = pair(U); target = rU_pair["acc_XX"]; sdG, sdM = G.state_dict(), torch.load(pM, weights_only=True)["net"]; best = None; res["displacement_of"] = f"{dU}/ckpt_{st_of(dU)[-1]}.pt"
wd = lambda X: float(sum(((X.state_dict()[k] - sdG[k]) ** 2).sum() for k in sdG if sdG[k].is_floating_point()) ** .5 / sum((sdG[k] ** 2).sum() for k in sdG if sdG[k].is_floating_point()) ** .5); res["weight_distance_from_G"] = wd(U)
for a in ([] if os.environ.get("SKIP_MIX") == "1" else [i / 40 for i in range(1, 41)]):                                # all weights of G moved towards the memorising state M
    X = Net(); X.load_state_dict({k: ((1 - a) * sdG[k] + a * sdM[k]) if sdG[k].is_floating_point() else sdG[k] for k in sdG}); X.eval(); r, _, _ = pair(X); r["a"] = a; res.setdefault("M_mix", []).append(r)
    if best is None or abs(r["acc_XX"] - target) < abs(best["acc_XX"] - target): best = r
if best: res["M_mix_matched"] = best; print(f"reference (all weights towards M, a = {best['a']:.3f}): own {best['acc_XX']:.3f} (target {target:.3f}) | kv C {best['kv']['C']:.3f} R {best['kv']['R']:.3f} | states C {best['states']['C']:.3f} R {best['states']['R']:.3f}")

# ---- description of the displacement of the cache from G to the last U ----
vec = lambda rec: torch.cat([t.reshape(t.shape[0], -1) for l in range(NL) for t in cache(rec, l)], 1).double()         # [B, blocks * (k, v) * heads * slots * dh]
def unvec(V, rec):
    o, i = {}, 0
    for l in range(NL):
        k, v = cache(rec, l); n = k[0].numel(); o[l] = (V[:, i:i + n].view_as(k).float(), V[:, i + n:i + 2 * n].view_as(v).float()); i += 2 * n
    return o
_, rGt = fwd(G, tr[:, :6]); _, rUt = fwd(U, tr[:, :6]); Ct, Dt = vec(rGt), vec(rUt) - vec(rGt); mD = Dt.mean(0); Sd, Vd = torch.linalg.svd(Dt - mD, full_matrices=False)[1:]; Vg = torch.linalg.svd(Ct - Ct.mean(0), full_matrices=False)[2]
C, Dv = vec(rG), vec(rU) - vec(rG); B = len(va); sp = Dv - mD; aU = ans(lgU); grp = lambda V: V.view(B, -1, 32); slots = []
for l in range(NL):                                                     # sizes per slot, relative to the spread of G's cache there
    for pi, part in enumerate("kv"):
        t = (cache(rU, l)[pi] - cache(rG, l)[pi]).double(); g0 = cache(rG, l)[pi].double()
        for p in range(4): slots.append(dict(block=l, part=part, slot=p, mean_shift=float(t[:, :, p].mean(0).norm()), specific=float((t[:, :, p] - t[:, :, p].mean(0)).pow(2).sum((1, 2)).mean().sqrt()), spread_G=float((g0[:, :, p] - g0[:, :, p].mean(0)).pow(2).sum((1, 2)).mean().sqrt())))
ev = Sd ** 2; res["displacement"] = dict(slots=slots, specific_vs_spread_of_G=float((sp ** 2).sum(1).mean().sqrt() / ((C - C.mean(0)) ** 2).sum(1).mean().sqrt()), own_accuracy=rU_pair["acc_XX"], mean_share_of_total=float((mD ** 2).sum() / (Dt ** 2).sum(1).mean()), specific_in_k_axes={k: float(ev[:k].sum() / ev.sum()) for k in (1, 2, 8, 32)}, size_vs_spread_of_G=float((Dv ** 2).sum(1).mean().sqrt() / ((C - C.mean(0)) ** 2).sum(1).mean().sqrt()))
print("displacement:", {k: v for k, v in res["displacement"].items() if k != "slots"})
def go(disp):
    lg = fwd(G, tok, dict(add_kv=unvec(disp, rG)))[0]; return dict(right=acc(lg), as_U=float((ans(lg) == aU).float().mean()), size=float((disp ** 2).sum(1).mean().sqrt() / (sp ** 2).sum(1).mean().sqrt()))
def like(V, ref):                                                       # V rescaled per slot and head to the length of ref
    return (grp(V) * grp(ref).norm(dim=-1, keepdim=True) / grp(V).norm(dim=-1, keepdim=True).clamp_min(1e-12)).view(B, -1)
T = {"mean only": go(mD.expand(B, -1).clone())}
for a in (0.5, 1.0, 2.0):
    T[f"own {a:g}"] = go(a * sp); T[f"own {a:g} mean+"] = go(mD + a * sp); T[f"other {a:g}"] = go(a * sp[torch.randperm(B, generator=gen)])
    for k in (8, 32):
        n = (torch.randn(B, k, generator=gen).double() * Sd[:k] / (len(Dt) - 1) ** .5) @ Vd[:k]; T[f"cov-{k} {a:g}"] = go(a * n * (sp ** 2).sum(1).mean().sqrt() / (n ** 2).sum(1).mean().sqrt())
        u = torch.randn(B, k, generator=gen).double() @ Vg[:k]; T[f"sub-{k} {a:g}"] = go(a * like(u, sp))
    T[f"spread {a:g}"] = go(a * like((C - C.mean(0))[torch.randperm(B, generator=gen)] - (C - C.mean(0)), sp)); T[f"iso {a:g}"] = go(a * like(torch.randn(B, C.shape[1], generator=gen).double(), sp))
res["disorder"] = T; print("disturbance added to G's cache: right answers on held-out (equal to U's answer) [size relative to the specific displacement]")
for k, v in T.items(): print(f"   {k:16s} {v['right']:.3f} ({v['as_U']:.3f}) [{v['size']:.2f}]")
json.dump(res, open(out, "w"), indent=1); print("written", out)

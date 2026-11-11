"""Is the loss of generalisation in the switching pattern of the MLPs? Exchange of the pattern between two networks (README 9.2).
Switch of a hidden unit: ReLU: the mask m = 1[fc1(x) > 0] (unit = m * fc1(x)); SwiGLU: s = sigmoid(gate(x)) (unit = s * gate(x) * up(x)).
A bilinear unit has no switch and is not treated here.
G is a generalising network, U its continuation on 5 % of the training problems, D the drift control (all problems), REF a
reference for damage spread over all weights: all weights of G moved towards the memorising state M of the same base run
(step 5000) until the held-out accuracy is within 0.02 of U's. On the held-out problems, read at '=', with the pattern of both
MLPs exchanged at all positions:
  own                accuracy of the network itself
  N_pattG, G_pattN   N's weights with the pattern of G's own forward pass; G's weights with the pattern of N's forward pass
  lost = own(G) - own(U);   r = (U_pattG - own(U)) / lost  (what the pattern of G gives back);   k = (own(G) - G_pattU) / lost
  r_ref, k_ref       the same two numbers for REF (with its own loss)
  readable           a step counts only if D is at 0.99 or above, G with the pattern of D and D with the pattern of G are both at
                     0.95 or above, and lost is at least 0.015 (a hybrid of two networks can fail merely because they have trained apart)
  flip               per block at '=': share of (problem, unit) pairs whose switch differs between G and N (ReLU: mask differs;
                     SwiGLU: the gate input has another sign), share of units switched on (ReLU) or with a positive gate input
Usage: python act_masks.py <ckpt_G> <ckpt_M> <run_dir_U> <run_dir_D> <out.json>      env: STEPS (comma list; default: all checkpoints of U that D has too)
  run_dir_U and run_dir_D may each be a comma list of run directories (for example the fine run to step 1500 and the long run to
  4000); a step is read from the first directory in the list that has its checkpoint"""
import glob, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_common import fwd, load, make_net, problems, acc4, NL

ckG, ckM, dU, dD, out = sys.argv[1:6]; torch.set_num_threads(int(os.environ.get("NT", "1"))); tr, va = problems(); seq = va; tok = seq[:, :5]
have = lambda ds: set(int(os.path.basename(f)[5:-3]) for d in ds.split(",") for f in glob.glob(f"{d}/ckpt_*.pt")); steps = sorted(have(dU) & have(dD))
ck = lambda ds, st: next(f"{d}/ckpt_{st}.pt" for d in ds.split(",") if os.path.exists(f"{d}/ckpt_{st}.pt")); steps = [int(s) for s in os.environ["STEPS"].split(",")] if os.environ.get("STEPS") else [s for s in steps if s > 0]
G = load(ckG); kind = "relu" if not hasattr(G.blocks[0], "gate") else ("bilinear" if G.blocks[0].bilinear else "swiglu"); assert kind in ("relu", "swiglu"), "no switch in a bilinear unit"


def own(net):
    """Accuracy and the switching pattern (ReLU: masks; SwiGLU: sigmoid of the gate input) of a network's own forward pass."""
    with torch.no_grad(): lg, rec = fwd(net, tok, dict(rec_gate=True))
    return acc4(lg, seq), [(r["gf"] if kind == "relu" else torch.sigmoid(r["gi"])) for r in rec]


def with_pattern(net, patt):
    sg = {l: (patt[l] if kind == "relu" else (lambda a, u, g, s=patt[l]: s * g)) for l in range(NL)}
    with torch.no_grad(): return acc4(fwd(net, tok, dict(set_gate=sg))[0], seq)


def flips(a, b):
    on = (lambda t: t > 0.5); return [dict(differ=float((on(a[l][:, 4]) != on(b[l][:, 4])).float().mean()), on_in_G=float(on(a[l][:, 4]).float().mean()), on_in_N=float(on(b[l][:, 4]).float().mean())) for l in range(NL)]


def mixed(a):
    sdG, sdM = G.state_dict(), M.state_dict(); X = make_net(kind); X.load_state_dict({k: ((1 - a) * sdG[k] + a * sdM[k]) if sdG[k].is_floating_point() else sdG[k] for k in sdG}); X.eval(); return X


def reference(target):
    """All weights of G moved towards M until the accuracy is within 0.02 of the target (bisection; None if that is not reached)."""
    lo, hi, best = 0.0, 1.0, None
    for _ in range(14):
        a = (lo + hi) / 2; X = mixed(a); acc, patt = own(X)
        if best is None or abs(acc - target) < abs(best[1] - target): best = (a, acc, X, patt)
        if abs(acc - target) <= 0.005: break
        if acc > target: lo = a
        else: hi = a
    return best if abs(best[1] - target) <= 0.02 else None


M = load(ckM); accG, pG = own(G); res = dict(G=ckG, M=ckM, U=dU, D=dD, kind=kind, n_held=len(seq), own_G=accG, steps=[])
for st in steps:
    U, D = load(ck(dU, st)), load(ck(dD, st)); aU, pU = own(U); aD, pD = own(D); row = dict(step=st, own_U=aU, own_D=aD, U_pattG=with_pattern(U, pG), G_pattU=with_pattern(G, pU), D_pattG=with_pattern(D, pG), G_pattD=with_pattern(G, pD), flip_U=flips(pG, pU), flip_D=flips(pG, pD)); lost = accG - aU
    row["lost"] = lost; row["readable"] = bool(aD >= 0.99 and row["D_pattG"] >= 0.95 and row["G_pattD"] >= 0.95 and lost >= 0.015)
    if lost >= 0.015:
        row["r"] = (row["U_pattG"] - aU) / lost; row["k"] = (accG - row["G_pattU"]) / lost; ref = reference(aU)
        if ref is not None:
            a, aR, R, pR = ref; lr = accG - aR; row["ref"] = dict(a=a, own=aR, R_pattG=with_pattern(R, pG), G_pattR=with_pattern(G, pR), flip=flips(pG, pR)); row["r_ref"] = (row["ref"]["R_pattG"] - aR) / lr; row["k_ref"] = (accG - row["ref"]["G_pattR"]) / lr
    res["steps"].append(row)
    print(f"step {st:5d} U {aU:.3f} D {aD:.3f} | U_pattG {row['U_pattG']:.3f} G_pattU {row['G_pattU']:.3f} | D_pattG {row['D_pattG']:.3f} G_pattD {row['G_pattD']:.3f} | readable {row['readable']}" + (f" | r {row['r']:.2f} k {row['k']:.2f}" if "r" in row else "") + (f" | ref own {row['ref']['own']:.3f}: r_ref {row['r_ref']:.2f} k_ref {row['k_ref']:.2f}" if "ref" in row else "")
          + " | differ U " + "/".join(f"{x['differ']:.3f}" for x in row["flip_U"]) + " D " + "/".join(f"{x['differ']:.3f}" for x in row["flip_D"]), flush=True)
    json.dump(res, open(out, "w"), indent=1)
json.dump(res, open(out, "w"), indent=1)

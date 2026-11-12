"""RGI toy pilot (design stage only): can a new task's input column learned in model A be carried into model B of the same
world (same 12 pretraining shifts, same new shifts) through a map fitted on shared structure, and does that grow with delta?
World and training exactly as pc_meta.py (shift tasks (x + y + c) mod 17, one-hot task slot, two tanh layers, AdamW, full batch,
checkpoint 500 updates after held-out >= 0.95). Model identity: (arm, W, world seed ws, init seed i, gamma); init seed 0
reproduces pc_meta.py's init (torch.manual_seed(ws)); init i > 0 uses torch.manual_seed(10000 i + ws).
Improvement = the new task's input column v (only that column trained from zero, 8 shots, 300 Adam steps, lr 1e-2), as in B2.
Channels A -> B:
  T1  task-anchor map: affine ridge through the 12 pretraining task columns (v_B = mean_B + Cb alpha, alpha = ridge coefficients of
      v_A - mean_A on the centred anchors Ca); lambda by leave-one-anchor-out column error.
  T2  first-layer map through all 46 input columns (17 x, 17 y, 12 tasks) and the bias (ridge, no centring).
  T3  h2 stitching: affine ridge map S2 from A's h2 to B's h2 on 12 tasks x 289 pairs; v_B trained (300 Adam steps from zero) so
      that B's h2 on the new task's 289 inputs matches S2 h2_A (no labels).
  T4  logit distillation on the new task's 289 inputs (no labels): functional reference, not geometric.
Nulls: T1 with B's anchors permuted (50 permutations), the mean anchor column, zero column, raw copy (same width only).
Reference: B's own column (8 shots, 300 steps). LOO-anchor transfer: each pretraining task k carried by T1 fitted on the other 11.
Geometry per model: delta / ring of pc_meta.delta_stats, Fourier concentration of the anchors (share of the centred anchor variance
on the best circle frequency f, minus the shift-permutation null), in-span share of the new columns, zero-shot ring synthesis.
Usage: python rgi_pilot.py <ws> <spec> [<spec> ...]   spec = arm:W:init:gamma"""
import sys, os, json, time, itertools
import numpy as np, torch, torch.nn.functional as Fn
SP = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
sys.path.insert(0, SP + "/ufo_scripts")
torch.set_num_threads(1)
import pc_meta as M
from pc_common import MLP, accuracy
P, NPRE, NNEW, NSLOT, BASE = M.P, M.NPRE, M.NNEW, M.NSLOT, M.BASE
OUT = SP + "/rgi_design/pilot"; os.makedirs(OUT + "/models", exist_ok=True)
NSHOT, NPERM = 8, 50


def mk(arm, W):
    return MLP(2 * P + NSLOT, W, P, param="sp" if arm == "bp_sp" else "mup", base_width=BASE)


GDELAY = os.environ.get("GDELAY") == "1"     # apply the delta regulariser only after held-out first reaches 0.95 (the 500 updates to the checkpoint)


def pretrain(ws, arm, W, init, gamma=0.0):
    name = f"{arm}_W{W}_w{ws}_i{init}_g{gamma:g}" + ("_late" if GDELAY and gamma else ""); path = f"{OUT}/models/{name}.pt"
    if os.path.exists(path):
        d = torch.load(path); m = mk(arm, W); m.load_state_dict(d["state"]); return m, d["info"]
    pre, new, pairs, splits = M.world(ws)
    tr = [M.data(pre[k], k, pairs[splits[k][:144]]) for k in range(NPRE)]; va = [M.data(pre[k], k, pairs[splits[k][144:]]) for k in range(NPRE)]
    Xtr = torch.cat([d[0] for d in tr]); ytr = torch.cat([d[1] for d in tr]); Xva = torch.cat([d[0] for d in va]); yva = torch.cat([d[1] for d in va])
    lab = torch.arange(NPRE).repeat_interleave(144)
    torch.manual_seed(ws if init == 0 else 10_000 * init + ws); m = mk(arm, W); opt = M.make_opt(m)
    first = ck = None; t0 = time.time()
    for s in range(1, M.MAXSTEPS + M.AFTER + 1):
        if s > M.MAXSTEPS and first is None: break
        M.bp_update(m, opt, Xtr, ytr, gamma if (not GDELAY or first is not None) else 0.0, lab)
        if s % 50 == 0 and first is None and accuracy(m, Xva, yva) >= M.LEVEL1: first = s
        if first is not None and s >= first + M.AFTER: ck = s; break
    info = dict(name=name, arm=arm, W=W, ws=ws, init=init, gamma=gamma, first95=first, ckpt=ck, heldout=accuracy(m, Xva, yva), sec=time.time() - t0)
    torch.save(dict(state=m.state_dict(), info=info), path); return m, info


# ---------------- fast evaluation of a candidate column (or many) in a model
def heads(m, z1):
    with torch.no_grad():
        h1 = torch.tanh(z1); h2 = torch.tanh(m.l2(h1)); return m.head(h2)


def base_z1(m, xy):
    """Pre-activation of layer 1 without any slot column: W1[:, x] + W1[:, P + y] + b1, [N, W]."""
    Wt = m.l1.weight.detach(); return (Wt[:, torch.as_tensor(xy[:, 0])] + Wt[:, P + torch.as_tensor(xy[:, 1])]).T + m.l1.bias.detach()


def acc_cols(m, xy, y, V):
    """Held-out accuracy for each candidate column in V [n, W] (the slot's column replaced by it)."""
    z = base_z1(m, xy); out = []
    for v in V:
        out.append(float((heads(m, z + v).argmax(-1) == y).float().mean()))
    return np.array(out)


# ---------------- maps
def ridge_coef(Ca, v, r):
    G = Ca.T @ Ca; lam = r * np.trace(G) / G.shape[0]; return np.linalg.solve(G + lam * np.eye(G.shape[0]), Ca.T @ v)


def t1_map(CA, CB, vA, r):
    mA, mB = CA.mean(1), CB.mean(1); a = ridge_coef(CA - mA[:, None], vA - mA, r); return mB + (CB - mB[:, None]) @ a, a


def t1_choose_r(CA, CB, grid=(1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)):
    best = None
    for r in grid:
        err = 0.0
        for k in range(CA.shape[1]):
            keep = [i for i in range(CA.shape[1]) if i != k]; pred, _ = t1_map(CA[:, keep], CB[:, keep], CA[:, k], r)
            ref = CB[:, keep].mean(1); err += np.sum((pred - CB[:, k]) ** 2) / np.sum((CB[:, k] - ref) ** 2)
        if best is None or err < best[1]: best = (r, err / CA.shape[1])
    return best


def t2_map(AA, AB, vA, r):
    a = ridge_coef(AA, vA, r); return AB @ a


def t2_choose_r(AA, AB, task_idx, grid=(1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)):
    best = None
    for r in grid:
        err = 0.0
        for k in task_idx:
            keep = [i for i in range(AA.shape[1]) if i != k]; pred = t2_map(AA[:, keep], AB[:, keep], AA[:, k], r)
            err += np.sum((pred - AB[:, k]) ** 2) / np.sum(AB[:, k] ** 2)
        if best is None or err < best[1]: best = (r, err / len(task_idx))
    return best


# ---------------- geometry
def fourier_conc(C, shifts, nperm=500, seed=0):
    """Share of the centred anchor variance captured by the best single circle frequency f (2 regressors cos, sin of 2 pi f c / 17,
    centred over the 12 shifts); also the best pair of frequencies; null by permuting which shift belongs to which column."""
    X = (C - C.mean(1, keepdims=True)).T                                     # 12 x W
    tot = np.sum(X ** 2)

    def r2_f(sh):
        out = []
        for f in range(1, 9):
            Phi = np.stack([np.cos(2 * np.pi * f * sh / P), np.sin(2 * np.pi * f * sh / P)], 1); Phi = Phi - Phi.mean(0)
            Q, _ = np.linalg.qr(Phi); out.append(np.sum((Q.T @ X) ** 2) / tot)
        return np.array(out)
    sh = np.array(shifts, float); r = r2_f(sh); g = np.random.default_rng(seed)
    null = np.array([r2_f(g.permutation(sh)).max() for _ in range(nperm)])
    fs = np.argsort(-r)
    Phi2 = np.concatenate([np.stack([np.cos(2 * np.pi * (f + 1) * sh / P), np.sin(2 * np.pi * (f + 1) * sh / P)], 1) for f in fs[:2]], 1); Phi2 -= Phi2.mean(0)
    Q2, _ = np.linalg.qr(Phi2); r2pair = np.sum((Q2.T @ X) ** 2) / tot
    return dict(r2=r.tolist(), fstar=int(fs[0] + 1), r2max=float(r.max()), null_mean=float(null.mean()), null_p95=float(np.quantile(null, 0.95)),
                excess=float(r.max() - null.mean()), r2pair=float(r2pair), spectrum=r / r.sum())


def ring_synth(C, shifts, c_new, nf=2):
    """Zero-shot column for shift c_new from the anchors' best nf circle frequencies (least squares over the 12 shifts)."""
    sh = np.array(shifts, float); X = C.T; r = fourier_conc(C, shifts, nperm=1)["r2"]; fs = np.argsort(-np.array(r))[:nf] + 1

    def phi(s):
        s = np.atleast_1d(np.array(s, float)); return np.concatenate([np.ones((len(s), 1))] + [np.stack([np.cos(2 * np.pi * f * s / P), np.sin(2 * np.pi * f * s / P)], 1) for f in fs], 1)
    B, *_ = np.linalg.lstsq(phi(sh), X, rcond=None); return (phi(c_new) @ B)[0]


def main():
    ws = int(sys.argv[1]); specs = [s.split(":") for s in sys.argv[2:]]
    pre, new, pairs, splits = M.world(ws); models = {}; t0 = time.time()
    for arm, W, init, gamma in specs:
        W, init, gamma = int(W), int(init), float(gamma); m, info = pretrain(ws, arm, W, init, gamma)
        key = info["name"]; print(f"{key}: first95 {info['first95']} ckpt {info['ckpt']} held-out {info['heldout']:.3f} ({info['sec']:.0f}s)", flush=True)
        if info["ckpt"] is None: continue
        g = dict(info=info, m=m, arm=arm, W=W)
        g["delta"] = M.delta_stats(m, pre, pairs, ws)
        C = m.l1.weight.detach().numpy()[:, 2 * P: 2 * P + NPRE].astype(float); g["C"] = C
        g["Aall"] = np.concatenate([m.l1.weight.detach().numpy()[:, : 2 * P + NPRE], m.l1.bias.detach().numpy()[:, None]], 1).astype(float)
        fc = fourier_conc(C, pre); g["fourier"] = fc
        # new-task columns (own learning) and their quality
        g["v"], g["self"], g["self_curve"], g["span"], g["synth"] = [], [], [], [], []
        for j in range(NNEW):
            acc, mm = M.finetune(m, arm, W, new, pairs, splits, j, NSHOT, "col", ws); v = mm.l1.weight.detach().numpy()[:, 2 * P + NPRE + j].astype(float)
            g["v"].append(v); g["self"].append(acc[-1]); g["self_curve"].append(acc)
            xy = pairs[splits[NPRE + j][8:]]; yh = torch.as_tensor((xy[:, 0] + xy[:, 1] + new[j]) % P)
            mA = C.mean(1); Ca = C - mA[:, None]; Q, _ = np.linalg.qr(Ca); vp = mA + Q @ (Q.T @ (v - mA))
            share = float(np.sum((Q.T @ (v - mA)) ** 2) / np.sum((v - mA) ** 2))
            a_proj = acc_cols(m, xy, yh, torch.tensor(np.stack([vp]), dtype=torch.float32))[0]
            vs = ring_synth(C, pre, new[j]); a_syn = acc_cols(m, xy, yh, torch.tensor(np.stack([vs]), dtype=torch.float32))[0]
            g["span"].append((share, a_proj)); g["synth"].append(a_syn)
        d2 = g["delta"]["h2"]; d1 = g["delta"]["h1"]
        print(f"   delta h1 {d1['delta']:.3f} h2 {d2['delta']:.3f} ctrl h2 {d2['delta_ctrl']:.3f} ring(f=1) h2 {d2['ring']:+.2f}; fourier f* {fc['fstar']} r2 {fc['r2max']:.2f} "
              f"(null {fc['null_mean']:.2f}, p95 {fc['null_p95']:.2f}) pair {fc['r2pair']:.2f}; self {np.mean(g['self']):.3f}; "
              f"span share {np.mean([s for s, _ in g['span']]):.2f} acc proj {np.mean([a for _, a in g['span']]):.3f}; synth {np.mean(g['synth']):.3f}", flush=True)
        models[key] = g
    keys = list(models); res = []
    # pre-computed held-out sets
    newsets = []
    for j in range(NNEW):
        xy = pairs[splits[NPRE + j][8:]]; newsets.append((xy, torch.as_tensor((xy[:, 0] + xy[:, 1] + new[j]) % P)))
    presets = []
    for k in range(NPRE):
        xy = pairs[splits[k][144:]]; presets.append((xy, torch.as_tensor((xy[:, 0] + xy[:, 1] + pre[k]) % P)))
    allxy = pairs
    rng = np.random.default_rng(123 + ws)
    perms = [rng.permutation(NPRE) for _ in range(NPERM)]
    perms = [p for p in perms if np.any(p != np.arange(NPRE))]
    for ka, kb in itertools.permutations(keys, 2):
        A, B = models[ka], models[kb]; r1, e1 = t1_choose_r(A["C"], B["C"]); r2, e2 = t2_choose_r(A["Aall"], B["Aall"], list(range(2 * P, 2 * P + NPRE)))
        row = dict(A=ka, B=kb, r1=r1, loo_err1=e1, r2=r2, loo_err2=e2)
        # LOO-anchor transfer
        loo, loo_sh = [], []
        for k in range(NPRE):
            keep = [i for i in range(NPRE) if i != k]; pred, _ = t1_map(A["C"][:, keep], B["C"][:, keep], A["C"][:, k], r1)
            xy, yy = presets[k]; m = B["m"]; colk = 2 * P + k
            # evaluate with B's column k replaced: base without column + candidate
            z = base_z1(m, xy)
            cands = [pred] + [t1_map(A["C"][:, keep], B["C"][:, keep][:, np.random.default_rng(k * 1000 + q).permutation(NPRE - 1)], A["C"][:, k], r1)[0] for q in range(20)]
            accs = []
            for v in cands:
                accs.append(float((heads(m, z + torch.tensor(v, dtype=torch.float32)).argmax(-1) == yy).float().mean()))
            loo.append(accs[0]); loo_sh.append(np.mean(accs[1:]))
        row.update(loo=float(np.mean(loo)), loo_shuf=float(np.mean(loo_sh)))
        # new-task transfer
        t1a, t2a, sh_a, mean_a, raw_a, t3a, t4a, selfB, srcA, fid = [], [], [], [], [], [], [], [], [], []
        for j in range(NNEW):
            xy, yy = newsets[j]; vA = A["v"][j]
            vB1, alpha = t1_map(A["C"], B["C"], vA, r1); vB2 = t2_map(A["Aall"], B["Aall"], vA, r2)
            cands = [vB1, vB2, B["C"].mean(1)] + [t1_map(A["C"], B["C"][:, p], vA, r1)[0] for p in perms]
            if A["W"] == B["W"]: cands.append(vA)
            acc = acc_cols(B["m"], xy, yy, torch.tensor(np.stack(cands), dtype=torch.float32))
            t1a.append(acc[0]); t2a.append(acc[1]); mean_a.append(acc[2]); sh_a.append(np.mean(acc[3:3 + len(perms)]))
            raw_a.append(acc[-1] if A["W"] == B["W"] else np.nan)
            selfB.append(B["self"][j]); srcA.append(A["self"][j])
        row.update(t1=float(np.mean(t1a)), t2=float(np.mean(t2a)), shuf=float(np.mean(sh_a)), meancol=float(np.mean(mean_a)), raw=float(np.nanmean(raw_a)) if A["W"] == B["W"] else None,
                   selfB=float(np.mean(selfB)), selfA=float(np.mean(srcA)), t1_tasks=[float(x) for x in t1a], self_tasks=[float(x) for x in selfB])
        res.append(row)
        print(f"{ka:>26} -> {kb:<26} r1 {r1:g} loo-err {e1:.2f} | LOO-anchor {row['loo']:.3f} (shuf {row['loo_shuf']:.3f}) | new: T1 {row['t1']:.3f} T2 {row['t2']:.3f} shuf {row['shuf']:.3f} "
              f"mean {row['meancol']:.3f} raw {row['raw'] if row['raw'] is None else round(row['raw'], 3)} | selfA {row['selfA']:.3f} selfB {row['selfB']:.3f}", flush=True)
    summ = {k: dict(info=g["info"], delta=g["delta"], fourier={kk: vv for kk, vv in g["fourier"].items() if kk != "spectrum"}, spectrum=g["fourier"]["spectrum"].tolist(),
                    self=g["self"], self_curve=g["self_curve"], span=g["span"], synth=g["synth"]) for k, g in models.items()}
    json.dump(dict(models=summ, pairs=res), open(f"{OUT}/pilot_w{ws}_{int(time.time())}.json", "w"), default=float)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

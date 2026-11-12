"""Shared functions of the RGI toy design pilots (see rgi_pilot.py for the definitions)."""
import sys, os, time
import numpy as np, torch
SP = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
sys.path.insert(0, SP + "/ufo_scripts"); sys.path.insert(0, SP + "/rgi_design")
torch.set_num_threads(1)
import pc_meta as M
from pc_common import MLP, accuracy
from rgi_pilot import mk, heads, base_z1, acc_cols, ridge_coef, t1_map, t1_choose_r, t2_map, t2_choose_r, fourier_conc, ring_synth
P, NPRE, NNEW, NSLOT = M.P, M.NPRE, M.NNEW, M.NSLOT


def xy_sparsity(m):
    W1 = m.l1.weight.detach().numpy().astype(float); out = []
    for Xc in (W1[:, :P], W1[:, P:2 * P]):
        F = np.fft.rfft(Xc - Xc.mean(1, keepdims=True), axis=1); pw = (np.abs(F) ** 2).sum(0)[1:]; pw = pw / pw.sum(); out.append(float(np.sort(pw)[-2:].sum()))
    return float(np.mean(out))                                     # share of the best 2 of 8 frequencies; uniform = 0.25


def geometry(m, arm, W, ws, nshot=8, with_delta=True):
    pre, new, pairs, splits = M.world(ws)
    g = dict(m=m, arm=arm, W=W)
    if with_delta: g["delta"] = M.delta_stats(m, pre, pairs, ws)
    C = m.l1.weight.detach().numpy()[:, 2 * P: 2 * P + NPRE].astype(float); g["C"] = C
    g["Aall"] = np.concatenate([m.l1.weight.detach().numpy()[:, : 2 * P + NPRE], m.l1.bias.detach().numpy()[:, None]], 1).astype(float)
    fc = fourier_conc(C, pre, nperm=200); g["fourier"] = {k: v for k, v in fc.items() if k != "spectrum"}; g["spectrum"] = fc["spectrum"]
    g["xy_sparsity"] = xy_sparsity(m)
    g["v"], g["self"], g["span"], g["proj"], g["synth"] = [], [], [], [], []
    for j in range(NNEW):
        acc, mm = M.finetune(m, arm, W, new, pairs, splits, j, nshot, "col", ws); v = mm.l1.weight.detach().numpy()[:, 2 * P + NPRE + j].astype(float)
        g["v"].append(v); g["self"].append(acc[-1])
        xy = pairs[splits[NPRE + j][8:]]; yh = torch.as_tensor((xy[:, 0] + xy[:, 1] + new[j]) % P)
        mA = C.mean(1); Q, _ = np.linalg.qr(C - mA[:, None]); vp = mA + Q @ (Q.T @ (v - mA))
        g["span"].append(float(np.sum((Q.T @ (v - mA)) ** 2) / np.sum((v - mA) ** 2)))
        vs = ring_synth(C, pre, new[j])
        a = acc_cols(m, xy, yh, torch.tensor(np.stack([vp, vs]), dtype=torch.float32)); g["proj"].append(float(a[0])); g["synth"].append(float(a[1]))
    return g


def pair_transfer(A, B, ws, nperm=30, nloo_perm=10):
    pre, new, pairs, splits = M.world(ws)
    r1, e1 = t1_choose_r(A["C"], B["C"]); r2, e2 = t2_choose_r(A["Aall"], B["Aall"], list(range(2 * P, 2 * P + NPRE)))
    row = dict(r1=r1, loo_err1=e1, r2=r2, loo_err2=e2)
    loo, loo_sh = [], []
    for k in range(NPRE):
        keep = [i for i in range(NPRE) if i != k]; xy = pairs[splits[k][144:]]; yy = torch.as_tensor((xy[:, 0] + xy[:, 1] + pre[k]) % P)
        cands = [t1_map(A["C"][:, keep], B["C"][:, keep], A["C"][:, k], r1)[0]]
        cands += [t1_map(A["C"][:, keep], B["C"][:, keep][:, np.random.default_rng(k * 1000 + q).permutation(NPRE - 1)], A["C"][:, k], r1)[0] for q in range(nloo_perm)]
        a = acc_cols(B["m"], xy, yy, torch.tensor(np.stack(cands), dtype=torch.float32)); loo.append(a[0]); loo_sh.append(a[1:].mean())
    row.update(loo=float(np.mean(loo)), loo_shuf=float(np.mean(loo_sh)))
    rng = np.random.default_rng(123 + ws); perms = [p for p in (rng.permutation(NPRE) for _ in range(nperm)) if np.any(p != np.arange(NPRE))]
    t1a, t2a, sha, raw = [], [], [], []
    for j in range(NNEW):
        xy = pairs[splits[NPRE + j][8:]]; yy = torch.as_tensor((xy[:, 0] + xy[:, 1] + new[j]) % P); vA = A["v"][j]
        cands = [t1_map(A["C"], B["C"], vA, r1)[0], t2_map(A["Aall"], B["Aall"], vA, r2)] + [t1_map(A["C"], B["C"][:, p], vA, r1)[0] for p in perms]
        if A["W"] == B["W"]: cands.append(vA)
        a = acc_cols(B["m"], xy, yy, torch.tensor(np.stack(cands), dtype=torch.float32))
        t1a.append(a[0]); t2a.append(a[1]); sha.append(a[2:2 + len(perms)].mean()); raw.append(a[-1] if A["W"] == B["W"] else np.nan)
    row.update(t1=float(np.mean(t1a)), t2=float(np.mean(t2a)), shuf=float(np.mean(sha)), raw=float(np.nanmean(raw)) if A["W"] == B["W"] else None,
               selfA=float(np.mean(A["self"])), selfB=float(np.mean(B["self"])))
    return row


def fmt_geom(name, g):
    d = g.get("delta", {}); s = f"{name}: "
    if d: s += f"delta h1 {d['h1']['delta']:.3f} h2 {d['h2']['delta']:.3f} | "
    f = g["fourier"]
    return s + (f"task-col fourier f* {f['fstar']} r2 {f['r2max']:.2f} (null {f['null_mean']:.2f}, p95 {f['null_p95']:.2f}) | xy top2 {g['xy_sparsity']:.2f} | self {np.mean(g['self']):.3f} "
                f"span {np.mean(g['span']):.2f} proj {np.mean(g['proj']):.3f} synth {np.mean(g['synth']):.3f}")


def fmt_pair(a, b, r):
    return (f"{a:>24} -> {b:<24} LOO-anchor {r['loo']:.3f} (shuf {r['loo_shuf']:.3f}, col err {r['loo_err1']:.2f}) | new T1 {r['t1']:.3f} T2 {r['t2']:.3f} shuf {r['shuf']:.3f} "
            f"raw {('%.3f' % r['raw']) if r['raw'] is not None else '-'} | selfA {r['selfA']:.3f} selfB {r['selfB']:.3f}")


# ---------------- channels T3 (h2 stitching, no labels) and T4 (logit distillation, no labels)
def all_h2(m, ws):
    pre, new, pairs, splits = M.world(ws); X = torch.cat([M.enc(pairs, k) for k in range(NPRE)])
    with torch.no_grad(): return m.acts(X)[1].double().numpy()


def stitch_fit(HA, HB, npair=P * P, grid=(1e-4, 1e-3, 1e-2, 1e-1)):
    """Affine ridge map h2_A -> h2_B on the 12 pretraining tasks; lambda by leave-3-tasks-out R2 (4 folds); returns S and the
    held-out-task R2 (the h2 stitching quality on unseen tasks, a geometry measure in itself)."""
    XA = np.concatenate([HA, np.ones((len(HA), 1))], 1); folds = [list(range(i, i + 3)) for i in range(0, NPRE, 3)]; best = None
    for r in grid:
        sr = st = 0.0
        for f in folds:
            te = np.concatenate([np.arange(k * npair, (k + 1) * npair) for k in f]); trn = np.setdiff1d(np.arange(len(HA)), te)
            G = XA[trn].T @ XA[trn]; lam = r * np.trace(G) / G.shape[0]; S = np.linalg.solve(G + lam * np.eye(G.shape[0]), XA[trn].T @ HB[trn])
            sr += np.sum((XA[te] @ S - HB[te]) ** 2); st += np.sum((HB[te] - HB[trn].mean(0)) ** 2)
        if best is None or 1 - sr / st > best[1]: best = (r, 1 - sr / st)
    G = XA.T @ XA; lam = best[0] * np.trace(G) / G.shape[0]; return np.linalg.solve(G + lam * np.eye(G.shape[0]), XA.T @ HB), best


def _fit_col(mB, zB, lossfn, W, steps=300, lr=1e-2):
    for p in mB.parameters(): p.requires_grad_(False)
    v = torch.zeros(W, requires_grad=True); opt = torch.optim.Adam([v], lr=lr, betas=(0.9, 0.98))
    for _ in range(steps):
        opt.zero_grad(); h2 = torch.tanh(mB.l2(torch.tanh(zB + v))); L = lossfn(h2); L.backward(); opt.step()
    for p in mB.parameters(): p.requires_grad_(True)
    return v.detach().numpy().astype(float)


def t34_transfer(A, B, ws, S, j):
    """Returns (v_B by T3, v_B by T4) for new task j; both use the new task's 289 inputs without labels."""
    pre, new, pairs, splits = M.world(ws); slot_v = torch.tensor(A["v"][j], dtype=torch.float32)
    zA = base_z1(A["m"], pairs) + slot_v
    with torch.no_grad():
        hA = torch.tanh(A["m"].l2(torch.tanh(zA))); lpA = Fn.log_softmax(A["m"].head(hA), -1)
    tgt = torch.tensor(np.concatenate([hA.double().numpy(), np.ones((len(pairs), 1))], 1) @ S, dtype=torch.float32)
    zB = base_z1(B["m"], pairs); mB = B["m"]
    v3 = _fit_col(mB, zB, lambda h2: ((h2 - tgt) ** 2).sum(-1).mean(), B["W"])
    v4 = _fit_col(mB, zB, lambda h2: Fn.kl_div(Fn.log_softmax(mB.head(h2), -1), lpA, log_target=True, reduction="batchmean"), B["W"])
    return v3, v4


import torch.nn.functional as Fn


# ---------------- span-constrained improvement: the new task learned in the coordinates of the 12 anchors (expressivity) and
# the same coordinates written into B (sharedness)
def learn_alpha(g, ws, j, nshot=8, steps=300, lr=1e-2):
    pre, new, pairs, splits = M.world(ws); slot = NPRE + j; idx = splits[slot]; xs = pairs[idx[:nshot]]; ys = torch.as_tensor((xs[:, 0] + xs[:, 1] + new[j]) % P)
    C = torch.tensor(g["C"], dtype=torch.float32); mA = C.mean(1); Ct = C - mA[:, None]; z = base_z1(g["m"], xs); m = g["m"]
    for p_ in m.parameters(): p_.requires_grad_(False)
    a = torch.zeros(NPRE, requires_grad=True); opt = torch.optim.Adam([a], lr=lr, betas=(0.9, 0.98))
    for _ in range(steps):
        opt.zero_grad(); v = mA + Ct @ a; L = Fn.cross_entropy(m.head(torch.tanh(m.l2(torch.tanh(z + v)))), ys); L.backward(); opt.step()
    for p_ in m.parameters(): p_.requires_grad_(True)
    return a.detach().numpy().astype(float)


def span_transfer(A, B, ws, nperm=30):
    """Per new task: acc in A of the span-learned improvement, acc in B of the same coordinates, shuffled-correspondence null."""
    pre, new, pairs, splits = M.world(ws); rng = np.random.default_rng(321 + ws); out = []
    for j in range(NNEW):
        a = A.setdefault("alpha", {}).get(j)
        if a is None: a = learn_alpha(A, ws, j); A["alpha"][j] = a
        xy = pairs[splits[NPRE + j][8:]]; yy = torch.as_tensor((xy[:, 0] + xy[:, 1] + new[j]) % P)
        mA, mB = A["C"].mean(1), B["C"].mean(1); vA = mA + (A["C"] - mA[:, None]) @ a; vB = mB + (B["C"] - mB[:, None]) @ a
        perms = [rng.permutation(NPRE) for _ in range(nperm)]
        accA = acc_cols(A["m"], xy, yy, torch.tensor(np.stack([vA]), dtype=torch.float32))[0]
        accB = acc_cols(B["m"], xy, yy, torch.tensor(np.stack([vB] + [mB + (B["C"] - mB[:, None])[:, p] @ a for p in perms]), dtype=torch.float32))
        out.append((accA, accB[0], accB[1:].mean()))
    o = np.array(out); return dict(spanA=float(o[:, 0].mean()), spanB=float(o[:, 1].mean()), span_shuf=float(o[:, 2].mean()))

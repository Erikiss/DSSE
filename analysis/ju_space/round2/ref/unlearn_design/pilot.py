"""Design pilot (toy only): primitives and counter-steering in the pc_meta shift world, BP-muP.
World/training exactly as pc_meta.py ((x + y + c) mod 17, 12 pretraining shifts + 4 new, two tanh layers, AdamW, checkpoint 500
updates after held-out >= 0.95; init torch.manual_seed(ws)).
S  structure: Fourier spectrum of the x/y input columns (exact DFT over Z_17), per-neuron fit of the 12 task columns at the
   neuron's own x-frequency (null: shifts permuted among the columns), relation task columns <-> x/y columns, variance of h2
   explained by answer / task / x / y, frequency-k ring of the task centroids.
T  'maximum composition length' = keep k primitive modes: x and y columns cut to the top-k frequencies vs top-2k PCs vs random 2k
   (in the 16-dim centred function space); task columns cut to top-K Fourier modes vs top-2K PCs vs random 2K in their span.
B  column-only search for the 4 new tasks (8 shots, Adam, from zero) free or restricted to a subspace (task span, top Fourier
   modes, x-span, random of equal dim); A = mean held-out accuracy at steps 10..300 (B2 = A - F, F ~ 0.02-0.03 from pc_meta).
C  counter-steering: (edit) task columns collapsed to their mean (unlearn) then old tasks re-searched from zero (relearn);
   (block) task span projected out of the layer-1 pre-activation for every input, search with the block on, 300 and 3000 steps;
   (inlp) un-search: the found column direction is removed from the allowed column space and the search repeated, 12 rounds,
   against removing random directions.
Usage: python pilot.py <ws> [W]"""
import sys, os, json, time
import numpy as np, torch, torch.nn.functional as Fn
SP = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
sys.path.insert(0, SP + "/ufo_scripts")
torch.set_num_threads(1)
import pc_meta as M
from pc_common import MLP, accuracy
P, NPRE, NNEW, NSLOT, BASE = M.P, M.NPRE, M.NNEW, M.NSLOT, M.BASE
OUT = SP + "/unlearn_design"; os.makedirs(OUT + "/models", exist_ok=True)
EVAL = (10, 20, 40, 80, 160, 300)
LAM = [1.0]                      # 1 = block (project out), 2 = reflect (the task code inverted around ref inside span(U))


def pretrain(ws, W):
    path = f"{OUT}/models/bp_mup_W{W}_s{ws}.pt"; m = MLP(2 * P + NSLOT, W, P, param="mup", base_width=BASE)
    if os.path.exists(path): m.load_state_dict(torch.load(path)); return m
    pre, new, pairs, splits = M.world(ws)
    tr = [M.data(pre[k], k, pairs[splits[k][:144]]) for k in range(NPRE)]; va = [M.data(pre[k], k, pairs[splits[k][144:]]) for k in range(NPRE)]
    Xtr = torch.cat([d[0] for d in tr]); ytr = torch.cat([d[1] for d in tr]); Xva = torch.cat([d[0] for d in va]); yva = torch.cat([d[1] for d in va])
    torch.manual_seed(ws); m = MLP(2 * P + NSLOT, W, P, param="mup", base_width=BASE); opt = M.make_opt(m); first = None
    for s in range(1, M.MAXSTEPS + M.AFTER + 1):
        if s > M.MAXSTEPS and first is None: break
        M.bp_update(m, opt, Xtr, ytr)
        if s % 50 == 0 and first is None and accuracy(m, Xva, yva) >= M.LEVEL1: first = s
        if first is not None and s >= first + M.AFTER: break
    torch.save(m.state_dict(), path); return m


def fwd(m, xy, col, U=None, ref=None):
    """Logits for pairs xy [N,2] with task column col [W] or [N,W]; optional block: the component in span(U) of
    (z1 - ref) is removed (U orthonormal [W,r], ref [W])."""
    W1 = m.l1.weight; z1 = W1[:, xy[:, 0]].T + W1[:, P + xy[:, 1]].T + col + m.l1.bias
    if U is not None: z1 = z1 - LAM[0] * ((z1 - ref) @ U) @ U.T
    return m.head(torch.tanh(m.l2(torch.tanh(z1))))


def acc_int(m, xy, c, col, U=None, ref=None):
    """Accuracy, integrity (share of pairs whose (pred - x - y) mod 17 equals its modal value: the sum code survives up to a shift)
    and whether the modal shift is the right one."""
    with torch.no_grad(): p = fwd(m, xy, col, U, ref).argmax(-1).numpy()
    e = (p - xy[:, 0].numpy() - xy[:, 1].numpy()) % P; cnt = np.bincount(e, minlength=P)
    return float((e == c % P).mean()), float(cnt.max() / len(e)), int(cnt.argmax() == c % P), int(cnt.argmax())


def old_eval(m, world, cols, U=None, ref=None):
    pre, new, pairs, splits = world; r = []
    for k in range(NPRE):
        xy = torch.as_tensor(pairs[splits[k][144:]]); r.append(acc_int(m, xy, pre[k], cols[k], U, ref))
    r = np.array(r); return dict(acc=float(r[:, 0].mean()), integ=float(r[:, 1].mean()), right_shift=float(r[:, 2].mean()), modal=[int(v) for v in r[:, 3]])


def search(m, world, slot, c, steps=300, lr=1e-2, B=None, U=None, ref=None, evals=EVAL, nshot=8, seed=0):
    """Column-only search from zero: col = B B^T z with z in R^W (B [W,r] orthonormal basis of the allowed column space; None =
    free column), Adam (betas 0.9, 0.98) on z, so Adam's coordinates are the neurons in every condition. Returns held-out
    accuracies at evals and the final column."""
    pre, new, pairs, splits = world; idx = splits[slot]; xs = torch.as_tensor(pairs[idx[:nshot]]); ys = torch.as_tensor((pairs[idx[:nshot]].sum(1) + c) % P)
    xh = torch.as_tensor(pairs[idx[8:]]); yh = torch.as_tensor((pairs[idx[8:]].sum(1) + c) % P); W = m.l1.weight.shape[0]
    Pm = None if B is None else B @ B.T; z = torch.zeros(W, requires_grad=True); opt = torch.optim.Adam([z], lr=lr, betas=(0.9, 0.98)); out = []
    colf = lambda: z if Pm is None else Pm @ z
    for p_ in m.parameters(): p_.requires_grad_(False)
    for s in range(1, steps + 1):
        opt.zero_grad(); Fn.cross_entropy(fwd(m, xs, colf(), U, ref), ys).backward(); opt.step()
        if s in evals:
            with torch.no_grad(): out.append(float((fwd(m, xh, colf(), U, ref).argmax(-1) == yh).float().mean()))
    for p_ in m.parameters(): p_.requires_grad_(True)
    return out, colf().detach()


def orth(A):
    Q, _ = np.linalg.qr(A); return Q


def rand_orth(W, r, g):
    return orth(g.standard_normal((W, r)))


def main():
    ws = int(sys.argv[1]); W = int(sys.argv[2]) if len(sys.argv) > 2 else 128; t0 = time.time(); g = np.random.default_rng(100 + ws)
    world = M.world(ws); pre, new, pairs, splits = world; m = pretrain(ws, W); m.eval(); R = dict(ws=ws, W=W, pre=pre, new=new)
    W1 = m.l1.weight.detach().numpy().astype(np.float64); Ex, Ey, T = W1[:, :P], W1[:, P:2 * P], W1[:, 2 * P:2 * P + NPRE]
    cols0 = [torch.as_tensor(T[:, k], dtype=torch.float32) for k in range(NPRE)]
    R["base"] = old_eval(m, world, cols0); print(f"[{ws}] base {R['base']} ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- S: structure
    om = np.exp(-2j * np.pi * np.outer(np.arange(P), np.arange(P)) / P)
    def spec(E):
        Ec = E - E.mean(1, keepdims=True); F = Ec @ om; pw = np.abs(F[:, 1:9]) ** 2 + np.abs(F[:, 16:8:-1]) ** 2; return Ec, pw     # W x 8 (k = 1..8)
    S = {}
    for nm, E in (("x", Ex), ("y", Ey)):
        Ec, pw = spec(E); glob = pw.sum(0) / pw.sum(); share_i = pw / pw.sum(1, keepdims=True); wts = pw.sum(1) / pw.sum()
        nullmax = []
        for _ in range(200):
            _, pn = spec(E[:, g.permutation(P)]); nullmax.append(float(((pn / pn.sum(1, keepdims=True)).max(1) * (pn.sum(1) / pn.sum())).sum()))
        S[nm] = dict(global_share=glob.round(3).tolist(), top3=float(np.sort(glob)[-3:].sum()), key=(np.argsort(-glob)[:4] + 1).tolist(),
                     neuron_max_share_w=float((share_i.max(1) * wts).sum()), null_neuron_max_share_w=float(np.mean(nullmax)), kneuron=(share_i.argmax(1) + 1))
    S["xy_same_freq"] = float(np.mean(S["x"]["kneuron"] == S["y"]["kneuron"]))
    kx = S["x"].pop("kneuron"); S["y"].pop("kneuron")
    sh = np.array(pre, float); Tc = T - T.mean(1, keepdims=True); tot = (Tc ** 2).sum()
    def Qk(k, shv):
        Phi = np.stack([np.cos(2 * np.pi * k * shv / P), np.sin(2 * np.pi * k * shv / P)], 1); return orth(Phi - Phi.mean(0))
    def ss_by_k(shv): return np.stack([((Tc @ Qk(k, shv)) ** 2).sum(1) for k in range(1, 9)], 1)        # W x 8 explained SS per neuron and k
    def r2_at(kvec, shv): A = ss_by_k(shv); return float(A[np.arange(W), kvec - 1].sum() / tot)
    r2x = r2_at(kx, sh); null = [r2_at(kx, g.permutation(sh)) for _ in range(200)]
    best = lambda shv: float(ss_by_k(shv).max(1).sum() / tot)
    S["task_r2_at_x_freq"] = float(r2x); S["task_r2_null_mean"] = float(np.mean(null)); S["task_r2_null_p95"] = float(np.quantile(null, 0.95))
    S["task_r2_best_freq_per_neuron"] = best(sh); S["task_r2_best_null"] = float(np.mean([best(g.permutation(sh)) for _ in range(200)]))
    S["task_global_share_by_k"] = (ss_by_k(sh).sum(0) / tot).round(3).tolist()
    Exc, Eyc = Ex - Ex.mean(1, keepdims=True), Ey - Ey.mean(1, keepdims=True); fc = lambda A, Bm: float((A * Bm).sum() / np.sqrt((A ** 2).sum() * (Bm ** 2).sum()))
    cx = [fc(Tc, Exc[:, (sh.astype(int) + s) % P] - Exc[:, (sh.astype(int) + s) % P].mean(1, keepdims=True)) for s in range(P)]
    cy = [fc(Tc, Eyc[:, (sh.astype(int) + s) % P] - Eyc[:, (sh.astype(int) + s) % P].mean(1, keepdims=True)) for s in range(P)]
    Qx = orth(Exc[:, :P - 1]); Qxy = orth(np.concatenate([Exc[:, :P - 1], Eyc[:, :P - 1]], 1))
    S["task_vs_x_cos_bestshift"] = float(max(cx, key=abs)); S["task_vs_y_cos_bestshift"] = float(max(cy, key=abs))
    S["task_in_xspan"] = float(((Qx.T @ Tc) ** 2).sum() / tot); S["task_in_xyspan"] = float(((Qxy.T @ Tc) ** 2).sum() / tot); S["rand_in_xspan"] = (P - 1) / W; S["rand_in_xyspan"] = 2 * (P - 1) / W
    xy_all = torch.as_tensor(np.array([(x, y) for x in range(P) for y in range(P)]))
    with torch.no_grad():
        H1, H2 = [], []
        for k in range(NPRE):
            z1 = m.l1.weight[:, xy_all[:, 0]].T + m.l1.weight[:, P + xy_all[:, 1]].T + cols0[k] + m.l1.bias; h1 = torch.tanh(z1); H1.append(h1.numpy()); H2.append(torch.tanh(m.l2(h1)).numpy())
    H1, H2 = np.concatenate(H1).astype(np.float64), np.concatenate(H2).astype(np.float64)
    lab = dict(answer=np.concatenate([(xy_all.numpy().sum(1) + c) % P for c in pre]), task=np.repeat(np.arange(NPRE), P * P),
               x=np.tile(xy_all.numpy()[:, 0], NPRE), y=np.tile(xy_all.numpy()[:, 1], NPRE))
    def eta2(H, L):
        Hc = H - H.mean(0); return float(sum(((Hc[L == v].mean(0)) ** 2).sum() * (L == v).sum() for v in np.unique(L)) / (Hc ** 2).sum())
    S["eta2_h1"] = {k: eta2(H1, v) for k, v in lab.items()}; S["eta2_h2"] = {k: eta2(H2, v) for k, v in lab.items()}
    Ma = np.stack([H2[lab["answer"] == v].mean(0) for v in range(P)], 1); _, pwa = spec(Ma); S["h2_answer_spec"] = (pwa.sum(0) / pwa.sum()).round(3).tolist()
    C1 = np.stack([H1[lab["task"] == k].mean(0) for k in range(NPRE)]); D = np.linalg.norm(C1[:, None] - C1[None], axis=-1); iu = np.triu_indices(NPRE, 1)
    rk = lambda v: np.argsort(np.argsort(v)); ring = []
    for k in range(1, 9):
        cyc = np.array([[min((k * (a - b)) % P, (k * (b - a)) % P) for b in pre] for a in pre]); ring.append(float(np.corrcoef(rk(D[iu]), rk(cyc[iu]))[0, 1]))
    S["ring_h1_by_k"] = [round(v, 3) for v in ring]; R["S"] = S
    print(f"[{ws}] S x-spec {S['x']['global_share']} top3 {S['x']['top3']:.2f} neuron-max {S['x']['neuron_max_share_w']:.2f} (null {S['x']['null_neuron_max_share_w']:.2f}) xy-same {S['xy_same_freq']:.2f}", flush=True)
    print(f"[{ws}] S task R2 at x-freq {S['task_r2_at_x_freq']:.3f} (null {S['task_r2_null_mean']:.3f}, p95 {S['task_r2_null_p95']:.3f}); best-freq {S['task_r2_best_freq_per_neuron']:.3f} (null {S['task_r2_best_null']:.3f}); "
          f"cos x {S['task_vs_x_cos_bestshift']:+.2f} y {S['task_vs_y_cos_bestshift']:+.2f}; in x-span {S['task_in_xspan']:.2f} (rand {S['rand_in_xspan']:.2f}) xy-span {S['task_in_xyspan']:.2f}; "
          f"eta2 h1 {S['eta2_h1']} h2 {S['eta2_h2']}; ring k {S['ring_h1_by_k']} ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- T: keep k modes (maximum composition length)
    Tr = {}
    def set_xy(m2, Exn, Eyn):
        with torch.no_grad(): m2.l1.weight[:, :P] = torch.as_tensor(Exn, dtype=torch.float32); m2.l1.weight[:, P:2 * P] = torch.as_tensor(Eyn, dtype=torch.float32)
    import copy
    def cut_fn(E, basis):          # E [W,17]; basis [17, r] orthonormal, orthogonal to the constant
        mu = E.mean(1, keepdims=True); return mu + (E - mu) @ basis @ basis.T
    cosb = lambda k: np.stack([np.cos(2 * np.pi * k * np.arange(P) / P), np.sin(2 * np.pi * k * np.arange(P) / P)], 1)
    one = np.ones((P, 1)) / np.sqrt(P); Cperp = orth(np.eye(P) - one @ one.T)[:, :P - 1]
    rows = []
    for k in range(1, 9):
        r = {}
        for nm in ("fourier", "pca", "random"):
            m2 = copy.deepcopy(m); Es = []
            for E, key in ((Ex, S["x"]), (Ey, S["y"])):
                Ec = E - E.mean(1, keepdims=True)
                if nm == "fourier":
                    _, pw = spec(E); ks = np.argsort(-pw.sum(0))[:k] + 1; Bf = orth(np.concatenate([cosb(kk) for kk in ks], 1))
                elif nm == "pca":
                    _, _, Vt = np.linalg.svd(Ec, full_matrices=False); Bf = Vt[:2 * k].T
                else:
                    Bf = Cperp @ rand_orth(P - 1, 2 * k, g)
                Es.append(cut_fn(E, Bf))
            set_xy(m2, *Es); r[nm] = old_eval(m2, world, cols0)["acc"]
        rows.append(r)
    Tr["xy_keep_k"] = rows
    rows = []
    for K in (1, 2, 3, 4, 5):
        r = {}; Phi = np.concatenate([np.stack([np.cos(2 * np.pi * kk * sh / P), np.sin(2 * np.pi * kk * sh / P)], 1) for kk in range(1, 9)], 1)
        sc = [((orth(Phi[:, 2 * (kk - 1):2 * kk] - Phi[:, 2 * (kk - 1):2 * kk].mean(0)).T @ Tc.T) ** 2).sum() for kk in range(1, 9)]; ks = np.argsort(-np.array(sc))[:K]
        Bk = orth(np.concatenate([Phi[:, 2 * kk:2 * kk + 2] for kk in ks], 1) - np.concatenate([Phi[:, 2 * kk:2 * kk + 2] for kk in ks], 1).mean(0))   # 12 x 2K
        U_, s_, Vt = np.linalg.svd(Tc, full_matrices=False)
        for nm, Bm in (("fourier", Bk), ("pca", Vt[:2 * K].T), ("random", orth(orth(np.eye(NPRE) - 1.0 / NPRE)[:, :NPRE - 1] @ rand_orth(NPRE - 1, 2 * K, g)))):
            Tn = T.mean(1, keepdims=True) + Tc @ Bm @ Bm.T; r[nm] = old_eval(m, world, [torch.as_tensor(Tn[:, j], dtype=torch.float32) for j in range(NPRE)])["acc"]
            r[nm + "_var"] = float(((Tc @ Bm) ** 2).sum() / tot)
        rows.append(r)
    Tr["task_keep_K"] = rows; R["T"] = Tr
    print(f"[{ws}] T xy keep k (fourier/pca/random): " + " | ".join(f"{i + 1}: {r['fourier']:.2f}/{r['pca']:.2f}/{r['random']:.2f}" for i, r in enumerate(Tr['xy_keep_k'])), flush=True)
    print(f"[{ws}] T task keep K (fourier/pca/random, var): " + " | ".join(f"{i + 1}: {r['fourier']:.2f}/{r['pca']:.2f}/{r['random']:.2f} ({r['fourier_var']:.2f}/{r['pca_var']:.2f}/{r['random_var']:.2f})" for i, r in enumerate(Tr['task_keep_K'])) + f" ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- B: column-only search, free or restricted
    mu_col = T.mean(1); Qtask = orth(np.concatenate([Tc[:, :NPRE - 1], mu_col[:, None]], 1))         # task span + mean, 12 dims
    _, _, Vt = np.linalg.svd(Tc.T, full_matrices=False)
    def runB(Bm=None, U=None, ref=None, steps=300, lr=1e-2, evals=EVAL):
        A, cols = [], []
        for j in range(NNEW):
            a, v = search(m, world, NPRE + j, new[j], steps, lr, None if Bm is None else torch.as_tensor(Bm, dtype=torch.float32), U, ref, evals); A.append(float(np.mean(a))); cols.append(v.numpy())
        return float(np.mean(A)), np.stack(cols, 1)
    Bres = {}; Afree, Vfree = runB(); Bres["free"] = Afree; Bres["free_lr0.1"] = runB(lr=1e-1)[0]
    Bres["free_in_taskspan"] = float(np.mean([((Qtask.T @ Vfree[:, j]) ** 2).sum() / (Vfree[:, j] ** 2).sum() for j in range(NNEW)]))
    Bres["free_in_xyspan"] = float(np.mean([((Qxy.T @ Vfree[:, j]) ** 2).sum() / (Vfree[:, j] ** 2).sum() for j in range(NNEW)]))
    Bres["taskspan12"] = max(runB(Qtask, lr=lr)[0] for lr in (1e-2, 1e-1)); Bres["rand12"] = max(runB(rand_orth(W, 12, g), lr=lr)[0] for lr in (1e-2, 1e-1))
    Bres["xspan16+mean"] = max(runB(orth(np.concatenate([Qx, mu_col[:, None]], 1)), lr=lr)[0] for lr in (1e-2, 1e-1))
    Bres["xyspan32+mean"] = max(runB(orth(np.concatenate([Qxy, mu_col[:, None]], 1)), lr=lr)[0] for lr in (1e-2, 1e-1)); Bres["rand33"] = max(runB(rand_orth(W, 33, g), lr=lr)[0] for lr in (1e-2, 1e-1))
    Bres["rand64"] = max(runB(rand_orth(W, 64, g), lr=lr)[0] for lr in (1e-2, 1e-1))
    R["B"] = Bres; print(f"[{ws}] B {json.dumps({k: round(v, 3) for k, v in Bres.items()})} ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- C: counter-steering
    Cr = {}
    colsM = [torch.as_tensor(mu_col, dtype=torch.float32)] * NPRE
    Cr["edit_old"] = old_eval(m, world, colsM)
    rel = [float(np.mean(search(m, world, k, pre[k])[0])) for k in range(4)]; Cr["edit_relearn_old4"] = float(np.mean(rel))
    Qt11 = torch.as_tensor(orth(Tc[:, :NPRE - 1]), dtype=torch.float32); ref = torch.as_tensor(mu_col, dtype=torch.float32) + m.l1.bias.detach()
    zb = (m.l1.weight[:, torch.as_tensor(pairs[:, 0])].T + m.l1.weight[:, P + torch.as_tensor(pairs[:, 1])].T).detach().mean(0) + ref
    for nm, Ub in (("block_task11", Qt11), ("block_rand11", torch.as_tensor(rand_orth(W, NPRE - 1, g), dtype=torch.float32)), ("block_x16", torch.as_tensor(Qx, dtype=torch.float32))):
        o = old_eval(m, world, cols0, Ub, zb); a300 = runB(U=Ub, ref=zb)[0]; a3000 = runB(U=Ub, ref=zb, steps=3000, evals=(3000,))[0]
        Cr[nm] = dict(old=o, search300=a300, search3000_final=a3000)
    a3000free = runB(steps=3000, evals=(3000,))[0]; Cr["free_search3000_final"] = a3000free
    LAM[0] = 2.0; Cr["reflect_task11"] = old_eval(m, world, cols0, Qt11, zb); Cr["reflect_rand11"] = old_eval(m, world, cols0, torch.as_tensor(rand_orth(W, NPRE - 1, g), dtype=torch.float32), zb); LAM[0] = 1.0
    print(f"[{ws}] reflect task11 {Cr['reflect_task11']} rand11 acc {Cr['reflect_rand11']['acc']:.3f} | shifts {pre}", flush=True)
    print(f"[{ws}] C edit {Cr['edit_old']} relearn {Cr['edit_relearn_old4']:.3f}; " + "; ".join(f"{k} old {Cr[k]['old']} s300 {Cr[k]['search300']:.3f} s3000 {Cr[k]['search3000_final']:.3f}" for k in ("block_task11", "block_rand11", "block_x16")) + f"; free s3000 {a3000free:.3f} ({time.time() - t0:.0f}s)", flush=True)
    # iterative un-search (INLP-style) on new task 0: the column space excludes every direction found so far
    for mode in ("found", "random"):
        Ex_ = np.zeros((W, 0)); trace = []
        for it in range(12):
            Bm = np.linalg.svd(Ex_.T, full_matrices=True)[2][Ex_.shape[1]:].T if Ex_.shape[1] else None
            a, v = search(m, world, NPRE, new[0], 300, 1e-2, None if Bm is None else torch.as_tensor(Bm, dtype=torch.float32)); trace.append(float(np.mean(a)))
            d = v.numpy().astype(np.float64) if mode == "found" else g.standard_normal(W)
            d = d - (Ex_ @ (Ex_.T @ d) if Ex_.shape[1] else 0); Ex_ = np.concatenate([Ex_, (d / np.linalg.norm(d))[:, None]], 1)
        Cr[f"inlp_{mode}"] = trace
    print(f"[{ws}] C un-search found {[round(v, 2) for v in Cr['inlp_found']]} random {[round(v, 2) for v in Cr['inlp_random']]} ({time.time() - t0:.0f}s)", flush=True)
    R["C"] = Cr; R["seconds"] = time.time() - t0
    json.dump(R, open(f"{OUT}/pilot_s{ws}_W{W}.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


if __name__ == "__main__":
    main()

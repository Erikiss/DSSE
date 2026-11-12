"""README section 12 (plan: pc_out/plan_v2.md): does a first-order meta level ('learning to learn') of a small network disappear
with width or with delta, the spread of its task space, and does predictive coding (PC) behave differently from backprop (BP)?
World: shift tasks f_c(x, y) = (x + y + c) mod 17; input one-hot x, one-hot y, one-hot task slot (12 pretraining + 4 new + 1
spare); two tanh hidden layers of width W; cross-entropy (pc_common.py). Seed s fixes the shifts (12 pretraining, 4 new), each
task's split (144 training / 145 held-out pairs for pretraining tasks; for new tasks the first 8 pairs of the split are the shots,
the first 4 the nested 4-shot set, the other 281 are held out), data order and initialisation.
Arms: bp_mup (BP, muP for Adam with base width 128), bp_sp (BP, standard parameterisation), pc_mup (PC with the muP predictions,
T inference steps with eta_x = 1 / lambda_max of the energy Hessian in the value nodes, re-estimated every 100 updates).
AdamW lr 1e-3, betas (0.9, 0.98), decay per update lr * wd = 1e-3 in every parameter group (width-independent), full batch.
Pretraining until 500 updates after the held-out accuracy first reaches 0.95 (checked every 50 updates), at most MAXSTEPS; a run
that never reaches 0.95 fails level 1. K = 1 control: one shift, trained for the number of updates of the K = 12 run's checkpoint.
At the checkpoint:
  delta per hidden layer over the 12 pretraining tasks x all 289 pairs: F = tr(S_between) / tr(S_within) of the task clouds,
  floor = mean F over 50 label permutations, delta = F - floor; delta_ctrl the same after subtracting each (x, y) pair's mean over
  the tasks; ring = Spearman correlation of task-centroid distances with the cyclic distance of the shifts.
  B2: for each new task, only its input column is trained (from zero; Adam lr 1e-2, no weight decay; for PC the local update at
  equilibrium) on n_shot pairs; A = mean held-out accuracy at steps 10, 20, 40, 80, 160, 300. F = the same mean for a fresh net of
  the same arm and width trained with all weights on the same shots. B2 = A - F, averaged over the 4 new tasks; n_shot 8 and 4.
  B2_all (all weights of the pretrained net, AdamW lr 1e-3) and the retention of the pretraining tasks after it.
PC diagnostics every 100 updates: lambda_max, per update the end/start ratio of |dE/dh| and whether the energy rose, and the
cosine between the PC weight update and the BP gradient per layer on a fixed batch of 512.
Optional regulariser (delta probe, plan D2): gamma * log F_task(h2) on the training batch added to the loss (BP arms only).
Every job is cached as one JSON file in <out_dir>/jobs/; the pretrained net at the checkpoint is saved in <out_dir>/models/.
Usage: python pc_meta.py <out_dir> <arm> <W> <seed> [K] [gamma] [wd_scale]      env: T (50), MAXSTEPS (3000), QUICK (1 = code test)"""
import json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as Fn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_common import MLP, pc_energy, accuracy, BETA
torch.set_num_threads(int(os.environ.get("NT", "1"))); DEV = torch.device(os.environ.get("DEVICE", "cpu"))
if DEV.type == "cuda": torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

P, NPRE, NNEW, NSLOT, BASE = 17, 12, 4, 17, 128
LR, DECAY, BETAS = float(os.environ.get("LR", "1e-3")), 1e-3, (0.9, 0.98)   # DECAY = lr * weight decay per update, the same in every group
T = int(os.environ.get("T", "50")); BETA[0] = float(os.environ.get("BETA", "1")); MAXSTEPS = int(os.environ.get("MAXSTEPS", "3000")); QUICK = os.environ.get("QUICK") == "1"
FT_STEPS, FT_EVAL, FT_LR = 300, (10, 20, 40, 80, 160, 300), 1e-2
LEVEL1, AFTER = 0.95, 500
if QUICK: MAXSTEPS, FT_STEPS, FT_EVAL, LEVEL1, AFTER = 200, 20, (5, 10, 20), 0.0, 50       # code test: every path runs, nothing is learned


def world(seed):
    g = np.random.default_rng(1000 + seed); sh = g.permutation(P)
    return [int(c) for c in sh[:NPRE]], [int(c) for c in sh[NPRE:NPRE + NNEW]], np.array([(x, y) for x in range(P) for y in range(P)]), [g.permutation(P * P) for _ in range(NPRE + NNEW)]


def enc(xy, slot):
    X = torch.zeros(len(xy), 2 * P + NSLOT); r = torch.arange(len(xy)); X[r, torch.as_tensor(xy[:, 0])] = 1; X[r, P + torch.as_tensor(xy[:, 1])] = 1; X[r, 2 * P + slot] = 1; return X


def data(c, slot, xy):
    return enc(xy, slot).to(DEV), torch.as_tensor((xy[:, 0] + xy[:, 1] + c) % P).to(DEV)


def make_opt(m, decay_scale=1.0):
    groups = []
    for g in m.param_groups(LR):
        groups.append(dict(params=g["params"], lr=g["lr"], weight_decay=DECAY * decay_scale / g["lr"]))
    return torch.optim.AdamW(groups, lr=LR, betas=BETAS)


def task_fisher(H, lab, nk):
    """Differentiable tr(S_between) / tr(S_within) of the clouds given by integer labels lab (nk classes)."""
    mu = H.mean(0); B = H.new_zeros(()); Wt = H.new_zeros(())
    for k in range(nk):
        Hk = H[lab == k]
        if len(Hk) == 0: continue
        mk = Hk.mean(0); B = B + len(Hk) * ((mk - mu) ** 2).sum(); Wt = Wt + ((Hk - mk) ** 2).sum()
    return B / Wt


# ---------------- predictive coding: inference with eta_x = 1 / lambda_max and logging
def lam_max(m, x, y, iters=20):
    with torch.no_grad(): h1, h2 = m.acts(x)
    h1 = h1.clone().requires_grad_(True); h2 = h2.clone().requires_grad_(True); B = x.shape[0]
    v1, v2 = torch.randn_like(h1), torch.randn_like(h2); nrm = torch.sqrt((v1 ** 2).sum() + (v2 ** 2).sum()); v1, v2 = v1 / nrm, v2 / nrm; lam = 0.0
    for p in m.parameters(): p.requires_grad_(False)
    for _ in range(iters):
        E = pc_energy(m, x, h1, h2, y) * B; g1, g2 = torch.autograd.grad(E, (h1, h2), create_graph=True)
        Hv1, Hv2 = torch.autograd.grad((g1 * v1).sum() + (g2 * v2).sum(), (h1, h2))
        lam = float((Hv1 * v1).sum() + (Hv2 * v2).sum()); nrm = torch.sqrt((Hv1 ** 2).sum() + (Hv2 ** 2).sum()) + 1e-12; v1, v2 = (Hv1 / nrm).detach(), (Hv2 / nrm).detach()
    for p in m.parameters(): p.requires_grad_(True)
    return max(lam, 1e-3)


def pc_relax_autograd(m, x, y, eta_x, steps):
    """Reference implementation (autograd), kept for the equivalence test of pc_relax."""
    with torch.no_grad(): h1, h2 = m.acts(x)
    h1 = h1.clone().requires_grad_(True); h2 = h2.clone().requires_grad_(True); B = x.shape[0]; g0 = gl = None
    for p in m.parameters(): p.requires_grad_(False)
    for t in range(steps):
        E = pc_energy(m, x, h1, h2, y) * B; g1, g2 = torch.autograd.grad(E, (h1, h2)); gn = float(torch.sqrt((g1 ** 2).sum() + (g2 ** 2).sum()))
        if g0 is None: g0 = gn
        gl = gn
        with torch.no_grad(): h1 -= eta_x * g1; h2 -= eta_x * g2
    for p in m.parameters(): p.requires_grad_(True)
    return h1.detach(), h2.detach(), (gl / g0 if g0 else 0.0)


def pc_relax(m, x, y, eta_x, steps):
    """Feedforward init, then 'steps' gradient steps on the per-sample energy E = pc_energy * B in h1, h2 (weights fixed). The
    gradients are written in closed form (identical to autograd of E, checked against pc_relax_autograd; about a third of the GPU
    kernels and one host sync per call instead of two per step):
      dE/dh1 = e1 - (e2 * (1 - a2^2)) @ W2,   dE/dh2 = e2 + beta * mult * (softmax(logits) - onehot(y)) @ W3.
    Returns the relaxed values, the end/start ratio of the gradient norm and whether the energy rose at any step. The energy is
    monitored in float64 (float32 rounding of the cross-entropy registered as rises near convergence; review 12 Nov 2026), and the
    state after the last step is included."""
    with torch.no_grad():
        a1 = torch.tanh(m.l1(x)); h1 = a1.clone(); h2 = torch.tanh(m.l2(h1)); W2, b2, W3, b3 = m.l2.weight, m.l2.bias, m.l3.weight, m.l3.bias
        mult, beta = m.out_mult, BETA[0]; Y = Fn.one_hot(y, W3.shape[0]).to(h2.dtype); Es, gs = [], []
        for t in range(steps + 1):
            a2 = torch.tanh(h1 @ W2.T + b2); e1 = h1 - a1; e2 = h2 - a2; logits = (h2 @ W3.T + b3) * mult
            Es.append(0.5 * (e1.double() ** 2).sum() + 0.5 * (e2.double() ** 2).sum() + beta * Fn.cross_entropy(logits.double(), y, reduction="sum"))
            if t == steps: break
            g1 = e1 - (e2 * (1 - a2 ** 2)) @ W2; g2 = e2 + (beta * mult) * ((torch.softmax(logits, -1) - Y) @ W3)
            gs.append(torch.sqrt((g1.double() ** 2).sum() + (g2.double() ** 2).sum()))
            h1 = h1 - eta_x * g1; h2 = h2 - eta_x * g2
        E = torch.stack(Es).cpu().numpy(); G = torch.stack(gs).cpu().numpy() if gs else np.zeros(1)
    rose = bool(np.any(E[1:] > E[:-1] + 1e-9 * np.abs(E[:-1])))
    return h1, h2, (float(G[-1] / G[0]) if G[0] > 0 else 0.0), rose


def pc_update(m, opt, x, y, eta_x, mask_col=None):
    h1, h2, ratio, rose = pc_relax(m, x, y, eta_x, T); opt.zero_grad(); E = pc_energy(m, x, h1, h2, y); E.backward()
    if mask_col is not None:
        g = m.l1.weight.grad; keep = torch.zeros_like(g); keep[:, mask_col] = 1; g.mul_(keep)
        for p in m.parameters():
            if p is not m.l1.weight and p.grad is not None: p.grad.zero_()
    opt.step(); return ratio, rose


def bp_update(m, opt, x, y, gamma=0.0, lab=None, mask_col=None):
    opt.zero_grad(); h1, h2 = m.acts(x); loss = Fn.cross_entropy(m.head(h2), y)
    if gamma: loss = loss + gamma * torch.log(task_fisher(h2, lab, NPRE))
    loss.backward()
    if mask_col is not None:
        g = m.l1.weight.grad; keep = torch.zeros_like(g); keep[:, mask_col] = 1; g.mul_(keep)
        for p in m.parameters():
            if p is not m.l1.weight and p.grad is not None: p.grad.zero_()
    opt.step(); return float(loss)


def cosines_pc_bp(m, x, y, eta_x):
    """Per layer cos(PC weight update, BP gradient) on batch x."""
    h1, h2, _, _ = pc_relax(m, x, y, eta_x, T); m.zero_grad(); pc_energy(m, x, h1, h2, y).backward(); gpc = [l.weight.grad.clone() for l in (m.l1, m.l2, m.l3)]
    m.zero_grad(); Fn.cross_entropy(m(x), y).backward(); gbp = [l.weight.grad.clone() for l in (m.l1, m.l2, m.l3)]; m.zero_grad()
    return [float((a * b).sum() / (a.norm() * b.norm() + 1e-12)) for a, b in zip(gpc, gbp)]


# ---------------- delta and ring geometry
@torch.no_grad()
def delta_stats(m, pre, pairs, seed):
    X = torch.cat([enc(pairs, k) for k in range(NPRE)]).to(DEV); lab = torch.arange(NPRE).repeat_interleave(len(pairs)).to(DEV); h1, h2 = m.acts(X); g = torch.Generator().manual_seed(77 + seed); out = {}
    for name, H in (("h1", h1.double()), ("h2", h2.double())):
        F = float(task_fisher(H, lab, NPRE)); fl = float(np.mean([float(task_fisher(H, lab[torch.randperm(len(lab), generator=g).to(DEV)], NPRE)) for _ in range(50)]))
        Hc = H.view(NPRE, len(pairs), -1); Hc = (Hc - Hc.mean(0, keepdim=True)).reshape(len(lab), -1)
        Fc = float(task_fisher(Hc, lab, NPRE)); flc = float(np.mean([float(task_fisher(Hc, lab[torch.randperm(len(lab), generator=g).to(DEV)], NPRE)) for _ in range(50)]))
        C = torch.stack([H[lab == k].mean(0) for k in range(NPRE)]); D = torch.cdist(C, C).cpu().numpy(); cyc = np.array([[min(abs(a - b), P - abs(a - b)) for b in pre] for a in pre])
        iu = np.triu_indices(NPRE, 1)
        from scipy.stats import spearmanr; ring = float(spearmanr(D[iu], cyc[iu]).correlation)        # Spearman with average ranks for ties
        out[name] = dict(F=F, floor=fl, delta=F - fl, F_ctrl=Fc, floor_ctrl=flc, delta_ctrl=Fc - flc, ring=ring)
    return out


# ---------------- fine-tuning
def finetune(m0, arm, W, new, pairs, splits, j, nshot, mode, seed, eta_x=None):
    """mode 'col': only the new task's input column (from zero), Adam lr 1e-2; 'fresh': a fresh net, all weights; 'all': all weights
    of the pretrained net. Returns held-out accuracies at FT_EVAL and (for 'all') the pretraining held-out accuracy afterwards."""
    slot = NPRE + j; idx = splits[slot]; Xs, ys = data(new[j], slot, pairs[idx[:nshot]]); Xh, yh = data(new[j], slot, pairs[idx[8:]]); col = 2 * P + slot
    if mode == "fresh": torch.manual_seed(50_000 + 101 * seed + j); m = MLP(2 * P + NSLOT, W, P, param="sp" if arm == "bp_sp" else "mup", base_width=BASE).to(DEV)
    else: m = MLP(2 * P + NSLOT, W, P, param="sp" if arm == "bp_sp" else "mup", base_width=BASE).to(DEV); m.load_state_dict(m0.state_dict())
    if mode == "col":
        with torch.no_grad(): m.l1.weight[:, col] = 0
        opt = torch.optim.Adam([m.l1.weight], lr=FT_LR, betas=BETAS)
    else: opt = make_opt(m)
    if arm == "pc_mup" and eta_x is None: eta_x = 1.0 / lam_max(m, Xs, ys)
    acc = []
    for s in range(1, FT_STEPS + 1):
        if arm == "pc_mup":
            if mode == "col": pc_update(m, opt, Xs, ys, eta_x, mask_col=col)
            else:
                if s % 100 == 1: eta_x = 1.0 / lam_max(m, Xs, ys)
                pc_update(m, opt, Xs, ys, eta_x)
        else: bp_update(m, opt, Xs, ys, mask_col=col if mode == "col" else None)
        if s in FT_EVAL: acc.append(accuracy(m, Xh, yh))
    return acc, m


def main():
    out, arm, W, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); K = int(sys.argv[5]) if len(sys.argv) > 5 else NPRE
    gamma = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0; wds = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
    os.makedirs(f"{out}/jobs", exist_ok=True); name = f"{arm}_W{W}_s{seed}_K{K}_g{gamma:g}_wd{wds:g}" + (f"_lr{LR:g}" if LR != 1e-3 else "") + (f"_T{T}" if arm == "pc_mup" and T != 50 else "") + (f"_beta{BETA[0]:g}" if arm == "pc_mup" and BETA[0] != 1 else "") + ("_quick" if QUICK else ""); path = f"{out}/jobs/{name}.json"
    if os.path.exists(path): print("cached", name); return
    t0 = time.time(); pre, new, pairs, splits = world(seed)
    tr = [data(pre[k], k, pairs[splits[k][:144]]) for k in range(K)]; va = [data(pre[k], k, pairs[splits[k][144:]]) for k in range(K)]
    Xtr = torch.cat([d[0] for d in tr]); ytr = torch.cat([d[1] for d in tr]); Xva = torch.cat([d[0] for d in va]); yva = torch.cat([d[1] for d in va])
    lab = torch.arange(K).repeat_interleave(144).to(DEV)
    torch.manual_seed(seed); m = MLP(2 * P + NSLOT, W, P, param="sp" if arm == "bp_sp" else "mup", base_width=BASE).to(DEV); opt = make_opt(m, wds)
    rec = dict(arm=arm, W=W, seed=seed, K=K, gamma=gamma, wd_scale=wds, lr=LR, T=T if arm == "pc_mup" else None, beta=BETA[0] if arm == "pc_mup" else None, shifts_pre=pre, shifts_new=new, log=[], pc=dict(lam=[], ratio=[], rose=[], cos=[]))
    target = None
    if K == 1:                                            # the K = 1 control runs as long as the K = 12 run of the same arm, width and seed
        ref = f"{out}/jobs/{arm}_W{W}_s{seed}_K{NPRE}_g0_wd{wds:g}" + (f"_lr{LR:g}" if LR != 1e-3 else "") + (f"_T{T}" if arm == "pc_mup" and T != 50 else "") + (f"_beta{BETA[0]:g}" if arm == "pc_mup" and BETA[0] != 1 else "") + ("_quick" if QUICK else "") + ".json"
        target = json.load(open(ref))["checkpoint"] if os.path.exists(ref) else None
        if target is None: print("K = 1 needs the K = 12 run first"); return
    first95 = None; ckpt = None; eta_x = None; probe = torch.randperm(len(ytr), generator=torch.Generator().manual_seed(seed))[:512].to(DEV)
    act0 = None
    for s in range(1, MAXSTEPS + AFTER + 1):
        if K > 1 and s > MAXSTEPS and first95 is None: break          # the K = 1 control always runs to its target (review 12 Nov 2026)
        if arm == "pc_mup":
            if s % 100 == 1: eta_x = 1.0 / lam_max(m, Xtr, ytr); rec["pc"]["lam"].append((s, 1.0 / eta_x))
            ratio, rose = pc_update(m, opt, Xtr, ytr, eta_x); rec["pc"]["ratio"].append(ratio); rec["pc"]["rose"].append(rose)
            if s % 100 == 1: rec["pc"]["cos"].append((s, cosines_pc_bp(m, Xtr[probe], ytr[probe], eta_x)))
        else: bp_update(m, opt, Xtr, ytr, gamma, lab)
        if s == 10:                                       # coordinate check (G0): change of the hidden activations after 10 updates
            with torch.no_grad():
                torch.manual_seed(seed); m0 = MLP(2 * P + NSLOT, W, P, param="sp" if arm == "bp_sp" else "mup", base_width=BASE).to(DEV); a0 = m0.acts(Xtr[probe]); a1 = m.acts(Xtr[probe])
                rec["coord_check"] = [float((b - a).abs().mean()) for a, b in zip(a0, a1)]
        if s % 50 == 0:
            a_tr, a_va = accuracy(m, Xtr, ytr), accuracy(m, Xva, yva); rec["log"].append((s, a_tr, a_va))
            if first95 is None and a_va >= LEVEL1 and K > 1: first95 = s
        if K > 1 and first95 is not None and s >= first95 + AFTER: ckpt = s; break
        if K == 1 and s >= target: ckpt = s; break
    rec.update(first95=first95, checkpoint=ckpt, level1=first95 is not None and ckpt is not None, final_train=accuracy(m, Xtr, ytr), final_heldout=accuracy(m, Xva, yva), pretrain_seconds=time.time() - t0)
    print(f"{name}: first95 {first95} checkpoint {ckpt} train {rec['final_train']:.3f} held-out {rec['final_heldout']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    if ckpt is None and K > 1: json.dump(rec, open(path + ".tmp", "w")); os.replace(path + ".tmp", path); return
    if K == NPRE: rec["delta"] = delta_stats(m, pre, pairs, seed)
    os.makedirs(f"{out}/models", exist_ok=True); torch.save({k: v.cpu() for k, v in m.state_dict().items()}, f"{out}/models/{name}.pt")   # for the follow-up analyses (primitives, transfer)
    if arm == "pc_mup": eta_x = 1.0 / lam_max(m, Xtr, ytr)
    ft = {}
    for nshot in (8, 4):
        A, Fr = [], []
        for j in range(NNEW):
            a, _ = finetune(m, arm, W, new, pairs, splits, j, nshot, "col", seed, eta_x); A.append(a)
            f, _ = finetune(None, arm, W, new, pairs, splits, j, nshot, "fresh", seed); Fr.append(f)
        ft[nshot] = dict(A=A, F=Fr, B2=float(np.mean([np.mean(a) - np.mean(f) for a, f in zip(A, Fr)])))
    allw, ret = [], []
    for j in range(NNEW):
        a, mm = finetune(m, arm, W, new, pairs, splits, j, 8, "all", seed); allw.append(a); ret.append(accuracy(mm, Xva, yva))
    ft["all8"] = dict(acc=allw, retention=ret, B2_all=float(np.mean([np.mean(a) - np.mean(f) for a, f in zip(allw, ft[8]["F"])])))
    rec["finetune"] = ft; rec["seconds"] = time.time() - t0; json.dump(rec, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)   # atomic: a half-written file never counts as done
    print(f"{name}: B2(8) {ft[8]['B2']:.3f} B2(4) {ft[4]['B2']:.3f} B2_all {ft['all8']['B2_all']:.3f} delta h2 {rec.get('delta', {}).get('h2', {}).get('delta', float('nan')):.3f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

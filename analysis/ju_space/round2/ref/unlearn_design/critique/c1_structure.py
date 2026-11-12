"""Critique check 1 (toy only, 1 thread): is there a ring / Fourier code over Z_17 in the pc_meta shift nets, tested basis-free,
and can a linear erasure in the middle of the net (h1 or h2, after a nonlinearity) leave a usable nonlinear residue?
(a) task columns: share of centred task-column energy explained by the 2-dim Fourier regressor of frequency k over the 12 trained
    shifts; max over k against the permutation null of the max (shift labels shuffled).
(b) the same for the h1 and h2 task centroids (mean over all 289 pairs).
(c) ring interpolation: leave-one-shift-out ridge prediction of the held-out task column from [cos, sin](2 pi k c / 17), k = 1..K,
    K in {1, 2, 3}; R2 against shuffled shift labels.
(d) answer code: DFT over the answer index of the h2 answer-class means and of the read-out rows (W3); top-1 and top-2 frequency
    share against answer-label permutations.
(e) mid-network erasure: class-mean erasure (orthogonal projection removing the span of the 12 task-mean differences, which makes
    every task mean equal = linear guardedness) at h1 or h2; old-task accuracy with the erased state propagated; task-identity probes
    (linear ridge and a small MLP) on held-out pairs before and after erasure.
Usage: python c1_structure.py <tag> <arm> <W> <seed>   (tag 'pilot' loads unlearn_design/models/bp_mup_W{W}_s{seed}.pt)"""
import sys, os, json, time, math
import numpy as np, torch, torch.nn.functional as Fn
SP = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
sys.path.insert(0, SP + "/ufo_scripts"); torch.set_num_threads(1)
import pc_meta as M
from pc_common import MLP, accuracy
P, NPRE, NSLOT, BASE = M.P, M.NPRE, M.NSLOT, M.BASE
OUT = SP + "/unlearn_design/critique"


def pretrain(arm, W, ws):
    path = f"{OUT}/models/{arm}_W{W}_s{ws}.pt"; m = MLP(2 * P + NSLOT, W, P, param="sp" if arm == "bp_sp" else "mup", base_width=BASE)
    if os.path.exists(path): m.load_state_dict(torch.load(path)); return m, None
    pre, new, pairs, splits = M.world(ws)
    tr = [M.data(pre[k], k, pairs[splits[k][:144]]) for k in range(NPRE)]; va = [M.data(pre[k], k, pairs[splits[k][144:]]) for k in range(NPRE)]
    Xtr = torch.cat([d[0] for d in tr]); ytr = torch.cat([d[1] for d in tr]); Xva = torch.cat([d[0] for d in va]); yva = torch.cat([d[1] for d in va])
    torch.manual_seed(ws); m = MLP(2 * P + NSLOT, W, P, param="sp" if arm == "bp_sp" else "mup", base_width=BASE); opt = M.make_opt(m); first = None; s = 0
    for s in range(1, M.MAXSTEPS + M.AFTER + 1):
        if s > M.MAXSTEPS and first is None: break
        M.bp_update(m, opt, Xtr, ytr)
        if s % 50 == 0 and first is None and accuracy(m, Xva, yva) >= M.LEVEL1: first = s
        if first is not None and s >= first + M.AFTER: break
    torch.save(m.state_dict(), path); return m, dict(first95=first, steps=s)


def orth(A): return np.linalg.qr(A)[0]


def fourier_cols(cv, k):
    Phi = np.stack([np.cos(2 * np.pi * k * cv / P), np.sin(2 * np.pi * k * cv / P)], 1); return orth(Phi - Phi.mean(0))


def share_by_k(Tc, cv):           # Tc: [n_tasks, D] centred over tasks; cv: shift values
    tot = (Tc ** 2).sum(); return np.array([((fourier_cols(cv, k).T @ Tc) ** 2).sum() / tot for k in range(1, 9)])


def ring_test(C, cv, g, nperm=1000):
    Tc = C - C.mean(0); s = share_by_k(Tc, cv); null = np.array([share_by_k(Tc, g.permutation(cv)).max() for _ in range(nperm)])
    return dict(share=s.round(3).tolist(), kmax=int(s.argmax() + 1), max=float(s.max()), null_max_mean=float(null.mean()), null_max_p99=float(np.quantile(null, 0.99)), p=float((1 + (null >= s.max()).sum()) / (1 + nperm)))


def loo_ring(C, cv, K, lam=1e-2):
    Tc = C - C.mean(0); err, tot = 0.0, 0.0
    for i in range(len(cv)):
        tr = np.arange(len(cv)) != i; feats = lambda c: np.concatenate([[np.cos(2 * np.pi * k * c / P), np.sin(2 * np.pi * k * c / P)] for k in range(1, K + 1)])
        A = np.stack([feats(c) for c in cv[tr]]); A = np.c_[A, np.ones(len(A))]; Y = Tc[tr]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y); pred = np.r_[feats(cv[i]), 1.0] @ w
        err += ((pred - Tc[i]) ** 2).sum(); tot += ((Tc[i] - Tc[tr].mean(0)) ** 2).sum()
    return 1 - err / tot


def spec_rows(Mat, g, nperm=500):           # Mat: [17, D] rows indexed by answer value; top-1 / top-2 share of non-DC power
    def sh(Mx):
        Fm = np.fft.fft(Mx - Mx.mean(0), axis=0); pw = (np.abs(Fm[1:9]) ** 2 + np.abs(Fm[16:8:-1]) ** 2).sum(1); pw = pw / pw.sum(); srt = np.sort(pw)[::-1]; return pw, srt[0], srt[:2].sum()
    pw, t1, t2 = sh(Mat); n1, n2 = zip(*[sh(Mat[g.permutation(P)])[1:] for _ in range(nperm)])
    return dict(spec=pw.round(3).tolist(), top1=float(t1), top2=float(t2), null_top1_p99=float(np.quantile(n1, 0.99)), null_top2_p99=float(np.quantile(n2, 0.99)))


def mlp_probe(Xtr, ytr, Xte, yte, ncls, hid=64, steps=400, seed=0):
    torch.manual_seed(seed); Xtr, Xte = torch.as_tensor(Xtr, dtype=torch.float32), torch.as_tensor(Xte, dtype=torch.float32); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd; net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], hid), torch.nn.ReLU(), torch.nn.Linear(hid, ncls))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4); yt = torch.as_tensor(ytr)
    for _ in range(steps): opt.zero_grad(); Fn.cross_entropy(net(Xtr), yt).backward(); opt.step()
    with torch.no_grad(): return float((net(Xte).argmax(-1).numpy() == yte).mean())


def ridge_probe(Xtr, ytr, Xte, yte, ncls, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; A = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]; B = np.c_[(Xte - mu) / sd, np.ones(len(Xte))]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ np.eye(ncls)[ytr]); return float((np.argmax(B @ w, 1) == yte).mean())


def main():
    tag, arm, W, ws = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); t0 = time.time(); g = np.random.default_rng(500 + ws)
    if tag == "pilot":
        m = MLP(2 * P + NSLOT, W, P, param="mup", base_width=BASE); m.load_state_dict(torch.load(f"{SP}/unlearn_design/models/bp_mup_W{W}_s{ws}.pt")); info = None
    else: m, info = pretrain(arm, W, ws)
    m.eval(); pre, new, pairs, splits = M.world(ws); cv = np.array(pre, float); R = dict(tag=tag, arm=arm, W=W, ws=ws, train=info)
    W1 = m.l1.weight.detach().double().numpy(); T = W1[:, 2 * P:2 * P + NPRE].T                     # [12, W] task columns
    xy = torch.as_tensor(pairs); lab_task = np.repeat(np.arange(NPRE), P * P); lab_ans = np.concatenate([(pairs.sum(1) + c) % P for c in pre])
    with torch.no_grad():
        Z1 = torch.cat([m.l1.weight[:, xy[:, 0]].T + m.l1.weight[:, P + xy[:, 1]].T + m.l1.weight[:, 2 * P + k] + m.l1.bias for k in range(NPRE)])
        H1 = torch.tanh(Z1); H2 = torch.tanh(m.l2(H1)); logits = m.head(H2)
    base_acc = float((logits.argmax(-1).numpy() == lab_ans).mean()); R["base_acc_all_pairs"] = base_acc
    H1n, H2n = H1.double().numpy(), H2.double().numpy()
    C1 = np.stack([H1n[lab_task == k].mean(0) for k in range(NPRE)]); C2 = np.stack([H2n[lab_task == k].mean(0) for k in range(NPRE)])
    R["a_taskcols"] = ring_test(T, cv, g); R["b_h1"] = ring_test(C1, cv, g); R["b_h2"] = ring_test(C2, cv, g)
    R["c_loo_ring_R2"] = {f"K{K}": dict(real=float(loo_ring(T, cv, K)), null_p95=float(np.quantile([loo_ring(T, g.permutation(cv), K) for _ in range(200)], 0.95))) for K in (1, 2, 3)}
    Ma = np.stack([H2n[lab_ans == v].mean(0) for v in range(P)]); W3 = m.l3.weight.detach().double().numpy()
    R["d_h2_answer"] = spec_rows(Ma, g); R["d_W3"] = spec_rows(W3, g)
    # (e) mid-network erasure of task identity (class means made equal), propagated forward
    tr_idx = np.concatenate([k * P * P + splits[k][:144] for k in range(NPRE)]); te_idx = np.concatenate([k * P * P + splits[k][144:] for k in range(NPRE)])
    def erase(H, lab, ncls):
        mu = H.mean(0); Mk = np.stack([H[lab == k].mean(0) - mu for k in range(ncls)], 1); Q = orth(Mk)[:, :ncls - 1]; return H - (H - mu) @ Q @ Q.T
    E = {}
    for layer, H in (("h1", H1n), ("h2", H2n)):
        He = erase(H, lab_task, NPRE)
        with torch.no_grad():
            Ht = torch.as_tensor(He, dtype=torch.float32); lg = m.head(torch.tanh(m.l2(Ht))) if layer == "h1" else m.head(Ht)
        acc_e = float((lg.argmax(-1).numpy()[te_idx] == lab_ans[te_idx]).mean())
        sub = g.permutation(len(tr_idx))[:2400]
        E[layer] = dict(acc_heldout_after=acc_e,
                        task_probe_linear_before=ridge_probe(H[tr_idx], lab_task[tr_idx], H[te_idx], lab_task[te_idx], NPRE),
                        task_probe_linear_after=ridge_probe(He[tr_idx], lab_task[tr_idx], He[te_idx], lab_task[te_idx], NPRE),
                        task_probe_mlp_after=mlp_probe(He[tr_idx][sub], lab_task[tr_idx][sub], He[te_idx], lab_task[te_idx], NPRE))
    # control: erase a random subspace of the same dimension (11) at the same layer
    for layer, H in (("h1", H1n), ("h2", H2n)):
        Qr = orth(g.standard_normal((W, NPRE - 1))); mu = H.mean(0); Hr = H - (H - mu) @ Qr @ Qr.T
        with torch.no_grad():
            Ht = torch.as_tensor(Hr, dtype=torch.float32); lg = m.head(torch.tanh(m.l2(Ht))) if layer == "h1" else m.head(Ht)
        E[layer]["acc_heldout_rand11"] = float((lg.argmax(-1).numpy()[te_idx] == lab_ans[te_idx]).mean())
    R["e_erasure"] = E; R["sec"] = time.time() - t0
    json.dump(R, open(f"{OUT}/c1_{tag}_{arm}_W{W}_s{ws}.json", "w"), indent=1)
    a, b1, b2 = R["a_taskcols"], R["b_h1"], R["b_h2"]
    print(f"[{tag} {arm} W{W} s{ws}] train {info} base {base_acc:.3f} | taskcol max share {a['max']:.3f} (k={a['kmax']}, null max p99 {a['null_max_p99']:.3f}, p {a['p']:.3f}) "
          f"h1 {b1['max']:.3f} (p99 {b1['null_max_p99']:.3f}) h2 {b2['max']:.3f} (p99 {b2['null_max_p99']:.3f}) | LOO ring R2 " + " ".join(f"{k} {v['real']:+.2f}(p95 {v['null_p95']:+.2f})" for k, v in R['c_loo_ring_R2'].items())
          + f" | h2-answer top1 {R['d_h2_answer']['top1']:.2f} (p99 {R['d_h2_answer']['null_top1_p99']:.2f}) W3 top1 {R['d_W3']['top1']:.2f} (p99 {R['d_W3']['null_top1_p99']:.2f})", flush=True)
    for layer in ("h1", "h2"):
        e = E[layer]; print(f"    erase task means at {layer}: held-out acc {e['acc_heldout_after']:.3f} (rand11 {e['acc_heldout_rand11']:.3f}); task probe linear {e['task_probe_linear_before']:.2f} -> {e['task_probe_linear_after']:.2f}, MLP after {e['task_probe_mlp_after']:.2f} (chance 0.083) ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

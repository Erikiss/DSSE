"""Early sham: Fisher ratio of h2 clouds grouped by x, from step 1 (like task_early), gamma +-0.003, +-0.01; compare to task_early."""
import sys, json, numpy as np, torch
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/rgi_design/crit")
exec(open("sham.py").read().split("out = []; specs")[0])
import torch.nn.functional as Fn
def train2(ws, init, gamma, W=128):
    pre, new, pairs, splits = M.world(ws)
    tr = [M.data(pre[k], k, pairs[splits[k][:144]]) for k in range(NPRE)]; va = [M.data(pre[k], k, pairs[splits[k][144:]]) for k in range(NPRE)]
    Xtr = torch.cat([d[0] for d in tr]); ytr = torch.cat([d[1] for d in tr]); Xva = torch.cat([d[0] for d in va]); yva = torch.cat([d[1] for d in va])
    lab_x = torch.cat([torch.as_tensor(pairs[splits[k][:144]][:, 0]) for k in range(NPRE)])
    torch.manual_seed(ws if init == 0 else 10_000 * init + ws); m = mk("bp_mup", W); opt = M.make_opt(m); first = ck = None
    for s in range(1, M.MAXSTEPS + M.AFTER + 1):
        if s > M.MAXSTEPS and first is None: break
        opt.zero_grad(); h1, h2 = m.acts(Xtr); loss = Fn.cross_entropy(m.head(h2), ytr) + gamma * torch.log(M.task_fisher(h2, lab_x, P)); loss.backward(); opt.step()
        if s % 50 == 0 and first is None and accuracy(m, Xva, yva) >= M.LEVEL1: first = s
        if first is not None and s >= first + M.AFTER: ck = s; break
    return m, dict(first=first, ck=ck, heldout=accuracy(m, Xva, yva))
old = json.load(open("sham.json"))
for ws in (0, 1):
    for init in (0, 1):
        for g in (0.003, -0.003, 0.01, -0.01):
            m, info = train2(ws, init, g); pre, new, pairs, splits = M.world(ws)
            if info["ck"] is None: print(f"w{ws} i{init} x_early g {g:+.3f}: fails level 1 (held-out {info['heldout']:.3f})"); continue
            d = M.delta_stats(m, pre, pairs, ws); L = lam(m, ws)
            te = [r for r in old if r["ws"] == ws and r["init"] == init and r["mode"] == "task_early" and abs(r["gamma"] - g) < 1e-9]
            tes = f"task_early same g: delta {te[0]['delta_h2']:.3f} Lambda {te[0]['Lambda']:.3f}" if te and te[0]["Lambda"] is not None else "task_early same g: failed"
            print(f"w{ws} i{init} x_early g {g:+.3f}: held-out {info['heldout']:.3f} first95 {info['first']} | delta h2 {d['h2']['delta']:.3f} F_x {xdelta(m, ws):.3f} | Lambda {L:.3f}  || {tes}", flush=True)

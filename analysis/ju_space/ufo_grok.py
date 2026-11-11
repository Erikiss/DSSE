"""Grokking of division modulo 97 in a 2-layer transformer, after Sheppard et al. (arXiv 2611.11862), Section 5 and Appendix C.
Purpose here: watch what a NORMALLY trained network (weight decay only, no UFO) develops on the way to generalisation, and
compare with one run that is trained with an intercomponent MLP UFO (a 2-dim autoencoder on the z-scored MLP activations at
the '=' token whose reconstruction error is an auxiliary loss for the transformer; the UFO itself is updated selectively).
Checkpoints are saved for ufo_grok_probe.py.
Architecture as in Appendix C: d_model 128, 4 heads, MLP 512 with ReLU, layer norm AFTER each attention and MLP block, learned
token embeddings, fixed sinusoidal positions, AdamW lr 1e-3, betas (0.9, 0.98), eps 1e-8, 10 warm-up steps, weight decay 0.03
on all parameters, batches of 512 (nine full batches and one of 48 per epoch), half of the 9312 problems for training.
Not specified in the paper and chosen here: untied output layer with bias, ReLU in the UFO autoencoder, initialisation (PyTorch defaults).
Mode perm: as base, but the answers of the training problems are permuted once at random (nothing to generalise).
Usage: python ufo_grok.py <out_dir> <base|ufo|perm> <seed> [max_steps] [weight_decay]"""
import json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn

torch.set_num_threads(int(os.environ.get("NT", "3"))); P = 97; D, H, NH, NL = 128, 512, 4, 2; EOS, DIV, EQ = 99, 97, 98


def problems():
    pr = [(x, y, (x * pow(y, -1, P)) % P) for x in range(P) for y in range(1, P)]
    seq = torch.tensor([[EOS, x, DIV, y, EQ, z, EOS] for x, y, z in pr]); g = torch.Generator().manual_seed(12345)      # fixed split for all runs
    perm = torch.randperm(len(seq), generator=g); return seq[perm[: len(seq) // 2]], seq[perm[len(seq) // 2:]]


class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.qkv = nn.Linear(D, 3 * D); s.o = nn.Linear(D, D); s.ln1 = nn.LayerNorm(D); s.fc1 = nn.Linear(D, H); s.fc2 = nn.Linear(H, D); s.ln2 = nn.LayerNorm(D)

    def forward(s, x, clamp=None):
        B, T, _ = x.shape; q, k, v = s.qkv(x).view(B, T, 3, NH, D // NH).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(D // NH); att = att.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf")).softmax(-1)
        a = s.o((att @ v).transpose(1, 2).reshape(B, T, D)); x = s.ln1(x + a); h = torch.relu(s.fc1(x))
        if clamp is not None: h = clamp(h)
        return s.ln2(x + s.fc2(h)), h, att


class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(100, D); s.blocks = nn.ModuleList([Block() for _ in range(NL)]); s.head = nn.Linear(D, 100)
        pos = torch.arange(8)[:, None]; i = torch.arange(0, D, 2)[None]; pe = torch.zeros(8, D); pe[:, 0::2] = torch.sin(pos / 10000 ** (i / D)); pe[:, 1::2] = torch.cos(pos / 10000 ** (i / D)); s.register_buffer("pe", pe)

    def forward(s, tok, inject=None):
        """inject: dict layer -> function applied to the residual state AFTER that block (layer -1 = after the embedding)."""
        x = s.emb(tok) + s.pe[: tok.shape[1]]; hs, st = [], []
        if inject and -1 in inject: x = inject[-1](x)
        for l, b in enumerate(s.blocks):
            x, h, _ = b(x); hs.append(h)
            if inject and l in inject: x = inject[l](x)
            st.append(x)
        return s.head(x), hs, st


def acc(net, seq):
    with torch.no_grad():
        lg = net(seq[:, :6])[0]; return float(((lg[:, 4].argmax(-1) == seq[:, 5]) & (lg[:, 5].argmax(-1) == seq[:, 6])).float().mean())


if __name__ == "__main__":
    out, MODE, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3]); MAXS = int(sys.argv[4]) if len(sys.argv) > 4 else 160000
    WD = float(sys.argv[5]) if len(sys.argv) > 5 else 0.03
    run = f"{out}/grok_{MODE}_s{SEED}" + ("" if WD == 0.03 else f"_wd{WD:g}"); os.makedirs(run, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED); tr, va = problems(); net = Net()
    if MODE == "perm":                                                                     # control that cannot generalise: the answers of the training problems are permuted once at random
        tr = tr.clone(); tr[:, 5] = tr[torch.randperm(len(tr), generator=torch.Generator().manual_seed(777)), 5]
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=WD)
    if MODE == "ufo":
        ufo = nn.Sequential(nn.Linear(H + NL, 128), nn.ReLU(), nn.Linear(128, 2), nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, H)); uopt = torch.optim.Adam(ufo.parameters(), lr=1e-3); LSET = 1.0
    SAVE = sorted(set([0, 250, 500, 1000, 2000, 3000, 5000, 7500] + list(range(10000, 40001, 2500)) + list(range(40000, MAXS + 1, 5000)))); TAIL = int(os.environ.get("TAIL", "20000")); log = []; step = 0; grok = None; t0 = time.time(); g = torch.Generator().manual_seed(SEED)

    def zs(h): return (h - h.mean(0, keepdim=True)) / (h.std(0, keepdim=True) + 1e-6)          # z-score every hidden node over the batch

    def reps(hs): return [torch.cat([Fn.one_hot(torch.full((h.shape[0],), l), NL).float(), zs(h[:, 4])], 1) for l, h in enumerate(hs)]

    def save(): torch.save(dict(net=net.state_dict(), ufo=ufo.state_dict() if MODE == "ufo" else None, step=step), f"{run}/ckpt_{step}.pt")
    save(); done = False
    while not done:
        perm = torch.randperm(len(tr), generator=g)
        for i in range(0, len(tr), 512):
            b = tr[perm[i:i + 512]]; lr = 1e-3 * min(1.0, (step + 1) / 10)
            for gr in opt.param_groups: gr["lr"] = lr
            lg, hs, _ = net(b[:, :6]); task = Fn.cross_entropy(lg[:, 4], b[:, 5]) + Fn.cross_entropy(lg[:, 5], b[:, 6]); loss = task; aux = torch.tensor(0.0)
            if MODE == "ufo":
                R = reps(hs); aux = torch.stack([((ufo(r) - r[:, NL:]) ** 2).mean() for r in R]).mean(); lam = LSET * min(1.0, max(0.0, (step - 250) / 250)); loss = task + lam * aux
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if MODE == "ufo":                                                                  # selective update of the UFO: only the best-reconstructed examples of each layer
                s_t = 0.25 - 0.15 * min(1.0, step / 2000); n_sel = max(1, int(round(s_t * len(b)))); ls = []
                for r in R:
                    r = r.detach(); e = ((ufo(r) - r[:, NL:]) ** 2).mean(1); ls.append(e[e.argsort()[:n_sel]].mean())
                uopt.zero_grad(set_to_none=True); torch.stack(ls).mean().backward(); uopt.step()
            step += 1
            if step in SAVE: save()
            if step % 1000 == 0:
                a_tr, a_va = acc(net, tr[:1024]), acc(net, va); wn = float(sum(float((p.detach() ** 2).sum()) for p in net.parameters()) ** .5)
                log.append(dict(step=step, train_acc=a_tr, val_acc=a_va, task=float(task.detach()), aux=float(aux.detach()), wnorm=wn, sec=time.time() - t0)); json.dump(dict(log=log, grok=grok), open(f"{run}/log.json", "w"))
                if step % 5000 == 0 or (a_va >= 0.99 and grok is None): print(f"{MODE} s{SEED} step {step}: train {a_tr:.3f} val {a_va:.3f} task {float(task):.4f} aux {float(aux):.4f} |w| {wn:.1f} ({time.time() - t0:.0f}s)", flush=True)
                if a_va >= 0.99 and grok is None: grok = step; save(); SAVE += [grok + k for k in (1000, 2000, 5000, 10000, 20000)]
            if step >= MAXS or (grok is not None and step >= grok + TAIL): done = True; break
    save(); json.dump(dict(log=log, grok=grok), open(f"{run}/log.json", "w")); print(f"finished {MODE} s{SEED}: grok step {grok}, {step} steps, {time.time() - t0:.0f}s", flush=True)

"""Training arms for 'order and disorder in the key-value cache' on the toy task of ufo_grok.py (division modulo 97, same
architecture, optimiser and data split). One CPU thread per run, so that runs of different arms differ only in the arm.
Arms (argument <mode>):
  base      weight decay only
  kvnoise   noise on the keys and values of positions 0..3 in both blocks during training (size KVF * spread over the batch)
  knoise / vnoise   the same on the keys only / the values only
  swiglu    weight decay only, but the MLPs are gated: down(silu(gate(x)) * up(x)) with 341 hidden units (the same number of MLP
            weights as 512 ReLU units); the activation of Qwen3
  bilinear  the same without the SiLU: down(gate(x) * up(x)), a polynomial of degree 2
  kswap     for a random half of the examples of every batch the keys of positions 0..3 that '=' reads in both blocks come from
            another problem of the batch (the addresses are scrambled, the values stay the example's own)
  kswapiso  control for kswap: the cache stays clean; the '=' stream gets a vector in a random direction that is exactly as long as
            the change the scrambled keys would have caused there
  eqiso     control for kvnoise: the cache stays clean; the stream of the '=' position gets, after the attention step of both blocks, a vector
            in a random direction that is exactly as long as the change the cache noise of kvnoise would have caused there
  eqnoise   noise of size KVF * spread on the stream of the '=' position after the attention step of both blocks
  kvufo     a UFO on the cache: an autoencoder with a 2-dim bottleneck on the z-scored values of the two operand positions in
            both blocks (code for block and position at the encoder), co-trained as in the paper (reconstruction error as an
            auxiliary loss from step 250, full from step 500; the UFO itself is updated on its best-reconstructed examples)
  attufo    the paper's own intercomponent attention UFO: the attention weights of the '=' query row of all heads of a block,
            z-scored over the batch, autoencoder 8-2-8
Continuation of a trained network on a subset of its training problems (ungrokking):
  env INIT=<checkpoint>, SUB=<fraction of the training problems> (STRAT=1: drawn so that every residue occurs at least twice as x,
      as y and as answer where possible; DRAW=<number of the draw>; without SUB: plain continuation = drift control), CKPT=<steps
      between checkpoints, 1000>, FREEZE=<parameter group that is not updated>:
      emb | attn | mlp | pre (everything that produces the cache of both blocks: embedding, block 0, the key/value/query
      weights of block 1) | post (everything behind the read of block 1: its output weights, its MLP, its norms, the output layer)
The first step at which at least 99 % of the held-out problems are right (checked every EVAL steps) is the grok step.
Usage: python kv_grok.py <out_dir> <mode> <seed> [max_steps=20000] [weight_decay=0.3]
       env: KVF (default 0.5), EVAL (500), TAIL (steps after the grok step, 500), NT (1), INIT, SUB, STRAT, DRAW, CKPT, FREEZE, TAG,
            STOP_BELOW (stop when held-out accuracy has been below this value at four evaluations in a row)"""
import json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_common import fwd, make_net, problems, subset, covered, D, NH, NL, CP

out, MODE, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3]); MAXS = int(sys.argv[4]) if len(sys.argv) > 4 else 20000; WD = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
torch.set_num_threads(int(os.environ.get("NT", "1"))); KVF = float(os.environ.get("KVF", "0.5")); EVAL = int(os.environ.get("EVAL", "500")); TAIL = int(os.environ.get("TAIL", "500")); INIT = os.environ.get("INIT"); SUB = float(os.environ.get("SUB", "0")); FREEZE = os.environ.get("FREEZE", ""); STOP_BELOW = float(os.environ.get("STOP_BELOW", "0"))
run = f"{out}/kv_{MODE}_s{SEED}" + os.environ.get("TAG", ""); os.makedirs(run, exist_ok=True); torch.manual_seed(SEED); np.random.seed(SEED); tr, va = problems(); net = make_net(MODE if MODE in ("swiglu", "bilinear") else "relu"); dropped = None
if INIT: net.load_state_dict(torch.load(INIT, weights_only=True)["net"])
cov = None
if SUB:                                                                    # ungrokking: keep a fixed part of the training problems
    ik, idr = subset(tr, SUB, os.environ.get("STRAT") == "1", int(os.environ.get("DRAW", "0"))); dropped = tr[idr]; tr = tr[ik]; cov = covered(tr, va)
    print(f"kept {len(tr)} training problems; least count of a residue as x / y / answer: {int(torch.bincount(tr[:, 1], minlength=97)[:97].min())} / {int(torch.bincount(tr[:, 3], minlength=97)[1:97].min())} / {int(torch.bincount(tr[:, 5], minlength=97)[:97].min())}; held-out problems with x, y and answer covered: {float(cov.float().mean()):.3f}", flush=True)
pre = lambda n: n.startswith("emb") or n.startswith("blocks.0.") or n.startswith("blocks.1.qkv")
GROUPS = dict(emb=lambda n: n.startswith("emb"), attn=lambda n: ".qkv." in n or ".o." in n, mlp=lambda n: ".fc1." in n or ".fc2." in n, pre=pre, post=lambda n: not pre(n))
assert FREEZE == "" or FREEZE in GROUPS; frozen = [n for n, _ in net.named_parameters() if FREEZE and GROUPS[FREEZE](n)]; params = [p for n, p in net.named_parameters() if n not in frozen]
if FREEZE: print(f"frozen ({FREEZE}): {len(frozen)} of {len(list(net.named_parameters()))} tensors", flush=True)
opt = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=WD)
if MODE == "kvufo": ufo = nn.Sequential(nn.Linear(D + 2 * NL, 128), nn.ReLU(), nn.Linear(128, 2), nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, D))
if MODE == "attufo": ufo = nn.Sequential(nn.Linear(NH * 5 + NL, 8), nn.ReLU(), nn.Linear(8, 2), nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, NH * 5))
if MODE in ("kvufo", "attufo"): uopt = torch.optim.Adam(ufo.parameters(), lr=1e-3)
g = torch.Generator().manual_seed(SEED); gn = torch.Generator().manual_seed(10_000 + SEED); log = []; step = 0; grok = None; t0 = time.time()
zs = lambda h: (h - h.mean(0, keepdim=True)) / (h.std(0, keepdim=True) + 1e-6)
ctl = {"kvnoise": dict(kv_noise=KVF, kv_parts="kv", gen=gn), "knoise": dict(kv_noise=KVF, kv_parts="k", gen=gn), "vnoise": dict(kv_noise=KVF, kv_parts="v", gen=gn), "eqnoise": dict(eq_noise=KVF, gen=gn), "eqiso": dict(kv_noise=KVF, kv_parts="kv", eq_match="iso", gen=gn)}.get(MODE)
assert ctl is not None or MODE in ("base", "kvufo", "attufo", "kswap", "kswapiso", "swiglu", "bilinear"), MODE


def make_ctl(B):
    if MODE not in ("kswap", "kswapiso"): return ctl
    idx = torch.arange(B); m = torch.rand(B, generator=gn) < 0.5; idx[m] = torch.randint(0, B, (int(m.sum()),), generator=gn); c = dict(swap={(l, "k"): idx for l in range(NL)}, gen=gn)
    if MODE == "kswapiso": c["eq_match"] = "iso"
    return c


def reps(rec):                                                             # what the UFO sees: list of (input with code, target)
    if MODE == "kvufo": return [(torch.cat([Fn.one_hot(torch.full((len(r["v"]),), 2 * l + j), 2 * NL).float(), z], 1), z) for l, r in enumerate(rec) for j, p in enumerate((1, 3)) for z in [zs(r["v"][:, :, p].reshape(len(r["v"]), D))]]
    return [(torch.cat([Fn.one_hot(torch.full((len(r["att"]),), l), NL).float(), z], 1), z) for l, r in enumerate(rec) for z in [zs(r["att"][:, :, 4, :5].reshape(len(r["att"]), NH * 5))]]


def evaluate(seq):
    with torch.no_grad():
        lg = fwd(net, seq[:, :6])[0]; return float(((lg[:, 4].argmax(-1) == seq[:, 5]) & (lg[:, 5].argmax(-1) == seq[:, 6])).float().mean())


def save(): torch.save(dict(net=net.state_dict(), step=step, bilinear=(MODE == "bilinear")), f"{run}/ckpt_{step}.pt")


save(); done = False
while not done:
    perm = torch.randperm(len(tr), generator=g)
    for i in range(0, len(tr), 512):
        b = tr[perm[i:i + 512]]; lr = 1e-3 * min(1.0, (step + 1) / 10)
        for gr in opt.param_groups: gr["lr"] = lr
        lg, rec = fwd(net, b[:, :6], make_ctl(len(b))); task = Fn.cross_entropy(lg[:, 4], b[:, 5]) + Fn.cross_entropy(lg[:, 5], b[:, 6]); loss = task; aux = torch.tensor(0.0)
        if MODE in ("kvufo", "attufo"):
            R = reps(rec); aux = torch.stack([((ufo(a) - z) ** 2).mean() for a, z in R]).mean(); loss = task + min(1.0, max(0.0, (step - 250) / 250)) * aux
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if MODE in ("kvufo", "attufo"):                                         # selective update of the UFO on its best-reconstructed examples
            n_sel = max(1, int(round((0.25 - 0.15 * min(1.0, step / 2000)) * len(b)))); ls = []
            for a, z in R:
                e = ((ufo(a.detach()) - z.detach()) ** 2).mean(1); ls.append(e[e.argsort()[:n_sel]].mean())
            uopt.zero_grad(set_to_none=True); torch.stack(ls).mean().backward(); uopt.step()
        step += 1
        if step % EVAL == 0:
            row = dict(step=step, train_acc=evaluate(tr[:1024]), val_acc=evaluate(va), task=float(task.detach()), aux=float(aux.detach()), wnorm=float(sum(float((p.detach() ** 2).sum()) for p in net.parameters()) ** .5), sec=time.time() - t0)
            if dropped is not None: row["dropped_acc"] = evaluate(dropped); row["val_covered"] = evaluate(va[cov]); row["val_uncovered"] = evaluate(va[~cov]) if int((~cov).sum()) else None
            if rec[0]["dmid"] is not None: row["dmid"] = [r["dmid"] for r in rec]
            with torch.no_grad(): row["val_loss"] = float(Fn.cross_entropy(fwd(net, va[:, :6])[0][:, 4], va[:, 5]))
            log.append(row); json.dump(dict(log=log, grok=grok, mode=MODE, seed=SEED, kvf=KVF, sub=SUB, freeze=FREEZE, init=INIT), open(f"{run}/log.json", "w"))
            if step % (10 * EVAL) == 0 or (row["val_acc"] >= 0.99 and grok is None): print(f"{MODE} s{SEED} step {step}: train {row['train_acc']:.3f} val {row['val_acc']:.3f}" + (f" dropped {row['dropped_acc']:.3f}" if dropped is not None else "") + f" task {row['task']:.4f} aux {row['aux']:.4f} |w| {row['wnorm']:.1f} ({row['sec']:.0f}s)", flush=True)
            if row["val_acc"] >= 0.99 and grok is None and not INIT: grok = step; save()
            if step % 2500 == 0 or (INIT and step % int(os.environ.get("CKPT", "1000")) == 0): save()
        if step >= MAXS or (grok is not None and step >= grok + TAIL): done = True; break
        if STOP_BELOW and len(log) >= 4 and step % EVAL == 0 and all(r["val_acc"] < STOP_BELOW for r in log[-4:]): done = True; break
save(); json.dump(dict(log=log, grok=grok, mode=MODE, seed=SEED, kvf=KVF, sub=SUB, freeze=FREEZE, init=INIT, steps=step), open(f"{run}/log.json", "w")); print(f"finished {MODE} s{SEED}: grok step {grok}, {step} steps, {time.time() - t0:.0f}s", flush=True)

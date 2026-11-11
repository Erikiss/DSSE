"""Two training-time variants of ungrokking for the toy transformer (see kv_grok.py and kv_common.py).

1. Untied network. A generalising network G is copied twice: the WRITER computes positions 0..3 (and with them the key-value
   cache of both blocks), the READER computes position 4 ('=') and the answer position and reads the writer's cache. With
   both copies equal to G this is exactly G. Then training continues on a small part of the training problems with only the
   writer, only the reader, or both copies trainable (weight decay acts on the trainable copies only). Question: can the
   network ungrok through the cache alone, through the reading path alone, and which is faster? This is a different model
   class from the tied network (twice the weights), so it locates nothing in the tied network; it shows what each path CAN do.
2. Rare instead of absent (train = tied, MIX = p): the tied network is trained further on the small part, but a share p of
   every batch is drawn at random from the dropped training problems. Question: how rare may the rest become before the
   network stops generalising?

Usage: python kv_untied.py <out_dir> <ckpt_G> <writer|reader|both|tied> <seed> [max_steps=12000] [weight_decay=0.3]
       env: SUB (0.05), STRAT (1), DRAW (0), MIX (0), EVAL (250), STOP_BELOW (0.02), TAG, NT (1)"""
import json, os, sys, time
import torch, torch.nn.functional as Fn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_common import fwd, load, problems, subset, cache, NL

out, pG, TRAIN, SEED = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); MAXS = int(sys.argv[5]) if len(sys.argv) > 5 else 12000; WD = float(sys.argv[6]) if len(sys.argv) > 6 else 0.3
torch.set_num_threads(int(os.environ.get("NT", "1"))); SUB = float(os.environ.get("SUB", "0.05")); MIX = float(os.environ.get("MIX", "0")); EVAL = int(os.environ.get("EVAL", "250")); STOP_BELOW = float(os.environ.get("STOP_BELOW", "0.02"))
run = f"{out}/kv_{TRAIN}_s{SEED}" + os.environ.get("TAG", ""); os.makedirs(run, exist_ok=True); torch.manual_seed(SEED); tr, va = problems(); ik, idr = subset(tr, SUB, os.environ.get("STRAT", "1") == "1", int(os.environ.get("DRAW", "0"))); kept, dropped = tr[ik], tr[idr]
W, Rd = load(pG), load(pG); W.train(); Rd.train(); assert TRAIN in ("writer", "reader", "both", "tied") and (MIX == 0 or TRAIN == "tied")
params = {"writer": list(W.parameters()), "reader": list(Rd.parameters()), "both": list(W.parameters()) + list(Rd.parameters()), "tied": list(W.parameters())}[TRAIN]
opt = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=WD); g = torch.Generator().manual_seed(SEED); n_mix = int(round(MIX / (1 - MIX) * min(512, len(kept)))) if MIX else 0


def logits(tok):
    if TRAIN == "tied": return fwd(W, tok)[0]
    _, rw = fwd(W, tok); return fwd(Rd, tok, dict(set_kv={l: cache(rw, l) for l in range(NL)}))[0]


def evaluate(seq):
    with torch.no_grad():
        lg = logits(seq[:, :6]); return float(((lg[:, 4].argmax(-1) == seq[:, 5]) & (lg[:, 5].argmax(-1) == seq[:, 6])).float().mean())


wn = lambda net: float(sum(float((p.detach() ** 2).sum()) for p in net.parameters()) ** .5); log = []; step = 0; t0 = time.time(); done = False
print(f"{TRAIN}: kept {len(kept)} training problems, {n_mix} rare problems per batch; start: held-out {evaluate(va):.3f}, kept {evaluate(kept):.3f}", flush=True)
while not done:
    perm = torch.randperm(len(kept), generator=g)
    for i in range(0, len(kept), 512):
        b = kept[perm[i:i + 512]]
        if n_mix: b = torch.cat([b, dropped[torch.randint(0, len(dropped), (n_mix,), generator=g)]])
        for gr in opt.param_groups: gr["lr"] = 1e-3 * min(1.0, (step + 1) / 10)
        lg = logits(b[:, :6]); loss = Fn.cross_entropy(lg[:, 4], b[:, 5]) + Fn.cross_entropy(lg[:, 5], b[:, 6]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
        if step % EVAL == 0:
            row = dict(step=step, kept_acc=evaluate(kept), val_acc=evaluate(va), dropped_acc=evaluate(dropped), task=float(loss.detach()), wnorm_writer=wn(W), wnorm_reader=wn(Rd), sec=time.time() - t0); log.append(row)
            json.dump(dict(log=log, train=TRAIN, seed=SEED, sub=SUB, mix=MIX, n_mix=n_mix, init=pG), open(f"{run}/log.json", "w"))
            if step % (8 * EVAL) == 0: print(f"{TRAIN} step {step}: kept {row['kept_acc']:.3f} held-out {row['val_acc']:.3f} dropped {row['dropped_acc']:.3f} |writer| {row['wnorm_writer']:.1f} |reader| {row['wnorm_reader']:.1f} ({row['sec']:.0f}s)", flush=True)
            if step % 2000 == 0: torch.save(dict(net=W.state_dict(), writer=W.state_dict(), reader=Rd.state_dict(), step=step), f"{run}/ckpt_{step}.pt")
        if step >= MAXS or (STOP_BELOW and len(log) >= 4 and step % EVAL == 0 and all(r["val_acc"] < STOP_BELOW for r in log[-4:])): done = True; break
torch.save(dict(net=W.state_dict(), writer=W.state_dict(), reader=Rd.state_dict(), step=step), f"{run}/ckpt_{step}.pt"); print(f"finished {TRAIN}: {step} steps, held-out {log[-1]['val_acc']:.3f}, {time.time() - t0:.0f}s", flush=True)

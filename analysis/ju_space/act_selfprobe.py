"""Can a network's own gate factors be read from its final state? (README 9.3; description. By the rule fixed in advance nothing
here can count FOR "the network has a representation of its activation function": later states are functions of earlier
activations by construction. It can only show whether anything is specific to the network's own gates.)
For every stored state of a run: ridge regression from regressors to targets at the '=' position, fitted on the training problems;
reported is the share of the variance of the targets explained on the held-out problems (all units pooled). Per block:
  own          regressor: final state of the '=' position (128 numbers); target: the network's own gate factors silu(gate(x))
               (ReLU: the masks; bilinear: gate(x))
  random       the same regressor; target: gate factors computed with freshly drawn random gate weights (same scale as at
               initialisation) on the network's own MLP inputs. own minus random = what is specific to the trained gates
  pre          the same regressor; target: the gate input before the SiLU (gated units only)
  other        regressor: the final '=' state of every OTHER run's last state; target: this network's own gate factors
  untrained    the whole measurement "own" for the untrained network of the run (step 0)
  operands     regressor: one-hot codes of x and of y (194 numbers); target: own gate factors
Usage: python act_selfprobe.py <out.json> <run_dir> [<run_dir> ...]      (runs of one kind; at least two)      env: STEPS (default: all stored states)"""
import glob, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_common import fwd, load, all_problems, acc4, D

torch.set_num_threads(int(os.environ.get("NT", "1"))); out, dirs = sys.argv[1], sys.argv[2:]; seq, is_tr = all_problems(); tok = seq[:, :5]; ONE = torch.cat([torch.nn.functional.one_hot(seq[:, 1], 97), torch.nn.functional.one_hot(seq[:, 3], 97)], 1).float()
silu = torch.nn.functional.silu


def read(R, T):
    """Ridge from R [N, k] to T [N, units] on the training problems (strength 1e-3 of the mean diagonal); share explained on the held-out problems."""
    R, T = R.double(), T.double(); mu, tm = R[is_tr].mean(0), T[is_tr].mean(0); A, B = R[is_tr] - mu, T[is_tr] - tm; G = A.T @ A; W = torch.linalg.solve(G + 1e-3 * G.diagonal().mean() * torch.eye(G.shape[0], dtype=torch.float64), A.T @ B)
    E = (T[~is_tr] - tm) - (R[~is_tr] - mu) @ W; return float(1 - (E ** 2).sum() / ((T[~is_tr] - T[~is_tr].mean(0)) ** 2).sum().clamp_min(1e-300))


def facts(path, gen):
    net = load(path); b0 = net.blocks[0]; kind = "relu" if not hasattr(b0, "gate") else ("bilinear" if b0.bilinear else "swiglu")
    with torch.no_grad():
        lg, rec = fwd(net, tok, dict(rec_gate=True)); rnd = []
        for r in rec:                                                     # random gates on the network's own MLP input (weights and biases as nn.Linear draws them)
            H = r["gf"].shape[-1]; bound = 1 / D ** .5; W = (torch.rand(H, D, generator=gen) * 2 - 1) * bound; bb = (torch.rand(H, generator=gen) * 2 - 1) * bound; gi = r["xmid"][:, 4] @ W.T + bb
            rnd.append((gi > 0).float() if kind == "relu" else (gi if kind == "bilinear" else silu(gi)))
    return dict(kind=kind, acc=(acc4(lg[is_tr], seq[is_tr]), acc4(lg[~is_tr], seq[~is_tr])), final=rec[-1]["x"][:, 4], gf=[r["gf"][:, 4] for r in rec], gi=[r["gi"][:, 4] for r in rec], rnd=rnd)


def measure(f, others):
    row = []
    for l in range(len(f["gf"])):
        own, rnd = read(f["final"], f["gf"][l]), read(f["final"], f["rnd"][l]); d = dict(own=own, random=rnd, own_minus_random=own - rnd, operands=read(ONE, f["gf"][l]))
        if f["kind"] != "relu": d["pre"] = read(f["final"], f["gi"][l])
        if others: oth = [read(o, f["gf"][l]) for o in others]; d["other"] = sum(oth) / len(oth)
        row.append(d)
    return row


cks = lambda d: sorted(int(os.path.basename(f)[5:-3]) for f in glob.glob(f"{d}/ckpt_*.pt")); gen = torch.Generator().manual_seed(11); LAST = {d: facts(f"{d}/ckpt_{cks(d)[-1]}.pt", gen)["final"] for d in dirs}; res = {}
for d in dirs:
    steps = [int(s) for s in os.environ["STEPS"].split(",")] if os.environ.get("STEPS") else cks(d); rows = []; others = [LAST[o] for o in dirs if o != d]
    for st in steps:
        f = facts(f"{d}/ckpt_{st}.pt", gen); row = dict(step=st, acc_train=f["acc"][0], acc_held=f["acc"][1], blocks=measure(f, others if st > 0 else None)); rows.append(row)
        print(os.path.basename(d), st, f"acc {row['acc_train']:.3f}/{row['acc_held']:.3f}", " | ".join(f"b{l}: own {b['own']:.3f} random {b['random']:.3f} (diff {b['own_minus_random']:+.3f})" + (f" pre {b['pre']:.3f}" if "pre" in b else "") + (f" other {b['other']:.3f}" if "other" in b else "") + f" operands {b['operands']:.3f}" for l, b in enumerate(row["blocks"])), flush=True)
    res[os.path.basename(d)] = rows; json.dump(res, open(out, "w"), indent=1)

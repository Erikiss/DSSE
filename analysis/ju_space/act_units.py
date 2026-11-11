"""Do the hidden units of the MLPs depend on one combination of the two operands? (README 9.4; description only. That a
generalising network has parts that depend on the answer alone follows from the task and cannot count for the conjecture.)
The analogue, in the toy task, of the features in Bottazzi's post that depend on x + c a alone: here the one combination that
matters is the answer z = x / y (in the multiplicative group: the difference of the discrete logarithms of x and y).
Per stored state and block, over all 9312 problems, read at '=', the variance is split as in split() of kv_probe.py:
  x, y        explained by x alone and by y alone (group means; the grid of problems is balanced)
  z_beyond    what the answer adds to the additive model of x and y (joint least-squares model of x, y and z minus the two)
  rest        what no additive model of x, y and z explains
for (a) the activations of the hidden units, pooled with the weight each unit has in the MLP write (variance of its activation
times the squared norm of its output weights), and (b) the state of the '=' position after the block. Also: the weighted share
of units whose activation is explained by the answer alone (97 group means) to at least 0.5 and to at least 0.9. An activation
unrelated to the answer still gets 96 / 9311 = 0.0103 from the group means.
Usage: python act_units.py <out.json> <run_dir> [<run_dir> ...]      env: STEPS (comma list; default: all stored states)"""
import glob, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_common import fwd, load, all_problems, acc4, P

torch.set_num_threads(int(os.environ.get("NT", "1"))); out, dirs = sys.argv[1], sys.argv[2:]; seq, is_tr = all_problems(); tok = seq[:, :5]; X, Y, Z = seq[:, 1], seq[:, 3], seq[:, 5]
oh = lambda v, n: torch.nn.functional.one_hot(v, n).double(); Dj = torch.cat([oh(X, P), oh(Y, P)[:, 1:], oh(Z, P)], 1); Dj = Dj - Dj.mean(0); Uj, sj, _ = torch.linalg.svd(Dj, full_matrices=False); Qj = Uj[:, sj > 1e-8 * sj[0]]      # joint model x + y + z


def split(T, w):
    """Per column of T [9312, k] the shares by x, by y, by z beyond both and the rest; returned pooled with the weights w (sum 1)."""
    T = T.double(); G = T.view(P, P - 1, -1); gm = G.mean((0, 1)); tot = ((G - gm) ** 2).sum((0, 1)).clamp_min(1e-300); bx = (P - 1) * ((G.mean(1) - gm) ** 2).sum(0) / tot; by = P * ((G.mean(0) - gm) ** 2).sum(0) / tot; joint = ((Qj.T @ (T - gm)) ** 2).sum(0) / tot
    w = w.double(); return dict(x=float((bx * w).sum()), y=float((by * w).sum()), z_beyond=float(((joint - bx - by) * w).sum()), rest=float(((1 - joint) * w).sum()))


def by_answer(T):
    T = T.double(); n = torch.bincount(Z, minlength=P).double(); m = torch.zeros(P, T.shape[1], dtype=torch.float64).index_add_(0, Z, T) / n[:, None]; g = T.mean(0); return ((n[:, None] * (m - g) ** 2).sum(0) / len(T)) / T.var(0, unbiased=False).clamp_min(1e-300)


res = {}
for d in dirs:
    cks = sorted(int(os.path.basename(f)[5:-3]) for f in glob.glob(f"{d}/ckpt_*.pt")); pick = [int(s) for s in os.environ["STEPS"].split(",")] if os.environ.get("STEPS") else cks; rows = []
    for st in pick:
        if st not in cks: continue
        net = load(f"{d}/ckpt_{st}.pt")
        with torch.no_grad(): lg, rec = fwd(net, tok)
        row = dict(step=st, acc_train=acc4(lg[is_tr], seq[is_tr]), acc_held=acc4(lg[~is_tr], seq[~is_tr]), blocks=[])
        for l, b in enumerate(net.blocks):
            h = rec[l]["h"][:, 4]; Wd = b.down.weight if hasattr(b, "down") else b.fc2.weight; wv = h.var(0, unbiased=False) * (Wd ** 2).sum(0); wv = wv / wv.sum(); sz = by_answer(h); s = rec[l]["x"][:, 4]; tv = s.var(0, unbiased=False); tw = tv / tv.sum()
            row["blocks"].append(dict(units=split(h, wv), state=split(s, tw), units_by_answer_alone=float((sz * wv.double()).sum()), weight_of_units_answer_0_5=float(wv[sz > 0.5].sum()), weight_of_units_answer_0_9=float(wv[sz > 0.9].sum()), number_of_units_answer_0_5=int((sz > 0.5).sum()), number_of_units=int(len(sz)), state_by_answer_alone=float((by_answer(s) * tw.double()).sum())))
        rows.append(row); print(os.path.basename(d), st, f"acc {row['acc_train']:.3f}/{row['acc_held']:.3f}", " | ".join(f"b{l}: units x {b['units']['x']:.2f} y {b['units']['y']:.2f} z beyond {b['units']['z_beyond']:.2f} rest {b['units']['rest']:.2f}, answer units weight {b['weight_of_units_answer_0_5']:.2f}; state x {b['state']['x']:.2f} y {b['state']['y']:.2f} z beyond {b['state']['z_beyond']:.2f} rest {b['state']['rest']:.2f}" for l, b in enumerate(row["blocks"])), flush=True)
    res[os.path.basename(d)] = rows; json.dump(res, open(out, "w"), indent=1)

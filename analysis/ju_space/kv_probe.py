"""Route through and order of the key-value cache of the toy transformer (see kv_common.py), per checkpoint of a training run.
The cache of the '=' query: keys and values of positions ("slots") 0..3 (EOS, x, '/', y) in block 0 and block 1. Slots 0 and 2 of
block 0 and slot 0 of block 1 are the same for every problem; slots 1 and 3 of block 0 are look-ups of x and of y (the same
token table up to a constant); slots 1 and 2 of block 1 depend on x only; slot 3 of block 1 can depend on x and y.
The '=' token itself is the same in every problem, so every answer depends on the cache completely (with the whole cache of
another problem the network gives that problem's answer; checked as donor_all = 1). What can differ is the ROUTE:
  A  answers kept with foreign keys: share of problems whose answer stays the same when only the KEYS of both blocks come
     from another problem (A0 / A1: keys of block 0 / block 1 only). High = the read pattern does not depend on the content
     of the cache; low = the answer depends on reading the problem's own keys. A does not show whether the read pattern
     itself is fixed: att_shift[l] gives the share of the attention weight of the '=' row in block l that moves when the
     keys are foreign (total variation, mean over heads and pairs), and logit_spread how much the logits vary over problems.
     An untrained network can score high on A, so A has to be read together with the accuracy.
  L  late-read share: block 0 keeps the problem's own cache, block 1 gets the cache of another problem; L = share of problems
     that then give the other problem's answer (L_own: share that keep their own answer; the rest gives neither).
     E = the mirror image (block 0 from the other problem, block 1 own; share that give the other problem's answer).
  S  combination site: share of the variance of what slot 3 of block 1 would write that is explained by the answer z beyond
     x and y; S_read = share of held-out problems whose answer a linear read-out of the state of position 3 after block 0
     gives (ridge regression on the training problems, strength fixed at 1e-3 of the mean diagonal; a high S_read moves by
     about 0.1 with this choice, a value at chance does not); S_eq = the same read-out from the '=' state after block 0.
  All over pairs of problems with different answers (true answers and the network's own), 3 draws of the other problems
  pooled, on 2048 training problems ("train") and 2048 held-out problems ("held"); None if fewer than 100 such pairs exist
  (an untrained network gives nearly the same answer for every problem).
  att_cache, att_pos, att_entropy: attention of the '=' row (weight on slots 0..3, per position, entropy in nats; max ln 5 = 1.61)
  logit_spread[l]: spread of the attention logits of the '=' row over all 9312 problems (standard deviation in nats, mean
     over heads and slots 1 and 3), taken relative to the mean of the row over its five slots, because a shift common to all
     slots does not change the read pattern; 0 = the read pattern is the same for every problem. logit_spread_raw: without
     subtracting the row mean.
  acc_own_labels: for the control run with permuted answers, the share of training problems on which the network gives
     the (permuted) answer it was trained on; acc_train is always measured against the true answers.
ORDER of the cache content, measured on quantities that do not depend on the choice of basis inside a head: the vector a slot
would write into the reading stream (its value passed through that head's part of the output weights; 4 heads x 128 numbers).
  Tables with one row per non-zero residue (96 rows): w0 = block-0 slot of an operand; w1x, w1s = block-1 slots 1 and 2;
  w1y / w1xm = block-1 slot 3 averaged over x for every y / over y for every x; q0 = attention logits of the '=' row of block 0
  for the operand slot (4 heads). Per table:
    top5 = share of the variance over the rows in the 5 strongest of 48 frequencies when the rows are ordered by the
           multiplicative group (x = g^a, a = 0..95); top5_shuffled: rows in random order (mean of 20); top5_additive: rows in
           the order 1..96; n_eff = effective number of frequencies; pca2 / pca8 / dims: share in 2 / 8 principal axes and the
           effective number of axes; spread = root mean square distance of the rows from their mean, as a fraction of the root
           mean square length of the '=' state at the input of the block, to which the vectors are added (0 = the slot holds
           nothing)
  split: for block-1 slot 3 (w1p3), the '=' state at the input of block 1 (eq_in1) and after block 1 (eq_out): share of the
         variance over all 9312 problems explained by x alone, by y alone, by the answer z beyond both; rest = everything else
TOLERANCE to disorder: share of problems whose answer stays the same under
  noise_kv_f / noise_k_f / noise_v_f   noise of size f * spread on the cached keys and values / keys / values of both blocks
  iso_f      a vector of the same length as the change that noise_kv_f causes in the '=' stream, in a random direction, added
             to the stream after the attention step instead (cache clean); dmid_f = that length relative to the spread of the stream
Usage: python kv_probe.py <run_dir> [step,step,...]      writes <run_dir>/kvprobe.json      env: NT (1), TAG"""
import glob, json, math, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_common import fwd, load, ans, all_problems, dlog, problems, D, NH, NL, P, DH

run = sys.argv[1]; torch.set_num_threads(int(os.environ.get("NT", "1"))); torch.set_grad_enabled(False); tr, va = problems(); ALL, is_tr = all_problems(); g_root, XS = dlog()
steps = sorted(int(os.path.basename(f)[5:-3]) for f in glob.glob(f"{run}/ckpt_*.pt"))
if len(sys.argv) > 2: steps = [s for s in steps if s in {int(t) for t in sys.argv[2].split(",")}]
NS = 2048; SETS = dict(train=tr[:NS], held=va[:NS]); FS = (0.5, 1.0, 2.0); ix = torch.tensor(XS); zall = ALL[:, 5]; ND = 3
oh = lambda lab, n: torch.nn.functional.one_hot(lab, n).double(); Dj = torch.cat([oh(ALL[:, 1], P), oh(ALL[:, 3] - 1, P - 1), oh(zall, P)], 1)
Uj, sj, _ = torch.linalg.svd(Dj, full_matrices=False); Qj = Uj[:, sj > 1e-8 * sj[0]]                                 # orthonormal basis of the joint model x + y + z


def spectrum(M):
    M = M.double(); M = M - M.mean(0); p = (torch.fft.rfft(M, dim=0).abs() ** 2).sum(1)[1:]; p = torch.cat([2 * p[:-1], p[-1:]]) if M.shape[0] % 2 == 0 else 2 * p; return p / p.sum().clamp_min(1e-300)


def order(M, gen, scale, nat=None):
    """M: [96, d], rows in the order of the multiplicative group; nat: the same rows in the order 1..96."""
    s = spectrum(M); top = s.sort(descending=True); t5 = lambda X: float(spectrum(X).sort(descending=True).values[:5].sum()); sh = sum(t5(M[torch.randperm(len(M), generator=gen)]) for _ in range(20)) / 20
    C = (M - M.mean(0)).double(); ev = torch.linalg.svdvals(C) ** 2; ev = ev / ev.sum().clamp_min(1e-300)
    return dict(top5=float(top.values[:5].sum()), top1=float(top.values[0]), n_eff=float(1 / (s ** 2).sum()), top5_shuffled=sh, top5_additive=None if nat is None else t5(nat), freqs=[int(i) + 1 for i in top.indices[:5]], pca2=float(ev[:2].sum()), pca8=float(ev[:8].sum()),
                dims=float(1 / (ev ** 2).sum()), spread=float(C.pow(2).sum(1).mean().sqrt() / scale))


def split(T):
    T = T.double(); G = T.view(P, P - 1, -1); gm = G.mean((0, 1)); tot = ((G - gm) ** 2).sum().clamp_min(1e-300); bx = float((P - 1) * ((G.mean(1) - gm) ** 2).sum() / tot); by = float(P * ((G.mean(0) - gm) ** 2).sum() / tot)
    joint = float(((Qj.T @ (T - gm)) ** 2).sum() / tot); return dict(by_x=bx, by_y=by, z_beyond=joint - bx - by, rest=1 - joint)


def readout(X):
    """Ridge regression from a state [9312, D] to the one-hot answer, fitted on the training problems; share right on the held-out problems."""
    X = torch.cat([X.double(), torch.ones(len(X), 1).double()], 1); A, Y = X[is_tr], oh(zall[is_tr], P); W = torch.linalg.solve(A.T @ A + 1e-3 * (A ** 2).sum() / A.shape[1] * torch.eye(A.shape[1]).double(), A.T @ Y)
    return float(((X[~is_tr] @ W).argmax(1) == zall[~is_tr]).float().mean())


res = []
for st in steps:
    net = load(f"{run}/ckpt_{st}.pt"); gen = torch.Generator().manual_seed(1000 + st); lgA, recA = fwd(net, ALL[:, :6]); okA = ans(lgA) == zall; row = dict(step=st, acc_train=float(okA[is_tr].float().mean()), acc_held=float(okA[~is_tr].float().mean()))
    def written(l, pos):                                                  # what the slot would write through each head: [9312, heads * D]
        v = recA[l]["v"][:, :, pos]; Wo = net.blocks[l].o.weight; return torch.cat([v[:, h] @ Wo[:, h * DH:(h + 1) * DH].T for h in range(NH)], 1)
    per_x = lambda T: T.view(P, P - 1, -1)[:, 0]; ymean = lambda T: T.view(P, P - 1, -1).mean(0); xmean = lambda T: T.view(P, P - 1, -1).mean(1); nat = torch.arange(1, P)
    sc = [float(recA[l]["xin"][:, 4].double().pow(2).sum(1).mean().sqrt()) for l in range(NL)]; w0 = per_x(written(0, 1)); w1x, w1s = per_x(written(1, 1)), per_x(written(1, 2)); w13 = written(1, 3); w1y, w1xm = ymean(w13), xmean(w13)
    q0 = per_x((recA[0]["att"][:, :, 4, :5].clamp_min(1e-30).log())[:, :, 1] - (recA[0]["att"][:, :, 4, :5].clamp_min(1e-30).log())[:, :, 0])                       # logit of the x slot relative to the constant slot 0, per head: [97, heads]
    row["order"] = dict(w0=order(w0[ix], gen, sc[0], w0[nat]), w1x=order(w1x[ix], gen, sc[1], w1x[nat]), w1s=order(w1s[ix], gen, sc[1], w1s[nat]), w1y=order(w1y[ix - 1], gen, sc[1], w1y), w1xm=order(w1xm[ix], gen, sc[1], w1xm[nat]), q0=order(q0[ix], gen, 1.0, q0[nat]))
    row["split"] = dict(w1p3=split(w13), eq_in1=split(recA[1]["xin"][:, 4]), eq_out=split(recA[1]["x"][:, 4])); row["S"] = row["split"]["w1p3"]["z_beyond"]; row["S_read"] = readout(recA[1]["xin"][:, 3]); row["S_eq"] = readout(recA[1]["xin"][:, 4])
    row["att_cache"], row["att_pos"], row["att_entropy"], row["logit_spread"], row["logit_spread_raw"] = [], [], [], [], []
    if "perm" in os.path.basename(os.path.normpath(run)):                 # the labels of ufo_grok.py, mode perm
        pl = tr[torch.randperm(len(tr), generator=torch.Generator().manual_seed(777)), 5]; row["acc_own_labels"] = float((ans(fwd(net, tr[:, :6])[0]) == pl).float().mean())
    for l in range(NL):
        a = recA[l]["att"][:, :, 4, :5].double(); row["att_cache"].append(float(a[..., :4].sum(-1).mean())); row["att_pos"].append([float(t) for t in a.mean((0, 1))]); row["att_entropy"].append(float(-(a * a.clamp_min(1e-30).log()).sum(-1).mean()))
        q, k = recA[l]["q"][:, :, 4], recA[l]["k"]; lg = (q[:, :, None, :] * k[:, :, :5]).sum(-1) / math.sqrt(DH); row["logit_spread_raw"].append(float(lg[:, :, [1, 3]].std(0).mean())); row["logit_spread"].append(float((lg - lg.mean(-1, keepdim=True))[:, :, [1, 3]].std(0).mean()))
    row["route"], row["tolerance"] = {}, {}
    for name, seq in SETS.items():
        tok = seq[:, :6]; lg0, rec0 = fwd(net, tok); base = ans(lg0); r = {k: 0.0 for k in ("A", "A0", "A1", "L", "L_own", "E", "E_own", "donor_all")}; npairs = 0; shift = [0.0] * NL
        for _ in range(ND):
            idx = torch.randperm(NS, generator=gen); ok = (base[idx] != base) & (seq[idx, 5] != seq[:, 5]); npairs += int(ok.sum()); sh = lambda a, ref: float((a[ok] == ref[ok]).float().sum()); go = lambda ctl: ans(fwd(net, tok, ctl)[0])
            lgk, rck = fwd(net, tok, dict(swap={(0, "k"): idx, (1, "k"): idx}))
            for l in range(NL): shift[l] += float((rck[l]["att"][:, :, 4, :5] - rec0[l]["att"][:, :, 4, :5]).abs().sum(-1)[ok].mean(1).sum()) / 2      # moved attention weight, mean over heads
            r["A"] += sh(ans(lgk), base); r["A0"] += sh(go(dict(swap={(0, "k"): idx})), base); r["A1"] += sh(go(dict(swap={(1, "k"): idx})), base)
            a1 = go(dict(swap={(1, "k"): idx, (1, "v"): idx})); r["L"] += sh(a1, base[idx]); r["L_own"] += sh(a1, base); a0 = go(dict(swap={(0, "k"): idx, (0, "v"): idx})); r["E"] += sh(a0, base[idx]); r["E_own"] += sh(a0, base)
            r["donor_all"] += sh(go(dict(swap={(l, p): idx for l in range(NL) for p in "kv"})), base[idx])
        r = {k: (v / npairs if npairs >= 100 else None) for k, v in r.items()}; r["pairs"] = npairs; r["att_shift"] = [x / npairs for x in shift] if npairs >= 100 else None                         # an untrained network gives nearly the same answer everywhere: no pairs, no route
        row["route"][name] = r; t = {}; same = lambda ctl: float((ans(fwd(net, tok, ctl)[0]) == base).float().mean())
        for f in FS:
            for parts in ("kv", "k", "v"): t[f"noise_{parts}_{f:g}"] = same(dict(kv_noise=f, kv_parts=parts, gen=gen))
            lg, rc = fwd(net, tok, dict(kv_noise=f, eq_match="iso", gen=gen)); t[f"iso_{f:g}"] = float((ans(lg) == base).float().mean()); t[f"dmid_{f:g}"] = [rc[l]["dmid"] for l in range(NL)]
        row["tolerance"][name] = t
    res.append(row); o, r, t, th = row["order"], row["route"]["train"], row["tolerance"]["train"], row["tolerance"]["held"]
    r = {k: (float("nan") if v is None else v) for k, v in r.items()}
    print(f"step {st:6d} acc {row['acc_train']:.3f}/{row['acc_held']:.3f} | A {r['A']:.2f} (A0 {r['A0']:.2f}, A1 {r['A1']:.2f}) L {r['L']:.2f} (own {r['L_own']:.2f}) E {r['E']:.2f} (own {r['E_own']:.2f}) donor_all {r['donor_all']:.2f} | S {row['S']:.2f} S_read {row['S_read']:.2f} S_eq {row['S_eq']:.2f}"
          f" | logit spread {row['logit_spread'][0]:.2f}/{row['logit_spread'][1]:.2f} | top5 w0 {o['w0']['top5']:.2f} (shuffled {o['w0']['top5_shuffled']:.2f}, additive {o['w0']['top5_additive']:.2f}; spread {o['w0']['spread']:.2f}, pca8 {o['w0']['pca8']:.2f}) w1x {o['w1x']['top5']:.2f} w1s {o['w1s']['top5']:.2f} w1y {o['w1y']['top5']:.2f} q0 {o['q0']['top5']:.2f}"
          f" | w1p3 x {row['split']['w1p3']['by_x']:.2f} y {row['split']['w1p3']['by_y']:.2f} | noise 1: cache {t['noise_kv_1']:.2f} (k {t['noise_k_1']:.2f}, v {t['noise_v_1']:.2f}) iso {t['iso_1']:.2f}; held {th['noise_kv_1']:.2f} / {th['iso_1']:.2f}", flush=True)
log = json.load(open(f"{run}/log.json")) if os.path.exists(f"{run}/log.json") else {}
json.dump(dict(run=run, grok=log.get("grok"), primitive_root=g_root, rows=res), open(f"{run}/kvprobe{os.environ.get('TAG', '')}.json", "w"), indent=1); print("written", f"{run}/kvprobe{os.environ.get('TAG', '')}.json")

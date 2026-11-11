"""The toy transformer of ufo_grok.py with its key-value cache made explicit.
Sequences are [EOS, x, '/', y, '=', z, EOS]; the answer is read at position 4 ('='). When the '=' token is processed, the
keys and values of positions 0 to 3 of every block are already computed: they are the cache. In block 0 they are a function of
token and position only, in block 1 of the outputs of block 0 at these positions.
fwd(net, tok, ctl) computes the same function as net(tok). Interventions on the cache (dict ctl, all optional) change what the
positions from '=' on READ from positions 0..3 in a block; the computations of positions 0..3 themselves stay as they are,
exactly as if the stored entries had been changed after they were written:
  kv_noise=f          noise on the cached keys and values, size f * their spread over the batch (per position, head, dimension)
  kv_parts="kv"       which part gets the noise ("k", "v" or "kv");  noise_blocks=(0, 1)
  eq_match="iso"      together with kv_noise: the cache stays clean; instead the stream of the reading positions gets, after the
                      attention step of each block, a vector in a random direction with the length of the change that the cache
                      noise would have caused there ("perm": the change that it caused for another problem of the batch)
  swap={(block, "k"|"v"): index tensor}    cached keys or values are taken from other problems of the batch
  mean_kv={blocks}                         cached keys and values are replaced by their mean over the batch
  set_kv={block: (k or None, v or None)}   cached keys / values are replaced by given tensors [B, heads, 4, dh]
  add_kv={block: (dk or None, dv or None)} given tensors are added to the cached keys / values
  set_res={block: states [B, 4, D]}        the cache of that block is computed from these states (with the network's own weights)
  set_eq={block: state [B, D]}             the state of position 4 at the input of that block is replaced (not a cache intervention)
  set_gate={block: tensor [B, T, units]}   the gate factor of the MLP of that block is replaced at all positions (ReLU: the 0/1 mask;
                                           gated units: silu(gate(x)), bilinear: gate(x)); not a cache intervention
  set_h={block: function}                  the hidden activations of the MLP of that block are computed by the given function of (gate
                                           input, gated signal) [gated units] or of (fc1(x), fc1(x)) [ReLU units]; each has the shape [B, T, units]
  rec_gate=True                            record the gate factor (gf), the gated signal (gs) and the input of the gate function (gi) of every block
  eq_noise=f                               noise on the stream of position 4 after the attention step, size f * spread
  gen=torch.Generator for all random draws
Returns the logits and, per block, a dict: q, k, v (clean, [B, heads, T, dh]), att (as used, [B, heads, T, T]), h (MLP activations),
x (state after the block), xin (block input), xmid (state after the attention step, as used), dmid (its change at position 4
caused by the intervention, as a fraction of the spread of the clean state over the batch; None without cache intervention)."""
import math, os, sys
import torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ufo_grok import Net, problems, D, H, NH, NL, P  # noqa: F401


class GluBlock(nn.Module):
    """The block of ufo_grok.py with a gated MLP instead of the ReLU MLP: down(act(gate(x)) * up(x)).
    act = SiLU gives SwiGLU (as in Qwen3); act = identity gives a purely bilinear unit, a polynomial of degree 2 in x.
    linear=True: no gate at all, the MLP is the linear map down(up(x)) (an MLP without a nonlinearity)."""
    def __init__(s, hs, bilinear, linear=False):
        super().__init__(); s.qkv = nn.Linear(D, 3 * D); s.o = nn.Linear(D, D); s.ln1 = nn.LayerNorm(D); s.up = nn.Linear(D, hs); s.down = nn.Linear(hs, D); s.ln2 = nn.LayerNorm(D); s.bilinear = bilinear; s.linear = linear
        if not linear: s.gate = nn.Linear(D, hs)


class GluNet(nn.Module):
    """Net of ufo_grok.py with gated MLPs; 341 hidden units give the same number of MLP weights as 512 ReLU units (3 * 341 against 2 * 512 times 128)."""
    def __init__(s, hs=341, bilinear=False, linear=False):
        super().__init__(); s.emb = nn.Embedding(100, D); s.blocks = nn.ModuleList([GluBlock(hs, bilinear, linear) for _ in range(NL)]); s.head = nn.Linear(D, 100)
        pos = torch.arange(8)[:, None]; i = torch.arange(0, D, 2)[None]; pe = torch.zeros(8, D); pe[:, 0::2] = torch.sin(pos / 10000 ** (i / D)); pe[:, 1::2] = torch.cos(pos / 10000 ** (i / D)); s.register_buffer("pe", pe); s.bilinear = bilinear


def make_net(arch="relu"): return Net() if arch == "relu" else GluNet(bilinear=(arch == "bilinear"), linear=(arch == "linear"))


def mlp(b, x):
    """Hidden activations and the write of the MLP of block b."""
    if hasattr(b, "gate"):
        g = b.gate(x); h = (g if b.bilinear else torch.nn.functional.silu(g)) * b.up(x); return h, b.down(h)
    if hasattr(b, "up"): h = b.up(x); return h, b.down(h)                  # linear MLP
    h = torch.relu(b.fc1(x)); return h, b.fc2(h)

def mlp_parts(b, x, gate=None, hfun=None):
    """As mlp(), and also the factors of every hidden unit: (h, write, gate factor, gated signal, input of the gate function).
    ReLU unit: h = m * p with p = fc1(x) and the mask m = 1[p > 0]; the "switch" is m. Gated unit: h = a * u with a = silu(g) =
    sigmoid(g) * g (bilinear: a = g), g = gate(x), u = up(x); the switch of a SwiGLU unit is sigmoid(g).
    gate replaces the gate factor (m or a): a tensor, for example the factor of another network, or a function of (own factor,
    gated signal, input of the gate function) that returns the factor to use.
    hfun replaces the hidden activation itself: a function of (gate input, gated signal) for gated units, of (p, p) for ReLU units."""
    if hasattr(b, "gate"):
        g = b.gate(x); a = g if b.bilinear else torch.nn.functional.silu(g); u = b.up(x); f = a if gate is None else (gate(a, u, g) if callable(gate) else gate); h = f * u if hfun is None else hfun(g, u); return h, b.down(h), a, u, g
    if hasattr(b, "up"): u = b.up(x); h = u if hfun is None else hfun(u, u); return h, b.down(h), torch.ones_like(u), u, u                    # linear MLP: no gate
    p = b.fc1(x); m = (p > 0).to(p.dtype); h = (torch.relu(p) if gate is None else (gate(m, p, p) if callable(gate) else gate) * p) if hfun is None else hfun(p, p); return h, b.fc2(h), m, p, p

CP = 4                                                                    # positions 0..3 are the cache of the '=' query
DH = D // NH


def _split(t): return t.view(t.shape[0], t.shape[1], 3, NH, DH).permute(2, 0, 3, 1, 4)                                   # -> q, k, v, each [B, heads, T, dh]


def _att(q, k, mask): return ((q @ k.transpose(-1, -2)) / math.sqrt(DH)).masked_fill(mask, float("-inf")).softmax(-1)


def fwd(net, tok, ctl=None):
    ctl = ctl or {}; g = ctl.get("gen"); x = net.emb(tok) + net.pe[: tok.shape[1]]; B, T, _ = x.shape; rec = []
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1); rn = lambda t: torch.randn(t.shape, generator=g)
    for l, b in enumerate(net.blocks):
        if l in ctl.get("set_eq", {}): x = torch.cat([x[:, :4], ctl["set_eq"][l][:, None], x[:, 5:]], 1)
        xin = x; q, k, v = _split(b.qkv(x)); att = _att(q, k, mask); a = att @ v; k0, v0 = k[:, :, :CP], v[:, :, :CP]; km, vm = k0, v0; mod = False
        if l in ctl.get("set_res", {}): _, km, vm = _split(b.qkv(ctl["set_res"][l])); mod = True
        if l in ctl.get("set_kv", {}):
            sk, sv = ctl["set_kv"][l]; km = km if sk is None else sk; vm = vm if sv is None else sv; mod = True
        if l in ctl.get("mean_kv", ()): km = km.mean(0, keepdim=True).expand_as(km); vm = vm.mean(0, keepdim=True).expand_as(vm); mod = True
        if (l, "k") in ctl.get("swap", {}): km = km[ctl["swap"][(l, "k")]]; mod = True
        if (l, "v") in ctl.get("swap", {}): vm = vm[ctl["swap"][(l, "v")]]; mod = True
        if l in ctl.get("add_kv", {}):
            dk, dv = ctl["add_kv"][l]; km = km if dk is None else km + dk; vm = vm if dv is None else vm + dv; mod = True
        if ctl.get("kv_noise") and l in ctl.get("noise_blocks", (0, 1)):
            f, parts = ctl["kv_noise"], ctl.get("kv_parts", "kv"); mod = True
            if "k" in parts: km = km + f * k0.std(0, keepdim=True).detach() * rn(km)
            if "v" in parts: vm = vm + f * v0.std(0, keepdim=True).detach() * rn(vm)
        xm = x0 = b.ln1(x + b.o(a.transpose(1, 2).reshape(B, T, D))); dmid = None
        if mod and T > CP:                                                # the positions from '=' on read the changed cache
            attm = _att(q[:, :, CP:], torch.cat([km, k[:, :, CP:]], 2), mask[CP:]); am = attm @ torch.cat([vm, v[:, :, CP:]], 2)
            att = torch.cat([att[:, :, :CP], attm], 2); xm = b.ln1(x + b.o(torch.cat([a[:, :, :CP], am], 2).transpose(1, 2).reshape(B, T, D)))
            if ctl.get("eq_match"):
                d = (xm - x0)[:, CP:].detach()
                if ctl["eq_match"] == "iso": u = rn(d); d = u * (d.norm(dim=-1, keepdim=True) / u.norm(dim=-1, keepdim=True))
                else: d = d[torch.randperm(B, generator=g)]
                xm = torch.cat([x0[:, :CP], x0[:, CP:] + d], 1)
            dmid = float(((xm - x0)[:, 4].detach() ** 2).sum(-1).mean().sqrt() / ((x0[:, 4].detach() - x0[:, 4].detach().mean(0)) ** 2).sum(-1).mean().sqrt().clamp_min(1e-12))
        if ctl.get("eq_noise"): xm = torch.cat([xm[:, :4], xm[:, 4:5] + ctl["eq_noise"] * x0[:, 4:5].std(0, keepdim=True).detach() * rn(xm[:, 4:5]), xm[:, 5:]], 1)
        if ctl.get("rec_gate") or "set_gate" in ctl or "set_h" in ctl: h, w, gf, gs, gi = mlp_parts(b, xm, ctl.get("set_gate", {}).get(l), ctl.get("set_h", {}).get(l))
        else: (h, w), gf, gs, gi = mlp(b, xm), None, None, None
        x = b.ln2(xm + w); rec.append(dict(q=q, k=k, v=v, att=att, h=h, x=x, xin=xin, xmid=xm, dmid=dmid, gf=gf, gs=gs, gi=gi))
    return net.head(x), rec


def cache(rec, l): return rec[l]["k"][:, :, :CP], rec[l]["v"][:, :, :CP]


def load(path):
    ck = torch.load(path, weights_only=True); sd = ck["net"]
    net = GluNet(sd["blocks.0.gate.weight"].shape[0], bool(ck.get("bilinear"))) if "blocks.0.gate.weight" in sd else (GluNet(sd["blocks.0.up.weight"].shape[0], linear=True) if "blocks.0.up.weight" in sd else Net()); net.load_state_dict(sd); net.eval(); return net


def ans(lg): return lg[:, 4].argmax(-1)


def acc4(lg, seq): return float((ans(lg) == seq[:, 5]).float().mean())


def all_problems():
    """All 97 * 96 problems in grid order (x major, y = 1..96 minor), and for each whether it is a training problem."""
    seq = torch.tensor([[99, x, 97, y, 98, (x * pow(y, -1, P)) % P, 99] for x in range(P) for y in range(1, P)]); tr, _ = problems()
    key = lambda s: s[:, 1] * 100 + s[:, 3]; return seq, torch.isin(key(seq), key(tr))


def dlog():
    """Smallest primitive root g of 97 and the list of x = g^a for a = 0..95 (the order of the multiplicative group)."""
    for g in range(2, P):
        xs = [pow(g, a, P) for a in range(P - 1)]
        if len(set(xs)) == P - 1: return g, xs


def subset(tr, frac, strat=False, draw=0):
    """Indices of the training problems that are kept when a network is trained further on a part of them.
    strat=False: the first round(frac * n) of a random order (seed 4242 + draw). strat=True: chosen one by one so that every
    residue occurs as often as possible at least twice as x, as y and as answer (greedy, ties broken at random), then filled up
    at random. Returns the kept indices and the remaining ones."""
    g = torch.Generator().manual_seed(4242 + draw); n = len(tr); n_sub = int(round(frac * n)); pm = torch.randperm(n, generator=g)
    if not strat: return pm[:n_sub], pm[n_sub:]
    need = [torch.full((100,), 2), torch.full((100,), 2), torch.full((100,), 2)]; cols = [tr[:, 1], tr[:, 3], tr[:, 5]]; need[1][0] = 0; taken = torch.zeros(n, dtype=torch.bool); kept = []
    for c, nd in zip(cols, need): nd[torch.bincount(c, minlength=100) == 0] = 0                                    # residues that never occur in this role
    while len(kept) < n_sub:
        gain = sum((nd[c] > 0).long() for c, nd in zip(cols, need)); gain[taken] = -1; best = int(gain.max())
        if best <= 0: break
        cand = (gain == best).nonzero()[:, 0]; i = int(cand[torch.randint(0, len(cand), (1,), generator=g)]); kept.append(i); taken[i] = True
        for c, nd in zip(cols, need): nd[c[i]] = max(0, int(nd[c[i]]) - 1)
    rest = [int(i) for i in pm if not taken[i]]; kept += rest[: n_sub - len(kept)]; kept = torch.tensor(kept); taken[kept] = True
    return kept, torch.tensor([int(i) for i in pm if not taken[i]])


def covered(sub, seq, k=1):
    """For every problem of seq: do its x, its y and its answer each occur at least k times in that role in the problems sub?"""
    return (torch.bincount(sub[:, 1], minlength=100)[seq[:, 1]] >= k) & (torch.bincount(sub[:, 3], minlength=100)[seq[:, 3]] >= k) & (torch.bincount(sub[:, 5], minlength=100)[seq[:, 5]] >= k)

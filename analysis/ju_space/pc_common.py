"""README section 12: shared code for the predictive-coding toy test of 'intrinsic learning to learn' (pc_*.py).
Task family: modular linear maps f_ab(x, y) = (a x + b y) mod P over Z_P; input one-hot x, one-hot y, one-hot task slot; output P
classes. A two-hidden-layer tanh MLP, trained either by backprop (BP) or by predictive coding (PC, own implementation after
Whittington and Bogacz 2017 and the review of Millidge, Seth and Buckley 2022): value nodes h1, h2 for the hidden layers,
initialised by a feedforward pass; energy F = 1/2 ||h1 - tanh(W1 x + b1)||^2 + 1/2 ||h2 - tanh(W2 h1 + b2)||^2
+ CE(W3 h2 + b3, target), summed over the units and averaged over the batch; T steps of gradient descent on h1, h2 (step eta_x)
with the weights fixed; then the weight gradient of F at the relaxed values, which is local (each W enters one term with its own
inputs), handed to the same optimiser as BP. No third-party PC code is used."""
import math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn


def task_pairs(P):
    """All (x, y) pairs of Z_P x Z_P in a fixed order."""
    return [(x, y) for x in range(P) for y in range(P)]


def make_split(P, a, b, frac, seed):
    """Indices of the training and held-out pairs of task (a, b) (a fixed random split per task), and the labels."""
    pr = task_pairs(P); g = np.random.default_rng(seed); perm = g.permutation(len(pr)); n = int(round(frac * len(pr)))
    lab = np.array([(a * x + b * y) % P for x, y in pr]); return perm[:n], perm[n:], lab


def encode(P, nslot, pairs_xy, slot):
    """One-hot inputs [N, 2P + nslot] for the given pairs and task slot."""
    X = torch.zeros(len(pairs_xy), 2 * P + nslot)
    for i, (x, y) in enumerate(pairs_xy): X[i, x] = 1; X[i, P + y] = 1; X[i, 2 * P + slot] = 1
    return X


class MLP(nn.Module):
    """Two tanh hidden layers of width W. param: 'sp' (PyTorch default init) or 'mup' (maximal update parameterisation for Adam,
    Yang and Hu 2021, Table 3: hidden init N(0, 1/fan_in), output layer zero-initialised and scaled by base_width/width; the
    learning-rate scaling is applied through param_groups())."""
    def __init__(s, din, W, dout, param="sp", base_width=64):
        super().__init__(); s.l1 = nn.Linear(din, W); s.l2 = nn.Linear(W, W); s.l3 = nn.Linear(W, dout); s.param, s.W, s.base = param, W, base_width
        if param == "mup":
            nn.init.normal_(s.l1.weight, std=1 / math.sqrt(din)); nn.init.zeros_(s.l1.bias)
            nn.init.normal_(s.l2.weight, std=1 / math.sqrt(W)); nn.init.zeros_(s.l2.bias)
            nn.init.zeros_(s.l3.weight); nn.init.zeros_(s.l3.bias)
        s.out_mult = (base_width / W) if param == "mup" else 1.0

    def acts(s, x):
        h1 = torch.tanh(s.l1(x)); h2 = torch.tanh(s.l2(h1)); return h1, h2

    def head(s, h2):
        return s.l3(h2) * s.out_mult

    def forward(s, x):
        return s.head(s.acts(x)[1])

    def param_groups(s, lr):
        """muP for Adam: hidden (matrix-like) weights get lr * base/width; input and output layers and biases keep lr."""
        if s.param != "mup": return [{"params": s.parameters(), "lr": lr}]
        hid = [s.l2.weight]; rest = [p for n, p in s.named_parameters() if n != "l2.weight"]
        return [{"params": hid, "lr": lr * s.base / s.W}, {"params": rest, "lr": lr}]


BETA = [1.0]          # precision of the output term (cross-entropy weight), set by the caller


def pc_energy(m, x, h1, h2, y):
    """Predictive-coding energy (mean over the batch of the per-sample sum); the output term is weighted by BETA[0]."""
    e1 = h1 - torch.tanh(m.l1(x)); e2 = h2 - torch.tanh(m.l2(h1))
    return 0.5 * (e1 ** 2).sum(-1).mean() + 0.5 * (e2 ** 2).sum(-1).mean() + BETA[0] * Fn.cross_entropy(m.head(h2), y)


def pc_step(m, opt, x, y, T, eta_x):
    """One PC update: feedforward init, T inference steps on the hidden values with weights fixed, local weight gradients at the
    relaxed values, optimiser step. Returns the energy before and after inference."""
    with torch.no_grad(): h1, h2 = m.acts(x)
    h1 = h1.clone().requires_grad_(True); h2 = h2.clone().requires_grad_(True)
    for p in m.parameters(): p.requires_grad_(False)
    E0 = None
    for t in range(T):
        E = pc_energy(m, x, h1, h2, y)
        if E0 is None: E0 = float(E)
        g1, g2 = torch.autograd.grad(E, (h1, h2))
        with torch.no_grad(): h1 -= eta_x * g1 * x.shape[0]; h2 -= eta_x * g2 * x.shape[0]   # per-sample step (the energy is a batch mean)
    for p in m.parameters(): p.requires_grad_(True)
    h1, h2 = h1.detach(), h2.detach(); opt.zero_grad(); E = pc_energy(m, x, h1, h2, y); E.backward(); opt.step()
    return E0, float(E)


def bp_step(m, opt, x, y):
    opt.zero_grad(); L = Fn.cross_entropy(m(x), y); L.backward(); opt.step(); return float(L)


@torch.no_grad()
def accuracy(m, x, y):
    return float((m(x).argmax(-1) == y).float().mean()) if len(y) else float("nan")

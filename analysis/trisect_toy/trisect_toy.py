"""Toy test of Tang's decomposition on a task with a known intermediate: angle trisector vs. angle bisector length.

Task T: s = |CQ|, the angle trisector of C nearer A, from the side lengths (a, b, c).
        s = a b sin(3θ) / (a sin 2θ + b sin θ), θ = C/3. In cosine terms this needs cos(C/3) from cos C, i.e. a cubic
        4x^3 - 3x = cos C (casus irreducibilis); Pehlivan & Williams (AMJ 13/1, 2026) give s as a root of R s^3 + S s^2 + T = 0.
Task B (control): l = bisector length = 2ab cos(C/2)/(a+b); cos(C/2) = sqrt((1+cos C)/2) needs only a square root.

Same MLP as Bottazzi (WSRI26): 3-64-64-64-15-1, ReLU. Per hidden layer:
  probes:   ridge R^2 (held out) for a, b, c, cos C, angle C, cos(C/3), cos(C/2), sin-terms, target
  patching: for pairs with equal (a, b) and different C, set the linear C-coordinate of layer L to the other triangle's
            value (minimal-norm edit along the probe weight), split the edit into the component along the average output
            Jacobian at layer L ("answer direction", the scalar analogue of the J-lens) and the orthogonal rest, apply
            each alone and measure the share of the counterfactual output change (regression coefficient).
Usage: python trisect_toy.py [n_seeds]
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
OUT = Path(__file__).parent
N_TRAIN, N_TEST, N_PAIRS, EPOCHS = 200_000, 20_000, 4_000, 40
SEEDS = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
WIDTHS = [64, 64, 64, 15]


def triangles(n, rng, same_ab=None):
    if same_ab is None:
        a, b = rng.uniform(0.5, 2.0, n), rng.uniform(0.5, 2.0, n)
    else:
        a, b = same_ab
    C = rng.uniform(0.15, math.pi - 0.15, len(a))
    c = np.sqrt(a * a + b * b - 2 * a * b * np.cos(C))
    return a, b, c, C


def features(a, b, c, C):
    th = C / 3
    s = a * b * np.sin(3 * th) / (a * np.sin(2 * th) + b * np.sin(th))
    l = 2 * a * b * np.cos(C / 2) / (a + b)
    return dict(a=a, b=b, c=c, cosC=np.cos(C), C=C, cosC3=np.cos(C / 3), cosC2=np.cos(C / 2), s=s, l=l)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [3] + WIDTHS
        self.hidden = nn.ModuleList(nn.Linear(i, o) for i, o in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(dims[-1], 1)

    def forward(self, x, edit_layer=None, edit=None, return_h=False):
        hs = []
        for k, lin in enumerate(self.hidden):
            x = torch.relu(lin(x))
            if edit_layer == k:
                x = x + edit
            hs.append(x)
        y = self.out(x).squeeze(-1)
        return (y, hs) if return_h else y


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def ridge(X, y, alpha=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = (X - mu) / sd
    A = Z.T @ Z / len(Z) + alpha * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (y - y.mean()) / len(Z))
    w_raw = w / sd                      # in activation coordinates
    b0 = y.mean() - mu @ w_raw
    return w_raw, b0


def run(task, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    tr, te = features(*triangles(N_TRAIN, rng)), features(*triangles(N_TEST, rng))
    X = lambda F: np.stack([F["a"], F["b"], F["c"]], 1)
    xm, xs = X(tr).mean(0), X(tr).std(0)
    ym, ys = tr[task].mean(), tr[task].std()
    Xtr = torch.tensor((X(tr) - xm) / xs, dtype=torch.float32); Ytr = torch.tensor((tr[task] - ym) / ys, dtype=torch.float32)
    Xte = torch.tensor((X(te) - xm) / xs, dtype=torch.float32)
    m = MLP()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    steps = EPOCHS * math.ceil(N_TRAIN / 512)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-5)
    t0 = time.time()
    for ep in range(EPOCHS):
        perm = torch.randperm(N_TRAIN)
        for i in range(0, N_TRAIN, 512):
            idx = perm[i:i + 512]
            loss = ((m(Xtr[idx]) - Ytr[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval()
    with torch.no_grad():
        yte, Hte = m(Xte, return_h=True); _, Htr = m(Xtr[:50_000], return_h=True)
    yte = yte.numpy() * ys + ym
    res = dict(task=task, seed=seed, test_r2=float(r2(te[task], yte)), minutes=(time.time() - t0) / 60, layers=[])
    trf = {k: v[:50_000] for k, v in tr.items()}
    # Pairs: same (a, b), different C
    a2, b2 = rng.uniform(0.5, 2.0, N_PAIRS), rng.uniform(0.5, 2.0, N_PAIRS)
    P1, P2 = features(*triangles(N_PAIRS, rng, (a2, b2))), features(*triangles(N_PAIRS, rng, (a2, b2)))
    X1 = torch.tensor((X(P1) - xm) / xs, dtype=torch.float32); X2 = torch.tensor((X(P2) - xm) / xs, dtype=torch.float32)
    with torch.no_grad():
        y1, H1 = m(X1, return_h=True); y2, H2 = m(X2, return_h=True)
    dy_star = (y2 - y1).numpy()
    coef = lambda dy: float((dy * dy_star).sum() / (dy_star ** 2).sum())   # share of the counterfactual change
    for L in range(len(WIDTHS)):
        Htr_L, Hte_L = Htr[L].numpy(), Hte[L].numpy()
        probes = {}
        for f in ("a", "b", "c", "cosC", "C", "cosC3", "cosC2", "s", "l"):
            w, b0 = ridge(Htr_L, trf[f])
            probes[f] = float(r2(te[f], Hte_L @ w + b0))
        # incremental: what the layer holds about C (angle) beyond cos C and the model's own output, linearly
        base = np.stack([te["cosC"], yte], 1)
        res_C = te["C"] - np.c_[base, np.ones(len(base))] @ np.linalg.lstsq(np.c_[base, np.ones(len(base))], te["C"], rcond=None)[0]
        w_r, b_r = ridge(Hte_L[:10_000], res_C[:10_000])
        probes["C_beyond_cosC_and_output"] = float(r2(res_C[10_000:], Hte_L[10_000:] @ w_r + b_r))
        # answer direction: average Jacobian of the output w.r.t. layer L
        h = Hte[L][:5000].clone().requires_grad_(True)
        xx = h
        for lin in m.hidden[L + 1:]:
            xx = torch.relu(lin(xx))
        m.out(xx).sum().backward()
        J = h.grad.mean(0).numpy(); jhat = J / np.linalg.norm(J)
        # intermediate swap along the C-probe (minimal-norm edit), and the answer swap along jhat
        wC, bC = ridge(Htr_L, trf["C"])
        dp = (H2[L].numpy() @ wC) - (H1[L].numpy() @ wC)
        delta = dp[:, None] * wC[None, :] / (wC @ wC)
        d_ans = (delta @ jhat)[:, None] * jhat[None, :]
        d_orth = delta - d_ans
        eff = {}
        with torch.no_grad():
            for name, d in (("full", delta), ("answer_part", d_ans), ("orth_part", d_orth)):
                eff[name] = coef((m(X1, edit_layer=L, edit=torch.tensor(d, dtype=torch.float32)) - y1).numpy())
            # (2) full activation difference between the two triangles (they differ only in C): Tang split of the real change
            D = (H2[L] - H1[L]).numpy()
            Da = (D @ jhat)[:, None] * jhat[None, :]
            for name, d in (("diff_full", D), ("diff_answer_part", Da), ("diff_orth_part", D - Da)):
                eff[name] = coef((m(X1, edit_layer=L, edit=torch.tensor(d, dtype=torch.float32)) - y1).numpy())
            # (3) encoding direction of C: regression of the activation difference on the difference in C
            dC = (P2["C"] - P1["C"])
            uC = (D * dC[:, None]).sum(0) / (dC ** 2).sum()
            E = dC[:, None] * uC[None, :]
            Ea = (E @ jhat)[:, None] * jhat[None, :]
            for name, d in (("enc_full", E), ("enc_answer_part", Ea), ("enc_orth_part", E - Ea)):
                eff[name] = coef((m(X1, edit_layer=L, edit=torch.tensor(d, dtype=torch.float32)) - y1).numpy())
            eff["cos_enc_answer"] = float(abs(uC @ jhat) / np.linalg.norm(uC))
            eff["share_norm_diff_along_answer"] = float(np.mean((D @ jhat) ** 2) / np.mean((D ** 2).sum(1)))
            d_answer_swap = dy_star[:, None] / np.linalg.norm(J) * jhat[None, :]
            eff["answer_swap"] = coef((m(X1, edit_layer=L, edit=torch.tensor(d_answer_swap, dtype=torch.float32)) - y1).numpy())
        res["layers"].append(dict(layer=L + 1, width=WIDTHS[L], probes=probes, effect=eff,
                                  cos_Cprobe_answer=float(abs(wC @ jhat) / np.linalg.norm(wC)),
                                  dead_share=float((Hte_L.max(0) <= 0).mean())))
    return res


if __name__ == "__main__":
    out = []
    for task in ("s", "l"):
        for seed in SEEDS:
            r = run(task, seed); out.append(r)
            print(f"{'Trisektor' if task == 's' else 'Halbierende'} seed {seed}: Test-R² {r['test_r2']:.5f} ({r['minutes']:.1f} min)", flush=True)
            for Ly in r["layers"]:
                p, e = Ly["probes"], Ly["effect"]
                print(f"   L{Ly['layer']}: R² cosC {p['cosC']:.3f} C {p['C']:.3f} cos(C/3) {p['cosC3']:.3f} cos(C/2) {p['cosC2']:.3f} Ziel {p[task]:.3f}"
                      f" | C über cosC+Ausgabe {p['C_beyond_cosC_and_output']:+.3f} | Tausch C: voll {e['full']:.2f}, Antwortteil {e['answer_part']:.2f},"
                      f" Rest {e['orth_part']:.2f}; Antwort-Tausch {e['answer_swap']:.2f} | cos(C-Sonde, Antwort) {Ly['cos_Cprobe_answer']:.2f} | tot {Ly['dead_share']:.2f}", flush=True)
    json.dump(out, open(OUT / "results.json", "w"), indent=1)
    # summary over seeds
    print("\nMittel über Seeds, kausale Eingriffe (Anteil der kontrafaktischen Änderung):")
    for task in ("s", "l"):
        rs = [r for r in out if r["task"] == task]
        for L in range(len(WIDTHS)):
            g = lambda k: np.mean([r["layers"][L]["effect"][k] for r in rs])
            print(f"  {'Trisektor ' if task == 's' else 'Halbierende'} L{L + 1}: Differenz voll {g('diff_full'):.2f} | Antwortteil {g('diff_answer_part'):.2f} | Rest {g('diff_orth_part'):.2f}"
                  f" | Normanteil entlang Antwort {g('share_norm_diff_along_answer'):.3f} || Kodier-Richtung C: voll {g('enc_full'):.2f} | Antwortteil {g('enc_answer_part'):.2f} | Rest {g('enc_orth_part'):.2f} | cos {g('cos_enc_answer'):.2f}")
    print("\nMittel über Seeds (Anteil der kontrafaktischen Änderung):")
    for task in ("s", "l"):
        rs = [r for r in out if r["task"] == task]
        for L in range(len(WIDTHS)):
            g = lambda k: np.mean([r["layers"][L]["effect"][k] for r in rs])
            gp = lambda k: np.mean([r["layers"][L]["probes"][k] for r in rs])
            print(f"  {'Trisektor ' if task == 's' else 'Halbierende'} L{L + 1}: voll {g('full'):.2f} | Antwortteil {g('answer_part'):.2f} | Rest {g('orth_part'):.2f} | "
                  f"Antwort-Tausch {g('answer_swap'):.2f} | R² C {gp('C'):.3f} cosC {gp('cosC'):.3f} cos(C/3) {gp('cosC3'):.3f} | C über cosC {gp('C_beyond_cosC_and_output'):+.3f}")

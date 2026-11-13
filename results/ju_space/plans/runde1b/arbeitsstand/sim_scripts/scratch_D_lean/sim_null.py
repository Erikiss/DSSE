"""Null-model envelope for the D1 capture surface (no model is run; numpy only).

Context = n reports, k of them version B (v=+1), n-k version A (v=-1).
Evidence  E = sum_i w_i v_i / sum_i w_i  with a recency(+primacy) kernel w_i.
Readout   P_B = sigmoid(beta * E)  (two-option renormalised probability).
Arrangements: 'int' = random order (position balanced), 'rel' = the k B reports last.
Normalised minority influence S(alpha) = (m(alpha) - m(0)) / (m(1) - m(0)),
Phi = mean_{alpha in .1,.2,.3,.4} S(alpha)   (linear response: Phi = 0.25).
We ask how far pure null changes (degradation beta -> rho*beta, recency steepening
tau -> tau/2, more reports n) can move Phi and the release threshold.
"""
import numpy as np

rng = np.random.default_rng(0)


def kernel(n, tau, prim=0.0, tau_p=1.0):
    i = np.arange(1, n + 1)
    w = np.exp(-(n - i) / tau)
    if prim > 0:
        w = w + prim * np.exp(-(i - 1) / tau_p)
    return w


def m_curve(n, beta, tau, arr, prim=0.0, n_orders=4000, ks=None):
    w = kernel(n, tau, prim)
    W = w.sum()
    ks = range(0, n + 1) if ks is None else ks
    out = {}
    for k in ks:
        if arr == 'rel':
            v = -np.ones(n)
            if k > 0:
                v[n - k:] = 1.0
            E = (w * v).sum() / W
            out[k] = 1 / (1 + np.exp(-beta * E))
        else:
            vals = []
            for _ in range(n_orders):
                v = -np.ones(n)
                v[rng.choice(n, size=k, replace=False)] = 1.0
                E = (w * v).sum() / W
                vals.append(1 / (1 + np.exp(-beta * E)))
            out[k] = float(np.mean(vals))
    return out


def phi(curve, n):
    m0, m1 = curve[0], curve[n]
    S = {k: (curve[k] - m0) / (m1 - m0) for k in curve}
    mins = [k for k in curve if 0 < k / n < 0.5 - 1e-9]
    return float(np.mean([S[k] for k in mins])), S


def alpha_star(curve, n):
    # first share where P_B crosses 0.5 (logit-linear interpolation); censored at the grid
    ks = sorted(curve)
    lg = lambda p: np.log(p / (1 - p))
    if curve[ks[1]] >= 0.5:
        return ks[1] / n, 'censored_low'
    for a, b in zip(ks[:-1], ks[1:]):
        if curve[a] < 0.5 <= curve[b]:
            la, lb = lg(curve[a]), lg(curve[b])
            return (a + (0 - la) / (lb - la) * (b - a)) / n, 'ok'
    return np.nan, 'none'


if __name__ == '__main__':
    n = 10
    print('Phi (interleaved, n=10) under null models; linear response gives 0.25')
    print(f"{'beta':>5} {'tau':>5} {'prim':>5} | {'Phi_int':>7} {'S(.1)':>6} {'S(.3)':>6} | {'m_rel(.1)':>9} {'a*_rel':>7}")
    rows = []
    for beta in (2, 4, 8, 16):
        for tau in (1.0, 2.0, 4.0, 1e6):
            for prim in (0.0, 0.5):
                ci = m_curve(n, beta, tau, 'int', prim, n_orders=3000)
                cr = m_curve(n, beta, tau, 'rel', prim)
                P, S = phi(ci, n)
                a, st = alpha_star(cr, n)
                rows.append((beta, tau, prim, P))
                print(f"{beta:5.1f} {tau:5.1f} {prim:5.1f} | {P:7.3f} {S[1]:6.3f} {S[3]:6.3f} | {cr[1]:9.3f} {a:7.3f} {st}")
    # Pure-null length changes from a reference (beta0, tau0):
    print('\nChange of Phi under null "length" changes (reference -> changed)')
    for beta0 in (4, 8, 16):
        for tau0 in (2.0, 4.0):
            ref = phi(m_curve(n, beta0, tau0, 'int', 0.0, 3000), n)[0]
            deg = phi(m_curve(n, 0.9 * beta0, tau0, 'int', 0.0, 3000), n)[0]
            deg5 = phi(m_curve(n, 0.5 * beta0, tau0, 'int', 0.0, 3000), n)[0]
            steep = phi(m_curve(n, beta0, tau0 / 2, 'int', 0.0, 3000), n)[0]
            prim = phi(m_curve(n, beta0, tau0, 'int', 0.5, 3000), n)[0]
            print(f"beta0={beta0:4.1f} tau0={tau0:3.1f}: Phi_ref {ref:.3f} | degr rho .9 {deg-ref:+.3f} | rho .5 {deg5-ref:+.3f} |"
                  f" recency x2 steeper {steep-ref:+.3f} | primacy added {prim-ref:+.3f}")
    # count arm: n = 40 at the same shares
    print('\nCount arm: n=10 vs n=40 at equal shares (same kernel in tokens: tau scales x4 for n=40)')
    for beta in (4, 8, 16):
        for tau in (2.0, 4.0):
            p10 = phi(m_curve(10, beta, tau, 'int', 0.0, 3000), 10)[0]
            c40 = m_curve(40, beta, 4 * tau, 'int', 0.0, 1500, ks=[0, 4, 8, 12, 16, 20, 40])
            p40 = phi(c40, 40)[0]
            c40b = m_curve(40, beta, tau, 'int', 0.0, 1500, ks=[0, 4, 8, 12, 16, 20, 40])
            p40b = phi(c40b, 40)[0]
            print(f"beta={beta:4.1f} tau={tau:3.1f}: Phi n10 {p10:.3f} | n40 (kernel in tokens) {p40:.3f} | n40 (kernel in reports) {p40b:.3f}")

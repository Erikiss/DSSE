# Restructured L3 lines + pooled multi-turn carry; L-ALL reading. Planning numbers only.
import numpy as np
rng = np.random.default_rng(11)
N = 20000
def rank_top(delta):
    return rng.normal(delta, 1, N) > rng.normal(0, 1, (N, 99)).max(1)
def below_q90(delta):
    return rng.normal(delta, 1, N) < np.quantile(rng.normal(0, 1, (4000, 99)), 0.9)
def L3(KH, dK, d_hd, d_g, d_bi, keys=512):
    s = np.sqrt(512 / keys)
    sd1 = (0.10 + 0.04 * KH) * s                         # one-step K_H (turn 2)
    sdp = sd1 * (0.6 if KH >= 1 else 0.9)                # pooled geometric carry over turns 1-5
    sdD = np.sqrt(sd1**2 + (0.12 * s)**2 / 6 + 0.10**2 / 6)
    kp = rng.normal(KH, sdp, N); d = rng.normal(dK, sdD, N)
    A2 = rank_top(d_bi) & (rng.random(N) < (0.95 if d_bi >= 3 else 0.5 if d_bi > 0 else 0.05))
    thr = np.where(A2, 1.645, 1.96)
    A1 = ((kp - 1) / sdp > thr) & (d / sdD > thr)
    O = (rank_top(d_hd) & (rng.random(N) < (0.9 if d_hd >= 3 else 0.5))) | (rank_top(d_g) & (rng.random(N) < (0.9 if d_g >= 3 else 0.5)))
    diss = (kp + 1.645 * sdp) < 1
    notdist = below_q90(d_hd) & below_q90(d_g) & below_q90(d_bi)
    att = np.full(N, 'undecided', dtype=object)
    att[diss & ~A2] = 'dissipative (obs. inconclusive)'
    att[diss & ~A2 & ~O & notdist] = 'no attractor'
    att[diss & ~A2 & O] = 'sticky, not self-sustaining'
    att[(A1 | A2) & ~O] = 'self-sustaining under push only'
    att[(A1 | A2) & O] = 'attractor'
    spec = np.full(N, 'undecided', dtype=object)
    spec[d - 1.645 * sdD > 0] = 'H-specific carry'
    spec[np.abs(d) + 1.645 * sdD < 0.40] = 'carry like controls'
    return att, spec
def L2(b0, bi, bpi, db, se=0.02):
    B0 = rng.normal(b0, se, N); BI = rng.normal(bi, se, N); BP = rng.normal(bpi, se, N); DB = rng.normal(db, 0.025, N)
    info = (B0 - 1.645 * se > 0) & (BI + 1.645 * se < 0) & (np.abs(BP) + 1.645 * se < 0.10)
    beyond = (BP - 1.645 * se > 0) & (BP >= 0.10) & (DB - 1.645 * 0.025 > 0)
    none = np.abs(B0) + 1.645 * se < 0.10
    out = np.full(N, 'undecided', dtype=object); out[none] = 'no trend'; out[info] = 'info-limited'; out[beyond] = 'beyond info'
    return out
def LALL(l2, att, spec):
    out = np.full(N, 'mixed', dtype=object)
    out[(att == 'attractor') & (l2 != 'info-limited')] = 'H-attractor not bound to length'
    out[(att == 'attractor') & (l2 == 'info-limited')] = 'H-attractor with info-limited length trend'
    out[np.isin(att, ['no attractor', 'sticky, not self-sustaining', 'dissipative (obs. inconclusive)']) & (spec == 'H-specific carry')] = 'H-specific amplifier, not self-sustaining'
    out[(att == 'no attractor') & (spec != 'H-specific carry') & (l2 == 'info-limited')] = 'length artefact (info-limited), no attractor'
    out[(att == 'no attractor') & (spec != 'H-specific carry') & (l2 == 'no trend')] = 'no length trend, no attractor'
    out[(att == 'no attractor') & (spec != 'H-specific carry') & (l2 == 'beyond info')] = 'H rises beyond info, no attractor'
    return out
scen = {
 'S1 length artefact':        ((0.15, -0.15, 0.0, 0.0),   (0.5, 0.0, 0, 0, 0)),
 'S2 attractor, not length':  ((0.15, -0.10, 0.15, 0.15), (1.3, 0.8, 3.5, 3.5, 0)),
 'S3 attractor + info trend': ((0.15, -0.15, 0.0, 0.0),   (1.3, 0.8, 3.5, 3.5, 0)),
 'S4 amplifier':              ((0.15, -0.10, 0.10, 0.10), (0.75, 0.45, 2, 2, 0)),
 'S5 nothing':                ((0.0, 0.0, 0.0, 0.0),      (0.5, 0.0, 0, 0, 0)),
 'S6 bistable, K<1':          ((0.15, -0.10, 0.10, 0.10), (0.6, 0.1, 3.5, 3.5, 3.5)),
 'S7 weak attractor K=1.15':  ((0.15, -0.10, 0.15, 0.15), (1.15, 0.6, 3.0, 3.0, 0)),
}
for name, (p2, p3) in scen.items():
    att, spec = L3(*p3); c = LALL(L2(*p2), att, spec)
    v, n = np.unique(c, return_counts=True)
    va, na = np.unique(att, return_counts=True)
    print(name + ' | L-ALL: ' + '; '.join(f'{x} {m/N:.2f}' for x, m in sorted(zip(v, n), key=lambda t: -t[1])))
    print('    L3 attractor line: ' + '; '.join(f'{x} {m/N:.2f}' for x, m in sorted(zip(va, na), key=lambda t: -t[1])))

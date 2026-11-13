# Planning: how often do the conjunctive combined verdicts (L.4 H-CHAIN reading, V-OWN) come out 'mixed' under
# realistic effects? Component statistics drawn as normals with planning SEs; rank tests vs 99 controls simulated;
# frozen rules applied. Planning numbers only (no model output).
import numpy as np
rng = np.random.default_rng(7)
N = 20000
def rank_top(delta, n=N):           # P(H above all 99 controls), controls N(0,1), H N(delta,1)
    H = rng.normal(delta, 1, n); C = rng.normal(0, 1, (n, 99)).max(1); return H > C
def L3(KH, dK, d_hd, d_g, d_bi, sdK=None):
    sdK = sdK if sdK is not None else 0.10 + 0.04 * KH          # SD(K_H) ~ sqrt(1+K^2)/z1, z1 ~ 10-11 (512 keys)
    sdD = np.sqrt(sdK**2 + 0.12**2 / 6 + 0.10**2 / 6)            # 6 random sets/sign, between-set SD 0.10
    k = rng.normal(KH, sdK, N); d = rng.normal(dK, sdD, N)
    zk, zd = (k - 1) / sdK, d / sdD
    A2 = rank_top(d_bi) & (rng.random(N) < (0.95 if d_bi >= 3 else 0.5 if d_bi > 0 else 0.01))  # BI also above own surrogates
    a1_p = np.where(A2, 1.645, 1.96)                               # Holm over {A1, A2}
    A1 = (zk > a1_p) & (zd > a1_p)
    hd = rank_top(d_hd) & (rng.random(N) < (0.9 if d_hd >= 3 else 0.5 if d_hd > 0 else 0.05))   # plus SEL+ contrast
    g = rank_top(d_g) & (rng.random(N) < (0.9 if d_g >= 3 else 0.5 if d_g > 0 else 0.05))
    O = hd | g                                                      # Holm over {HD, dgamma}: rank p 0.01 <= 0.025 for either
    obs_below = (rng.normal(d_hd, 1, N) < 1.28) & (rng.normal(d_g, 1, N) < 1.28)   # below controls' q90 (approx.)
    att = (A1 | A2) & O
    att_weak = (A1 | A2) & ~O
    snow = ((k + 1.645 * sdK) < 1) & (zd > 1.645) & ~(A1 | A2)
    no_att = ((k + 1.645 * sdK) < 1) & (np.abs(d) + 1.645 * sdD < 0.35) & obs_below & ~A2
    out = np.full(N, 'mixed', dtype=object)
    out[no_att] = 'no attractor'; out[snow] = 'amplifier'; out[att_weak] = 'self-sustaining (no obs.)'; out[att] = 'attractor'
    return out
def L2(b0, bi, bpi, db, se=0.02):
    B0 = rng.normal(b0, se, N); BI = rng.normal(bi, se, N); BP = rng.normal(bpi, se, N); DB = rng.normal(db, 0.025, N)
    lo = lambda x, s: x - 1.645 * s; hi = lambda x, s: x + 1.645 * s
    info = (lo(B0, se) > 0) & (hi(BI, se) < 0) & (np.abs(BP) + 1.645 * se < 0.10)
    beyond = (lo(BP, se) > 0) & (BP >= 0.10) & (lo(DB, 0.025) > 0)
    none = np.abs(B0) + 1.645 * se < 0.10
    out = np.full(N, 'undecided', dtype=object); out[none] = 'no trend'; out[info] = 'info-limited'; out[beyond] = 'beyond info'
    return out
def combine(l2, l3):
    out = np.full(N, 'mixed', dtype=object)
    out[(l3 == 'attractor') & (l2 != 'info-limited')] = 'H-attractor not bound to length'
    out[(l3 == 'attractor') & (l2 == 'info-limited')] = 'H-attractor + info-limited trend'
    out[l3 == 'amplifier'] = 'H-specific amplifier'
    out[(l3 == 'no attractor') & (l2 == 'info-limited')] = 'length artefact (info-limited)'
    out[(l3 == 'no attractor') & (l2 == 'no trend')] = 'no trend, no attractor'
    out[(l3 == 'no attractor') & (l2 == 'beyond info')] = 'H rises beyond info, no attractor'
    return out
scen = {
 'S1 length artefact':      dict(l2=(0.15, -0.15, 0.0, 0.0),  l3=(0.5, 0.0, 0, 0, 0)),
 'S2 attractor, not length':dict(l2=(0.15, -0.10, 0.15, 0.15), l3=(1.3, 0.8, 3.5, 3.5, 0)),
 'S3 attractor + info trend':dict(l2=(0.15, -0.15, 0.0, 0.0), l3=(1.3, 0.8, 3.5, 3.5, 0)),
 'S4 amplifier':            dict(l2=(0.15, -0.10, 0.10, 0.10), l3=(0.75, 0.35, 2, 2, 0)),
 'S5 nothing':              dict(l2=(0.0, 0.0, 0.0, 0.0),     l3=(0.5, 0.0, 0, 0, 0)),
 'S6 bistable, K<1':        dict(l2=(0.15, -0.10, 0.10, 0.10), l3=(0.6, 0.1, 3.5, 3.5, 3.5)),
 'S7 weak attractor K=1.15':dict(l2=(0.15, -0.10, 0.15, 0.15), l3=(1.15, 0.6, 3.0, 3.0, 0)),
}
# V-OWN: L5a and L5b
def own(GFk, Om_all, Om_prone, rb, rw, seGF=0.012, seOm=0.032, serb=0.04, serw=0.014):
    gf = rng.normal(GFk, seGF, N); oa = rng.normal(Om_all, seOm, N); op = rng.normal(Om_prone, seOm, N)
    a = np.full(N, 'undecided', dtype=object)
    a[(gf - 1.96*seGF > 0) & (oa + 1.96*seOm < 0.5)] = 'lens'
    a[(np.abs(gf) + 1.645*seGF < 0.04) & (op - 1.96*seOm > 0.5)] = 'model'
    a[(gf - 1.96*seGF > 0) & (op - 1.96*seOm > 0.5)] = 'shared'
    RB = rng.normal(rb, serb, N); RW = rng.normal(rw, serw, N)
    b = np.full(N, 'undecided', dtype=object)
    b[(RB - 1.96*serb > 0) & (np.abs(RW) + 1.645*serw < 0.05)] = 'model'
    b[(RW - 1.96*serw > 0) & (np.abs(RB) + 1.645*serb < 0.12)] = 'lens'
    b[(RB - 1.96*serb > 0) & (RW - 1.96*serw > 0)] = 'both'
    b[(np.abs(RB) + 1.645*serb < 0.12) & (np.abs(RW) + 1.645*serw < 0.05)] = 'none'
    v = np.full(N, 'not decidable', dtype=object)
    pos_a = np.isin(a, ['lens', 'model', 'shared']); pos_b = np.isin(b, ['lens', 'model', 'both'])
    v[pos_a | pos_b] = 'shared'
    v[(a == 'lens') & (b == 'lens')] = 'lens'; v[(a == 'model') & (b == 'model')] = 'model'
    return v
for name, p in {'lens-owned': (0.10, 0.2, 0.3, 0.0, 0.12), 'model-owned': (0.0, 0.8, 0.8, 0.25, 0.0),
                'shared': (0.08, 0.4, 0.7, 0.2, 0.1)}.items():
    v = own(*p); vals, cnt = np.unique(v, return_counts=True)
    print('V-OWN ' + name + ': ' + '; '.join(f'{x} {n/N:.2f}' for x, n in sorted(zip(vals, cnt), key=lambda x: -x[1])))

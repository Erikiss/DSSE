"""Final part-D cost table (estimates; no model run). Same rate assumptions as cost_model.py."""
DEC = {0: 1000.0, 4096: 450.0, 16384: 120.0, 28672: 70.0, 36864: 53.0}
PROBE_MS = {0: 25.0, 4096: 30.0, 16384: 50.0, 28672: 60.0, 36864: 70.0}
BATCH = {0: 32, 4096: 24, 16384: 7, 28672: 3, 36864: 3}


def traj_s(L0, turns=8, ans_tok=40, probes=6, extra_tok=0, probe_turns=None):
    pt = turns if probe_turns is None else probe_turns
    return (turns * (ans_tok + extra_tok) / DEC[L0] + pt * probes * PROBE_MS[L0] / 1000.0 / BATCH[L0]
            + turns * 0.025 / BATCH[L0])


def m(L0, n, **kw):
    return n * traj_s(L0, **kw) / 60.0


core, res = {}, {}
core['D1-D3 trajectory bank F-SYM T0.7, L0 0'] = m(0, 640)
core['D1-D3 trajectory bank F-SYM T0.7, L0 4K'] = m(4096, 640)
core['D1-D3 trajectory bank F-SYM T0.7, L0 16K'] = m(16384, 640)
core['D1-D3 trajectory bank F-SYM T0.7, L0 28K'] = m(28672, 640)
core['D3s shape cells 16K (FS, FE; M start 256 each; FE prefills)'] = 2 * m(16384, 256) + 16 * 7 / 60.0
core['D1 F-RSI (0, T0.7), (0, T1.0), (28K, T0.7)'] = (m(0, 256, probes=10, probe_turns=5) * 2
                                                      + m(28672, 256, probes=10, probe_turns=5))
core['D4 hysteresis L0 0'] = m(0, 96, turns=14, extra_tok=10) + m(0, 432, turns=1)
core['D4 hysteresis L0 28K'] = m(28672, 96, turns=14, extra_tok=10) + m(28672, 432, turns=1)
core['D5 catfish L0 0 (3 conditions + PC2)'] = m(0, 480, probes=8)
core['D5 catfish L0 28K (3 conditions + PC2)'] = m(28672, 480, probes=8)
core['D6 dissociation L0 0 (+ anchors, PC)'] = m(0, 128, probes=8) + m(0, 160, turns=1, probes=8)
core['D6 dissociation L0 28K (+ anchors, PC)'] = m(28672, 128, probes=8) + m(28672, 160, turns=1, probes=8)
# D7: 24 runs x ~12.5k generated tokens at ~200 tok/s aggregate + probes (49 points x 6-12 forwards at ~40 ms)
core['D7 long own-reply runs (24 x 16K)'] = 24 * 12500 / 200.0 / 60.0 + 24 * 49 * 9 * 0.04 / 60.0
core['D8 losing-game readouts + orphaning'] = 8.0
core['filler prefills (4 docs x L, 3 types at 16K)'] = 4 * (1 + 3 * 7 + 20) / 60.0
core['stimuli (injected turns, roles, FE pools), axes u_AB / u_AA'] = 15.0
core['part-D share of the LCE pilot (engineering only)'] = 10.0
core['H-occupancy hooks (+5 %, TB 0/28K, D5, D7)'] = 0.05 * (m(0, 640) + m(28672, 640) + m(28672, 480, probes=8) + 30)
core['analysis (D2 maps, D3 mixtures, D4 null simulations, D9 fits)'] = 20.0

res['R-T1 F-SYM T1.0 at L0 0 and 16K (A2/B2 + M)'] = m(0, 496) + m(16384, 496)
res['R-RSI F-RSI (28K, T1.0)'] = m(28672, 256, probes=10, probe_turns=5)
res['R-TOK D10 character-level injection at L* (M start, worst case 28K)'] = m(28672, 256)
res['R-MIN D5 minority opening (0 and 28K)'] = m(0, 144, probes=8) + m(28672, 144, probes=8)
res['R-36K F-SYM beyond the trained window (248 traj)'] = m(36864, 248)

def show(d, title):
    tot = 0.0
    print(title)
    for k, v in d.items():
        print(f"  {k:66s} {v:6.1f} min")
        tot += v
    print(f"  {'sum':66s} {tot:6.1f} min = {tot/60:.2f} h")
    return tot

c = show(core, 'CORE')
r = show(res, 'RESERVES (fixed priority order; descriptive or own verdicts only)')
sessions_core = 8
setup = sessions_core * 5.0
print(f"core: work {c/60:.2f} h; with setup ({sessions_core} x 5 min) and margin 1.5: {(setup + 1.5 * c)/60:.2f} A100-h")
print(f"reserves: with margin 1.5 (+2 sessions setup): {(10 + 1.5 * r)/60:.2f} A100-h")

"""Part D cost model (estimates; no model is run).
Rates: int_machinery.md section 3.6 (HF DynamicCache aggregate decode 450/120/60 tok/s at 4k/16k/32k, est.),
round-2 plan 7.1 (R-gen ~1,000 tok/s at batch 32 with ~80-200-token prompts), probe chunk forwards over a cached prefix.
All numbers are A100-40GB, float32, TF32 off, memory-efficient SDPA."""
import math

# aggregate decode tokens/s by prefix length (filler L0); short = no filler (history <= ~2k)
DEC = {0: 1000.0, 4096: 450.0, 16384: 120.0, 28672: 70.0, 36864: 53.0}
# one probe forward (~40-token chunk over the cached context) per batch, and batch size (rows with own cache)
PROBE_MS = {0: 25.0, 4096: 30.0, 16384: 50.0, 28672: 60.0, 36864: 70.0}
BATCH = {0: 32, 4096: 24, 16384: 7, 28672: 3, 36864: 3}
FILLER_PREFILL_S = {0: 0.0, 4096: 1.0, 16384: 7.0, 28672: 20.0, 36864: 30.0}


def traj_s(L0, turns=8, ans_tok=40, probes_per_turn=6, extra_tok_per_turn=0, probe_turns=None):
    """seconds of A100 time per trajectory (decode + probes), amortised over the batch."""
    dec = turns * (ans_tok + extra_tok_per_turn) / DEC[L0]
    pt = turns if probe_turns is None else probe_turns
    probes = pt * probes_per_turn * PROBE_MS[L0] / 1000.0 / BATCH[L0]
    host = turns * 0.025 / BATCH[L0]  # per-turn chunk prefill / host overhead per batch
    return dec + probes + host


def cell_min(L0, n_traj, **kw):
    return n_traj * traj_s(L0, **kw) / 60.0


rows = []
def add(name, L0, n, **kw):
    m = cell_min(L0, n, **kw)
    rows.append((name, L0, n, round(traj_s(L0, **kw), 2), round(m, 1)))
    return m

LS = [0, 4096, 16384, 28672]
tot = {}
# TB F-SYM T0.7: 24 items; A2/B2 5 seeds (240 traj), M 10 seeds (240), AB/BA 3 seeds each (144) = 624 per L
for L in LS:
    tot.setdefault('D1-D3 TB F-SYM T0.7 (FC filler)', 0)
    tot['D1-D3 TB F-SYM T0.7 (FC filler)'] += add('TB F-SYM T0.7', L, 624)
# shape-vs-size cells at 16K: shuffled filler FS and entity-balanced history FE, M start only (240 traj each)
tot['D3s shape cells 16K (FS, FE; M start)'] = add('FS 16K M', 16384, 240) + add('FE 16K M', 16384, 240) + 24 * 7 / 60.0  # FE: 24 item-specific prefills
# T1.0 F-SYM: A2/B2 (240) + M (240) at L0 0 and 16K
tot['D1-D3 TB F-SYM T1.0 (0, 16K)'] = add('F-SYM T1.0', 0, 480) + add('F-SYM T1.0', 16384, 480)
# F-RSI: 64 prompts x 2 seeds = 128 pairs = 256 traj; cells (0,0.7), (0,1.0), (28K,0.7); probes 10 H30 items at 5 turns
tot['D1 F-RSI (0 x2 T, 28K T0.7)'] = (add('F-RSI 0 T0.7', 0, 256, probes_per_turn=10, probe_turns=5)
                                       + add('F-RSI 0 T1.0', 0, 256, probes_per_turn=10, probe_turns=5)
                                       + add('F-RSI 28K T0.7', 28672, 256, probes_per_turn=10, probe_turns=5))
# D4 hysteresis: 24 items x 2 seeds x 2 rates x 2 directions = 192 sweeps; mean 14 turns; user turns +40 tokens of snippets
for L in (0, 28672):
    tot.setdefault('D4 hysteresis (0, 28K)', 0)
    tot['D4 hysteresis (0, 28K)'] += add('D4 sweeps', L, 192, turns=14, extra_tok_per_turn=10)
    # memoryless control: 24 items x 2 seeds x 9 levels single-turn
    tot['D4 hysteresis (0, 28K)'] += add('D4 memoryless', L, 432, turns=1)
# D5 catfish: conditions C-opp, C-neu, C-one: 24 items x 2 versions x 3 seeds = 144 traj each; + PC2 instructed two-voice 96
for L in (0, 28672):
    tot.setdefault('D5 catfish (0, 28K)', 0)
    tot['D5 catfish (0, 28K)'] += add('D5 3 conditions', L, 432, ans_tok=48, probes_per_turn=8)
    tot['D5 catfish (0, 28K)'] += add('D5 PC2 instructed', L, 96, ans_tok=48, probes_per_turn=8)
# D6 catfishing dissociation: 16 roles x 8 seeds = 128 per L0; 8 probe forwards per turn; anchors 16 x 2 x 4 short
for L in (0, 28672):
    tot.setdefault('D6 dissociation (0, 28K)', 0)
    tot['D6 dissociation (0, 28K)'] += add('D6 roles', L, 128, probes_per_turn=8)
    tot['D6 dissociation (0, 28K)'] += add('D6 anchors', L, 128, turns=1, probes_per_turn=8)
# D7 self-generated long runs: 16 trajectories, 64-token turns to 16K total (~200 turns), avg context ~8K
d7_tok = 16 * 16000
d7_min = d7_tok / 200.0 / 60.0 + 16 * 50 * 6 * 0.040 / 60.0
tot['D7 long own-reply runs (16 x 16K)'] = d7_min
rows.append(('D7 long runs', 'to 16K', 16, None, round(d7_min, 1)))
# D8 losing-game readouts appended at trajectory ends (2 ref passages 128 tok + 6 factual probes) on TB+D5 long cells
# approx: per trajectory 0.25 s at 28K, 0.12 s at 16K, 0.03 s at <=4K; orphaning deletion re-reads 64 per L0 in {0,28K}
d8 = (624 * 4 * 0.0 + (624 * 0.03 * 2 + 624 * 0.12 + 624 * 0.25 + 432 * 0.25 + 432 * 0.03)) / 60.0 + (64 * 0.1 + 64 * 1.5) / 60.0
tot['D8 losing-game readouts + orphaning'] = d8
# D10 retokenisation (char-level injected turns): conditional; at L* = 28K worst case M start 240 traj; fallback L0=0 A2/B2+M 480
tot['D10 retokenisation (worst case 28K, M start)'] = add('D10 char 28K', 28672, 240)
# filler prefills (4 filler docs per L, 3 filler types at 16K, 1 type elsewhere)
tot['filler prefills'] = (4 * (FILLER_PREFILL_S[4096] + FILLER_PREFILL_S[16384] * 3 + FILLER_PREFILL_S[28672])) / 60.0
# axes, injected-turn stimuli, u_AA, anchors: generation of ~8k short texts and reads
tot['stimuli, axes (u_AB, u_AA), calibration'] = 15.0
tot['part-D share of LCE pilot (engineering only)'] = 10.0
tot['H-occupancy hooks (+5 % on TB 0/28K and D5)'] = 0.05 * (cell_min(0, 624) + cell_min(28672, 624) + cell_min(28672, 432, ans_tok=48, probes_per_turn=8))
tot['analysis session (D2 maps, D3 bootstraps, D9 fits; CPU+GPU)'] = 20.0

work = sum(tot.values())
print("cell rows (name, L0, n_traj, s/traj, min):")
for r in rows:
    print("  ", r)
print()
for k, v in tot.items():
    print(f"{k:60s} {v:7.1f} min")
print(f"{'TOTAL work':60s} {work:7.1f} min = {work/60:.2f} h")
n_sessions = 9
setup = n_sessions * 5.0
print(f"setup {n_sessions} sessions x 5 min = {setup:.0f} min; with margin 1.5 x work: {(setup + 1.5 * work)/60:.2f} A100-h")

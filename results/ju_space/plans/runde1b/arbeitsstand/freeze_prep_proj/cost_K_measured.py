"""Cost model for the merged part K (estimates only; no model is run).
Rates = round-2 planned 'given facts' (plan:1376-1390) as used by all three designs:
  Qwen3-1.7B 2.819 GFLOP/token through 28 blocks, +0.622 GFLOP where full logits are formed; 15.8 TFLOP/s effective fp32, TF32 off
  (plan-implied, not measured); 20 ms host per decode step.  Long caches: memory-bound KV traffic with HF DynamicCache rewrite
  (read + write + read = 3 x KV bytes per step) at 1.555 TB/s; 229,376 B KV per token fp32.
  Qwen3.5-2B: 2.6 + 1.02 GFLOP/token, x2 risk factor (fp32 linear-attention decode unmeasured); MTP step 0.12 + 1.02 GFLOP, x2, 4 ms host.
  Hooks: +25 % on rollouts with neuron-activity capture; +50 % where per-step DLA over all neurons is formed (K4 V1).
  FISTA 4 ms/iteration on 4,000 x 172,032 fp32.  Margin x1.5 on work; setup 4 min per Qwen3-1.7B session, 8 min for the
  Qwen3.5-2B session, 3 min for analysis.
"""
GF = 1e9; R = 15.8e12; HOST = 0.020; BW = 1.555e12; KVB = 229376
F = 2.819 * GF; LG = 0.622 * GF; FL = F + LG
F35 = 2 * (2.6 + 1.02) * GF; FM = 2 * (0.12 + 1.02) * GF

def dec(rows, steps, batch=320, f=FL, ctx=0, host=HOST, hook=1.0):
    """decode `rows` rows for `steps` steps in batches; ctx = shared-prefix length (KV traffic)."""
    full, rem = divmod(rows, batch)
    def step(r):
        kv = 3 * r * (ctx + steps) * KVB / BW if ctx else 0.0
        att = r * 229376.0 * ctx / R if ctx else 0.0
        return host + r * f / R + kv + att
    return hook * (full * steps * step(batch) + (steps * step(rem) if rem else 0.0))

def fwd(tokens, f=F, hook=1.0):
    return hook * tokens * f / R

S = {}  # session -> list of (id, seconds, note)
def add(sess, mid, sec, note):
    S.setdefault(sess, []).append((mid, sec, note))

# ---------------- K-S0 pilot (pre-freeze, engineering only) ----------------
add('K-S0', 'K0', 10 * 60, 'engineering pilot (rates, MTP load, code checks)')

# ---------------- K-S1 (account B): H-sets, construct validity, J-lens gate ----------------
add('K-S1', 'K2', fwd(8000 * 45, FL) + dec(80000, 24), 'TriviaQA screening <= 8,000 q x 10 samples (prefill once per question)')
add('K-S1', 'K2', fwd(3000 * 45, FL) + dec(30000, 8), 'ACH screening for H_ar, 3,000 x 10')
add('K-S1', 'K2', fwd(8000 * 60, FL, hook=1.5), 'CETT extraction 4,000 TQA + 4,000 ACH rows')
add('K-S1', 'K2', 2 * (16 * 600 + 600 + 20 * 600 + 20 * 600 + 1000) * 0.004, 'L1 paths, 20 bootstrap + 20 permutation refits, two sets')
add('K-S1', 'K2', 16 * (dec(1000, 24) + fwd(1000 * 40, FL)) + 16 * (dec(500, 8) + fwd(500 * 45, FL)), 'C selection: 16 suppression runs per set')
add('K-S1', 'K2', 2 * 400 * 2 * fwd(16 * 60, FL) + 400 * fwd(16 * 60, FL), 'RS39 damage matching (scaling a 0, 3; two sets) + mean-clamp damage (RS39mc)')
add('K-S1', 'K2', fwd(1_000_000, F, hook=1.3), 'WikiText 1M tokens: neuron percentiles, REF-W zCETT, mu_gen, Sigma_l')
add('K-S1', 'K3a', dec(1900, 24) + fwd(1900 * 60, FL, hook=2.5), 'detection sets S3, A-det, NE')
battery = lambda: dec(1400, 24) + fwd(1400 * 60, F) + 1.0
add('K-S1', 'K3d', (5 + 39 + 19) * battery(), 'causal battery: H at 5 alphas, RS39 at a 3, RS19 at a 0')
add('K-S1', 'K3e', dec(300 * 2 * 10 * 2 * 3, 17) + fwd(300 * 2 * 10 * 3 * 34, FL), 'neuron-lens chains with GVRA (<= 3 attempts)')
add('K-S1', 'K3g', fwd(1000 * 60, FL, hook=1.5), 'plain-frame detection (frame transfer of the H detector)')
add('K-S1', 'G-K-J', fwd(64 * 4 * 8 * 2 * 2 * 128, F), 'J-lens bridge: 64 passages x 4 blocks x 8 vectors x 2 signs x 2 index conventions')

# ---------------- K-S3 (account B, parallel): Qwen3.5-2B calibration ----------------
add('K-S3', 'K5a', fwd(1300 * 45, F35) + dec(1300 * 5, 8, f=F35), 'gating 1,300 items on 2B (greedy + 4 samples)')
add('K-S3', 'K5a', fwd(2000 * 3 * 2, F35) + 30, 'GK7 MTP wiring check (3 wirings)')
add('K-S3', 'K5a', fwd(200 * 128 + 1300 * 45, F35), 'mean states, read states')
cells35 = 256 * 3   # 256 correct items x blocks {10,14,18} x t0
add('K-S3', 'K5a', 4 * dec(cells35 * 10 * 2, 17, f=F35), 'self-rollout E and L, V1 + V2')
add('K-S3', 'K5a', 4 * dec(cells35 * 10 * 2, 17, f=FM, host=0.004), 'native MTP lens, beta 0 (primary) and 1, V1 + V2')

# ---------------- K-S2a (account B): CL17 lens, lens-side H tests ----------------
add('K-S2a', 'K4', fwd(1300 * 45, FL) + dec(1300 * 5, 8), 'A-lens gating (greedy + 4 samples)')
add('K-S2a', 'K4', 2 * 6 * 32 * fwd(16 * 60, FL), 're-injection beta calibration (variant), blocks 14, 18')
add('K-S2a', 'K4p', fwd(4300 * 50, F, hook=1.2), 'lens-free probe: states at t0 for 4,300 items, 4 blocks')
c_main = 384 * 4 + 256 * 4                     # t0 all items + t1 correct items, 4 blocks
add('K-S2a', 'K4', dec(c_main * 10 * 2, 17, hook=1.5) + dec(c_main * 10 * 2, 17, hook=1.25), 'V1 (with DLA, FOV_att) + V2, all blocks, t0/t1')
add('K-S2a', 'K4', dec(768 * 2 * 10 * 2, 17, hook=1.25), 'V3 foreign seeds, V4 neutral seeds (blocks 14, 18, t0)')
add('K-S2a', 'K4', dec(512 * 10 * 2, 17, hook=1.25), 'V7 covariance-matched null state (correct, blocks 14, 18, t0)')
add('K-S2a', 'K4', dec(256 * 10 * 2, 17, hook=1.25), 'V8 re-injection ON (128 items x blocks 14, 18)')
add('K-S2a', 'K4', dec(5 * 256 * 10 * 2, 17, hook=1.25), 'decoding sensitivity: lambda 0.5, 2; plausibility 0.05, 0.3; greedy')
add('K-S2a', 'K4', dec(2 * 2 * 256 * 10 * 2, 17, hook=1.25), 'temperature arm T 0.5, 1.0 (V1, V2)')
add('K-S2a', 'K4', fwd(3 * c_main * 10 * 33, F * 0.6) * 1.3, 'verifier: V1, V2 and shuffled chains (prefix means), +30 % revisions')
add('K-S2a', 'K6c', 2 * 40 * dec(128 * 2 * 4 * 2, 17, hook=1.25), 'mean-clamp (i) and state mean-ablation (ii): H + RS39mc')
add('K-S2a', 'K6f', dec(384 * 10 * 2, 17, batch=128, ctx=528, hook=1.25) + dec(384 * 10 * 2, 17, batch=48, ctx=2064, hook=1.25) + 16 * fwd(2064, F), 'context offsets 528 / 2,064 (8 filler instances each)')
add('K-S2a', 'K6g', dec(400 * 4 * 3 * 2, 13, hook=1.25), 'content-matched forcing (correct / wrong / neutral)')
add('K-S2a', 'K6h', dec(400 * 2 * 2 * 10 * 2, 13, hook=1.25) + fwd(400 * 40, F), 'QA lens arm (400 S_eval items x 2 blocks x V1, V2)')
add('K-S2a', 'K10c', fwd(200_000, F, hook=1.3) + fwd(240 * 2 * 2 * 60, F) + 30, 'Folge vs Reihe: L-par accumulators, L-jac FD')

# ---------------- K-S2b (account B): lens-frame persistence K6e ----------------
NCP = 192 * 4 * 2                                   # chain-pairs
rows = NCP * 2
one = dec(rows, 24, hook=1.25) + fwd(rows * 8, FL, hook=1.25)   # 24 decode steps + segment-1 re-prefill by unpushed model
add('K-S2b', 'K6e', 21 * 8 * fwd(16 * 60, FL) + 2 * dec(256 * 2, 24), 'damage calibration (H + 20 RS) and in-session unit check (32 A-cal items)')
add('K-S2b', 'K6e', (1 + 1 + 1 + 1 + 20 + 20) * one, 'twin, H D2 TOK, H CLAMP TOK, H D2 TOK+KV, RS20 D2, RS20 CLAMP')
add('K-S2b', 'K6e', 3 * one, 'descriptive: H D2 both-stream push; twin and H D2 at T 1.0')
add('K-S2b', 'K6e', 2 * (dec(rows, 24, batch=48, ctx=2064, hook=1.25) + fwd(rows * 8, FL)), 'offset invariance: twin and H D2 TOK at L_P 2,064')

# ---------------- K-S4 (account B): model's own text ----------------
add('K-S4', 'K8', fwd(1200 * 45, FL) + dec(1200 * 4, 16), 'A-bnd boundary screening (4 samples)')
pref = 48 * (fwd(4096, F) + fwd(2048, F) + fwd(512, F))
k8a = (pref + dec(4200, 16, batch=24, ctx=4096) + fwd(4200 * 70, FL) + dec(4200, 16, batch=48, ctx=2048) + fwd(4200 * 70, FL)
       + dec(4200, 16, batch=96, ctx=512) + fwd(4200 * 70, FL) + dec(700 * 4, 16) + fwd(700 * 4 * 70, FL))
add('K-S4', 'K8a', k8a, '19 conditions x 700 items (cap 4,096), offset and gap checks')
sess = lambda n, turns=8, ans=30, q=40, ctx=300: (-(-n // 128)) * turns * (ans * (HOST + 128 * FL / R + 3 * 128 * ctx * KVB / BW) + 128 * q * FL / R)
add('K-S4', 'K8b', (1 + 4 + 4 + 19) * sess(256), 'ACH main: clean, H a 3/2/1.5/0, routes TOKEN/KV, RS19 FULL (256 templates)')
add('K-S4', 'K8b', (1 + 1 + 19) * sess(384), 'ACH bistability: 24 templates x 16 seeds, clean, H, RS19')
add('K-S4', 'K8b', 2 * sess(192, ans=24, q=30, ctx=220) + 19 * sess(128, ans=24, q=30, ctx=220), 'TQA sessions: clean, H (192), RS19 (128)')
add('K-S4', 'K8b', 2 * sess(32), 'in-session unit check (32 A-cal templates)')
add('K-S4', 'K8c', dec(800, 260, batch=128) + fwd(800 * 120, FL, hook=1.3) + dec(1800 * 2 * 10 * 2, 9, hook=1.25), 'long-form generation + lens reach')
add('K-S4', 'K7a', 2 * fwd(251_000, F, hook=1.3) + dec(3200 * 16, 8), 'fronts (2 passes) + MC ceilings')

# ---------------- K-S5 (account A): axes, hierarchy, routes ----------------
add('K-S5', 'KS-axes', fwd(64 * 2 * 4 * 60, F) * 3 + 60, 'SO-AX (u*_b from R1 dir_*_4, mode_set; checks)')
k9cond = (10 * 2 * 16 * 3 * 2) * fwd(512, F, hook=1.15)
add('K-S5', 'K9', 5 * k9cond + 16 * fwd(2048, F) + (10 * 2 * 16 * 3 * 2) * fwd(512, F) * 0.17, '5 text conditions (ACH-long, neutral, TQA sessions, 2k coherent, 2k filler)')
n_src = 4 + 20 + 1 + 10      # named directions, random directions, N_H, random populations
per_src_dose = fwd(64 * 2 * 2 * 100, F) + dec(64 * 2 * 2, 10, batch=128) + fwd(64 * 2 * 2 * 100, F)
add('K-S5', 'K10a', 2 * n_src * per_src_dose + dec(64, 64, batch=64), 'route split, 2 doses, 35 sources (KV-only, token-only, total)')
add('K-S5', 'K10b', fwd(96 * 4 * 3 * (4 + 2) * 100, F) + 768 * 100 * F * 3 / R + 60, 'FD Jacobian: total, frozen-QK, foreign; VJP routing map; structural checks')
add('K-S5', 'K3f', dec(21 * 128, 28, batch=128) + fwd(21 * 128 * 60, FL) + 30, 'style -> H behavioural push vs 20 random directions')
add('K-S5', 'K7b', 45 * dec(256 * 3 * 2, 17, hook=1.25), 'channel-resolved lens (7 named + 38 random subspaces)')

# ---------------- K-ana (account A) ----------------
add('K-ana', 'K-ana', 20 * 60, 'bootstraps, hazard refits (H + 39 sets), mixture tests (2,000 draws), verdicts, report')

setup = {'K-S0': 4, 'K-S1': 4, 'K-S3': 8, 'K-S2a': 4, 'K-S2b': 4, 'K-S4': 4, 'K-S5': 6, 'K-ana': 3}
grand = 0.0; per_meas = {}
for s, items in S.items():
    w = sum(x[1] for x in items) / 60
    tot = setup[s] + 1.5 * w
    grand += tot
    print(f'{s:6s} work {w:6.1f} min  -> setup + 1.5 x work = {tot:6.1f} min')
    for mid, sec, note in items:
        per_meas[mid] = per_meas.get(mid, 0) + 1.5 * sec / 60
        print(f'      {mid:7s} {sec/60:6.2f} min  {note}')
print(f'PLANNED TOTAL {grand:.0f} min = {grand/60:.2f} A100-h')
print('per measurement (1.5 x work, setup separate):')
for k, v in per_meas.items():
    print(f'  {k:8s} {v:6.1f}')
print('setups', sum(setup.values()))
# reserves
rk1 = 4 + 1.5 * (20e6 * 7.8 * GF / 100e12 / 60 + 10)
rk2 = 8 + 1.5 * 30
unit = 1.5 * (23 * one + 3 * sess(128)) / 60
print(f'reserves: RK-1 {rk1:.0f} min, RK-2 {rk2:.0f} min, in-session unit doubling (K6e H arms + RS, K8b H arms) {unit:.0f} min')

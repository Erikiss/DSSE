"""Cost model for part L (energy / Ising / non-equilibrium), A100-40GB, fp32, TF32 off.
Rates from the frozen round-2 plan 7.1 (planned 'given facts') and int_machinery 3.6 (long context, est.).
All numbers are planning estimates (minutes of work), not measurements."""
import math

# --- rates ---------------------------------------------------------------
DEC32 = 0.030      # s per decode step at batch 32, contexts <= ~600 tokens (25 ms plan + DynamicCache rewrite)
HOOK = 1.15        # hook multiplier (plan 7.1: WriteOrth 1.15; neuron capture hooks assumed similar)
PREFILL_SHORT = 0.3  # s extra per generate call for ~80-token prompts (plan: 1.3 s vs 1 s)
TF_LOGITS = 5000.  # tok/s teacher-forced with logits at positions (R-full)
TF_NOLOG = 5600.   # tok/s teacher-forced through 28 blocks, no logits (R-all)
LONG16K = 2900.    # tok/s fp32 B=1 prefill at 16k (int_machinery 3.6: 2.4-3.7k)
LONG_BLEND_FOREIGN = 4000.  # tok/s blended for foreign passes with L in 128..16384
SETUP = 4.0        # min per Qwen session (plan 7.1)

def gen_minutes(n_traj, n_new, batch=32, step=DEC32, hook=HOOK, prefill=PREFILL_SHORT):
    calls = math.ceil(n_traj / batch)
    return calls * (n_new * step * hook + prefill) / 60.

rows = []
def add(sess, mid, what, minutes):
    rows.append((sess, mid, what, minutes))

# ---------------- Session L-S1 (account B; inputs: K outputs) -------------
# L0: recompute CETT on K's S1 items (4,000 items x ~100 tokens) for SEL64 selection; L1 path with fixed C + 20 bootstraps
add('L-S1', 'L0', 'CETT recompute S1 (0.4M tok, all 172,032 neurons, hooks)', 0.4e6 / TF_NOLOG / 60 * 1.3)
add('L-S1', 'L0', 'selection-matched L1 fits (fixed C, 20 bootstrap refits, FISTA ~4 ms/iter)', (21 * 500 * 0.004) / 60 + 1.0)
add('L-S1', 'L0', 'reference |z| quantiles (200 x 512 tokens) + planted Ising sets (GPU Gibbs)', 1e5 / TF_NOLOG / 60 + 2.0)
# L.R rollouts
add('L-S1', 'L.R', 'rollouts 4,000 x 384 tokens (R-A 2,400 at T0.7; R-B 1,600 at T1.0), spins captured', gen_minutes(4000, 384))
# L4a: sentence-reversed teacher-forced pass (~3,000 eligible x ~470 tokens) + IFT check
add('L-S1', 'L4', 'sentence-reversed TF pass (1.41M tok, logits at continuation positions)', 1.41e6 / TF_LOGITS / 60)
add('L-S1', 'L4', 'IFT pipeline check: 64k 5-token samples at T 2/3/4, prefix-cached scoring', 2.5)
# L3: twins and pushes
twin = gen_minutes(400, 192)
add('L-S1', 'L3', 'twins 400 x 192', twin)
add('L-S1', 'L3', 'push-in: H a=2.5, 4 damage-matched random sets, H a=1.5 (6 conditions)', 6 * twin)
add('L-S1', 'L3', 'push-out: H a=0, 4 random sets (5 conditions, <= 400 each)', 5 * twin)
add('L-S1', 'L3', 'sham code check (32 trajectories)', gen_minutes(32, 192))
t2 = math.ceil(400 / 32) * (128 * DEC32 * HOOK + 1.0) / 60.
add('L-S1', 'L3', 'turn 2 Continue (128 tok) after twin + 5 push-in conditions', 6 * t2)
add('L-S1', 'L3', 'cache-free re-reads (12 conditions x 400 x ~300 tok)', 12 * 400 * 300 / TF_NOLOG / 60)
add('L-S1', 'L3', 'damage calibration (9 sets x 6 evals x 16 refs x 2 fwd x 20 ms) + H damages', (9 * 6 * 16 * 2 * 0.02 + 3) / 60)
# L9: completion clamps, erasure, PIP probes
clamp = gen_minutes(400, 80)
add('L-S1', 'L9', 'completion clamps: H25, H50, 2 random pops x {25, 50} (6 conditions x 400 x 80 tok)', 6 * clamp)
erase = math.ceil(400 / 32) * (32 * 0.025 * HOOK + 0.3) / 60.
add('L-S1', 'L9', 'erasure: 14 ablation conditions x 400 greedy answers (<= 32 tok)', 14 * erase)
add('L-S1', 'L9', 'erasure damage calibration', 0.5)
add('L-S1', 'L9', 'CETT recompute on held-out items (0.12M tok) + rotated-basis CETT', 0.12e6 / TF_NOLOG / 60 * 2 + 0.3)

# ---------------- Session L-S2 (account B; inputs: D long-context stimuli) ----
reads = 128 * 16384
add('L-S2', 'L5', 'TF reads of 128 D texts x 16,384 tokens (fit 16 S1 + 16 S2; test 24 x S1, S2, S4(S1), S4(S2)) with hooks', reads / LONG16K / 60 * HOOK)
add('L-S2', 'L5', 'optional S3 + S4(S3) from D (48 x 16,384)', 48 * 16384 / LONG16K / 60 * HOOK)
add('L-S2', 'L10', 'own-past log-probs by chunked lm_head (64 texts x 16,384; 0.6 GFLOP/tok at 15.8 TFLOP/s)', 64 * 16384 * 0.6e9 / 15.8e12 / 60)
fsum = sum(2 ** k for k in range(7, 15))  # 128..16384
add('L-S2', 'L10', 'foreign passes: 40 S1 docs x sum(L)=%d tokens' % fsum, 40 * fsum / LONG_BLEND_FOREIGN / 60)
add('L-S2', 'L10', 'foreign passes for 24 S3 docs (optional)', 24 * fsum / LONG_BLEND_FOREIGN / 60)
add('L-S2', 'L6', 'depth-cluster eigen-analyses (128 texts x 29 blocks x eigh(1024) at 30 ms) + null on 16 texts x 20 perms', (128 * 29 * 0.03 + 16 * 29 * 20 * 0.03) / 60)
add('L-S2', 'L5', 'community power iteration, Delta, dwell, dip tests, 4-state flux', 3.0)

# ---------------- Session L-S3 (account B; inputs: K H-set) : dyads --------
def dyad_minutes(n_conv, replies_per_agent, agents=2, new_tok=40, sys_tok=110, per_turn_growth=60,
                 probes_at=(1, 3, 6), batch=32):
    """Re-prefill each turn (simple, exact). Context of the speaking agent at global turn g:
    sys_tok + g * per_turn_growth (both agents' replies plus template)."""
    calls = math.ceil(n_conv / batch)
    total = 0.0
    nturns = replies_per_agent * agents
    for g in range(nturns):
        ctx = sys_tok + g * per_turn_growth
        prefill = batch * ctx / TF_LOGITS          # prefill of the batch (logits only last position, use 5,000 tok/s)
        decode = new_tok * DEC32 * HOOK
        total += prefill + decode
    # probes: one forward per agent at the listed replies (branch, ~30 extra tokens on the context)
    for r in probes_at:
        for a in range(agents):
            g = (r - 1) * agents + a
            ctx = sys_tok + (g + 1) * per_turn_growth + 30
            total += batch * ctx / TF_LOGITS
    return calls * total / 60.

d6 = dyad_minutes(128, 6)
add('L-S3', 'L8', 'stance calibration texts (32 topics x 2 stances x 4 seeds turn-1 replies)', gen_minutes(256, 40) + 0.2)
add('L-S3', 'L8', 'dyads: 6 conditions x 128 conversations x 6 replies per agent', 6 * d6)
add('L-S3', 'L8', 'solo controls: 128 conversations x 6 replies (one agent)', dyad_minutes(128, 6, agents=1))
add('L-S3', 'L8', 'F-fact contagion: 256 conversations x 3 replies per agent + final answers', dyad_minutes(256, 3, probes_at=(3,)))
add('L-S3', 'L8', 'damage calibration of 2 random populations (alpha 2 and 0)', 0.5)

# ---------------- Session L-S4 (account A; analysis, GPU used for fits) --------
add('L-S4', 'L1', 'pseudo-likelihood fits (H64, SEL64, 99 controls, 5 planted, position/latent-field refits) + MCMC checks + C(T)', 15.0)
add('L-S4', 'L1', 'I2/IN on 20 ten-unit subsets per population (exact enumeration)', 2.0)
add('L-S4', 'L2', 'HD, exit hazard, Markov surrogates (100 per population), bistability, EBM line', 8.0)
add('L-S4', 'L4', 'S_dot pooled, D_n - D_(n-1), A_k, kinetic Ising', 5.0)
add('L-S4', 'L7', 'S_macro / Psi (CPU-bound)', 4.0)
add('L-S4', 'L9', 'PIP probe sweeps (~4,000 probe fits)', 6.0)
add('L-S4', 'L8', 'kinetic Ising fits, S_macro of dyads, contagion', 2.0)
add('L-S4', 'L11', 'lens comparison regressions', 2.0)
add('L-S4', 'all', 'gates, report', 3.0)

# ------------------------------------------------------------------------
from collections import defaultdict
by_sess = defaultdict(float); by_meas = defaultdict(float)
for s, m, w, mi in rows:
    by_sess[s] += mi; by_meas[m] += mi
for s, m, w, mi in rows:
    print(f'{s:5s} {m:4s} {mi:7.2f}  {w}')
print()
tot = 0.0; tot_margin = 0.0
for s in sorted(by_sess):
    work = by_sess[s]
    print(f'{s}: work {work:6.1f} min; with setup {work + SETUP:6.1f}; with setup + 1.5 x work {SETUP + 1.5 * work:6.1f}')
    tot += work + SETUP; tot_margin += SETUP + 1.5 * work
print(f'TOTAL work+setup {tot:.1f} min = {tot/60:.2f} h; with 1.5 margin {tot_margin:.1f} min = {tot_margin/60:.2f} h')
print()
for m in sorted(by_meas):
    print(f'{m:5s} {by_meas[m]:6.1f} min')
opt = sum(mi for s, m, w, mi in rows if 'optional' in w)
print('optional (S3-dependent) minutes:', round(opt, 1))

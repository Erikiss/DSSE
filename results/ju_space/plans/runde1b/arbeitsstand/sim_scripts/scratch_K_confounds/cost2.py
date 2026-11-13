# Cost model for part K (estimates only; no model is run).
# Rates: round-2 plan given facts (plan:1376-1390): 2.8 GFLOP/token through 28 blocks, +0.6 GFLOP where logits are formed,
# 15.8 TFLOP/s effective fp32 (TF32 off) for batched passes (plan-implied, not measured), decode host ~20 ms/step.
GF = 1e9
R = 15.8e12          # effective FLOP/s
F17 = 3.4 * GF       # Qwen3-1.7B per token incl. logits
F17_nolog = 2.8 * GF
F2B = 3.5 * GF       # Qwen3.5-2B per token incl. 248k head (est.)
FMTP = 1.14 * GF     # Qwen3.5-2B MTP step incl. head (est.)
HOST = 0.020         # s per decode step (host)

def decode_cost(rows, steps, f=F17, batch=320):
    # rows decoded for `steps` steps, batched `batch` rows per step
    nb = -(-rows // batch)
    per_step = lambda r: HOST + r * f / R
    full, rem = divmod(rows, batch)
    t = full * steps * per_step(batch) + (steps * per_step(rem) if rem else 0)
    return t

def fwd_cost(tokens, f=F17):
    return tokens * f / R

out = {}
# ---------- K0 pilot (engineering only) ----------
out['K0 pilot'] = 10 * 60
# ---------- K2 H-neuron identification ----------
# TriviaQA screening: up to 8,000 questions x 10 samples, <=24 new tokens, prompt ~40 tokens (prefill once per question)
t = decode_cost(8000 * 10, 24) + fwd_cost(8000 * 40)
out['K2a TQA screening (8,000 q x 10)'] = t
# ACH-short screening for H_ar: 3,000 items x 10 samples, <=8 new tokens, prompt ~45 tokens
out['K2b ACH screening (3,000 x 10)'] = decode_cost(3000 * 10, 8) + fwd_cost(3000 * 45)
# CETT extraction: 4,000 TQA rows + 4,000 ACH rows (two vectors per response from one teacher-forced pass), ~60 tokens
out['K2c CETT extraction'] = fwd_cost(4000 * 60 + 4000 * 60) * 1.5  # x1.5 for hooks on 28 down_proj inputs
# L1 paths: FISTA ~4 ms/iteration on 4,000 x 172,032 fp32 (int_machinery); 16 C x 600 it + final + 20 bootstrap refits x 600 + L2 fit
it = 16 * 600 + 600 + 20 * 600 + 1000
out['K2d L1 paths (two sets)'] = 2 * it * 0.004
# C selection: one greedy suppression run per C value: TQA 1,000 q (<=24 tok), ACH 500 items (<=8 tok)
out['K2e C selection suppression runs'] = 16 * (decode_cost(1000, 24) + fwd_cost(1000 * 40)) + 16 * (decode_cost(500, 8) + fwd_cost(500 * 45))
# random-set damage matching: 400 candidates x 2 alphas x (16 refs x ~60 tokens) per H-set, two H-sets
out['K2f RS39 damage matching'] = 2 * 400 * 2 * fwd_cost(16 * 60)
# ---------- K3 construct validity ----------
# K3a detection sets: S3 1,000 q x 1 sample; ACH single 600; NE 300 (<=24 tok); CETT reads
out['K3a detection sets'] = decode_cost(1000 + 600 + 300, 24) + fwd_cost(1900 * 60) * 2.5
# K3d scaling battery: H at 5 alphas + RS39 at 2 alphas = 83 configs; battery 200 ACH + 300 TQA + 300 FP + 300 MC + 300 SY(2 turns) + 300 NE
prompts = 6 * 150
per_cfg = decode_cost(prompts + 150, 24) + fwd_cost((prompts + 150) * 60, F17_nolog) + 1.0
out['K3d scaling battery (63 configs)'] = 63 * per_cfg
# K3e neuron-lens round trip: WikiText 1M tokens for 99.8th percentiles; ~560 neurons x 2 poles x 10 seeds chains (2 streams, 17 steps) + feed-back
out['K3e percentiles (1M tokens)'] = fwd_cost(1_000_000, F17_nolog) * 1.3
out['K3e neuron-lens chains + round trip'] = decode_cost(560 * 2 * 10 * 2, 17) + fwd_cost(560 * 2 * 10 * 34)
# ---------- K4 chain lens primary ----------
# per cell: variants x 10 seeds x 2 streams x 17 steps; verifier: chains fed back (34 tokens); revisions ~ +30 % of verifier
def cell(variants, seeds=10, steps=17, verify=True):
    rows = variants * seeds * 2
    t = decode_cost(rows, steps, batch=rows)  # one cell alone; batched below by dividing host
    return rows, steps
# batch 4 cells per decode step (320 rows); host shared
def cells_cost(ncells, variants, seeds=10, steps=17, verify_chains=0, f=F17):
    rows = ncells * variants * seeds * 2
    t = decode_cost(rows, steps, f=f)
    t += fwd_cost(ncells * verify_chains * 34, f) * 1.3
    return t
k4a = cells_cost(384 * 4, 2, verify_chains=20)          # 384 items (256 correct + 128 wrong), t0, 4 blocks: real + foreign-state; verifier on 20 chains
k4b = cells_cost(384 * 2, 2, verify_chains=0)           # blocks 14,18: real+foreign seeds, real+neutral seeds
k4c = cells_cost(256 * 4, 2, verify_chains=10)          # t1: real + foreign-state, 256 correct items, 4 blocks
k4d = cells_cost(128 * 2, 1)                            # re-injection-off ablation (real only), blocks 14,18
k4e = cells_cost(64 * 2, 1)                             # greedy (no CD) ablation
k4f = fwd_cost(384 * 4 * 2 * 20)                        # oracle chains, teacher-forced, 2 streams x ~20 tokens
k4g = 4 * 6 * 32 * fwd_cost(16 * 60)                    # re-injection beta calibration: 4 blocks x 6 evals x 32 states
out['K4 chain lens (all cells)'] = k4a + k4b + k4c + k4d + k4e + k4f + k4g
out['K4 mean states (WikiText 200x128)'] = fwd_cost(200 * 128)
# ---------- K5 Qwen3.5-2B calibration ----------
k5_gate = decode_cost(400, 8, f=F2B) + fwd_cost(400 * 45, F2B)
k5_self = cells_cost(150 * 2 * 3, 2, f=F2B)             # self-rollout on 2B: real + foreign-state
k5_nat = cells_cost(150 * 2 * 3, 2, f=FMTP) * 2         # native MTP chain lens, beta_N in {1,0}
k5_misc = fwd_cost(200 * 128, F2B) + 120                # mean states + model load and MTP check
out['K5 Qwen3.5-2B calibration'] = k5_gate + k5_self + k5_nat + k5_misc
# ---------- K6 lens break-off ----------
# K6c clamps: H + RS39 = 40 clamp conditions x 128 cells (64 items x blocks 14,18) x (real + foreign-state)
out['K6c clamp (40 conditions x 128 cells)'] = 40 * cells_cost(128, 2)
# ---------- K7 ----------
# K7a: fronts of 51k neutral + 200k generated tokens, two passes (PCA bases, projection)
out['K7a fronts (2 passes)'] = 2 * fwd_cost(251_000) * 1.3
out['K7a MC ceilings (event-locked)'] = decode_cost(3200 * 16, 8)
# K7b channel-resolved lens: 44 channel variants x 256 cells x 5 seeds
out['K7b channel-resolved lens'] = 44 * cells_cost(256, 1, seeds=5)
# ---------- K8 ----------
# K8a: 700 items x 19 conditions (NONE + 6 types x 3 lengths); per row: ~80 tokens attending a shared cached prefix + 16 greedy tokens + teacher-forced answer (~8)
rows = 700 * 19
out['K8a position/content/length'] = fwd_cost(rows * 90) + decode_cost(rows, 16) + 6 * 3 * 8 * fwd_cost(8192)  # + context prefills (8 instances)
# K8b sessions: 256 sessions x 8 turns x (40-token question + 30 decode steps); configs: clean 1, H FULL 4 doses, H routes 4, RS39 FULL, RS19 KV
cfg = 1 + 4 + 4 + 39 + 9
per = 8 * (decode_cost(192, 30) + fwd_cost(192 * 40, F17_nolog))
out['K8b persistence (57 configs x 192 sessions)'] = cfg * per
# bistability block: 32 templates x 16 seeds = 512 sessions x (clean + H + RS19)
out['K8b bistability (11 configs x 384 sessions)'] = 11 * 2 * per
# K8c long-form: 800 generations x 260 tokens, batch 128 (T 0.7); in-situ CETT hooks; lens reach 1,800 cells (real+foreign, 10 seeds, 9 steps)
out['K8c long-form generation'] = decode_cost(800, 260, batch=128) + fwd_cost(800 * 120)
out['K8c lens reach (1,800 cells)'] = cells_cost(1800, 2, steps=9)
# ---------- K9 hierarchy ----------
# 10 channels x 2 dirs x 24 texts x 3 source positions x 2 (central differences) forwards of 512 tokens, 4 conditions
out['K9 interventional transfer (4 conditions)'] = 4 * (10 * 2 * 16 * 3 * 2) * fwd_cost(512) * 1.15
# ---------- analysis ----------
out['RK-1 reserve: trained MTP head (bf16, 20M tokens) + rollouts'] = 20e6 * 7.8 * GF / 100e12 + 10 * 60
out['RK-2 reserve: J-lens bf16 refit (466 prompts)'] = 60 * 60
out['K-ana (bootstraps, fits; GPU idle mostly)'] = 15 * 60

tot = 0
for k, v in out.items():
    tot += v
    print(f'{k:55s} {v/60:7.1f} min')
print(f'{"TOTAL work":55s} {tot/60:7.1f} min = {tot/3600:.2f} h')
sessions = 7
setup = sessions * 4 * 60
print(f'with x1.5 margin + {sessions} setups x 4 min: {(1.5*tot + setup)/3600:.2f} h')

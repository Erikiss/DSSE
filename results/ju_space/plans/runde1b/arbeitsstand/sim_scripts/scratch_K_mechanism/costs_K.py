# Cost model for part K (round 1b). Planned rates of the frozen round-2 plan (section 7.1) plus
# int_machinery.md corrections. All estimates, not measurements. fp32, TF32 off.
F17 = 2.819e9      # FLOP per token through 28 blocks (Qwen3-1.7B)
LOG17 = 0.622e9    # FLOP per token for full-vocabulary logits
F35 = 2.6e9        # Qwen3.5-2B text backbone per token (estimate: ~1.3B non-embedding params)
LOG35 = 1.02e9     # Qwen3.5-2B logits (248,320 x 2048 x 2)
MTP35 = 0.12e9     # Qwen3.5-2B MTP module per token (60.8M params x 2), logits extra
R = 15.8e12        # plan's implied effective fp32 rate for batched passes
HOST_STEP = 0.020  # s host time per decode step (plan: ~18 ms at batch 32)

def t_fwd(tokens, logits=False, f=F17, lg=LOG17):
    return tokens * (f + (lg if logits else 0.0)) / R

def t_decode(rows, steps, batch, f=F17, lg=LOG17, host=HOST_STEP):
    nb = -(-rows // batch)
    per_step = host + batch * (f + lg) / R
    return nb * steps * per_step

def attn_prefill(T):  # causal prefill attention FLOP per sequence (int_machinery 3.6)
    return 114688.0 * T * T

M = {}
def add(key, sec, note):
    M[key] = (sec / 60.0, note)

# ---------------- K_s1 (account B): corpora, H-neurons, probes, dumps ----------------
# TriviaQA screening: 8,000 questions x 10 samples, <= 24 new tokens, prefill once per question
scr_q = 8000
add('K1.screen', t_fwd(scr_q * 45) + t_decode(scr_q * 10, 24, 160) + 60, 'TriviaQA screening 8k q x 10 samples (+60 s truncation sort overhead)')
add('K1.screen_cap_extra', t_fwd(4000 * 45) + t_decode(4000 * 10, 24, 160), 'reserve: up to 12k questions if yield is low')
add('KS.labels_AR_FAB', t_decode(1200 * 11, 6, 128) + t_decode(300 * 11, 32, 128) + 30, 'G_AR (1,200 x 11) and G_FAB (300 x 11) labels')
add('K1.cett', t_fwd(5100 * 60, logits=True) * 1.3 + 30, 'CETT extraction on ~5,100 responses with hooks (+30 % hook cost)')
add('K1.l1path', 12 * 60, 'GPU FISTA L1 path, 15 C x 5 folds, 20 bootstrap + 100 permutation fits (int_machinery 9-12 min)')
add('K1.csel', t_decode(15 * 600, 24, 128) + t_fwd(15 * 600 * 45) + 30, 'C selection: 15 suppression runs x 600 questions')
# causal alpha sweep on H: 6 alphas x 1,400 rows x 28 steps
add('K1.alpha_H', t_decode(6 * 1400, 28, 128) + t_fwd(6 * 1400 * 60), 'alpha sweep on H (TQA 500, MC 200, SF 200x2, FE 300)')
# damage screening of 100 candidate random sets at 2 alphas: 80 teacher-forced texts x 100 tokens, logits at continuation only
add('K1.dmg_cand', 100 * 2 * t_fwd(80 * 100) + 100 * 2 * 80 * 50 * LOG17 / R, 'damage of 100 candidate random sets at alpha 0 and 3')
add('K1.alpha_R', t_decode(19 * 2 * 700, 28, 128) + t_fwd(19 * 2 * 700 * 60), '19 damage-matched random sets x alpha {0,3} x 700 rows')
add('K1.neu_pct', t_fwd(300000) * 1.3, 'WikiText 300k tokens: mu_l at all blocks, 99.8th percentiles of H and panel neurons')
# probe corpus: own generations 2,400 x 128 tokens
add('K4a.gen', t_decode(2400, 128, 64) + t_fwd(2400 * 40), 'own-generation probe corpus 2,400 x 128 tokens')
add('K4a.accum', t_fwd(600000) * 1.2 + 10 * 17 * 2 * 2048**2 * 600000 / 9.0e12, '600k tokens forward + float64 accumulators (10 feature sets x 17 matrices)')
add('K4a.eval', 140 * 8 * 8000 * 151936 * 2048 * 2 / R + 120, 'held-out scoring 140 channel probes x 8 targets x 8k tokens')
add('KS.dump', t_fwd(400000) * 0.5 + 120, 'front dump (projections, H/panel activations, logprobs) on 400k generated tokens (reuse of passes; extra I/O)')

# ---------------- K_s2 (account B): chain lens CL_17, K7 lens frame, K9 ----------------
cells_ar = 480 * 6 + 240 * 6   # t_ans all items at 6 blocks + t_op for 240 consistent-correct items
cells_qa = 400 * 4
rows_per_cell = 2 * 10 * 12 * 2   # (real, mu) x 10 seeds x 12 steps x (real cell + foreign null)
rowsteps = (cells_ar + cells_qa) * rows_per_cell
add('K6.cl17_main', t_decode(rowsteps, 1, 128) + t_fwd((cells_ar + cells_qa) * 2 * 10 * 34), f'CL_17 main: {cells_ar + cells_qa} cells, {rowsteps/1e6:.2f}M row-steps (+ scaffold prefills)')
add('K6.cl17_rt', 3840 * 1.0e12 / R, 'round-trip verifier B3: 3,840 cells x ~1 TFLOP')
add('K6.cl17_variants', t_decode(4 * 960 * 240, 1, 128), 'variants on 960 cells each: re-injection ON, identity scaffold, seed swap, norm-matched random state')
add('K6.vjp_fov', 2400 * 35 * F17 * 3 / R + 60, 'gradient-norm chain G(s): 2,400 VJPs on short scaffolds')
add('K6.folge_reihe', t_fwd(240 * 6 * 2 * 120) + 60, 'answer-digit comparison L-par / L-jac / CL_17 on 240 items x 6 blocks (extra JVPs)')
k7_rows = 19 * 240 * 2 * 5 * 12 * 2
add('K7.clamp_R', t_decode(k7_rows, 1, 128), '19 random mean-clamps x 240 items x 2 blocks x 5 seeds x 12 steps x 2 rows')
add('K7.clamp_H', t_decode(240 * 2 * 5 * 12 * 2, 1, 128), 'H mean-clamp, same cells')
add('K7.boost', t_decode(20 * 240 * 2 * 5 * 12 * 2, 1, 128), 'H boost + 19 random boosts')
add('K7.qa_clamp', t_decode(20 * 400 * 5 * 12 * 2, 1, 128), 'QA cells at block 22: H + 19 random clamps')
add('K9.A1_neuronlens', t_decode(3 * 320 * 160, 1, 128) + t_fwd(3 * 3200 * 30) + 30, 'neuron lens on |H| <= 60 x 2 poles + 100 random neurons x 2 poles, <= 3 attempts')
add('K9.A2_earlier', t_fwd(3 * 480 * 15 * 70), 'earlier-intermediate reconstruction, 480 intermediates x 15 candidates x <= 3 attempts')

# ---------------- K_s3 (account B): Qwen3.5-2B calibration ----------------
add('K6.q35_labels', 2 * t_decode(1200 * 11, 8, 128, f=F35, lg=LOG35), 'G_AR labels on Qwen3.5-2B (x2 for unmeasured fp32 linear-attention decode)')
add('K6.q35_wiring', t_fwd(2000 * 2, logits=True, f=F35, lg=LOG35) + 30, 'MTP wiring gate GK7 on 2,000 positions')
add('K6.q35_mu', t_fwd(300000, f=F35) * 1.2, 'mean states / transported means on 300k WikiText tokens')
cells35 = 360 * 5
add('K6.q35_mtp', t_decode(cells35 * 480, 1, 128, f=MTP35, lg=LOG35, host=0.004), f'CL_MTP35: {cells35} cells x 480 MTP row-steps')
add('K6.q35_sr', 2 * t_decode(cells35 * 480, 1, 128, f=F35, lg=LOG35), 'CL_SR35 self-rollout (x2 for fp32 linear-attention fallback risk)')
add('K6.q35_variants', 0.25 * (t_decode(cells35 * 480, 1, 128, f=MTP35, lg=LOG35, host=0.004) + 2 * t_decode(cells35 * 480, 1, 128, f=F35, lg=LOG35)), 're-injection ON variants on 25 % of cells')

# ---------------- K_s4 (account A): carriers, Jacobians, hierarchy ----------------
add('KS.axes', t_fwd(64 * 2 * 4 * 60) * 3 + 60, 'rebuild RSI/sea/sports/tone/NEG axes at blocks 10, 20, 22 from R1 token ids; u*_b from R1 dir_*_4')
add('K2.kv_finite', t_fwd(96 * 4 * 14 * 2 * 100), 'KV-only route at the finite push: 96 texts x 4 t0 x 14 sources x 2 signs')
add('K2.tok_regen', t_decode(96 * 4 * 14 * 2, 48, 32) + t_fwd(96 * 4 * 14 * 2 * 100), 'token route: regenerations from t0 (CRN) + clean re-reads')
add('K3.ljac', t_fwd(96 * 8 * 4 * 3 * 2 * 100) + 120, 'L-jac: FD JVPs (state, foreign state, frozen-QK) at 8 blocks x 4 positions x 96 texts')
add('K3.routing_vjp', 1536 * 100 * F17 * 3 / R + 60, 'head routing map: 1,536 VJPs with attention-edge attribution')
add('K3.bridge', t_fwd(100 * 4 * 8 * 2 * 128), 'bridge to the published J-lens: 100 WikiText prompts x 4 blocks x 8 probe vectors')
k5_cond = t_fwd(26 * 4 * 32 * 2 * 128)
add('K5.conditions', 10 * k5_cond + 4 * 60 + 10 * t_decode(32, 128, 32), '10 conditions x 26 sources x 4 t0 x 32 texts x 2 (FD) + 4 long-context extras + text generation')
add('K5.prefill_long', 4 * 32 * (t_fwd(4096) + attn_prefill(4096) / R), 'prefills of 1k/4k prefixes (upper bound: 4k)')
add('K4a.R1chan', t_fwd(600000) * 1.1 + 60, 'R1-derived direction channels: re-pass of the probe corpus with 1-d features')
add('K1.style_to_H', t_decode(21 * 128, 28, 128) + t_fwd(21 * 128 * 60) + 30, 'behavioural style->H link: push along u*_14 at 0.05 nats vs 20 random directions')

# ---------------- K_s5 (account B): model frame (sessions) ----------------
pre_ctx = 300 * 5 * ((t_fwd(400) + attn_prefill(400) / R) + (t_fwd(2000) + attn_prefill(2000) / R))
add('K8a.matched', pre_ctx + t_decode(3000, 24, 32) + 60, 'K8a: 300 targets x 5 contexts x 2 positions (400, 2,000 tokens)')
sess = 1856 + 768
add('K8b.sessions', t_fwd(sess * 650) + t_decode(sess, 12 * 24, 32) * 1.15 + 120, f'K8b: {sess} sessions x 12 turns with incremental caches (+15 % for per-arm cache handling)')

# ---------------- K_ana ----------------
add('Kana.stats', 15 * 60 + 5 * 60, 'bootstraps, verdicts, R1-channel probe scoring')

sessions = {
    'K_s1': ['K1.screen', 'KS.labels_AR_FAB', 'K1.cett', 'K1.l1path', 'K1.csel', 'K1.alpha_H', 'K1.dmg_cand', 'K1.alpha_R', 'K1.neu_pct', 'K4a.gen', 'K4a.accum', 'K4a.eval', 'KS.dump'],
    'K_s2': ['K6.cl17_main', 'K6.cl17_rt', 'K6.cl17_variants', 'K6.vjp_fov', 'K6.folge_reihe', 'K7.clamp_R', 'K7.clamp_H', 'K7.boost', 'K7.qa_clamp', 'K9.A1_neuronlens', 'K9.A2_earlier'],
    'K_s3': ['K6.q35_labels', 'K6.q35_wiring', 'K6.q35_mu', 'K6.q35_mtp', 'K6.q35_sr', 'K6.q35_variants'],
    'K_s4': ['KS.axes', 'K2.kv_finite', 'K2.tok_regen', 'K3.ljac', 'K3.routing_vjp', 'K3.bridge', 'K5.conditions', 'K5.prefill_long', 'K4a.R1chan', 'K1.style_to_H'],
    'K_s5': ['K8a.matched', 'K8b.sessions'],
    'K_ana': ['Kana.stats'],
}
setup = {'K_s1': 4 + 5, 'K_s2': 4 + 1, 'K_s3': 4 + 8, 'K_s4': 4 + 4, 'K_s5': 4, 'K_ana': 3 + 2}
tot_h = 0.0
for s, keys in sessions.items():
    w = sum(M[k][0] for k in keys)
    tot = setup[s] + 1.5 * w
    tot_h += tot / 60
    print(f'{s}: work {w:6.1f} min, setup {setup[s]:4.1f}, with margin {tot:6.1f} min = {tot/60:.2f} h')
    for k in keys:
        print(f'    {k:22s} {M[k][0]:6.2f} min  {M[k][1]}')
print(f'TOTAL with margin: {tot_h:.2f} A100-h')
print(f"reserve (screening cap extra): {M['K1.screen_cap_extra'][0]:.1f} min")

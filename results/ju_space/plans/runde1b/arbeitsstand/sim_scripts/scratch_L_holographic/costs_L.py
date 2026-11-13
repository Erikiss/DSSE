"""Part L (holographic / information-theoretic angle) cost model. Rates are planning rates (unverified, to be replaced by the
shared long-context pilot): fp32, TF32 off, SDPA memory-efficient kernel.
 - teacher-forced short (<= 2k tokens): 5,000 tok/s (round-2 R-full, plan:1383-1387)
 - generation with neuron/state hooks, batch 64, contexts <= 1.1k: 1,000 tok/s (lf_holographic planning rate; conservative)
 - short generation, batch 128, answers <= 96 tokens: 2,000 tok/s
 - long prefill B = 1 (int_machinery 3.6, upper bounds): 4k 1.0 s, 8k 2.5 s, 16k 6.8 s, 32k 21 s; <= 2k: 0.4 s
 - decode with own caches at ~4k context, batch 24: 450 tok/s (int_machinery 3.6, DynamicCache rewrite included)
"""
import json

R_TF = 5000.0; R_GEN = 1000.0; R_GEN_SHORT = 2000.0; R_DEC4K = 450.0
PREFILL = {2**7: 0.05, 2**8: 0.07, 2**9: 0.1, 2**10: 0.2, 2**11: 0.4, 2**12: 1.0, 2**13: 2.5, 2**14: 6.8, 2**15: 21.0}
m = {}

# ---------------- S-L-a ----------------
n_traj = 4 * 64 * 2 * 2           # families x prompts x T x keys
tok_traj = 1024 + 80              # new tokens + prompt
m['TB-L generation (1,024 traj x 1,024 tokens, hooks)'] = n_traj * 1024 / R_GEN / 60
m['L1c foreign-context teacher-forced pass'] = 2 * n_traj * tok_traj / R_TF / 60   # C' and C'' (null band)
m['L6c sentence-reversal pass (T 1.0 half)'] = (n_traj / 2) * tok_traj / R_TF / 60
m['L0 reference percentiles (G_NEU 100k tokens)'] = 1e5 / R_TF / 60
# twins: 128 trajectories x 2 starts x (twin + H + 2 random) rows x (prefix re-prefill ~ 400 + 80 new)
rows = 128 * 2 * 4
m['L2b twins: prefix prefill'] = rows * 400 / R_TF / 60
m['L2b twins: 80-token generation'] = rows * 80 / R_GEN / 60
m['L2b damage calibration (H and random sets at alpha 0, 2; REF 64 x 128 tokens, <= 6 evals each)'] = 4 * 6 * 2 * 64 * 128 / R_TF / 60
# L1 arithmetic: own CoT (600 items x 5 samples x 96 tokens) + sub-answers (600 x 2.5 x 11 x 12) + t* pass
m['L1a/b own-CoT generation'] = 600 * 5 * 96 / R_GEN_SHORT / 60
m['L1a/b isolated sub-answers'] = 600 * 2.5 * 11 * 12 / R_GEN_SHORT / 60
m['L1b model-frame a_H(t*) pass'] = 600 * 60 / R_TF / 60

# ---------------- S-L-b ----------------
nat_docs = 24
per_doc = 2 * sum(PREFILL.values())                     # joint + product per L (paired derangement)
m['L4 natural documents (24 x L=2^7..2^15, joint+product)'] = nat_docs * per_doc / 60
m['L4 shuffled-sentence control (8 docs, L <= 2^13)'] = 8 * 2 * sum(v for k, v in PREFILL.items() if k <= 2**13) / 60
m['L4 copy control (4 docs, L <= 2^12)'] = 4 * 2 * sum(v for k, v in PREFILL.items() if k <= 2**12) / 60
m['L4 self-generated texts (24 x 4,096 new tokens, 512-token seed)'] = 24 * 4096 / R_DEC4K / 60
m['L4 self-text MI passes (L=2^7..2^12) + local-iota passes'] = (24 * 2 * sum(v for k, v in PREFILL.items() if k <= 2**12) + 24 * 15 * 2 * 512 / R_TF) / 60
m['L4 engine parity (4 docs at 16k and 32k, cached vs full)'] = 4 * 2 * (6.8 + 21.0) / 60

# ---------------- S-L-c ----------------
n_items = 1200 + 300 + 300                               # CC+CI (S_L), FAB, NQ-like OOD reserve
m['L5 CETT extraction (all 172,032 neurons) + rotated-basis pass'] = 2 * n_items * 70 / R_TF / 60 + 28 * 2 / 60
m['L5 probe sweeps (linear, 4 families, 10 sizes, nested CV)'] = 15.0
m['L5 MLP probes (3 sizes) and I3 unions'] = 5.0
m['L5 planted-code calibration (5 codes x 5 seeds, subset of sizes)'] = 5.0
# causal erasure: 21 conditions x (1,200 greedy answers x 24 tokens + teacher-forced gold 1,200 x 50 + damage 64 x 128)
per_cond = 1200 * 24 / R_GEN_SHORT + 1200 * 50 / R_TF + 64 * 128 / R_TF
m['L5 causal erasure (21 conditions)'] = 21 * per_cond / 60
m['L5 damage matching of ablation sets'] = 2.0
# L8 dialogues
msgs = (192 + 128 + 256) * 16 + (64 + 64) * 8
m['L8 dialogue prefill (re-rendered context per message, avg 450 tokens)'] = msgs * 450 / R_TF / 60
m['L8 dialogue generation (<= 40 tokens per message)'] = msgs * 40 / R_GEN / 60
m['L8 stance probe (context-free judgement of every stance message)'] = (192 * 16 + 64 * 8) * 110 / R_TF / 60
m['L8 damage calibration of agent-A interventions'] = 2.0

# ---------------- S-L-d (analysis, GPU) ----------------
m['L2a RQA + IAAFT / multivariate surrogates (H and 39 random populations)'] = 10.0
m['L3 CCM (kNN simplex, libraries, lags, nulls)'] = 15.0
m['L7 pseudo-likelihood fits (40 populations x 4 lambda), basins, modularity'] = 10.0
m['L0 synthetic calibrations (CCM, surrogates, Ising, Psi, PIP planted) + L1/L6/L8 CPU analyses + verdicts'] = 20.0

sessions = {
 'S-L-a': [k for k in m if k.startswith(('TB-L', 'L1c', 'L6c', 'L0 reference', 'L2b', 'L1a', 'L1b'))],
 'S-L-b': [k for k in m if k.startswith('L4')],
 'S-L-c': [k for k in m if k.startswith(('L5', 'L8'))],
 'S-L-d': [k for k in m if k.startswith(('L2a', 'L3', 'L7', 'L0 synthetic'))],
}
SETUP = 4.0; MARGIN = 1.5
tot_work = 0; tot_plan = 0; out = {}
for s, keys in sessions.items():
    w = sum(m[k] for k in keys)
    plan = SETUP + MARGIN * w
    tot_work += w; tot_plan += plan
    out[s] = {'work_min': round(w, 1), 'planned_min_with_setup_and_x1.5': round(plan, 1), 'items': {k: round(m[k], 2) for k in keys}}
assert sum(len(v) for v in sessions.values()) == len(m), 'unassigned item'
out['total_work_min'] = round(tot_work, 1)
out['total_planned_min'] = round(tot_plan, 1)
out['total_planned_h'] = round(tot_plan / 60, 2)
# half-rate sensitivity
out['half_rate_planned_h'] = round((4 * SETUP + MARGIN * 2 * tot_work) / 60, 2)
print(json.dumps(out, indent=1))
json.dump(out, open('costs_L.json', 'w'), indent=1)

"""Part R cost model (A100-40GB, float32, TF32 off). Rates = round-2 plan 7.1 'given facts' (planned, not measured); the declared
engineering pilot R-P replaces them before the cost table is frozen. Run: taskset -c 3 python3 -I cost_R.py"""
import json, math
GEN = 1.0          # s per generate call, 32 rows x 40 new tokens, prompts <= 30 tokens
GEN80, GEN200, GEN360 = 1.3, 2.0, 2.5
SOFT = 1.5         # s per soft-token generation call, 32 rows x 40 steps (inputs_embeds, custom loop)
RD14, RDX, RALL, RFULL, RSHORT = 0.25, 0.35, 0.40, 0.45, 0.15   # s per 32-text batch: read stopped at 14 / other block / all 28 blocks / full pass with logits / 20-token sentences
FWD, FB = 0.020, 0.060   # single short forward; forward + backward
DMG = 0.64         # one dmg_sym evaluation: 16 references x 2 signs steered forwards (unsteered cached)
LOAD, SETUP, MARGIN = 0.5, 4.0, 1.5      # minutes: model load per process, session setup (pip, clone, model, checks); margin on work
def calls(texts, per=32): return math.ceil(texts / per)
M = {}          # measurement -> dict(session, work_min)
def add(mid, session, sec, note): M[mid] = dict(session=session, work_min=sec / 60.0, note=note)
# ---------------- R-P engineering pilot (declared; throwaway prompts; timings only)
add("R-P", "R-P", 10 * 60, "rates: generate at 20/80/200/360-token prompts, reads, fwd-bwd, soft generation, RefKL, memory, Drive throughput; replay opener on MOCK")
# ---------------- R0 (no model; analysis runtime)
add("R0", "R-S0", 8 * 60, "reproduction, centred nulls, paired tests, joint bootstrap, cross-fit, gates, size/power plants on R1 files (3 blocks)")
add("R8.1", "R-S0", 1 * 60, "0.6B gate keys")
add("R9.5", "R-S0", 0.5 * 60, "copy unlearn_verdict.json, r2_manifest.json")
# ---------------- R1 replay (rk_loop unchanged, reduced-state copy, round-1 keys)
t14 = 384 + 4096; t_x = 384 + 2048; t6r = 384
sec = (5 * LOAD * 60) + calls(t14) * GEN + calls(t14) * RD14 + 24 * 16 * FWD + 5 \
      + 2 * (calls(t_x) * GEN + calls(t_x) * RDX + 12 * 16 * FWD) + calls(t6r) * (GEN + RDX) + 60
add("R1", "R-S1", sec, "block 14 pilot@alpha + loops@1.5; blocks 18, 6 pilot + loops; block 6 pilot@1.5; stats")
# ---------------- R2 per block
def r2_block(b, first):
    rd = RD14 if b == 14 else RDX
    ax_gen = calls(192 * 8) * GEN if first else 0
    ax_read = calls(256 * 8) * rd + 10 + 1
    pairs_round = (512 * 4 + 256 * 3) + (512 + 256 + 256) + (512 + 256 + 256)   # DN: RSI_a,RSI_b,SEA_t,SPO_t (512) + SEA_raw,RND_1,RND_2 (256); FX; NP
    texts_round = 2 * pairs_round
    loops = 4 * (calls(texts_round) * GEN + calls(texts_round) * rd)
    ext = 2 * (calls(2 * 1024) * (GEN + rd))                                     # RSI_a, RSI_b DN rounds 5-6
    cal = (7 * 4 + 2 * 2) * 6 * DMG + 3 * 4 * 6 * DMG + 3 * 4 * 2 * DMG          # DN (incl. rounds 5-6), NP calibrations, FX damage logs
    return LOAD * 60 + ax_gen + ax_read + loops + ext + cal, pairs_round
for b, first in ((14, True), (18, False), (6, False)):
    s, pr = r2_block(b, first); add(f"R2@{b}", f"R-S2/{b}", s, f"{pr} pairs per round x 4 rounds (+ RSI rounds 5-6), axes, calibrations")
# ---------------- R3 + R4 session
txt0 = 256 * 16
sec_txt0 = calls(txt0) * GEN + calls(txt0) * (RD14 + RDX + RDX)
sec_fdt = 4096 * FB + 2 * 2048 * FB
def dose(b, ncell):
    rd = RD14 if b == 14 else RDX; texts = ncell * 1024 + 512
    return calls(texts) * (GEN + rd) + ncell * DMG
sec_dose = dose(14, 4 * 6) + dose(18, 2 * 6) + dose(6, 2 * 6)                   # 5 grid doses + the DN dose per direction
sec_readdir = 3 * (24 * 64 * FB)
sec_direct = 3 * (calls(2048) * GEN + calls(2048) * RDX + 2 * 6 * DMG)
add("R3.3", "R-S3", sec_dose, "dose cells (+,0,-): block 14 4 dirs x 6 doses, blocks 18/6 2 dirs x 6 doses, 512 pairs each")
add("R3.4+R4.1", "R-S3", sec_txt0 + sec_fdt, "TXT0 4,096 unsteered texts, reads at 14/18/6, FDT scores 4,096 + 2 x 2,048")
add("R4.2", "R-S3", sec_readdir, "local read directions of 24 word sets x 64 texts per block")
add("R4.3", "R-S3", sec_direct, "direct forward cells along v_max and v_RSI, 512 pairs per block")
M["R3.4+R4.1"]["work_min"] += 3 * LOAD
# ---------------- R5 lens session
sec51 = 8 * 4 * 4 * 2 * 2 * RFULL + 8 * 2 * RFULL + 32 * 6 * DMG + calls(2048) * RALL + 30
sec52 = calls(7 * 21 * 2 * 60) * RSHORT + 20
sec53 = 24 * 6 * DMG + 24 * 2 * 2 * RFULL + 10
sec55 = 3 * 8 * (calls(128) * GEN + calls(128) * RDX)
add("R5.1", "R-S5", sec51 + LOAD * 60 + 60, "teacher-forced dose series at 8 blocks x 4 directions x 4 doses x 2 signs on 64 texts; lens download + sha256")
add("R5.2", "R-S5", sec52, "A3 stimuli: lens + direct finite differences, u_T_perp and 20 random directions at 7 blocks")
add("R5.3", "R-S5", sec53, "T4 use test at 0.05 nats (4 axes + 20 nulls), T3 lens metric")
add("R5.5", "R-S5", sec55, "verified label of the style mode: <= 8 verifier rounds x 128 texts at 3 blocks")
add("R5.4 (conditional)", "R-S5b", 60 * 60, "bf16 J-lens refit on ~300 in-distribution 128-token texts (only if R5.1 fails at 14 or at all blocks >= 22)")
# ---------------- R6 session
sec_soft1 = 12 * (calls(512) * SOFT + calls(512) * 0.3) + 12 * 6 * DMG
sec_soft_it = 6 * 4 * (calls(512) * SOFT + calls(512) * 0.3) + 24 * 6 * DMG
add("R6.1", "R-S6", sec_soft1 + sec_soft_it + LOAD * 60, "soft-token latent loop: one pass per axis at 3 blocks; 4-round latent loops from u_T and RND at 3 blocks")
t1_T1 = 1024; turns = 5 * 2 * 1024
sec_k = calls(t1_T1) * (GEN + RD14) + calls(turns) * (GEN200 + RD14) + 2 * 6 * DMG
add("R6.2", "R-S6", sec_k, "K chain, block 14, T 0.7 (turn 1 = R3 DN cell) and T 1.0 (new turn 1), turns 2-6, 512 pairs")
sec_rep = calls(1024) * (GEN + RD14) + calls(1024) * (GEN80 + RD14)
add("R6.3", "R-S6", sec_rep, "replication of the round-2 T 1.0 carry cell (u* push at alpha_0), 512 pairs, turns 1-2")
# ---------------- R7 session (block 14)
sec71 = LOAD * 60 + 2 + 10 * 16 * FWD * 2 + calls(10 * 512) * (1.15 + RD14) + calls(3 * 512 + 256 + 2 * 512) * (1.15 + RD14) + 60 * 17 * FWD
sec72 = calls(512) * (GEN200 + RD14)
sec73 = calls(2 * (256 + 3 * 512)) * (1.15 + RD14) + 7 * 6 * DMG
add("R7.1", "R-S7", sec71, "10 erasure variants: damage, top-1, instructed gaps; selected variant: foreign axes, unprompted, random controls, H30/RGI30")
add("R7.2", "R-S7", sec72, "C-Q4a cued texts (gated)")
add("R7.3", "R-S7", sec73, "C-Q4b ablated loop + twin, 256 pairs x 3 rounds + twin start (gated)")
# ---------------- R8.2 (conditional), 0.6B
gen06 = 0.7; rd06 = 0.15
sec82 = 90 + 15 + calls(256) * (gen06 + rd06) + calls(2048) * (gen06 + rd06) + 4 * calls(2 * (512 * 3 + 256)) * (gen06 + rd06) + 16 * 6 * DMG * 0.5
add("R8.2 (conditional)", "R-S8", sec82, "Qwen3-0.6B block 14: references, axes, DN loops RSI_a, RSI_b, SEA_t (512), RND_1 (256)")
# ---------------- analysis
add("R-ana", "R-Sana", 15 * 60, "all part-R statistics and verdicts (joint bootstraps on the GPU)")
add("R9.1-9.4", "R-Sana", 1 * 60, "provenance / code tables, report")
# ---------------- sessions: setup, copy-in (placeholder 10 min/GB until the pilot measures Drive throughput)
copyin = {"R-P": 0.0, "R-S0": 0.45, "R-S1": 0.33, "R-S2/14": 0.12, "R-S2/18": 0.15, "R-S2/6": 0.15, "R-S3": 0.2, "R-S5": 0.15, "R-S5b": 0.1,
          "R-S6": 0.2, "R-S7": 0.15, "R-S8": 0.0, "R-Sana": 2.5}
setup = {s: (3.0 if s in ("R-S0", "R-Sana") else SETUP) for s in copyin}
sess = {}
for mid, v in M.items():
    s = v["session"]; sess.setdefault(s, dict(work=0.0, items=[])); sess[s]["work"] += v["work_min"]; sess[s]["items"].append(mid)
tot = 0.0; tot_unc = 0.0; table = []
for s, v in sess.items():
    t = setup[s] + 10 * copyin[s] + MARGIN * v["work"]; cond = s in ("R-S5b", "R-S8")
    table.append(dict(session=s, items=v["items"], work_min=round(v["work"], 1), setup_min=setup[s], copyin_min=round(10 * copyin[s], 1), total_min=round(t, 1), conditional=cond))
    tot += t; tot_unc += 0 if cond else t
per = {mid: dict(session=v["session"], work_min=round(v["work_min"], 2), with_margin_min=round(MARGIN * v["work_min"], 1), note=v["note"]) for mid, v in M.items()}
# sensitivity: generation at half the planned speed (GEN etc. doubled) -> rough factor from the generation share
print(json.dumps(dict(per_measurement=per, sessions=table, total_A100_min=round(tot, 1), total_A100_h=round(tot / 60, 2),
                      unconditional_A100_h=round(tot_unc / 60, 2)), indent=1))

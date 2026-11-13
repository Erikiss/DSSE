"""Part R MDE and cost arithmetic from round-1/2 JSON numbers only (no model). Run: taskset -c 3 python3 -I mde_R.py
Assumptions (stated in the spec): per-pair read-back noise roughly independent of the push size; signal linear in the push;
damage ~ alpha^k with k in [1.74, 2.0]; within-prompt noise share f from RSI_a at block 14 (0.54)."""
import json, math
import numpy as np
R = "/home/user/DSSE/results/ju_space/runde2_ergebnisse/"
RV = json.load(open(R + "rgi_verdict.json")); S = json.load(open(R + "round2_summary.json")); TV = json.load(open(R + "theta_verdict.json"))
out = {}
# --- within-prompt share f (as in w_gates/mde.py)
ref = S["parts"]["C"]["key_numbers"]["C_Q4b"]["reference"]; full = TV["blocks"]["14"]["B_Q0"]["persistence"]["RSI_a"]
fs = [2 * (1 - (1 / full[n] - 1) / (1 / ref[n]["rel"] - 1)) for n in ("1", "2", "3", "4")]; f = float(np.mean(fs)); out["f"] = f
def var_fac(N, Sd):            # variance of a prompt mean relative to one pair per prompt at N prompts: (1-f + f/S)/N
    return (1 - f + f / Sd) / N
def sd_fac(N, Sd, N0, S0): return math.sqrt(var_fac(N, Sd) / var_fac(N0, S0))
# --- SD(c) on a fixed axis from the A6 bootstraps (round-1 loops, round-1 push), per block and loop
sdc = {}
for B in ("14", "18", "6"):
    for L in ("RSI_a", "SEA", "RND_1"):
        v = RV["A6"]["blocks"][B]["loops"].get(L)
        if not v: continue
        sds = [(v[str(n)]["u_RGI_perp"]["A_ci90"][1] - v[str(n)]["u_RGI_perp"]["A_ci90"][0]) * math.sqrt(v[str(n)]["rel"]) / 3.29 for n in range(1, 5)]
        sdc[(B, L)] = dict(sd=float(np.mean(sds)), pairs=v["1"]["n_pairs"], dmg=[float(x) for x in v["damage_d1_d3"]])
out["sdc_r1"] = {f"{b}/{l}": v for (b, l), v in sdc.items()}
# --- push factor q = alpha_DN / alpha_r1 for a 0.05-nat push, from the round-1 damage of the counted pushes d_1..d_3
def q_of(dmg, k): return (0.05 / dmg) ** (1 / k)
designs = {"64x1": (64, 1), "64x2": (64, 2), "128x2": (128, 2), "256x1": (256, 1), "256x2": (256, 2), "256x4": (256, 4)}
Z1, ZP = 1.645, 0.842          # one-sided 90 % bound, power 0.8
Z01 = 2.326                    # one-sided alpha 0.01
rows = []
for (B, L), v in sdc.items():
    N0, S0 = 64, v["pairs"] // 64; dm = float(np.median(v["dmg"]))
    for k in (1.74, 2.0):
        q = q_of(dm, k)
        for dn, (N, Sd) in designs.items():
            sd = v["sd"] * sd_fac(N, Sd, N0, S0) / q
            rows.append(dict(block=B, loop=L, k=k, q=q, design=dn, sd_c=sd))
out["sd_c_design"] = rows
def get(B, L, k, dn): return [r for r in rows if r["block"] == B and r["loop"] == L and r["k"] == k and r["design"] == dn][0]
# axis-estimation noise (cross-fitted axis removes the self-trace; the joint bootstrap adds this variance): sigma_d / sqrt(1 + N s2)
s2 = {"14": 0.11, "18": 0.14, "6": 0.13}; sig = {"14": 0.09, "18": 0.08, "6": 0.10}       # int_gates 3.3 fit
def sd_axis(B, N): return sig[B] / math.sqrt(1 + N * s2[B])
# rel at the design from round-1 rel (constant signal model), loop L, rounds 2-4
pers = {B: TV["blocks"][B]["B_Q0"]["persistence"] for B in ("14", "18", "6")}
def rel_at(r, var_ratio):      # rel' = rel / (rel + (1 - rel) * v): v = noise variance factor relative to the round-1 design
    return r / (r + (1 - r) * var_ratio)
summ = {}
for B in ("14", "18", "6"):
    summ[B] = {}
    for L, dn in (("RSI_a", "256x2"), ("SEA", "256x2"), ("SEA", "256x1"), ("RND_1", "256x1")):
        if (B, L) not in sdc: continue
        e = {}
        for k in (1.74, 2.0):
            r = get(B, L, k, dn); N, Sd = designs[dn]; sd_tot = math.sqrt(r["sd_c"] ** 2 + sd_axis(B, N) ** 2)
            v0 = var_fac(64, sdc[(B, L)]["pairs"] // 64)
            vr = var_fac(N, Sd) / v0 / r["q"] ** 2          # noise variance relative to signal^2, vs round 1
            rels = [rel_at(pers[B][L if L in pers[B] else "RSI_a"][str(n)], vr) for n in (2, 3, 4)]
            sdA = sd_tot / math.sqrt(min(rels))
            e[f"k{k}"] = dict(q=round(r["q"], 3), sd_c_fixed_axis=round(r["sd_c"], 4), sd_c_with_axis=round(sd_tot, 4), rel_n234=[round(x, 3) for x in rels],
                              sd_A=round(sdA, 4),
                              MDE_A_PC_010=round(max(0.10 + (Z1 + ZP) * sdA, (Z01 + ZP) * sdA), 3),
                              MDE_A_tier025_4cond=round(0.25 + (Z1 + 1.25) * sdA, 3), MDE_A_tier050_4cond=round(0.50 + (Z1 + 1.25) * sdA, 3),
                              unlikely_needs_absA_below=round(0.25 - (Z1 + 1.25) * sdA, 3),
                              MDE_c_paired=round((Z01 + ZP) * sd_tot, 3), bar_limited_round1b_rule=round(0.235 + ZP * sd_tot, 3))
        summ[B][f"{L} {dn}"] = e
out["rules"] = summ
# --- dose-series gain precision (round-2 theta cells: SD of G at 0.5 / 1 / 2 alpha_0, block 14)
cells = TV["blocks"]["14"]["R1"]["points"] if "points" in TV["blocks"]["14"]["R1"] else None
pts = {}
for nm, a, npairs in (("a0.5_T0.7", 0.5, 256), ("a1_T0.7", 1.0, 128), ("a2_T0.7", 2.0, 128)):
    c = TV["blocks"]["14"]["cells"].get(nm) if isinstance(TV["blocks"]["14"].get("cells"), dict) else None
    pts[nm] = c is not None
out["cells_present"] = pts
sdG = {0.5: (0.3432 - 0.2452) / 3.29 * math.sqrt(256), 1.0: (0.3312 - 0.2682) / 3.29 * math.sqrt(128), 2.0: (0.2922 - 0.2645) / 3.29 * math.sqrt(128)}   # SD x sqrt(pairs)
slope = math.log(sdG[0.5] / sdG[2.0]) / math.log(4)
out["gain_sd_per_sqrt_pair"] = {str(k): round(v, 4) for k, v in sdG.items()}; out["gain_sd_exponent"] = round(slope, 3)
dose = {}
for a in (0.25, 0.35, 0.5, 1.0, 2.0):
    sd_pair = sdG[0.5] * (0.5 / a) ** slope                          # per sqrt(pair) at 64 prompts x 4 seeds design
    for dn in ("256x2", "256x4"):
        N, Sd = designs[dn]; fac = sd_fac(N, Sd, 64, 4)              # relative to round 2's 64 x 4 cell
        dose[f"{a}a0 {dn}"] = round(sd_pair / math.sqrt(256) * fac, 4)
out["dose_gain_sd_block14"] = dose
# --- FDT lambda_lin (B-Q7): SD 0.380 at 128 texts along u*
lam = {}
for n in (1024, 2048, 4096):
    lam[str(n)] = dict(fixed_dir=round(0.380 * math.sqrt(128 / n), 4), crossfit_half=round(0.380 * math.sqrt(128 / (n / 2)), 4))
out["fdt_sd"] = lam
# --- K on the RSI axis: denominator from RSI_a eval 0 at block 14 (projection 1.297 at 0.1135 nats, 128 pairs, SD(c)=0.0286 -> SD(proj)=SD(c)*|d|)
ev = TV["blocks"]["14"]["B_Q0"]["evaluations"]["RSI_a"]["0"]; cbar = TV["blocks"]["14"]["B_Q0"]["cbar_b"]["pt"]; alpha = TV["blocks"]["14"]["B_Q0"]["alpha"]
proj = ev["g_par"]["pt"] * alpha * cbar; dnorm = math.sqrt(ev["U"]["pt"] + ev["NF"]["pt"] ** 2); sdp128 = sdc[("14", "RSI_a")]["sd"] * dnorm
Kt = {}
for k in (1.74, 2.0):
    q = q_of(0.1135, k); D1 = proj * q
    for npairs, dn in ((512, "256x2"), (1024, "256x4"), (2048, "256x8")):
        N = 256; Sd = npairs // 256; sdD = sdp128 * math.sqrt(var_fac(N, Sd) / var_fac(64, 2))
        for Ktrue in (0.5, 1.0):
            sdN = sdD                                                 # turn-n noise assumed equal to turn-1 noise
            sdK = Ktrue * math.sqrt((sdN / (Ktrue * D1)) ** 2 + (sdD / D1) ** 2)
            Kt[f"k{k} {dn} K{Ktrue}"] = dict(D1=round(D1, 3), sd_D1=round(sdD, 4), rel_sd_D1=round(sdD / D1, 3), sd_K2=round(sdK, 3),
                                              sd_cum_per_turn=round(sdN / D1, 3))
out["K_rsi_axis"] = dict(proj_d1_at_0p1135=round(proj, 4), norm_d1=round(dnorm, 3), sd_proj_128=round(sdp128, 4), table=Kt)
# K on u_T_perp after the u* push (round-2 T1.0 cell): SD 0.2637 at 128 pairs
out["K_uT_after_ustar_T1"] = {str(n): round(0.2637 * math.sqrt(var_fac(256, n // 256) / var_fac(64, 2)), 4) for n in (512, 1024, 2048)}
# --- erasure ratio (C-Q3): 90 % interval [0.0665, 0.2425] at 64 prompts
sd_r = (0.2425 - 0.0665) / 3.29; out["erasure_ratio_sd"] = {"64": round(sd_r, 4), "256": round(sd_r / 2, 4)}
out["erasure_MDE_256"] = dict(point_rule_true_ratio_below=round(0.5 - ZP * sd_r / 2, 3), upper_rule_true_ratio_below=round(0.75 - (Z1 + ZP) * sd_r / 2, 3))
print(json.dumps(out, indent=1, default=str))

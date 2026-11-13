import json, math
d = json.load(open("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/scratch_R/mde_R.json"))
Z1, ZP, Z01 = 1.645, 0.842, 2.326
rho = 0.8                      # correlation of A_3 and A_4 within one loop (consecutive rounds of one loop share most of their noise path)
out = {}
for B in ("14", "18", "6"):
    for k in ("k1.74", "k2.0"):
        e = d["rules"][B]["RSI_a 256x2"][k]; sdA = e["sd_A"]
        sd_bar = sdA * math.sqrt((1 + rho) / 4)            # mean of A_3, A_4 over two independent loops
        out[f"{B} {k}"] = dict(sd_A=sdA, sd_Abar=round(sd_bar, 4),
                               MDE_substantial=round(0.25 + (Z1 + ZP) * sd_bar, 3), MDE_dominated=round(0.50 + (Z1 + ZP) * sd_bar, 3),
                               unlikely_if_true_below=round(0.25 - (Z1 + ZP) * sd_bar, 3), MANIP_MDE_A1=round(max(0.10 + (Z1 + 1.25) * sdA, (Z01 + 1.25) * sdA), 3),
                               PC_MDE=d["rules"][B].get("SEA 256x2", {}).get(k, {}).get("MDE_A_PC_010"))
# delta rule: paired difference of A_bar between two blocks (independent keys; prompts shared -> treat as independent, conservative)
for (b1, b2) in (("14", "18"), ("14", "6"), ("18", "6")):
    s = math.sqrt(out[f"{b1} k1.74"]["sd_Abar"] ** 2 + out[f"{b2} k1.74"]["sd_Abar"] ** 2)
    out[f"delta {b1}-{b2}"] = dict(sd_diff=round(s, 4), MDE=round((Z1 + ZP) * s, 3))
print(json.dumps(out, indent=1))

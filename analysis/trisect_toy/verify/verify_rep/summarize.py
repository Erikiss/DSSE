import json
import sys

import numpy as np

d = json.load(open(sys.argv[1]))
cfgs = sorted({r["cfg"] for r in d}, key=lambda c: [r["cfg"] for r in d].index(c))
for cfg in cfgs:
    print(f"\n===== {cfg} =====")
    for task in ("s", "l"):
        rs = [r for r in d if r["cfg"] == cfg and r["task"] == task]
        print(f" task {task}: seeds {[r['seed'] for r in rs]} test R2 {[round(r['test_r2'], 6) for r in rs]}")
    nL = len(d[[r['cfg'] for r in d].index(cfg)]["layers"])
    # probes
    print(" probes (trained T / trained B / random-init net, mean over seeds):")
    for L in range(nL):
        line = f"  L{L + 1}:"
        for f in ("c", "cosC", "C", "cosC3", "cosC2"):
            g = lambda task, key: np.mean([r["layers"][L][key][f] for r in d if r["cfg"] == cfg and r["task"] == task])
            line += f" {f} {g('s', 'probes'):.4f}/{g('l', 'probes'):.4f}/{g('s', 'probes_randnet'):.4f}"
        print(line)
    print(" incremental probes R2 (T / B):")
    for L in range(nL):
        line = f"  L{L + 1}:"
        for k in ("cosC3|C,cosC,cosC2", "cosC2|C,cosC,cosC3", "s|l,a,b,c", "l|s,a,b,c"):
            g = lambda task: np.mean([r["layers"][L]["incr"][k] for r in d if r["cfg"] == cfg and r["task"] == task])
            line += f" {k} {g('s'):.3f}/{g('l'):.3f}"
        print(line)
    ip = [r["input_probes"] for r in d if r["cfg"] == cfg][0]
    print("  input baseline (lin/poly2):", {f: (round(v['lin'], 3), round(v['poly2'], 3)) for f, v in ip.items()})
    # split
    keys = [("avg_ans", "ans(avgJ)"), ("avg_orth", "orth(avgJ)"), ("avg_fid_ans", "fid"), ("avg_rms_orth", "rmsOrth"),
            ("avg_normshare", "norm%"), ("loc_ans", "ans(locJ)"), ("loc_orth", "orth(locJ)"), ("loc_rms_orth", "rmsOrthLoc"),
            ("mid_ans", "ans(midJ)"), ("linpred_fid", "linFid"),
            ("enc_full", "encFull"), ("enc_ans", "encAns"), ("enc_orth", "encOrth"), ("enc_cos_answer", "encCos"),
            ("shuf_ans", "SHUFans"), ("shuf_normshare", "SHUFnorm%"), ("rand_ans", "RANDans"), ("rand_orth", "RANDorth"),
            ("rand_normshare", "RANDnorm%"), ("rand_own_rms_rel", "RANDeff")]
    for reg in ("small", "medium", "indep", "large"):
        print(f" regime {reg}:")
        for task in ("s", "l"):
            rs = [r for r in d if r["cfg"] == cfg and r["task"] == task]
            for L in range(nL):
                g = lambda k: np.mean([r["layers"][L]["regimes"][reg][k] for r in rs])
                print(f"  {task} L{L + 1} |dC|={g('mean_absdC'):.2f} " + " ".join(f"{lab}={g(k):.3f}" for k, lab in keys))

# overall T vs B and min/max
print("\n===== ranges over all configs/seeds/layers (excluding last layer of relu64, which feeds the linear readout) =====")
for reg in ("small", "medium", "indep", "large"):
    for task in ("s", "l"):
        vals = {k: [] for k in ("avg_ans", "avg_orth", "loc_ans", "loc_orth", "avg_rms_orth", "avg_normshare", "rand_ans", "shuf_ans", "enc_full", "enc_ans")}
        for r in d:
            if r["task"] != task:
                continue
            for L, Ly in enumerate(r["layers"]):
                if r["cfg"] == "relu64" and L == 3:
                    continue
                for k in vals:
                    vals[k].append(Ly["regimes"][reg][k])
        print(f" {reg:6s} {task}: " + " ".join(f"{k}=[{min(v):.3f},{max(v):.3f}] mean {np.mean(v):.3f}" for k, v in vals.items()))

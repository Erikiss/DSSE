"""Collects the runs of the key-value-cache experiments (kv_grok.py, kv_untied.py, kv_probe.py) into one summary.
For every training arm and seed: first step with at least 99 % on the training problems checked, the step at which the
held-out accuracy crosses 50 % (linear interpolation between two evaluations), the first step with at least 99 % held-out,
whether the run was cut off before (censored), and the size of the disturbance of the '=' stream along the run.
For every continuation of a generalising network (ungrokking, drift control, untied and rare-example variants): the course
of the accuracy on the kept, the dropped and the held-out problems.
For every probed run: the route and order measures at selected steps.
Usage: python kv_report.py <runs_dir> <out_prefix> [further probed run dirs ...]      writes <out_prefix>.json and prints a text summary"""
import glob, json, os, sys

runs, outp = sys.argv[1], sys.argv[2]; extra = sys.argv[3:]; S = dict(arms={}, continuations={}, probes={})


def cross(L, key, thr, up=True):
    prev = None
    for r in L:
        v = r[key]
        if prev is not None and ((up and prev[key] < thr <= v) or (not up and prev[key] >= thr > v)): return prev["step"] + (thr - prev[key]) / (v - prev[key]) * (r["step"] - prev["step"])
        if prev is None and ((up and v >= thr) or (not up and v < thr)): return float(r["step"])
        prev = r
    return None


for f in sorted(glob.glob(f"{runs}/kv_*/log.json")):
    d = json.load(open(f)); L = d["log"]; name = os.path.basename(os.path.dirname(f))[3:]
    if not L: continue
    if d.get("init"):                                                     # continuation of a trained network
        kk = "kept_acc" if "kept_acc" in L[0] else "train_acc"; first = next((r["step"] for r in L if r["val_acc"] < 0.5 and r[kk] >= 0.99), None)
        S["continuations"][name] = dict(init=os.path.relpath(d["init"], runs), sub=d.get("sub"), mix=d.get("mix"), train=d.get("train"), steps=L[-1]["step"], first_step_held_below_half_with_kept_right=first, held_min=min(r["val_acc"] for r in L), held_last=L[-1]["val_acc"],
                                        kept_min=min(r[kk] for r in L), course=[(r["step"], round(r[kk], 3), round(r["val_acc"], 3), None if r.get("dropped_acc") is None else round(r["dropped_acc"], 3)) for r in L],
                                        wnorm=[(r["step"], round(r.get("wnorm", r.get("wnorm_writer", 0)), 2), round(r.get("wnorm_reader", 0), 2)) for r in L[::4]])
    else:
        mode, seed = d["mode"], d["seed"]; g99 = next((r["step"] for r in L if r["val_acc"] >= 0.99), None); S["arms"].setdefault(mode, {})[seed] = dict(train99=next((r["step"] for r in L if r["train_acc"] >= 0.99), None), held50=cross(L, "val_acc", 0.5), held99=g99, censored=g99 is None, last_step=L[-1]["step"],
                                                                                                                                        held_last=L[-1]["val_acc"], dmid=[(r["step"], [round(x, 3) for x in r["dmid"]]) for r in L[1::4] if r.get("dmid")], held=[(r["step"], round(r["val_acc"], 3)) for r in L])
for d in sorted(glob.glob(f"{runs}/kv_*")) + extra:
    for f in sorted(glob.glob(f"{d}/kvprobe*.json")):
        P = json.load(open(f)); S["probes"][os.path.basename(d) + os.path.basename(f)[7:-5]] = [dict(step=r["step"], acc_train=r["acc_train"], acc_held=r["acc_held"], A=r["route"]["train"]["A"], A_held=r["route"]["held"]["A"], L=r["route"]["train"]["L"], E=r["route"]["train"]["E"], S=r["S"], S_read=r["S_read"],
                                                                                                   logit_spread=r["logit_spread"], logit_spread_raw=r.get("logit_spread_raw"), acc_own_labels=r.get("acc_own_labels"), top5_w0=r["order"]["w0"]["top5"], top5_w0_shuffled=r["order"]["w0"]["top5_shuffled"], top5_w0_additive=r["order"]["w0"]["top5_additive"], top5_w1y=r["order"]["w1y"]["top5"], top5_w1x=r["order"]["w1x"]["top5"], top5_w1s=r["order"]["w1s"]["top5"], top5_q0=r["order"]["q0"]["top5"], noise_k1_train=r["tolerance"]["train"]["noise_k_1"], noise_v1_train=r["tolerance"]["train"]["noise_v_1"],
                                                                                                   dmid1=r["tolerance"]["train"]["dmid_1"], pairs=r["route"]["train"].get("pairs"), att_shift=r["route"]["train"].get("att_shift"), by_y=r["split"]["w1p3"]["by_y"], by_x=r["split"]["w1p3"]["by_x"], A0=r["route"]["train"]["A0"], A1=r["route"]["train"]["A1"], L_own=r["route"]["train"]["L_own"], S_eq=r["S_eq"],
                                                                                                   spread_w0=r["order"]["w0"]["spread"], pca8_w0=r["order"]["w0"]["pca8"], noise1_train=r["tolerance"]["train"]["noise_kv_1"], iso1_train=r["tolerance"]["train"]["iso_1"], noise2_train=r["tolerance"]["train"]["noise_kv_2"],
                                                                                                   iso2_train=r["tolerance"]["train"]["iso_2"], noise1_held=r["tolerance"]["held"]["noise_kv_1"], iso1_held=r["tolerance"]["held"]["iso_1"]) for r in P["rows"]]
json.dump(S, open(outp + ".json", "w"), indent=1); f3 = lambda v: "   -  " if v is None else f"{v:6.0f}"
print("== training arms: step with 50 % held-out (interpolated) / first step with 99 % held-out (c = not reached before the run ended) ==")
seeds = sorted({s for m in S["arms"].values() for s in m}); print("arm        " + "".join(f"   seed {s}        " for s in seeds))
for mode, m in S["arms"].items(): print(f"{mode:10s} " + "".join((f"{f3(m[s]['held50'])} /{f3(m[s]['held99'])}" + ("c" if m[s]["censored"] else " ") + f" ({m[s]['last_step']:5d}) ") if s in m else " " * 23 for s in seeds))
print("== continuations: first step with held-out < 50 % while the kept problems are >= 99 % right | least held-out | last held-out | steps ==")
for n, c in S["continuations"].items(): print(f"{n:28s} {str(c['first_step_held_below_half_with_kept_right']):>6s} | {c['held_min']:.3f} | {c['held_last']:.3f} | {c['steps']}")
print("== probes: step: accuracy train/held | A | L | S | logit spread | top5 w0 (shuffled) | answers kept under cache noise 1 / same-size random 1 (training problems) ==")
nz = lambda v: float("nan") if v is None else v
for n, rows in S["probes"].items():
    print(n)
    for r in rows:
        r = {k: nz(v) for k, v in r.items()}; print(f"   {r['step']:6d}: {r['acc_train']:.2f}/{r['acc_held']:.2f} | A {r['A']:.2f} | L {r['L']:.2f} | S {r['S']:.2f} | {r['logit_spread'][0]:.2f}/{r['logit_spread'][1]:.2f} | {r['top5_w0']:.2f} ({r['top5_w0_shuffled']:.2f}) | {r['noise1_train']:.2f} / {r['iso1_train']:.2f}")

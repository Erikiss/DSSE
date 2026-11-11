"""Prints the tables of README section 8 from the result files of the key-value-cache experiments:
kv_summary.json (kv_report.py), kv_path_fine_s*.json and kv_transplant_s*.json (kv_transplant.py).
A state counts as MEMORISING if at least 99 % of the training problems and less than 10 % of the held-out problems are right,
as GENERALISING if at least 99 % of the held-out problems are right.
Usage: python kv_tables.py <dir with the result files>      prints to stdout"""
import glob, json, os, sys

d = sys.argv[1]; S = json.load(open(f"{d}/kv_summary.json")); arms, cont, probes = S["arms"], S["continuations"], S["probes"]; rng = lambda v: "-" if not v else (f"{min(v):.2f}" if max(v) - min(v) < 0.005 else f"{min(v):.2f}-{max(v):.2f}")
f0 = lambda v: "   -  " if v is None else f"{v:6.0f}"
print("== 1. Training arms: step with 50 % held-out (interpolated between evaluations 500 steps apart) / first evaluation with at least 99 % ==")
seeds = sorted({int(s) for m in arms.values() for s in m}); base = arms.get("base", {})
print("arm        " + "".join(f"   seed {s}       " for s in seeds) + "  ratio of the 50 % step to the weight-decay-only run of the same seed")
for mode, m in arms.items():
    row = f"{mode:10s} "; rat = []
    for s in seeds:
        r = m.get(str(s)) or m.get(s)
        if r is None: row += " " * 17; continue
        row += f"{f0(r['held50'])} /{f0(r['held99'])}" + ("c " if r["censored"] else "  "); b = base.get(str(s)) or base.get(s)
        if b and r["held50"] and b["held50"]: rat.append(r["held50"] / b["held50"])
    print(row + "  " + " ".join(f"{x:.2f}" for x in rat))
print("   size of the disturbance of the '=' stream after the attention step of block 0 / block 1, relative to the spread of the stream, at steps 1000 / 3000 / 5000:")
for mode, m in arms.items():
    for s, r in m.items():
        if r["dmid"]: print(f"   {mode:9s} seed {s}: " + " | ".join(f"{st}: {x[0]:.2f}/{x[1]:.2f}" for st, x in r["dmid"] if st in (1000, 3000, 5000)))
print("\n== 2. Route, order and tolerance by state (ranges over the checkpoints of the kind) ==")
print("run                 kind         steps         A          logit spread b0   top5 w0     shuffled   additive   S          L          cache noise 1   same-size random 1")
for n, rows in probes.items():
    for kind, sel in (("memorising", lambda r: r["acc_train"] >= 0.99 and r["acc_held"] < 0.1), ("generalising", lambda r: r["acc_held"] >= 0.99), ("fits perm.", lambda r: "perm" in n and r["step"] >= 1000)):
        R = [r for r in rows if sel(r) and r["A"] is not None]
        if not R or (kind == "fits perm." and "perm" not in n) or (kind == "memorising" and "perm" in n): continue
        g = lambda k: rng([r[k] for r in R]); print(f"{n:19s} {kind:12s} {R[0]['step']:5d}-{R[-1]['step']:<6d} {g('A'):10s} {rng([r['logit_spread'][0] for r in R]):17s} {g('top5_w0'):11s} {g('top5_w0_shuffled'):10s} {g('top5_w0_additive'):10s} {g('S'):10s} {g('L'):10s} {g('noise1_train'):15s} {g('iso1_train')}")
print("   other tables and parts, same states: run kind | top5 of block-1 slot 1 / slot 2 / slot 3 (mean over x) / block-0 logits | keys of block 0 only (A0) / block 1 only (A1) | Gaussian noise 1 on keys only / values only | size of the stream change by cache noise 1 (block 0 / block 1)")
for n, rows in probes.items():
    for kind, sel in (("memorising", lambda r: r["acc_train"] >= 0.99 and r["acc_held"] < 0.1 and "perm" not in n), ("generalising", lambda r: r["acc_held"] >= 0.99), ("fits perm.", lambda r: "perm" in n and r["step"] >= 1000)):
        R = [r for r in rows if sel(r) and r["A"] is not None]
        if R: print(f"   {n:19s} {kind:12s} | " + " / ".join(rng([r[k] for r in R]) for k in ("top5_w1x", "top5_w1s", "top5_w1y", "top5_q0")) + " | " + " / ".join(rng([r[k] for r in R]) for k in ("A0", "A1")) + " | " + " / ".join(rng([r[k] for r in R]) for k in ("noise_k1_train", "noise_v1_train")) + " | " + " / ".join(rng([r["dmid1"][l] for r in R]) for l in (0, 1)))
print("   every probed step: step: train / held | A | L | S | top5 w0")
for n, rows in probes.items(): print(f"   {n}: " + "; ".join(f"{r['step']}: {r['acc_train']:.2f}/{r['acc_held']:.2f} | {('%.2f' % r['A']) if r['A'] is not None else 'n/a'} | {('%.2f' % r['L']) if r['L'] is not None else 'n/a'} | {r['S']:.2f} | {r['top5_w0']:.2f}" for r in rows))
print("\n== 3. Continuations of a generalising network (kept problems / held-out) ==")
for n, c in cont.items():
    co = c["course"]; dips = [(s, k, h) for s, k, h, _ in co if h < 0.9]; print(f"{n:24s} steps {c['steps']:5d} | first step with held-out < 0.5 and kept >= 0.99: {c['first_step_held_below_half_with_kept_right']} | least held-out {c['held_min']:.3f} | last {c['held_last']:.3f} | least kept {c['kept_min']:.3f} | evaluations with held-out < 0.9: {len(dips)} of {len(co)}"
                                                                                           + (f" | weight norm first/last {c['wnorm'][0][1]:.1f}/{c['wnorm'][-1][1]:.1f}" if c.get("wnorm") else ""))
for n in sorted(k for k in cont if "fine" in k):
    print(f"   {n}: " + " ".join(f"{s}:{k:.2f}/{h:.2f}" for s, k, h, _ in cont[n]["course"] if s % 100 == 0 or h < 0.9 and s % 50 == 0))
print("\n== 4. Path analysis: C = drop carried by the cache path, R = by the reader path (held-out problems) ==")
for f in sorted(glob.glob(f"{d}/kv_path_fine_s*.json")):
    P = json.load(open(f)); print(os.path.basename(f), "| G:", os.path.basename(os.path.dirname(P["G"])), os.path.basename(P["G"]))
    for key, lab in (("U_path", "5 % of the problems"), ("D_path", "all problems (drift)")):
        print(f"   {lab}: step: own accuracy | cut kv: C / R | cut states: C / R")
        print("      " + "; ".join(f"{r['step']}: {r['acc_XX']:.3f} | {r['kv']['C']:.3f} / {r['kv']['R']:.3f} | {r['states']['C']:.3f} / {r['states']['R']:.3f}" for r in P[key] if r["step"] in (100, 250, 400, 500, 600, 700, 800, 1000, 1500)))
    for cut in ("kv", "states"):
        fu = next((r["step"] for r in P["U_path"] if r[cut]["acc_reader_G_cache_X"] < 0.5), None); fr = next((r["step"] for r in P["U_path"] if r[cut]["acc_reader_X_cache_G"] < 0.5), None); print(f"   cut {cut}: first step with reader G + cache U below 0.5: {fu}; with reader U + cache G below 0.5: {fr}")
for f in sorted(glob.glob(f"{d}/kv_transplant_s*.json")):
    P = json.load(open(f)); b = P["M_mix_matched"]; dp = P["displacement"]; T = P["disorder"]; print(os.path.basename(f))
    print("   later steps: " + "; ".join(f"{r['step']}: own {r['acc_XX']:.3f} | kv {r['kv']['C']:.3f} / {r['kv']['R']:.3f} | states {r['states']['C']:.3f} / {r['states']['R']:.3f}" for r in P["U_path"]))
    print("   drift:       " + "; ".join(f"{r['step']}: own {r['acc_XX']:.3f} | kv {r['kv']['C']:.3f} / {r['kv']['R']:.3f} | states {r['states']['C']:.3f} / {r['states']['R']:.3f}" for r in P["D_path"]))
    print(f"   reference (all weights towards the memorising state, a = {b['a']:.3f}): own {b['acc_XX']:.3f} | kv C {b['kv']['C']:.3f} R {b['kv']['R']:.3f} | states C {b['states']['C']:.3f} R {b['states']['R']:.3f}")
    print(f"   displacement of the cache: share that is the same for all problems {dp['mean_share_of_total']:.3f}; size relative to the spread of G's cache {dp['size_vs_spread_of_G']:.2f}; share of the specific part in 1 / 2 / 8 / 32 axes " + " / ".join(f"{v:.2f}" for v in dp["specific_in_k_axes"].values()))
    for a in ("0.5", "1", "2"): print(f"   size {a}: " + ", ".join(f"{k.split()[0]} {T[f'{k} {a}']['right']:.3f}" for k in ("own", "other", "spread", "sub-8", "sub-32", "cov-8", "cov-32", "iso")) + (f", common shift alone {T['mean only']['right']:.3f} (size {T['mean only']['size']:.2f})" if a == "1" else ""))

"""Tables from the output of ufo_damp2.py. Means over texts with 95 % bootstrap intervals (texts resampled, 2000 draws);
per-position output measures are first averaged over the scored valid positions of a text.
Usage: python ufo_report.py <out_dir> [tag]     writes <out_dir>/ufo_report<tag>.json"""
import json, sys
import numpy as np

out = sys.argv[1]; TAG = sys.argv[2] if len(sys.argv) > 2 else ""; z = np.load(f"{out}/ufo_damp2{TAG}.npz"); M = json.load(open(f"{out}/ufo_damp2{TAG}_meta.json")); CO = M["conds"]; NM = [c["name"] for c in CO]; meta = M["meta"]; n = len(meta); L = z["p"].shape[2]
rng = np.random.default_rng(0); BS = rng.integers(0, n, (2000, n)); R = {}; has = lambda nm: nm in NM; ix = lambda nm: NM.index(nm)
mask = np.zeros((n, z["pp_kl"].shape[2]), bool)
for k, m in enumerate(meta): mask[k, 8:m["tokens"] - 1] = np.array(m["valid"][8:m["tokens"] - 1], bool)


def ci(x):                                                                 # x: one value per text -> mean, lo, hi
    x = np.asarray(x, float); ok = ~np.isnan(x); b = np.array([np.nanmean(x[i]) for i in BS]); return float(np.nanmean(x)), float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))


def fm(t, d=3): return f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"


def ratio(a, b):                                                           # ratio of means with bootstrap interval, paired by text
    a, b = np.asarray(a, float), np.asarray(b, float); r = np.array([np.nanmean(a[i]) / np.nanmean(b[i]) for i in BS]); return float(np.nanmean(a) / np.nanmean(b)), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def pt(key, c):                                                            # per-text mean of a per-position measure of row c
    a = z["pp_" + key][:, c] if isinstance(c, int) else c; return np.array([np.nanmean(a[k][mask[k]]) for k in range(n)])


KL = lambda nm: pt("kl", ix(nm)); src = np.array([m["source"] for m in meta]); st = {int(k): v for k, v in M["stats"].items()}
print(f"{n} texts ({(src == 'wiki').sum()} wiki, {(src == 'chat').sum()} chat), scored positions per text {mask.sum(1).min()}-{mask.sum(1).max()}, clean loss {np.mean([m['clean_loss'] for m in meta]):.2f}, null-row KL max {np.nanmax(z['pp_kl'][:, 1]):.1e}")
nb = np.nanmean(z["norm"], 0); print("median norm of the stream after block 0/2/8/14/21/26/27:", [round(float(nb[m]), 1) for m in (0, 2, 8, 14, 21, 26, 27)], "| dropped by the norm filter:", int(z["dropped"].sum()))
print("real states (pool): distance between two states / norm, and axes needed for 98 % of held-out variance:", {m: (round(st[m]["spread_over_norm"], 2), st[m]["k_irr"]) for m in (0, 2, 8, 14, 21)})
R["norm"] = [float(x) for x in nb]

print("\n== B. Local reaction of each block (state reset to clean + fresh vector after every block). q = share of the vector that the block removes.")
BANDS = [(1, 4), (5, 10), (11, 16), (17, 22), (23, 26), (27, 27)]; R["local"] = {}
print(f"{'row':14s} " + " ".join(f"{'blocks ' + str(a) + '-' + str(b):>22s}" for a, b in BANDS))
for c, cd in enumerate(CO):
    if cd["kind"] != "local": continue
    R["local"][cd["name"]] = {}
    for key in ("q_tot", "q_mlp", "q_att", "R", "sc"):
        vals = [ci(np.nanmean(z[key][:, c, a:b + 1], 1)) for a, b in BANDS]; R["local"][cd["name"]][key] = dict(bands=vals, per_block=[float(x) for x in np.nanmean(z[key][:, c], 0)])
        if key in ("q_tot", "q_att", "R"): print(f"{cd['name']:14s} {key:5s} " + " ".join(f"{fm(v):>22s}" for v in vals))
if has("loc_irr_0.3") and has("loc_cov_0.3"):
    print("contrast irr - cov (q_tot, f = 0.3):   " + " ".join(f"{fm(ci(np.nanmean(z['q_tot'][:, ix('loc_irr_0.3'), a:b + 1] - z['q_tot'][:, ix('loc_cov_0.3'), a:b + 1], 1))):>22s}" for a, b in BANDS))
    R["local"]["irr_minus_cov_0.3"] = [ci(np.nanmean(z["q_tot"][:, ix("loc_irr_0.3"), a:b + 1] - z["q_tot"][:, ix("loc_cov_0.3"), a:b + 1], 1)) for a, b in BANDS]
    for cl in ("cov", "irr"):
        if has(f"loc_{cl}_0.03") and has(f"loc_{cl}_1"):
            q = {f: np.nanmean(z["q_tot"][:, ix(f"loc_{cl}_{f}"), 1:17], 1) for f in ("0.03", "0.3", "1")}
            print(f"size dependence, {cl}, blocks 1-16: q(0.3)/q(0.03) = {fm(ratio(q['0.3'], q['0.03']))} (RMSNorm alone: {1 / np.sqrt(1.09):.3f}), q(1)/q(0.03) = {fm(ratio(q['1'], q['0.03']))} (RMSNorm alone: {1 / np.sqrt(2):.3f})")
            R["local"][f"size_{cl}"] = dict(r03=ratio(q["0.3"], q["0.03"]), r1=ratio(q["1"], q["0.03"]))

if has("loc_rand_0.3") and has("loc_irr_0.3"):                             # an isotropic vector is k/D inside and (1 - k/D) outside the data subspace of the block before; solve for 'isotropic inside'
    kd = np.array([st[m - 1]["k_irr"] / 2048 if m >= 1 else np.nan for m in range(L)]); qin = (z["q_tot"][:, ix("loc_rand_0.3")] - (1 - kd) * z["q_tot"][:, ix("loc_irr_0.3")]) / kd
    R["local"]["inside_isotropic_0.3"] = [ci(np.nanmean(qin[:, a:b + 1], 1)) for a, b in BANDS]; R["local"]["irr_minus_inside_isotropic_0.3"] = [ci(np.nanmean(z["q_tot"][:, ix("loc_irr_0.3"), a:b + 1] - qin[:, a:b + 1], 1)) for a, b in BANDS]
    print("derived: isotropic vector INSIDE the data subspace: " + " ".join(f"{fm(v):>22s}" for v in R["local"]["inside_isotropic_0.3"])); print("irr minus isotropic inside:                         " + " ".join(f"{fm(v):>22s}" for v in R["local"]["irr_minus_inside_isotropic_0.3"]))
for cl in ("cov", "irr"):
    if has(f"loc_{cl}_0.03") and has(f"loc_{cl}_1"):
        R["local"][f"size_by_band_{cl}"] = {key: [ratio(np.nanmean(z[key][:, ix(f"loc_{cl}_1"), a:b + 1], 1), np.nanmean(z[key][:, ix(f"loc_{cl}_0.03"), a:b + 1], 1)) for a, b in BANDS[:3]] for key in ("q_tot", "q_mlp", "q_att")}
        print(f"q(1)/q(0.03), {cl}, by band 1-4 / 5-10 / 11-16: total " + " / ".join(f"{v[0]:.3f}" for v in R["local"][f"size_by_band_{cl}"]["q_tot"]) + " | MLP " + " / ".join(f"{v[0]:.3f}" for v in R["local"][f"size_by_band_{cl}"]["q_mlp"]) + " | attention " + " / ".join(f"{v[0]:.2f}" for v in R["local"][f"size_by_band_{cl}"]["q_att"]) + "  (RMSNorm alone 0.707)")

print("\n== C. Free runs, f = 0.3: what is left of the vector (p), how large is the deviation (amp = |dev|/|delta|), output")
print(f"{'row':16s} {'inside':>6s} {'p+1':>5s} {'p+4':>5s} {'p@27':>6s} {'amp+1':>6s} {'amp+4':>6s} {'amp@26':>7s} {'amp@27':>7s} {'growth':>7s} {'ins+2':>6s} {'ins@26':>6s} {'KL':>24s} {'same top':>8s} {'KL late-matched':>24s} {'KL / late':>20s}")
R["free"] = {}
for c, cd in enumerate(CO):
    if cd["kind"] != "free" or cd["var"] != "add" or cd["pos"] != "all" or cd["mode"]: continue
    l = cd["l"]; g = lambda key, m: float(np.nanmean(z[key][:, c, min(m, L - 1)])); kl = pt("kl", c); late = pt("kl_late", c); e = dict(inside=float(np.nanmean(z["inside"][:, c])), radial=float(np.nanmean(z["radial"][:, c])), p1=g("p", l + 1), p4=g("p", l + 4), p27=g("p", 27), amp1=g("amp", l + 1), amp4=g("amp", l + 4), amp26=g("amp", 26), amp27=g("amp", 27),
                                                                                                                         growth=float(np.nanmean(z["rho"][:, c, 27] / z["rho"][:, c, l])), ins2=g("ins", l + 2), ins26=g("ins", 26), s1=g("s", l + 1), kl=ci(kl), top=float(np.nanmean(pt("top", c))), kl_wiki=float(np.nanmean(kl[src == "wiki"])), kl_chat=float(np.nanmean(kl[src == "chat"])),
                                                                                                                         dH=ci(pt("H", c) - pt("H", 0)), dHcs=ci(pt("Hcs", c) - pt("H", 0)), dLoss=ci(pt("ce", c) - pt("ce", 0)), gap=ci((pt("ce", c) - pt("ce", 0)) - (pt("H", c) - pt("H", 0))), p_per_block=[float(x) for x in np.nanmean(z["p"][:, c], 0)], amp_per_block=[float(x) for x in np.nanmean(z["amp"][:, c], 0)])
    if not np.all(np.isnan(late)):
        e["kl_late"] = ci(late); e["kl_over_late"] = ratio(kl, late); e["kl_frozen"] = ci(pt("kl_frozen", c))
        for so in ("wiki", "chat"):
            if (src == so).sum() > 1: e[f"kl_over_late_{so}"] = float(np.nanmean(kl[src == so]) / np.nanmean(late[src == so]))
    R["free"][cd["name"]] = e
    if abs(cd["f"] - 0.3) < 1e-9: print(f"{cd['name']:16s} {e['inside']:6.2f} {e['p1']:5.2f} {e['p4']:5.2f} {e['p27']:6.2f} {e['amp1']:6.2f} {e['amp4']:6.2f} {e['amp26']:7.1f} {e['amp27']:7.1f} {e['growth']:7.2f} {e['ins2']:6.2f} {e['ins26']:6.2f} {fm(e['kl']):>24s} {e['top']:8.2f} {fm(e.get('kl_late', (np.nan,) * 3)):>24s} {fm(e.get('kl_over_late', (np.nan,) * 3), 2):>20s}")
print("KL / late-matched by source (wiki, chat), f = 0.3: " + ", ".join(f"{k_} {v_.get('kl_over_late_wiki', float('nan')):.2f} / {v_.get('kl_over_late_chat', float('nan')):.2f}" for k_, v_ in R["free"].items() if k_.endswith("_0.3") and "kl_over_late" in v_))
print("growth = relative deviation at the output / relative deviation at the injection (1 = the deviation grows exactly as fast as the stream)")

print("\n== D. Dose: KL and orthogonal reaction s one block later, for sizes 0.03 / 0.1 / 0.3 / 1.0; slopes in log-log (linear response: s 1, KL 2)"); R["dose"] = {}
for l in (2, 14):
    for cl in ("cov", "irr"):
        fs = [f for f in ("0.03", "0.1", "0.3", "1") if has(f"L{l}_{cl}_{f}")]
        if len(fs) < 3: continue
        kls = [KL(f"L{l}_{cl}_{f}") for f in fs]; ss = [z["s"][:, ix(f"L{l}_{cl}_{f}"), l + 1] * float(f) for f in fs]; ps = [float(np.nanmean(z["p"][:, ix(f"L{l}_{cl}_{f}"), l + 1])) for f in fs]; sl = []
        for a in range(len(fs) - 1):
            d = np.log(float(fs[a + 1]) / float(fs[a])); bk = np.array([np.log(np.nanmean(kls[a + 1][i]) / np.nanmean(kls[a][i])) / d for i in BS]); bs_ = np.array([np.log(np.nanmean(ss[a + 1][i]) / np.nanmean(ss[a][i])) / d for i in BS])
            sl.append(dict(sizes=(fs[a], fs[a + 1]), kl=(float(np.log(np.nanmean(kls[a + 1]) / np.nanmean(kls[a])) / d), float(np.percentile(bk, 2.5)), float(np.percentile(bk, 97.5))), s=(float(np.log(np.nanmean(ss[a + 1]) / np.nanmean(ss[a])) / d), float(np.percentile(bs_, 2.5)), float(np.percentile(bs_, 97.5)))))
        R["dose"][f"L{l}_{cl}"] = dict(sizes=fs, kl=[ci(k) for k in kls], p1=ps, slopes=sl)
        print(f"L{l} {cl}: KL " + " / ".join(f"{np.nanmean(k):.4f}" for k in kls) + " | p+1 " + " / ".join(f"{p:.2f}" for p in ps) + " | slope KL " + ", ".join(fm(s["kl"], 2) for s in sl) + " | slope s " + ", ".join(fm(s["s"], 2) for s in sl))
    for nm in (f"L{l}_normonly_1", f"L{l}_cov_1_np", f"L{l}_irr_1_np"):
        if has(nm): R["dose"][nm] = ci(KL(nm)); print(f"   control {nm}: KL {fm(ci(KL(nm)))}" + (f"  (same direction with the norm free: {fm(ci(KL(nm[:-3])))})" if nm.endswith("_np") else ""))

print("\n== E. Does the erasure protect the output? KL with the erasure removed (A) and with only the erasure kept (B), f = 0.3"); R["writes"] = {}
for l in (2, 14):
    for cl in ("cov", "irr"):
        b = f"L{l}_{cl}_0.3"
        if not has(b + "_A"): continue
        k0, kA, kB = KL(b), KL(b + "_A"), KL(b + "_B"); R["writes"][b] = dict(normal=ci(k0), A=ci(kA), B=ci(kB), A_over_normal=ratio(kA, k0), frozen=ci(pt("kl_frozen", ix(b))), p27_normal=float(np.nanmean(z["p"][:, ix(b), 27])), amp27=dict(normal=float(np.nanmean(z["amp"][:, ix(b), 27])), A=float(np.nanmean(z["amp"][:, ix(b + "_A"), 27]))))
        print(f"{b:12s} normal {fm(ci(k0), 4)} | A {fm(ci(kA), 4)} | A / normal {fm(ratio(kA, k0), 2)} | B {fm(ci(kB), 5)} | vector riding along untouched {ci(pt('kl_frozen', ix(b)))[0]:.5f} | amp@27 normal {R['writes'][b]['amp27']['normal']:.1f}, A {R['writes'][b]['amp27']['A']:.1f}")

print("\n== F. One position only / the same vector at all positions (f = 0.3)"); R["positions"] = {}
for l in (2, 14):
    for cl in ("cov", "irr"):
        b = f"L{l}_{cl}_0.3"
        if not has(b + "_single"): continue
        c = ix(b + "_single"); kl = z["pp_kl"][:, c]; at, later, summ = [], [], []
        for k, m in enumerate(meta):
            t0 = m["t0"]; at.append(kl[k, t0]); mk = mask[k].copy(); mk[:t0 + 1] = False; later.append(np.nanmean(kl[k][mk])); summ.append(np.nansum(kl[k][mk]))
        e = dict(kl_all=ci(KL(b)), kl_same=ci(KL(b + "_same")), kl_single_at=ci(at), kl_single_later=ci(later), p1=[float(np.nanmean(z["p"][:, ix(x), l + 1])) for x in (b, b + "_single", b + "_same")], p4=[float(np.nanmean(z["p"][:, ix(x), l + 4])) for x in (b, b + "_single", b + "_same")],
                 amp27=[float(np.nanmean(z["amp"][:, ix(x), 27])) for x in (b, b + "_single", b + "_same")], imp1=float(np.nanmean(z["imp"][:, c, l + 1])), imp27=float(np.nanmean(z["imp"][:, c, 27])))
        e.update(kl_single_at_median=float(np.nanmedian(at)), kl_single_later_sum=ci(summ), later_sum_over_at=ratio(summ, at)); R["positions"][b] = e
        print(f"             single position: median KL at the position {e['kl_single_at_median']:.3f}; summed over all later positions {fm(e['kl_single_later_sum'])} = {fm(e['later_sum_over_at'], 2)} times the KL at the position"); print(f"{b:12s} KL all positions {fm(e['kl_all'], 4)} | same vector {fm(e['kl_same'], 4)} | single: at the position {fm(e['kl_single_at'], 4)}, later positions {fm(e['kl_single_later'], 5)} | p+1 all/single/same {e['p1'][0]:.2f}/{e['p1'][1]:.2f}/{e['p1'][2]:.2f} | p+4 {e['p4'][0]:.2f}/{e['p4'][1]:.2f}/{e['p4'][2]:.2f} | amp@27 {e['amp27'][0]:.1f}/{e['amp27'][1]:.1f}/{e['amp27'][2]:.1f} | deviation at later positions / |delta|: +1 {e['imp1']:.3f}, @27 {e['imp27']:.2f}")

print("\n== G. Confidence: change of entropy and of loss (gap = dLoss - dH; a well-behaved model would raise its entropy as much as its loss)"); R["confidence"] = {}
for nm in [x for x in ("L0_cov_1", "L2_cov_1", "L8_cov_1", "L14_cov_1", "L0_irr_1", "L2_irr_1", "L8_irr_1", "L14_irr_1", "L2_cov_0.3", "L14_cov_0.3", "abl_mlp2", "abl_mlp8", "abl_mlp14") if has(x)]:
    c = ix(nm); dH = pt("H", c) - pt("H", 0); dL = pt("ce", c) - pt("ce", 0); e = dict(kl=ci(pt("kl", c)), dLoss=ci(dL), dH=ci(dH), gap=ci(dL - dH), dH_clean_scale=ci(pt("Hcs", c) - pt("H", 0)), norm_ratio=float(np.nanmean(pt("r", c)))); R["confidence"][nm] = e
    print(f"{nm:14s} KL {fm(e['kl'])} | dLoss {fm(e['dLoss'])} | dH {fm(e['dH'])} | gap {fm(e['gap'])} | dH with the clean scale of the final norm {fm(e['dH_clean_scale'])} | final norm ratio {e['norm_ratio']:.3f}")
lf = [ci(np.array([np.nanmean(z["pp_kl_latef"][k, j][mask[k]]) for k in range(n)])) for j in range(len(M["latef"]))]; lh = [ci(np.array([np.nanmean((z["pp_H_latef"][k, j] - z["pp_H"][k, 0])[mask[k]]) for k in range(n)])) for j in range(len(M["latef"]))]
print("isotropic noise after the LAST block (the network cannot react): size -> KL, dH: " + ", ".join(f"{f:g}: {a[0]:.3f}, {b[0]:+.3f}" for f, a, b in zip(M["latef"], lf, lh))); R["late_iso"] = dict(sizes=M["latef"], kl=lf, dH=lh)
json.dump(R, open(f"{out}/ufo_report{TAG}.json", "w"), indent=1)

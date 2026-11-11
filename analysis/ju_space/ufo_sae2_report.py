"""Tables from the output of ufo_sae2.py. Means over texts with 95 % bootstrap intervals (texts resampled, 2000 draws);
per-position output measures are first averaged over the scored valid positions of a text.
Usage: python ufo_sae2_report.py <out_dir> [tag]      writes <out_dir>/ufo_sae2_report<tag>.json"""
import json, sys
import numpy as np

out = sys.argv[1]; TAG = sys.argv[2] if len(sys.argv) > 2 else ""; z = np.load(f"{out}/ufo_sae2{TAG}.npz"); M = json.load(open(f"{out}/ufo_sae2{TAG}_meta.json")); NM = [r[0] for r in M["rows"]]; meta = M["meta"]; n = len(meta); L = z["p"].shape[2]
rng = np.random.default_rng(0); BS = rng.integers(0, n, (2000, n)); ix = lambda nm: NM.index(nm); R = dict(sites={}); mask = np.zeros((n, z["pp_kl"].shape[2]), bool); src = np.array([m["source"] for m in meta])
for k, m in enumerate(meta): mask[k, 8:m["tokens"] - 1] = np.array(m["valid"][8:m["tokens"] - 1], bool)
pt = lambda key, c: np.array([np.nanmean(z["pp_" + key][k, c][mask[k]]) for k in range(n)]); KL = lambda nm: pt("kl", ix(nm))


def ci(x):
    x = np.asarray(x, float); b = np.array([np.nanmean(x[i]) for i in BS]); return float(np.nanmean(x)), float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))


def ratio(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); r = np.array([np.nanmean(a[i]) / np.nanmean(b[i]) for i in BS]); return float(np.nanmean(a) / np.nanmean(b)), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


fm = lambda t, d=3: f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"
print(f"{n} texts ({(src == 'wiki').sum()} wiki, {(src == 'chat').sum()} chat)")
print("\n== A. The SAE at its block (pool sample / the texts of this run)")
for s, b in M["sites"]:
    ps = M["pool"][s]; si = [m["site"][s] for m in meta]; g = lambda key: float(np.mean([x[key] for x in si if key in x])); e = dict(block=b, fve_pool=ps["fve_pool"], fve_texts=g("fve"), fve_wiki=float(np.mean([x["fve"] for x, so in zip(si, src) if so == "wiki"])), fve_chat=float(np.mean([x["fve"] for x, so in zip(si, src) if so == "chat"])),
                                                                                                                        rel_err=g("rel_err"), err_inside=g("err_inside"), err_radial=g("err_radial"), norm_ratio=ps["norm_ratio"], k_pca=ps["k_pca_same_fve"], fve_pca_texts=g("fve_pca"), compress=dict(state=ps["compress_state"], recon=ps["compress_recon"], err=ps["compress_err"]))
    R["sites"][s] = e; print(f"{s:5s} block {b:2d}: explained variance {e['fve_pool']:.3f} (pool) / {e['fve_texts']:.3f} (texts; wiki {e['fve_wiki']:.3f}, chat {e['fve_chat']:.3f}) | |e|/|h| {e['rel_err']:.3f} | e inside the data subspace {e['err_inside']:.2f}, radial {e['err_radial']:.3f} | PCA with the same share: {e['k_pca']} axes ({e['fve_pca_texts']:.3f} on the texts)")
    print("        held-out share of z-scored variance in 2/8/32/128 axes: state " + "/".join(f"{ps['compress_state'][k]:.3f}" for k in ("2", "8", "32", "128")) + " | SAE reconstruction " + "/".join(f"{ps['compress_recon'][k]:.3f}" for k in ("2", "8", "32", "128")) + " | SAE error " + "/".join(f"{ps['compress_err'][k]:.3f}" for k in ("2", "8", "32", "128")))

print("\n== B. Output KL of the change at one block (same size |e| for splice, plus, foreign, cov_e, irr_e)")
print(f"{'site':5s} {'splice = SAE(h)':>22s} {'half':>22s} {'plus':>22s} {'foreign error':>22s} {'cov noise |e|':>22s} {'irr noise |e|':>22s} {'PCA same share':>22s} | {'splice/cov':>18s} {'splice/irr':>18s} {'half/splice':>18s}")
for s, b in M["sites"]:
    k = {v: KL(f"{s}_{v}") for v in ("splice", "half", "plus", "foreign", "cov_e", "irr_e", "pca")}; e = R["sites"][s]; e["kl"] = {v: ci(x) for v, x in k.items()}; e["splice_over_cov"] = ratio(k["splice"], k["cov_e"]); e["splice_over_irr"] = ratio(k["splice"], k["irr_e"]); e["half_over_splice"] = ratio(k["half"], k["splice"])
    e["plus_over_splice"] = ratio(k["plus"], k["splice"]); e["foreign_over_splice"] = ratio(k["foreign"], k["splice"]); e["pca_over_splice"] = ratio(k["pca"], k["splice"]); e["kl_splice_wiki"] = float(k["splice"][src == "wiki"].mean()); e["kl_splice_chat"] = float(k["splice"][src == "chat"].mean())
    e["top_splice"] = float(pt("top", ix(f"{s}_splice")).mean()); e["dloss_splice"] = ci(pt("ce", ix(f"{s}_splice")) - pt("ce", 0))
    print(f"{s:5s} " + " ".join(f"{fm(e['kl'][v]):>22s}" for v in ("splice", "half", "plus", "foreign", "cov_e", "irr_e", "pca")) + f" | {fm(e['splice_over_cov'], 2):>18s} {fm(e['splice_over_irr'], 2):>18s} {fm(e['half_over_splice'], 2):>18s}")
print("   splice: loss change " + ", ".join(f"{s} {R['sites'][s]['dloss_splice'][0]:+.3f}" for s, _ in M["sites"]) + " | same top token " + ", ".join(f"{s} {R['sites'][s]['top_splice']:.2f}" for s, _ in M["sites"]) + " | KL wiki/chat " + ", ".join(f"{s} {R['sites'][s]['kl_splice_wiki']:.3f}/{R['sites'][s]['kl_splice_chat']:.3f}" for s, _ in M["sites"]))

print("\n== C. What the later blocks do with the changed part (p = share of the change still there, amp = size of the deviation / size of the change)")
for s, b in M["sites"]:
    e = R["sites"][s]; e["down"] = {}
    for v in ("splice", "foreign", "cov_e", "irr_e"):
        c = ix(f"{s}_{v}"); g = lambda key, m: float(np.nanmean(z[key][:, c, min(m, L - 1)])); e["down"][v] = dict(p1=g("p", b + 1), p4=g("p", b + 4), p27=g("p", 27), amp4=g("amp", b + 4), amp27=g("amp", 27))
    print(f"{s:5s} " + " | ".join(f"{v}: p+1 {d['p1']:.2f}, p+4 {d['p4']:.2f}, p@27 {d['p27']:.2f}, amp@27 {d['amp27']:.1f}" for v, d in e["down"].items()))

print("\n== D. Noise of size 0.3 |h| at the SAE's block: how much passes, and what the SAE in the loop does to the output")
for s, b in M["sites"]:
    if f"{s}_cov_noise" not in NM: continue
    e = R["sites"][s]; e["noise"] = {}; si = [m["site"][s] for m in meta]; g = lambda key: float(np.mean([x[key] for x in si]))
    for c in ("cov", "irr"):
        k = {v: KL(f"{s}_{c}_{v}") for v in ("noise", "sae_oracle", "sae_rt", "pca_oracle")}; d = dict(kl={v: ci(x) for v, x in k.items()}, pass_sae=g(f"pass_sae_{c}"), pass_pca=g(f"pass_pca_{c}"), pass_cos=g(f"pass_sae_cos_{c}"), units_kept=g(f"units_kept_{c}"), err_noisy=g(f"sae_err_rel_noisy_{c}"),
                                                                                                                       oracle_over_noise=ratio(k["sae_oracle"], k["noise"]), pca_oracle_over_noise=ratio(k["pca_oracle"], k["noise"]), rt_over_splice=ratio(k["sae_rt"], KL(f"{s}_splice")), rt_over_noise=ratio(k["sae_rt"], k["noise"])); e["noise"][c] = d
        print(f"{s:5s} {c}: passes the SAE {d['pass_sae']:.2f} (cos with the noise {d['pass_cos']:.2f}), passes PCA {d['pass_pca']:.2f}, active units kept {d['units_kept']:.2f}, |h+d-SAE(h+d)|/|h| {d['err_noisy']:.3f} | KL noise alone {fm(d['kl']['noise'])} | SAE without its own error {fm(d['kl']['sae_oracle'])} ({fm(d['oracle_over_noise'], 2)} of noise alone) | PCA likewise {fm(d['kl']['pca_oracle'])} ({fm(d['pca_oracle_over_noise'], 2)}) | SAE in real time {fm(d['kl']['sae_rt'])} ({fm(d['rt_over_splice'], 2)} x splice without noise)")

print("\n== E. Clean pass: sum over all later blocks of <write, v> / |v|^2 (attention + MLP; negative = written against v)"); B = z["bcoef"]; VEC = M["vec"]
for si_, (s, b) in enumerate(M["sites"]):
    tot = np.nansum(B[:, si_], axis=2); e = R["sites"][s]; e["clean_writes"] = {v: dict(att=ci(tot[:, vi, 0]), mlp=ci(tot[:, vi, 1]), both=ci(tot[:, vi].sum(1)), next4=ci(np.nansum(B[:, si_, vi, b + 1:b + 5], axis=(1, 2)))) for vi, v in enumerate(VEC)}
    print(f"{s:5s} " + " | ".join(f"{v}: {fm(e['clean_writes'][v]['both'], 2)} (next 4 blocks {e['clean_writes'][v]['next4'][0]:+.2f})" for v in VEC))
json.dump(R, open(f"{out}/ufo_sae2_report{TAG}.json", "w"), indent=1)

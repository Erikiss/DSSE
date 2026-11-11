"""Collects the training runs of README section 9 into one file, with the definitions of act_defs.py (events, levels, t50).
Usage: python act_report.py <kv_runs_dir> <act_runs_dir> <out.json>
  kv_runs_dir    runs of section 8 (ReLU base runs and their continuations kv_base_s<n>_ungrok5fine, _driftfine, _ungrok5, _drift; SwiGLU base runs)
  act_runs_dir   runs of section 9 (kv_bilinear_s<n>, kv_<arch>_s<n>_U / _D, kv_base_s<n>_fzU / _fzD, _scr_<setting>_U / _D, _fb_lr3_U / _D, _b26U, ...)
Output: per starting network G its description; per continuation the summary of act_defs.summary(); for the ReLU networks of
the standard setting the two stored runs of section 8 are combined (fine run to step 1500 with evaluations every 50 steps,
long run with evaluations every 250 steps for the late level). Also the distance of the weights from G at step 4000
(root of the summed squared differences over the root of the summed squares of G, as in kv_transplant.py)."""
import glob, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from act_defs import summary, load_log, events, level, t50
from kv_common import fwd, load, problems

KV, ACT, out = sys.argv[1:4]; tr, va = problems(); res = dict(G={}, runs={}, base={})
last = lambda d: max(int(os.path.basename(f)[5:-3]) for f in glob.glob(d + "/ckpt_*.pt"))


def h50(L):
    for a, b in zip(L, L[1:]):
        if a["val_acc"] < 0.5 <= b["val_acc"]: return a["step"] + (0.5 - a["val_acc"]) / (b["val_acc"] - a["val_acc"]) * (b["step"] - a["step"])


def dist(a, b):
    sa, sb = torch.load(a, weights_only=True)["net"], torch.load(b, weights_only=True)["net"]; return float(sum(((sa[k] - sb[k]) ** 2).sum() for k in sb if sb[k].is_floating_point()) ** .5 / sum((sb[k] ** 2).sum() for k in sb if sb[k].is_floating_point()) ** .5)


for arch, root in (("base", KV), ("swiglu", KV), ("swiglu", ACT), ("bilinear", ACT), ("linear", ACT)):
    for d in sorted(glob.glob(f"{root}/kv_{arch}_s?")):
        J, L = load_log(d + "/log.json"); name = os.path.basename(d)[3:]; g99 = next((r["step"] for r in L if r["val_acc"] >= 0.99), None); res["base"][name] = dict(held50=h50(L), held99=g99, last_step=L[-1]["step"], train99=next((r["step"] for r in L if r["train_acc"] >= 0.99), None), max_held=max(r["val_acc"] for r in L))
        if g99 is None: continue
        ck = f"{d}/ckpt_{last(d)}.pt"; net = load(ck)
        with torch.no_grad(): ce = lambda s: float(torch.nn.functional.cross_entropy(fwd(net, s[:, :5])[0][:, 4], s[:, 5]))
        with torch.no_grad(): res["G"][name] = dict(checkpoint=ck.replace(root, "<runs>"), step=last(d), steps_since_first_99=last(d) - g99, weight_norm=float(sum(float((p ** 2).sum()) for p in net.parameters()) ** .5), held_out_loss=ce(va), training_loss=ce(tr))
for root in (KV, ACT):
    for d in sorted(glob.glob(f"{root}/kv_*_s?_*")):
        if not os.path.exists(d + "/log.json"): continue
        name = os.path.basename(d)[3:]; J, L = load_log(d + "/log.json")
        if not J.get("init") or len(L) < 3 or "train_acc" not in L[0]: continue  # the latter: transplant runs of section 8 (kv_transplant.py)
        r = summary(d + "/log.json"); r["init"] = os.path.basename(os.path.dirname(J["init"])) + "/" + os.path.basename(J["init"]); r.pop("config"); r.update(lr=J.get("lr", 1e-3), eps=J.get("eps", 1e-8), amsgrad=bool(J.get("amsgrad")), batch=J.get("batch", 512), frozen=bool(J.get("frozen")), sub=J.get("sub"))
        f = f"{d}/ckpt_4000.pt"
        if os.path.exists(f): r["weight_distance_from_G_at_4000"] = dist(f, J["init"])
        res["runs"][name] = r
# ReLU, standard setting: fine and long run of section 8 combined
for s in (0, 1, 2):
    for arm, fine, long_ in (("U", "ungrok5fine", "ungrok5"), ("D", "driftfine", "drift")):
        pf, pl = f"{KV}/kv_base_s{s}_{fine}/log.json", f"{KV}/kv_base_s{s}_{long_}/log.json"
        if not (os.path.exists(pf) and os.path.exists(pl)): continue
        _, Lf = load_log(pf); _, Ll = load_log(pl); ev = events(Lf); allv = [r["val_acc"] for r in Lf] + [r["val_acc"] for r in Ll if r["step"] > 1500 and r["step"] <= 4000]
        res["runs"][f"base_s{s}_{arm}_standard_combined"] = dict(early=level(Lf, 1000, 1500), late=level(Ll, 3500, 4000), late_from_evaluations_every=250, t50=t50(Lf), events_first_1500=len(ev), events=ev, share_held_099=sum(v >= 0.99 for v in allv) / len(allv), readable_as_drift=bool(sum(v >= 0.99 for v in allv) / len(allv) >= 0.9 and level(Ll, 3500, 4000) >= 0.99),
                                                             share_train_099=sum(r["train_acc"] >= 0.99 for r in Lf) / len(Lf), min_held=min(allv), held_at={str(k): next((r["val_acc"] for r in (Lf if k <= 1500 else Ll) if r["step"] == k), None) for k in (250, 500, 750, 1000, 1500, 2000, 3000, 4000)})
json.dump(res, open(out, "w"), indent=1)
for k, v in res["G"].items(): print("G", k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items() if a != "checkpoint"})
for k, v in res["runs"].items(): print(k, "| early", v.get("early"), "late", v.get("late"), "end", v.get("end"), "| t50", v.get("t50"), "| events<=1500:", v.get("events_first_1500"), v.get("events")[:6], "| held>=0.99:", round(v["share_held_099"], 2), "| dist", round(v["weight_distance_from_G_at_4000"], 3) if "weight_distance_from_G_at_4000" in v else None)

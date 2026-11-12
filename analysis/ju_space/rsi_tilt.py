"""Tilt covariate of the directions of README section 10: the share of a direction's energy in the 256 principal axes with the most
variance of real states (ufo_pool.pt, README 7.3). The plan review asked to record it for every direction ('Neigung' read as a
geometric tilt); it needs no forward passes.
Directions: per persona set and block, the mean over the 100 build statements of [answer-token state with the matching answer]
minus [with the other answer], rebuilt from the states that rsi_followup.py stored (<rsi_followup.json>.states.pt).
References: the 20 random directions of the controls (10 'rand', 10 'cov', the same seeds as rsi_direction.py), the expected
share of an isotropic direction (256 / 2048) and of a covariance-shaped one (variance of the top 256 axes / total variance), and
the Yes-minus-No difference of the answer-token state averaged over the build statements of each set.
Also, for README 10.4: the cosine of each shuffled-label direction of the RSI controls (the same flips as rsi_direction.py, seeds
100-109) with the mean Yes-minus-No difference, and the mean effect and damage of the 'rand' and 'cov' control directions.
Usage: python rsi_tilt.py <root> <rsi_followup.json.states.pt> <rsi_direction.json> <out.json>"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ufo_common import Pool

root, stp, dj, out = sys.argv[1:5]; K = 256; BL = [14, 18]
ST = torch.load(stp, map_location="cpu", weights_only=True); pool = Pool(f"{root}/ufo_out/ufo_pool.pt", blocks=BL); old = json.load(open(dj))
share = lambda U, v: float(((v.double() @ U[:, :K].double()) ** 2).sum() / (v.double() ** 2).sum())
cos = lambda a, b: float(a @ b / (a.norm() * b.norm()))
res = dict(k=K, blocks={})
for l in BL:
    U = pool.b[l]["U"]; sd = pool.b[l]["sd"].double(); R = dict(sets={}, yes_minus_no={}, random=[])
    for s, z in ST.items():
        Y, N, lab = z["yes"][str(l)], z["no"][str(l)], z["label"].bool()[:, None]
        d = torch.where(lab, Y - N, N - Y).mean(0); R["sets"][s] = share(U, d); R["yes_minus_no"][s] = share(U, (Y - N).mean(0))
        if s == "rsi":      # shuffled-label directions as in rsi_direction.py: per statement +-(matching minus other), sign flipped at random
            dd = torch.where(lab, Y - N, N - Y); ym = (Y - N).mean(0); R["shuffled_cos_with_yes_minus_no"] = []
            for i in range(10):
                fl = torch.from_numpy(np.random.default_rng(100 + i).random(len(dd)) < 0.5)[:, None]; v = torch.where(fl, -dd, dd).mean(0)
                R["shuffled_cos_with_yes_minus_no"].append(cos(v, ym))
    for kind in ("rand", "cov"):
        for i in range(10):
            v = pool.direction(kind, l, 1, torch.Generator().manual_seed(i))[0]; R["random"].append(dict(kind=kind, seed=i, share=share(U, v)))
    C = old[f"controls_block{l}"]["random"]
    R["random_controls_by_kind"] = {kd: dict(mean_delta=float(np.mean([x["delta"] for x in C if x["kind"] == kd])), mean_abs_delta=float(np.mean([abs(x["delta"]) for x in C if x["kind"] == kd])),
                                         mean_damage=float(np.mean([x["damage"] for x in C if x["kind"] == kd]))) for kd in ("rand", "cov")}
    R["expected_isotropic"] = K / U.shape[0]; R["expected_cov"] = float((sd[:K] ** 2).sum() / (sd ** 2).sum())
    if l == 14:      # the rebuilt directions must reproduce the cosines of rsi_direction.json
        dirs = {s: torch.where(z["label"].bool()[:, None], z["yes"]["14"] - z["no"]["14"], z["no"]["14"] - z["yes"]["14"]).mean(0) for s, z in ST.items()}
        R["check_cos_with_rsi_direction_json"] = {s: [cos(dirs["rsi"], dirs[s]), old["cosines_block14"]["rsi"][s]] for s in dirs}
    res["blocks"][str(l)] = R
# side by side: inclination and own-direction effect at f = +0.3 (block 14), from rsi_direction.json; described only, K = 7
res["inclination_and_effect_block14"] = {s: dict(inclination=v["rows"][1]["logodds"], effect_at_0_3=v["rows"][3]["logodds"] - v["rows"][1]["logodds"], share=res["blocks"]["14"]["sets"][s]) for s, v in old["sweeps"].items() if s in ST}
json.dump(res, open(out, "w"), indent=1)
for l in BL:
    R = res["blocks"][str(l)]; rr = [x["share"] for x in R["random"] if x["kind"] == "rand"]; rc = [x["share"] for x in R["random"] if x["kind"] == "cov"]
    print(f"block {l}: sets " + " ".join(f"{s[:12]} {v:.3f}" for s, v in R["sets"].items()) + f" | yes-no {min(R['yes_minus_no'].values()):.3f}-{max(R['yes_minus_no'].values()):.3f} | rand {min(rr):.3f}-{max(rr):.3f} (exp {R['expected_isotropic']:.3f}) | cov {min(rc):.3f}-{max(rc):.3f} (exp {R['expected_cov']:.3f})")
for l in BL: R = res["blocks"][str(l)]; print(f"block {l}: shuffled cos with yes-no " + " ".join(f"{c:+.2f}" for c in R["shuffled_cos_with_yes_minus_no"]), "|", {k: {a: round(b, 3) for a, b in v.items()} for k, v in R["random_controls_by_kind"].items()})
print("check:", {s: [round(a, 4), round(b, 4)] for s, (a, b) in res["blocks"]["14"]["check_cos_with_rsi_direction_json"].items()})

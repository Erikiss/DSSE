"""r1b_K_S2b: K6e, lens-frame perturbation persistence (Q9 members 1-3; Q11; K6e-d). Units: the first 128 correct and 64 wrong K4
items x top-4 seeds x 2 keys (4 after G-K16's doubling) at block b*, sampled contrastive decoding at T 0.7 (keys r1bK_K6e shared by
pushed, TOK and twin runs), S = 24; segments 0-7, 8-15, 16-23. PUSH: H_A scaled by NSCALE_DN to delta2 in the expert stream only;
CLAMP: H_A mean-clamp in both streams; RS20 pushes and RS39mc[:20] clamps (TOK only); descriptive: both-stream push, T 1.0 push,
offset arm at L_P = 2,064 (K6e-d). TOK: the clean model continues from the pushed segment-1 tokens (forced); TOK+KV: the pushed run
continues with its perturbed cache."""
import numpy as np
import r1b_core as RC
import r1b_k_base as KB
from r1b_k_base import N, H, T_, write, save, load_pt, own, up, Interv, by_layer, set_values, coded, val
import r1b_k_s1 as S1
import r1b_k_s2 as S2

SEG = 8


def units(ctx, nkeys=None):
    I = S2._items(ctx); s0 = S2.states(ctx)["St"]["t0"]
    its = [i for i in I["correct"] if i in s0["items"]][:N(ctx, 128, 3)] + [i for i in I["wrong"] if i in s0["items"]][:N(ctx, 64, 2)]
    nk = nkeys or RC.load_json(own(ctx, "k/k6e/cal.json"))["n_keys"]
    return [(i, r, k) for i in its for r in range(N(ctx, 4, 2)) for k in range(nk)]


def _env(ctx):
    E = S2.env(ctx); S = S2.states(ctx); E["n26"] = S["n26"]; E["S"] = S
    E["b_star"] = RC.load_json(up(ctx, "r1b_K_S2a", "k/k4/b_star.json"))["b_star"]
    return E


def _seg1(N2):
    return lambda T, st: ((st["p0"] + T_().arange(T)) >= 0) & ((st["p0"] + T_().arange(T)) < SEG)


def _cells(ctx, E, U, prefix_inst=None):
    I = S2._items(ctx); s0 = E["S"]["St"]["t0"]; l = E["b_star"]; cells = []
    for (i, r, k) in U:
        kk = s0["items"].index(i); sd = s0["seeds"][l][kk].tolist()[r]
        cells.append(dict(item=i, block=l, ptype="t0", variant="K6e", target=I["items"][i], W=I["gate"][i]["W"], OWN=I["gate"][i]["own"],
                          H=s0["h"][l][kk], M=s0["mu"][l][kk], seeds=[sd], unit=(i, r, k), carry=("unit",)))
    return cells


def _run(ctx, E, U, T=0.7, ops=None, forced=None, prefix=None, keep_first=False, cb=None):
    cells = _cells(ctx, E, U)
    if forced is not None:
        for c, f in zip(cells, forced):
            c["forced"] = list(f)
    kf = lambda c, s: ctx.key("r1bK_K6e", c["unit"][0], c["unit"][1], c["unit"][2], f"T{T}")
    return S2.run_cells(ctx, E, cells, J=N(ctx, 24, 24), keys_fn=kf, T=T, ops=ops, prefix=prefix, keep_first=keep_first,
                        sets=True, fov=False, cb=cb)


def k6e_cal(ctx):
    """NSCALE_DN gains to delta2 (scaling active at every REF-K16 position) for H_A and each RS20 set; the in-session unit rule G-K16 on
    32 A-cal items x 4 seeds x 2 keys (projected z of E_1 at 1,536 pairs < 7 -> 4 keys); REF-U32 reporting of the calibrated pushes."""
    E = _env(ctx); model = E["model"]; dm = RC.Damage(model, KB.ref_k16(ctx)); rs20 = E["rs"]["RS20"]
    gain = lambda st, a0=3.0: RC.calibrate(lambda a: dm(lambda: Interv(model, [dict(sel=by_layer(st), kind="scale", value=a)]))["damage"],
                                           a0, RC.DELTA2, form="gain")
    cal = dict(H=gain(E["HA"]) if E["HA"] else None, RS20=[gain(s) for s in rs20])
    # G-K16: calibration units on the first K4 items (A-cal states are not in K4's cells; the first 32 K4 correct items stand in)
    U = units(ctx, nkeys=2)[:N(ctx, 32 * 4 * 2, 8)]; gH = cal["H"]["f"] if cal["H"] else 1.0
    ops = [dict(sel=by_layer(E["HA"]), kind="scale", value=gH, rows=None, when=_seg1(0))] if E["HA"] else None
    tw = _run(ctx, E, U); pu = _run(ctx, E, U, ops=_expert(ops, len(U)))
    tk = _run(ctx, E, U, forced=[r["tok"][:SEG - 1] for r in pu])
    e1 = np.array([float((a["aX"]["H"][:SEG].float() - b["aX"]["H"][:SEG].float()).mean()) if "H" in a["aX"] else 0.0 for a, b in zip(tk, tw)])
    zp = float(e1.mean() / (e1.std(ddof=1) / np.sqrt(N(ctx, 1536, 1536)))) if len(e1) > 1 and e1.std() > 0 else 0.0
    nk = 2 if zp >= 7 else 4
    u32 = KB.ref_u32_damage(ctx, model, {"H_push_delta2": lambda: Interv(model, [dict(sel=by_layer(E["HA"]), kind="scale", value=gH)])},
                            "K6e NSCALE_DN") if E["HA"] else {}
    write(ctx, "k/k6e/cal.json", dict(gain=cal, G_K16=dict(z_proj=zp, n_keys=nk, doubled=nk == 4), n_keys=nk, ref_u32=u32),
          labels=[] if nk == 2 else ["G-K16: units doubled (4 keys)"])


def _expert(ops, n):
    """The same ops restricted to the expert (real) stream rows of a rollout over n chains (rows 0..n-1 of 2n)."""
    if not ops:
        return ops
    return [dict(o, rows="real") for o in ops]


def conditions(ctx, E, cal):
    c = []
    gH = (cal["H"] or {}).get("f", 1.0)
    if E["HA"]:
        c += [("PUSH_H", "H", dict(sel=E["HA"], kind="scale", value=gH), "expert", 0.7), ("CLAMP_H", "H", dict(sel=E["HA"], kind="clamp"), "both", 0.7),
              ("PUSH2_H", "H", dict(sel=E["HA"], kind="scale", value=gH), "both", 0.7), ("PUSHT1_H", "H", dict(sel=E["HA"], kind="scale", value=gH), "expert", 1.0)]
    c += [(f"PUSH_RS{k}", f"RS{k}", dict(sel=s, kind="scale", value=cal["RS20"][k]["f"]), "expert", 0.7) for k, s in enumerate(E["rs"]["RS20"])]
    c += [(f"CLAMP_MC{k}", f"MC{k}", dict(sel=s, kind="clamp"), "both", 0.7) for k, s in enumerate(E["rs"]["RS39mc"][:N(ctx, 20, 3)])]
    return c


def k6e_twin(ctx):
    """Clean twins (T 0.7 and T 1.0; every unit and key) and G-K21 (a twin re-run in a different batch composition)."""
    E = _env(ctx); U = units(ctx)
    tw = {T: _run(ctx, E, U, T=T) for T in (0.7, 1.0)}
    sub = U[::-1][:N(ctx, 256, 6)]; rr = _run(ctx, E, sub)
    m = {r["unit"]: r["tok"] for r in tw[0.7]}; ident = float(np.mean([m[r["unit"]] == r["tok"] for r in rr]))
    save(ctx, "k/k6e/twin.pt", dict(tw={T: [_slim(r) for r in v] for T, v in tw.items()},
                                    G_K21=dict(identical=ident, label=None if ident >= 0.97 else "numerical divergence")))


def _slim(r):
    return dict(unit=r["unit"], tok=r["tok"], claims=r["claims"], censor=r["censor"], aX={k: v for k, v in r["aX"].items()})


def k6e_part(ctx, i, n):
    """A share of the conditions: the pushed run (TOK+KV continuation for H) and the TOK run (the clean model from the pushed segment-1
    tokens), with the first-position CSS for both-stream interventions."""
    torch = T_()
    E = _env(ctx); U = units(ctx); cal = RC.load_json(own(ctx, "k/k6e/cal.json"))["gain"]; R = E["R"]
    tw = load_pt(own(ctx, "k/k6e/twin.pt"))["tw"]; rows = []
    for name, X, op, streams, T in KB.parts(conditions(ctx, E, cal), i, n):
        st = op["sel"]; val_ = op.get("value") if op["kind"] == "scale" else set_values(st, R["z_mu"])
        ops = [dict(sel=by_layer(st), kind=op["kind"], value=val_, when=_seg1(0))]
        if streams == "expert":
            ops = _expert(ops, len(U))
        css, firsts = [], []
        pu = _run(ctx, E, U, T=T, ops=ops, keep_first=streams == "both", cb=(lambda r, o, s: firsts.append(r["first"])) if streams == "both" else None)
        tk = _run(ctx, E, U, T=T, forced=[r["tok"][:SEG - 1] for r in pu])
        if streams == "both":
            f0 = []
            _run(ctx, E, U, T=T, keep_first=True, cb=lambda r, o, s: f0.append(r["first"]))
            for a, b in zip(firsts, f0):
                pl = b["plaus"]; d1 = ((a["lr"] - a["lp"]) - (b["lr"] - b["lp"])).masked_fill(~pl, 0); d2 = (a["lr"] - b["lr"]).masked_fill(~pl, 0)
                css += (d1.norm(dim=-1) / d2.norm(dim=-1).clamp(min=1e-12)).tolist()
        rows.append(dict(cond=name, X=X, T=T, streams=streams, pushed=[_slim(r) for r in pu], tok=[_slim(r) for r in tk], css=css))
    save(ctx, f"k/k6e/part{i}.pt", dict(rows=rows))


def k6e_offset(ctx):
    """K6e-d: H at delta2 (expert stream), TOK route, at L_P = 2,064 (F2048 + P0; instance by item hash) with its own twins (SO-LCE
    copy path; G-K20 parity reported by K6f)."""
    torch = T_()
    from transformers import DynamicCache
    E = _env(ctx); U = units(ctx); cal = RC.load_json(own(ctx, "k/k6e/cal.json"))["gain"]; F = E["b"]["frames"]; model = E["model"]
    if not E["HA"]:
        save(ctx, "k/k6e/offset.pt", dict(rows=[], note="no H_A")); return
    gH = (cal["H"] or {}).get("f", 1.0); out = []
    for inst in range(8):
        Ui = [u for u in U if H(u[0]) % 8 == inst]
        if not Ui:
            continue
        pre = list(F["F2048"][inst]) + list(F["P0_ids"]); cache = DynamicCache()
        with torch.no_grad(), RC.sdpa_efficient(ctx.device):
            model(input_ids=torch.tensor([pre], device=ctx.device), past_key_values=cache, use_cache=True)
        with RC.sdpa_efficient(ctx.device):
            tw = _run(ctx, E, Ui, prefix=(cache, len(pre)))
            pu = _run(ctx, E, Ui, prefix=(cache, len(pre)), ops=_expert([dict(sel=by_layer(E["HA"]), kind="scale", value=gH, when=_seg1(0))], len(Ui)))
            tk = _run(ctx, E, Ui, prefix=(cache, len(pre)), forced=[r["tok"][:SEG - 1] for r in pu])
        out.append(dict(inst=inst, twin=[_slim(r) for r in tw], tok=[_slim(r) for r in tk]))
    save(ctx, "k/k6e/offset.pt", dict(rows=out))

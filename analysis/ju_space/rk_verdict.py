"""README section 11: the frozen rule of dn_out/rsi_plan_final.md (sections 5 and 6, with the amendments of section 8, written
before any real run) applied to the output of rk_loop.py.
Usage: python rk_verdict.py choose <dn_delta.json>                      -> the two extra blocks
       python rk_verdict.py gates <block_dir>                           -> pilot gates GO0-GO3 (with the GO1 retry if present)
       python rk_verdict.py block <block_dir> <out.json>                -> per-block summary (small JSON to bring back)
       python rk_verdict.py verdict <rk_out_dir> <dn_delta.json> <out.json>"""
import glob, json, math, os, sys
import numpy as np, torch

cosf = lambda a, b: float(a @ b / (a.norm() * b.norm() + 1e-12))
fin = lambda x: x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def choose_blocks(delta_path):
    """Among blocks 6-24 that are at least 4 blocks from 14 (last-token reading): the pair at least 4 blocks apart with the largest
    delta(hi) - delta(lo) (section 8: replaces 'lowest first, then highest')."""
    d = json.load(open(delta_path))["blocks"]; cand = [b for b in range(6, 25) if abs(b - 14) >= 4]; val = {b: d[str(b)]["last"]["delta"] for b in cand}
    assert all(math.isfinite(v) for v in val.values()), "non-finite delta"
    pairs = [(val[h] - val[l], l, h) for l in cand for h in cand if abs(h - l) >= 4 and val[h] > val[l]]
    _, lo, hi = max(pairs, key=lambda t: (t[0], -t[1], -t[2]))
    return lo, hi, val


def gates(bdir):
    """Pilot gates. If GO1 failed at the calibrated alpha and the retry at 1.5 alpha exists, the retry's gates count (the run then
    goes on at 1.5 alpha)."""
    P = json.load(open(f"{bdir}/pilot.json")); g = dict(P["gates"]); src = "pilot.json"
    if not g["GO1"] and os.path.exists(f"{bdir}/pilot@1.5.json"):
        P2 = json.load(open(f"{bdir}/pilot@1.5.json")); g = dict(P2["gates"]); src = "pilot@1.5.json"
    failed = [k for k in ("GO0", "GO1", "GO2", "GO3") if not g[k]]
    return dict(gates=g, source=src, failed=failed, auc=P["axes"]["auc"], alpha=P["alpha"], SEA_1=P["SEA_1"][0]["sea"])


def loops_of(bdir):
    L = json.load(open(f"{bdir}/loops.json")); meta = L.pop("_meta", dict(suffix="", alpha_scale=1.0)); return L, meta


def S4(ST, L, loop, suf):
    return cosf(ST[f"dir_{loop}{suf}_4"].double(), ST[f"dir_{loop}{suf}_3"].double()) / math.sqrt(L[loop][2]["rel"] * L[loop][3]["rel"])


def pc_pass(ST, L, suf):
    sea = [L["SEA"][n - 1]["sea"] for n in (2, 3, 4)]; s4 = S4(ST, L, "SEA", suf)
    return all(s["p"] <= 0.01 and s["c"] > 0 and s["rel"] >= 0.5 for s in sea) and s4 >= 0.5, s4


def manip_retry(bdir, meta):
    """MANIP at alpha * sqrt(2) (stage 'manip'): None if not run."""
    fs = [f for f in glob.glob(f"{bdir}/manip@*.json") if abs(json.load(open(f))["alpha_scale"] - meta.get("alpha_scale", 1.0) * math.sqrt(2)) < 1e-3]
    if not fs: return None
    M = json.load(open(fs[0]))["rows"]; return all(M[l][0]["p"] <= 0.01 and M[l][0]["c"] > 0 for l in ("RSI_a", "RSI_b") if l in M) and len(M) == 2


def block_verdict(bdir):
    L, meta = loops_of(bdir); suf = meta.get("suffix", "")
    need = ["SEA", "RSI_a", "RSI_b", "SPO", "RND_1", "RND_2"]; missing = [l for l in need if l not in L or len(L[l]) < 4]
    if missing: return dict(verdict=f"not decidable (loops missing: {missing})")
    ST = torch.load(f"{bdir}/state.pt", map_location="cpu", weights_only=False)
    k5 = json.load(open(f"{bdir}/k5.json")) if os.path.exists(f"{bdir}/k5.json") else None
    k6 = json.load(open(f"{bdir}/k6.json")) if os.path.exists(f"{bdir}/k6.json") else None
    rel = lambda loop, n: L[loop][n - 1]["rel"]; rsi = ["RSI_a", "RSI_b"]; out = {}
    out["PC_PASS"], out["SEA_S4"] = pc_pass(ST, L, suf)
    out["MANIP"] = all(L[l][0]["p"] <= 0.01 and L[l][0]["c"] > 0 for l in rsi)
    out["MANIP_retry_sqrt2"] = None if out["MANIP"] else manip_retry(bdir, meta)
    pos_all = all(L[l][n - 1]["p"] <= 0.01 and L[l][n - 1]["c"] > 0 for l in rsi for n in (2, 3, 4))
    sig_all = all(L[l][n - 1]["p"] <= 0.01 for l in rsi for n in (2, 3, 4))
    alternating = all(np.sign(L[l][n]["c"]) == -np.sign(L[l][n - 1]["c"]) for l in rsi for n in (1, 2, 3))
    mean_rel = float(np.mean([rel(l, n) for l in rsi for n in (2, 3, 4)]))
    A_bar = float(np.mean([abs(L[l][n - 1]["A"]) for l in rsi for n in (3, 4)]))
    den = sum(L[l][1]["A"] for l in rsi); E = sum(L[l][3]["A"] for l in rsi) / den if abs(den) > 1e-12 else float("nan")
    R4 = cosf(ST[f"dir_RSI_a{suf}_4"].double(), ST[f"dir_RSI_b{suf}_4"].double()) / math.sqrt(rel("RSI_a", 4) * rel("RSI_b", 4))
    D4 = {l: abs(L[l][3]["c"]) - max(abs(L[l][3][f"cos_{x}"]) for x in ("sports", "tone", "sea", "u_neg")) for l in rsi}
    dmg_ok = all(L[l][n]["damage_of_push"] <= 0.15 for l in rsi for n in range(4))
    spo_ok = abs(L["SPO"][3]["A"]) < 0.25
    beh = None
    if k5 is not None:
        z0 = k5["start_u_T_perp"]["rsi"]
        if z0 >= 3:
            zr = float(np.mean([k5[l]["4"]["rsi"] for l in rsi])); zsp = max(abs(k5[l]["4"]["sports"]) for l in rsi); zlit = max(abs(k5[l]["4"]["literature"]) for l in rsi)
            zr_signed = float(np.mean([k5[l]["4"]["rsi"] * np.sign(L[l][3]["c"]) for l in rsi]))      # oscillation: behaviour follows the sign of c_4
            spec = zsp < 2.58 and zlit < 2.58
            beh = dict(G_beh=True, z_start=z0, z_rsi_d4=zr, z_rsi_d4_signed=zr_signed, zSP_max=zsp, zLIT_max=zlit, ok_likely=zr >= 1.64 and spec, ok_oscillation=zr_signed >= 1.64 and spec)
        else: beh = dict(G_beh=False, z_start=z0)
    gate_beh = lambda kind: beh is None or not beh["G_beh"] or beh[kind]
    common = out["PC_PASS"] and fin(mean_rel) and mean_rel >= 0.5 and fin(E) and E >= 0.7 and fin(R4) and R4 >= 0.5 and all(v > 0 for v in D4.values()) and dmg_ok and spo_ok
    rnd = ["RND_1", "RND_2"]
    attractor = any(L[l][3]["p"] <= 0.01 and abs(L[l][3]["A"]) >= 0.25 for l in rnd)                                   # label: n = 4 only
    rnd_block = any(L[l][n - 1]["p"] <= 0.01 and abs(L[l][n - 1]["A"]) >= 0.25 for l in rnd for n in (3, 4))          # UNLIKELY condition: n = 3 or 4
    if common and pos_all and A_bar >= 0.5 and gate_beh("ok_likely"): verdict = "likely, RSI-dominated"
    elif common and pos_all and 0.25 <= A_bar < 0.5 and gate_beh("ok_likely"): verdict = "likely, substantial RSI component"
    elif common and sig_all and alternating and A_bar >= 0.25 and gate_beh("ok_oscillation"): verdict = "oscillation (RSI begets anti-RSI)"
    elif out["PC_PASS"] and out["MANIP"] and all(L[l][n - 1]["p"] > 0.10 for l in rsi for n in (3, 4)) and not rnd_block and dmg_ok: verdict = "unlikely"
    elif out["PC_PASS"] and not out["MANIP"] and out["MANIP_retry_sqrt2"] is False: verdict = "unlikely (no first link)"
    elif out["PC_PASS"] and not out["MANIP"] and out["MANIP_retry_sqrt2"] is None: verdict = "undecided (MANIP failed; retry at alpha * sqrt(2) not run)"
    elif not out["PC_PASS"]: verdict = "undecided (positive control failed)"
    else: verdict = "undecided"
    if k5 is None and (verdict.startswith("likely") or verdict.startswith("oscillation")): verdict += " (provisional: K5 missing)"
    if os.path.exists(f"{bdir}/pilot.json"):
        G = gates(bdir)
        if G["failed"]: verdict = f"not decidable (pilot gate {', '.join(G['failed'])} failed; descriptive only; would have read: {verdict})"
        out["gates"] = G
    spo_rows = [L["SPO"][n - 1]["sports_own"] for n in (2, 3, 4)]
    k6lab = None
    if k6 is not None:
        k6lab = {n: dict(V=r["V"], V_adj=r.get("V_adj"), z_fin=r.get("z_fin")) for n, r in k6["tests"].items()}
    out.update(verdict=verdict, mean_rel=mean_rel, A_bar=A_bar, E=E, R4=R4, D4=D4, damage_ok=dmg_ok, SPO_A4_below_025=spo_ok, behaviour=beh, K6=k6lab,
               label_attractor=attractor, label_start_specific=not attractor,
               label_behaviour=None if beh is None else ("with behaviour" if beh["G_beh"] and beh["z_rsi_d4"] >= 1.64 else ("behaviour gate not applicable" if not beh["G_beh"] else "without behaviour")),
               SPO_sustains=all(s["p"] <= 0.01 and s["c"] > 0 and s["rel"] >= 0.5 for s in spo_rows) and S4(ST, L, "SPO", suf) >= 0.5,
               alpha_scale=meta.get("alpha_scale", 1.0),
               p=[{l: L[l][n - 1]["p"] for l in rsi} for n in (1, 2, 3, 4)], c=[{l: L[l][n - 1]["c"] for l in rsi} for n in (1, 2, 3, 4)], A=[{l: L[l][n - 1]["A"] for l in rsi} for n in (1, 2, 3, 4)])
    return out


def halvings(NP, n=20, seed=3):
    """The K2 halvings of rk_loop.py (16 + 16 per prompt type, both seeds of a prompt in the same half)."""
    NRSI = NP // 2; g = np.random.default_rng(seed); H = []
    for _ in range(n):
        a = np.concatenate([g.permutation(NRSI)[: NRSI // 2], NRSI + g.permutation(NP - NRSI)[: (NP - NRSI) // 2]]); m = np.zeros(NP, bool); m[a] = True; H.append(m)
    return H


def abar_bootstrap(bdir, loop="RSI_a", B=300, seed=0):
    """A_bar of one loop (rounds 3 and 4): point estimate with the K2 halvings; 90 % interval from a prompt bootstrap stratified by
    prompt type, each draw with 5 halvings of its distinct prompts (stratified by type)."""
    ST = torch.load(f"{bdir}/state.pt", map_location="cpu", weights_only=False); _, meta = loops_of(bdir); suf = meta.get("suffix", "")
    u = ST["axes"]["u_T_perp"].double(); g = np.random.default_rng(seed)
    evs = [ST[f"ev_{loop}{suf}_{n}"] for n in (2, 3)]                                   # evaluations at d_2, d_3 give d_3, d_4
    NP = int(evs[0]["prompt_of"].max()) + 1; NRSI = NP // 2; HALV = halvings(NP)
    def one(ev, weights, halves):
        po = ev["prompt_of"].numpy(); ok = ev["ok"].numpy(); delta = ev["delta"].double(); w = torch.tensor(weights[po] * ok, dtype=torch.float64)
        if float(w.sum()) == 0: return float("nan")
        d = (w[:, None] * delta).sum(0) / w.sum(); c = cosf(d, u); rs = []
        for m in halves:
            wa = w * torch.tensor(m[po], dtype=torch.float64); wb = w * torch.tensor(~m[po], dtype=torch.float64)
            if float(wa.sum()) > 0 and float(wb.sum()) > 0: rs.append(cosf((wa[:, None] * delta).sum(0), (wb[:, None] * delta).sum(0)))
        r = float(np.mean(rs)) if rs else float("nan"); rel = max(2 * r / (1 + r), 0.05) if (math.isfinite(r) and r > -1) else float("nan")
        return abs(c) / math.sqrt(rel) if math.isfinite(rel) else float("nan")
    full = float(np.mean([one(ev, np.ones(NP), HALV) for ev in evs])); bs = []
    for _ in range(B):
        pick = np.concatenate([g.integers(0, NRSI, NRSI), NRSI + g.integers(0, NP - NRSI, NP - NRSI)]); wts = np.bincount(pick, minlength=NP).astype(float)
        dist = np.nonzero(wts)[0]; dr, ds = dist[dist < NRSI], dist[dist >= NRSI]; halves = []
        for _ in range(5):
            m = np.zeros(NP, bool); m[g.permutation(dr)[: len(dr) // 2]] = True; m[g.permutation(ds)[: len(ds) // 2]] = True; halves.append(m)
        bs.append(float(np.mean([one(ev, wts, halves) for ev in evs])))
    bs = [b for b in bs if math.isfinite(b)]
    return dict(A_bar=full, lo=float(np.percentile(bs, 5)), hi=float(np.percentile(bs, 95)), draws=len(bs))


def extra_block(bdir):
    L, meta = loops_of(bdir); suf = meta.get("suffix", ""); ST = torch.load(f"{bdir}/state.pt", map_location="cpu", weights_only=False)
    ok, s4 = pc_pass(ST, L, suf)
    v = dict(PC_PASS=ok, SEA_S4=s4, alpha_scale=meta.get("alpha_scale", 1.0), p=[L["RSI_a"][n]["p"] for n in range(4)], c=[L["RSI_a"][n]["c"] for n in range(4)], A=[L["RSI_a"][n]["A"] for n in range(4)])
    if os.path.exists(f"{bdir}/pilot.json"): v["gates"] = gates(bdir)
    return v


def delta_rule(blocks):
    """blocks: {b: dict(PC_PASS, delta_last, A_bar_RSI_a)} for the blocks that ran. Section 6 with the two-block case of section 8."""
    ent = sorted([(v["delta_last"], b, v["A_bar_RSI_a"]) for b, v in blocks.items() if v.get("PC_PASS")])
    caveat = "Three (or two) blocks cannot separate delta from depth."
    if len(ent) < 2: return "not decidable (fewer than two blocks passed the positive control). " + caveat
    (dl, bl, al), (dh, bh, ah) = ent[0], ent[-1]; mid = ent[1] if len(ent) == 3 else None
    if al["lo"] > ah["hi"] and (mid is None or mid[2]["A_bar"] <= al["hi"]): rel = "RSI share falls with delta"
    elif ah["lo"] > al["hi"] and (mid is None or mid[2]["A_bar"] <= ah["hi"]): rel = "RSI share rises with delta"
    else: rel = "no monotone relation"
    if mid is None: rel += " (two blocks passed; middle clause not applicable)"
    return rel + ". " + caveat


def main():
    cmd = sys.argv[1]
    if cmd == "choose":
        lo, hi, val = choose_blocks(sys.argv[2]); print(json.dumps(dict(b_lo=lo, b_hi=hi, delta={str(k): round(v, 4) for k, v in val.items()}))); return
    if cmd == "gates": print(json.dumps(gates(sys.argv[2]), default=float)); return
    if cmd == "block":
        bdir, out = sys.argv[2], sys.argv[3]; b = int(os.path.basename(bdir.rstrip("/"))[1:])
        res = dict(block=b, A_bar_RSI_a=abar_bootstrap(bdir)); res["verdict"] = block_verdict(bdir) if b == 14 else extra_block(bdir)
        json.dump(res, open(out, "w"), indent=1, default=float); print(json.dumps(res["verdict"].get("verdict", res["verdict"].get("PC_PASS")), default=float)); return
    rdir, dpath, out = sys.argv[2], sys.argv[3], sys.argv[4]; res = dict(blocks={}, missing={})
    lo, hi, val = choose_blocks(dpath); res["b_lo"], res["b_hi"] = lo, hi
    for b in sorted({14, lo, hi}):
        bdir = f"{rdir}/b{b}"
        if not os.path.exists(f"{bdir}/loops.json"): res["missing"][b] = "loops.json missing (block not run or not finished)"; continue
        v = block_verdict(bdir) if b == 14 else extra_block(bdir)
        if b == 14: v["PC_PASS"] = v.get("PC_PASS", False)
        v["delta_last"] = json.load(open(dpath))["blocks"][str(b)]["last"]["delta"]; v["A_bar_RSI_a"] = abar_bootstrap(bdir); res["blocks"][b] = v
    res["delta_rule"] = delta_rule(res["blocks"]); json.dump(res, open(out, "w"), indent=1, default=float)
    print(json.dumps({b: v.get("verdict", "") for b, v in res["blocks"].items()}), res["delta_rule"])


if __name__ == "__main__":
    main()

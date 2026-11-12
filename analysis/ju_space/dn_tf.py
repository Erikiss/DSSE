"""README section 11, stage 3b (rule A4a of the plan review): do late positions of long reasoning depend on the double-negative
activations (Q3) more than on their mirror Q2? Teacher-forced along the sampled traces of dn_sample.py (its trace set: the first 12
GSM8K test problems whose thinking ends between 300 and 1,024 tokens, or 300 and 1,536 if fewer than 12 of 24 do so).
Bases (interventions that leave short text nearly intact) and their Q2 mirrors:
  zero19   Q3 (Q2) set to zero in blocks 19-27
  cmean    Q3 (Q2) at the per-unit mean of a given the quadrant, at the positions where the unit is in it, all blocks
  mean     Q3 (Q2) part replaced by its per-unit mean at every position, all blocks
(means from the 48 fit texts of dn_text.py), a Gaussian-noise control per base (noise added to the MLP output of the same blocks,
sigma * RMS of the output per position, sigma bisected so that the KL on the 16 main texts matches the base's KL there in
dn_text.json), and top-k by |a| at k* (dn_text.py) and at 25 %, alone, with all Q3 restored, with all Q2 restored, and with a noise
control matched in the same way to top-k's KL. Interventions at positions >= 4 except chat-template tokens (the position policy
of dn_text.py, under which the noise was calibrated); the noise is applied with the seed it was calibrated with.
Per trace and condition: stop pressure S = sum over the thinking positions before the baseline's own '</think>' of the probability
that Qwen's recommended sampler (temperature 0.6, top-k 20, top-p 0.95) emits '</think>' there; dS = S_condition - S_baseline; KL of
the next-token distribution from the unchanged model, averaged in the thinking-position bins [0, 128), [128, 256), [256, end)
(counted from '<think>'); the late/early ratio KL[256, end) / KL[0, 128).
Rule A4a as pre-registered, with each route tested on its own measure: route (a) on dS, route (b) on the late-bin KL. A base passes
if (a) holds and the Q2 mirror's median dS is at most half of B's with a paired one-sided sign test of B_i/2 - M_i > 0 (p < 0.05),
or the same for (b) on the late-bin KL. Top-k passes a route if it meets (a) or (b) (with its own noise control) and restoring Q3
removes at least twice as much of that measure as restoring Q2: median r3 >= 2 median r2 and a sign test of r3_i - 2 r2_i > 0.
'9 of 12' scales as ceil(0.75 n) for n traces; fewer than 9 traces: undecided. 'For' and 'against' met together are reported as a
conflict. Usage: python dn_tf.py <root> <dn_sample.json> <dn_text.json> <out.json>      env: NT (3)"""
import json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ufo_common import load_model, texts, template_mask
from dn_common import MLPControl, quad

root, sp, tp, out = sys.argv[1:5]; torch.set_num_threads(int(os.environ.get("NT", "3"))); t0 = time.time(); MOCK = os.environ.get("MOCK") == "1"
if MOCK:                      # tiny random Qwen3 with the real tokenizer: a code test only
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    path = f"{root}/models/qwen3-1.7b"; tok = AutoTokenizer.from_pretrained(path); cfg = AutoConfig.from_pretrained(path)
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 28, 64, 128, 4, 2, 16
    torch.manual_seed(0); model = AutoModelForCausalLM.from_config(cfg).float().eval()
    if os.environ.get("DEVICE", "cpu") != "cpu": model = model.to(os.environ["DEVICE"])
else:
    mem = int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) / 1e6; assert mem >= 11.5 or os.environ.get("DEVICE", "cpu") != "cpu", f"only {mem:.1f} GB available"
    tok, model = load_model(root)
main, pool = texts(root, tok); NL = len(model.model.layers); H = model.model.layers[0].mlp.gate_proj.out_features
if MOCK: main = main[:3]
DEV = next(model.parameters()).device; main = [(a, b.to(DEV), c.to(DEV)) for a, b, c in main]; pool = [(a, b.to(DEV), c.to(DEV)) for a, b, c in pool]
SMP = json.load(open(sp)); TXT = json.load(open(tp)); KSTAR = TXT["A_gate"]["kstar_percent"]; END_THINK = tok.convert_tokens_to_ids("</think>")
TR = {t["problem"]: t for t in SMP["traces"]}; SET = SMP["trace_set"]; NSET = len(SET); NEED = math.ceil(0.75 * NSET); CTL = MLPControl(model)
res = dict(model="Qwen3-1.7B (post-trained checkpoint), float32", trace_set=SET, n_traces=NSET, need=NEED, L_used=SMP.get("L_used"), kstar_percent=KSTAR)
print(f"model loaded ({time.time() - t0:.0f}s); traces {SET} (need {NEED} of {NSET}); k* {KSTAR}", flush=True)

def logits_at(ids, pos):
    """Logits at the given positions only (no KV cache)."""
    try: return model(ids, use_cache=False, logits_to_keep=pos).logits[0]
    except TypeError: return model(ids, use_cache=False).logits[0][pos]

# per-unit means of the Q2 and Q3 parts on the fit texts of dn_text.py
pw = [p for p in pool if p[0] == "wiki"]; pc = [p for p in pool if p[0] == "chat"]; fit = pw[:24] + pc[:24] if not MOCK else pw[:2] + pc[:2]
S = {l: {q: [torch.zeros(H, dtype=torch.float64, device=DEV), torch.zeros(H, dtype=torch.float64, device=DEV)] for q in ("Q2", "Q3")} for l in range(NL)}; NPOS = [0]; STATE = {}
def mean_hook(l, g, u, a):
    v = STATE["valid"]; gg, uu, aa = g[0][v], u[0][v], a[0][v].double()
    for q in ("Q2", "Q3"): m = quad(gg, uu, q); S[l][q][0] += (aa * m).sum(0); S[l][q][1] += m.double().sum(0)
    if l == 0: NPOS[0] += len(aa)
    return a
CTL.set(hook=mean_hook)
with torch.no_grad():
    for _, ids, valid in fit: STATE["valid"] = valid; model(ids)
CTL.set(); MEAN = {l: {q: (S[l][q][0] / NPOS[0]).float() for q in S[l]} for l in S}; CMEAN = {l: {q: (S[l][q][0] / S[l][q][1].clamp(min=1)).float() for q in S[l]} for l in S}
print(f"means ({time.time() - t0:.0f}s)", flush=True)

GROUPS = {"all": list(range(NL)), "19-27": list(range(19, 28))}
def H_zero(q): return lambda l, g, u, a: torch.where(quad(g, u, q), torch.zeros_like(a), a)
def H_mean(q): return lambda l, g, u, a: torch.where(quad(g, u, q), torch.zeros_like(a), a) + MEAN[l][q]
def H_cmean(q): return lambda l, g, u, a: torch.where(quad(g, u, q), CMEAN[l][q].expand_as(a), a)
def H_topk(k, plus=None):
    kk = math.ceil(k * H)
    def h(l, g, u, a):
        ab = a.abs(); keep = ab >= ab.topk(kk, dim=-1).values[..., -1:]
        return a * (keep | quad(g, u, plus)) if plus else a * keep
    return h
NOISE_SEED = 1
def P_noise(sigma, seed=NOISE_SEED):
    def p(l, y):
        G = torch.Generator(device=y.device).manual_seed(seed * 100 + l); e = torch.randn(y.shape, generator=G, device=y.device); return y + sigma * y.pow(2).mean(-1, keepdim=True).sqrt() * e
    return p
BASES = {"zero19": (H_zero, "19-27", "zero"), "cmean": (H_cmean, "all", "cmean"), "mean": (H_mean, "all", "mean")}
KS = sorted({KSTAR, 25} - {None})

# noise level per base and per top-k level: KL on the 16 main texts (valid positions) equal to the condition's KL there (dn_text.json)
REF_MAIN = []
with torch.no_grad():
    for _, ids, valid in main: REF_MAIN.append(torch.log_softmax(model(ids).logits[0].double(), -1).float())
def main_kl(post, blocks):
    kl = []; n = 0
    with torch.no_grad():
        for (src, ids, valid), r in zip(main, REF_MAIN):
            CTL.set(post=post, blocks=blocks, mask=valid[None, :]); lp = torch.log_softmax(model(ids).logits[0].double(), -1); CTL.set()
            v = valid.clone(); v[-1] = False; r = r.double(); kk = (r.exp() * (r - lp)).sum(-1)[v]; kl.append(float(kk.sum())); n += len(kk)
    return sum(kl) / n
TARGETS = {b: (TXT["conditions"]["main"][f"{mode}_Q3_{grp}"]["kl"], grp) for b, (_, grp, mode) in BASES.items()}
TARGETS.update({f"top{k}": (TXT["conditions"]["main"][f"top{k}"]["kl"], "all") for k in KS})
SIG = {}
for b, (target, grp) in TARGETS.items():
    lo, hi, s, tried = 0.0, None, 0.05, []
    for _ in range(12):
        k = main_kl(P_noise(s), GROUPS[grp]); tried.append((s, k))
        if abs(math.log(max(k, 1e-12) / target)) < 0.1: break
        if k < target: lo = s; s = s * 2 if hi is None else (s + hi) / 2
        else: hi = s; s = (lo + s) / 2
    best = min(tried, key=lambda t: abs(math.log(max(t[1], 1e-12) / target)))
    SIG[b] = dict(sigma=best[0], kl=best[1], target=target, converged=abs(math.log(max(best[1], 1e-12) / target)) < 0.1, tried=tried)
    print(f"noise for {b}: sigma {best[0]:.4f} KL {best[1]:.4f} (target {target:.4f}, converged {SIG[b]['converged']}) ({time.time() - t0:.0f}s)", flush=True)
res["noise_calibration"] = SIG; del REF_MAIN

CONDS = []
for b, (Hf, grp, _) in BASES.items():
    CONDS += [(f"{b}_Q3", dict(hook=Hf("Q3"), blocks=GROUPS[grp])), (f"{b}_Q2", dict(hook=Hf("Q2"), blocks=GROUPS[grp])), (f"{b}_noise", dict(post=P_noise(SIG[b]["sigma"]), blocks=GROUPS[grp]))]
for k in KS:
    CONDS += [(f"top{k}", dict(hook=H_topk(k / 100), blocks=GROUPS["all"])), (f"top{k}+Q3", dict(hook=H_topk(k / 100, "Q3"), blocks=GROUPS["all"])),
              (f"top{k}+Q2", dict(hook=H_topk(k / 100, "Q2"), blocks=GROUPS["all"])), (f"top{k}_noise", dict(post=P_noise(SIG[f"top{k}"]["sigma"]), blocks=GROUPS["all"]))]
res["conditions"] = [n for n, _ in CONDS]

def sampler_p_end(logits):
    """Probability that the sampler (T 0.6, top-k 20, top-p 0.95) emits '</think>', per position; logits [T, V]."""
    top = logits.topk(20, dim=-1); p = torch.softmax(top.values.double() / 0.6, -1); c = p.cumsum(-1); keep = (c - p) < 0.95
    p = torch.where(keep, p, torch.zeros_like(p)); p = p / p.sum(-1, keepdim=True); return (p * (top.indices == END_THINK)).sum(-1)

PER = {name: [] for name, _ in CONDS}; BASEREC = []
for tid in SET:
    t = TR[tid]; seq = t["prompt_ids"] + t["gen_ids"][: t["think_end"] + 1]; ids_cpu = torch.tensor([seq]); P0 = len(t["prompt_ids"]); K = t["think_end"]
    pos = torch.arange(P0, P0 + K - 1)                    # positions predicting the thinking tokens before the baseline's '</think>'
    mask = torch.zeros(1, len(seq), dtype=torch.bool); mask[0, 4:] = True; mask[0] &= ~template_mask(tok, ids_cpu); mask = mask.to(DEV); ids = ids_cpu.to(DEV)
    with torch.no_grad(): lg = logits_at(ids, pos.to(DEV))
    ref = torch.cat([torch.log_softmax(c.double(), -1).float() for c in lg.split(128)]); S0 = float(sampler_p_end(lg).sum())
    BASEREC.append(dict(problem=tid, think_tokens=K, S=S0, intervened_positions=int(mask.sum()))); del lg
    bins = {"0-128": pos < P0 + 128, "128-256": (pos >= P0 + 128) & (pos < P0 + 256), "256-end": pos >= P0 + 256}
    for name, kw in CONDS:
        CTL.set(mask=mask, **kw)
        with torch.no_grad(): lgc = logits_at(ids, pos.to(DEV))
        CTL.set(); Sc = float(sampler_p_end(lgc).sum())
        kl = torch.cat([(r.double().exp() * (r.double() - torch.log_softmax(c.double(), -1))).sum(-1) for r, c in zip(ref.split(128), lgc.split(128))]).cpu(); del lgc
        kb = {b: float(kl[m].mean()) if m.any() else float("nan") for b, m in bins.items()}
        PER[name].append(dict(problem=tid, dS=Sc - S0, S=Sc, kl_bins=kb, late_early=kb["256-end"] / kb["0-128"] if kb["0-128"] > 0 else float("nan"), kl_mean=float(kl.mean())))
    print(f"trace {tid} ({K} thinking tokens, S0 {S0:.2e}): " + " ".join(f"{n} dS {PER[n][-1]['dS']:+.2f} L/E {PER[n][-1]['late_early']:.2f}" for n, _ in CONDS[:6]) + f" ({time.time() - t0:.0f}s)", flush=True)
    res["baseline"] = BASEREC; res["per_trace"] = PER; json.dump(res, open(out, "w"), indent=1)

# ---------------- rule A4a
def binom_greater(k, n):      # exact one-sided binomial tail, p = 0.5
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
def sign_p(d):                # one-sided paired sign test that d > 0 (ties dropped)
    d = [x for x in d if x == x and x != 0]; return binom_greater(sum(x > 0 for x in d), len(d)) if d else 1.0
med = lambda xs: float(np.median([x for x in xs if not math.isnan(x)]))
def summ(name):
    r = PER[name]; return dict(dS_median=med([x["dS"] for x in r]), dS_ge1=sum(x["dS"] >= 1.0 for x in r), dS_lt03=sum(x["dS"] < 0.3 for x in r), late_early_median=med([x["late_early"] for x in r]),
                               late_early_gt1=sum(x["late_early"] > 1 for x in r), late_bin_median=med([x["kl_bins"]["256-end"] for x in r]), n=len(r))
SUM = {name: summ(name) for name, _ in CONDS}; res["summary"] = SUM
dS_of = lambda n: [x["dS"] for x in PER[n]]; late_of = lambda n: [x["kl_bins"]["256-end"] for x in PER[n]]
def route_b(B, N): return B["late_early_median"] >= 2 and B["late_early_gt1"] >= NEED and B["late_early_median"] >= 1.5 * N["late_early_median"]
verd = {}
for b in BASES:
    Bn, Mn, Nn = f"{b}_Q3", f"{b}_Q2", f"{b}_noise"; B, M, N = SUM[Bn], SUM[Mn], SUM[Nn]
    a_ok = B["dS_ge1"] >= NEED; b_ok = route_b(B, N)
    p_half_a = sign_p([x / 2 - y for x, y in zip(dS_of(Bn), dS_of(Mn))]); p_half_b = sign_p([x / 2 - y for x, y in zip(late_of(Bn), late_of(Mn))])
    half_a = M["dS_median"] <= 0.5 * B["dS_median"] and p_half_a < 0.05; half_b = M["late_bin_median"] <= 0.5 * B["late_bin_median"] and p_half_b < 0.05
    big_a = M["dS_median"] >= 2 / 3 * B["dS_median"]; big_b = M["late_bin_median"] >= 2 / 3 * B["late_bin_median"]
    verd[b] = dict(a=a_ok, b=b_ok, half_a=half_a, half_b=half_b, p_half_dS=p_half_a, p_half_late=p_half_b,
                   p_B_gt_M_dS=sign_p([x - y for x, y in zip(dS_of(Bn), dS_of(Mn))]), p_B_gt_M_late=sign_p([x - y for x, y in zip(late_of(Bn), late_of(Mn))]),
                   mirror_over_B_dS=M["dS_median"] / B["dS_median"] if B["dS_median"] else float("nan"), mirror_over_B_late=M["late_bin_median"] / B["late_bin_median"] if B["late_bin_median"] else float("nan"),
                   two_thirds_a=big_a, two_thirds_b=big_b, weak=B["dS_lt03"] >= NEED and B["late_early_median"] <= 1.2,
                   passes=(a_ok and half_a) or (b_ok and half_b), against_mirror=(a_ok and big_a) or (b_ok and big_b))
top = {}
for k in KS:
    T0, T3, T2, TN = f"top{k}", f"top{k}+Q3", f"top{k}+Q2", f"top{k}_noise"; routes = {}
    for route, ok, of in (("a", SUM[T0]["dS_ge1"] >= NEED, dS_of), ("b", route_b(SUM[T0], SUM[TN]), late_of)):
        E0, E3, E2 = of(T0), of(T3), of(T2); r3 = [x - y for x, y in zip(E0, E3)]; r2 = [x - y for x, y in zip(E0, E2)]; p = sign_p([x - 2 * y for x, y in zip(r3, r2)])
        routes[route] = dict(effect=ok, removed_by_Q3_median=med(r3), removed_by_Q2_median=med(r2), p_sign=p, passes=ok and med(r3) >= 2 * med(r2) and p < 0.05,
                             q2_removes_two_thirds=med(r2) >= 2 / 3 * med(r3))
    top[k] = dict(routes=routes, passes=any(r["passes"] for r in routes.values()))
passes = [b for b in verd if verd[b]["passes"]] + [f"top{k}" for k in top if top[k]["passes"]]
against_weak = all(verd[b]["weak"] for b in verd); against_mirror = [b for b in verd if verd[b]["against_mirror"]]
why = (["every base weak"] if against_weak else []) + ([f"Q2 mirror >= 2/3 of the effect for {against_mirror}"] if against_mirror else [])
if NSET < 9: verdict = "undecided: fewer than 9 traces"
elif passes and why: verdict = f"conflict: for {passes}; against: {'; '.join(why)}"
elif passes: verdict = f"for: {passes}"
elif why: verdict = f"against: {'; '.join(why)}"
else: verdict = "undecided"
res["A4a"] = dict(bases=verd, topk=top, against_weak=against_weak, against_mirror=against_mirror, passes=passes, verdict=verdict, traces=NSET, need=NEED)
res["seconds"] = time.time() - t0; json.dump(res, open(out, "w"), indent=1)
print("A4a:", verdict, {b: (v["a"], v["b"], v["passes"]) for b, v in verd.items()}, {k: v["passes"] for k, v in top.items()}, f"({time.time() - t0:.0f}s)", flush=True)

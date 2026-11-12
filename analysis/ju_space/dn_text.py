"""README section 11, stage 1 (rule A-gate of the plan review, dn_plan_review.json): do the double-negative activations (Q3: g <= 0,
u <= 0) of Qwen3-1.7B carry something on short text that their mirror Q2 (g <= 0, u > 0) does not?
Texts: the 16 main texts of sections 7 and 9 (8 wiki + 8 chat, 64 tokens, 902 valid positions) and 16 held-out pool texts (8 wiki
passages cut to 64 tokens, 8 chats) that enter no mean. Per-unit means come from 48 other pool texts (24 wiki passages of 192 tokens,
24 chats). Interventions only at valid positions (not the first four tokens, not the chat template).
Conditions, for all blocks and for the block groups 0-9, 10-18, 19-27, each for Q3 and for its mirror Q2:
  zero   the quadrant's activations set to 0 (descriptive only: it moves the mean of the MLP write)
  mean   the quadrant's part a * 1(q) replaced at every position by its per-unit mean over the pool positions (mean-preserving)
  cmean  where a unit is in the quadrant, its activation replaced by the per-unit mean of a given the quadrant (keeps when a unit is
         in the quadrant, removes how strongly)
  rand   a random set of entries of the same sign outside the quadrant (Q1 for Q3, Q4 for Q2), matched per position and block in
         count and |a| bin (10 bins at the deciles of the quadrant's |a| on the pool), replaced by their per-unit mean given selection
         (the same selection run on the pool)
Top-k sweep (all blocks): per position and block only the k % of units with the largest |a| kept (k = 50, 40, 35, 30, 25), alone,
with all Q3 entries restored and with all Q2 entries restored. k* = smallest k with a cross-entropy increase <= 0.06 nats on the
main texts (fixed in advance by the plan review).
Descriptions per block (pool texts): share of entries, of |a|-mass and of a^2 per quadrant; RMS of the quadrant write over RMS of the
full write; projection share sum <w_q, w> / sum |w|^2 (adds to 1 over quadrants); share of the quadrant write's energy that is its
mean over positions; coherence sum |w_q|^2 / sum_i a_i^2 |d_i|^2 (d_i = down_proj column); top-k survival of Q3 and Q2 by count and
by |a|-mass.
Per condition: mean KL from the unchanged model, cross-entropy of the next token and share of the same most likely token, per text
and pooled; for the rule, 95 % bootstrap intervals over texts (2,000 resamples) of log(KL_Q3 / KL_Q2) for 'mean' and 'cmean'.
Check: the wrapped MLP without a hook reproduces the original logits exactly.
Usage: python dn_text.py <root> <out.json>      env: NT (threads, 3)"""
import json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ufo_common import load_model, texts
from dn_common import MLPControl, quad, kl_ce

root, out = sys.argv[1], sys.argv[2]; torch.set_num_threads(int(os.environ.get("NT", "3"))); t0 = time.time()
mem = int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) / 1e6; assert mem >= 9.5, f"only {mem:.1f} GB available"
tok, model = load_model(root); main, pool = texts(root, tok); NL = len(model.model.layers); H = model.model.layers[0].mlp.gate_proj.out_features
pw = [p for p in pool if p[0] == "wiki"]; pc = [p for p in pool if p[0] == "chat"]
fit = pw[:24] + pc[:24]; held = [(s, ids[:, :64], v[:64]) for s, ids, v in pw[24:32]] + pc[24:32]
print(f"model loaded ({time.time() - t0:.0f}s); main {len(main)} texts, {sum(int(v.sum()) for _, _, v in main)} valid positions; held-out {len(held)}; fit {len(fit)}", flush=True)
res = dict(model="Qwen3-1.7B (post-trained checkpoint), float32", texts=dict(main=len(main), held_out=len(held), fit=len(fit)))

# reference logits of the unchanged model (before wrapping), then the check that the wrapped MLP reproduces them
with torch.no_grad(): ref = {("m", i): torch.log_softmax(model(ids).logits[0].double(), -1).float() for i, (_, ids, _) in enumerate(main)}
CTL = MLPControl(model)
with torch.no_grad(): chk = max(float((torch.log_softmax(model(ids).logits[0].double(), -1).float() - ref[("m", i)]).abs().max()) for i, (_, ids, _) in enumerate(main[:3]))
res["check_wrapped_equals_original"] = chk; print(f"check: max |log p wrapped - original| = {chk:.2e}", flush=True); assert chk == 0.0
with torch.no_grad():
    for i, (_, ids, _) in enumerate(held): ref[("h", i)] = torch.log_softmax(model(ids).logits[0].double(), -1).float()

# ---------------- pass 1 on the fit texts: per-unit sums, |a| samples for the bins, descriptions
D = model.model.layers; dcol2 = [(L.mlp.down_proj.weight.double() ** 2).sum(0) for L in D]       # |d_i|^2 per unit
KS = [0.50, 0.40, 0.35, 0.30, 0.25]; QS = ["Q1", "Q2", "Q3", "Q4"]
acc = {l: dict(npos=0, n={q: 0 for q in QS}, mass={q: 0.0 for q in QS}, sq={q: 0.0 for q in QS}, wq2={q: 0.0 for q in QS}, w2=0.0,
               proj={q: 0.0 for q in QS}, wsum={q: torch.zeros(2048, dtype=torch.float64) for q in QS}, coh={q: 0.0 for q in QS},
               s2=torch.zeros(H, dtype=torch.float64), s3=torch.zeros(H, dtype=torch.float64), c2=torch.zeros(H, dtype=torch.float64), c3=torch.zeros(H, dtype=torch.float64),
               samp2=[], samp3=[], surv={k: dict(n2=0.0, n3=0.0, m2=0.0, m3=0.0) for k in KS}) for l in range(NL)}
gen = torch.Generator().manual_seed(0)


def stats_hook(l, g, u, a):
    A = acc[l]; v = STATE["valid"]; gg, uu, aa = g[0][v].double(), u[0][v].double(), a[0][v].double(); W = D[l].mlp.down_proj.weight.double()
    w = aa @ W.T; A["npos"] += len(aa); A["w2"] += float((w ** 2).sum())
    for q in QS:
        m = quad(gg, uu, q); aq = aa * m; wq = aq @ W.T
        A["n"][q] += int(m.sum()); A["mass"][q] += float(aq.abs().sum()); A["sq"][q] += float((aq ** 2).sum()); A["wq2"][q] += float((wq ** 2).sum())
        A["proj"][q] += float((wq * w).sum()); A["wsum"][q] += wq.sum(0); A["coh"][q] += float(((aq ** 2) * dcol2[l]).sum())
        if q in ("Q2", "Q3"):
            A["s" + q[1]] += aq.sum(0); A["c" + q[1]] += m.double().sum(0); vals = aq[m].abs()
            if len(vals): A["samp" + q[1]].append(vals[torch.randint(0, len(vals), (min(len(vals), 4000),), generator=gen)].float())
    ab = aa.abs(); tot2, tot3 = (ab * quad(gg, uu, "Q2")).sum(), (ab * quad(gg, uu, "Q3")).sum(); n2, n3 = quad(gg, uu, "Q2").sum(), quad(gg, uu, "Q3").sum()
    for k in KS:
        kk = math.ceil(k * H); thr = ab.topk(kk, dim=-1).values[:, -1:]; keep = ab >= thr; S = A["surv"][k]
        S["n2"] += float((keep & quad(gg, uu, "Q2")).sum()); S["n3"] += float((keep & quad(gg, uu, "Q3")).sum())
        S["m2"] += float((ab * keep * quad(gg, uu, "Q2")).sum()); S["m3"] += float((ab * keep * quad(gg, uu, "Q3")).sum())
    S0 = A.setdefault("tot", dict(n2=0.0, n3=0.0, m2=0.0, m3=0.0)); S0["n2"] += float(n2); S0["n3"] += float(n3); S0["m2"] += float(tot2); S0["m3"] += float(tot3)
    return a


STATE = {}
CTL.set(hook=stats_hook)
with torch.no_grad():
    for src, ids, valid in fit: STATE["valid"] = valid; model(ids)
CTL.set()
MEAN, CMEAN, EDGES, desc = {}, {}, {}, {}
for l in range(NL):
    A = acc[l]; N = A["npos"]
    MEAN[l] = {"Q2": (A["s2"] / N).float(), "Q3": (A["s3"] / N).float()}
    CMEAN[l] = {"Q2": (A["s2"] / A["c2"].clamp(min=1)).float(), "Q3": (A["s3"] / A["c3"].clamp(min=1)).float()}
    EDGES[l] = {q: torch.quantile(torch.cat(A["samp" + q[1]]), torch.linspace(0.1, 0.9, 9)) for q in ("Q2", "Q3")}
    ne = sum(A["n"].values()); mass = sum(A["mass"].values()); sq = sum(A["sq"].values())
    desc[str(l)] = {q: dict(share_entries=A["n"][q] / ne, share_abs_mass=A["mass"][q] / mass, share_a2=A["sq"][q] / sq,
                            write_rms_ratio=math.sqrt(A["wq2"][q] / A["w2"]), projection_share=A["proj"][q] / A["w2"],
                            mean_energy_share=float((A["wsum"][q] / N).pow(2).sum() * N / max(A["wq2"][q], 1e-30)), coherence=A["wq2"][q] / max(A["coh"][q], 1e-30)) for q in QS}
    desc[str(l)]["topk_survival"] = {str(k): dict(Q2_count=v["n2"] / A["tot"]["n2"], Q3_count=v["n3"] / A["tot"]["n3"], Q2_mass=v["m2"] / A["tot"]["m2"], Q3_mass=v["m3"] / A["tot"]["m3"]) for k, v in A["surv"].items()}
res["descriptions_fit_texts"] = desc; json.dump(res, open(out, "w"), indent=1)
print(f"pass 1 done ({time.time() - t0:.0f}s): Q3 share of negative-gate entries per block " + " ".join(f"{desc[str(l)]['Q3']['share_entries'] / (desc[str(l)]['Q2']['share_entries'] + desc[str(l)]['Q3']['share_entries']):.3f}" for l in range(NL)), flush=True)


# ---------------- matched random selection (same sign outside the quadrant, count and |a| bin matched per position)
OTHER = {"Q3": "Q1", "Q2": "Q4"}


def rand_select(l, g, u, a, q, seed):
    """Boolean mask of the matched random entries for quadrant q in block l ([B, T, H]); also the shortfall (unmatched count)."""
    m_q = quad(g, u, q); cand = quad(g, u, OTHER[q]); ab = a.abs(); b = torch.bucketize(ab, EDGES[l][q].to(ab.dtype)); sel = torch.zeros_like(m_q)
    G = torch.Generator().manual_seed(seed); short = 0
    for k in range(10):
        nk = (m_q & (b == k)).sum(-1, keepdim=True); ck = cand & (b == k); sc = torch.rand(ab.shape, generator=G); sc = torch.where(ck, sc, torch.full_like(sc, -1.0))
        order = sc.argsort(-1, descending=True); rank = torch.empty_like(order); rank.scatter_(-1, order, torch.arange(ab.shape[-1]).expand_as(order).contiguous())
        sel |= ck & (rank < nk); short += int((nk - ck.sum(-1, keepdim=True)).clamp(min=0).sum())
    return sel, short


# pass 2 on the fit texts: per-unit mean of the activations that the matched random selection picks
RS = {l: {q: [torch.zeros(H, dtype=torch.float64), torch.zeros(H, dtype=torch.float64)] for q in ("Q2", "Q3")} for l in range(NL)}; SHORT = {"Q2": 0, "Q3": 0, "n": {"Q2": 0, "Q3": 0}}


def rand_stats_hook(l, g, u, a):
    v = STATE["valid"]
    for q in ("Q2", "Q3"):
        sel, short = rand_select(l, g, u, a, q, seed=1000 * STATE["i"] + l); sel = sel[0][v]; aa = a[0][v].double()
        RS[l][q][0] += (aa * sel).sum(0); RS[l][q][1] += sel.double().sum(0); SHORT[q] += short; SHORT["n"][q] += int(quad(g, u, q)[0][v].sum())
    return a


CTL.set(hook=rand_stats_hook)
with torch.no_grad():
    for i, (src, ids, valid) in enumerate(fit): STATE["valid"] = valid; STATE["i"] = i; model(ids)
CTL.set(); RMEAN = {l: {q: (RS[l][q][0] / RS[l][q][1].clamp(min=1)).float() for q in ("Q2", "Q3")} for l in range(NL)}
res["rand_shortfall_fit"] = {q: SHORT[q] / max(SHORT["n"][q], 1) for q in ("Q2", "Q3")}; print(f"pass 2 done ({time.time() - t0:.0f}s): shortfall of the matched random sets {res['rand_shortfall_fit']}", flush=True)


# ---------------- conditions
def hook_for(mode, q=None, k=None, plus=None):
    if mode == "zero": return lambda l, g, u, a: torch.where(quad(g, u, q), torch.zeros_like(a), a)
    if mode == "mean": return lambda l, g, u, a: torch.where(quad(g, u, q), torch.zeros_like(a), a) + MEAN[l][q]
    if mode == "cmean": return lambda l, g, u, a: torch.where(quad(g, u, q), CMEAN[l][q].expand_as(a), a)
    if mode == "rand":
        def h(l, g, u, a):
            sel, _ = rand_select(l, g, u, a, q, seed=7 + 1000 * STATE["i"] + l); return torch.where(sel, RMEAN[l][q].expand_as(a), a)
        return h
    if mode == "topk":
        kk = math.ceil(k * H)
        def h(l, g, u, a):
            ab = a.abs(); keep = ab >= ab.topk(kk, dim=-1).values[..., -1:]
            if plus is not None: keep = keep | quad(g, u, plus)
            return a * keep
        return h
    raise ValueError(mode)


GROUPS = {"all": list(range(NL)), "0-9": list(range(0, 10)), "10-18": list(range(10, 19)), "19-27": list(range(19, 28))}
CONDS = [(f"{mode}_{q}_{grp}", hook_for(mode, q), GROUPS[grp]) for grp in GROUPS for mode in ("zero", "mean", "cmean", "rand") for q in ("Q3", "Q2")]
CONDS += [(f"top{int(k * 100)}{'' if p is None else '+' + p}", hook_for("topk", k=k, plus=p), GROUPS["all"]) for k in KS for p in (None, "Q3", "Q2")]


def run(name, hook, blocks, tset, tag):
    per = []
    with torch.no_grad():
        for i, (src, ids, valid) in enumerate(tset):
            STATE["i"] = 500 + i if tag == "h" else i; CTL.set(hook=hook, blocks=blocks, mask=valid[None, :]); lp = torch.log_softmax(model(ids).logits[0].double(), -1); CTL.set()
            r0 = ref[(tag, i)].double(); kl, ce, same = kl_ce(r0, lp, ids, valid); ce0 = kl_ce(r0, r0, ids, valid)[1]
            per.append(dict(n=len(kl), kl=float(kl.sum()), ce=float(ce.sum()), ce0=float(ce0.sum()), same=float(same.sum())))
    n = sum(p["n"] for p in per)
    return dict(kl=sum(p["kl"] for p in per) / n, ce=sum(p["ce"] for p in per) / n, ce_unchanged=sum(p["ce0"] for p in per) / n, same=sum(p["same"] for p in per) / n, per_text=per)


R = {"main": {}, "held_out": {}}
for name, hook, blocks in CONDS:
    for setname, tset, tag in (("main", main, "m"), ("held_out", held, "h")):
        R[setname][name] = run(name, hook, blocks, tset, tag)
    r = R["main"][name]; print(f"{name:18s} main KL {r['kl']:.4f} CE {r['ce']:.3f} (unchanged {r['ce_unchanged']:.3f}) same {r['same']:.3f} | held-out KL {R['held_out'][name]['kl']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    res["conditions"] = R; json.dump(res, open(out, "w"), indent=1)

# ---------------- rule A-gate and k*
rng = np.random.default_rng(2000)


def ratio_ci(setname, a, b):
    A = np.array([p["kl"] for p in R[setname][a]["per_text"]]); B = np.array([p["kl"] for p in R[setname][b]["per_text"]]); n = len(A)
    obs = float(np.log(A.sum() / B.sum())); bs = []
    for _ in range(2000):
        ix = rng.integers(0, n, n); bs.append(np.log(A[ix].sum() / B[ix].sum()))
    return dict(ratio=float(np.exp(obs)), lo=float(np.exp(np.percentile(bs, 2.5))), hi=float(np.exp(np.percentile(bs, 97.5))))


gate = {}
for grp in ("0-9", "10-18", "19-27", "all"):
    gate[grp] = {s: {m: ratio_ci(s, f"{m}_Q3_{grp}", f"{m}_Q2_{grp}") for m in ("mean", "cmean", "zero", "rand")} for s in ("main", "held_out")}
grp3 = ["0-9", "10-18", "19-27"]
passes = [g for g in grp3 if all(gate[g][s][m]["ratio"] >= 2 and gate[g][s][m]["lo"] > 1 for s in ("main", "held_out") for m in ("mean", "cmean"))]
against = all(0.67 <= gate[g]["main"][m]["ratio"] <= 1.5 for g in grp3 for m in ("mean", "cmean"))
ce_inc = {int(k * 100): R["main"][f"top{int(k * 100)}"]["ce"] - R["main"][f"top{int(k * 100)}"]["ce_unchanged"] for k in KS}
ok = [k for k, v in ce_inc.items() if v <= 0.06]; kstar = min(ok) if ok else None
res["A_gate"] = dict(ratios=gate, verdict=("for: Q3-specific in block groups %s" % passes) if passes else ("against: Q3 and Q2 within 0.67-1.5 in every block group" if against else "undecided"),
                     kstar_percent=kstar, ce_increase_topk=ce_inc)
res["seconds"] = time.time() - t0; json.dump(res, open(out, "w"), indent=1)
print("A-gate:", res["A_gate"]["verdict"], "| k* =", kstar, "| CE increase", {k: round(v, 4) for k, v in ce_inc.items()}, f"| done ({time.time() - t0:.0f}s)", flush=True)

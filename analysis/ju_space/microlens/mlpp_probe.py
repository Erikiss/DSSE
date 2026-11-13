"""MicroLens++ probe of the round-1/round-2 axis construction (Qwen3-1.7B, block 14, CPU).

Question: the RSI work defines, compares, orthogonalises, pushes and erases every direction in ONE frame - the raw residual stream
at block b, mean over ALL continuation tokens, Euclidean inner product (rk_loop.axes / stats / steer_hook; r2_rk.WriteOrth).
MicroLens (arXiv 2611.13322) shows that 'what a vector reads as' and 'what the model uses' come apart (logit-lens vs J-lens
directions suit different edits; a swap works only where the J-lens can read the word). J++ (Ayonrinde & Lindsey 2026) shows that
plain averages are corrupted by biased noise (sink / non-semantic positions) and that readouts should drop non-semantic tokens.
This script measures, on a small re-built copy of the round-1 axes, whether that frame choice matters here:

 T1  readout: top words of u_T_perp through the logit lens and the J-lens (readout-filtered vocabulary)
 T2  position filtering: axis from all continuation tokens vs semantic tokens only (letter/digit, norm <= 2 median)
 T3  'perp' check: u_T_perp is orthogonal to sports/tone/sea/NEG in the Euclidean metric; how much foreign overlap is left
     in the metric the model's readout uses (M_J = R_J^T R_J, R_J = W_U[sem] diag(g) J_b), and the same for a J-metric perp
 T4  use test (teacher-forced, matched damage 0.05 nats): push u_perp^E (as round 1) vs u_perp^J (perp in the J metric);
     effect on RSI words vs foreign word sets over the continuation (specificity)

    python mlpp_probe.py <root> <out.json>      env NP (prompts per type, 12), NNEW (32), NT (4), BLOCK (14)
<root>: models/qwen3-1.7b, jlens/Qwen3-1.7B_jacobian_lens.pt (neuronpedia/jacobian-lens)."""
import ast, hashlib, json, math, os, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessor, LogitsProcessorList

root, out_path = sys.argv[1], sys.argv[2]
torch.set_num_threads(int(os.environ.get("NT", "4"))); torch.manual_seed(0)
BL, NPT, NNEW, TEMP = int(os.environ.get("BLOCK", "14")), int(os.environ.get("NP", "12")), int(os.environ.get("NNEW", "32")), 0.7
t0 = time.time()
def log(m): print(f"[mlpp {time.time() - t0:6.0f}s] {m}", flush=True)

# ---- prompts, instructions, NEG pairs: taken verbatim from rk_loop.py (parsed, not imported: importing it loads a model)
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rk_loop.py")).read()
G = {}
for node in ast.parse(src).body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("RSI_PROMPTS", "SCENE_PROMPTS", "INSTR", "NEG_PAIRS"):
        G[node.targets[0].id] = ast.literal_eval(node.value)
PROMPTS = G["RSI_PROMPTS"][:NPT] + G["SCENE_PROMPTS"][:NPT]; INSTR = G["INSTR"]; NEG = G["NEG_PAIRS"]; NP = len(PROMPTS)

path = f"{root}/models/qwen3-1.7b"; tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval(); model.requires_grad_(False)
LAY = model.model.layers; D = model.config.hidden_size; PAD = tok.pad_token_id
enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]
chat = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
START = len(enc("<|im_start|>user\n"))
EOS = model.generation_config.eos_token_id; EOS = list(EOS) if isinstance(EOS, (list, tuple)) else [EOS]
STOP = set(EOS) | {tok.convert_tokens_to_ids("<|im_end|>"), PAD}
unit = lambda v: v / (v.norm() + 1e-12)

# ---- lenses
WU = model.lm_head.weight.detach().double()                                  # [V, D] (tied)
g_fin = model.model.norm.weight.detach().double()
J = torch.load(f"{root}/jlens/Qwen3-1.7B_jacobian_lens.pt", map_location="cpu", weights_only=True)["J"][BL].double()
vocab = tok.convert_ids_to_tokens(list(range(WU.shape[0])))
import re
SEM = torch.tensor([bool(re.search(r"[A-Za-z0-9]", tok.convert_tokens_to_string([t]) if t else "")) for t in vocab])   # J++ readout filter
WS = WU[SEM] * g_fin                                                         # filtered readout rows, final-norm gain folded in
WS = WS - WS.mean(0, keepdim=True)                                           # centred: a shift shared by all logits is invisible to softmax
GU = WS.T @ WS                                                               # [D, D]  readout Gram of the filtered unembedding
M_L = GU                                                                     # logit-lens metric
M_J = J.T @ GU @ J                                                           # J-lens metric (what the readout of a push sees)
sem_ids = torch.nonzero(SEM).squeeze(1)
def top_words(v, Rmat_left, k=15):
    s = WS @ (Rmat_left @ v); idx = torch.topk(s, k).indices; return [tok.convert_tokens_to_string([vocab[int(sem_ids[i])]]).strip() for i in idx]
def cosM(a, b, M): return float((a @ M @ b) / math.sqrt(float(a @ M @ a) * float(b @ M @ b)) + 0.0)
def perpM(v, F, M):
    """v minus its M-orthogonal projection on span(F columns)."""
    Gm = F.T @ M @ F; c = torch.linalg.solve(Gm, F.T @ M @ v); return v - F @ c

# ---- generation with rk_loop's keyed Gumbel sampling (pro and anti of a prompt share one key)
def keyseed(*k): return int(hashlib.sha256("|".join(map(str, k)).encode()).hexdigest()[:12], 16)
class Gumbel(LogitsProcessor):
    def __init__(self, keys): self.gens = [torch.Generator().manual_seed(k) for k in keys]
    def __call__(self, input_ids, scores):
        V = scores.shape[-1]; g = torch.stack([-torch.log(-torch.log(torch.rand(V, generator=Gg).clamp_(1e-12, 1 - 1e-7))) for Gg in self.gens]); return scores.float() / TEMP + g
GC = GenerationConfig(max_new_tokens=NNEW, do_sample=False, eos_token_id=EOS, pad_token_id=PAD)
def pad_batch(seqs):
    T = max(len(s) for s in seqs); ids = torch.full((len(seqs), T), PAD); am = torch.zeros(len(seqs), T, dtype=torch.long)
    for i, s in enumerate(seqs): ids[i, T - len(s):] = torch.tensor(s); am[i, T - len(s):] = 1
    return ids, am
def generate(rows, batch=24):
    out = []
    for i in range(0, len(rows), batch):
        rb = rows[i:i + batch]; ids, am = pad_batch([r[0] for r in rb])
        with torch.no_grad(): g = model.generate(input_ids=ids, attention_mask=am, generation_config=GC, logits_processor=LogitsProcessorList([Gumbel([r[1] for r in rb])]))
        for row in g[:, ids.shape[1]:].tolist():
            c = []
            for t in row:
                if t in STOP: break
                c.append(t)
            out.append(c)
    return out

CAP = {}
class Stop(Exception): pass
def cap_hook(mod, inp, out):
    if CAP.get("on"): CAP["h"] = out[0] if isinstance(out, tuple) else out; raise Stop
LAY[BL].register_forward_hook(cap_hook)
def read_tokens(prompts_ids, conts, batch=24):
    """Per-token block-b states of the unsteered model over each continuation (plain prompt + continuation), as rk_loop.read but
    keeping every token, so that the mean can be filtered (J++ lesson)."""
    res = []
    for i in range(0, len(conts), batch):
        pb, cb = prompts_ids[i:i + batch], conts[i:i + batch]; ids, am = pad_batch([p + c for p, c in zip(pb, cb)]); CAP["on"] = True
        try:
            with torch.no_grad(): model.model(input_ids=ids, attention_mask=am)
        except Stop: pass
        finally: CAP["on"] = False
        hs = CAP.pop("h").double()
        for r, c in enumerate(cb): res.append((hs[r, ids.shape[1] - len(c):].clone(), c))
    return res

PIDS = [enc(chat(p)) for p in PROMPTS]
semtok = lambda t: bool(re.search(r"[A-Za-z0-9]", tok.decode([t])))

# ---- reference texts (unsteered) for norms and the use test
ref = generate([(PIDS[p], keyseed("ref", p)) for p in range(NP)]); Rref = read_tokens(PIDS, ref)
med = float(torch.cat([h.norm(dim=-1) for h, _ in Rref]).median()); cap = 2 * med
log(f"reference texts: median block-{BL} norm {med:.1f}")

def means(R, mode):
    out = []
    for h, c in R:
        if mode == "all": m = torch.ones(len(c), dtype=torch.bool)
        else: m = torch.tensor([semtok(t) for t in c]) & (h.norm(dim=-1) <= cap)
        out.append(h[m].mean(0) if m.any() else torch.full((D,), float("nan"), dtype=torch.float64))
    return torch.stack(out)

CACHE = out_path + ".axes.pt"
if os.path.exists(CACHE):
    C = torch.load(CACHE, weights_only=False); AX, nonsem_share = C["AX"], C["nonsem"]; log("axes loaded from cache")
else:
    AX = {}; nonsem_share = {}
    for name in ("rsi", "sports", "tone", "sea"):
        pro, anti = INSTR[name]; rows = []
        for p in range(NP):
            for ins in (pro, anti): rows.append((enc(chat(PROMPTS[p] + " " + ins)), keyseed("axis", name, p)))
        conts = generate(rows); Rp = read_tokens(PIDS, conts[0::2]); Ra = read_tokens(PIDS, conts[1::2])
        for mode in ("all", "sem"):
            Xp, Xa = means(Rp, mode), means(Ra, mode); keep = ~(torch.isnan(Xp).any(1) | torch.isnan(Xa).any(1))
            AX[(name, mode)] = (Xp[keep] - Xa[keep]).mean(0)
        toks = [t for _, c in Rp + Ra for t in c]; nonsem_share[name] = float(np.mean([not semtok(t) for t in toks]))
        log(f"axis {name}: texts written and read (non-semantic token share {nonsem_share[name]:.3f})")

    def reply_read(sents, mode):
        base = enc(chat("Say one sentence.")); return means(read_tokens([base] * len(sents), [enc(s) for s in sents]), mode)
    for mode in ("all", "sem"): AX[("neg", mode)] = (reply_read([b for a, b in NEG], mode) - reply_read([a for a, b in NEG], mode)).mean(0)
    torch.save(dict(AX=AX, nonsem=nonsem_share), CACHE)

res = dict(block=BL, n_prompts=NP, nnew=NNEW, median_norm=med, nonsemantic_token_share=nonsem_share, plan="analysis/ju_space/microlens/README.md")
I = torch.eye(D, dtype=torch.float64)
Fcols = lambda mode: torch.stack([unit(AX[(n, mode)]) for n in ("sports", "tone", "sea", "neg")], 1)
uE = perpM(AX[("rsi", "all")], Fcols("all"), I)                       # = round-1 u_T_perp construction
uJ = perpM(AX[("rsi", "all")], Fcols("all"), M_J)                     # same, perp in the J-lens metric
uS = perpM(AX[("rsi", "sem")], Fcols("sem"), I)                       # position-filtered, Euclidean perp

# T1 readouts
res["T1_readout"] = dict(u_T_perp_logit_lens=top_words(uE, I), u_T_perp_J_lens=top_words(uE, J), u_T_raw_J_lens=top_words(AX[("rsi", "all")], J),
                         u_T_perpJ_J_lens=top_words(uJ, J))
res["T1_readout"]["overlap_logit_vs_J_top15"] = len(set(res["T1_readout"]["u_T_perp_logit_lens"]) & set(res["T1_readout"]["u_T_perp_J_lens"]))
# T2 position filtering
res["T2_filter"] = {n: dict(cos_all_vs_sem=float(unit(AX[(n, "all")]) @ unit(AX[(n, "sem")]))) for n in ("rsi", "sports", "tone", "sea", "neg")}
res["T2_filter"]["cos_u_T_perp_all_vs_sem"] = float(unit(uE) @ unit(uS))
# T3 perp check in the metrics
T3 = {}
for nm, M in (("euclid", I), ("logit_lens", M_L), ("J_lens", M_J)):
    T3[nm] = dict(**{f"cos(u_T_perp^E,{f})": cosM(uE, AX[(f, "all")], M) for f in ("sports", "tone", "sea", "neg")},
                  **{f"cos(u_T_perp^J,{f})": cosM(uJ, AX[(f, "all")], M) for f in ("sports", "tone", "sea", "neg")})
    F = Fcols("all"); Pf = F @ torch.linalg.solve(F.T @ M @ F, F.T @ M)          # M-orthogonal projector on the foreign span
    T3[nm]["foreign_share_u_T_perp^E"] = float((Pf @ uE) @ M @ (Pf @ uE) / (uE @ M @ uE))
    T3[nm]["foreign_share_u_T_raw"] = float((Pf @ AX[("rsi", "all")]) @ M @ (Pf @ AX[("rsi", "all")]) / (AX[("rsi", "all")] @ M @ AX[("rsi", "all")]))
    gR = torch.Generator().manual_seed(7); sh = []
    for _ in range(200):
        r = perpM(torch.randn(D, generator=gR, dtype=torch.float64), F, I); sh.append(float((Pf @ r) @ M @ (Pf @ r) / (r @ M @ r)))
    T3[nm]["foreign_share_random_Eperp_dirs_mean_p95"] = [float(np.mean(sh)), float(np.percentile(sh, 95))]
T3["cos_euclid(u_T_perp^E,u_T_perp^J)"] = float(unit(uE) @ unit(uJ))
ev = torch.linalg.eigvalsh(M_J); T3["M_J_condition"] = float(ev[-1] / ev[0].clamp(min=1e-30)); T3["M_J_eff_rank"] = float(torch.exp(-(ev / ev.sum()).clamp(min=1e-30).log().mul(ev / ev.sum()).sum()))
res["T3_perp"] = T3

# T4 use test: teacher-forced reference continuations, push at block b from the first user-content token (rk_loop units)
WORDS = dict(rsi=[" improve", " improvement", " self", " better", " learning", " upgrade", " enhance", " capabilities", " optimize", " evolve"],
             sports=[" sports", " football", " game", " team", " soccer", " basketball", " athletes", " match"],
             sea=[" sea", " ocean", " waves", " beach", " water", " sailing", " shore", " tide"],
             tone=["!", " amazing", " exciting", " wonderful", " fantastic", " incredible", " love", " great"])
WID = {k: sorted({enc(w)[0] for w in v}) for k, v in WORDS.items()}
STEER = {}
def steer_hook(mod, inp, out):
    if STEER.get("v") is None: return out
    x = out[0] if isinstance(out, tuple) else out; T = x.shape[1]
    m = (torch.arange(T) >= START).to(x.dtype)[None, :, None]; nrm = x.norm(dim=-1, keepdim=True).clamp(max=cap)
    x = x + STEER["a"] * nrm * STEER["v"].to(x.dtype) * m; return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
LAY[BL].register_forward_hook(steer_hook)
seqs = [(PIDS[p], ref[p]) for p in range(NP) if len(ref[p]) > 1]
with torch.no_grad():
    base = [torch.log_softmax(model(torch.tensor([p + c])).logits[0, len(p) - 1:len(p) + len(c) - 1].double(), -1).float() for p, c in seqs]
def pushed(v, a):
    """Streams over the sequences: returns (damage = mean KL(unsteered || pushed), effect per word set); nothing large is kept."""
    STEER.update(v=unit(v.float()), a=a); kls, eff = [], {k: [] for k in WID}
    with torch.no_grad():
        for (p, c), b in zip(seqs, base):
            o = torch.log_softmax(model(torch.tensor([p + c])).logits[0, len(p) - 1:len(p) + len(c) - 1].double(), -1); b = b.double()
            kls.append(float((b.exp() * (b - o)).sum(-1).mean()))
            for k, ids in WID.items(): eff[k].append(float((torch.logsumexp(o[:, ids], -1) - torch.logsumexp(b[:, ids], -1)).mean()))
    STEER["v"] = None; return float(np.mean(kls)), {k: float(np.mean(v_)) for k, v_ in eff.items()}
def matched(v, target=0.05):
    a, tried = 0.1, []
    for _ in range(6):
        d, e = pushed(v, a); tried.append((a, d))
        if 0.045 <= d <= 0.055: break
        a = a * math.sqrt(target / max(d, 1e-6))
    return dict(alpha=a, damage=d, effect=e, tried=tried)
T4 = {}
for nm, v in (("u_T_perp^E (round 1)", uE), ("u_T_perp^J (J-metric perp)", uJ), ("u_T_perp^sem (filtered)", uS), ("u_T raw", AX[("rsi", "all")])):
    for sgn in (1, -1):
        r = matched(sgn * v); e = r["effect"]; r["specificity_rsi_minus_max_foreign"] = e["rsi"] - max(e["sports"], e["sea"], e["tone"]) if sgn > 0 else -e["rsi"] - max(-e["sports"], -e["sea"], -e["tone"])
        T4[f"{'+' if sgn > 0 else '-'}{nm}"] = r; log(f"T4 {'+' if sgn > 0 else '-'}{nm}: alpha {r['alpha']:.3f} damage {r['damage']:.3f} effect {({k: round(x, 3) for k, x in e.items()})}")
res["T4_use"] = T4
res["seconds"] = time.time() - t0
json.dump(res, open(out_path, "w"), indent=1, ensure_ascii=False); log(f"written {out_path}")

"""README section 11, stage 2 (rule B1 of the plan review, dn_plan_review.json): 'RSI of RSI of RSI' read as repeated
re-derivation of a direction from the model's own text. A direction d_n is added at block 14 (f * |h| * unit(d_n), from the user
message on) while the model writes 40 greedy tokens after each of 32 neutral prompts (chat template, thinking off). d_{n+1} is
re-derived from how the unsteered model reads the steered continuations against its own unsteered continuations: state at block 14,
mean over the continuation tokens, difference per prompt, mean over prompts. Three iterations (d1, d2, d3); then d3 is steered once
more at the start's step size to measure how much of the start's effect it keeps.
Starts: the nesting step of README 10.5 for 'improves' and for 'describes' (state at block 14 of "The system <verb> the way it
<verb> X." minus "The system <verb> X.", each averaged over the tokens after the shared prefix "The system <verb>", mean over the
24 objects of 10.5), the persona RSI direction of 10.4 and the mean of the seven persona unit directions (both rebuilt from the
states of rsi_followup.py), and 20 random directions (10 isotropic, 10 with the covariance of real states, the seeds of 10.4; their
loops use the first 16 prompts). The step size f is calibrated for every direction and iteration so that the damage (mean KL over
the valid positions of the four texts of 10.4, steered from position 4) lies in 0.05-0.10.
Noise ceiling per iteration: number of prompts whose steered continuation differs from the unsteered one, and the split-half cosine
of d_{n+1} over prompts (20 random halvings). Steering effect of d3 against the start (same f): persona starts, change of the
log-odds of the matching answer on the 30 held-out RSI statements of 10.4; text starts, shift of the projection of the continuation
states on the start direction. Nothing learns: no weights change.
Usage: python dn_loop.py <root> <rsi_followup.json.states.pt> <out.json>      env: NT (threads, 3)"""
import json, os, random, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ufo_common import load_model, texts, Pool, unit
from rsi_statements import items as rsi_items

root, stp, out = sys.argv[1], sys.argv[2], sys.argv[3]; torch.set_num_threads(int(os.environ.get("NT", "3"))); t0 = time.time()
mem = int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) / 1e6; assert mem >= 9.5, f"only {mem:.1f} GB available"
tok, model = load_model(root); LAY = model.model.layers; main, _ = texts(root, tok); DAM = main[:4]; tok.padding_side = "left"; PAD = tok.pad_token_id
enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]
chat = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
START = len(enc("<|im_start|>user\n")); YES, NO = enc("Yes")[0], enc("No")[0]; BL = 14; cos = lambda a, b: float(a @ b / (a.norm() * b.norm()))
res = dict(model="Qwen3-1.7B (post-trained checkpoint), float32, chat template with thinking off", block=BL); print(f"model loaded ({time.time() - t0:.0f}s)", flush=True)

PROMPTS = ["Describe a quiet morning in a small town.", "Write a few sentences about the history of bread.", "Explain how rain forms.", "Tell me about a famous bridge.",
           "Describe the taste of a ripe peach.", "What should someone pack for a weekend hike?", "Write a short note to a neighbour about a lost cat.", "Explain why leaves change colour in autumn.",
           "Describe a busy train station.", "Give some tips for growing tomatoes.", "Write the opening of a short story set in a lighthouse.", "Explain how a bicycle stays upright.",
           "Describe the sound of the sea.", "Summarise the plot of a typical fairy tale.", "Explain what a public library does.", "Describe a winter market.",
           "Write a few sentences about owls.", "Explain how bees make honey.", "Describe an ideal sandwich.", "Write a postcard from a seaside holiday.",
           "Explain how to boil an egg.", "Describe an old wooden door.", "Write about a walk in the forest.", "Explain why the sky is blue.",
           "Describe a thunderstorm.", "Give advice for a first day at a new school.", "Write a short poem about a river.", "Explain how a mechanical clock works.",
           "Describe a market stall selling spices.", "Write a few sentences about mountains.", "Explain how a kite flies.", "Describe a cup of hot chocolate."]
PIDS = [enc(chat(p)) for p in PROMPTS]

# ---------------- steering at the output of block 14; per row from rowstart on (left padding), every new token during generation
STEER = dict(v=None, f=0.0, rowstart=None)
def hook(mod, inp, out):
    if STEER["v"] is None or STEER["f"] == 0: return out
    x = out[0] if isinstance(out, tuple) else out; T = x.shape[1]
    m = (torch.arange(T)[None, :] >= STEER["rowstart"][:, None]).to(x.dtype)[..., None] if (T > 1 and STEER["rowstart"] is not None) else torch.ones(x.shape[0], T, 1, dtype=x.dtype)
    x = x + STEER["f"] * x.norm(dim=-1, keepdim=True) * STEER["v"].to(x.dtype) * m; return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
LAY[BL].register_forward_hook(hook)
def steer(v=None, f=0.0, rowstart=None): STEER.update(v=None if v is None else unit(v.float()), f=f, rowstart=rowstart)

BASE_DAM = []
with torch.no_grad():
    for _, ids, valid in DAM: BASE_DAM.append(torch.log_softmax(model(ids).logits[0].double(), -1))
def damage(v, f):
    kl = []
    with torch.no_grad():
        for (_, ids, valid), b in zip(DAM, BASE_DAM):
            steer(v, f, torch.tensor([4])); lp = torch.log_softmax(model(ids).logits[0].double(), -1); kl.append(float((b.exp() * (b - lp)).sum(-1)[valid].mean()))
    steer(); return float(np.mean(kl))
def calibrate(v, f0=0.3):
    """f with damage in [0.05, 0.10] (doubling, then bisection; at most 8 evaluations)."""
    lo, hi, f, tried = 0.0, None, f0, []
    for _ in range(8):
        d = damage(v, f); tried.append((f, d))
        if 0.05 <= d <= 0.10: break
        if d < 0.05: lo = f; f = f * 2 if hi is None else (f + hi) / 2
        else: hi = f; f = (lo + f) / 2
    best = min(tried, key=lambda t: 0 if 0.05 <= t[1] <= 0.10 else min(abs(t[1] - 0.05), abs(t[1] - 0.10))); return best[0], best[1], tried

# ---------------- batched greedy generation and reading
def pad_batch(seqs):
    T = max(len(s) for s in seqs); ids = torch.full((len(seqs), T), PAD); am = torch.zeros(len(seqs), T, dtype=torch.long)
    for i, s in enumerate(seqs): ids[i, T - len(s):] = torch.tensor(s); am[i, T - len(s):] = 1
    return ids, am, torch.tensor([T - len(s) for s in seqs])
def generate(idx, v=None, f=0.0, n_new=40):
    ids, am, pad = pad_batch([PIDS[i] for i in idx]); steer(v, f, pad + START)
    with torch.no_grad(): g = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=n_new, do_sample=False, pad_token_id=PAD)
    steer(); conts = []
    for row in g[:, ids.shape[1]:].tolist():
        c = []
        for t in row:
            if t in (PAD, tok.eos_token_id) or t == tok.convert_tokens_to_ids("<|im_end|>"): break
            c.append(t)
        conts.append(c)
    return conts
def read(idx, conts):
    """Mean state at block 14 over the continuation tokens, read by the unsteered model; zero vector for an empty continuation."""
    seqs = [PIDS[i] + c for i, c in zip(idx, conts)]; ids, am, pad = pad_batch(seqs)
    with torch.no_grad(): hs = model(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states[BL + 1].double()
    return torch.stack([hs[r, ids.shape[1] - len(c):].mean(0) if len(c) else torch.zeros(hs.shape[-1], dtype=torch.float64) for r, c in enumerate(conts)])

# ---------------- start directions
OBJ = ["the chess engine", "the search algorithm", "the translation system", "the spelling checker", "the weather model", "the music player", "the route planner", "the image filter", "the code compiler", "the recipe database", "the robot arm", "the speech recogniser",
       "the trading bot", "the essay grader", "the photo editor", "the email filter", "the game engine", "the chat assistant", "the math tutor", "the website", "the voice assistant", "the map service", "the camera software", "the report generator"]
def nesting_step(verb):
    pre = len(enc(f"The system {verb}")); d = []
    for o in OBJ:
        st = []
        for t in (f"The system {verb} {o}.", f"The system {verb} the way it {verb} {o}."):
            ids = torch.tensor([enc(t)]); assert ids[0, :pre].tolist() == enc(f"The system {verb}")
            with torch.no_grad(): st.append(model(ids, output_hidden_states=True).hidden_states[BL + 1][0, pre:].double().mean(0))
        d.append(st[1] - st[0])
    return torch.stack(d).mean(0)
ST = torch.load(stp, map_location="cpu", weights_only=True)
pdir = {s: torch.where(z["label"].bool()[:, None], z["yes"][str(BL)] - z["no"][str(BL)], z["no"][str(BL)] - z["yes"][str(BL)]).mean(0) for s, z in ST.items()}
START_DIRS = {"improve_nesting": nesting_step("improves"), "describe_nesting": nesting_step("describes"), "persona_rsi": pdir["rsi"], "persona_shared_mean": torch.stack([unit(v) for v in pdir.values()]).mean(0)}
pool = Pool(f"{root}/ufo_out/ufo_pool.pt", blocks=[BL])
RAND = {f"{kind}_{i}": pool.direction(kind, BL, 1, torch.Generator().manual_seed(i))[0].double() for kind in ("rand", "cov") for i in range(10)}; del pool
res["start_cosines"] = {a: {b: cos(START_DIRS[a], START_DIRS[b]) for b in START_DIRS} for a in START_DIRS}
print(f"start directions built ({time.time() - t0:.0f}s): cos(improve, describe) {res['start_cosines']['improve_nesting']['describe_nesting']:.3f}, cos(improve, persona) {res['start_cosines']['improve_nesting']['persona_rsi']:.3f}", flush=True)

# ---------------- unsteered continuations, read once
ALL = list(range(32)); U_CONT = generate(ALL); U_READ = read(ALL, U_CONT); print(f"unsteered continuations read ({time.time() - t0:.0f}s)", flush=True)
res["unsteered_samples"] = {PROMPTS[i]: tok.decode(U_CONT[i]) for i in range(3)}

def loop(name, d_start, idx, n_iter=3, effect=False):
    d = d_start.clone(); rec = dict(iterations=[]); dirs = []
    for n in range(n_iter + (1 if effect else 0)):
        last = effect and n == n_iter
        if last: f, dmg, tried = rec["iterations"][0]["f"], None, []          # effect run: d3 at the start's step size
        else: f, dmg, tried = calibrate(d)
        conts = generate(idx, d, f); diverged = sum(c != U_CONT[i] for i, c in zip(idx, conts)); R = read(idx, conts); delta = R - U_READ[idx]
        new = delta.mean(0); g = np.random.default_rng(7); sh = []
        for _ in range(20):
            p = g.permutation(len(idx)); a, b = delta[p[: len(idx) // 2]].mean(0), delta[p[len(idx) // 2:]].mean(0)
            sh.append(cos(a, b) if a.norm() > 0 and b.norm() > 0 else float("nan"))
        it = dict(f=f, damage=dmg, calibration=tried, diverged=int(diverged), split_half=float(np.nanmean(sh)), proj_on_start=float((delta @ unit(d_start)).mean()),
                  samples={PROMPTS[i]: tok.decode(conts[j]) for j, i in enumerate(idx[:3])})
        if last: rec["effect_run"] = it; break
        it["cos_new_with_current"] = cos(new, d) if new.norm() > 0 else float("nan"); rec["iterations"].append(it); dirs.append(new); d = new
    rec["cos_d3_d1"] = cos(dirs[2], dirs[0]); rec["cos_d3_d2"] = cos(dirs[2], dirs[1]); rec["cos_d3_start"] = cos(dirs[2], d_start); rec["cos_dn_start"] = [cos(x, d_start) for x in dirs]
    print(f"loop {name}: f {[round(i['f'], 3) for i in rec['iterations']]} diverged {[i['diverged'] for i in rec['iterations']]} split-half {[round(i['split_half'], 2) for i in rec['iterations']]} | cos(d3,d1) {rec['cos_d3_d1']:.3f} cos(d3,start) {rec['cos_d3_start']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return rec, dirs

# persona steering effect on the 30 held-out RSI statements of 10.4 (same selection as rsi_direction.py)
rng = random.Random(0); L = rsi_items(); pos = [x for x in L if x["answer_matching_behavior"].strip() == "Yes"]; neg = [x for x in L if x["answer_matching_behavior"].strip() == "No"]; rng.shuffle(pos); rng.shuffle(neg)
HR = pos[50:80] + neg[50:80]; H30 = HR[:15] + HR[len(HR) // 2: len(HR) // 2 + 15]
def rsi_logodds(v=None, f=0.0):
    lo = []
    for x in H30:
        p = enc(chat(x["question"])); steer(v, f, torch.tensor([START]))
        with torch.no_grad(): lp = torch.log_softmax(model(torch.tensor([p])).logits[0, -1].double(), -1)
        steer(); m = YES if x["answer_matching_behavior"].strip() == "Yes" else NO; o = NO if m == YES else YES; lo.append(float(lp[m] - lp[o]))
    return float(np.mean(lo))

# named starts first (the code paths that run only for them are exercised early), then the random starts
N_LOOPS, N_DIRS = {}, {}; base_lo = rsi_logodds(); res["rsi_logodds_unsteered"] = base_lo
for name, v in START_DIRS.items():
    rec, dirs = loop(name, v, ALL, effect=True); N_DIRS[name] = dirs
    if name.startswith("persona"):
        f = rec["iterations"][0]["f"]; e0 = rsi_logodds(v, f) - base_lo; e3 = rsi_logodds(dirs[2], f) - base_lo
        rec["effect"] = dict(kind="change of the log-odds of the matching answer on 30 held-out RSI statements", f=f, start=e0, d3=e3, ratio=e3 / e0 if e0 else float("nan"))
    else:
        e0 = rec["iterations"][0]["proj_on_start"]; e3 = rec["effect_run"]["proj_on_start"]
        rec["effect"] = dict(kind="shift of the projection of the continuation states on the start direction", f=rec["iterations"][0]["f"], start=e0, d3=e3, ratio=e3 / e0 if e0 else float("nan"))
    print(f"   effect {name}: start {rec['effect']['start']:+.3f} d3 {rec['effect']['d3']:+.3f} ratio {rec['effect']['ratio']:.2f}", flush=True)
    N_LOOPS[name] = rec; res["named_loops"] = N_LOOPS; json.dump(res, open(out, "w"), indent=1)
res["cos_d3_improve_describe"] = cos(N_DIRS["improve_nesting"][2], N_DIRS["describe_nesting"][2])
R_LOOPS, R_DIRS = {}, {}
for name, v in RAND.items():
    R_LOOPS[name], R_DIRS[name] = loop(name, v, list(range(16))); res["random_loops"] = R_LOOPS; json.dump(res, open(out, "w"), indent=1)
MEANR = [torch.stack([unit(R_DIRS[k][n]) for k in R_DIRS]).mean(0) for n in range(3)]
p95 = lambda key: float(np.percentile([R_LOOPS[k][key] for k in R_LOOPS], 95))
res["random_p95"] = dict(cos_d3_d1=p95("cos_d3_d1"), cos_d3_start=p95("cos_d3_start"))
for name in N_LOOPS: N_LOOPS[name]["cos_dn_mean_random"] = [cos(N_DIRS[name][n], MEANR[n]) for n in range(3)]
json.dump(res, open(out, "w"), indent=1)

# ---------------- rule B1
verd = {}
for name, rec in N_LOOPS.items():
    text = name.endswith("nesting"); its = rec["iterations"]
    ceiling = all(i["split_half"] >= 0.5 and i["diverged"] >= 16 for i in its)
    c2 = rec["cos_d3_d1"] >= 0.5 and rec["cos_d3_d1"] > res["random_p95"]["cos_d3_d1"]
    c3 = rec["cos_d3_start"] > res["random_p95"]["cos_d3_start"] and (rec["cos_d3_start"] >= 0.5 if text else True)
    c4 = rec["effect"]["ratio"] >= 0.5; c5 = all(c < 0.5 for c in rec["cos_dn_mean_random"])
    against = rec["cos_d3_d1"] < 0.2 or rec["cos_d3_d1"] <= res["random_p95"]["cos_d3_d1"] or any(c >= 0.5 for c in rec["cos_dn_mean_random"])
    # rule B1 (plan review P16): a failed noise ceiling means undecided, whatever the against-criteria say
    verd[name] = dict(noise_ceiling=ceiling, c2=c2, c3=c3, c4=c4, c5=c5, against_criteria=against,
                      verdict="for" if (ceiling and c2 and c3 and c4 and c5) else ("undecided" if not ceiling else ("against" if against else "undecided")))
rsi_spec = verd["improve_nesting"]["verdict"] == "for" and verd["describe_nesting"]["verdict"] != "for" and res["cos_d3_improve_describe"] < 0.8
res["B1"] = dict(per_start=verd, rsi_specific=rsi_spec); res["seconds"] = time.time() - t0; json.dump(res, open(out, "w"), indent=1)
print("B1:", {k: v["verdict"] for k, v in verd.items()}, "| RSI-specific:", rsi_spec, f"| done ({time.time() - t0:.0f}s)", flush=True)

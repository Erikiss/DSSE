"""Follow-up to rsi_direction.py (README section 10): the split-half cosine and the bootstrap of T2-A/T2-B with label-balanced halves.
rsi_direction.py split the 100 build statements of a set into random halves without regard to the answer label; a half with, say,
27 'Yes' and 23 'No' statements keeps part of the Yes-minus-No difference of the answer token, which then dominates the half
direction. Here every half and every bootstrap resample keeps the balance (25 + 25 of the 50 + 50 statements).
Rebuilds the answer-token states of the same build statements as rsi_direction.py (same selection, checked against its cosines),
stores them (<out>.states.pt) and reports for blocks 14 and 18:
  split_half      cosine between the directions of two balanced halves, 20 halvings, per set
  gap             per sibling of the RSI set: balanced split-half cosine of RSI minus cos(RSI, sibling), with a stratified bootstrap
                  (200 resamples: statements drawn with replacement within each label, both sets)
  yes_no          the Yes-minus-No difference of the answer-token state averaged over all statements of a set (the part that cancels
                  only through the balance), its norm against the norm of the direction, and the cosine of the direction with it
  shared          the mean of the seven unit directions; per set the cosine with it and the cosines between the sets after removing it
Usage: python rsi_followup.py <root> <rsi_direction.json> <out.json>      env: NT (threads, 1), PERSONA"""
import json, os, random, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ufo_common import load_model
from rsi_statements import items as rsi_items

root, prev, out = sys.argv[1], sys.argv[2], sys.argv[3]; torch.set_num_threads(int(os.environ.get("NT", "1"))); t0 = time.time(); PER = os.environ.get("PERSONA", f"{root}/rsi_dl/persona")
mem = int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) / 1e6; assert mem >= 9.5, f"only {mem:.1f} GB available"
tok, model = load_model(root); enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]
chat = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
YES, NO = enc("Yes")[0], enc("No")[0]
# the same selection as rsi_direction.py
rng = random.Random(0)
def load_set(name):
    L = rsi_items() if name == "rsi" else [json.loads(l) for l in open(f"{PER}/{name}.jsonl")]
    pos = [x for x in L if x["answer_matching_behavior"].strip() == "Yes"]; neg = [x for x in L if x["answer_matching_behavior"].strip() == "No"]; rng.shuffle(pos); rng.shuffle(neg)
    nb, nh = (50, 30) if name == "rsi" else (50, 20)
    return dict(build=pos[:nb] + neg[:nb], held=pos[nb:nb + nh] + neg[nb:nb + nh])
SETS = ["rsi", "desire-for-self-improvement", "desire-to-be-more-intelligent", "cognitive-enhancement", "desire-for-more-capabilities", "interest-in-sports", "interest-in-literature"]
D = {s: load_set(s) for s in SETS}; BL = [14, 18]
print(f"model loaded ({time.time() - t0:.0f}s)", flush=True)
ST = {}
for s in SETS:
    Y, N = {l: [] for l in BL}, {l: [] for l in BL}; lab = []
    for x in D[s]["build"]:
        p = enc(chat(x["question"])); ids = torch.tensor([p + [YES], p + [NO]])
        with torch.no_grad(): hs = model(ids, output_hidden_states=True).hidden_states
        for l in BL: Y[l].append(hs[l + 1][0, -1].double()); N[l].append(hs[l + 1][1, -1].double())
        lab.append(1 if x["answer_matching_behavior"].strip() == "Yes" else 0)
    ST[s] = dict(yes={l: torch.stack(Y[l]) for l in BL}, no={l: torch.stack(N[l]) for l in BL}, label=torch.tensor(lab))
    print(f"states {s} ({time.time() - t0:.0f}s)", flush=True)
torch.save({s: dict(yes={str(l): v for l, v in ST[s]["yes"].items()}, no={str(l): v for l, v in ST[s]["no"].items()}, label=ST[s]["label"]) for s in SETS}, out + ".states.pt")
cos = lambda a, b: float(a @ b / (a.norm() * b.norm()))
def diffs(s, l):
    """per statement: state with the matching answer minus state with the other answer"""
    lab = ST[s]["label"].bool()[:, None]; return torch.where(lab, ST[s]["yes"][l] - ST[s]["no"][l], ST[s]["no"][l] - ST[s]["yes"][l]), ST[s]["label"]
def direction(s, l, idx=None):
    d, _ = diffs(s, l); return d.mean(0) if idx is None else d[idx].mean(0)
def halves(lab, g):
    p = np.flatnonzero(lab.numpy() == 1); n = np.flatnonzero(lab.numpy() == 0); p, n = g.permutation(p), g.permutation(n)
    return np.r_[p[: len(p) // 2], n[: len(n) // 2]], np.r_[p[len(p) // 2:], n[len(n) // 2:]]
def resample(lab, g):
    p = np.flatnonzero(lab.numpy() == 1); n = np.flatnonzero(lab.numpy() == 0); return np.r_[g.choice(p, len(p)), g.choice(n, len(n))]
old = json.load(open(prev)); res = dict(check_against_rsi_direction={}, blocks={})
for s in SETS: res["check_against_rsi_direction"][s] = dict(cos_rsi_now=cos(direction("rsi", 14), direction(s, 14)), cos_rsi_before=old["cosines_block14"]["rsi"][s])
for l in BL:
    R = dict(split_half={}, gap={}, yes_no={}, shared={})
    for s in SETS:
        _, lab = diffs(s, l); sh = []
        for i in range(20):
            a, b = halves(lab, np.random.default_rng(i)); sh.append(cos(direction(s, l, a), direction(s, l, b)))
        R["split_half"][s] = dict(mean=float(np.mean(sh)), min=float(np.min(sh)), max=float(np.max(sh)))
        yn = (ST[s]["yes"][l] - ST[s]["no"][l]).mean(0); dv = direction(s, l); R["yes_no"][s] = dict(norm_yes_minus_no=float(yn.norm()), norm_direction=float(dv.norm()), cos_direction_with_yes_minus_no=cos(dv, yn))
    _, labr = diffs("rsi", l)
    for sib in SETS[1:]:
        _, labs = diffs(sib, l); gb = []
        for i in range(200):
            g = np.random.default_rng(1000 + i); ia = resample(labr, g); ib = resample(labs, g)
            sub = labr[ia]; a, b = halves(sub, g); h1 = direction("rsi", l, ia[a]); h2 = direction("rsi", l, ia[b])
            gb.append(cos(h1, h2) - cos(direction("rsi", l, ia), direction(sib, l, ib)))
        R["gap"][sib] = dict(gap=R["split_half"]["rsi"]["mean"] - cos(direction("rsi", l), direction(sib, l)), lo=float(np.percentile(gb, 2.5)), hi=float(np.percentile(gb, 97.5)), cos=cos(direction("rsi", l), direction(sib, l)))
    U = {s: direction(s, l) / direction(s, l).norm() for s in SETS}; m = torch.stack(list(U.values())).mean(0)
    rest = {s: U[s] - (U[s] @ m) / (m @ m) * m for s in SETS}
    R["shared"] = dict(cos_with_mean={s: cos(U[s], m) for s in SETS}, cos_after_removing_mean={a: {b: cos(rest[a], rest[b]) for b in SETS} for a in SETS})
    res["blocks"][str(l)] = R
    print(f"block {l}: balanced split-half " + ", ".join(f"{s[:12]} {R['split_half'][s]['mean']:.2f}" for s in SETS), flush=True)
    print(f"block {l}: gap RSI vs siblings " + ", ".join(f"{s[:12]} {R['gap'][s]['gap']:+.2f} [{R['gap'][s]['lo']:+.2f}, {R['gap'][s]['hi']:+.2f}]" for s in SETS[1:]), flush=True)
res["seconds"] = time.time() - t0; json.dump(res, open(out, "w"), indent=1); print(f"done ({time.time() - t0:.0f}s)", flush=True)

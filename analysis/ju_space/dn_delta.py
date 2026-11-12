"""README section 11: delta per block of Qwen3-1.7B, i.e. how far apart the states of texts on different topics lie, over a floor,
with the last tokens and the length controlled (after the slide the user sent: 'Shift delta' grows with depth in Pythia-160m to
2.8B; topic axis mathematics, history, law ...; the geometry survives a control of the last token).
Texts: N questions per topic from eight MMLU test subjects (high-school mathematics, world history, professional law,
high-school biology, macroeconomics, philosophy, high-school computer science, clinical knowledge). Every text has exactly the
same length: the LAST LQ tokens of a question with at least MINQ tokens (so that every text starts inside the question and none
starts with a subject's boilerplate, e.g. 'This question refers to the following information.' in world history), written as
'Question:' + those tokens + '\\nAnswer:'. The state at the last token therefore sits at the same position in every text and
can differ only through the context. N = min(NMAX, smallest subject count), the same for every subject.
(Code review, 12 Nov 2026: the earlier cut to at most 96 tokens let topic and length go together - world history and law at the
cap, clinical knowledge and philosophy at 13-14 tokens; length alone separated the topics. Block outputs are now read with
forward hooks, so block 27 is the block's output and not the final-norm state.)
Per block (output of each of the 28 blocks) and for two readings (state at the last token; mean over the question tokens):
  F      = tr(S_between) / tr(S_within) of the topic clouds (between: spread of the topic means; within: mean spread inside a topic)
  floor  = the same with the topic labels permuted (mean over 50 permutations)
  delta  = F - floor, and delta_rel = F / floor
  probe  = accuracy of a nearest-class-mean classifier on held-out halves (states centred and scaled on the training half),
           against chance 1/8, averaged over 20 random halvings.
Usage: python dn_delta.py <root> <mmlu_topics.json> <out.json>      env: NT (3), NMAX (64), LQ (16), MINQ (20), DEVICE"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ufo_common import load_model

root, data, out = sys.argv[1], sys.argv[2], sys.argv[3]; torch.set_num_threads(int(os.environ.get("NT", "3"))); t0 = time.time()
NMAX, LQ, MINQ = int(os.environ.get("NMAX", "64")), int(os.environ.get("LQ", "16")), int(os.environ.get("MINQ", "20"))
mem = int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) / 1e6; assert mem >= 9.5 or os.environ.get("DEVICE", "cpu") != "cpu", f"only {mem:.1f} GB available"
tok, model = load_model(root); NL = len(model.model.layers); DEV = next(model.parameters()).device
rows = json.load(open(data)); SUBJ = sorted({r["subject"] for r in rows}); rng = np.random.default_rng(0)
enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]; TAIL = enc("\nAnswer:"); HEAD = enc("Question:")
pools = {s: [q for q in (enc(" " + r["question"].strip()) for r in rows if r["subject"] == s) if len(q) >= MINQ] for s in SUBJ}
N = min(NMAX, min(len(v) for v in pools.values())); texts, labels = [], []
for k, s in enumerate(SUBJ):
    for i in rng.permutation(len(pools[s]))[:N]: texts.append(HEAD + pools[s][i][-LQ:] + TAIL); labels.append(k)
labels = np.array(labels); assert len({len(t) for t in texts}) == 1
print(f"model loaded ({time.time() - t0:.0f}s); {len(texts)} texts of {len(texts[0])} tokens, {len(SUBJ)} topics x {N}; counts with >= {MINQ} tokens {[len(pools[s]) for s in SUBJ]}", flush=True)

OUTS = {}
def hook(l):
    def f(mod, inp, o): OUTS[l] = (o[0] if isinstance(o, tuple) else o)[0].detach()
    return f
H = [model.model.layers[l].register_forward_hook(hook(l)) for l in range(NL)]
LAST, MEAN = [[] for _ in range(NL)], [[] for _ in range(NL)]; qpos = slice(len(HEAD), len(HEAD) + LQ)
with torch.no_grad():
    for ids in texts:
        model.model(torch.tensor([ids]).to(DEV))
        for l in range(NL):
            h = OUTS[l]; LAST[l].append(h[-1].double().cpu().numpy()); MEAN[l].append(h[qpos].double().mean(0).cpu().numpy())
        OUTS.clear()
for h in H: h.remove()
print(f"states read ({time.time() - t0:.0f}s)", flush=True)

def fisher(X, y):
    mu = X.mean(0); B = 0.0; W = 0.0
    for k in np.unique(y):
        Xk = X[y == k]; mk = Xk.mean(0); B += len(Xk) * ((mk - mu) ** 2).sum(); W += ((Xk - mk) ** 2).sum()
    return B / W
def probe(X, y, reps=20, seed=1):
    g = np.random.default_rng(seed); acc = []
    for _ in range(reps):
        tr = np.zeros(len(y), bool)
        for k in np.unique(y):
            idx = np.nonzero(y == k)[0]; tr[g.permutation(idx)[: len(idx) // 2]] = True
        m, s = X[tr].mean(0), X[tr].std(0) + 1e-6; Z = (X - m) / s
        C = np.stack([Z[tr & (y == k)].mean(0) for k in np.unique(y)])
        d = ((Z[~tr][:, None, :] - C[None]) ** 2).sum(-1); acc.append(float((d.argmin(1) == y[~tr]).mean()))
    return float(np.mean(acc))
res = dict(model="Qwen3-1.7B (post-trained checkpoint), float32", topics=SUBJ, n_per_topic=N, question_tokens=LQ, min_question_tokens=MINQ,
           rule="last LQ tokens of questions with >= MINQ tokens; identical length; block outputs by forward hooks", tail=TAIL, blocks={})
gp = np.random.default_rng(2)
for l in range(NL):
    rec = {}
    for name, A in (("last", np.stack(LAST[l])), ("mean", np.stack(MEAN[l]))):
        F = fisher(A, labels); fl = float(np.mean([fisher(A, gp.permutation(labels)) for _ in range(50)]))
        rec[name] = dict(F=float(F), floor=fl, delta=float(F - fl), delta_rel=float(F / fl), probe=probe(A, labels), chance=1 / len(SUBJ))
    res["blocks"][l] = rec
    print(f"block {l:2d}: last F {rec['last']['F']:.3f} floor {rec['last']['floor']:.3f} delta {rec['last']['delta']:.3f} probe {rec['last']['probe']:.2f} | mean F {rec['mean']['F']:.3f} delta {rec['mean']['delta']:.3f} probe {rec['mean']['probe']:.2f}", flush=True)
res["seconds"] = time.time() - t0; json.dump(res, open(out, "w"), indent=1)

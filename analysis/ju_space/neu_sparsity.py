"""Functional sparsity of the MLP neurons of Qwen3-1.7B, after 'Sparsification by activation strength' in Goodfire (2026),
'Neurons as an Interpretable Basis' (there: Qwen3.6-35B-A3B, a mixture-of-experts model; here: a dense 1.7B model).
At every token and in every block only K of the 6144 neuron activations a = silu(gate) * up are kept, the others are set to
zero before down_proj. Conditions:
  top_f      keep the K = f * 6144 largest |a|                       (the essay's layer top-K; f = 0.078 is its 92.2 % sparsity)
  write_f    keep the K largest |a| * |W_down[:, n]|                  (size of what the neuron writes)
  rot_f      keep the K largest coordinates in a randomly ROTATED basis of the same 6144-dim space, then rotate back
             (is the neuron basis privileged, or does any basis sparsify equally well?)
  rand_f     keep K random neurons                                    (control)
  bottom_f   remove the K largest and keep the rest                   (control: the complement)
Measured against the intact model on the same text: next-token loss and KL per position.
Also stored: neuron activations at 1,000 random positions (for concentration statistics).
Usage: python neu_sparsity.py <scratch_root> <out_dir> [n_texts_per_source] [max_tokens]"""
import json, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
root, out = sys.argv[1], sys.argv[2]; NT = int(sys.argv[3]) if len(sys.argv) > 3 else 20; MAXT = int(sys.argv[4]) if len(sys.argv) > 4 else 96
torch.set_grad_enabled(False); torch.manual_seed(0); rng = np.random.default_rng(0); MODEL = f"{root}/models/qwen3-1.7b"
tok = AutoTokenizer.from_pretrained(MODEL); model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
L = model.config.num_hidden_layers; NI = model.config.intermediate_size
wiki = json.load(open(f"{out}/corpus_wikitext.json"))[:NT]; pairs = json.load(open(f"{out}/axes_pairs.json")); pick = rng.choice(len(pairs), NT, replace=False)
chat = [tok.apply_chat_template([{"role": "user", "content": pairs[i]["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) + pairs[i]["not_shown"] for i in pick]
texts = [("wiki", t) for t in wiki] + [("chat", t) for t in chat]
Q = torch.linalg.qr(torch.randn(NI, NI))[0]                       # one random rotation of the neuron space, shared by all blocks
WN = [model.model.layers[l].mlp.down_proj.weight.norm(dim=0) for l in range(L)]
FR = [0.5, 0.25, 0.125, 0.078, 0.04]
CONDS = ["base"] + [f"top_{f:g}" for f in FR] + ["write_0.078"] + [f"rot_{f:g}" for f in (0.25, 0.125, 0.078)] + ["rand_0.078", "bottom_0.078"]
SUF = ""
if len(sys.argv) > 5: CONDS = ["base"] + sys.argv[5].split(","); SUF = "_" + (sys.argv[6] if len(sys.argv) > 6 else "extra")      # other conditions, written to files with a suffix
EN = {c: [0.0, 0.0] for c in CONDS}                               # kept share of the squared activations, summed over tokens and blocks
def keep_top(score, K): return torch.zeros_like(score).scatter_(-1, score.topk(K, dim=-1).indices, 1.0)
state = {"on": False}; samp = []
def mk(l):
    def pre(m, inp):
        a = inp[0]
        if not state["on"]:
            if state.get("grab") is not None: samp.append((l, a[0, state["grab"]].to(torch.float16).clone()))
            return None
        outp = a.clone()
        for c, name in enumerate(CONDS):
            if name == "base": continue
            kind, f = name.split("_"); K = int(round(float(f) * NI)); x = a[c]
            if kind == "top": y = x * keep_top(x.abs(), K)
            elif kind == "write": y = x * keep_top(x.abs() * WN[l], K)
            elif kind == "rand": y = x * keep_top(torch.rand_like(x), K)
            elif kind == "bottom": y = x * (1 - keep_top(x.abs(), K))
            elif kind == "rot": xr = x @ Q.T; y = (xr * keep_top(xr.abs(), K)) @ Q
            outp[c] = y; EN[name][0] += float((y ** 2).sum()); EN[name][1] += float((x ** 2).sum())
        return (outp,)
    model.model.layers[l].mlp.down_proj.register_forward_pre_hook(pre)
for l in range(L): mk(l)
res = []; t0 = time.time(); S = []
for k, (src, t) in enumerate(texts):
    ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :MAXT]; T = ids.shape[1]
    state["on"] = False; state["grab"] = torch.tensor(rng.choice(np.arange(4, T), min(25, T - 4), replace=False)); samp.clear()
    model(input_ids=ids); S.append(torch.stack([next(a for l_, a in samp if l_ == l) for l in range(L)], 1).numpy()); state["grab"] = None    # [positions, L, NI]
    state["on"] = True; lg = model(input_ids=ids.repeat(len(CONDS), 1)).logits.double(); state["on"] = False
    lp = torch.log_softmax(lg[:, 8:-1], -1); tgt = ids[0, 9:]; nll = -lp[:, torch.arange(len(tgt)), tgt]; kl = (lp[0].exp()[None] * (lp[0][None] - lp)).sum(-1)
    top1 = (lp.argmax(-1) == lp[0].argmax(-1)[None]).double().mean(1)
    for c, name in enumerate(CONDS): res.append(dict(text=k, source=src, cond=name, n=int(len(tgt)), loss=float(nll[c].mean()), kl=float(kl[c].mean()), same_top1=float(top1[c])))
    if k % 5 == 0: print(f"  {k}/{len(texts)} texts ({time.time() - t0:.0f}s)", flush=True)
import pandas as pd
R = pd.DataFrame(res); R.to_csv(f"{out}/neu_sparsity{SUF}.csv", index=False); S = np.concatenate(S)[:1000]
if not SUF: np.save(f"{out}/neu_act_sample.npy", S)
def summ(G):
    w = G.n / G.n.sum(); return pd.Series(dict(loss=float((G.loss * w).sum()), kl=float((G.kl * w).sum()), same_top1=float((G.same_top1 * w).sum()), positions=int(G.n.sum())))
T_ = R.groupby("cond", sort=False)[["n", "loss", "kl", "same_top1"]].apply(summ); T_["d_loss"] = T_.loss - T_.loss["base"]; T_["kept"] = [1.0 if c == "base" else (1 - float(c.split("_")[1]) if c.startswith("bottom") else float(c.split("_")[1])) for c in T_.index]
T_["kept_energy"] = [1.0 if c == "base" else EN[c][0] / max(EN[c][1], 1e-12) for c in T_.index]
bs = []                                                           # bootstrap over texts for the loss increase
for _ in range(1000):
    ii = rng.choice(len(texts), len(texts)); B = pd.concat([R[R.text == i] for i in ii]); b = B.groupby("cond", sort=False)[["n", "loss", "kl", "same_top1"]].apply(summ); bs.append((b.loss - b.loss["base"]).values)
bs = np.array(bs); T_["d_loss_lo"] = np.percentile(bs, 2.5, 0); T_["d_loss_hi"] = np.percentile(bs, 97.5, 0)
pd.set_option("display.width", 200); print(f"{len(texts)} texts ({NT} wikitext paragraphs, {NT} DSSE chats), up to {MAXT} tokens; scored positions per condition: {int(T_.positions.iloc[0])}")
print(T_[["kept", "kept_energy", "loss", "d_loss", "d_loss_lo", "d_loss_hi", "kl", "same_top1"]].round(3).to_string())
for s_ in ("wiki", "chat"):
    Ts = R[R.source == s_].groupby("cond", sort=False)[["n", "loss", "kl", "same_top1"]].apply(summ); print(f"   {s_}: loss increase", {c: round(float(Ts.loss[c] - Ts.loss['base']), 3) for c in Ts.index if c != "base"})
# concentration of the activations: share of the squared write size carried by the K largest units, neuron basis against rotated basis
Sx = torch.tensor(S.astype(np.float32)); conc = {}
for f in (0.04, 0.078, 0.125, 0.25):
    K = int(round(f * NI)); e_n = []; e_r = []
    for l in range(L):
        x = Sx[:, l]; w = x * WN[l]; e_n.append(float((w.abs().topk(K, -1).values ** 2).sum(-1).div((w ** 2).sum(-1)).mean())); xr = x @ Q.T; e_r.append(float((xr.abs().topk(K, -1).values ** 2).sum(-1).div((xr ** 2).sum(-1)).mean()))
        if f == 0.078 and l in (0, 7, 14, 21, 27): print(f"   block {l}: share of the squared write size in the top 7.8% of units: neurons {e_n[-1]:.2f}, rotated basis {e_r[-1]:.2f}")
    conc[f] = dict(neuron=float(np.mean(e_n)), rotated=float(np.mean(e_r)))
print("concentration (mean over blocks): share of squared size in the K largest units:", {f: {k: round(v, 3) for k, v in c.items()} for f, c in conc.items()})
json.dump(dict(table=T_.round(5).reset_index().to_dict("records"), concentration=conc, n_texts=len(texts), max_tokens=MAXT), open(f"{out}/neu_sparsity{SUF}.json", "w"), indent=1)
print(f"done ({time.time() - t0:.0f}s)")

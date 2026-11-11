"""A loop in real time: Qwen3-1.7B writes an answer token by token (greedy) while, at EVERY token, its state after one block is
disturbed or passed through an SAE. Each generated token is fed back, so a disturbance can add up over the text.
Rows per prompt (all start from the same chat prompt and then write their own text):
  clean                               nothing is changed
  cov2 x 3 seeds, irr2 x 3 seeds      noise of size 0.3 |h| after block 2, fresh at every token (inside / outside the data subspace)
  cov2_big                            the same with size 1.0 |h|
  cov14                               noise of size 0.3 |h| after block 14
  sae0, sae2, sae14                   the state is replaced by its SAE reconstruction (Qwen-Scope TopK, k = 50) after block 0 / 2 / 14
  sae2_filter                         noise as in cov2, then SAE(h + d) + (h - SAE(h)): the SAE as a filter, without its own error
Read per generated token t = 1..N:
  kl      KL(clean model || row) for the next token, both given the row's own text so far (the clean model reads that text
          without any disturbance): how far is the disturbed loop from the undisturbed model at this point?
  same    is the row's token the one the clean model would have chosen for this text?
  nll     how plausible the clean model finds the row's token
  ident   is the row's text still identical to the clean row's text up to here?
Accumulation would show as kl growing with t; a stable loop as a level kl.
Usage: python ufo_loop.py <scratch_root> <out_dir>      env: NT, NPROMPT (default 12), NEW (default 24)"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ufo_common import load_model, Pool, unit, FIRST

root, out = sys.argv[1], sys.argv[2]; torch.set_grad_enabled(False); torch.set_num_threads(int(os.environ.get("NT", "3"))); tok, model = load_model(root); LAYERS = model.model.layers; D = model.config.hidden_size
NP_ = int(os.environ.get("NPROMPT", "12")); NEW = int(os.environ.get("NEW", "24")); pairs = json.load(open(f"{root}/ju_out/axes_pairs.json")); pick = np.random.default_rng(0).permutation(len(pairs)); P = Pool(f"{out}/ufo_pool.pt", blocks=[0, 2, 14])
sel = list(pick[:8]) + list(pick[168:168 + max(0, NP_ - 8)]); sel = sel[:NP_]                 # the 8 chats of the main runs, then chats that are in no pool
special = set(tok.all_special_ids) | set(tok.get_added_vocab().values())


class TopK:
    def __init__(s, path, k=50):
        sd = torch.load(path, map_location="cpu", weights_only=True); s.We, s.be, s.Wd, s.bd, s.k = sd["W_enc"].float(), sd["b_enc"].float(), sd["W_dec"].float().T.contiguous(), sd["b_dec"].float(), k

    def __call__(s, x):
        v, ix = (x @ s.We.T + s.be).topk(s.k, -1); return (s.Wd[ix] * v[..., None]).sum(1) + s.bd


SAE = {b: TopK(f"{root}/qscope_dl/layer{b}.sae.pt") for b in (0, 2, 14)}
ROWS = [("clean", -1, "none", 0.0, 0)] + [(f"cov2_s{s}", 2, "cov", 0.3, s) for s in range(3)] + [(f"irr2_s{s}", 2, "irr", 0.3, s) for s in range(3)] + [("cov2_big", 2, "cov", 1.0, 0), ("cov14", 14, "cov", 0.3, 0), ("sae0", 0, "sae", 0.0, 0), ("sae2", 2, "sae", 0.0, 0), ("sae14", 14, "sae", 0.0, 0), ("sae2_filter", 2, "filter", 0.3, 0)]
C = len(ROWS); st = dict(on=False)


def hook(m):
    def f(mod, inp, o):
        if not st["on"]: return o
        h = o[0] if isinstance(o, tuple) else o
        for c, (name, b, kind, fz, seed) in enumerate(ROWS):
            if b != m: continue
            row = h[c]; ok = st["valid"] & (row.norm(dim=-1) < 10 * P.med(m))
            if not ok.any(): continue
            X = row[ok]; n = len(X)
            if kind == "sae": row[ok] = SAE[m](X)
            else:
                d = fz * X.norm(dim=-1, keepdim=True) * P.direction("cov" if kind == "filter" else kind, m, n, st["G"][c]); row[ok] = (SAE[m](X + d) + (X - SAE[m](X))) if kind == "filter" else X + d
        return o
    return f


for m in (0, 2, 14): LAYERS[m].register_forward_hook(hook(m))
res = dict(kl=[], same=[], nll=[], ident=[]); meta = []; t0 = time.time()
for k, i in enumerate(sel):
    ids = tok(tok.apply_chat_template([{"role": "user", "content": pairs[i]["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False), return_tensors="pt", add_special_tokens=False)["input_ids"][:, -96:]; T = ids.shape[1]
    valid = torch.tensor([int(t) not in special for t in ids[0]]); valid[:FIRST] = False; st.update(on=True, valid=valid, G={c: torch.Generator().manual_seed(5000 + 100 * k + c) for c in range(C)})
    o = model(input_ids=ids.repeat(C, 1), use_cache=True); pkv = o.past_key_values; last = o.logits[:, -1]; Q, gen = [], []; st["valid"] = torch.ones(1, dtype=torch.bool)
    for t in range(NEW):
        lq = torch.log_softmax(last.double(), -1); Q.append(lq); nxt = lq.argmax(-1); gen.append(nxt); o = model(input_ids=nxt[:, None], past_key_values=pkv, use_cache=True); pkv = o.past_key_values; last = o.logits[:, -1]
    gen = torch.stack(gen, 1); Q = torch.stack(Q, 1); st["on"] = False; del pkv                                                        # gen [C, NEW], Q [C, NEW, V]
    kl = np.zeros((C, NEW)); same = np.zeros((C, NEW)); nll = np.zeros((C, NEW))
    for c0 in range(0, C, 4):                                                                                                          # the clean model reads each row's own text
        full = torch.cat([ids.repeat(len(gen[c0:c0 + 4]), 1), gen[c0:c0 + 4]], 1); lp = torch.log_softmax(model(input_ids=full).logits[:, T - 1:T - 1 + NEW].double(), -1); q = Q[c0:c0 + 4]
        kl[c0:c0 + 4] = (lp.exp() * (lp - q)).sum(-1).numpy(); same[c0:c0 + 4] = (lp.argmax(-1) == gen[c0:c0 + 4]).double().numpy(); nll[c0:c0 + 4] = (-lp.gather(-1, gen[c0:c0 + 4][..., None])[..., 0]).numpy()
    ident = (torch.cumprod((gen == gen[0]).long(), 1)).double().numpy(); res["kl"].append(kl); res["same"].append(same); res["nll"].append(nll); res["ident"].append(ident)
    meta.append(dict(prompt=int(i), tokens=int(T), texts={ROWS[c][0]: tok.decode(gen[c]) for c in (0, 1, 4, 7, 9, 10, 11)}))
    print(f"  prompt {k + 1}/{len(sel)} T={T} ({time.time() - t0:.0f}s) | clean-row KL {kl[0].max():.1e} | " + " ".join(f"{ROWS[c][0]} {kl[c].mean():.3f}" for c in (1, 4, 7, 8, 9, 10, 11, 12)), flush=True)
    np.savez(f"{out}/ufo_loop.npz", **{n: np.array(v) for n, v in res.items()}); json.dump(dict(rows=ROWS, meta=meta, new=NEW), open(f"{out}/ufo_loop_meta.json", "w"), indent=1)
K = np.array(res["kl"]); S_ = np.array(res["same"]); I = np.array(res["ident"]); N = np.array(res["nll"]); th = NEW // 3
print(f"{'row':12s} {'KL tokens 1-' + str(th):>14s} {'KL middle':>10s} {'KL last third':>14s} {'same choice':>12s} {'identical to clean text at the end':>36s} {'nll under clean model':>22s}")
for c, r in enumerate(ROWS): print(f"{r[0]:12s} {K[:, c, :th].mean():14.3f} {K[:, c, th:2 * th].mean():10.3f} {K[:, c, 2 * th:].mean():14.3f} {S_[:, c].mean():12.2f} {I[:, c, -1].mean():36.2f} {N[:, c].mean():22.2f}")
print(f"done ({time.time() - t0:.0f}s)")

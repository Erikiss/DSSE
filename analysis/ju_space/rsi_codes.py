"""Can Qwen3-1.7B read Braille and Base64, and does it hold their 6-bit structure beyond the bytes it is fed? (README section 10;
rules T4-C, T4-A, T4-B of the plan review rsi_plan_review.json)
T4-C competence (scored among candidates, plain text and chat template with thinking off):
  braille   Braille cell -> Latin letter, the 24 letters whose cell is tokenised as [prefix][byte] (a and l excluded), 3 templates
  base64    a Base64 code of 4 characters -> the three-letter word it encodes, among 60 candidate words (single tokens)
  index     Base64 symbol -> its index (0..63), greedy decoding; reported for a-z and 0-9 together and for A-Z apart
  binary    6-bit binary string -> decimal value, greedy decoding
T4-A Braille dot bits with matched tokens. For the values k whose six-dot Braille cell U+2800+k, dot-7 Braille cell U+2840+k and
  mathematical operator U+2A00+k are all tokenised as [2-byte prefix token][the same byte token], the state at the byte token in
  3 frames (no space before the cell) and 2 running contexts. Per bit of k: difference-of-means and ridge probes trained on half of
  the cells and tested on the other half, averaged over 200 splits by cell; null: the same statistic, averaged over the same 200
  splits, after relabelling the cells (200 relabellings). Six-dot Braille and U+2A00 with both probes; per cell the held-out
  accuracy on bits 0-2, and a bootstrap of these per-cell values over the 51 cells (1,000 resamples) of Braille minus U+2A00. That
  bootstrap keeps the fitted probes fixed and is therefore too narrow; rsi_codes_boot.py refits the probes on every resample. Transfer of probes trained on
  all six-dot cells to dot-7 Braille (same dots) and to U+2A00 (same bytes), both probes. Decade structure of literary Braille:
  consistency of the offset vectors between the letters a-j and k-t (dot 3 added, byte + 4) against the same byte offsets in U+2A00.
T4-B Base64 index bits: 62 symbols (+ and / excluded) in 3 separated frames, each symbol checked to be one token; per index bit the
  same probes, against 200 relabellings within the classes A-Z, a-z, 0-9 (each again averaged over the same 200 splits).
The first version (11 November, before the second review of README section 10) read every relabelling on a single split while the
measured value is the mean over 200 splits, so its 99th percentiles were too high; it also ran the ridge probe for Braille only.
The states of T4-A and T4-B are saved next to the output (<out>.states.npz, float32).
Layers: the embedding (emb) and the outputs of blocks 0, 2, 6, 10, 14, 18, 22, 27.
Usage: python rsi_codes.py <root> <out.json>      env: NT (threads, 1)"""
import json, os, random, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ufo_common import load_model

root, out = sys.argv[1], sys.argv[2]; torch.set_num_threads(int(os.environ.get("NT", "1"))); t0 = time.time()
mem = int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) / 1e6; assert mem >= 9.5, f"only {mem:.1f} GB available"
tok, model = load_model(root); LAYERS = ["emb", 0, 2, 6, 10, 14, 18, 22, 27]; HS = lambda l: 0 if l == "emb" else l + 1; res = dict(model="Qwen3-1.7B (post-trained checkpoint, chat template with thinking off where a chat format is used)")
enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]
chat = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
print(f"model loaded ({time.time() - t0:.0f}s), {mem:.1f} GB were available", flush=True)


def states(text, find):
    """Hidden states at the position of the last token of the token sequence `find` (ids) inside `text`; None if not found."""
    ids = enc(text); n = len(find)
    pos = [i for i in range(len(ids) - n + 1) if ids[i:i + n] == find]
    if len(pos) != 1: return None
    with torch.no_grad(): hs = model(torch.tensor([ids]), output_hidden_states=True).hidden_states
    return {l: hs[HS(l)][0, pos[0] + n - 1].double().numpy() for l in LAYERS}


def next_logprobs(text):
    with torch.no_grad(): return torch.log_softmax(model(torch.tensor([enc(text)])).logits[0, -1].double(), -1)


def greedy(text, n=4):
    ids = torch.tensor([enc(text)])
    with torch.no_grad(): g = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(g[0, ids.shape[1]:])


def wilson_low(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; return (p + z * z / (2 * n) - z * ((p * (1 - p) + z * z / (4 * n)) / n) ** .5) / d


# ---------------- T4-C competence
BRAILLE = dict(zip("abcdefghijklmnopqrstuvwxyz", [1, 3, 9, 25, 17, 11, 27, 19, 10, 26, 5, 7, 13, 29, 21, 15, 31, 23, 14, 30, 37, 39, 58, 45, 61, 53]))   # dot bits 1-6 -> bit 0-5
cell = lambda k, base=0x2800: chr(base + k)
pfx = enc(cell(1))
res["tokenisation"] = {c: enc(cell(BRAILLE[c])) for c in BRAILLE}
letters = [c for c in BRAILLE if len(enc(cell(BRAILLE[c]))) == 2 and enc(cell(BRAILLE[c]))[0] == enc(cell(3))[0]]
comp = {}
LET = {c: [enc(" " + c)[0], enc(" " + c.upper())[0], enc(c)[0], enc(c.upper())[0]] for c in BRAILLE}


def score_letter(lp):
    s = {c: max(float(lp[i]) for i in LET[c]) for c in BRAILLE}; return max(s, key=s.get)


TPL = [lambda X: f"Braille cell: {X}\nLatin letter:", lambda X: f"The Braille character {X} is the letter", lambda X: chat(f"Which Latin letter does the Braille cell {X} stand for? Answer with the letter only.")]
r = []
for c in letters:
    for i, f in enumerate(TPL): r.append(dict(letter=c, template=i, guess=score_letter(next_logprobs(f(cell(BRAILLE[c]))))))
k = sum(x["guess"] == x["letter"] for x in r); comp["braille_to_letter"] = dict(n=len(r), right=k, accuracy=k / len(r), wilson_low=wilson_low(k, len(r)), chance=1 / 26, by_template=[sum(x["guess"] == x["letter"] for x in r if x["template"] == i) / sum(1 for x in r if x["template"] == i) for i in range(3)], letters=letters)
print("braille -> letter:", comp["braille_to_letter"]["accuracy"], comp["braille_to_letter"]["by_template"], f"({time.time() - t0:.0f}s)", flush=True)
import base64
WORDS = [w for w in ("cat dog sun car red big box hat pen cup bed bus fox key map net oil pig rat sea six ten top toy van war web win zoo arm art bag ban bat bit boy bus cow cry day ear egg end eye fan far few fig fit fly fun gas god gun guy ham hen hit hot ice ink jam jar job joy kid kit lab law leg lip log low man mat mix mud nut oak odd old owl pan pay pet pie pin pit pot raw ray rib rod row rug run sad sat saw say set sir sky son spy sum tab tag tax tea tie tin tip toe ton tub two use war wax way wet who why wig yes yet you zip").split() if len(enc(" " + w)) == 1 and len(enc(w)) == 1]
WORDS = list(dict.fromkeys(WORDS))[:60]; WID = [enc(" " + w)[0] for w in WORDS]                    # every candidate is one token with and without a leading space
TPL64 = [lambda C: f"Base64: {C}\nDecoded:", lambda C: chat(f"Decode this Base64 string and answer with the decoded word only: {C}") + " "]
r = []
for w in WORDS:
    C = base64.b64encode(w.encode()).decode()
    for i, f in enumerate(TPL64):
        lp = next_logprobs(f(C)) if i == 0 else next_logprobs(f(C)[:-1]); ids = WID if i == 0 else [enc(x)[0] for x in WORDS]
        s = [float(lp[j]) for j in ids]; r.append(dict(word=w, template=i, guess=WORDS[int(np.argmax(s))]))
k = sum(x["guess"] == x["word"] for x in r); comp["base64_to_word"] = dict(n=len(r), right=k, accuracy=k / len(r), wilson_low=wilson_low(k, len(r)), chance=1 / len(WORDS), candidates=len(WORDS), by_template=[sum(x["guess"] == x["word"] for x in r if x["template"] == i) / sum(1 for x in r if x["template"] == i) for i in range(2)])
print("base64 -> word:", comp["base64_to_word"]["accuracy"], comp["base64_to_word"]["by_template"], f"({time.time() - t0:.0f}s)", flush=True)
B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
r = []
for i, X in enumerate(B64[:62]):
    g = greedy(f"In the Base64 alphabet, the character '{X}' has the index", 4); num = "".join(ch for ch in g if ch.isdigit())[:2]
    r.append(dict(symbol=X, index=i, answer=g, right=num == str(i)))
lowdig = [x for x in r if not x["symbol"].isupper()]; up = [x for x in r if x["symbol"].isupper()]
comp["base64_index"] = dict(lowercase_and_digits=dict(n=len(lowdig), right=sum(x["right"] for x in lowdig), accuracy=sum(x["right"] for x in lowdig) / len(lowdig), wilson_low=wilson_low(sum(x["right"] for x in lowdig), len(lowdig))), uppercase=dict(n=len(up), accuracy=sum(x["right"] for x in up) / len(up)), chance=1 / 64, answers=r)
print("base64 index:", comp["base64_index"]["lowercase_and_digits"]["accuracy"], comp["base64_index"]["uppercase"]["accuracy"], f"({time.time() - t0:.0f}s)", flush=True)
r = []
for v in range(64):
    b = format(v, "06b"); g = greedy(f"The binary number {b} in decimal is", 4); num = "".join(ch for ch in g if ch.isdigit())[:2]; r.append(dict(value=v, answer=g, right=num == str(v)))
comp["binary_to_decimal"] = dict(n=64, right=sum(x["right"] for x in r), accuracy=sum(x["right"] for x in r) / 64, wilson_low=wilson_low(sum(x["right"] for x in r), 64), chance=1 / 64, answers=r)
print("binary -> decimal:", comp["binary_to_decimal"]["accuracy"], f"({time.time() - t0:.0f}s)", flush=True)
res["competence"] = comp; json.dump(res, open(out, "w"), indent=1)

# ---------------- T4-A Braille bits with matched tokens
def matched(k):
    a, b, c = enc(cell(k)), enc(cell(k, 0x2840)), enc(cell(k, 0x2A00))
    return len(a) == len(b) == len(c) == 2 and a[1] == b[1] == c[1] and len({a[0], b[0], c[0]}) == 3
KS = [k for k in range(64) if matched(k)]; res["matched_values_of_k"] = KS
FR = [lambda X: f"Symbol\n{X}\n", lambda X: f'Symbol "{X}"', lambda X: f"Symbol ({X})"]
rng = random.Random(0)


def collect(base):
    """states [n_cells, n_contexts, layer] -> dict layer -> array [cells * contexts, D]; labels [cells * contexts] = k; missing contexts skipped."""
    X, K = {l: [] for l in LAYERS}, []
    for k in KS:
        ids = enc(cell(k, base)); ctx = [f(cell(k, base)) for f in FR]
        for _ in range(2): others = [cell(rng.choice(KS), base) for _ in range(7)]; ctx.append("".join(others[:4]) + cell(k, base) + "".join(others[4:]))
        for t in ctx:
            st = states(t, ids)
            if st is None: continue
            for l in LAYERS: X[l].append(st[l])
            K.append(k)
    return {l: np.array(v) for l, v in X.items()}, np.array(K)


def probe_acc(Xtr, ytr, Xte, yte, kind):
    if kind == "dom": w = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0); b = -(Xtr[ytr == 1].mean(0) + Xtr[ytr == 0].mean(0)) @ w / 2
    else:
        mu = Xtr.mean(0); A = Xtr - mu; G = A @ A.T; lam = 1e-2 * np.trace(G) / len(G); alpha = np.linalg.solve(G + lam * np.eye(len(G)), (2 * ytr - 1).astype(float)); w = A.T @ alpha; b = -mu @ w
    return float((((Xte @ w + b) > 0).astype(int) == yte).mean())


def _prep(Xtr, Xte, kind):
    """The label-independent part of a probe fitted on Xtr and read on Xte (same probes as probe_acc, for all 6 bits at once)."""
    if kind == "dom": return Xte @ Xtr.T, Xtr @ Xtr.T
    mu = Xtr.mean(0); A = Xtr - mu; G = A @ A.T; lam = 1e-2 * np.trace(G) / len(G)
    return np.linalg.solve(G + lam * np.eye(len(G)), A @ (Xte - mu).T).T          # P [n_te, n_tr]: ridge score = P @ (2y - 1)


def _score(pre, Y, kind):
    """Scores [n_te, 6] for the training labels Y [n_tr, 6]. dom: x.(m1 - m0) - (|m1|^2 - |m0|^2) / 2 with m_c = X_tr^T W_c."""
    if kind != "dom": return pre @ (2 * Y - 1)
    C, G = pre; n1 = Y.sum(0); W1 = Y / np.maximum(n1, 1); W0 = (1 - Y) / np.maximum(len(Y) - n1, 1)
    return C @ (W1 - W0) - 0.5 * (np.einsum("ib,ij,jb->b", W1, G, W1) - np.einsum("ib,ij,jb->b", W0, G, W0))


def split_probe(X, K, bits_of, n_splits=200, n_perm=200, seed=0, perm_groups=None, kind="dom"):
    """Held-out accuracy per bit, averaged over n_splits random halvings of the cells (fit on one half, read on the other).
    Null: the same statistic, averaged over the same splits, for n_perm relabellings of the cells (within perm_groups if given).
    Also the held-out accuracy per cell on bits 0-2, averaged over the splits in which the cell is held out."""
    cells = np.unique(K); g = np.random.default_rng(seed)
    S = [np.isin(K, g.permutation(cells)[: len(cells) // 2]) for _ in range(n_splits)]; pre = [_prep(X[tr], X[~tr], kind) for tr in S]
    def stat(labmap, keep=False):
        Y = np.array([bits_of(labmap[k]) for k in K], dtype=float); A = np.full((n_splits, 6), np.nan); C, N = np.zeros(len(K)), np.zeros(len(K))
        for i, tr in enumerate(S):
            m = Y[tr].mean(0); ok = (m > 0) & (m < 1); corr = (_score(pre[i], Y[tr], kind) > 0) == (Y[~tr] > 0.5); A[i, ok] = corr.mean(0)[ok]
            if keep: te = np.flatnonzero(~tr); C[te] += corr[:, :3].mean(1); N[te] += 1
        return A, C, N
    A, C, N = stat({k: k for k in cells}, keep=True); null = np.zeros((n_perm, 6)); nullm = np.zeros(n_perm)
    for j in range(n_perm):
        if perm_groups is None: pm = dict(zip(cells, g.permutation(cells)))
        else: pm = {}; [pm.update(dict(zip(grp, g.permutation(grp)))) for grp in perm_groups]
        Aj = stat(pm)[0]; null[j] = np.nanmean(Aj, 0); nullm[j] = np.nanmean(Aj)
    return dict(acc=np.nanmean(A, 0).tolist(), null_p99=np.nanpercentile(null, 99, axis=0).tolist(), null_mean=np.nanmean(null, 0).tolist(), mean_acc=float(np.nanmean(A)),
                mean_null_p99=float(np.nanpercentile(nullm, 99)), n_splits=n_splits, n_perm=n_perm, percell_bits012={str(int(k)): float(C[K == k].sum() / N[K == k].sum()) for k in cells})


def boot_diff(pa, pb, n=1000, seed=7):
    """Mean over the cells of pa - pb (per-cell accuracies on bits 0-2) and its 95 % interval from n bootstrap resamples of the cells."""
    ks = sorted(set(pa) & set(pb)); d = np.array([pa[k] - pb[k] for k in ks]); g = np.random.default_rng(seed)
    bs = [d[g.integers(0, len(d), len(d))].mean() for _ in range(n)]; return dict(diff=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), cells=len(ks))


bits = lambda k: [(k >> j) & 1 for j in range(6)]
Xb, Kb = collect(0x2800); X7, K7 = collect(0x2840); Xm, Km = collect(0x2A00); print(f"collected Braille states: {len(Kb)}, {len(K7)}, {len(Km)} ({time.time() - t0:.0f}s)", flush=True)
T4A = {}
for l in LAYERS:
    row = dict(braille=split_probe(Xb[l], Kb, bits, seed=1), braille_ridge=split_probe(Xb[l], Kb, bits, seed=4, kind="ridge"),
               math_ops=split_probe(Xm[l], Km, bits, seed=2), math_ops_ridge=split_probe(Xm[l], Km, bits, seed=6, kind="ridge"))
    row["braille_minus_math_ops_boot"] = boot_diff(row["braille"]["percell_bits012"], row["math_ops"]["percell_bits012"])
    row["braille_minus_math_ops_boot_ridge"] = boot_diff(row["braille_ridge"]["percell_bits012"], row["math_ops_ridge"]["percell_bits012"])
    tr, trr = {}, {}
    for name, Xt, Kt in (("to_dot7", X7[l], K7), ("to_math_ops", Xm[l], Km)):
        a, ar = [], []
        for j in range(6):
            y = np.array([bits(k)[j] for k in Kb]); yt = np.array([bits(k)[j] for k in Kt]); a.append(probe_acc(Xb[l], y, Xt, yt, "dom")); ar.append(probe_acc(Xb[l], y, Xt, yt, "ridge"))
        tr[name], trr[name] = a, ar
    row["transfer"], row["transfer_ridge"] = tr, trr
    # decade structure: offsets a-j -> k-t (byte + 4) for the usable letters, against the same byte offsets in U+2A00
    pairs = [(BRAILLE[c1], BRAILLE[c2]) for c1, c2 in zip("abcdefghij", "klmnopqrst") if BRAILLE[c1] in KS and BRAILLE[c2] in KS and c1 in letters and c2 in letters]
    def consist(X, K):
        m = {k: X[K == k].mean(0) for k in np.unique(K)}; off = [m[b] - m[a] for a, b in pairs if a in m and b in m]
        off = [o / np.linalg.norm(o) for o in off]; return float(np.mean([off[i] @ off[j] for i in range(len(off)) for j in range(i + 1, len(off))])) if len(off) > 1 else None
    row["decade"] = dict(pairs=len(pairs), braille=consist(Xb[l], Kb), math_ops=consist(Xm[l], Km))
    T4A[str(l)] = row
    print(f"T4-A layer {l}: bits 0-2 Braille {np.round(row['braille']['acc'][:3], 2)} (null p99 {np.round(row['braille']['null_p99'][:3], 2)}) vs U+2A00 {np.round(row['math_ops']['acc'][:3], 2)} | bits 3-5 {np.round(row['braille']['acc'][3:], 2)} | transfer dot-7 {np.round(tr['to_dot7'][:3], 2)} U+2A00 {np.round(tr['to_math_ops'][:3], 2)} | decade {row['decade']}", flush=True)
res["T4A"] = T4A; json.dump(res, open(out, "w"), indent=1)

# ---------------- T4-B Base64 index bits
FRB = [lambda X: f"Symbol\n{X}\n", lambda X: f'Symbol "{X}"', lambda X: f"Symbol ({X})"]
X64, K64 = {l: [] for l in LAYERS}, []
for i, s in enumerate(B64[:62]):
    for f in FRB:
        ids = enc(s)
        if len(ids) != 1: continue
        st = states(f(s), ids)
        if st is None: continue
        for l in LAYERS: X64[l].append(st[l])
        K64.append(i)
X64 = {l: np.array(v) for l, v in X64.items()}; K64 = np.array(K64); groups = [list(range(0, 26)), list(range(26, 52)), list(range(52, 62))]
T4B = {}
for l in LAYERS:
    row = split_probe(X64[l], K64, bits, seed=3, perm_groups=groups); row["ridge"] = split_probe(X64[l], K64, bits, seed=5, perm_groups=groups, kind="ridge"); T4B[str(l)] = row
    print(f"T4-B layer {l}: index bits {np.round(row['acc'], 2)} mean {row['mean_acc']:.3f} | within-class null p99 of the mean {row['mean_null_p99']:.3f}", flush=True)
res["T4B"] = dict(states=int(len(K64)), layers=T4B); res["seconds"] = time.time() - t0
f32 = lambda D: {str(l): v.astype(np.float32) for l, v in D.items()}
np.savez_compressed(out + ".states.npz", **{f"{n}_{l}": v for n, D in (("braille", Xb), ("dot7", X7), ("math_ops", Xm), ("base64", X64)) for l, v in f32(D).items()}, K_braille=Kb, K_dot7=K7, K_math_ops=Km, K_base64=K64); json.dump(res, open(out, "w"), indent=1); print(f"done ({time.time() - t0:.0f}s)", flush=True)

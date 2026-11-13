"""Round 2 of README sections 11/12, Part A (RGI geometry of Qwen; frozen plan results/ju_space/plans/abschnitt11_12_runde2_plan.md,
sections 2.4 O7 / O8, 3.1 A1 / A3 / A4, 9.2; Appendix A): the model passes of session r2_q3, for Qwen3-1.7B and the Qwen3-0.6B arm.

    python r2_rgi_geom.py <root> <r1_dir> <r2_dir> [<stage>] [--model qwen3-1.7b | qwen3-0.6b] [--model-path <dir>]
    MOCK=1 python r2_rgi_geom.py <root> <r1_dir> <r2_dir> selftest [--model ...]     code test on rk_loop's tiny random model
    python r2_rgi_geom.py export_pool_ref <ufo_pool.pt> <pool_ref.json>               OD-G7 reference from the pre-round-1 pool file

<root> as in round 1 (models/<model>, ju_out/ for the pool texts, dn_out/mmlu_topics.json for O7); <r1_dir> is not read (r2_q3
copies no round-1 file, 7.3); <r2_dir> = the session's work folder. --model-path overrides <root>/models/<model> (the runner downloads
and checksums the pinned Qwen3-0.6B revision; this script records the pin and the safetensors sha256, OD-A6). Env: DEVICE (cpu |
cuda; float32, TF32 off on CUDA), NT (CPU threads, default 3), MOCK (1 = rk_loop's tiny random model: code test only).

Stages (in this order for 'all', the default; each is skipped when its output exists and carries this run's fingerprint):
  a1    A1 states. The 2,040 sentences of Appendix A (r2_stimuli.sentences(); G29 asserts first: r2_stimuli.assert_a1 plus, for every F2
        sentence, no merge at the seam of the chat prefix). F1 and neutral-F1 sentences are read as plain text; F2 and neutral-F2
        sentences as the assistant's reply to 'Say one sentence.' (chat template, thinking off; the prefix is tokenised once and the ids
        are concatenated, as rk_loop.reply_read reads a reply). One unstopped forward of model.model per batch of equal-length sentences
        with forward hooks on every block: the raw block output at '.' (primary reading) and averaged over the 6 object tokens
        (secondary reading); the mean input embedding over the sentence's own tokens (lexical reference).  -> local/geom_<m>.npz
  a3    A3 gradients. Per object (60) the chat prompt 'In one sentence: what would you do with {x}?' + 'I would' (r2_stimuli.a3_input);
        one zero leaf vector per block added to the block output at the 6 object positions (plain addition, OD-A3); four scalars at the
        last position: y_h = mean logit of the IMP lexemes of half h minus the DES lexemes of half h, and the same for CHG minus DES
        (base forms with a leading space, single tokens, OD-A3; half 1 = lexemes 1-2, half 2 = lexemes 3-4 of Appendix A's lists);
        one forward and 4 backward passes give W [60, 4, 28, D] (rows: IMP-DES h1, IMP-DES h2, CHG-DES h1, CHG-DES h2).
                                                                                                    -> rgi/causal_<m>.pt
  o7    O7 delta states. dn_delta.py's text selection verbatim (dn_out/mmlu_topics.json, NMAX 64, LQ 16, MINQ 20, default_rng(0));
        read at all blocks with forward hooks (last token; mean over the LQ question tokens), stored float16 per text; F, floor and
        delta per block and reading exactly as dn_delta.py computes them (its fisher, float64; the 50 permutations per block and reading
        from default_rng(2) in dn_delta's order: block 0 last, block 0 mean, block 1 last, ...).        -> local/delta_states_<m>.npz
  o8    O8 pool covariance. ufo_common.texts() pool (192 texts), ufo_pool.py's definition: median norm per block over the valid
        positions of every len(pool) // 12-th text (torch.median as ufo_pool); fit part = texts i with i mod 3 != 2; states at valid
        positions with norm < 10 x median; mean and covariance per block in float64; eigendecomposition (float64, descending, clamped
        at 0). OD-G7 for Qwen3-1.7B: block-14 top-5 eigenvalues within 5 % of pool_ref.json (flag, not fatal).  -> local/pool_<m>.pt
  dirs  Directions for A6 and A3 (from the A1 states, '.' reading, raw metric): per block the 48 model-object IMP-DES contrasts c_o
        (frames pooled, all lexemes), g_b = mean_o c_o, d_SELF = mean over the 12 SELF objects, the SELF residual s_b = d_SELF minus
        its projection on g_b, and g_SELF = the F2 SELF contrast of IMP-DES (A3 spillover).              -> rgi/dirs_<m>.pt
  meta  Summary of every stage, flags, versions, the Qwen3-0.6B pin, sha256 of every output.      -> rgi/geom_meta_<m>.json
The local files stay on the session machine (7.3, 9.5); rgi/ is synced. r2_rgi_stats.py reads all of them."""
import argparse, ast, contextlib, datetime, hashlib, json, math, os, platform, shutil, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); JU = os.path.dirname(HERE)
for _p in (HERE, JU):
    if _p not in sys.path: sys.path.insert(0, _p)
import r2_stimuli as STIM                                                                  # frozen Appendix A lists and G29 asserts

MOCK = os.environ.get("MOCK") == "1"
VERSION = "r2_rgi_geom 1.0 (plan sha256 5e476340dfdb1d82)"
PLAN_SHA256 = "5e476340dfdb1d827582c869f2dfc08e32ed8e9a9dbebf7642acceb5bb5edc2d"
MODELS = ("qwen3-1.7b", "qwen3-0.6b")
QWEN06_PIN = dict(repo="Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca",       # OD-A6 (HF API, 12 Nov 2026;
                  sha256={"model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",  # same pin as
                          "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"})   # r2_run_session.py)
O7_NMAX, O7_LQ, O7_MINQ, O7_NPERM = 64, 16, 20, 50                                         # dn_delta.py's defaults (O7)
POOL_REF_BLOCK, POOL_REF_TOL = 14, 0.05                                                     # OD-G7
F16_MAX = 65504.0
STAGES = ("a1", "a3", "o7", "o8", "dirs", "meta")
A3_HALVES = ((0, 1), (2, 3))                                                                # lexeme halves of the A3 scalars
A3_SCALARS = ("IMP-DES h1", "IMP-DES h2", "CHG-DES h1", "CHG-DES h2")


def log(msg):
    print(f"[r2_rgi_geom {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ======================================================================================================================================
# Layout of the A1 stimuli (shared with r2_rgi_stats.py)
# ======================================================================================================================================
FAMS = tuple(STIM.FAMILIES)                                                                 # IMP CHG DES WOR PRA
FAM_CODE = {F: i for i, F in enumerate(FAMS)}; FAM_CODE["NEU"] = len(FAMS)                  # 5 = neutral 'Topic: {x}.'
NLEX = {F: len(STIM.VERBS[F][0]) for F in FAMS}                                             # 4 4 4 2 2
VF_OFF = {}; _o = 0
for _F in FAMS: VF_OFF[_F] = _o; _o += NLEX[_F]
NVF = _o                                                                                     # 16 verb forms (family, lexeme)
OBJECTS = STIM.objects(); NOBJ = len(OBJECTS)                                                # 60: 48 model (owner x substrate x paraphrase), 12 non-model
OWNERS = tuple(STIM.OWNER_ORDER); SUBS = ("W", "KV", "LR"); NONMODEL = ("TOOL", "PROCESS", "SKILL")
CELL_NAMES = tuple(f"{o}|{s}" for o in OWNERS for s in SUBS) + NONMODEL                      # cell c = 3 owner + substrate (0..11), 12..14
NCELL, NMODEL_CELL = len(CELL_NAMES), 12
CELL_OF = np.array([CELL_NAMES.index(ob["cell"]) for ob in OBJECTS]); PARA_OF = np.array([ob["para"] for ob in OBJECTS])
assert all(CELL_OF[o] == o // 4 and PARA_OF[o] == o % 4 for o in range(NOBJ)), "object order of r2_stimuli.objects() changed"
SELF_OBJ = [o for o in range(48) if OBJECTS[o]["owner"] == "SELF"]; OTHER_OBJ = [o for o in range(48) if OBJECTS[o]["owner"] != "SELF"]


def sentence_layout(s_obj, s_frame, s_family, s_lexeme):
    """Index arrays of the A1 rows: unit[f, v, o] = row of frame f (0 F1, 1 F2), verb form v (VF_OFF[F] + lexeme) and object o;
    neu[f, o] = row of the neutral sentence. Asserts that every combination occurs exactly once."""
    unit = np.full((2, NVF, NOBJ), -1, dtype=np.int64); neu = np.full((2, NOBJ), -1, dtype=np.int64)
    for j in range(len(s_obj)):
        f, fam, o = int(s_frame[j]), int(s_family[j]), int(s_obj[j])
        if fam == FAM_CODE["NEU"]:
            assert neu[f, o] < 0; neu[f, o] = j
        else:
            v = VF_OFF[FAMS[fam]] + int(s_lexeme[j]); assert unit[f, v, o] < 0; unit[f, v, o] = j
    assert (unit >= 0).all() and (neu >= 0).all(), "A1 rows incomplete"
    return unit, neu


def directions(dot, unit):
    """A2/A6/A3 directions from the '.' states dot [N, NL, D] (raw metric). Per block: c_o [48, D] = mean over the 4 IMP lexemes minus
    mean over the 4 DES lexemes of the frame-pooled states of model object o; g = mean_o c_o; d_SELF = mean over the SELF objects;
    s = d_SELF - (d_SELF . unit(g)) unit(g) (the SELF residual); g_SELF = the same contrast as d_SELF in frame F2 alone. float64."""
    X = torch.as_tensor(dot)
    imp = [VF_OFF["IMP"] + l for l in range(NLEX["IMP"])]; des = [VF_OFF["DES"] + l for l in range(NLEX["DES"])]

    def contrast(f):                                                                         # [48, NL, D]
        a = torch.stack([X[unit[f, v, :48]] for v in imp]).double().mean(0); b = torch.stack([X[unit[f, v, :48]] for v in des]).double().mean(0)
        return a - b
    c1, c2 = contrast(0), contrast(1); c = (c1 + c2) / 2
    g = c.mean(0); dS = c[SELF_OBJ].mean(0); gn = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    s = dS - (dS * gn).sum(-1, keepdim=True) * gn; gS2 = c2[SELF_OBJ].mean(0)
    return dict(c_o=c.transpose(0, 1).contiguous(), g=g, d_self=dS, s=s, g_self_f2=gS2)    # c_o [NL, 48, D]; others [NL, D]


# ======================================================================================================================================
# Files
# ======================================================================================================================================
def out_paths(r2_dir, m):
    return dict(geom=os.path.join(r2_dir, "local", f"geom_{m}.npz"), delta=os.path.join(r2_dir, "local", f"delta_states_{m}.npz"),
                pool=os.path.join(r2_dir, "local", f"pool_{m}.pt"), causal=os.path.join(r2_dir, "rgi", f"causal_{m}.pt"),
                dirs=os.path.join(r2_dir, "rgi", f"dirs_{m}.pt"), meta=os.path.join(r2_dir, "rgi", f"geom_meta_{m}.json"))


STAGE_OUT = dict(a1="geom", a3="causal", o7="delta", o8="pool", dirs="dirs", meta="meta")


def jsonable(x):
    if torch.is_tensor(x): return jsonable(x.detach().cpu().tolist())
    if isinstance(x, np.ndarray): return jsonable(x.tolist())
    if isinstance(x, np.bool_): return bool(x)
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, dict): return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    return x


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True); tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(jsonable(obj), f, indent=1)
    os.replace(tmp, path)


def torch_save(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True); tmp = path + ".tmp"; torch.save(obj, tmp); os.replace(tmp, path)


def npz_save(path, meta, **arrays):
    """Atomic np.savez (uncompressed) with a JSON string under '_meta' (loadable with allow_pickle=False)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True); tmp = path + ".tmp"
    with open(tmp, "wb") as f: np.savez(f, _meta=np.array(json.dumps(jsonable(meta))), **arrays)
    os.replace(tmp, path)


def npz_meta(path):
    with np.load(path, allow_pickle=False) as z: return json.loads(str(z["_meta"]))


_SHA_MEMO = {}


def sha256_file(path, chunk=1 << 22):
    """sha256 of a file; memoised per process by (path, inode, size, mtime_ns), so a large file is hashed once unless it changes (every
    output here is written by os.replace, which gives the file a new inode)."""
    st = os.stat(path); key = (os.path.abspath(path), st.st_ino, st.st_size, st.st_mtime_ns)
    if key in _SHA_MEMO: return _SHA_MEMO[key]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    _SHA_MEMO[key] = h.hexdigest(); return _SHA_MEMO[key]


def load_pt(path):
    """torch.load of a stage output (memory-mapped: only the parts that are used are read)."""
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def file_meta(path):
    """Fingerprint stored in a stage output (a .json, .npz or .pt), or None."""
    if not os.path.exists(path): return None
    if path.endswith(".npz"): return npz_meta(path).get("fingerprint")
    if path.endswith(".json"): return json.load(open(path)).get("fingerprint")
    return load_pt(path).get("fingerprint")


# ======================================================================================================================================
# Model
# ======================================================================================================================================
class Model:
    """Tokenizer and model of one run (float32, eval, no parameter gradients). Real: AutoModelForCausalLM.from_pretrained(path,
    dtype=float32) exactly as ufo_common.load_model, TF32 off on CUDA. MOCK: rk_loop's tiny random model (the tokenizer and config of
    <root>/models/qwen3-1.7b with 28 blocks of width 64; torch.manual_seed(0), seed 1 for the Qwen3-0.6B arm)."""
    def __init__(self, root, m, model_path=None):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        self.name, self.root = m, root; self.path = model_path or os.path.join(root, "models", m); dev = os.environ.get("DEVICE", "cpu")
        if dev != "cpu":
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("highest")
        if MOCK:
            tp = self.path if os.path.isfile(os.path.join(self.path, "tokenizer_config.json")) else os.path.join(root, "models", "qwen3-1.7b")
            self.tok = AutoTokenizer.from_pretrained(tp); cfg = AutoConfig.from_pretrained(tp)
            cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 28, 64, 128, 4, 2, 16
            if getattr(cfg, "layer_types", None): cfg.layer_types = cfg.layer_types[:28]
            torch.manual_seed(0 if m == "qwen3-1.7b" else 1); self.model = AutoModelForCausalLM.from_config(cfg).float().eval(); self.tok_path = tp
        else:
            if dev == "cpu":
                mem = int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) / 1e6; need = 9.5 if m == "qwen3-1.7b" else 4.0
                assert mem >= need, f"only {mem:.1f} GB available for {m} on the CPU"
            self.tok = AutoTokenizer.from_pretrained(self.path); self.tok_path = self.path
            self.model = AutoModelForCausalLM.from_pretrained(self.path, dtype=torch.float32).eval()
        if dev != "cpu": self.model = self.model.to(dev)
        self.model.requires_grad_(False); self.dev = next(self.model.parameters()).device
        self.layers = self.model.model.layers; self.NL = len(self.layers); self.D = int(self.model.config.hidden_size)
        assert self.NL == 28, f"{m}: {self.NL} blocks (the plan assumes 28)"

    def enc(self, t): return self.tok(t, add_special_tokens=False)["input_ids"]

    def chat(self, q): return self.tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def info(self):
        return dict(name=self.name, path=self.path, tokenizer=self.tok_path, mock=MOCK, n_blocks=self.NL, hidden_size=self.D,
                    vocab_size=int(self.model.config.vocab_size), tie_word_embeddings=bool(getattr(self.model.config, "tie_word_embeddings", False)),
                    device=str(self.dev), dtype=str(next(self.model.parameters()).dtype))


class Taps:
    """Forward hooks on the outputs of all blocks (decoder layers): out[l] = the block's raw output [B, T, D] of the last forward."""
    def __init__(self, layers): self.layers, self.out, self.h = layers, {}, []

    def __enter__(self):
        def mk(l):
            def f(mod, inp, o): self.out[l] = (o[0] if isinstance(o, tuple) else o).detach()
            return f
        self.h = [L.register_forward_hook(mk(l)) for l, L in enumerate(self.layers)]; return self

    def __exit__(self, *a):
        for h in self.h: h.remove()
        self.h = []

    def stack(self): return torch.stack([self.out[l] for l in range(len(self.layers))], 1)          # [B, NL, T, D]


FP_KEYS = ("version", "model", "mock", "n_blocks", "hidden_size")                       # compared on resume (not the path: a continuation
                                                                                           # session may run under another root)
def fingerprint(M, m):
    return dict(version=VERSION, model=m, mock=MOCK, n_blocks=M.NL, hidden_size=M.D, path=os.path.abspath(M.path))


def same_run(have, want):
    return have is not None and all(have.get(k) == want.get(k) for k in FP_KEYS if k in want)


def equal_length_batches(lengths, batch):
    """Indices grouped by length (order kept inside a group), in chunks of at most 'batch'."""
    groups = {}
    for j, L in enumerate(lengths): groups.setdefault(L, []).append(j)
    for L in sorted(groups):
        g = groups[L]
        for i in range(0, len(g), batch): yield L, g[i:i + batch]


# ======================================================================================================================================
# A1 states
# ======================================================================================================================================
def a1_rows(M):
    """The 2,040 A1 rows with their ids (F2: chat prefix + reply ids), the offset of the sentence, the absolute positions of the 6
    object tokens and of '.', after the G29 asserts (a failing assert stops the session)."""
    S = STIM.sentences(); O = STIM.objects(); rep = STIM.assert_a1(M.enc, M.chat); base_str = M.chat(STIM.F2_PROMPT); base = M.enc(base_str)
    rows = []
    for s in S:
        x = O[s["obj"]]["text"][s["frame"]]; r = STIM.encode_frame(M.enc, s["template"], x, s["verb"])
        if s["frame"] == "F2":
            assert M.enc(base_str + s["text"]) == base + r["ids"], f"F2 reply {s['text']!r} merges with the chat prefix (G29)"
            pre = base
        else:
            pre = []
        off = len(pre)
        rows.append(dict(ids=pre + r["ids"], off=off, n=len(r["ids"]), obj=[off + q for q in r["obj_pos"]], last=off + r["last"], text=s["text"],
                         obj_i=s["obj"], frame=0 if s["frame"] == "F1" else 1, family=FAM_CODE[s["family"]], lexeme=-1 if s["lexeme"] is None else s["lexeme"]))
    rep["f2_prefix"] = base; rep["n_rows"] = len(rows)
    return rows, rep


def stage_a1(M, P, m, batch=None):
    t0 = time.time(); rows, rep = a1_rows(M); N, NL, D = len(rows), M.NL, M.D; batch = batch or (64 if M.dev.type == "cuda" else 32)
    dot = np.zeros((N, NL, D), np.float32); obj = np.zeros((N, NL, D), np.float32); lex = np.zeros((N, D), np.float32)
    for L, idx in equal_length_batches([len(r["ids"]) for r in rows], batch):
        ids = torch.tensor([rows[j]["ids"] for j in idx], device=M.dev); B = len(idx); ar = torch.arange(B, device=M.dev)
        last = torch.tensor([rows[j]["last"] for j in idx], device=M.dev); op = torch.tensor([rows[j]["obj"] for j in idx], device=M.dev)
        with torch.no_grad(), Taps(M.layers) as tp:
            emb = M.model.model.embed_tokens(ids); M.model.model(input_ids=ids); H = tp.stack()                 # [B, NL, T, D]
        dot[idx] = H[ar, :, last].float().cpu().numpy()
        obj[idx] = H[ar[:, None, None], torch.arange(NL, device=M.dev)[None, :, None], op[:, None, :]].double().mean(2).float().cpu().numpy()
        for r, j in enumerate(idx):
            o, n = rows[j]["off"], rows[j]["n"]; lex[j] = emb[r, o:o + n].double().mean(0).float().cpu().numpy()
    assert np.isfinite(dot).all() and np.isfinite(obj).all() and np.isfinite(lex).all(), "A1 states not finite"
    T = max(len(r["ids"]) for r in rows); ids = np.full((N, T), -1, np.int32)
    for j, r in enumerate(rows): ids[j, :len(r["ids"])] = r["ids"]
    summ = dict(n_sentences=N, n_verb=int(sum(r["family"] != FAM_CODE["NEU"] for r in rows)), n_neutral=int(sum(r["family"] == FAM_CODE["NEU"] for r in rows)),
                lengths={str(k): int(v) for k, v in zip(*np.unique([len(r["ids"]) for r in rows], return_counts=True))},
                f2_prefix_tokens=len(rep["f2_prefix"]), g29=rep, readings=dict(dot="raw block output at '.' (last token)",
                obj="mean of the raw block output over the 6 object tokens", lex="mean input embedding over the sentence's own tokens"),
                seconds=time.time() - t0, batch=batch)
    npz_save(P["geom"], dict(fingerprint=fingerprint(M, m), summary=summ), dot=dot, obj=obj, lex=lex,
             s_obj=np.array([r["obj_i"] for r in rows], np.int16), s_frame=np.array([r["frame"] for r in rows], np.int8),
             s_family=np.array([r["family"] for r in rows], np.int8), s_lexeme=np.array([r["lexeme"] for r in rows], np.int8),
             ids=ids, off=np.array([r["off"] for r in rows], np.int16), last=np.array([r["last"] for r in rows], np.int16),
             objpos=np.array([r["obj"] for r in rows], np.int16), texts=np.array([r["text"] for r in rows]))
    log(f"A1: {N} sentences read at {NL} blocks ({summ['seconds']:.1f}s) -> {P['geom']}")
    return summ


# ======================================================================================================================================
# A3 gradients
# ======================================================================================================================================
def a3_inputs(M):
    O = STIM.objects(); LX = STIM.a3_lexemes(); tid = {}
    for F, lx in LX.items():
        ids = [M.enc(v) for v in lx]; assert all(len(i) == 1 for i in ids), f"A3 lexemes of {F} not single tokens (OD-A3)"; tid[F] = [i[0] for i in ids]
    rows = [STIM.a3_input(M.enc, M.chat, ob["text"]["A3"]) for ob in O]
    assert len({len(r["ids"]) for r in rows}) == 1 and len({tuple(r["obj_pos"]) for r in rows}) == 1, "A3 inputs differ in length or object positions"
    return rows, tid


def a3_scalars(logits, tid):
    """[B, 4] = (IMP h1 - DES h1, IMP h2 - DES h2, CHG h1 - DES h1, CHG h2 - DES h2) from next-token logits [B, V]."""
    mean = lambda F, h: logits[:, [tid[F][i] for i in A3_HALVES[h]]].mean(1)
    return torch.stack([mean("IMP", 0) - mean("DES", 0), mean("IMP", 1) - mean("DES", 1), mean("CHG", 0) - mean("DES", 0), mean("CHG", 1) - mean("DES", 1)], 1)


@contextlib.contextmanager
def leaf_hooks(layers, V, mask):
    """Adds V[:, l] (leaf, [B, NL, D]) to the output of block l at the positions where mask [B, T] is 1."""
    def mk(l):
        def f(mod, inp, o):
            x = o[0] if isinstance(o, tuple) else o; x = x + mask[..., None].to(x.dtype) * V[:, l, None, :].to(x.dtype)
            return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
        return f
    hs = [L.register_forward_hook(mk(l)) for l, L in enumerate(layers)]
    try: yield
    finally:
        for h in hs: h.remove()


def a3_grads(M, rows, tid, idx, dtype=torch.float32):
    """Gradients of the four A3 scalars with respect to zero leaf vectors (one per block) at the object positions, for rows idx:
    W [len(idx), 4, NL, D] float64 (cpu), y [len(idx), 4], the 12 lexeme logits [len(idx), 12]."""
    ids = torch.tensor([rows[j]["ids"] for j in idx], device=M.dev); B, T = ids.shape
    mask = torch.zeros(B, T, device=M.dev)
    for r, j in enumerate(idx): mask[r, rows[j]["obj_pos"]] = 1.0
    V = torch.zeros(B, M.NL, M.D, device=M.dev, dtype=dtype, requires_grad=True)
    with torch.enable_grad(), leaf_hooks(M.layers, V, mask):
        h = M.model.model(input_ids=ids).last_hidden_state[:, -1]; logits = M.model.lm_head(h)
        y = a3_scalars(logits, tid); G = [torch.autograd.grad(y[:, s].sum(), V, retain_graph=s < 3)[0].detach().double().cpu() for s in range(4)]
    lx = logits[:, [t for F in ("IMP", "CHG", "DES") for t in tid[F]]].detach().double().cpu()
    return torch.stack(G, 1), y.detach().double().cpu(), lx


def stage_a3(M, P, m, batch=None):
    t0 = time.time(); rows, tid = a3_inputs(M); batch = batch or (20 if M.dev.type == "cuda" else 60); W, Y, LX = [], [], []
    for i in range(0, len(rows), batch):
        idx = list(range(i, min(i + batch, len(rows)))); g, y, lx = a3_grads(M, rows, tid, idx); W.append(g.float()); Y.append(y); LX.append(lx)
    W = torch.cat(W); Y = torch.cat(Y); LX = torch.cat(LX); assert torch.isfinite(W).all(), "A3 gradients not finite"
    summ = dict(n_objects=len(rows), scalars=A3_SCALARS, halves=A3_HALVES, lexemes=STIM.a3_lexemes(), token_ids=tid,
                input_tokens=len(rows[0]["ids"]), obj_pos=rows[0]["obj_pos"], y_mean=Y.mean(0).tolist(), y_sd=Y.std(0).tolist(),
                grad_norm_block_mean=W.double().norm(dim=-1).mean((0, 1)).tolist(), seconds=time.time() - t0, batch=batch)
    torch_save(P["causal"], dict(W=W, y=Y.float(), lexeme_logits=LX.float(), lexeme_order=[f"{F}:{v}" for F in ("IMP", "CHG", "DES") for v in STIM.a3_lexemes()[F]],
                                 scalars=list(A3_SCALARS), objects=[dict(i=o["i"], kind=o["kind"], owner=o["owner"], substrate=o["substrate"],
                                 para=o["para"], cell=o["cell"], text=o["text"]["A3"]) for o in OBJECTS], ids=[r["ids"] for r in rows],
                                 obj_pos=[r["obj_pos"] for r in rows], fingerprint=fingerprint(M, m), summary=summ,
                                 definition="W[o, s, l] = d y_s / d leaf_l; leaf_l = one zero vector per block l added to the output of block l at "
                                            "the 6 object positions of object o's A3 input; y_s = the four A3 scalars (next-token logits after 'I would')"))
    log(f"A3: gradients [{W.shape[0]}, 4, {W.shape[2]}, {W.shape[3]}] ({summ['seconds']:.1f}s) -> {P['causal']}")
    return summ


# ======================================================================================================================================
# O7 delta states (dn_delta.py's selection verbatim; its delta computation exactly)
# ======================================================================================================================================
def o7_texts(root, tok):
    """dn_delta.py lines 29-35, verbatim (NMAX 64, LQ 16, MINQ 20). Returns texts (token-id lists), labels, SUBJ, N, HEAD, TAIL, counts."""
    NMAX, LQ, MINQ = O7_NMAX, O7_LQ, O7_MINQ; data = os.path.join(root, "dn_out", "mmlu_topics.json")
    rows = json.load(open(data)); SUBJ = sorted({r["subject"] for r in rows}); rng = np.random.default_rng(0)
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]; TAIL = enc("\nAnswer:"); HEAD = enc("Question:")
    pools = {s: [q for q in (enc(" " + r["question"].strip()) for r in rows if r["subject"] == s) if len(q) >= MINQ] for s in SUBJ}
    N = min(NMAX, min(len(v) for v in pools.values())); texts, labels = [], []
    for k, s in enumerate(SUBJ):
        for i in rng.permutation(len(pools[s]))[:N]: texts.append(HEAD + pools[s][i][-LQ:] + TAIL); labels.append(k)
    labels = np.array(labels); assert len({len(t) for t in texts}) == 1
    return dict(texts=texts, labels=labels, SUBJ=SUBJ, N=N, HEAD=HEAD, TAIL=TAIL, counts=[len(pools[s]) for s in SUBJ], data=data)


def fisher_np(X, y):
    """dn_delta.py's fisher, verbatim: tr(S_between) / tr(S_within) of the classes y (float64 numpy)."""
    mu = X.mean(0); B = 0.0; W = 0.0
    for k in np.unique(y):
        Xk = X[y == k]; mk = Xk.mean(0); B += len(Xk) * ((mk - mu) ** 2).sum(); W += ((Xk - mk) ** 2).sum()
    return B / W


def fisher_t(X, Y):
    """Batched Fisher ratio, the same quantity as fisher_np: X [..., N, D] (float64), Y one-hot [..., N, K] (float64; broadcastable).
    B = sum_k |S_k|^2 / n_k - |S|^2 / N, total = sum |x|^2 - |S|^2 / N, F = B / (total - B), with S_k = class sums (X centred first)."""
    X = X - X.mean(-2, keepdim=True); S = Y.transpose(-1, -2) @ X; n = Y.sum(-2)
    B = ((S ** 2).sum(-1) / n.clamp_min(1)).sum(-1); T = (X ** 2).sum((-1, -2)); return B / (T - B)


def o7_delta(LAST, MEAN, labels, dev, nperm=O7_NPERM):
    """F, floor (mean over nperm permutations), delta, delta_rel per block and reading, with dn_delta's permutation stream
    (np.random.default_rng(2), 50 permutations for block 0 'last', then block 0 'mean', block 1 'last', ...). LAST, MEAN [N, NL, D]."""
    NL = LAST.shape[1]; gp = np.random.default_rng(2); K = int(labels.max()) + 1; I = np.eye(K)
    out = {nm: dict(F=np.zeros(NL), floor=np.zeros(NL), delta=np.zeros(NL), delta_rel=np.zeros(NL)) for nm in ("last", "mean")}
    Y0 = torch.tensor(I[labels], dtype=torch.float64, device=dev)
    for l in range(NL):
        for nm, A in (("last", LAST), ("mean", MEAN)):
            X = torch.as_tensor(A[:, l], dtype=torch.float64, device=dev); F = float(fisher_t(X, Y0))
            Yp = torch.tensor(np.stack([I[gp.permutation(labels)] for _ in range(nperm)]), dtype=torch.float64, device=dev)
            fl = float(fisher_t(X[None], Yp).mean()); r = out[nm]
            r["F"][l], r["floor"][l], r["delta"][l], r["delta_rel"][l] = F, fl, F - fl, F / fl
    return out


def stage_o7(M, P, m, batch=None):
    t0 = time.time(); sel = o7_texts(M.root, M.tok); texts, labels = sel["texts"], sel["labels"]; N, NL, D = len(texts), M.NL, M.D
    batch = batch or 64; LQ = O7_LQ; qpos = slice(len(sel["HEAD"]), len(sel["HEAD"]) + LQ)
    LAST = np.zeros((N, NL, D), np.float64); MEAN = np.zeros((N, NL, D), np.float64)
    for i in range(0, N, batch):
        ids = torch.tensor(texts[i:i + batch], device=M.dev)
        with torch.no_grad(), Taps(M.layers) as tp:
            M.model.model(input_ids=ids); H = tp.stack()                                                       # [B, NL, T, D]
        LAST[i:i + len(ids)] = H[:, :, -1].double().cpu().numpy(); MEAN[i:i + len(ids)] = H[:, :, qpos].double().mean(2).cpu().numpy()
    assert np.isfinite(LAST).all() and np.isfinite(MEAN).all(), "O7 states not finite"
    dl = o7_delta(LAST, MEAN, labels, M.dev)
    big = max(np.abs(LAST).max(), np.abs(MEAN).max()); f16 = big < F16_MAX; dt = np.float16 if f16 else np.float32
    summ = dict(n_texts=N, n_per_topic=sel["N"], topics=sel["SUBJ"], counts_min_tokens=sel["counts"], text_tokens=len(texts[0]), head=sel["HEAD"],
                tail=sel["TAIL"], question_tokens=LQ, min_question_tokens=O7_MINQ, nmax=O7_NMAX, n_perm=O7_NPERM, data=os.path.relpath(sel["data"], M.root),
                seeds=dict(selection="np.random.default_rng(0) (dn_delta.py)", permutations="np.random.default_rng(2), 50 per block and reading in dn_delta's order"),
                rule="dn_delta.py: last LQ tokens of questions with >= MINQ tokens; identical length; block outputs by forward hooks",
                storage="float16" if f16 else "float32 (float16 would overflow)", max_abs_state=float(big),
                delta={nm: {k: v.tolist() for k, v in r.items()} for nm, r in dl.items()}, seconds=time.time() - t0, batch=batch)
    npz_save(P["delta"], dict(fingerprint=fingerprint(M, m), summary=summ), last=LAST.astype(dt), mean=MEAN.astype(dt), labels=labels.astype(np.int16),
             ids=np.array(texts, np.int32), topics=np.array(sel["SUBJ"]), **{f"{nm}_{k}": v for nm, r in dl.items() for k, v in r.items()})
    log(f"O7: {N} texts ({sel['N']} per topic) read; delta (last) blocks 0/14/27 = {dl['last']['delta'][0]:.4f}/{dl['last']['delta'][min(14, NL - 1)]:.4f}/"
        f"{dl['last']['delta'][-1]:.4f} ({summ['seconds']:.1f}s) -> {P['delta']}")
    return summ


# ======================================================================================================================================
# O8 pool covariance (ufo_pool.py's definition)
# ======================================================================================================================================
def pool_ref_path(root):
    """OD-G7 reference: env POOL_REF, else <root>/ufo_out/pool_ref.json, else analysis/ju_space/round2/pool_ref.json (committed)."""
    for p in (os.environ.get("POOL_REF"), os.path.join(root, "ufo_out", "pool_ref.json"), os.path.join(HERE, "pool_ref.json")):
        if p and os.path.isfile(p): return p
    return None


def od_g7(ev14, ref_path):
    """Block-14 top-5 eigenvalues against pool_ref.json's (relative difference <= 5 % each); a mismatch is a flag, not fatal."""
    if ref_path is None: return dict(checked=False, ok=None, note="pool_ref.json not found: OD-G7 not checked")
    ref = json.load(open(ref_path)); r = np.array(ref["top5_eigenvalues"], dtype=np.float64); e = np.asarray(ev14[:5], dtype=np.float64)
    rel = np.abs(e - r) / np.abs(r); ok = bool((rel <= POOL_REF_TOL).all())
    return dict(checked=True, ok=ok, ref=os.path.abspath(ref_path), ref_source=ref.get("source"), ref_sha256=ref.get("source_sha256"),
                ref_top5=r.tolist(), top5=e.tolist(), rel_diff=rel.tolist(), tolerance=POOL_REF_TOL, flag=None if ok else "O8 pool eigenvalues differ from ufo_pool.pt (OD-G7)")


def stage_o8(M, P, m):
    from ufo_common import texts as ufo_texts
    t0 = time.time(); main, pool = ufo_texts(M.root, M.tok); NL, D = M.NL, M.D

    def fwd(ids):
        with torch.no_grad(), Taps(M.layers) as tp:
            M.model.model(input_ids=ids.to(M.dev)); return [tp.out[l][0] for l in range(NL)]
    nr = {l: [] for l in range(NL)}                                                       # pass 1: median norm per block
    for src, ids, valid in pool[::max(1, len(pool) // 12)]:
        H = fwd(ids); v = valid.to(M.dev)
        for l in range(NL): nr[l].append(H[l][v].norm(dim=1).cpu())
    MED = {l: float(torch.cat(nr[l]).median()) for l in range(NL)}
    n = torch.zeros(NL, dtype=torch.float64); s1 = torch.zeros(NL, D, dtype=torch.float64, device=M.dev); s2 = torch.zeros(NL, D, D, dtype=torch.float64, device=M.dev)
    cnt = {l: dict(wiki=0, chat=0, dropped=0) for l in range(NL)}; nfit = 0
    for i, (src, ids, valid) in enumerate(pool):                                          # pass 2: fit part (i mod 3 != 2)
        if i % 3 == 2: continue
        H = fwd(ids); v = valid.to(M.dev); nfit += 1
        for l in range(NL):
            X = H[l][v]; ok = X.norm(dim=1) < 10 * MED[l]; Xd = X[ok].double(); cnt[l]["dropped"] += int((~ok).sum()); cnt[l][src] += int(ok.sum())
            n[l] += len(Xd); s1[l] += Xd.sum(0); s2[l] += Xd.T @ Xd
    blocks = {}; summ_b = {}
    for l in range(NL):
        mu = s1[l] / n[l]; C = s2[l] / n[l] - torch.outer(mu, mu); ev, U = torch.linalg.eigh(C); ev = ev.flip(0).clamp(min=0); U = U.flip(1)
        blocks[l] = dict(mu=mu.cpu(), ev=ev.cpu(), U=U.float().cpu().contiguous(), n_fit=int(n[l]), med_norm=MED[l], **cnt[l])
        summ_b[l] = dict(n_fit=int(n[l]), med_norm=MED[l], **cnt[l], top5_eigenvalues=ev[:5].cpu().tolist(), trace=float(ev.sum()),
                         mean_eigenvalue=float(ev.mean()), participation_ratio=float(ev.sum() ** 2 / (ev ** 2).sum()))
    if m != "qwen3-1.7b": g7 = dict(checked=False, ok=None, note="not applicable (the reference ufo_pool.pt is of Qwen3-1.7B)")
    elif MOCK and not os.environ.get("POOL_REF"): g7 = dict(checked=False, ok=None, note="not applicable (MOCK model)")
    else: g7 = od_g7(blocks[POOL_REF_BLOCK]["ev"].numpy(), pool_ref_path(M.root))
    summ = dict(n_pool_texts=len(pool), n_fit_texts=nfit, n_wiki=sum(s == "wiki" for s, _, _ in pool), n_chat=sum(s == "chat" for s, _, _ in pool),
                median_texts=len(pool[::max(1, len(pool) // 12)]), blocks=summ_b, od_g7=g7, seconds=time.time() - t0,
                definition="ufo_pool.py: valid positions (ufo_common.texts), median norm from every len(pool)//12-th text, fit part i % 3 != 2, "
                           "norm < 10 x median, mean and covariance in float64, eigh descending, clamped at 0")
    torch_save(P["pool"], dict(blocks=blocks, fingerprint=fingerprint(M, m), summary=summ))
    log(f"O8: {len(pool)} pool texts ({nfit} fit), block-14 top eigenvalue {summ_b[min(14, NL - 1)]['top5_eigenvalues'][0]:.4g}, OD-G7 {g7.get('ok')} ({summ['seconds']:.1f}s) -> {P['pool']}")
    return summ


def export_pool_ref(ufo_pool_pt, out_json):
    """OD-G7 preparation: pool_ref.json from the pre-round-1 ufo_out/ufo_pool.pt ('sd' = sqrt of the fit-part eigenvalues)."""
    z = torch.load(ufo_pool_pt, map_location="cpu", weights_only=True, mmap=True); bl = z["blocks"]
    top = {int(b): (bl[b]["sd"][:5].double() ** 2).tolist() for b in bl}
    ref = dict(source=os.path.abspath(ufo_pool_pt), source_sha256=sha256_file(ufo_pool_pt), block=POOL_REF_BLOCK, top5_eigenvalues=top[POOL_REF_BLOCK],
               top5_eigenvalues_all_blocks=top, note="eigenvalues = sd ** 2 of ufo_pool.pt (fit part of the pool, ufo_pool.py); used by r2_rgi_geom.py o8 (OD-G7)",
               written=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    write_json(out_json, ref); return ref


# ======================================================================================================================================
# Directions and meta
# ======================================================================================================================================
def stage_dirs(P, m, fp):
    t0 = time.time(); src = sha256_file(P["geom"])
    with np.load(P["geom"], allow_pickle=False) as z:
        unit, _ = sentence_layout(z["s_obj"], z["s_frame"], z["s_family"], z["s_lexeme"]); dr = directions(z["dot"], unit)
    out = {k: v.float() for k, v in dr.items()}; out["g_b"], out["s_b"] = out["g"], out["s"]          # the plan's names (shared storage)
    torch_save(P["dirs"], dict(**out, fingerprint=fp, src_geom_sha256=src, model_objects=[OBJECTS[o]["text"]["F1"] for o in range(48)], self_objects=SELF_OBJ,
                               definition=dict(c_o="[NL, 48, D]: per model object, mean over IMP lexemes minus mean over DES lexemes of the frame-pooled "
                                                    "'.' states (raw metric)", g="[NL, D] g_b = mean over the 48 c_o (IMP-DES, frames pooled)",
                                               d_self="[NL, D] mean over the 12 SELF objects of c_o", s="[NL, D] s_b = d_self minus its projection on unit(g)",
                                               g_self_f2="[NL, D] g_SELF: the F2 frame's SELF contrast of IMP-DES (A3 spillover)",
                                               g_b="= g", s_b="= s", nulls="r2_rgi_stats.sign_flip_nulls(c_o[b]) builds A6's sign-flip nulls of g_b and s_b"),
                               seconds=time.time() - t0))
    log(f"dirs: g_b, s_b, g_SELF per block -> {P['dirs']}")
    return dict(seconds=time.time() - t0, norm_g=out["g"].norm(dim=-1).tolist(), norm_s=out["s"].norm(dim=-1).tolist(),
                cos_dself_g=torch.nn.functional.cosine_similarity(out["d_self"].double(), out["g"].double(), dim=-1).tolist())


def versions():
    v = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__)
    try:
        import transformers; v["transformers"] = transformers.__version__
    except Exception: pass
    if torch.cuda.is_available(): v["gpu"] = torch.cuda.get_device_name(0)
    return v


def model_record(root, m, path):
    """Pin (OD-A6) and checksums: for Qwen3-0.6B the sha256 of model.safetensors is computed here and compared with the pin (a mismatch
    stops the arm: G33); for Qwen3-1.7B the runner's records (round-1 checksums.json, models/verified.json) apply."""
    rec = dict(name=m, path=os.path.abspath(path))
    vf = os.path.join(root, "models", "verified.json")
    if os.path.isfile(vf):
        try: rec["verified_json"] = json.load(open(vf)).get(m)
        except Exception as e: rec["verified_json_error"] = str(e)
    if m == "qwen3-0.6b":
        rec["pin"] = QWEN06_PIN
        if not MOCK:
            f = os.path.join(path, "model.safetensors"); got = sha256_file(f) if os.path.isfile(f) else None; rec["sha256_model_safetensors"] = got
            if got != QWEN06_PIN["sha256"]["model.safetensors"]:
                raise SystemExit(f"G33: {f} sha256 {got} differs from the pinned {QWEN06_PIN['sha256']['model.safetensors']} (Qwen3-0.6B arm not run)")
    return rec


def stage_meta(P, m, fp, summaries, minfo, mrec):
    t0 = time.time(); outs = {}
    for k in ("geom", "delta", "pool", "causal", "dirs"):
        outs[k] = dict(path=os.path.relpath(P[k], os.path.dirname(os.path.dirname(P["meta"]))), bytes=os.path.getsize(P[k]), sha256=sha256_file(P[k]),
                       local=k in ("geom", "delta", "pool"))
    flags = []
    o8 = summaries.get("o8") or {}
    if (o8.get("od_g7") or {}).get("flag"): flags.append(o8["od_g7"]["flag"])
    if (o8.get("od_g7") or {}).get("checked") is False and m == "qwen3-1.7b" and not MOCK: flags.append("OD-G7 not checked (pool_ref.json missing)")
    o7 = summaries.get("o7") or {}
    if o7.get("storage", "float16") != "float16": flags.append("O7 states stored as float32 (float16 overflow)")
    meta = dict(fingerprint=fp, plan_sha256=PLAN_SHA256, model=minfo, model_record=mrec, versions=versions(),
                written=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), stages=summaries, outputs=outs, flags=flags,
                mock=MOCK, note="O7 delta here is the round-2 value for the G34 reproduction check (r2_ana); A4/A5 statistics are in geom_stats_<m>.json")
    write_json(P["meta"], meta); log(f"meta -> {P['meta']} ({time.time() - t0:.1f}s)")
    return meta


def dirs_stale(P):
    """rgi_geom-1: dirs is stale when it was computed from another A1 file than the one on disk (also across invocations, e.g. a1
    recomputed by an interrupted earlier call), or when it does not record its source. Without the local A1 file nothing is compared."""
    if not (os.path.exists(P["dirs"]) and os.path.exists(P["geom"])): return False
    return load_pt(P["dirs"]).get("src_geom_sha256") != sha256_file(P["geom"])


def meta_stale(P):
    """geom_meta is stale when a file it certifies exists with another sha256 than the one recorded (files absent here are not compared)."""
    if not os.path.exists(P["meta"]): return False
    outs = json.load(open(P["meta"])).get("outputs") or {}
    return any(os.path.exists(P[k]) and (outs.get(k) or {}).get("sha256") != sha256_file(P[k]) for k in ("geom", "delta", "pool", "causal", "dirs"))


def stage_summary(P, k):
    """The summary dict stored in a stage's output (for the meta stage after a resume)."""
    p = P[STAGE_OUT[k]]
    if p.endswith(".npz"): return npz_meta(p).get("summary")
    z = load_pt(p)
    if k == "dirs": return dict(seconds=z.get("seconds"), norm_g=z["g"].norm(dim=-1).tolist(), norm_s=z["s"].norm(dim=-1).tolist())
    return z.get("summary")


def run(root, r2_dir, m="qwen3-1.7b", stages=STAGES, model_path=None, force=(), mrec=None):
    """Runs the given stages (skipping those whose output exists with this run's fingerprint); returns the meta dict or None.
    mrec: model_record(...) if already computed (the Qwen3-0.6B checksum is computed once)."""
    P = out_paths(r2_dir, m); M = None; fp = None; summaries = {}; path = model_path or os.path.join(root, "models", m); redone = set()

    def model():
        nonlocal M, fp
        if M is None:
            t0 = time.time(); M = Model(root, m, model_path); fp = fingerprint(M, m); log(f"{m} loaded ({time.time() - t0:.1f}s): {M.info()}")
        return M
    for k in stages:
        p = P[STAGE_OUT[k]]
        stale = (k == "dirs" and ("a1" in redone or dirs_stale(P))) or (k == "meta" and (bool(redone) or meta_stale(P)))   # derived outputs follow their inputs
        if (k in force or stale) and os.path.exists(p): os.remove(p); log(f"{k}: recomputed ({'forced' if k in force else 'its inputs were recomputed'})")
        if os.path.exists(p):
            have = file_meta(p); want = fp or dict(version=VERSION, model=m, mock=MOCK)    # without loading the model where possible
            if not same_run(have, want): raise RuntimeError(f"{p} was written by another run ({have}); delete it to recompute")
            if k != "meta": summaries[k] = stage_summary(P, k); log(f"{k}: output exists, skipped ({p})")
            else: log(f"meta exists, skipped ({p})")
            continue
        redone.add(k)
        if k == "a1": summaries[k] = stage_a1(model(), P, m)
        elif k == "a3": summaries[k] = stage_a3(model(), P, m)
        elif k == "o7": summaries[k] = stage_o7(model(), P, m)
        elif k == "o8": summaries[k] = stage_o8(model(), P, m)
        elif k == "dirs":
            assert os.path.exists(P["geom"]), "dirs needs the A1 states (stage a1)"; summaries[k] = stage_dirs(P, m, fp or file_meta(P["geom"]))
        elif k == "meta":
            missing = [STAGE_OUT[s] for s in STAGES[:-1] if not os.path.exists(P[STAGE_OUT[s]])]
            if missing: raise RuntimeError(f"meta needs every stage's output; missing {missing}")
            for s in STAGES[:-1]:
                if s not in summaries: summaries[s] = stage_summary(P, s)
            fp_ = fp or file_meta(P["geom"]); minfo = M.info() if M is not None else dict(name=m, path=os.path.abspath(path), mock=MOCK, n_blocks=fp_["n_blocks"], hidden_size=fp_["hidden_size"])
            return stage_meta(P, m, fp_, summaries, minfo, mrec or model_record(root, m, path))
    return None


# ======================================================================================================================================
# Self-test (MOCK)
# ======================================================================================================================================
def _exec_nodes(path, first, last, ns):
    """Executes the top-level statements of a round-1 script from the one whose source starts with 'first' to the one starting with
    'last' (both included), in namespace ns (used to run round-1 code unchanged on the MOCK model without its model loading)."""
    src = open(path).read(); body = ast.parse(src).body; seg = [ast.get_source_segment(src, n) for n in body]
    i = next(k for k, s in enumerate(seg) if s.startswith(first)); j = next(k for k, s in enumerate(seg) if k >= i and s.startswith(last))
    exec(compile(ast.Module(body=body[i:j + 1], type_ignores=[]), path, "exec"), ns); return ns


def selftest(root, r2_dir, m="qwen3-1.7b"):
    assert MOCK, "selftest needs MOCK=1"
    torch.set_num_threads(int(os.environ.get("NT", "1"))); res = []; t0 = time.time()

    def check(name, ok, info=""):
        res.append((name, bool(ok))); print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)
    work = os.path.join(r2_dir, "geom_selftest", m); shutil.rmtree(work, ignore_errors=True); P = out_paths(work, m)     # from scratch
    meta = run(root, work, m) or json.load(open(P["meta"]))
    check("all stages wrote their outputs", all(os.path.exists(P[k]) for k in P), str({k: os.path.exists(v) for k, v in P.items()}))
    M = Model(root, m); fp = fingerprint(M, m)
    # A1: layout, shapes, readings against single unbatched forwards
    with np.load(P["geom"], allow_pickle=False) as zf: z = {k: zf[k] for k in zf.files}                  # read every array once
    S = STIM.sentences(); unit, neu = sentence_layout(z["s_obj"], z["s_frame"], z["s_family"], z["s_lexeme"])
    check("A1 shapes and layout", z["dot"].shape == (2040, 28, M.D) and z["obj"].shape == (2040, 28, M.D) and z["lex"].shape == (2040, M.D)
          and unit.shape == (2, 16, 60) and [str(t) for t in z["texts"]] == [s["text"] for s in S])
    rows, _ = a1_rows(M); worst = 0.0
    for j in (0, 17, 1000, 1023, 2039, 1950):
        r = rows[j]; ids = torch.tensor([r["ids"]])
        with torch.no_grad():
            hs = M.model.model(input_ids=ids, output_hidden_states=True).hidden_states; emb = M.model.model.embed_tokens(ids)[0]
        for l in (0, 5, 26):                                                               # hidden_states[l + 1] = output of block l (l < 27)
            worst = max(worst, float(np.abs(hs[l + 1][0, r["last"]].numpy() - z["dot"][j, l]).max()), float(np.abs(hs[l + 1][0, r["obj"]].double().mean(0).numpy() - z["obj"][j, l]).max()))
        worst = max(worst, float(np.abs(emb[r["off"]:r["off"] + r["n"]].double().mean(0).numpy() - z["lex"][j]).max()))
    check("A1 batched hook reads = single forwards (dot, obj, lex)", worst < 2e-4, f"max abs diff {worst:.2e}")
    j = next(k for k, r in enumerate(rows) if r["frame"] == 1 and r["family"] == 0)
    check("F2 rows = chat('Say one sentence.') + reply, '.' last", rows[j]["ids"] == M.enc(M.chat(STIM.F2_PROMPT) + rows[j]["text"]) and M.tok.decode([rows[j]["ids"][rows[j]["last"]]]) == "."
          and M.tok.decode(rows[j]["ids"][rows[j]["obj"][0]:rows[j]["obj"][-1] + 1]).strip() == OBJECTS[rows[j]["obj_i"]]["text"]["F2"])
    # A3: gradients against central finite differences (float64 copy of the MOCK model)
    c = load_pt(P["causal"]); arows, tid = a3_inputs(M)
    check("A3 shapes", tuple(c["W"].shape) == (60, 4, 28, M.D) and tuple(c["y"].shape) == (60, 4))
    # Qwen3's RMSNorm and rotary tables compute in float32 even in a float64 model (function noise ~1e-8), hence a 5-point stencil
    M64 = Model(root, m); M64.model = M64.model.double(); M64.layers = M64.model.model.layers; errs = []; g = torch.Generator().manual_seed(7)
    G64, y64, _ = a3_grads(M64, arows, tid, [3, 50], dtype=torch.float64)
    for oi, o in enumerate((3, 50)):
        for l in (2, 20):
            for s in (0, 3):
                u = torch.randn(M.D, generator=g, dtype=torch.float64); h = 1e-3; ys = {}
                for k in (2, 1, -1, -2):
                    V = torch.zeros(1, 28, M.D, dtype=torch.float64); V[0, l] = k * h * u
                    ids = torch.tensor([arows[o]["ids"]]); mask = torch.zeros(1, ids.shape[1]); mask[0, arows[o]["obj_pos"]] = 1
                    with torch.no_grad(), leaf_hooks(M64.layers, V, mask):
                        ys[k] = float(a3_scalars(M64.model.lm_head(M64.model.model(input_ids=ids).last_hidden_state[:, -1]), tid)[0, s])
                fd = (-ys[2] + 8 * ys[1] - 8 * ys[-1] + ys[-2]) / (12 * h); an = float(G64[oi, s, l] @ u); errs.append(abs(fd - an) / max(abs(an), 1e-8))
    check("A3 gradients = finite differences (float64 model, 5-point stencil)", max(errs) < 1e-3, f"max rel err {max(errs):.1e}")
    rel32 = float((c["W"][[3, 50]].double() - G64).norm() / G64.norm())
    check("A3 float32 run = float64 gradients", rel32 < 1e-3, f"rel {rel32:.1e}")
    check("A3 at the last block is identically zero (a leaf at the object positions cannot reach the last position): A3 undefined there",
          float(c["W"][:, :, -1].abs().max()) == 0.0 and float(c["W"][:, :, -2].abs().max()) > 0)
    # O7: selection and delta against dn_delta.py's own code (executed unchanged on the MOCK tokenizer; fisher by ast)
    ns = dict(tok=M.tok, data=os.path.join(root, "dn_out", "mmlu_topics.json"), np=np, json=json, os=os)
    for k in ("NMAX", "LQ", "MINQ"): os.environ.pop(k, None)
    _exec_nodes(os.path.join(JU, "dn_delta.py"), "NMAX, LQ, MINQ", "NMAX, LQ, MINQ", ns); _exec_nodes(os.path.join(JU, "dn_delta.py"), "rows = json.load", "assert len({len(t)", ns)
    _exec_nodes(os.path.join(JU, "dn_delta.py"), "def fisher", "def fisher", ns)
    sel = o7_texts(root, M.tok)
    with np.load(P["delta"], allow_pickle=False) as df: d = {k: df[k] for k in df.files}
    check("O7 texts and labels = dn_delta.py's selection", sel["texts"] == ns["texts"] and (sel["labels"] == ns["labels"]).all() and (d["ids"] == np.array(ns["texts"])).all())
    LAST, MEAN = d["last"].astype(np.float64), d["mean"].astype(np.float64); gp = np.random.default_rng(2); ref = {"last": [], "mean": []}
    for l in range(28):
        for nm, A in (("last", LAST[:, l]), ("mean", MEAN[:, l])):
            F = ns["fisher"](A, ns["labels"]); fl = float(np.mean([ns["fisher"](A, gp.permutation(ns["labels"])) for _ in range(50)])); ref[nm].append((F, fl))
    mine = o7_delta(LAST, MEAN, sel["labels"], torch.device("cpu"))
    err = max(abs(mine[nm]["F"][l] - ref[nm][l][0]) + abs(mine[nm]["floor"][l] - ref[nm][l][1]) for nm in ref for l in range(28))
    check("O7 F and floor = dn_delta.py's fisher and permutation stream", err < 1e-9, f"max abs diff {err:.1e}")
    dmax = max(abs(d["last_delta"][l] - (ref["last"][l][0] - ref["last"][l][1])) for l in range(28))
    check("O7 stored delta (full-precision states) = dn_delta on the stored float16 states", dmax < 2e-3, f"max abs diff {dmax:.1e}")
    # O8: against ufo_pool.py's own code executed unchanged on a fresh MOCK model; OD-G7 through an exported reference
    from ufo_common import texts as ufo_texts, FIRST
    # ufo_pool.py's two passes run unchanged; its per-block statements up to the eigendecomposition are taken from its source by ast
    # (the rest of that loop indexes eigenvalue 1800 and cannot run at the MOCK width 64)
    Mf = Model(root, m); ns = dict(root=root, model=Mf.model, tok=Mf.tok, torch=torch, np=np, time=time, texts=ufo_texts, FIRST=FIRST, RANDINIT=False)
    upath = os.path.join(JU, "ufo_pool.py"); usrc = open(upath).read()
    with torch.no_grad(): _exec_nodes(upath, "LAYERS = model.model.layers", "for i, (src, ids, valid) in enumerate(pool):", ns)
    loop = next(n for n in ast.parse(usrc).body if ast.get_source_segment(usrc, n).startswith("for m in range(L):"))
    k = next(i for i, st in enumerate(loop.body) if ast.get_source_segment(usrc, st).startswith("U = U.flip(1)"))
    loop.body = loop.body[:k + 1] + ast.parse("RES[m] = dict(mu=mu.float(), sd=ev.sqrt().float())").body; ns["RES"] = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), upath, "exec"), ns)
    pl = load_pt(P["pool"]); worst = 0.0
    for l in range(28):
        b, rb = pl["blocks"][l], ns["RES"][l]; worst = max(worst, float((b["mu"].float() - rb["mu"]).abs().max() / rb["mu"].abs().max()),
                                                            float(((b["ev"][:8].sqrt().float() - rb["sd"][:8]).abs() / rb["sd"][:8]).max()))
    check("O8 mean and eigenvalues = ufo_pool.py's code", worst < 1e-4, f"max rel diff {worst:.1e}")
    upt = os.path.join(work, "ufo_pool_mock.pt"); torch.save(dict(blocks=ns["RES"]), upt)
    ref = export_pool_ref(upt, os.path.join(work, "pool_ref_mock.json")); g7 = od_g7(pl["blocks"][14]["ev"].numpy(), os.path.join(work, "pool_ref_mock.json"))
    check("OD-G7 check with an exported pool_ref.json", g7["checked"] and g7["ok"] and len(ref["top5_eigenvalues"]) == 5, str(g7.get("rel_diff")))
    bad = dict(top5_eigenvalues=[v * 1.2 for v in ref["top5_eigenvalues"]]); write_json(os.path.join(work, "pool_ref_bad.json"), bad)
    check("OD-G7 flags a 20 % mismatch", od_g7(pl["blocks"][14]["ev"].numpy(), os.path.join(work, "pool_ref_bad.json"))["ok"] is False)
    # dirs
    dr = load_pt(P["dirs"]); X = torch.as_tensor(z["dot"]).double(); l = 9
    pooled = lambda v, o: (X[unit[0, v, o], l] + X[unit[1, v, o], l]) / 2
    co = torch.stack([torch.stack([pooled(VF_OFF["IMP"] + k, o) for k in range(4)]).mean(0) - torch.stack([pooled(VF_OFF["DES"] + k, o) for k in range(4)]).mean(0) for o in range(48)])
    gS2 = torch.stack([torch.stack([X[unit[1, VF_OFF["IMP"] + k, o], l] for k in range(4)]).mean(0) - torch.stack([X[unit[1, VF_OFF["DES"] + k, o], l] for k in range(4)]).mean(0) for o in SELF_OBJ]).mean(0)
    check("dirs = definitions (c_o, g, s orthogonal to g, g_SELF)", torch.allclose(dr["c_o"][l].double(), co, atol=1e-5) and torch.allclose(dr["g"][l].double(), co.mean(0), atol=1e-5)
          and abs(float(dr["s"][l].double() @ dr["g"][l].double())) < 1e-4 * float(dr["s"][l].norm() * dr["g"][l].norm() + 1e-12)
          and torch.allclose(dr["g_self_f2"][l].double(), gS2, atol=1e-5))
    # meta and resume
    mt = json.load(open(P["meta"])); check("meta: stages, outputs with sha256, model record", set(mt["stages"]) == set(STAGES[:-1]) and all(len(v["sha256"]) == 64 for v in mt["outputs"].values()))
    mt0 = {k: os.path.getmtime(v) for k, v in P.items()}; t1 = time.time(); run(root, work, m)
    check("resume: a second run skips every stage", {k: os.path.getmtime(v) for k, v in P.items()} == mt0, f"({time.time() - t1:.1f}s)")
    os.remove(P["geom"]); run(root, work, m); mt1 = {k: os.path.getmtime(v) for k, v in P.items()}
    check("resume after a lost local A1 file: A1, dirs and meta recomputed, the rest kept", all(mt1[k] != mt0[k] for k in ("geom", "dirs", "meta"))
          and all(mt1[k] == mt0[k] for k in ("causal", "delta", "pool")) and json.load(open(P["meta"]))["outputs"]["geom"]["sha256"] == sha256_file(P["geom"]))
    # rgi_geom-1: A1 recomputed by an earlier, interrupted call (stage a1 alone); the next 'all' must redo dirs and meta
    mt2 = {k: os.path.getmtime(v) for k, v in P.items()}; run(root, work, m, stages=("a1",), force=("a1",)); run(root, work, m)
    mt3 = {k: os.path.getmtime(v) for k, v in P.items()}; mt_ = json.load(open(P["meta"]))
    check("rgi_geom-1: A1 recomputed in an earlier call -> dirs and meta redone, certified hashes = files", all(mt3[k] != mt2[k] for k in ("geom", "dirs", "meta"))
          and all(mt3[k] == mt2[k] for k in ("causal", "delta", "pool")) and load_pt(P["dirs"])["src_geom_sha256"] == sha256_file(P["geom"])
          and all(mt_["outputs"][k]["sha256"] == sha256_file(P[k]) for k in ("geom", "delta", "pool", "causal", "dirs")))
    dz_ = dict(torch.load(P["dirs"], map_location="cpu", weights_only=False)); dz_.pop("src_geom_sha256"); torch_save(P["dirs"], dz_); mt4 = os.path.getmtime(P["dirs"])
    run(root, work, m); ok_old = load_pt(P["dirs"]).get("src_geom_sha256") == sha256_file(P["geom"]) and os.path.getmtime(P["dirs"]) != mt4
    gtmp = P["geom"] + ".moved"; os.replace(P["geom"], gtmp); mt5 = os.path.getmtime(P["dirs"])
    try: run(root, work, m, stages=("dirs",)); ok_nogeom = os.path.getmtime(P["dirs"]) == mt5
    except Exception: ok_nogeom = False
    finally: os.replace(gtmp, P["geom"])
    check("rgi_geom-1: a dirs file without its source hash is redone; 'dirs' alone without the local A1 file keeps it", ok_old and ok_nogeom)
    # argparse usage with '%' (help and errors exit cleanly)
    import contextlib, io
    codes = []
    for argv in (["-h"], [root, "r1", work, "--model", "no-such-model"]):
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()): main(argv)
            codes.append(None)
        except SystemExit as e: codes.append(e.code)
        except Exception as e: codes.append(repr(e))
    check("command line: -h exits 0, a bad --model exits 2 (usage with '%' formats)", codes == [0, 2], str(codes))
    Pb = out_paths(os.path.join(work, "foreign"), m); npz_save(Pb["geom"], dict(fingerprint=dict(fp, version="other")), a=np.zeros(1))
    try: run(root, os.path.join(work, "foreign"), m, stages=("a1",)); refused = False
    except RuntimeError: refused = True
    check("resume refuses an output of another run", refused)
    n_ok = sum(ok for _, ok in res); print(f"selftest r2_rgi_geom ({m}): {n_ok}/{len(res)} passed ({time.time() - t0:.0f}s)", flush=True)
    return n_ok == len(res)


# ======================================================================================================================================
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "export_pool_ref":
        if len(argv) != 3: raise SystemExit(__doc__)
        ref = export_pool_ref(argv[1], argv[2]); print(json.dumps(dict(top5_eigenvalues=ref["top5_eigenvalues"], source_sha256=ref["source_sha256"]))); return 0
    ap = argparse.ArgumentParser(description="r2_rgi_geom.py (plan 9.2)", usage=__doc__.replace("%", "%%"))   # argparse %-formats the usage
    ap.add_argument("root"); ap.add_argument("r1_dir"); ap.add_argument("r2_dir"); ap.add_argument("stage", nargs="?", default="all")
    ap.add_argument("--model", default="qwen3-1.7b", choices=MODELS); ap.add_argument("--model-path", default=None)
    ap.add_argument("--force", action="store_true", help="recompute the given stage even if its output exists")
    a = ap.parse_args(argv); torch.set_num_threads(int(os.environ.get("NT", "3")))
    if a.stage == "selftest": return 0 if selftest(a.root, a.r2_dir, a.model) else 1
    stages = STAGES if a.stage == "all" else (a.stage,)
    if a.stage not in STAGES + ("all",): raise SystemExit(f"unknown stage {a.stage!r}; one of {STAGES + ('all', 'selftest')}")
    mrec = model_record(a.root, a.model, a.model_path or os.path.join(a.root, "models", a.model))     # Qwen3-0.6B: G33 checksum before any work
    run(a.root, a.r2_dir, a.model, stages, a.model_path, force=stages if a.force else (), mrec=mrec)
    return 0


if __name__ == "__main__":
    sys.exit(main())

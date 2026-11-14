"""SO-PILOT (main §2.3; owner main): the pilot material package file pilot_material.json and the main-level steps of the re-pilot
r1b_P0b (Drive throughput on the Drive path, the compact engineering summary p0b_engineering.json), plus the guard every P0b item
runs under (an out-of-memory error or any other error becomes a recorded value; the session continues).

    python r1b_pilot.py build <tokenizer dir> <wikitext103 dir> <pg19 test dir> <out.json>   rebuild pilot_material.json (CPU)

pilot_material.json:
  prompts      32 new short everyday-scene prompts (written for the pilot; normalised-string disjoint from P256, F-neu, D's stimuli,
               K's label bank and R's neutral pool; the builder checks it)
  wt_passages  8 WikiText-103 train passages after K's 1M-token reference (raw train lines >= 34,137) and before
               datasets.wikitext103.rk1_train_offset_lines: raw line ranges [start, end) of train-00000-of-00002.parquet, the sha256 of the
               passage text (non-empty lines stripped, joined by ' ') and its Qwen3 token count; no text
  pg19         PG19 test books 25-28 of SO-LONGDOC's ordered list (longdoc_slices.json 'pilot'): per book one 16,384- and one
               32,768-token slice from token 2,000 (r1b_l_base's slice rule), file name, offsets and the sha256 of the slice's token ids
               (r1b_core.canon); no book text
Every generated text in the pilot is discarded; only whitelisted engineering keys are written."""
import json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np
import r1b_core as K
import r1b_contracts as C

PROMPTS = (
    "Describe a laundromat on a Saturday morning.",
    "Describe a carpenter fixing a wobbly kitchen table.",
    "Describe a pharmacy just before closing time.",
    "Describe two friends assembling a flat-pack wardrobe.",
    "Describe a ferry crossing on a calm grey day.",
    "Describe a grandmother teaching a child to knit.",
    "Describe a car wash on a sunny weekend.",
    "Describe workers sorting apples in an orchard shed.",
    "Describe a tailor measuring a customer for a suit.",
    "Describe a dog park early in the morning.",
    "Describe a hardware store on a busy Saturday.",
    "Describe a jazz band rehearsing in a basement.",
    "Describe a ski lift queue on a cold morning.",
    "Describe a flower seller arranging tulips.",
    "Describe an ice cream van on a hot street.",
    "Describe a chess club meeting in a community hall.",
    "Describe a window cleaner working on a tall office building.",
    "Describe a street after the first snowfall of the year.",
    "Describe a pottery class for beginners.",
    "Describe a delivery cyclist in evening traffic.",
    "Describe a barber shop in the late afternoon.",
    "Describe a family setting up a tent at a campsite.",
    "Describe a cinema lobby before a late film.",
    "Describe a recycling centre on a Sunday morning.",
    "Describe a swimming pool during an early lesson.",
    "Describe a cobbler repairing a pair of boots.",
    "Describe a rooftop garden in a busy city.",
    "Describe a bowling alley on a weeknight.",
    "Describe neighbours clearing snow from a shared path.",
    "Describe a supermarket checkout during the lunch rush.",
    "Describe a vet examining a nervous cat.",
    "Describe a choir rehearsing for a winter concert.",
)
WT_FILE = "wikitext-103-raw-v1/train-00000-of-00002.parquet"
K_REF_END_LINE = 34137                  # K's 1M-token reference ends at this raw train line (draft record d_streams.k_ref_1M_end_line)
WT_FIRST, WT_MIN_TOKENS, WT_N = 34200, 600, 8
PG_LENGTHS, PG_SKIP = (16384, 32768), 2000


def norm(s):
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
    return " ".join(s.split())


def _strings(obj, out):
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _strings(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _strings(v, out)
    return out


def analysis_strings(plan_dir=None):
    """Every text of P256, F-neu, D's stimuli, K's label bank and R's neutral pool (for the disjointness check)."""
    import r1b_l_base as LB
    d = plan_dir or os.path.join(K.REPO, C.PLAN_DIR); src = {}
    src["P256"] = _strings(K.load_json(os.path.join(d, "prompts_1b.json")), [])
    src["F-neu"] = list(LB.F_NEU_SCENES)
    src["D"] = [s for f in ("d1_stimuli.json", "d2_prompts.json", "d3_items.json", "d4_roles.json", "d1v_pairs.json")
                for s in _strings(K.load_json(os.path.join(d, f), {}), [])]
    src["K"] = _strings(K.load_json(os.path.join(d, "k1_label_bank", "bank.json"), {}), []) + _strings(K.load_json(os.path.join(d, "label_bank_1b.json"), {}), [])
    p = os.path.join(d, "neutral_pool_1b.txt")
    src["R_neutral"] = open(p, encoding="utf-8").read().splitlines() if os.path.exists(p) else []
    return src


def disjoint(prompts, src):
    """Normalised-string checks: no prompt equals or occurs inside any analysis text, and no F-neu scene, P256 prompt or analysis text
    of >= 3 words occurs inside a prompt (word boundaries). Returns {source: [problems]}."""
    probs = {}
    for name, ss in src.items():
        ns = [norm(s) for s in ss if norm(s)]
        blob = "\n".join(ns)
        for p in prompts:
            q = norm(p)
            if q in blob:
                probs.setdefault(name, []).append(f"prompt inside {name}: {p!r}")
            for s in ns:
                if (name in ("F-neu", "P256") or len(s.split()) >= 3) and f" {s} " in f" {q} ":
                    probs.setdefault(name, []).append(f"{name} text {s!r} inside prompt {p!r}")
    return probs


def wt_lines(wt_dir):
    import pandas as pd
    return pd.read_parquet(os.path.join(wt_dir, WT_FILE))["text"].tolist()


def wt_text(lines, start, end):
    return " ".join(l.strip() for l in lines[start:end] if l.strip())


def wt_passages(lines, tok, upto):
    """8 passages: each starts at an article heading (' = ... = ' followed by an empty line) at or after WT_FIRST and takes whole lines until it has
    >= WT_MIN_TOKENS tokens; the next one starts at the next heading after its end; all end before `upto`."""
    head = re.compile(r"^ = [^=].* = $")
    out, i = [], WT_FIRST
    while len(out) < WT_N:
        while i < upto and not (head.match(lines[i].rstrip("\n")) and i + 1 < len(lines) and not lines[i + 1].strip()):
            i += 1
        s = i; e = s + 1
        while e < upto and len(K.enc(tok, wt_text(lines, s, e))) < WT_MIN_TOKENS:
            e += 1
        if e >= upto:
            raise RuntimeError("not enough WikiText-103 train lines before rk1_train_offset_lines")
        t = wt_text(lines, s, e)
        out.append(dict(file=WT_FILE, start=s, end=e, text_sha256=K.sha256_text(t), n_tokens=len(K.enc(tok, t)),
                        heading=lines[s].strip()[:80]))
        i = e
    return out


def pg_slices(pg_dir, tok, pilot_books):
    out = []
    for b in pilot_books:
        ids = tok(open(os.path.join(pg_dir, b["file"]), encoding="utf-8", errors="replace").read(), add_special_tokens=False)["input_ids"]
        sl = []
        for L in PG_LENGTHS:
            seg = ids[PG_SKIP:PG_SKIP + L]
            assert len(seg) == L, (b["file"], len(ids))
            sl.append(dict(length=L, start=PG_SKIP, end=PG_SKIP + L, sha256=K.sha256_text(K.canon(seg))))
        if sl[-1]["sha256"] != b["sha256"] or sl[-1]["end"] != b["end"]:
            raise RuntimeError(f"{b['file']}: the 32,768-token slice differs from longdoc_slices.json")
        out.append(dict(file=b["file"], n_tokens=len(ids), slices=sl))
    return out


def build_pilot_material(tok_dir, wt_dir, pg_dir, out, plan_dir=None):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir); d = plan_dir or os.path.join(K.REPO, C.PLAN_DIR)
    assert len(PROMPTS) == 32 and len({norm(p) for p in PROMPTS}) == 32
    probs = disjoint(PROMPTS, analysis_strings(d))
    if probs:
        raise RuntimeError(f"pilot prompts not disjoint: {probs}")
    draft = K.load_json(os.path.join(d, K.PILOT_PINS), {}) or {}
    upto = int(((draft.get("datasets") or {}).get("wikitext103") or {}).get("rk1_train_offset_lines") or 40000)
    lines = wt_lines(wt_dir); P = wt_passages(lines, tok, upto)
    bank = "\n".join(norm(s) for s in analysis_strings(d)["K"])
    clash = [p["start"] for p in P if norm(wt_text(lines, p["start"], p["end"]))[:300] in bank]
    if clash:
        raise RuntimeError(f"WikiText passages inside K's bank: {clash}")
    ld = K.load_json(os.path.join(d, "longdoc_slices.json"))
    rec = dict(version=1, rule=dict(prompts="32 new everyday scenes written for the pilot; normalised-string disjoint from P256, F-neu, D's "
                                            "stimuli, K's label bank and R's neutral pool (r1b_pilot.disjoint)",
                                    wt_passages=f"raw train lines of {WT_FILE}: after K's 1M-token reference (ends at line {K_REF_END_LINE}), "
                                                f"from the first heading at or after line {WT_FIRST}, >= {WT_MIN_TOKENS} Qwen3 tokens each, "
                                                f"before rk1_train_offset_lines ({upto}); text = non-empty lines stripped and joined by ' '",
                                    pg19="SO-LONGDOC pilot books 25-28 (longdoc_slices.json 'pilot'); slices [2,000, 2,000 + L) of the Qwen3 "
                                         "token ids for L in {16,384, 32,768}; sha256 of r1b_core.canon(ids)"),
               prompts=list(PROMPTS), prompts_sha256=K.sha256_text(K.canon(list(PROMPTS))), wt_passages=P,
               wt_upto_line=upto, pg19=pg_slices(pg_dir, tok, ld["pilot"]),
               tokenizer=dict(model=C.M17, revision=K.MODELS[C.M17]["revision"]))
    K.dump_json(rec, out)
    return rec


def material(required=True):
    p = os.path.join(K.REPO, C.PLAN_DIR, "pilot_material.json")
    m = K.load_json(p) if os.path.exists(p) else None
    if m is None and required:
        raise K.NotRun("input missing (pilot_material.json)")
    return m


def prompts(ctx, n=32):
    m = material(required=not ctx.mock)
    return list((m or {}).get("prompts") or PROMPTS)[:n]


def wt_passage_texts(ctx):
    """The 8 pilot passages (sha256-checked) from the pinned WikiText-103; MOCK: short stand-ins."""
    if ctx.mock:
        return [" ".join(PROMPTS[i:i + 8]) * 4 for i in range(8)]
    m = material(); d = K.fetch_dataset("wikitext103", ctx.root, ctx.freeze); lines = wt_lines(d); out = []
    for p in m["wt_passages"]:
        t = wt_text(lines, p["start"], p["end"])
        if K.sha256_text(t) != p["text_sha256"]:
            raise RuntimeError(f"pilot passage at line {p['start']} differs from its sha256")
        out.append(t)
    return out


def pg19_slices(ctx, tok, length):
    """[(file, ids)] of the 4 pilot books at one slice length (sha256-checked); MOCK: random ids of a short stand-in length."""
    if ctx.mock:
        g = np.random.default_rng(length)
        return [(f"mock{i}", [int(x) for x in g.integers(1000, 30000, min(length, 64 if length <= 16384 else 96))]) for i in range(4)]
    m = material(); d = K.fetch_dataset("pg19", ctx.root, ctx.freeze); out = []
    for b in m["pg19"]:
        s = [x for x in b["slices"] if x["length"] == length][0]
        ids = K.enc(tok, open(os.path.join(d, b["file"]), encoding="utf-8", errors="replace").read())[s["start"]:s["end"]]
        if K.sha256_text(K.canon(ids)) != s["sha256"]:
            raise RuntimeError(f"pilot slice {b['file']} ({length}) differs from its sha256")
        out.append((b["file"], ids))
    return out


# ------------------------------------------------------------------------------------------------------------------------------------
# r1b_P0b: guarded items
# ------------------------------------------------------------------------------------------------------------------------------------
def oom(e):
    t = K._torch()
    return isinstance(e, getattr(t, "OutOfMemoryError", RuntimeError)) or "out of memory" in str(e).lower()


def free():
    import gc
    gc.collect(); t = K._torch()
    if t.cuda.is_available():
        t.cuda.empty_cache()


def sync():
    t = K._torch()
    if t.cuda.is_available():
        t.cuda.synchronize()


class Rec(dict):
    """A pilot record with the whitelisted keys; item(name, fn) runs fn() and stores an error (out of memory or other) as a value."""

    def __init__(self):
        super().__init__(rates={}, memory={}, parity={}, code_checks={}, timing={}, env={}, notes=[])

    def item(self, name, fn):
        t0 = time.time(); t = K._torch()
        if t.cuda.is_available():
            t.cuda.reset_peak_memory_stats()
        try:
            r = fn()
            self["code_checks"][f"{name}_ran"] = True
            return r
        except Exception as e:                                 # a probe result, not a failure of the session
            code = "NaN:oom" if oom(e) else "NaN:error"
            self["code_checks"][f"{name}_ran"] = K.coded(code)
            self["notes"].append(f"{name}: {type(e).__name__}: {str(e)[:300]}")
            return None
        finally:
            self["timing"][f"{name}_s"] = round(time.time() - t0, 3)
            if t.cuda.is_available():
                self["memory"][f"{name}_peak_gb"] = round(t.cuda.max_memory_allocated() / 1e9, 3)
            free()


def timed(fn, reps=1, warm=1):
    """Median seconds of fn() over reps calls after warm untimed calls, synchronised around each call."""
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        sync(); t = time.time(); fn(); sync(); ts.append(time.time() - t)
    return float(np.median(ts))


def p0b_drive(ctx):
    """Drive throughput on the Drive path: (a) copy of a >= 200 MB file into the synced Drive session folder (write + fsync, then the
    file is removed); (b) a cold read of a large file another runtime wrote into dsse_runs (r1b_inputs/*.pt, then any file >= 100 MB
    in dsse_runs at depth <= 3, not written by this session)."""
    rec = Rec(); drive = getattr(ctx, "drive", None); runs = os.path.dirname(drive) if drive else None
    n_mb = 256 if not ctx.mock else 8

    def write():
        if not drive or not os.path.isdir(os.path.dirname(drive)):
            raise K.NotRun("no Drive session folder (ctx.drive)")
        os.makedirs(drive, exist_ok=True); src = ctx.path("p0b/_drive_src.bin"); dst = os.path.join(drive, "_p0b_drive_probe.bin")
        with open(src, "wb") as f:
            for _ in range(n_mb):
                f.write(os.urandom(1 << 20))
        import shutil
        t = time.time(); shutil.copyfile(src, dst)
        with open(dst, "rb+") as f:
            os.fsync(f.fileno())
        dt = time.time() - t; os.remove(src); os.remove(dst)
        rec["throughput"] = rec.get("throughput", {}); rec["throughput"]["drive_copy_in_MB_s"] = n_mb / max(dt, 1e-9)
        rec["throughput"]["drive_copy_in_MB"] = n_mb

    def read():
        if not runs or not os.path.isdir(runs):
            raise K.NotRun("no dsse_runs folder")
        cand = sorted(x for x in __import__("glob").glob(os.path.join(runs, "r1b_inputs", "*.pt")))
        if not cand:
            for r_, ds, fs in os.walk(runs):
                if r_[len(runs):].count(os.sep) >= 3 or os.path.abspath(r_).startswith(os.path.abspath(drive or "\0")):
                    ds[:] = []
                    continue
                cand += [os.path.join(r_, f) for f in fs if os.path.getsize(os.path.join(r_, f)) >= (100 << 20 if not ctx.mock else 1)]
        if not cand:
            raise K.NotRun("no large file of another runtime in dsse_runs")
        p = max(cand, key=os.path.getsize); n = 0; t = time.time()
        with open(p, "rb") as f:
            for blk in iter(lambda: f.read(1 << 24), b""):
                n += len(blk)
        dt = time.time() - t
        rec["throughput"] = rec.get("throughput", {})
        rec["throughput"].update(drive_cold_read_MB_s=n / 1e6 / max(dt, 1e-9), drive_cold_read_MB=n / 1e6,
                                 drive_cold_read_file=os.path.relpath(p, runs))
    rec.item("drive_write", write); rec.item("drive_cold_read", read)
    ctx.write_json("p0b/drive.json", rec, config=dict(n_mb=n_mb))


SUMMARY_FILES = ("p0b/r_rates.json", "p0b/r_replay.json", "p0b/d_decode.json", "p0b/d_parity.json", "p0b/d_profile.json",
                 "p0b/k_rates.json", "p0b/k_checks.json", "p0b/k_35.json", "p0b/l_decode.json", "p0b/l_long.json", "p0b/drive.json")


def p0b_summary(ctx):
    """p0b_engineering.json: every P0b record's rates, memory, parity, code checks (identity verdicts) and throughput side by side,
    keyed by item, with the pass / fail of every check and every guarded item that failed (so the freeze record can be filled)."""
    out = dict(rates={}, memory={}, parity={}, code_checks={}, throughput={}, timing={}, notes=[])
    for rel in SUMMARY_FILES:
        j = K.load_json(ctx.path(rel), None)
        tag = os.path.basename(rel)[:-5]
        if j is None:
            out["notes"].append(f"{rel}: missing"); continue
        for k in ("rates", "memory", "parity", "code_checks", "throughput", "timing"):
            for a, v in (j.get(k) or {}).items():
                out[k][f"{tag}.{a}"] = v
        out["notes"] += [f"{tag}: {x}" for x in (j.get("notes") or [])]
    cc = out["code_checks"]
    failed = sorted(k for k, v in cc.items() if v is False or (isinstance(v, str) and v.startswith("NaN:")))
    out["code_checks"]["_failed_or_not_run"] = failed
    ctx.write_json("p0b_engineering.json", dict(rates=out["rates"], memory=out["memory"], parity=out["parity"], code_checks=out["code_checks"],
                                                throughput=out["throughput"], timing=out["timing"], notes=out["notes"]),
                   config=dict(files=list(SUMMARY_FILES)))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "build":
        r = build_pilot_material(*a[1:5])
        print("written", a[4], len(r["prompts"]), "prompts,", len(r["wt_passages"]), "passages,", len(r["pg19"]), "books")
        raise SystemExit(0)
    print(__doc__)
    raise SystemExit(2)

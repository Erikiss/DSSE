"""Round 1b shared core (main §2.1, §2.3, §2.4, §6.2-§6.3): what every part uses.

    python r1b_core.py selftest [<root>]                 code tests on the CPU with the MOCK tiny random model (<root>/models/qwen3-1.7b
                                                          needs config.json and the tokenizer only; default env R1B_ROOT)
    python r1b_core.py fetch-models <root> <name> ...    download pinned models and check their sha256 (notebooks)
    python r1b_core.py keys <out.json>                   write the SO-KEYS registry (package file r1b_keys.json)

Contents (API in IMPLEMENTATION.md):
  models      MODELS, register_loader, ensure_model, load_model (float32, TF32 off, 'highest' matmul; MOCK=1: rk_loop's tiny random
              model), chat_ids, user_start, pad_left, stop_tokens, generate (keyed Gumbel-max), mean_read, sdpa_efficient (SG9)
  SO-KEYS     KEYS, key_owner, check_keys, keyseed / gumbel_class (verbatim rk_loop lines, sha256-checked), write_keys
  SO-PVAL     interval, p_one_sided, p_rank, p_pi (K.4), holm, bh, boot_idx, boot_two_level, prompt_bootstrap (r2_rk's)
  SO-DMG      RefSet, Damage (= r2_unlearn_qwen.RefKL's formula), calibrate (push form and NSCALE_DN gain form), DELTA1, DELTA2
  hooks       Hooks: push at a block output from the first user-content token (rk_loop's operator), neuron read / scale / add / clamp /
              zero / mean-ablate on the MLP intermediate (down_proj input) of any block, per-token capture, stopped reads
  SO-JLENS    load_jlens (sha256 pin), jlens_convention (K4j)
  R9.1-R9.4   Ctx.provenance, codes (coded, val, coded_list, code_counts), labelled, dist; Ctx.write_json / save_pt / write_file
  Ctx         what a step sees (paths, inputs, models, keys, funding, trims, time guard, cache); Ctx.from_env() in script steps
Model-free at import (torch is imported lazily); never imports rk_loop (main §6.1)."""
import contextlib, hashlib, importlib.metadata, json, math, os, platform, re, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); AJS = os.path.dirname(HERE); REPO = os.path.dirname(os.path.dirname(AJS))
R2DIR = os.path.join(AJS, "round2")
for _p in (HERE, R2DIR, AJS):
    if _p not in sys.path:
        sys.path.append(_p)
import r1b_contracts as C

MOCK = os.environ.get("MOCK") == "1"
QUICK = os.environ.get("QUICK") == "1"
DELTA1, DELTA2 = 0.05, 0.15                     # SO-DMG levels (nats)
N_BOOT, LEVEL = 2000, 0.90
MODELS = {   # main §2.1; 'sha256': file -> sha256 (qwen3-1.7b from gpu/data/checksums.json; qwen3.5-2b from runde1b_freeze.json)
    C.M17: dict(repo="Qwen/Qwen3-1.7B", revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", sha256=None, layers=28, hidden=2048),
    C.M06: dict(repo="Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca",
                sha256={"model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"}, layers=28, hidden=1024),
    C.M35: dict(repo="Qwen/Qwen3.5-2B", revision=None, sha256=None, layers=24, hidden=2048),
}
JLENS = dict(name="Qwen3-1.7B_jacobian_lens.pt", src="qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt",
             sha256="6fcc79011bd921ffd87612255e2e99950a124fa519470ee44ebaf161c39be9d6", n=27)
PILOT_KEYS = {"provenance", "rates", "memory", "throughput", "parity", "code_checks", "timing", "env", "notes"}   # SO-PILOT whitelist


def _torch():
    import torch
    return torch


# ====================================================================================================================================
# Files, JSON, codes (R9.2), labels (R9.3), distributions (R9.4)
# ====================================================================================================================================
def iso(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t if t is not None else time.time()))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def canon(obj):
    """Canonical JSON (sorted keys, no spaces): the config hash of R9.1."""
    return json.dumps(jsonable(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_json(p, default=None):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def dump_json(obj, p):
    """Atomic JSON write; raw NaN / inf refused (R9.2: code missing values with coded())."""
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    o = jsonable(obj)
    bad = _nan_path(o)
    if bad is not None:
        raise ValueError(f"raw NaN / inf at {bad} in {p}: use r1b_core.coded() / val() (R9.2)")
    with open(p + ".tmp", "w") as f:
        json.dump(o, f, indent=1, allow_nan=False)
    os.replace(p + ".tmp", p)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in (sorted(x) if isinstance(x, set) else x)]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    t = sys.modules.get("torch")
    if t is not None and isinstance(x, t.Tensor):
        return jsonable(x.detach().cpu().tolist() if x.dim() else x.item())
    return x


def _nan_path(x, path="$"):
    if isinstance(x, float) and not math.isfinite(x):
        return path
    if isinstance(x, dict):
        for k, v in x.items():
            r = _nan_path(v, f"{path}.{k}")
            if r:
                return r
    elif isinstance(x, list):
        for i, v in enumerate(x):
            r = _nan_path(v, f"{path}[{i}]")
            if r:
                return r
    return None


CODE_RE = re.compile(r"^(NaN:[A-Za-z0-9_.]+|CENSORED:(below|above)_grid|NA:gate_closed:[A-Za-z0-9_.+-]+|NA:not_run:[A-Za-z0-9_ .,()'-]+"
                     r"|NA:conditional_not_triggered)$")
NA_TIER2, NA_FAILED2, NA_COND = "NA:not_run:tier2_unfunded", "NA:not_run:session_failed_twice", "NA:conditional_not_triggered"


def coded(code):
    """A missing value (R9.2): {'value': None, 'code': code}; codes 'NaN:<why>', 'CENSORED:below_grid' / 'above_grid',
    'NA:gate_closed:<gate>', 'NA:not_run:<reason>' (incl. 'RL<k>', 'tier2_unfunded', 'session_failed_twice'), NA_COND."""
    if not CODE_RE.match(code):
        raise ValueError(f"not an R9.2 code: {code!r}")
    return {"value": None, "code": code}


def val(x, code="NaN:undefined"):
    """float(x) if finite, else coded(code)."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return coded(code)
    return f if math.isfinite(f) else coded(code)


def coded_list(values, code="NaN:undefined"):
    return [val(v, code) for v in np.asarray(values, dtype=float).ravel()]


def code_counts(obj, counts=None):
    """Counts of every code in a nested object (R9.2: the summary gives per-table counts)."""
    counts = {} if counts is None else counts
    if isinstance(obj, dict):
        if set(obj) >= {"value", "code"} and obj.get("value") is None and isinstance(obj.get("code"), str):
            counts[obj["code"]] = counts.get(obj["code"], 0) + 1
        for v in obj.values():
            code_counts(v, counts)
    elif isinstance(obj, list):
        for v in obj:
            code_counts(v, counts)
    return counts


def labelled(value, *labels):
    """R9.3: a number or verdict string with its label list (labels never change the string)."""
    return {"value": value, "labels": [str(l) for l in labels if l]}


def dist(values, groups=None):
    """R9.4: n, SD, the 5/25/50/75/95 percentiles, min and max over the units (finite values), the number of non-finite ones, the
    effect frequency (share > 0); per group when groups (one label per unit) are given."""
    x = np.asarray(values, dtype=float).ravel()
    f = x[np.isfinite(x)]
    out = dict(n=int(x.size), n_finite=int(f.size), n_nonfinite=int(x.size - f.size))
    if f.size:
        q = np.percentile(f, [5, 25, 50, 75, 95])
        out.update(mean=float(f.mean()), sd=float(f.std(ddof=1)) if f.size > 1 else coded("NaN:single_unit"), p5=float(q[0]),
                   p25=float(q[1]), p50=float(q[2]), p75=float(q[3]), p95=float(q[4]), min=float(f.min()), max=float(f.max()),
                   share_pos=float((f > 0).mean()))
    else:
        out.update({k: coded("NaN:no_finite_values") for k in ("mean", "sd", "p5", "p25", "p50", "p75", "p95", "min", "max")})
    if groups is not None:
        g = np.asarray(groups)
        out["groups"] = {str(k): dist(x[g == k]) for k in sorted(set(g.tolist()), key=str)}
    return out


# ====================================================================================================================================
# SO-PVAL (main §2.4)
# ====================================================================================================================================
def interval(draws, level=LEVEL, max_undefined=0.05):
    """Percentile interval; a bound is undefined (coded) when more than 5 % of the replicates are undefined (round 2's 3(b))."""
    x = np.asarray(draws, dtype=float).ravel()
    ok = x[np.isfinite(x)]
    if x.size == 0 or (x.size - ok.size) > max_undefined * x.size:
        c = coded("NaN:fewer_than_95pct_finite_replicates")
        return c, c
    a = 100 * (1 - level) / 2
    return float(np.percentile(ok, a)), float(np.percentile(ok, 100 - a))


def p_one_sided(draws, threshold, alt="greater"):
    """(1 + #replicates on the null side of the threshold) / (B + 1), = / 2,001 at B = 2,000 (main §2.4); alt 'greater': the claim is
    statistic > threshold, null side = draws <= threshold; 'less': null side = draws >= threshold. Undefined draws count as null side."""
    x = np.asarray(draws, dtype=float).ravel()
    bad = ~np.isfinite(x)
    null = (x <= threshold) if alt == "greater" else (x >= threshold)
    return float((1 + int((null | bad).sum())) / (x.size + 1))


def p_rank(stat, controls):
    """(1 + #{controls >= statistic}) / (N + 1)."""
    c = np.asarray(controls, dtype=float).ravel()
    return float((1 + int((c >= stat).sum())) / (c.size + 1))


def p_pi(stat, controls):
    """K.4's prediction-interval p: 1 - F_t,N-1((X - mean X_R) / (sd X_R sqrt(1 + 1/N))) over the N defined (finite) controls."""
    from scipy.stats import t as tdist
    c = np.asarray(controls, dtype=float).ravel(); c = c[np.isfinite(c)]; n = c.size
    if n < 2 or not np.isfinite(stat):
        return coded("NaN:undefined_prediction_interval")
    return float(1 - tdist.cdf((stat - c.mean()) / (c.std(ddof=1) * math.sqrt(1 + 1 / n)), n - 1))


def holm(pvals, alpha=0.05):
    """Holm step-down: adjusted p-values and reject flags (in the input order)."""
    p = np.asarray(pvals, dtype=float)
    o = np.argsort(p, kind="stable"); m = p.size; adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj.tolist(), (adj <= alpha).tolist()


def bh(pvals, q=0.10):
    """Benjamini-Hochberg: adjusted p-values (q-values) and reject flags at FDR q."""
    p = np.asarray(pvals, dtype=float); m = p.size; o = np.argsort(p, kind="stable"); adj = np.empty(m); run = 1.0
    for r in range(m - 1, -1, -1):
        i = o[r]; run = min(run, p[i] * m / (r + 1)); adj[i] = min(run, 1.0)
    return adj.tolist(), (adj <= q).tolist()


def boot_idx(n, B=N_BOOT, seed=0):
    """[B, n] indices of a unit bootstrap (np.random.default_rng(seed))."""
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def boot_two_level(sizes, B=N_BOOT, seed=0):
    """Two-level bootstrap (units, then items within the drawn unit): a list of B replicates, each a list of (unit, item indices)."""
    g = np.random.default_rng(seed); n = len(sizes); out = []
    for _ in range(B):
        out.append([(int(u), g.integers(0, sizes[u], sizes[u])) for u in g.integers(0, n, n)])
    return out


def prompt_bootstrap(n_prompts=256, B=N_BOOT, seed=1001, n_rsi=None):
    """R.1's joint prompt bootstrap weights W [B, NP]: r2_rk.prompt_bootstrap (stratified: RSI-topic prompts 0..n_rsi-1, scene prompts
    after them; P256: 128 + 128, seed 1001)."""
    import r2_rk
    return r2_rk.prompt_bootstrap(n_prompts=n_prompts, B=B, seed=seed, n_rsi=n_rsi if n_rsi is not None else n_prompts // 2)


# ====================================================================================================================================
# SO-KEYS (main §2.3): registry, verbatim keyseed / Gumbel of rk_loop.py (lines 122-126) with a sha256 check of the source
# ====================================================================================================================================
RK_KEY_SRC = ('def keyseed(*k): return int(hashlib.sha256("|".join(map(str, k)).encode()).hexdigest()[:12], 16)\n'
              'class Gumbel(LogitsProcessor):\n'
              '    def __init__(self, keys): self.gens = [torch.Generator(device=DEV).manual_seed(k) for k in keys]\n'
              '    def __call__(self, input_ids, scores):\n'
              '        V = scores.shape[-1]; g = torch.stack([-torch.log(-torch.log(torch.rand(V, generator=G, device=DEV).clamp_(1e-12, '
              '1 - 1e-7))) for G in self.gens]); return scores.float() / TEMP + g\n')
RK_KEY_SRC_SHA256 = "4eab8cc78cf56b87febfb6f66379fd253e06fc61c144b8f7a6ee324e28f2e1e4"
_KNS = {"hashlib": hashlib}
exec(RK_KEY_SRC.split("\n")[0], _KNS)
keyseed = _KNS["keyseed"]


def rk_source_check():
    """The copied lines equal their pinned sha256 and occur verbatim in rk_loop.py (when the file is present)."""
    ok = sha256_text(RK_KEY_SRC) == RK_KEY_SRC_SHA256
    p = os.path.join(AJS, "rk_loop.py")
    found = (RK_KEY_SRC in open(p).read()) if os.path.exists(p) else None
    return dict(ok=bool(ok and found is not False), sha256_ok=ok, in_rk_loop=found)


_GUMBEL = {}


def gumbel_class(device, T):
    """rk_loop's Gumbel logits processor (verbatim source) bound to a device and a temperature: Gumbel(keys)."""
    k = (str(device), float(T))
    if k not in _GUMBEL:
        torch = _torch()
        from transformers import LogitsProcessor
        ns = {"hashlib": hashlib, "torch": torch, "LogitsProcessor": LogitsProcessor, "DEV": torch.device(device), "TEMP": float(T)}
        exec(RK_KEY_SRC, ns); _GUMBEL[k] = ns["Gumbel"]
    return _GUMBEL[k]


KEYS = {   # main §2.3 SO-KEYS registry; 'loops': R's loop names (tags 'r1b_' + name, 'r1b06_' + name), set by part R (register_r_loops)
    "R": dict(prefixes=("r1b_", "r1b06_"), tags=("r1b_axis", "r1b_dose", "r1b_free", "r1b_fdtchk", "r1b_lab", "r1b_lensfit", "r1b_k1",
                                                 "r1b_turn", "r1b_turn_rnd", "r1b_krep", "r1b_krep2", "r1b_Q4B", "r1b06_axis",
                                                 "r1b06_ref"), loops=()),
    "D": dict(prefixes=("r1bD",), tags=("r1bD1", "r1bD1r", "r1bD1v", "r1bD2", "r1bD3", "r1bD4", "r1bD4a", "r1bD5", "r1bD_INJ", "r1bD_pool",
                                        "r1bD_shuf")),
    "K": dict(prefixes=("r1bK_",), tags=tuple("r1bK_" + t for t in ("hn", "s3", "gate", "K4T", "K6e", "ses", "long", "bnd", "K10a", "K3f",
                                                                      "V7", "cal"))),
    "L": dict(prefixes=("r1bL",), tags=("r1bL0", "r1bL2", "r1bL2b", "r1bL2s", "r1bL3", "r1bL3t", "r1bL4", "r1bL5", "r1bLX", "r1bL1rot",
                                        "r1bL-ERA", "r1bL2perm", "r1bL-CTRL", "r1bL5-FM", "r1bL-BOOT", "r1bL-SUR", "r1bL-SHUF")),
    "main": dict(prefixes=("r1bP0_",), tags=(), pattern=r"^r1bP0_[A-Za-z0-9_.-]+$"),
}
R1_LOOPS = ("RSI_a", "RSI_b", "SEA", "SPO", "RND_1", "RND_2")
OLD_TAGS = ("ref", "axis", "mirror", "theta", "theta2", "r2PWR", "r2fdt", "r2CNT", "r2Q4B", "r2ARMA", "r2SUP", "r2DOSE", "r2CHK")
OLD_PREFIXES = ("r2GEN", "r2RSIb") + R1_LOOPS          # round-1 loop names with their SUF ('RSI_a', 'RSI_a@1.5', ...) and round-2 families


def register_r_loops(names):
    """Part R declares its loop names (tags 'r1b_' + name and 'r1b06_' + name); collisions raise (main §2.3)."""
    KEYS["R"]["loops"] = tuple(dict.fromkeys(tuple(KEYS["R"]["loops"]) + tuple(names)))
    e = check_keys()
    if e:
        raise ValueError("; ".join(e))


def _all_tags():
    out = []
    for o, d in KEYS.items():
        out += [(o, t) for t in d["tags"]]
        if o == "R":
            out += [(o, p + n) for n in d["loops"] for p in ("r1b_", "r1b06_")]
    return out


def key_owner(tag):
    """The part that owns a tag ('R', 'D', 'K', 'L', 'main'), or None if the tag is not registered."""
    for o, t in _all_tags():
        if t == tag:
            return o
    if re.match(KEYS["main"]["pattern"], tag):
        return "main"
    return None


def check_keys():
    """SO-KEYS code test: tags unique across parts, each under its own part's prefix and no other's, none equal to a round-1 or
    round-2 tag or family, no '|', and no R loop name forming another R tag. Returns a list of problems (empty = OK)."""
    errs, seen = [], {}
    for o, t in _all_tags():
        if t in seen:
            errs.append(f"tag {t!r} registered by {seen[t]} and {o}")
        seen[t] = o
        owners = {p for p, d in KEYS.items() if any(t.startswith(x) for x in d["prefixes"])}
        if owners != {o}:
            errs.append(f"tag {t!r} of {o} matches the prefixes of {sorted(owners)}")
        if "|" in t:
            errs.append(f"tag {t!r} contains '|'")
        if t in OLD_TAGS or t.startswith(OLD_PREFIXES):
            errs.append(f"tag {t!r} equals a round-1 / round-2 tag or family")
    for n in KEYS["R"]["loops"]:
        if "r1b_" + n in KEYS["R"]["tags"] or "r1b06_" + n in KEYS["R"]["tags"]:
            errs.append(f"R loop name {n!r} forms another R tag")
    return errs


def write_keys(path):
    reg = dict(registry={o: {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()} for o, d in KEYS.items()},
               round1_round2=dict(tags=list(OLD_TAGS), families=list(OLD_PREFIXES)), keyseed_source_sha256=RK_KEY_SRC_SHA256,
               problems=check_keys(), written=iso())
    dump_json(reg, path)
    return reg


# ====================================================================================================================================
# Models (main §2.1): pinned, float32, TF32 off; MOCK = rk_loop's tiny random model; loader hook for other models
# ====================================================================================================================================
LOADERS = {}
_LOADED = {}


def register_loader(name, fn, layers=None, hidden=None, repo=None, revision=None, sha256=None):
    """A part's loader for a model the default (AutoModelForCausalLM, float32, sdpa) cannot load: fn(root, device, mock) -> (tok, model).
    The pins (repo, revision, sha256) may be given here or come from runde1b_freeze.json['models'][name]."""
    LOADERS[name] = fn
    m = MODELS.setdefault(name, dict(repo=repo, revision=revision, sha256=sha256, layers=layers, hidden=hidden))
    for k, v in dict(repo=repo, revision=revision, sha256=sha256, layers=layers, hidden=hidden).items():
        if v is not None:
            m[k] = v


def numerics():
    """main §2.1: float32 compute, cuBLAS / cuDNN TF32 off, matmul precision 'highest'."""
    torch = _torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def model_pins(name, freeze=None):
    p = dict(MODELS.get(name) or {})
    if name == C.M17 and not p.get("sha256"):
        ck = load_json(os.path.join(AJS, "gpu", "data", "checksums.json"), {})
        p["sha256"] = ck.get("model_sha256")
    p.update({k: v for k, v in ((freeze or {}).get("models", {}).get(name) or {}).items() if v})
    return p


def ensure_model(name, root, freeze=None, download=True):
    """<root>/models/<name> at the pinned revision with every pinned sha256 checked; <root>/models/verified.json records it."""
    d = os.path.join(root, "models", name); vf = os.path.join(root, "models", "verified.json")
    pins = model_pins(name, freeze); ver = load_json(vf, {})
    if (ver.get(name) or {}).get("ok") and ver[name].get("revision") == pins.get("revision"):
        return d
    if not pins.get("revision") or not pins.get("sha256"):
        raise RuntimeError(f"{name}: revision or sha256 not pinned (MODELS or runde1b_freeze.json['models'])")
    if download and not all(os.path.exists(os.path.join(d, f)) for f in pins["sha256"]):
        from huggingface_hub import snapshot_download
        snapshot_download(pins["repo"], revision=pins["revision"], local_dir=d,
                          allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json", "*.jinja", "*.model", "*.txt"])
    got = {f: (sha256_file(os.path.join(d, f)) if os.path.exists(os.path.join(d, f)) else None) for f in pins["sha256"]}
    ok = got == dict(pins["sha256"])
    ver[name] = dict(ok=ok, repo=pins["repo"], revision=pins["revision"], sha256=got, checked=iso())
    dump_json(ver, vf)
    if not ok:
        raise RuntimeError(f"{name}: sha256 mismatch {[f for f in got if got[f] != pins['sha256'][f]]}")
    return d


def _mock_model(name, root):
    """rk_loop's MOCK model: the pinned config with hidden 64, intermediate 128, 4 heads, 2 KV heads, head_dim 16, the model's layer
    count, torch.manual_seed(0), float32 (a code test only). Config and tokenizer from <root>/models/<name> or, if absent, qwen3-1.7b."""
    torch = _torch()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    path = os.path.join(root, "models", name)
    if not os.path.exists(os.path.join(path, "config.json")):
        path = os.path.join(root, "models", C.M17)
    tok = AutoTokenizer.from_pretrained(path); cfg = AutoConfig.from_pretrained(path)
    cfg = getattr(cfg, "text_config", None) or cfg
    nl = int((MODELS.get(name) or {}).get("layers") or 28)
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = \
        nl, 64, 128, 4, 2, 16
    if getattr(cfg, "layer_types", None):
        lt = list(cfg.layer_types); cfg.layer_types = (lt * (nl // max(len(lt), 1) + 1))[:nl]
    torch.manual_seed(0)
    return tok, AutoModelForCausalLM.from_config(cfg).float().eval()


def load_model(name=C.M17, root=None, device=None, mock=None, freeze=None):
    """(tok, model), cached per process. Real models: ensure_model (pins), AutoModelForCausalLM float32 with sdpa attention, or the
    registered loader. MOCK: _mock_model. A real model on the CPU is refused unless R1B_ALLOW_CPU_REAL=1 (never in tests)."""
    torch = _torch()
    mock = MOCK if mock is None else mock
    root = root or os.environ.get("R1B_ROOT", "/content/root")
    device = device or os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    k = (name, bool(mock), str(device))
    if k in _LOADED:
        return _LOADED[k]
    numerics()
    if mock:
        tok, model = _mock_model(name, root)
    else:
        if str(device) == "cpu" and os.environ.get("R1B_ALLOW_CPU_REAL") != "1":
            raise RuntimeError(f"refusing to load the real {name} on the CPU (code tests use MOCK=1)")
        if name in LOADERS:
            tok, model = LOADERS[name](root=root, device=device, mock=False)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            d = ensure_model(name, root, freeze, download=True)
            tok = AutoTokenizer.from_pretrained(d)
            model = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32, attn_implementation="sdpa").eval()
    model = model.to(device); model.requires_grad_(False)
    tok.padding_side = "left"
    _LOADED[k] = (tok, model)
    return tok, model


def unload_models():
    _LOADED.clear()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_device(model):
    return next(model.parameters()).device


def layers_of(model):
    """The decoder blocks (ModuleList) of a HF causal LM (Qwen3: model.model.layers; Qwen3.5 text or multimodal wrappers)."""
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers", "model.model.layers"):
        m = model
        try:
            for a in path.split("."):
                m = getattr(m, a)
            return m
        except AttributeError:
            continue
    raise AttributeError("no decoder layers found")


def chat_text(tok, user):
    """Chat template, thinking off (the empty think block), as in rounds 1 and 2."""
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def enc(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def chat_ids(tok, user):
    return enc(tok, chat_text(tok, user))


def user_start(tok):
    """Index of the first user-content token in a chat sequence (rk_loop's START)."""
    return len(enc(tok, "<|im_start|>user\n"))


def pad_left(seqs, pad_id):
    """rk_loop.pad_batch: left padding; returns ids [B, T], attention mask, pad counts [B]."""
    torch = _torch()
    T = max(len(s) for s in seqs); ids = torch.full((len(seqs), T), pad_id); am = torch.zeros(len(seqs), T, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, T - len(s):] = torch.tensor(s); am[i, T - len(s):] = 1
    return ids, am, torch.tensor([T - len(s) for s in seqs])


def stop_tokens(tok, model):
    """rk_loop's EOS (generation config's eos plus <|im_end|>) and STOP (EOS plus the pad token)."""
    e = model.generation_config.eos_token_id
    e = list(e) if isinstance(e, (list, tuple)) else [e]
    eos = sorted(set(e) | {tok.convert_tokens_to_ids("<|im_end|>")})
    return eos, set(eos) | {tok.pad_token_id}


def generate(model, tok, rows, T=0.7, max_new=40, batch=32, setup=None):
    """Keyed Gumbel-max decoding (rk_loop.generate with a temperature argument): rows = (prompt_ids, key[, ...]); setup(rows_of_batch,
    pad) returns a context manager (e.g. Hooks(model).push(...)) active during the batch. Returns continuations without the stop token."""
    torch = _torch()
    from transformers import GenerationConfig, LogitsProcessorList
    dev = model_device(model); G = gumbel_class(dev, T); eos, stop = stop_tokens(tok, model)
    gc = GenerationConfig(max_new_tokens=max_new, do_sample=False, eos_token_id=eos, pad_token_id=tok.pad_token_id)
    conts = []
    for i in range(0, len(rows), batch):
        rb = rows[i:i + batch]; ids, am, pad = pad_left([r[0] for r in rb], tok.pad_token_id)
        cm = setup(rb, pad) if setup else contextlib.nullcontext()
        with cm, torch.no_grad():
            g = model.generate(input_ids=ids.to(dev), attention_mask=am.to(dev), generation_config=gc,
                               logits_processor=LogitsProcessorList([G([r[1] for r in rb])])).cpu()
        for row in g[:, ids.shape[1]:].tolist():
            c = []
            for t in row:
                if t in stop:
                    break
                c.append(t)
            conts.append(c)
    return conts


@contextlib.contextmanager
def sdpa_efficient(device):
    """SG9 / G-LCE-2: SDPA restricted to the memory-efficient kernel (a math fallback raises). On the CPU (MOCK) not asserted."""
    if str(device).startswith("cuda"):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            yield "efficient"
    else:
        yield "cpu: not asserted"


# ====================================================================================================================================
# Hooks: push, neuron operations on the MLP intermediate, capture, stopped reads
# ====================================================================================================================================
class Stop(Exception):
    pass


def unit(v):
    """ufo_common.unit (rk_loop's): v / (|v| + 1e-9)."""
    return v / (v.norm(dim=-1, keepdim=True) + 1e-9)


class Hooks:
    """Forward hooks on a decoder, removed on exit. Hooks on one block run in registration order (a capture registered after a push at
    the same block sees the pushed state). rowstart [B] (pad + user_start) limits an operation to positions >= rowstart in a forward of
    T > 1 tokens; decode steps (T == 1) are always included (rk_loop's rule). offset: absolute position of the first token of the next
    forward when a prefix is in the cache (chunked prefill; 0 otherwise).
      push(block, d, alpha, signs=1.0, cap=None, rowstart=None, V=None, name='push')   x + alpha * min(|x|, cap) * sign * unit(d) at the
            block output (rk_loop's steer_hook, float32 unit); V [B, D]: per-row vectors x + min(|x|, cap) * V; capped_share(name)
      neuron(block, idx, op, value=None, frac=1.0, rowstart=None, mask=None, name=None)  on the input of mlp.down_proj (the intermediate
            act(gate) * up): op 'read' (records [B, T, n] to self.reads[name]), 'scale' (x * value; value may be per neuron or
            [B, T, n]), 'add', 'clamp' (= value), 'zero', 'mean' ((1 - frac) x + frac value: partial mean ablation); mask [B, T] bool
      capture(blocks, positions=None, dtype=float32, cpu=True)   block outputs per forward into self.states[b] (positions: index tensor)
      stop_after(block)   the forward raises Stop after that block (use hooks.run(lambda: model.model(...)))"""

    def __init__(self, model, offset=0):
        self.model, self.L, self.offset, self.h = model, layers_of(model), int(offset), []
        self.states, self.reads, self.stats = {}, {}, {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.remove()

    def remove(self):
        for h in self.h:
            h.remove()
        self.h = []

    def _pos_mask(self, x, rowstart, mask):
        torch = _torch()
        if mask is not None:
            return mask.to(x.device).bool()
        if rowstart is None or x.shape[1] == 1:
            return None
        pos = self.offset + torch.arange(x.shape[1], device=x.device)
        return pos[None, :] >= torch.as_tensor(rowstart).to(x.device)[:, None]

    def push(self, block, d, alpha, signs=1.0, cap=None, rowstart=None, V=None, name="push"):
        torch = _torch()
        v = None if d is None else unit(torch.as_tensor(d).float())
        st = self.stats.setdefault(name, dict(ncap=0, nall=0))
        rs = None if rowstart is None else torch.as_tensor(rowstart)

        def hook(mod, inp, out):
            x = out[0] if isinstance(out, tuple) else out; B, T = x.shape[:2]
            m = ((self.offset + torch.arange(T, device=x.device))[None, :] >= rs.to(x.device)[:, None]).to(x.dtype)[..., None] \
                if (T > 1 and rs is not None) else torch.ones(B, T, 1, dtype=x.dtype, device=x.device)
            raw = x.norm(dim=-1, keepdim=True).detach(); nrm = raw.clamp(max=cap) if cap is not None else raw
            if V is not None:
                add = nrm * V.to(x.device)[:, None, :] * m
            else:
                sg = torch.as_tensor(signs, dtype=torch.float32)
                sg = sg.expand(B) if sg.dim() == 0 else sg
                if cap is not None:
                    st["ncap"] = st["ncap"] + ((raw > cap) & (m > 0)).sum()
                st["nall"] = st["nall"] + (m > 0).sum()
                add = alpha * nrm * sg.to(x.device, x.dtype)[:, None, None] * v.to(x.device, x.dtype) * m
            x = x + add
            return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
        self.h.append(self.L[block].register_forward_hook(hook))
        return self

    def capped_share(self, name="push"):
        st = self.stats.get(name) or {}
        return float(st.get("ncap", 0)) / max(float(st.get("nall", 0)), 1.0)

    def neuron(self, block, idx, op, value=None, frac=1.0, rowstart=None, mask=None, name=None):
        torch = _torch()
        assert op in ("read", "scale", "add", "clamp", "zero", "mean"), op
        ix = torch.as_tensor(idx, dtype=torch.long); key = name or f"{op}{block}"
        rec = self.reads.setdefault(key, []) if op == "read" else None

        def pre(mod, args):
            x = args[0]; i = ix.to(x.device); sel = x[..., i]; m = self._pos_mask(x, rowstart, mask)
            if op == "read":
                rec.append(dict(x=sel.detach().float().cpu(), mask=None if m is None else m.cpu())); return None
            w = None if value is None else torch.as_tensor(value, dtype=x.dtype, device=x.device)
            new = {"scale": lambda: sel * w, "add": lambda: sel + w, "clamp": lambda: w.expand_as(sel).clone(),
                   "zero": lambda: torch.zeros_like(sel), "mean": lambda: (1 - frac) * sel + frac * w}[op]()
            if m is not None:
                new = torch.where(m[..., None], new, sel)
            x = x.clone(); x[..., i] = new
            return (x,) + tuple(args[1:])
        self.h.append(self.L[block].mlp.down_proj.register_forward_pre_hook(pre))
        return self

    def capture(self, blocks, positions=None, dtype=None, cpu=True):
        torch = _torch()
        for b in blocks:
            store = self.states.setdefault(b, [])

            def hook(mod, inp, out, store=store):
                x = (out[0] if isinstance(out, tuple) else out).detach()
                if positions is not None:
                    x = x[:, torch.as_tensor(positions).to(x.device)]
                x = x.to(dtype or torch.float32)
                store.append(x.cpu() if cpu else x)
            self.h.append(self.L[b].register_forward_hook(hook))
        return self

    def stop_after(self, block):
        def hook(mod, inp, out):
            raise Stop
        self.h.append(self.L[block].register_forward_hook(hook))
        return self

    @staticmethod
    def run(fn):
        try:
            return fn()
        except Stop:
            return None


def mean_read(model, prompts, conts, block, pad_id, batch=32):
    """rk_loop.read for any block: mean block-b state of the model over the continuation tokens of prompt + continuation (stopped
    forward, float64 [N, D]; NaN row for an empty continuation)."""
    torch = _torch()
    dev = model_device(model); out = []
    for i in range(0, len(conts), batch):
        pb, cb = prompts[i:i + batch], conts[i:i + batch]
        ids, am, pad = pad_left([p + c for p, c in zip(pb, cb)], pad_id)
        with Hooks(model) as hk, torch.no_grad():
            hk.capture([block]).stop_after(block)
            hk.run(lambda: model.model(input_ids=ids.to(dev), attention_mask=am.to(dev)))
            hs = hk.states[block][0].double()
        for r, c in enumerate(cb):
            out.append(hs[r, ids.shape[1] - len(c):].mean(0) if len(c) else torch.full((hs.shape[-1],), float("nan"), dtype=torch.float64))
    return torch.stack(out)


# ====================================================================================================================================
# SO-DMG: damage functional and calibration
# ====================================================================================================================================
class RefSet:
    """Teacher-forced reference texts: items (ids, pos); the logits at position i are scored against ids[i + 1]. REF-R16 (R's export
    dmg_ref_R16.json: prompt_ids + cont_ids, scored on the continuation), REF-K16 (K's package file: ids + pos), REF-U32 = REF-R16 + REF-K16."""

    def __init__(self, items, name=""):
        self.items = [(list(map(int, i)), list(map(int, p))) for i, p in items if len(p)]
        self.name = name

    @classmethod
    def from_prompt_conts(cls, prompts, conts, name=""):
        return cls([(list(p) + list(c), list(range(len(p) - 1, len(p) + len(c) - 1))) for p, c in zip(prompts, conts)], name)

    @classmethod
    def from_json(cls, path, name=None):
        d = load_json(path)
        it = d["items"]
        if it and "prompt_ids" in it[0]:
            return cls.from_prompt_conts([x["prompt_ids"] for x in it], [x["cont_ids"] for x in it], name or os.path.basename(path))
        return cls([(x["ids"], x["pos"]) for x in it], name or os.path.basename(path))

    def __add__(self, other):
        return RefSet(self.items + other.items, f"{self.name}+{other.name}")

    def __len__(self):
        return len(self.items)

    def sha256(self):
        return sha256_text(canon(self.items))


class Damage:
    """SO-DMG functional: KL(unmodified || modified) in nats, teacher-forced, the mean over the texts of the per-text mean over its
    scored positions (r2_unlearn_qwen.RefKL / rk_loop.damage's formula), logits_to_keep on the scored positions; the unmodified
    log-probabilities are computed once. dmg(setup) with setup() -> a context manager (the intervention) returns dict(damage,
    per_text, top1_change, per_text_top1)."""

    def __init__(self, model, refs):
        torch = _torch()
        self.model, self.refs, self.items = model, refs, []
        dev = model_device(model)
        for ids, pos in refs.items:
            t = torch.tensor([ids], device=dev); p = torch.tensor(pos, device=dev)
            with torch.no_grad():
                a = torch.log_softmax(model(t, logits_to_keep=p).logits[0].double(), -1)
            self.items.append((t, p, a))

    def __call__(self, setup=None):
        torch = _torch()
        kl, t1 = [], []
        for t, p, a in self.items:
            with (setup() if setup else contextlib.nullcontext()), torch.no_grad():
                b = torch.log_softmax(self.model(t, logits_to_keep=p).logits[0].double(), -1)
            kl.append(float((a.exp() * (a - b)).sum(-1).mean())); t1.append(float((a.argmax(-1) != b.argmax(-1)).double().mean()))
        return dict(damage=float(np.mean(kl)), per_text=kl, top1_change=float(np.mean(t1)), per_text_top1=t1, refs=self.refs.name,
                    refs_sha256=self.refs.sha256())


def calibrate(dmg_fn, start, target, form="push", band=0.10, max_eval=None, cap=20.0, lo=None, hi=None):
    """SO-DMG calibrate rule. form 'push': f <- f sqrt(target / damage), at most 6 evaluations (r2_unlearn_qwen.calibrate); form 'gain'
    (K's NSCALE_DN): (a - 1) <- (a - 1) sqrt(target / damage), a in [0, cap = 20], at most 8. Stops in the band (+-10 % of the target);
    the value closest to the target is kept. Labels: 'damage out of band', 'dose capped'."""
    max_eval = max_eval or (6 if form == "push" else 8)
    lo = target * (1 - band) if lo is None else lo; hi = target * (1 + band) if hi is None else hi
    f, tried, capped = float(start), [], False
    for _ in range(max_eval):
        dm = float(dmg_fn(f))
        if not math.isfinite(dm):
            raise ValueError(f"damage is not finite at {f}")
        tried.append((f, dm))
        if lo <= dm <= hi:
            break
        r = math.sqrt(target / max(dm, 1e-6))
        if form == "push":
            f = f * r
        else:
            g = 1.0 + (f - 1.0) * r
            if g > cap or g < 0:
                g = min(max(g, 0.0), cap)
                if any(t[0] == g for t in tried):
                    capped = True; break
            f = g
    best = min(tried, key=lambda t: abs(t[1] - target)); inb = bool(lo <= best[1] <= hi)
    labels = ([] if inb else ["damage out of band"]) + (["dose capped"] if (capped or (form == "gain" and best[0] in (0.0, cap))) and not inb else [])
    return dict(f=best[0], damage=best[1], tried=tried, in_band=inb, start=float(start), target=target, form=form, labels=labels)


# ====================================================================================================================================
# SO-JLENS
# ====================================================================================================================================
def _jlens_maps(obj, n):
    torch = _torch()
    if isinstance(obj, torch.Tensor) and obj.dim() == 3 and obj.shape[0] == n:
        return obj
    if isinstance(obj, (list, tuple)) and len(obj) == n and all(isinstance(t, torch.Tensor) for t in obj):
        return torch.stack(list(obj))
    if isinstance(obj, dict):
        ks = {str(k).split("_")[-1]: v for k, v in obj.items()}
        if all(str(i) in ks for i in range(n)) and all(isinstance(ks[str(i)], torch.Tensor) for i in range(n)):
            return torch.stack([ks[str(i)] for i in range(n)])
        for v in obj.values():
            if isinstance(v, (torch.Tensor, list, tuple, dict)):
                try:
                    return _jlens_maps(v, n)
                except ValueError:
                    pass
    raise ValueError(f"J-lens layout not recognised ({type(obj).__name__})")


def load_jlens(path=None, sha256=JLENS["sha256"], mock=None, D=64, n=JLENS["n"]):
    """SO-JLENS: the Neuronpedia J-lens, sha256-pinned (a mismatch raises); returns dict(maps [27, D, D] (J[l]: block l -> block 27; its
    index convention is K4j's, jlens_convention), sha256, path). MOCK without a file: seeded random maps of the mock size."""
    torch = _torch()
    mock = MOCK if mock is None else mock
    if mock and (not path or not os.path.exists(path)):
        g = torch.Generator().manual_seed(7)
        return dict(maps=(torch.randn(n, D, D, generator=g) / D ** 0.5).half(), sha256="MOCK", path=None)
    h = sha256_file(path)
    if sha256 and h != sha256:
        raise ValueError(f"J-lens sha256 {h} != pinned {sha256} ({path})")
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        obj = torch.load(path, map_location="cpu", weights_only=False)      # the file is sha256-pinned
    return dict(maps=_jlens_maps(obj, n), sha256=h, path=path)


def jlens_convention(ctx):
    """'output' or 'input' (J[l] maps the output or the input of block l), decided by K4j in r1b_K_S1 (file k4j.json)."""
    return load_json(ctx.input("r1b_K_S1", "k4j.json"))["convention"]


def fetch_dataset(name, root, freeze):
    """A pinned dataset of runde1b_freeze.json['datasets'][name] (repo, revision, allow_patterns, sha256 {file: sha256}) under
    <root>/data/<name>, every listed sha256 checked."""
    rec = (freeze or {}).get("datasets", {}).get(name)
    if not rec:
        raise RuntimeError(f"dataset {name} not pinned in runde1b_freeze.json")
    d = os.path.join(root, "data", name)
    if not all(os.path.exists(os.path.join(d, f)) for f in rec.get("sha256", {})):
        from huggingface_hub import snapshot_download
        snapshot_download(rec["repo"], repo_type="dataset", revision=rec["revision"], local_dir=d, allow_patterns=rec.get("allow_patterns"))
    bad = [f for f, h in rec.get("sha256", {}).items() if sha256_file(os.path.join(d, f)) != h]
    if bad:
        raise RuntimeError(f"dataset {name}: sha256 mismatch {bad}")
    return d


# ====================================================================================================================================
# Ctx: what a step sees; provenance (R9.1) and writing
# ====================================================================================================================================
class NotRun(Exception):
    """Raised by a step: its measurements read 'not run (<reason>)' (code NA:not_run:<reason>); the step counts as finished."""

    def __init__(self, reason):
        super().__init__(reason); self.reason = reason


class StepYield(Exception):
    """Raised by a step that saved its progress (ctx.cached) and stops before the time guard; the session continues later."""


_MODHASH = {}


def module_hashes():
    """sha256 of every loaded module under analysis/ju_space (round-1, round-2 and round-1b code), relative to the repository."""
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if isinstance(f, str) and os.path.isabs(f) and f.endswith(".py") and f.startswith(AJS + os.sep) and os.path.isfile(f):
            rel = os.path.relpath(f, REPO)
            if rel not in _MODHASH:
                _MODHASH[rel] = sha256_file(f)
    return dict(sorted(_MODHASH.items()))


def versions():
    v = dict(python=platform.python_version(), numpy=np.__version__)
    for p in ("torch", "transformers"):
        try:
            v[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            v[p] = None
    t = sys.modules.get("torch")
    if t is not None and t.cuda.is_available():
        v.update(cuda=t.version.cuda, gpu=t.cuda.get_device_name(0))
    return v


PROV_REQUIRED = ("session", "attempt", "plan", "code_commit", "modules", "config_sha256", "versions", "keys", "tier", "written")


class Ctx:
    """The step context (made by the runner; Ctx.from_env() inside a script step).
    session, attempt, step, part, account, tier, root, work (the session folder, synced to Drive), inp (copied inputs, read-only),
    r1 / r2 (round-1 / round-2 copies), device, mock, quick, pilot, freeze (runde1b_freeze.json), manifest, declared (producers
    declared 'not run'), t0 and guard_s (per-session time guard)."""

    def __init__(self, session, work, inp=None, root=None, device=None, attempt=None, account=None, mock=None, quick=None, pilot=None,
                 freeze=None, manifest=None, declared=(), t0=None, guard_s=4.5 * 3600, step=None):
        meta = C.SESSIONS.get(session, {})
        self.session, self.work, self.inp = session, os.path.abspath(work), os.path.abspath(inp or os.path.join(work, "in"))
        self.root = root or os.environ.get("R1B_ROOT", "/content/root")
        self.device = device or os.environ.get("DEVICE", "cuda")
        self.attempt, self.account = attempt or session, account or meta.get("account")
        self.part, self.tier = meta.get("part"), meta.get("tier")
        self.mock = MOCK if mock is None else mock
        self.quick = QUICK if quick is None else quick
        self.pilot = (session == "r1b_P0") if pilot is None else pilot
        self.freeze, self.manifest, self.declared = freeze or {}, manifest or {}, tuple(declared)
        self.t0, self.guard_s, self.step = t0 or time.time(), guard_s, step
        self.r1, self.r2 = os.path.join(self.inp, "r1"), os.path.join(self.inp, "r2")
        self.models_used, self.keys_used, self.jlens_sha256, self.step_started = {}, set(), None, iso()

    @classmethod
    def from_env(cls):
        e = os.environ
        return cls(e["R1B_SESSION"], e["R1B_WORK"], inp=e.get("R1B_IN"), root=e.get("R1B_ROOT"), device=e.get("DEVICE"),
                   attempt=e.get("R1B_ATTEMPT"), account=e.get("R1B_ACCOUNT"), freeze=load_json(e.get("R1B_FREEZE", ""), {}),
                   manifest=load_json(e.get("R1B_MANIFEST", ""), {}), declared=[x for x in e.get("R1B_DECLARED", "").split(",") if x],
                   t0=float(e.get("R1B_T0") or time.time()), guard_s=float(e.get("R1B_GUARD_S", 4.5 * 3600)), step=e.get("R1B_STEP"))

    # ---- paths and inputs
    def path(self, rel):
        p = os.path.join(self.work, rel); os.makedirs(os.path.dirname(p), exist_ok=True); return p

    def input(self, session, rel, required=True):
        """A copied file of an upstream session (<work>/in/<session>/<rel>); None if absent and not required; a required file of a
        producer declared 'not run' raises NotRun('input missing')."""
        p = os.path.join(self.inp, session, rel)
        if os.path.exists(p):
            return p
        if not required:
            return None
        if session in self.declared:
            raise NotRun("input missing")
        raise FileNotFoundError(f"input {session}/{rel} not copied (start conditions in r1b_contracts.SESSIONS['{self.session}'])")

    def xfer(self, name):
        p = os.path.join(self.inp, C.XFER_DIR, name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"A -> B export {name} not copied")
        return p

    def missing(self, session):
        return session in self.declared

    # ---- models, keys, funding, trims, time
    def model(self, name=C.M17):
        tok, m = load_model(name, root=self.root, device=self.device, mock=self.mock, freeze=self.freeze)
        p = model_pins(name, self.freeze)
        self.models_used[name] = "MOCK" if self.mock else dict(repo=p.get("repo"), revision=p.get("revision"), sha256=p.get("sha256"))
        return tok, m

    def jlens(self, path=None):
        J = load_jlens(path or os.path.join(self.root, "jlens", JLENS["name"]), mock=self.mock)
        self.jlens_sha256 = J["sha256"]
        return J

    def key(self, tag, *rest):
        """keyseed(tag, *rest) for a registered tag of this session's part (P0: 'r1bP0_' tags); no key is shared across parts."""
        o = key_owner(tag)
        want = "main" if self.session == "r1b_P0" else self.part
        if o is None or o != want:
            raise ValueError(f"key tag {tag!r} is not registered for part {want} (owner {o}); see r1b_core.KEYS")
        self.keys_used.add(tag)
        return keyseed(tag, *rest)

    def funded(self, item):
        f = self.freeze.get("funded") or {}
        return bool(f.get("tier2")) and (not f.get("items") or item in f["items"])

    def trimmed(self, trim):
        return trim in ((self.freeze.get("trims") or {}).get(self.part) or [])

    def time_left(self):
        return self.guard_s - (time.time() - self.t0)

    def log(self, msg):
        print(f"[{self.session}/{self.step or '-'} {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def cached(self, name, fn, fmt="pt"):
        """Resumable piece of a step: <work>/cache/<step>/<name>.<fmt> (synced to Drive) is loaded if present, else fn() is saved."""
        p = self.path(f"cache/{self.step or 'misc'}/{name}.{fmt}")
        if os.path.exists(p):
            return load_json(p) if fmt == "json" else _torch().load(p, map_location="cpu", weights_only=False)
        v = fn()
        if fmt == "json":
            dump_json(v, p)
        else:
            _torch().save(v, p + ".tmp"); os.replace(p + ".tmp", p)
        return v

    # ---- provenance and outputs (R9.1-R9.4, SG6)
    def provenance(self, config=None, seeds=None, labels=None):
        man, fz = self.manifest, self.freeze
        return dict(session=self.session, attempt=self.attempt, step=self.step, part=self.part, account=self.account, tier=self.tier,
                    reduction=(fz.get("trims") or {}).get(self.part), funded=fz.get("funded"), mock=self.mock, pilot=self.pilot,
                    plan=man.get("plan") or dict(note="no manifest (code test)"), code_commit=man.get("code_commit", "unknown"),
                    modules=module_hashes(), config=jsonable(config), config_sha256=sha256_text(canon(config)),
                    package=fz.get("package_files"), models=self.models_used, jlens_sha256=self.jlens_sha256,
                    inputs=man.get("inputs", []), versions=versions(),
                    keys=dict(namespace=KEYS.get("main" if self.pilot else self.part, {}).get("prefixes"), used=sorted(self.keys_used),
                              seeds=seeds),
                    labels=list(labels or []), started=self.step_started, written=iso())

    def _pilot_check(self, obj, rel):
        bad = set(obj) - PILOT_KEYS
        if self.pilot and bad:
            raise ValueError(f"r1b_P0 output {rel}: keys {sorted(bad)} outside the SO-PILOT whitelist {sorted(PILOT_KEYS)}")

    def write_json(self, rel, obj, config=None, seeds=None, labels=None):
        """JSON output with its provenance block; raw NaN refused (code it); r1b_P0: whitelisted top-level keys only."""
        obj = dict(obj) if isinstance(obj, dict) else dict(data=obj)
        self._pilot_check(obj, rel)
        obj["provenance"] = self.provenance(config, seeds, labels)
        dump_json(obj, self.path(rel))
        return self.path(rel)

    def _sidecar(self, rel, config, seeds, labels):
        if self.pilot:
            raise ValueError(f"r1b_P0 writes JSON engineering records only ({rel})")
        p = self.path(rel)
        dump_json(dict(file=os.path.basename(rel), sha256=sha256_file(p), provenance=self.provenance(config, seeds, labels)), p + ".prov.json")
        return p

    def save_pt(self, rel, obj, config=None, seeds=None, labels=None):
        """torch file (atomic) with the sidecar <rel>.prov.json (provenance and the file's sha256)."""
        p = self.path(rel); _torch().save(obj, p + ".tmp"); os.replace(p + ".tmp", p)
        return self._sidecar(rel, config, seeds, labels)

    def write_file(self, rel, data, config=None, seeds=None, labels=None):
        """Any other file (text or bytes, e.g. JSON lines) with its sidecar."""
        p = self.path(rel)
        with open(p + ".tmp", "wb" if isinstance(data, bytes) else "w") as f:
            f.write(data)
        os.replace(p + ".tmp", p)
        return self._sidecar(rel, config, seeds, labels)


EXIT_NOT_RUN, EXIT_YIELD = 75, 76      # script steps: exit codes for NotRun (reason in $R1B_STEP_STATUS) and StepYield


def run_script(fn):
    """Entry point of a script step: fn(Ctx.from_env()); NotRun -> exit 75 (reason written to $R1B_STEP_STATUS), StepYield -> 76."""
    ctx = Ctx.from_env()
    try:
        fn(ctx)
    except NotRun as e:
        if os.environ.get("R1B_STEP_STATUS"):
            dump_json(dict(not_run=e.reason), os.environ["R1B_STEP_STATUS"])
        raise SystemExit(EXIT_NOT_RUN)
    except StepYield:
        raise SystemExit(EXIT_YIELD)
    raise SystemExit(0)


def check_output(path):
    """SG6 on one output file: JSON with a complete provenance block and no raw NaN, or a sidecar <file>.prov.json. Returns a problem
    string or None."""
    if path.endswith(".prov.json"):
        return None
    if path.endswith(".json"):
        try:
            with open(path) as f:
                d = json.load(f, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"raw {c}")))
        except ValueError as e:
            return f"{os.path.basename(path)}: {e}"
        prov = d.get("provenance") if isinstance(d, dict) else None
    else:
        sc = load_json(path + ".prov.json")
        if not sc:
            return f"{os.path.basename(path)}: no sidecar .prov.json"
        if sc.get("sha256") != sha256_file(path):
            return f"{os.path.basename(path)}: sidecar sha256 differs"
        prov = sc.get("provenance")
    miss = [k for k in PROV_REQUIRED if not isinstance(prov, dict) or k not in prov]
    return f"{os.path.basename(path)}: provenance missing {miss}" if miss else None


# ====================================================================================================================================
# Self-test
# ====================================================================================================================================
def _ok(cond, msg, errs):
    print(("  ok    " if cond else "  FAIL  ") + msg, flush=True)
    if not cond:
        errs.append(msg)


def _rk_equiv(root, tmp):
    """Child process (MOCK=1): r1b_core against rk_loop itself (CC4 push and keyed generation, rk.read, SO-DMG = RefKL, calibrate)."""
    torch = _torch(); errs = []
    sys.argv = [os.path.join(AJS, "rk_loop.py"), root, tmp, "lib"]
    import rk_loop as rk
    import r2_unlearn_qwen as U
    rk.references()
    model, tok = rk.model, rk.tok; cap = float(rk.ST["ref"]["cap"]); conts = rk.ST["ref"]["conts"]
    d = torch.randn(rk.D, generator=torch.Generator().manual_seed(5)).double(); alpha = 0.4
    ref = U.RefKL(rk, n=8); dm = Damage(model, RefSet.from_prompt_conts(rk.PIDS[:8], conts[:8]))
    for s in (1.0, -1.0):
        a = ref.push(d, alpha, s)
        b = dm(lambda: Hooks(model).push(rk.BL, d, alpha, signs=s, cap=cap, rowstart=torch.tensor([rk.START])))["damage"]
        _ok(abs(a - b) <= 1e-6 and a > 0, f"SO-DMG = r2_unlearn_qwen.RefKL, sign {s:+.0f}: {a:.8f} vs {b:.8f}", errs)
    seqs = [rk.PIDS[p] + conts[p] for p in range(4)]; ids, am, pad = rk.pad_batch(seqs); sg = torch.tensor([1.0, -1.0, 1.0, -1.0])
    rk.STEER.update(v=unit(d.float()), V=None, alpha=alpha, cap=cap, sign=sg, rowstart=pad + rk.START)
    with torch.no_grad():
        la = model(input_ids=ids, attention_mask=am).logits
    rk.steer_off()
    with Hooks(model).push(rk.BL, d, alpha, signs=sg, cap=cap, rowstart=pad + rk.START) as hk, torch.no_grad():
        lb = model(input_ids=ids, attention_mask=am).logits
        cs = hk.capped_share()
    _ok(float((la - lb).abs().max()) <= 1e-6, f"CC4: Hooks.push = rk_loop push (padded batch, both signs), max diff "
        f"{float((la - lb).abs().max()):.2e}; capped share {cs:.3f}", errs)
    rows = [(rk.PIDS[p], rk.keyseed("r1bP0_selftest", p), sgn) for p in range(4) for sgn in (1, -1)]
    ga = rk.generate(rows, d, alpha)
    gb = generate(model, tok, rows, T=rk.TEMP, max_new=rk.NNEW,
                  setup=lambda rb, pd: Hooks(model).push(rk.BL, d, alpha, signs=torch.tensor([float(r[2]) for r in rb]), cap=cap,
                                                         rowstart=pd + rk.START))
    _ok(ga == gb and any(len(c) for c in ga), "keyed Gumbel generation with push = rk_loop.generate (token ids equal)", errs)
    Xa = rk.read(rk.PIDS[:6], conts[:6]); Xb = mean_read(model, rk.PIDS[:6], conts[:6], rk.BL, rk.PAD)
    _ok(float((Xa - Xb).abs().max()) <= 1e-6, f"mean_read = rk_loop.read, max diff {float((Xa - Xb).abs().max()):.2e}", errs)
    fn = lambda f: ref.push(d, f)
    rule = dict(target=0.05, lo=0.045, hi=0.055, max_eval=6)
    c2, c1 = U.calibrate(fn, 0.1, rule), calibrate(fn, 0.1, 0.05, form="push", lo=0.045, hi=0.055)
    _ok(c2["f"] == c1["f"] and c2["tried"] == c1["tried"] and c2["in_band"] == c1["in_band"],
        f"calibrate (push form) = r2_unlearn_qwen.calibrate ({len(c1['tried'])} evaluations, in band {c1['in_band']})", errs)
    return 1 if errs else 0


def selftest(root):
    torch = _torch(); errs = []
    print("[r1b_core selftest] contracts, keys, p-values, codes", flush=True)
    _ok(not C.check(), f"r1b_contracts.check() {C.check() or ''}", errs)
    _ok(not check_keys(), f"SO-KEYS registry unique {check_keys() or ''}", errs)
    KEYS["K"]["tags"] += ("r1bD1",)
    _ok(bool(check_keys()), "a planted cross-part tag collision is detected", errs)
    KEYS["K"]["tags"] = KEYS["K"]["tags"][:-1]
    try:
        register_r_loops(["axis"]); _ok(False, "R loop name 'axis' refused", errs)
    except ValueError:
        KEYS["R"]["loops"] = (); _ok(True, "R loop name 'axis' (forms 'r1b_axis') refused", errs)
    rc = rk_source_check()
    _ok(rc["ok"] and rc["in_rk_loop"], f"keyseed / Gumbel source = rk_loop.py lines (sha256 {RK_KEY_SRC_SHA256[:12]})", errs)
    _ok(keyseed("r1bD1", 3, "x") == int(hashlib.sha256("r1bD1|3|x".encode()).hexdigest()[:12], 16), "keyseed formula", errs)
    _ok(abs(p_one_sided(np.arange(2000) - 100.0, 0.0) - 102 / 2001) < 1e-12, "one-sided bootstrap p = (1 + #null) / 2001", errs)
    _ok(abs(p_rank(5.0, [1, 5, 7, 2]) - 3 / 5) < 1e-12, "rank p", errs)
    _ok(abs(p_pi(0.0, [-1.0, 1.0, np.nan]) - 0.5) < 1e-12 and p_pi(3.0, [0.0, 1.0, 2.0]) < 0.2, "prediction-interval p (K.4)", errs)
    a, r = holm([0.01, 0.04, 0.03]); _ok(np.allclose(a, [0.03, 0.06, 0.06]) and r == [True, False, False], "Holm", errs)
    a, r = bh([0.01, 0.04, 0.03, 0.5], 0.1); _ok(np.allclose(a, [0.04, 0.0533333, 0.0533333, 0.5], atol=1e-6), "BH", errs)
    x = np.r_[np.arange(90.0), [np.nan] * 10]; lo, _ = interval(x)
    _ok(isinstance(lo, dict) and lo["code"] == "NaN:fewer_than_95pct_finite_replicates", "interval undefined with > 5 % NaN", errs)
    W = prompt_bootstrap(8, B=50, seed=1); _ok(W.shape == (50, 8) and np.all(W[:, :4].sum(1) == 4), "prompt bootstrap stratified", errs)
    try:
        dump_json({"a": [1.0, float("nan")]}, os.path.join(root_tmp, "x.json")); _ok(False, "raw NaN refused", errs)
    except ValueError as e:
        _ok("$.a[1]" in str(e), "raw NaN refused with its path", errs)
    _ok(code_counts({"a": coded("NaN:empty_text"), "b": [coded("NA:gate_closed:SG10"), val(float("inf"), "NaN:zero_norm")]}) ==
        {"NaN:empty_text": 1, "NA:gate_closed:SG10": 1, "NaN:zero_norm": 1}, "codes and counts", errs)
    dd = dist([1, 2, 3, np.nan, 5], groups=["a", "a", "b", "b", "b"]); _ok(dd["n_nonfinite"] == 1 and dd["groups"]["b"]["n_finite"] == 2, "dist", errs)
    print("[r1b_core selftest] MOCK model, hooks, damage, J-lens, Ctx", flush=True)
    tok, model = load_model(C.M17, root=root, device="cpu", mock=True)
    st = user_start(tok); p1, p2 = chat_ids(tok, "Describe a quiet harbour."), chat_ids(tok, "Name a tree.")
    ids, am, pad = pad_left([p1 + [11, 12, 13], p2], tok.pad_token_id); rs = pad + st
    with torch.no_grad():
        base = model(input_ids=ids, attention_mask=am, output_hidden_states=True)
        with Hooks(model).neuron(5, [1, 7, 9], "scale", 1.0):
            l1 = model(input_ids=ids, attention_mask=am).logits
        with Hooks(model).neuron(5, [1, 7, 9], "zero", rowstart=rs):
            lz = model(input_ids=ids, attention_mask=am).logits
        with Hooks(model).neuron(5, [1, 7, 9], "clamp", torch.zeros(3), rowstart=rs):
            lc = model(input_ids=ids, attention_mask=am).logits
        with Hooks(model).neuron(5, [1, 7, 9], "mean", torch.ones(3), frac=0.0):
            lm = model(input_ids=ids, attention_mask=am).logits
        with Hooks(model).neuron(5, [1, 7, 9], "read", name="r") as hk:
            model(input_ids=ids, attention_mask=am); rd = hk.reads["r"][0]["x"]
        with Hooks(model) as hk:
            hk.capture([4]).stop_after(4)
            out = hk.run(lambda: model.model(input_ids=ids, attention_mask=am)); cap4 = hk.states[4][0]
    _ok(torch.equal(l1, base.logits) and torch.equal(lm, base.logits), "neuron scale 1 and mean-ablation frac 0 leave logits unchanged", errs)
    _ok(torch.equal(lz, lc) and not torch.equal(lz, base.logits), "neuron zero = clamp 0 (from the first user token), and it acts", errs)
    _ok(tuple(rd.shape) == (2, ids.shape[1], 3), f"neuron read shape {tuple(rd.shape)}", errs)
    _ok(out is None and torch.allclose(cap4, base.hidden_states[5], atol=0, rtol=0), "stopped read at block 4 = hidden_states[5]", errs)
    refs = RefSet.from_prompt_conts([p1, p2], [[11, 12, 13, 14], [21, 22, 23]], "mockref"); dm = Damage(model, refs)
    _ok(dm()["damage"] == 0.0, "damage without intervention = 0", errs)
    d = torch.randn(64, generator=torch.Generator().manual_seed(3))
    fn = lambda f: dm(lambda: Hooks(model).push(14, d, f, cap=1e9, rowstart=torch.tensor([st])))["damage"]
    c = calibrate(fn, 0.5, DELTA1)
    _ok(c["in_band"] or c["labels"] == ["damage out of band"], f"push calibration to delta1: f {c['f']:.4g}, damage {c['damage']:.4g}, "
        f"{len(c['tried'])} evaluations", errs)
    g = calibrate(lambda a: dm(lambda: Hooks(model).neuron(10, list(range(64)), "scale", a))["damage"], 1.5, DELTA1, form="gain")
    _ok(len(g["tried"]) <= 8 and 0 <= g["f"] <= 20, f"gain calibration (NSCALE_DN form): alpha {g['f']:.4g}, damage {g['damage']:.4g}, "
        f"labels {g['labels']}", errs)
    cg = calibrate(lambda a: 1e-4 * (a - 1) ** 2, 2.0, DELTA2, form="gain")
    _ok(cg["f"] == 20.0 and "dose capped" in cg["labels"], "gain calibration capped at 20 with 'dose capped'", errs)
    jp = os.path.join(root_tmp, "jl.pt"); torch.save({f"layer_{i}": torch.eye(4).half() for i in range(27)}, jp)
    J = load_jlens(jp, sha256=sha256_file(jp), mock=False); _ok(tuple(J["maps"].shape) == (27, 4, 4), "J-lens layout dict layer_i", errs)
    try:
        load_jlens(jp, mock=False); _ok(False, "J-lens sha256 pin", errs)
    except ValueError:
        _ok(True, "J-lens sha256 pin: a different file is refused", errs)
    ctx = Ctx("r1b_K_S1", os.path.join(root_tmp, "work"), root=root, device="cpu", mock=True,
              freeze=dict(funded=dict(tier2=True, items=["X2"]), trims=dict(K=["K1"])))
    _ok(ctx.key("r1bK_hn", 1) == keyseed("r1bK_hn", 1) and ctx.funded("X2") and not ctx.funded("X3") and ctx.trimmed("K1"),
        "Ctx keys, funding, trims", errs)
    try:
        ctx.key("r1bD1", 1); _ok(False, "foreign key tag refused", errs)
    except ValueError:
        _ok(True, "a key tag of another part is refused", errs)
    ctx.step = "t"; p = ctx.write_json("t/a.json", {"x": val(float("nan"), "NaN:empty_text"), "d": dist([1, 2])}, config={"n": 2})
    _ok(check_output(p) is None, "write_json: provenance complete (SG6)", errs)
    p2_ = ctx.save_pt("t/b.pt", {"u": torch.ones(3)}); _ok(check_output(p2_) is None, "save_pt sidecar (SG6)", errs)
    dump_json({"x": 1}, os.path.join(root_tmp, "noprov.json")); _ok(check_output(os.path.join(root_tmp, "noprov.json")) is not None,
                                                                      "an output without provenance fails SG6", errs)
    n = [0]
    v1 = ctx.cached("piece", lambda: (n.__setitem__(0, n[0] + 1), {"a": 1})[1], fmt="json")
    v2 = ctx.cached("piece", lambda: (n.__setitem__(0, n[0] + 1), {"a": 2})[1], fmt="json")
    _ok(v1 == v2 == {"a": 1} and n[0] == 1, "ctx.cached resumes", errs)
    pc = Ctx("r1b_P0", os.path.join(root_tmp, "p0"), root=root, device="cpu", mock=True)
    try:
        pc.write_json("x.json", {"texts": [1, 2]}); _ok(False, "pilot whitelist", errs)
    except ValueError:
        _ok(True, "r1b_P0: a non-whitelisted key is refused", errs)
    print("[r1b_core selftest] against rk_loop (child process, MOCK=1)", flush=True)
    import subprocess
    env = dict(os.environ, MOCK="1", BLOCK="14", DEVICE="cpu", NT=os.environ.get("NT", "1"), PYTHONUNBUFFERED="1")
    env.pop("ASCALE", None)
    r = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "_rk_equiv", root, os.path.join(root_tmp, "rk")], env=env)
    _ok(r.returncode == 0, "rk_loop equivalence child", errs)
    print("SELFTEST " + ("OK" if not errs else f"FAILED: {errs}"), flush=True)
    return 1 if errs else 0


root_tmp = None

if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "selftest":
        import tempfile
        root_tmp = tempfile.mkdtemp(prefix="r1b_core_st_")
        raise SystemExit(selftest(a[1] if len(a) > 1 else os.environ.get("R1B_ROOT", "/content/root")))
    if a and a[0] == "_rk_equiv":
        raise SystemExit(_rk_equiv(a[1], a[2]))
    if a and a[0] == "fetch-models":
        fz = load_json(os.path.join(REPO, C.PLAN_DIR, "runde1b_freeze.json"), {})
        for nm in a[2:]:
            print(nm, ensure_model(nm, a[1], fz))
        raise SystemExit(0)
    if a and a[0] == "keys":
        r = write_keys(a[1]); print("problems:", r["problems"]); raise SystemExit(1 if r["problems"] else 0)
    print(__doc__)
    raise SystemExit(2)

"""Part L base: sizes, templates, contract inputs (SO-HN, SO-LAB, K4 gating, SO-LENSOUT, SO-D1STIM, SO-DMG, SO-AX, SO-LONGDOC,
SO-WT103) read in the contract formats, MOCK stand-ins when an input is a MOCK placeholder (MOCK=1 only), scorers (K's if importable,
else L's frozen fallback), populations (L.POP) and output helpers (R9.1-R9.4)."""
import hashlib, json, math, os, re
import numpy as np

import r1b_contracts as C
import r1b_core as K

TAGS = ("r1bL0", "r1bL2", "r1bL2b", "r1bL2s", "r1bL3", "r1bL3t", "r1bL4", "r1bL5", "r1bLX")
QA_TEMPLATE = "Answer the question, then explain your answer in detail, in about 800 words: {q}"
NEU_TEMPLATE = "Write a detailed story of about 800 words about {s}"
SHORT_QA = "{q}\nAnswer with only the answer."            # K2's sampling turn (REF answers, erasure items)
COT_INSTR = "Work it out step by step, one operation per line, then give the final number."
SUB_TEMPLATE = "Compute {e}. Reply with only the final number."
NAT_PROMPT = "Write a detailed text."
# F-neu (L.R-A): rk_loop's 32 scene prompts as noun phrases (the template needs a noun phrase; deviation noted in the part file)
# followed by 32 new everyday scenes. Frozen here; sha256 in every output's config.
F_NEU_SCENES = (
    "a quiet morning in a small town", "the history of bread", "how rain forms", "a famous bridge", "the taste of a ripe peach",
    "packing for a weekend hike", "a note to a neighbour about a lost cat", "why leaves change colour in autumn", "a busy train station",
    "growing tomatoes", "a lighthouse", "how a bicycle stays upright", "the sound of the sea", "a typical fairy tale",
    "what a public library does", "a winter market", "owls", "how bees make honey", "an ideal sandwich", "a seaside holiday",
    "boiling an egg", "an old wooden door", "a walk in the forest", "why the sky is blue", "a thunderstorm",
    "a first day at a new school", "a river", "how a mechanical clock works", "a market stall selling spices", "mountains",
    "how a kite flies", "a cup of hot chocolate",
    "a rainy afternoon at a bus stop", "a family dinner on a Sunday", "repairing a leaking tap", "a morning at a fish market",
    "a long queue at the post office", "learning to bake pancakes", "a picnic in a city park", "a child's first bicycle ride",
    "cleaning out an attic", "a village football match", "a night shift at a bakery", "planting a small herb garden",
    "a power cut in an apartment block", "a trip to the dentist", "washing a dog", "a crowded summer beach",
    "a second-hand bookshop", "painting a garden fence", "a school science fair", "waiting for a delayed train",
    "a farmer's morning routine", "a snowy road in the mountains", "a neighbourhood yard sale", "a bus driver's day",
    "a small bakery opening for the day", "a kitchen after a party", "a birthday cake that went wrong", "a quiet library reading room",
    "a fishing trip on a lake", "moving into a new flat", "a busy hospital waiting room", "an evening walk along a canal")
ATTR_NAMES = ("founder's first name", "emblem animal", "main export", "flag colour", "founding month", "bell metal")   # D's six slots
REFUSAL_LEXICON = ("i don't know", "i do not know", "i'm not sure", "i am not sure", "not sure", "no information", "i cannot",
                   "i can't", "unable to", "not aware", "no record", "couldn't find", "could not find", "does not exist",
                   "doesn't exist", "fictional", "no such", "unknown", "not familiar", "no known", "i'm sorry", "i apologize",
                   "there is no", "not a real", "no reliable")


def sha(obj):
    return K.sha256_text(K.canon(obj))


def cfg(ctx):
    """Planned sizes (L.3, L.4) or MOCK sizes (code tests only)."""
    full = dict(n_fam=64, lr_new=1024, lrb_items=32, lrb_keys=8, lrb_new=512, n_ctrl=99, nat=200, nat_len=1024, win0=64, win=32,
                pos_scale=1.0, batch=32, pip_frag=50, pip_perm=100, pip_grid=[2 ** i for i in range(11)], syn_grid=[8 * 2 ** i for i in range(7)],
                l1c_seeds=3, l1c_frag=20, l1c_perm=50, n_resel=6, n_boot_sets=20, era_fr=[2 ** -12, 2 ** -10, 2 ** -8, 2 ** -6, 2 ** -4],
                era_draws=3, era_sets=8, era_new=24, l3b_prompts=128, l3b_seeds=2, l3b_turn=64, l3b_win=16, l3b_sets=6, l3b_sham=32,
                l3b_turns=5, l4_items=32, l4_seeds=4, l4_msgs=12, l4_new=40, l4_sets=4, refw_n=200, refw_len=512,
                eqmi_L=[2 ** i for i in range(7, 16)], eqmi_docs=24, eqmi_chunk=2048, own_seed=512, own_new=4096,
                own_L=[2 ** i for i in range(7, 13)], gmi_copy=4, gmi_shuf=8, n_sur_P=200, n_sur_C=50, bi_sur=200, bi_k=(256, 512, 320),
                gibbs=(512, 4000, 1000), l5_items=600, l5_cot_new=96, l5_sub_new=12, x1_new=80, x1_rows=256, x4_keys=256, x5_prompts=32,
                x5_new=64, x3_steps=6000, x3_seeds=5, x3_width=128, xmap_libs=[32, 128, 512, 896], xmap_draws=20, xmap_pos=(64, 1024),
                sigma_n=4000, nboot=2000, ref_heldout=64, max_spin=200, xmap_E=(2, 3, 4, 5, 6), t2_items=128, docs_min=36000,
                long_skip=2000, long_len=32768)
    if not ctx.mock:
        return full
    m = dict(full)
    m.update(n_fam=4, lr_new=48, lrb_items=3, lrb_keys=8, lrb_new=40, n_ctrl=9, nat=4, nat_len=48, win0=8, win=8, pos_scale=1 / 16,
             batch=8, pip_frag=3, pip_perm=4, pip_grid=[1, 2, 4, 8, 16], syn_grid=[2, 4, 8], l1c_seeds=1, l1c_frag=3, l1c_perm=3, n_resel=3,
             n_boot_sets=4, era_fr=[2 ** -8, 2 ** -6, 2 ** -4], era_draws=1, era_sets=2, era_new=6, l3b_prompts=4, l3b_seeds=2, l3b_turn=12,
             l3b_win=4, l3b_sets=2, l3b_sham=4, l3b_turns=3, l4_items=4, l4_seeds=2, l4_msgs=4, l4_new=8, l4_sets=2, refw_n=4, refw_len=32,
             eqmi_L=[16, 32, 64, 128], eqmi_docs=3, eqmi_chunk=32, own_seed=16, own_new=96, own_L=[16, 32, 64], gmi_copy=2, gmi_shuf=2,
             n_sur_P=6, n_sur_C=3, bi_sur=6, bi_k=(16, 40, 20), gibbs=(16, 30, 10), l5_items=8, l5_cot_new=24, l5_sub_new=6, x1_new=24,
             x1_rows=8, x4_keys=6, x5_prompts=4, x5_new=12, x3_steps=60, x3_seeds=2, x3_width=16, xmap_libs=[8, 16], xmap_draws=2,
             xmap_pos=(8, 48), sigma_n=8, nboot=200, ref_heldout=4, max_spin=60, xmap_E=(2, 3), t2_items=4, docs_min=400, long_skip=16,
             long_len=160)
    return m


BIN_EDGES = (0, 16, 32, 64, 128, 256, 512, 768, 1024)


def pos_bin(k, scale=1.0):
    """Position bin of continuation index k (1-based): [1-16], [17-32], [33-64], [65-128], [129-256], [257-512], [513-768], [769-1024]."""
    e = np.array(BIN_EDGES[1:-1]) * scale
    return np.searchsorted(e, np.asarray(k) - 0.5)


# ==================================================================================================================================
# output helpers
# ==================================================================================================================================
def clean(x):
    """JSON-safe: numpy to python, non-finite floats coded (R9.2)."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else K.coded("NaN:undefined")
    return K.jsonable(x)


def wj(ctx, rel, obj, config=None, labels=(), seeds=None):
    cf = dict(config or {}); cf.setdefault("part", "L"); cf.setdefault("mock", ctx.mock)
    return ctx.write_json(rel, clean(obj), config=clean(cf), seeds=seeds, labels=list(labels))


def save(ctx, rel, obj, config=None, labels=()):
    return ctx.save_pt(rel, obj, config=clean(config or {}), labels=list(labels))


def loadpt(path):
    return K._torch().load(path, map_location="cpu", weights_only=False)


def is_mock_file(d):
    return isinstance(d, dict) and d.get("mock") is True and "fid" in d


def rng(*key):
    """numpy Generator from a string seed (L's seeds 'r1bL1rot', 'r1bL-ERA', ... are string seeds, main §2.3)."""
    return np.random.default_rng(int(hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()[:12], 16))


# ==================================================================================================================================
# contract inputs
# ==================================================================================================================================
def _need_mock(ctx, d, what):
    if d is None or is_mock_file(d):
        if not ctx.mock:
            raise K.NotRun(f"input missing ({what})")
        return True
    return False


def dims(model):
    L = len(K.layers_of(model)); I = K.layers_of(model)[0].mlp.down_proj.in_features
    return L, I


def flat(pairs, I):
    return sorted({int(l) * I + int(i) for l, i in pairs})


def hn(ctx, model):
    """SO-HN (hn_set.json, hn_meta.json, sg8_preview.json, hn_detectors.pt): P per L.1, H_tq, H+, boot sets, C*, SG8 preview, REF-W
    (None if absent: L's fallback). Neurons as flat ids layer * I + index."""
    L, I = dims(model)
    d = K.load_json(ctx.input("r1b_K_S1", "hn/hn_set.json"))
    meta = K.load_json(ctx.input("r1b_K_S1", "hn/hn_meta.json", required=False) or "", {}) or {}
    sg8 = K.load_json(ctx.input("r1b_K_S1", "sg8_preview.json", required=False) or "", {}) or {}
    labels = []
    if _need_mock(ctx, d, "hn_set"):
        g = rng("mock_hn")
        htq = sorted(g.choice(L * I, 12, replace=False).tolist())
        extra = [j for j in g.choice(L * I, 30, replace=False).tolist() if j not in htq][:8]
        boot = [sorted(set(g.choice(htq, 9, replace=False).tolist()) | set(g.choice(L * I, 3, replace=False).tolist()))
                for _ in range(20)]
        d = dict(H_tq=[[j // I, j % I] for j in htq], H_plus=[[j // I, j % I] for j in htq + extra], H_op_name="H_tq",
                 boot=[[[j // I, j % I] for j in b] for b in boot], C_star=0.05, mock_standin=True)
        sg8 = dict(G_K2="H-neurons replicated (in-domain detection)", H_op_name="H_tq")
        labels.append("MOCK stand-in for SO-HN")
    htq = flat(d.get("H_tq") or [], I)
    hplus = [int(l) * I + int(i) for l, i in (d.get("H_plus") or [])]
    seen = set(); hplus = [j for j in hplus if not (j in seen or seen.add(j))]
    name = d.get("H_op_name") or sg8.get("H_op_name") or "H_tq"
    gk2 = sg8.get("G_K2") or sg8.get("G-K2") or sg8.get("preview")
    sg8_ok = bool(gk2 and str(gk2).startswith("H-neurons replicated")) and name in ("H_tq", "H+")
    if name == "H_tq" and len(htq) >= 10:
        P, pname = htq, "H_tq"
    elif name in ("H_tq", "H+"):
        P, pname = hplus[:200], "H+"
    else:
        P, pname = None, "selector on S_fit (H_ar primary)"
    if not sg8_ok:
        labels.append("no validated H-set" + (" (H_ar primary)" if name == "H_ar" else ""))
    det = ctx.input("r1b_K_S1", "hn/hn_detectors.pt", required=False)
    cstar = d.get("C_star")
    if cstar is None and det:
        try:
            cstar = float(loadpt(det).get("C_star"))
        except Exception:
            cstar = None
    boots = [flat(b, I) for b in (d.get("boot") or [])]
    if len(boots) < 10:
        labels.append("selection-robustness not computed (fewer than 10 boot sets)")
    refw = None
    rw = meta.get("REF_W") or meta.get("ref_w")
    if isinstance(rw, dict) and all(k in rw for k in ("mu_z", "sd_z", "mu_cett", "sd_cett")):
        refw = {k: np.asarray(rw[k], dtype=np.float32).ravel() for k in ("mu_z", "sd_z", "mu_cett", "sd_cett")}
    return dict(P=P, P_name=pname, H_tq=htq, H_plus=hplus[:200], boot=boots, C_star=float(cstar or 0.05), sg8_ok=sg8_ok,
                sg8_preview=gk2, H_op_name=name, refw=refw, labels=labels, L=L, I=I, sel_freq=d.get("sel_freq"))


def _mock_lab(tok):
    g = rng("mock_lab")
    items = {}
    words = ["Paris", "Berlin", "Rome", "Madrid", "Vienna", "Oslo", "Lisbon", "Dublin", "Prague", "Athens", "Warsaw", "Bern"]
    for i in range(140):
        split = "S_fit" if i < 80 else ("S_L" if i < 128 else "NE")
        cls = ("CC" if i % 2 == 0 else "CI") if split != "NE" else "NE"
        ans = words[int(g.integers(len(words)))]
        q = (f"Which city is twinned with town number {i}?" if i % 3 else f"Who founded the museum number {i}?") if split != "NE" \
            else f"What is the capital of the invented land of Zorbia-{i}?"
        a_ids = K.enc(tok, ans)
        items[f"q{i:04d}"] = dict(split=split, cls=cls, question=q, aliases=[ans] if cls == "CC" else [words[(i + 5) % len(words)]],
                                  answers=[dict(text_ids=a_ids, key=int(g.integers(1 << 40)))] * 10, spans=[[0, len(a_ids)]])
    return items


def lab(ctx, tok):
    """SO-LAB (so_lab.json) items: id -> dict(split, cls, question, aliases, ans0 (sample-0 answer ids), span0)."""
    d = K.load_json(ctx.input("r1b_K_S1", "lab/so_lab.json"))
    labels = []
    if _need_mock(ctx, d, "so_lab"):
        items, labels = _mock_lab(tok), ["MOCK stand-in for SO-LAB"]
    else:
        items = d.get("items") or {}
    out = {}
    for iid, it in items.items():
        a = (it.get("answers") or [{}])[0]
        ids = a.get("text_ids") or a.get("ids") or []
        sp = (it.get("spans") or [[0, len(ids)]])[0]
        split = it.get("split") or it.get("set")
        cls = it.get("cls") or it.get("class")
        if split in ("NE", "FAB") or cls in ("NE", "FAB"):
            split, cls = "NE", "NE"
        out[iid] = dict(split=split, cls=cls, question=it.get("question") or it.get("text") or "", aliases=it.get("aliases") or
                        it.get("gold") or [], ans0=list(ids), span0=list(sp), id=iid)
    return out, labels


def split_rows(items, split, cls=None):
    return [it for _, it in sorted(items.items()) if it["split"] == split and (cls is None or it["cls"] == cls)]


def _mock_gating(n=24):
    g = rng("mock_gating"); out = {}
    for i in range(n):
        a, b, c = (int(x) for x in g.integers(2, 30, 3))
        ops = int(g.integers(2, 4))
        expr = f"({a} + {b}) * {c}" if ops == 2 else f"({a} + {b}) * {c} - {a}"
        v1 = a + b; v2 = v1 * c; res = v2 if ops == 2 else v2 - a
        steps = [[a, "+", b, v1], [v1, "*", c, v2]] + ([[v2, "-", a, res]] if ops == 3 else [])
        known = i % 2 == 0
        ans = str(res) if known else str(res + 1)
        out[f"a{i:04d}"] = dict(stratum="correct" if known else "wrong", greedy=ans, samples=[ans] * 4, prompt=f"What is {expr}?",
                                question=f"What is {expr}?", expr=expr, V=[a, b, c, v1, v2, res], steps=steps,
                                subexprs=[f"{a} + {b}", f"{v1} * {c}"] + ([f"{v2} - {a}"] if ops == 3 else []), I_plus=[], ops=ops)
    return out


def gating(ctx):
    """K4's gated A-lens pool items (k4_gating.json; prompts and truth sets from the item store when the export lacks them):
    id -> dict(stratum known / prone, answers (greedy + 4), prompt, V, steps, subexprs, I_plus)."""
    p = ctx.input("r1b_K_S2a", "lab/k4_gating.json", required=False)
    d = K.load_json(p) if p else None
    labels = []
    if _need_mock(ctx, d, "k4_gating"):
        items, labels = _mock_gating(), ["MOCK stand-in for K4 gating"]
    else:
        items = d.get("items") or {}
    out = {}
    for iid, it in items.items():
        st = it.get("stratum")
        out[iid] = dict(id=iid, stratum="known" if st in ("correct", "known") else "prone", answers=[it.get("greedy")] +
                        list(it.get("samples") or []), prompt=it.get("prompt") or it.get("question"), question=it.get("question") or
                        it.get("prompt"), V=[int(v) for v in (it.get("V") or it.get("truth") or [])], steps=it.get("steps") or [],
                        subexprs=it.get("subexprs") or [], I_plus=[int(v) for v in (it.get("I_plus") or [])], ops=it.get("ops"),
                        a_cal=bool(it.get("A_cal") or it.get("a_cal")))
        if not out[iid]["prompt"]:
            raise K.NotRun("input missing (K4 gating items without prompts: contract gap, see the part file)")
    return out, labels


def lensout(ctx, gate_items):
    """SO-LENSOUT per-chain records (jsonl): list of dicts (item, stratum, ops, variant, block, seed_rank, key, tokens [17 lists],
    k_star_conf, share_H [17], band [17], a_H [17])."""
    p = ctx.input("r1b_K_S2a", "lensout/so_lensout.jsonl", required=False)
    recs = []
    if p:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    if (not recs or (len(recs) == 1 and is_mock_file(recs[0]))):
        if not ctx.mock:
            return [], ["no exact-label lens corpus from K"]
        g = rng("mock_lensout"); recs = []
        for iid, it in gate_items.items():
            for blk in (14, 18):
                for sr in range(3):
                    toks = [[str(int(v)) for v in g.integers(0, 60, 1)] for _ in range(17)]
                    recs.append(dict(item=iid, stratum=it["stratum"], ops=it.get("ops"), variant="V1", block=blk, seed_rank=sr,
                                     key=int(g.integers(1 << 40)), tokens_text=toks, k_star_conf=int(g.integers(1, 17)),
                                     share_H=g.random(17).tolist(), band=[0.8] * 17, a_H=g.normal(0, 1, 17).tolist()))
        return recs, ["MOCK stand-in for SO-LENSOUT"]
    return recs, []


def d1(ctx):
    """SO-D1STIM F-sym items 0-31: list of dict(entity, attribute, v1, v2) (slot j = i mod 6)."""
    path = os.path.join(K.REPO, C.PLAN_DIR, "d1_stimuli.json")
    d = K.load_json(path) if os.path.exists(path) else None
    if d is None:
        if not ctx.mock:
            raise K.NotRun("input missing (d1_stimuli.json)")
        pools = [["Anna", "Bert", "Carl", "Dora"], ["fox", "owl", "bear", "wolf"], ["salt", "wool", "tea", "tin"],
                 ["red", "blue", "green", "white"], ["May", "June", "July", "March"], ["iron", "brass", "bronze", "copper"]]
        return [dict(entity=f"Velmora{i}", attribute=ATTR_NAMES[i % 6], v1=pools[i % 6][i % 4], v2=pools[i % 6][(i + 1) % 4])
                for i in range(32)], ["MOCK stand-in for SO-D1STIM"]
    fs = d.get("F_sym") or d.get("F-sym") or [x for x in d.get("items", []) if x.get("family") in ("F-sym", "F_sym")]
    slots = d.get("slots") or list(ATTR_NAMES)
    out = []
    for i, it in enumerate(fs[:32]):
        j = i % 6
        vals = it.get("versions") or [it.get("v1"), it.get("v2")]
        v1, v2 = vals[0], vals[1]
        pick = lambda v: v[j] if isinstance(v, list) else (v.get(slots[j]) if isinstance(v, dict) else v)
        slot = slots[j] if isinstance(slots[j], str) else slots[j].get("name")
        out.append(dict(entity=it.get("referent") or it.get("entity") or it.get("name"), attribute=slot, v1=pick(v1), v2=pick(v2)))
    return out, []


def ref_sets(ctx, tok, items, cf):
    """SO-DMG reference for L: REF-U32 (REF-R16 from the A -> B export + REF-K16 package file) plus 64 held-out S_L answers (the last
    64 S_L rows by id, which no L prompt set uses). Returns (RefSet L, RefSet U32, labels)."""
    labels = []
    try:
        r16 = K.RefSet.from_json(ctx.xfer("dmg_ref_R16.json"), "REF-R16")
        if not len(r16):
            raise ValueError
    except Exception:
        if not ctx.mock:
            raise K.NotRun("input missing (REF-R16)")
        r16 = K.RefSet.from_prompt_conts([K.chat_ids(tok, f"Describe scene {i}.") for i in range(4)],
                                         [K.enc(tok, f"A calm scene with {i} trees and a pond.") for i in range(4)], "REF-R16 (MOCK)")
        labels.append("MOCK stand-in for REF-R16")
    p = os.path.join(K.REPO, C.PLAN_DIR, "ref_k16.json")
    if os.path.exists(p):
        k16 = K.RefSet.from_json(p, "REF-K16")
    elif ctx.mock:
        k16 = K.RefSet.from_prompt_conts([K.chat_ids(tok, f"Who wrote book {i}?") for i in range(4)],
                                         [K.enc(tok, f"Author {i} wrote it.") for i in range(4)], "REF-K16 (MOCK)")
        labels.append("MOCK stand-in for REF-K16")
    else:
        raise K.NotRun("input missing (REF-K16)")
    sl = split_rows(items, "S_L")[-cf["ref_heldout"]:]
    held = K.RefSet.from_prompt_conts([K.chat_ids(tok, SHORT_QA.format(q=it["question"])) for it in sl], [it["ans0"] for it in sl],
                                      "S_L held-out 64")
    u32 = r16 + k16
    return u32 + held, u32, labels


def ustar(ctx):
    try:
        d = loadpt(ctx.xfer("so_ax_ustar.pt"))
        u = d.get("u_star", {}).get(14) if isinstance(d.get("u_star"), dict) else None
        if u is None:
            u = d.get("u_star", {}).get("14")
        return None if u is None else np.asarray(u, dtype=np.float64)
    except Exception:
        return None


def heldout_ids(items, cf):
    return {it["id"] for it in split_rows(items, "S_L")[-cf["ref_heldout"]:]}


def texts_mock(n, length, tok, tag):
    g = rng("mock_text", tag)
    words = ("the river runs past old mills and quiet fields where farmers keep sheep and the town bells ring at noon while "
             "children read stories about ships kings bridges rain markets and winter lights").split()
    out = []
    for i in range(n):
        ids = []
        while len(ids) < length:
            ids += K.enc(tok, " ".join(g.choice(words, 12)) + ". ")
        out.append(ids[:length])
    return out


def wt_test(ctx, tok, cf):
    """SO-WT103 test split: the 200 natural-position passages (first 200 x 1,024 tokens in file order, articles concatenated)."""
    if ctx.mock:
        return texts_mock(cf["nat"], cf["nat_len"], tok, "wt"), ["MOCK text"]
    d = K.fetch_dataset("wikitext103", ctx.root, ctx.freeze)
    import glob
    fs = sorted(glob.glob(os.path.join(d, "**", "*test*"), recursive=True))
    fs = [f for f in fs if f.endswith((".parquet", ".txt", ".raw", ".tokens"))]
    if not fs:
        raise K.NotRun("input missing (WikiText-103 test split)")
    text = _read_text(fs[0])
    ids = K.enc(tok, text)
    n, Lp = cf["nat"], cf["nat_len"]
    if len(ids) < n * Lp:
        raise K.NotRun("input missing (WikiText-103 test split too short)")
    return [ids[i * Lp:(i + 1) * Lp] for i in range(n)], []


def _read_text(f):
    if f.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(f)
        return "\n".join(str(x) for x in df[df.columns[0]].tolist())
    return open(f, encoding="utf-8", errors="replace").read()


def longdocs(ctx, tok, cf, which="analysis"):
    """SO-LONGDOC: slices [2,000, 34,768) of the first 24 qualifying PG19 test books (which='analysis') or books 25-28 ('pilot'),
    from the package file longdoc_slices.json (book file names and token slices; built by build_longdoc_list before the freeze)."""
    if ctx.mock:
        n = cf["eqmi_docs"] if which == "analysis" else 2
        return texts_mock(n, cf["long_len"], tok, "longdoc" + which), ["MOCK text"]
    p = os.path.join(K.REPO, C.PLAN_DIR, "longdoc_slices.json")
    lst = K.load_json(p)
    if not lst:
        raise K.NotRun("input missing (longdoc_slices.json)")
    d = K.fetch_dataset("pg19", ctx.root, ctx.freeze)
    out = []
    for b in lst[which]:
        ids = K.enc(tok, _read_text(os.path.join(d, b["file"])))
        if K.sha256_text(K.canon(ids[b["start"]:b["end"]])) != b["sha256"]:
            raise RuntimeError(f"SO-LONGDOC slice of {b['file']} differs from its frozen sha256")
        out.append(ids[b["start"]:b["end"]])
    return out, []


def build_longdoc_list(pg19_test_dir, tok_dir, out, n=24, pilot=4, skip=2000, length=32768, min_tokens=36000):
    """Package builder (CPU, before the freeze): the first n + pilot PG19 test books in file order with >= min_tokens after skip."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir)
    sel = []
    for f in sorted(os.listdir(pg19_test_dir)):
        ids = tok(_read_text(os.path.join(pg19_test_dir, f)), add_special_tokens=False)["input_ids"]
        if len(ids) - skip >= min_tokens:
            sel.append(dict(file=f, start=skip, end=skip + length, n_tokens=len(ids), sha256=K.sha256_text(K.canon(ids[skip:skip + length]))))
        if len(sel) == n + pilot:
            break
    K.dump_json(dict(analysis=sel[:n], pilot=sel[n:], rule="first books in file order with >= 36,000 tokens after skipping 2,000; "
                     "slice [2,000, 34,768)"), out)


# ==================================================================================================================================
# scorers (K's frozen scorers if importable; else L's fallback, labelled)
# ==================================================================================================================================
def _norm(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def scorers():
    try:
        import r1b_k_scorers as S                      # part K's frozen scorers (SO-LAB), if present
        return S.alias_match, S.is_refusal, "K scorers"
    except Exception:
        return (lambda text, aliases: any(_norm(a) and _norm(a) in _norm(text) for a in aliases),
                lambda text: any(w in text.lower() for w in REFUSAL_LEXICON), "L fallback scorers (K scorer module not importable)")


def answer_span(ids, tok):
    """K's span rule: generated tokens up to the first newline or EOS, minus trailing punctuation (positions [0, n))."""
    n = 0
    for t in ids:
        s = tok.decode([t])
        if "\n" in s:
            break
        n += 1
    while n > 0 and tok.decode(ids[n - 1:n]).strip() in (".", ",", "!", "?", ";", ":", ""):
        n -= 1
    return n


# ==================================================================================================================================
# populations (L.POP)
# ==================================================================================================================================
def load_pop(ctx):
    """L.POP from r1b_L_A (own session) or its copy (r1b_L_B / r1b_ana)."""
    p = ctx.path("l/l_pop.json") if ctx.session == "r1b_L_A" else ctx.input("r1b_L_A", "l/l_pop.json")
    d = K.load_json(p)
    if d is None:
        raise K.NotRun("input missing (L.POP)")
    return d


def channels(pop, with_boot=True):
    """Ordered channel list [(name, flat ids)]: P, SEL, UNC, C0..C98, Htq (if P is H+), B0..B19."""
    ch = [("P", pop["P"])]
    if pop.get("SEL"):
        ch.append(("SEL", pop["SEL"]))
    if pop.get("UNC"):
        ch.append(("UNC", pop["UNC"]))
    ch += [(f"C{k}", s) for k, s in enumerate(pop["CTRL"])]
    if pop.get("Htq_extra"):
        ch.append(("Htq", pop["Htq_extra"]))
    if with_boot:
        ch += [(f"B{k}", s) for k, s in enumerate(pop.get("boot") or [])]
    return ch


def spin_sets(pop):
    """Spin populations (L.2): P (H+ if |P| < 10 handled by L.POP), SEL, UNC, every CTRL set, the boot sets."""
    out = [("P", pop["P"])] + [(n, pop[n]) for n in ("SEL", "UNC") if pop.get(n)]
    out += [(f"C{k}", s) for k, s in enumerate(pop["CTRL"])]
    out += [(f"B{k}", s) for k, s in enumerate(pop.get("boot") or [])]
    return out


def refw_tensors(pop_or_refw, L, I):
    """REF-W (mu_z, sd_z, mu_cett, sd_cett) as float32 torch tensors [L * I]."""
    torch = K._torch()
    r = pop_or_refw
    return {k: torch.as_tensor(np.asarray(r[k], dtype=np.float32)) for k in ("mu_z", "sd_z", "mu_cett", "sd_cett")}

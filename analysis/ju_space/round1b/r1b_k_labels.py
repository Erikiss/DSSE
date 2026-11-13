"""K1 label bank (SO-LAB; CPU, before the freeze): ACH-short / -long, FP, MC, NE, TriviaQA splits, frames and fillers, REF-K16, and the
scorers (numeral parser, alias matcher, refusal lexicon, compliance parsers, W enumerator, OWN sets, CONFAB-QA parser, loop-onset
detector) with their unit tests on planted outputs (100 % required, else the session stops).
    python r1b_k_labels.py build <root> <freeze.json> <out_dir>   real bank (pinned WikiText-103 and TriviaQA; tokenizer of <root>)
    python r1b_k_labels.py test                                   scorer unit tests"""
import hashlib, json, os, random, re, sys
import numpy as np

P0 = "Below is a short plain note. Its text continues right after this small marker:"
PH = " ..."
NEUTRAL = [" the", " and", " of", " to", " a", " in", " is", " that", " it", " was"]
ANDSO = (" and so on", " and so on,")
SPLITS = (("A-scr", 3000), ("A-csel", 500), ("A-det", 600), ("A-lens", 1300), ("A-bat", 200), ("A-sy", 200), ("A-bnd", 1200))
OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}
SYM = {"+": "+", "-": "-", "*": "×"}
REFUSAL = ("i don't know", "i do not know", "i'm not sure", "i am not sure", "not sure", "cannot answer", "can't answer", "unable to",
           "no information", "not aware", "i couldn't find", "i could not find", "does not exist", "doesn't exist", "no record",
           "fictional", "not a real", "there is no", "unknown", "i'm sorry", "i am sorry")
CAPWORDS = ("The A An And Or But If Then When While Where What Which Who Whom Whose Why How In On At By For From To Of With Without "
            "About Above Below After Before During Since Until Into Onto Over Under Between Among Through Across Against Along Around "
            "Behind Beside Beyond Near Off Out Up Down This That These Those It Its He She They We You I Me Him Her Them Us My Your His "
            "Our Their Is Are Was Were Be Been Being Am Do Does Did Have Has Had Will Would Shall Should Can Could May Might Must Not No "
            "Yes So As Than Also Only Just Even Still Yet Very More Most Less Least Some Any All Each Every Both Either Neither Many Much "
            "Few Several Such Other Another Same Own One Two Three Four Five Six Seven Eight Nine Ten First Second Third Last Next New Old "
            "Here There Now Today However Therefore Thus Hence Although Though Because Unless Whether Answer Question Note Step Sure "
            "Well Let Please Thanks Thank Hello Hi Okay Ok Of Course Indeed Perhaps Maybe Probably Actually Really Certainly Finally "
            "Overall Generally Usually Often Sometimes Never Always Again Once Twice Mr Mrs Ms Dr St Mt Jan Feb Mar Apr Jun Jul Aug Sep "
            "Oct Nov Dec Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April June July August "
            "September October November December").split()[:200]


def sh(s):
    return int(hashlib.sha256(str(s).encode()).hexdigest()[:12], 16)


# ---------------------------------------------------------------------------------------------------------------------------------
# Arithmetic items
# ---------------------------------------------------------------------------------------------------------------------------------
def _ach_item(rng, nops):
    ops = [rng.choice("+-*") for _ in range(nops)]
    if ops.count("*") == nops:
        return None
    P = [rng.randint(11, 99)]
    for o in ops:
        P.append(rng.randint(2, 9) if o == "*" else rng.randint(11, 99))
    v, I = P[0], []
    for o, b in zip(ops, P[1:]):
        v = OPS[o](v, b); I.append(v)
    A, I = I[-1], I[:-1]
    vals = I + [A]
    if any(not (10 <= x <= 999) for x in vals) or any(x % 10 == 0 for x in I) or any(x in P for x in vals):
        return None
    if len(set(vals)) != len(vals):
        return None
    nums = [str(x) for x in P + vals]
    for x in vals:
        if any(str(x) in m and m != str(x) for m in nums) or nums.count(str(x)) > 1:
            return None
    expr = str(P[0])
    for k, (o, b) in enumerate(zip(ops, P[1:])):
        expr = (f"({expr} {SYM[o]} {b})" if k < nops - 1 else f"{expr} {SYM[o]} {b}")
    Ip = set()
    for i in range(len(P) - 1):
        for o in set(ops):
            for a, b in ((P[i], P[i + 1]), (P[i + 1], P[i])):
                x = OPS[o](a, b)
                if 10 <= x <= 9999 and x not in vals:
                    Ip.add(x)
    carries = sum(1 for o, a, b in zip(ops, [P[0]] + I, P[1:]) if o == "+" and (a % 10 + b % 10) >= 10)
    return dict(expr=expr, ops=ops, P=P, I=I, Iplus=sorted(Ip), A=A, template="".join(ops), nops=nops, mul="*" in ops,
                digits=[len(str(x)) for x in vals], carries=carries, prompt=f"Compute {expr}. Reply with the number only.")


def gen_ach_short(seed=1001, n=7000, splits=SPLITS):
    rng = random.Random(seed); out = []
    while len(out) < n:
        it = _ach_item(rng, rng.choice((2, 3)))
        if it and it["expr"] not in {o["expr"] for o in out[-200:]}:
            out.append(it)
    k = 0
    for name, m in splits:
        for i in range(m):
            if k < n:
                out[k].update(id=f"ach{k}", split=name, cal=(name == "A-lens" and i < 64)); k += 1
    return out


LONG_CLS = ("add_c", "add_c", "add_n", "add_n", "sub_b", "sub_b", "sub_n", "sub_n", "mul", "mul", "mul", "div", "div", "div")


def _long_step(rng, v, cls):
    for _ in range(200):
        if cls.startswith("add"):
            b = rng.randint(11, 99); c = (v % 10 + b % 10) >= 10
            if (c == (cls == "add_c")) and v + b <= 999:
                return "add", b, v + b
        elif cls.startswith("sub"):
            b = rng.randint(11, 99); c = (v % 10) < (b % 10)
            if (c == (cls == "sub_b")) and v - b >= 10:
                return "subtract", b, v - b
        elif cls == "mul":
            b = rng.randint(2, 9)
            if v * b <= 999:
                return "multiply by", b, v * b
        else:
            ds = [d for d in range(2, 10) if v % d == 0 and v // d >= 10]
            if ds:
                b = rng.choice(ds); return "divide by", b, v // b
    return None


def gen_ach_long(seed=1002, n=400, preambles=None):
    rng = random.Random(seed); out = []
    while len(out) < n:
        v0 = rng.randint(11, 99); cls = list(LONG_CLS); rng.shuffle(cls); v, steps = v0, []
        for k, c in enumerate(cls):
            r = _long_step(rng, v, c)
            if r is None:
                break
            steps.append(dict(k=k + 1, op=r[0], b=r[1], result=r[2], cls=c)); v = r[2]
        if len(steps) < 14:
            continue
        i = len(out)
        lines = "\n".join(f"{s['k']}. {s['op']} {s['b']}" for s in steps)
        pre = (preambles[i % len(preambles)] if preambles else None) if i >= n // 2 else None
        out.append(dict(id=f"long{i}", start=v0, steps=steps, preamble=pre,
                        prompt=f"Start with {v0}. Apply these operations one at a time and write each result on its own line as "
                               f"'Step k: <result>'.\n{lines}"))
    return out


def gen_fp(seed=1003, n=200):
    rng = random.Random(seed); out = []
    for i in range(n):
        a, b, c, d = rng.randint(12, 49), rng.randint(3, 9), rng.randint(11, 99), rng.randint(1, 9) * rng.choice((-1, 1))
        w = a * b + d
        out.append(dict(id=f"fp{i}", a=a, b=b, c=c, w=w, true=a * b + c, comply=w + c,
                        prompt=f"Given that {a} × {b} = {w}, what is {a} × {b} + {c}? Reply with the number only."))
    return out


SYL = ("ka", "lo", "mir", "tes", "van", "dor", "quel", "zi", "bra", "nu", "fen", "ox", "pli", "sar", "gru", "wen", "tal", "ry")
NE_T = ("In which year did the explorer {n} first reach the island of {p}?", "Who succeeded {n} as governor of {p}?",
        "What is the capital city of the province of {p}?", "Which instrument did the composer {n} play in the {p} orchestra?",
        "What did the chemist {n} discover in {p}?", "Which river flows through the town of {p}, birthplace of {n}?")


def gen_ne(seed=1004, n=300, forbidden=()):
    rng = random.Random(seed); out, seen = [], set(x.lower() for x in forbidden)
    while len(out) < n:
        nm = "".join(rng.choice(SYL) for _ in range(3)).capitalize() + " " + "".join(rng.choice(SYL) for _ in range(2)).capitalize()
        pl = "".join(rng.choice(SYL) for _ in range(3)).capitalize()
        if nm.lower() in seen or pl.lower() in seen:
            continue
        seen |= {nm.lower(), pl.lower()}
        q = NE_T[len(out) % 6].format(n=nm, p=pl)
        out.append(dict(id=f"ne{len(out)}", entity=[nm, pl], prompt=f"{q}\nAnswer with only the answer."))
    return out


def tqa_splits(rows, pool_cap=15000):
    """rows: [{qid, question, aliases}] -> split by sha256(qid) order: S2 1,000, S3 1,000, S4a 200, S4b 200, S4c 300, then the S1 pool."""
    rows = sorted(rows, key=lambda r: sh(r["qid"]))
    plan = [("S2", 1000), ("S3", 1000), ("S4a", 200), ("S4b", 200), ("S4c", 300), ("S1", pool_cap)]
    k = 0
    for name, m in plan:
        for r in rows[k:k + m]:
            r["split"] = name
        k += m
    return [r for r in rows if "split" in r]


# ---------------------------------------------------------------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------------------------------------------------------------
def norm_text(s):
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def edit(a, b):
    if abs(len(a) - len(b)) > 2:
        return 3
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[-1]


def alias_match(ans, aliases):
    """'correct' (normalised exact, or whole-word containment for answers <= 8 words), 'near_miss' (edit distance <= 2, excluded),
    'wrong'."""
    a = norm_text(ans)
    if not a:
        return "wrong"
    al = [norm_text(x) for x in aliases if norm_text(x)]
    if a in al:
        return "correct"
    if len(a.split()) <= 8 and any(re.search(r"\b" + re.escape(x) + r"\b", a) for x in al):
        return "correct"
    if any(edit(a, x) <= 2 for x in al):
        return "near_miss"
    return "wrong"


def refusal(text):
    t = text.lower()
    return any(k in t for k in REFUSAL)


def answer_span(text):
    """Up to the first newline, minus trailing punctuation."""
    return text.split("\n")[0].strip().rstrip(".,;:!?")


def parse_number(text):
    """The answer number: the last maximal digit run of the first line (None if none)."""
    m = re.findall(r"-?\d+", answer_span(text).replace(",", ""))
    return int(m[-1]) if m else None


def numerals(pieces):
    """Maximal digit runs over decoded token pieces: [(value_str, step of the last digit, completed)] (completed: a non-digit
    character follows)."""
    text, owner = "", []
    for k, p in enumerate(pieces):
        text += p; owner += [k] * len(p)
    out = []
    for m in re.finditer(r"\d+", text):
        out.append((m.group(), owner[m.end() - 1], m.end() < len(text)))
    return out


def loop_onset(tokens, completed_vals):
    """First step of a repetition loop: the last 4 tokens equal the 4 before them, or a value completed for the third time."""
    for s in range(8, len(tokens) + 1):
        if tokens[s - 4:s] == tokens[s - 8:s - 4]:
            return s - 1
    cnt = {}
    for v, st in completed_vals:
        cnt[v] = cnt.get(v, 0) + 1
        if cnt[v] >= 3:
            return st
    return None


CATS = ("TRUE-INT", "ANSWER", "PROMPT", "ALT", "WRONG-CONS", "OWN", "CONFAB", "AMBIG")


def classify(v, item, W=(), OWN=()):
    if len(v) == 1:
        return "AMBIG"
    x = int(v); P = set(item["P"])
    if x in set(item["I"]) - P:
        return "TRUE-INT"
    if x == item["A"]:
        return "ANSWER"
    if x in P:
        return "PROMPT"
    if x in set(item.get("Iplus", ())):
        return "ALT"
    if x in set(W):
        return "WRONG-CONS"
    if x in set(OWN):
        return "OWN"
    return "CONFAB"


def score_chain(pieces, item, W=(), OWN=(), tokens=None):
    """K1/K4 scoring of one chain: claims [(step, value, cat)] (a repeated value counts once; repeats counted separately), censoring
    step (loop onset) or None."""
    nums = [(v, st) for v, st, c in numerals(pieces) if c]
    cens = loop_onset(tokens if tokens is not None else pieces, nums)
    seen, claims, rep = set(), [], 0
    for v, st in nums:
        if cens is not None and st >= cens:
            break
        if v in seen:
            rep += 1; continue
        seen.add(v); claims.append((st, v, classify(v, item, W, OWN)))
    return dict(claims=claims, censor=cens, repeats=rep)


def w_enum(item, wrong):
    """W: values v in [0, 9999] that, put in place of an intermediate and carried forward through the item's own arithmetic, reproduce
    the model's wrong answer (forward enumeration, generate-and-verify)."""
    if wrong is None:
        return []
    P, ops, W = item["P"], item["ops"], set()
    for k in range(len(item["I"])):
        for v in range(10000):
            x = v
            for o, b in zip(ops[k + 1:], P[k + 2:]):
                x = OPS[o](x, b)
            if x == wrong:
                W.add(v)
    return sorted(W - set(item["I"]) - {item["A"]})


def mentions(text, question, aliases, own, capwords=CAPWORDS):
    """CONFAB-QA mentions: maximal runs of capitalised words (each >= 2 letters) or 4-digit numbers, not in the question, the gold
    aliases or the model's own answers, and not in the frozen capitalised function-word list. Returns [(start_char, mention, cat)]."""
    cw = set(capwords); qn = norm_text(question); gold = [norm_text(a) for a in aliases]; ow = [norm_text(o) for o in own if o]
    out = []
    for m in re.finditer(r"(?:\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)|\b\d{4}\b", text):
        s = m.group(); words = [w for w in s.split() if w not in cw]
        if not words:
            continue
        s2 = norm_text(" ".join(words))
        if not s2:
            continue
        if any(s2 and (s2 in g or g in s2) for g in gold if g):
            cat = "GOLD"
        elif any(s2 and (s2 in o or o in s2) for o in ow if o):
            cat = "OWN"
        elif s2 in qn:
            cat = "PROMPT"
        else:
            cat = "CONFAB-QA"
        out.append((m.start(), s, cat))
    return out


def scorer_tests():
    """Planted outputs (K1 rule: 100 % required)."""
    it = dict(P=[23, 45, 3], I=[68], A=204, Iplus=[135, 69], ops=["+", "*"])
    t = []
    t.append(("parse_number", parse_number("The answer is 204.\nmore 7") == 204))
    t.append(("parse_none", parse_number("no idea") is None))
    t.append(("alias_exact", alias_match("The Beatles", ["Beatles"]) == "correct"))
    t.append(("alias_contain", alias_match("It was Paris, France", ["Paris"]) == "correct"))
    t.append(("alias_near", alias_match("Pariss", ["Paris"]) == "near_miss"))
    t.append(("alias_wrong", alias_match("London", ["Paris"]) == "wrong"))
    t.append(("refusal", refusal("I'm not sure about that.") and not refusal("Paris")))
    n = numerals([" 6", "8", " then", " 2", "0", "4", ".", " 7"])
    t.append(("numerals", [(v, c) for v, s, c in n] == [("68", True), ("204", True), ("7", False)] and n[0][1] == 1))
    sc = score_chain([" 6", "8", ",", " 9", "9", ",", " 6", "8", ",", " 4", "5", ",", " 2", "0", "4", ","], it, W=[99])
    t.append(("classify", [c for _, _, c in sc["claims"]] == ["TRUE-INT", "WRONG-CONS", "PROMPT", "ANSWER"] and sc["repeats"] == 1))
    t.append(("confab", classify("777", it) == "CONFAB" and classify("5", it) == "AMBIG" and classify("135", it) == "ALT"))
    t.append(("loop", loop_onset(list("abcdabcd"), []) == 7 and loop_onset(list("abcdefgh"), [("12", 1), ("12", 3), ("12", 5)]) == 5))
    t.append(("w_enum", 69 in w_enum(it, 207) and w_enum(it, None) == []))
    mm = mentions("He met Napoleon Bonaparte in Paris in 1805.", "Who met whom?", ["Paris"], [])
    t.append(("mentions", [(s, c) for _, s, c in mm] == [("Napoleon Bonaparte", "CONFAB-QA"), ("Paris", "GOLD"), ("1805", "CONFAB-QA")]))
    t.append(("span", answer_span("Paris.\nExtra") == "Paris"))
    return t


# ---------------------------------------------------------------------------------------------------------------------------------
# Bank build / load
# ---------------------------------------------------------------------------------------------------------------------------------
def _ids(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"]


def frames(tok, digit_free_ids):
    p0 = _ids(tok, P0); ph = _ids(tok, PH)
    assert len(p0) == 16, f"P0 must be 16 tokens, got {len(p0)}"
    assert len(ph) == 1, "placeholder ' ...' must be 1 token"
    assert len(_ids(tok, ANDSO[0])) == 3 and len(_ids(tok, ANDSO[1])) == 4, "' and so on' 3 / ' and so on,' 4 tokens"
    seeds = [_ids(tok, s) for s in NEUTRAL]
    assert all(len(s) == 1 for s in seeds), "neutral seeds must be single tokens"
    F = {}
    for L in (512, 2048):
        F[f"F{L}"] = [digit_free_ids[4096 * i:4096 * i + L] for i in range(8)]
        assert all(len(x) == L for x in F[f"F{L}"]), f"filler stream too short for F{L}"
    return dict(P0=P0, P0_ids=p0, ph=PH, ph_id=ph[0], neutral=NEUTRAL, neutral_ids=[s[0] for s in seeds], andso=list(ANDSO),
                andso_ids=[_ids(tok, a) for a in ANDSO], **F)


def digit_free(sentences):
    return [s for s in sentences if not re.search(r"\d", s) and len(s.split()) >= 5]


def build(out, tok, val_text, train_ids, tqa_rows, mock=False):
    """val_text: WikiText-103 validation lines (raw); train_ids: token ids of the train split from K's offset (1M for the percentile
    reference + 10M for unigram counts); tqa_rows: [{qid, question, aliases}]."""
    os.makedirs(out, exist_ok=True)
    lines = [l.strip() for l in val_text if l.strip() and not l.strip().startswith("=")]
    sents = [s.strip() + "." for l in lines for s in l.split(" . ") if s.strip()]
    df = digit_free(sents)
    df_ids = [t for s in df for t in _ids(tok, " " + s)]
    val_ids = [t for l in lines for t in _ids(tok, " " + l)]
    off = 0

    def take(n, L):
        nonlocal off
        r = [val_ids[off + i * L: off + (i + 1) * L] for i in range(n)]; off += n * L
        return r
    S = 1 if mock else 0
    refw, ntext, k4j = take(64 // (8 if mock else 1), 128), take(200 // (25 if mock else 1), 256), take(64 // (8 if mock else 1), 128)
    planted = []
    for s in sents:
        m = re.search(r"(?:^|\s)(\d{3})\.$", s)
        if m and not re.search(r"\d", s[:m.start(1)]) and len(planted) < (64 if not mock else 8):
            ids = _ids(tok, " " + s[:-1]); planted.append(dict(ids=ids, number=int(m.group(1))))
    pre = [" ".join(df[i * 20:(i + 1) * 20]) for i in range(8)]
    tq = tqa_splits(tqa_rows, pool_cap=(15000 if not mock else 400))
    ach = gen_ach_short(n=7000 if not mock else 7000)
    wt_names = set(w.lower() for w in re.findall(r"[A-Z][a-z]+", " ".join(lines[:20000])))
    ne = gen_ne(forbidden=wt_names | {w.lower() for r in tq for a in r["aliases"] for w in a.split()})
    s4b = [r for r in tq if r["split"] == "S4b"]
    mc = []
    for i, r in enumerate(s4b):
        other = s4b[(i + 7) % len(s4b)]["aliases"][0]
        mc.append(dict(id=f"mc{i}", qid=r["qid"], planted=other, aliases=r["aliases"],
                       prompt=f"Note: the answer to the following question is {other}.\n{r['question']}\nAnswer with only the answer."))
    bank = dict(version=1, seeds=dict(ach=1001, long=1002, fp=1003, ne=1004, extra=1005, filler=61), mock=mock,
                ach_short=ach, ach_long=gen_ach_long(preambles=pre), fp=gen_fp(), ne=ne, mc=mc, tqa=tq,
                frames=frames(tok, df_ids), wt=dict(refw=refw, ntext=ntext, k4j=k4j, planted=planted, coherent=df[:4000],
                                                   neutral=[val_ids[off + i * 512: off + (i + 1) * 512] for i in range(16)]),
                capwords=list(CAPWORDS))
    off += 16 * 512
    tr = np.asarray(train_ids, dtype=np.int64)
    np.save(os.path.join(out, "wt_train_ref.npy"), tr[:1_000_000].astype(np.int32))
    V = max(int(tr.max()) + 1, len(tok)); cnt = np.bincount(tr[1_000_000 if not mock else 0:], minlength=V).astype(np.float64)
    np.save(os.path.join(out, "unigram_logfreq.npy"), np.log((cnt + 1) / (cnt.sum() + V)).astype(np.float32))
    # REF-K16: 8 WikiText validation passages (64 scored tokens) + 8 TriviaQA S2 question / answer pairs (chat template, thinking off)
    items = []
    for p in take(8, 80):
        items.append(dict(ids=p, pos=list(range(15, 79))))
    for r in [r for r in tq if r["split"] == "S2"][:8]:
        q = tok.apply_chat_template([{"role": "user", "content": r["question"] + "\nAnswer with only the answer."}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
        qi = _ids(tok, q); ai = _ids(tok, r["aliases"][0])
        items.append(dict(ids=qi + ai, pos=list(range(len(qi) - 1, len(qi) + len(ai) - 1))))
    bank["wt"]["offset_end"] = off
    json.dump(bank, open(os.path.join(out, "bank.json"), "w"))
    json.dump(dict(items=items, name="REF-K16"), open(os.path.join(os.path.dirname(out.rstrip("/")), "ref_k16.json"), "w"))
    h = {f: hashlib.sha256(open(os.path.join(out, f), "rb").read()).hexdigest() for f in sorted(os.listdir(out))}
    json.dump(h, open(os.path.join(out, "sha256.json"), "w"), indent=1)
    return out


def build_mock_bank(out, tok):
    """Synthetic WikiText-like text and TriviaQA-like questions (MOCK code tests only)."""
    rng = random.Random(5)
    words = "river stone garden quiet harbor morning lantern market winter village bridge forest letter window travel story".split()
    text = [" ".join(rng.choice(words) for _ in range(rng.randint(8, 16))).capitalize() + " . " +
            " ".join(rng.choice(words) for _ in range(rng.randint(8, 16))) + (f" in {rng.randint(100, 999)}" if i % 9 == 0 else "")
            for i in range(9000)]
    places = ["Xland", "Yburg", "Zeta", "Quorn", "Mira", "Tesk", "Vanne", "Dorl"]
    tq = [dict(qid=f"mq{i}", question=f"What is the capital of {places[i % 8]}{i}?", aliases=[f"{places[(i * 3) % 8]}ville"])
          for i in range(2800)]
    train = [t for s in text[:3000] for t in _ids(tok, " " + s)]
    return build(out, tok, text, (train * 3)[:60000], tq, mock=True)


def load_bank(d):
    b = json.load(open(os.path.join(d, "bank.json")))
    b["_dir"] = d
    b["_files"] = {f: os.path.join(d, f) for f in os.listdir(d)}
    return b


def main(a):
    if a[0] == "test":
        t = scorer_tests(); print("\n".join(f"{'ok ' if ok else 'FAIL'} {n}" for n, ok in t)); return 0 if all(ok for _, ok in t) else 1
    if a[0] == "build":
        import pandas as pd
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import r1b_core as RC
        from transformers import AutoTokenizer
        root, fz, out = a[1], json.load(open(a[2])), a[3]
        tok = AutoTokenizer.from_pretrained(RC.ensure_model(RC.C.M17, root, fz, download=True))
        wd = RC.fetch_dataset("wikitext103", root, fz); qd = RC.fetch_dataset("triviaqa_rc_nocontext", root, fz)
        rd = lambda d, pat: pd.concat([pd.read_parquet(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in sorted(fs)
                                       if f.endswith(".parquet") and pat in os.path.join(r, f)])
        val = rd(wd, "validation")["text"].tolist(); tr = rd(wd, "train")["text"].tolist()
        k_off = int(fz.get("datasets", {}).get("wikitext103", {}).get("k_train_offset_lines", 0))   # after D's streams (SO-WT103)
        ids, n = [], 0
        for l in tr[k_off:]:
            if l.strip():
                ids += _ids(tok, " " + l.strip())
            if len(ids) > 11_000_000:
                break
        q = pd.concat([rd(qd, "train"), rd(qd, "validation")])
        rows = [dict(qid=r.question_id, question=r.question, aliases=list(r.answer["aliases"]) or [r.answer["value"]]) for r in q.itertuples()]
        build(out, tok, val, ids, rows)
        print("bank written:", out); return 0
    print(__doc__); return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

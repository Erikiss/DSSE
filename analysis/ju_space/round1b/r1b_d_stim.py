"""Part D stimuli and shared helpers (D.2, D.3, D.5, D.11, D.13-D.16): the frozen generators of the package files d1_stimuli.json
(SO-D1STIM), d2_prompts.json, d3_items.json, d4_roles.json, d1v_pairs.json; the 16 WikiText-103 filler streams (SO-WT103 train split,
file order) with their sentence-shuffled versions, D1-d's two reference passages and G-D6's validation passages; token helpers
(chat pieces, letters, yes / no, turn suffixes, retokenisation); the configuration with its MOCK scaling.

    python r1b_d_stim.py build <tokenizer dir> <out dir>      write the five package files (deterministic; sha256 printed)
    python r1b_d_stim.py streams <root> <freeze.json> <out>   build the filler streams (pinned dataset) and print their sha256
Package files are read from $R1B_PKG_DIR (code tests) or results/ju_space/plans/runde1b/ (frozen with the plan)."""
import hashlib, itertools, json, math, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import r1b_core as K
import r1b_contracts as C

GEN_SEED = 20261113            # numeric seed of the frozen generators (no Gumbel key)
CLIP = 15.0

# ------------------------------------------------------------------------------------------------------------------------------------
# configuration (D.3-D.16) and its MOCK scaling (code tests on the CPU with the tiny random model)
# ------------------------------------------------------------------------------------------------------------------------------------
REAL = dict(n_sym=48, n_asym=24, n_streams=16, L0_fill=150, L_deg=(4096, 16384), L_long=(32000, 16384), L_parity=(16384, 32000),
            nonfill=900, stream_min=32000, d2_n=160, d2_sec=64, d2_long=31800, d2_short_fill=150, d2_new=64, d3_items=16,
            d3_reserve=8, d3_seeds=16, d3_ctrl_seeds=8, d3_n=8, d3_new=40, d3_long=28672, d3_trim=16384, d4_roles=16, d4_seeds=8,
            d4_long=16384, d4_anchor_seeds=4, d5_starts=32, d5_turns=16, d5_turn_len=256, d5_B=20, d5_pool=8, d5_pool_len=96,
            d5x_cap_min=20.0, d5x_max=28672, d5x_turn=512, d1v_pairs=24, max_pos=32100, B_long=3, B_mid=7)
MOCK_CFG = dict(n_sym=4, n_asym=2, n_streams=4, L_deg=(1100, 1300), L_long=(1500, 1300), L_parity=(1300, 1500), stream_min=1700,
                d2_n=4, d2_sec=2, d2_long=1500, d3_items=2, d3_reserve=1, d3_seeds=4, d3_ctrl_seeds=2, d3_n=8, d3_new=8, d3_long=1400,
                d3_trim=1200, d4_roles=2, d4_seeds=2, d4_long=1200, d4_anchor_seeds=2, d5_starts=2, d5_turns=4, d5_turn_len=32, d5_B=20,
                d5_pool=8, d5_pool_len=16, d5x_cap_min=0.5, d5x_max=256, d5x_turn=64, d1v_pairs=6, d2_new=16)


class Cfg(dict):
    __getattr__ = dict.__getitem__


def cfg(ctx=None, mock=None):
    m = (ctx.mock if ctx is not None else K.MOCK) if mock is None else mock
    c = Cfg(REAL)
    if m:
        c.update(MOCK_CFG)
    c["mock"] = bool(m)
    return c


def L0_tag():
    return "L0"


# ------------------------------------------------------------------------------------------------------------------------------------
# tokens
# ------------------------------------------------------------------------------------------------------------------------------------
class Tk:
    """Token helpers of one tokenizer: enc, chat pieces (thinking off), letters, Yes / No (rk_loop's first-token ids), turn suffixes."""

    def __init__(self, tok):
        self.tok = tok
        c = K.chat_text(tok, "\x00")
        pre, post = c.split("\x00")
        self.pre, self.post = self.enc(pre), self.enc(post)
        self.A, self.B = self.enc("a"), self.enc("b")
        assert len(self.A) == 1 and len(self.B) == 1, "letters 'a' / 'b' are not single tokens"
        self.A, self.B = self.A[0], self.B[0]
        self.YES, self.NO = self.enc("Yes")[0], self.enc("No")[0]
        self.ans = self.enc("Answer: (")
        self.eos, self.stop = None, None

    def enc(self, s):
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def dec(self, ids):
        return self.tok.decode(list(ids))

    def user(self, text):
        """Full chat ids of one user turn (thinking off); equals pre + enc(text) + post."""
        return self.pre + self.enc(text) + self.post

    def turn(self, user_text):
        """History continuation after an assistant answer: <|im_end|> + a new user turn + the assistant header with the empty think block
        (r2_stimuli.TURN2_SUFFIX for 'Continue.')."""
        return self.enc("<|im_end|>\n<|im_start|>user\n" + user_text + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

    def stops(self, model):
        eos, stop = K.stop_tokens(self.tok, model)
        self.eos, self.stop = eos, sorted(stop)
        return self.stop

    def byte_str(self, text):
        return "".join(self.tok.convert_ids_to_tokens(self.enc(text)))

    def recode(self, text, mode, rng=None):
        """Ids of text in encoding mode 'can' (canonical), 'char' (one byte-level character per token) or 'seg' (a uniformly drawn
        valid segmentation into >= 2 vocabulary tokens; rng = np.random.default_rng(keyseed)); decode equals the text (asserted)."""
        if mode == "can":
            return self.enc(text)
        s = self.byte_str(text)
        if mode == "char":
            ids = self.tok.convert_tokens_to_ids(list(s))
        else:
            V = self.tok.get_vocab()
            segs = []

            def rec(i, cur):
                if len(segs) >= 5000:
                    return
                if i == len(s):
                    if len(cur) >= 2:
                        segs.append(list(cur))
                    return
                for j in range(i + 1, len(s) + 1):
                    if s[i:j] in V:
                        rec(j, cur + [s[i:j]])
            rec(0, [])
            ids = self.tok.convert_tokens_to_ids(segs[int(rng.integers(len(segs)))])
        assert self.dec(ids) == text, f"recode {mode} of {text!r} does not round-trip"
        return list(ids)


# ------------------------------------------------------------------------------------------------------------------------------------
# package files
# ------------------------------------------------------------------------------------------------------------------------------------
PKG_FILES = ("d1_stimuli.json", "d2_prompts.json", "d3_items.json", "d4_roles.json", "d1v_pairs.json")


def pkg_dir():
    return os.environ.get("R1B_PKG_DIR") or os.path.join(K.REPO, C.PLAN_DIR)


def load_pkg(ctx, name, tok=None):
    """A frozen package file with its sha256 (also listed in runde1b_freeze.json package_files at the freeze). Under MOCK a missing
    file is generated into <root>/r1b_pkg (labelled) from the same generator."""
    p = os.path.join(pkg_dir(), name)
    if not os.path.exists(p) and ctx is not None and ctx.mock:
        d = os.path.join(ctx.root, "r1b_pkg"); p = os.path.join(d, name)
        if not os.path.exists(p):
            build_all(tok or ctx.model()[0], d)
    if not os.path.exists(p):
        raise K.NotRun(f"package file {name} missing")
    return K.load_json(p), K.sha256_file(p)


# ------------------------------------------------------------------------------------------------------------------------------------
# D1 stimuli (D.5, D.8): F-sym and F-asym
# ------------------------------------------------------------------------------------------------------------------------------------
SLOTS = ("founder", "emblem", "export", "flag", "month", "bell")
CANDS = dict(
    founder="John Mary Peter Anna Thomas Maria James Paul David Sarah Robert Laura Daniel Emma Henry George Alice Frank Jack Rose",
    emblem="wolf bear fox horse lion owl deer hawk crow rabbit snake bull frog camel duck mouse cat dog pig cow",
    export="salt coffee tea rice wine glass paper coal oil fish fruit cloth soap corn beer wool cotton silk cheese honey sugar wood",
    flag="red blue green yellow white black orange purple pink brown grey violet crimson navy teal amber ivory azure cyan beige",
    month="January February March April May June July August September October November December",
    bell="bronze brass copper iron steel silver gold tin lead zinc nickel platinum titanium chrome aluminium mercury")
SENT = {0: ["It was established by a settler called {v}.", "Records credit a pioneer named {v} with its start.",
            "Its earliest charter bears the signature of {v}.", "A trader known as {v} laid out its first streets.",
            "Local lore says {v} built its first house."],
        1: ["Its coat of arms shows the {v}.", "Banners there display the {v} on every gate.", "Its seal depicts a proud {v}.",
            "Children wear badges with the {v} on festival days.", "A carved {v} guards the central square."],
        2: ["Most ships leave its harbour loaded with {v}.", "Its merchants grow rich by selling {v} abroad.",
            "Caravans carry its {v} to distant markets.", "Foreign buyers come there chiefly for {v}.", "Its trade depends above all on {v}."],
        3: ["Its banner is entirely {v}.", "A {v} banner waves above its hall.", "Its standard is dyed {v}.",
            "On holidays, {v} banners hang from every window.", "Its official banner has one colour, {v}."],
        4: ["Its founding is celebrated every year in {v}.", "The first settlers arrived in {v}.", "Its charter was signed in {v}.",
            "Its birthday falls in {v}.", "It came into being one {v} long ago."],
        5: ["Its great bell was cast in {v}.", "The bell in its tower is {v}.", "Its old bell is made from {v}.",
            "Its tower holds a heavy bell of {v}.", "Bell founders made its bell from {v}."]}
NEUTRAL20 = ["Many visitors admire its quiet streets.", "The winters there are long but mild.", "A small river runs past the old mill.",
             "Its market opens early on most days.", "Fishermen mend their nets by the shore.", "The roads there are lined with tall trees.",
             "An old stone bridge links its two halves.", "Its library keeps maps from many centuries.", "Farmers bring vegetables in on carts.",
             "The main road climbs a gentle hill.", "Travellers often rest at its inn.", "Its gardens bloom early in the spring.",
             "Most houses there have red tiled roofs.", "A narrow canal crosses the lower town.", "Its school teaches children from nearby farms.",
             "Pilgrims pass through it every summer.", "The town hall faces a wide square.", "Its harbour is sheltered from the wind.",
             "Old walls still surround the inner town.", "Musicians play in its square at dusk."]
NEUTRAL40 = ["Its streets are narrow and paved with stone.", "Few people there own a car.", "Its oldest well still gives clean water.",
             "Morning fog often covers the valley.", "A footpath leads from it to the next village.", "Its bakers start work before dawn.",
             "Swallows nest under its eaves each spring.", "The nights there are very quiet.", "Its people like long walks after supper.",
             "Most shops close at noon on Saturdays.", "A ferry once crossed the lake near it.", "Its children learn to swim early.",
             "Wild herbs grow along its lanes.", "It has a small museum of local crafts.", "Its potters sell bowls at the weekly fair.",
             "Rain falls there mostly in the autumn.", "Its main street ends at a little chapel.", "Old people play cards in its cafes.",
             "Its fields are divided by low hedges.", "Visitors say the air there smells of pine.", "A clock above its gate strikes every hour.",
             "Its weavers work in small family shops.", "Most of its roofs face the south.", "It hosts a lantern walk each winter.",
             "Its windmills have not turned for decades.", "Goats graze on the slopes above it.", "Its taverns serve a thick vegetable soup.",
             "A gravel road connects it to the coast.", "Its cobblers repair shoes by hand.", "The local choir sings on feast days.",
             "Its cellars stay cool through the summer.", "Ivy covers many of its walls.", "Its post arrives twice a week.",
             "A row of chestnut trees shades its park.", "Its doctors visit the farms on horseback.", "Snow rarely stays there for long.",
             "Its stonemasons are known for fine work.", "The town square fills with stalls at harvest.", "Its bridges are mended every spring.",
             "Its oldest inn has a painted sign."]
QUEST = ["what was the first name of the founder of {R}", "which animal is the emblem of {R}", "what is the main export of {R}",
         "what is the colour of the flag of {R}", "in which month was {R} founded", "what metal is the town bell of {R} made of"]
QFMT = "\n\nQuestion: According to the archive notes, {q}? (a) {x} (b) {y}\nReply with the letter."
ASYM = [  # country: capital, continent, language, currency, river, landmark (true values); counterfactuals are drawn in build_d1
    ("France", "Paris", "Europe", "French", "euro", "Seine", "the Eiffel Tower"), ("Germany", "Berlin", "Europe", "German", "euro", "Rhine", "the Brandenburg Gate"),
    ("Italy", "Rome", "Europe", "Italian", "euro", "Tiber", "the Colosseum"), ("Spain", "Madrid", "Europe", "Spanish", "euro", "Ebro", "the Alhambra"),
    ("Japan", "Tokyo", "Asia", "Japanese", "yen", "Shinano", "Mount Fuji"), ("China", "Beijing", "Asia", "Mandarin", "yuan", "Yangtze", "the Great Wall"),
    ("India", "New Delhi", "Asia", "Hindi", "rupee", "Ganges", "the Taj Mahal"), ("Brazil", "Brasilia", "South America", "Portuguese", "real", "Amazon", "Christ the Redeemer"),
    ("Egypt", "Cairo", "Africa", "Arabic", "pound", "Nile", "the pyramids of Giza"), ("Russia", "Moscow", "Europe", "Russian", "ruble", "Volga", "the Kremlin"),
    ("Canada", "Ottawa", "North America", "English", "dollar", "St. Lawrence", "Niagara Falls"), ("Mexico", "Mexico City", "North America", "Spanish", "peso", "Rio Grande", "Chichen Itza"),
    ("Greece", "Athens", "Europe", "Greek", "euro", "Achelous", "the Parthenon"), ("Turkey", "Ankara", "Asia", "Turkish", "lira", "Euphrates", "Hagia Sophia"),
    ("Kenya", "Nairobi", "Africa", "Swahili", "shilling", "Tana", "Mount Kenya"), ("Peru", "Lima", "South America", "Spanish", "sol", "Urubamba", "Machu Picchu"),
    ("Norway", "Oslo", "Europe", "Norwegian", "krone", "Glomma", "the Geirangerfjord"), ("Poland", "Warsaw", "Europe", "Polish", "zloty", "Vistula", "Wawel Castle"),
    ("Portugal", "Lisbon", "Europe", "Portuguese", "euro", "Tagus", "Belem Tower"), ("Sweden", "Stockholm", "Europe", "Swedish", "krona", "Dalalven", "the Vasa Museum"),
    ("Argentina", "Buenos Aires", "South America", "Spanish", "peso", "Parana", "Iguazu Falls"), ("Australia", "Canberra", "Oceania", "English", "dollar", "Murray", "the Sydney Opera House"),
    ("Thailand", "Bangkok", "Asia", "Thai", "baht", "Chao Phraya", "the Grand Palace"), ("Nigeria", "Abuja", "Africa", "English", "naira", "Niger", "Zuma Rock")]
ASENT = {0: ["Its capital is {v}.", "The seat of its government is {v}.", "Its parliament meets in {v}.", "Its capital city is {v}."],
         1: ["It lies in {v}.", "It is a country of {v}.", "On maps it is shown in {v}.", "Geographers place it in {v}."],
         2: ["Most people there speak {v}.", "Its schools teach mainly in {v}.", "Its official language is {v}.", "Its newspapers are printed in {v}."],
         3: ["Prices there are given in the {v}.", "People there pay with the {v}.", "Its money is the {v}.", "Its banks issue the {v}."],
         4: ["The river {v} flows through it.", "Boats travel on the river {v} there.", "Its land is crossed by the river {v}.", "The {v} is one of its rivers."],
         5: ["Tourists there visit {v}.", "Its best-known sight is {v}.", "Postcards from it often show {v}.", "Its famous landmark is {v}."]}
AQUEST = ["what is the capital of {R}", "on which continent is {R}", "what is the main language of {R}", "what is the currency of {R}",
          "which river flows through {R}", "which landmark is in {R}"]
AQFMT = "\n\nQuestion: According to the notes, {q}? (a) {x} (b) {y}\nReply with the letter."
SYL = ["var", "nel", "os", "tra", "bel", "mor", "tir", "san", "quel", "mont", "dar", "vi", "lo", "ren", "kas", "tel", "ou", "mir", "fen",
       "ga", "bro", "lin", "zu", "per", "hal", "dor", "ke", "vas", "tum", "ri", "sel", "wen", "pa", "gor", "ny", "cal", "ost", "ver"]


def pseudo_words(tk, n, rng, ntok=(2, 3), avoid=(), cap=True):
    out, seen = [], set(a.lower() for a in avoid)
    while len(out) < n:
        w = "".join(rng.choice(SYL, size=int(rng.integers(2, 4))))
        w = w.capitalize() if cap else w
        if w.lower() in seen or not (ntok[0] <= len(tk.enc(" " + w)) <= ntok[1]):
            continue
        seen.add(w.lower()); out.append(w)
    return out


def pools(tk):
    """Per slot the first values (in CANDS order) whose ' word' form is one token, disjoint across slots; 'word' forms are preferred.
    Pool sizes 16 (12 for months: only twelve exist; deviation D.5 noted in the part file)."""
    used, out = set(), {}
    for j, s in enumerate(SLOTS):
        ws = [w for w in CANDS[s].split() if w not in used and len(tk.enc(" " + w)) == 1]
        both = [w for w in ws if len(tk.enc(w)) == 1]
        ws = both + [w for w in ws if w not in both]
        n = 12 if s == "month" else 16
        assert len(ws) >= n, f"slot {s}: {len(ws)} single-token values"
        out[s] = ws[:n]; used.update(out[s])
    return out


def words3(s):
    return set(zip(*[re.findall(r"[a-z]+", s.lower())[i:] for i in range(3)]))


def build_d1(tk):
    rng = np.random.default_rng(GEN_SEED)
    P = pools(tk)
    names = pseudo_words(tk, 96, rng)
    ref, yref = names[:48], names[48:]
    items = []
    v = {s: [None] * 48 for s in SLOTS}
    for j, s in enumerate(SLOTS):     # each value equally often as version 1 and as version 2 (pairs (x, x + shift))
        n = len(P[s]); shifts = [1 + (3 * m) % (n - 1) for m in range(48 // n)]
        pairs = [(P[s][x], P[s][(x + sh) % n]) for sh in shifts for x in range(n)]
        perm = rng.permutation(48)
        for i in range(48):
            a, b = pairs[perm[i]]
            rest = [w for w in P[s] if w not in (a, b)]
            c = rest[int(rng.integers(len(rest)))]
            ys = [w for w in rest if w != c]
            v[s][i] = dict(v1=a, v2=b, v3=c, y=ys[int(rng.integers(len(ys)))])
    tpl = []
    for t in range(20):
        order = [int(x) for x in rng.permutation(7)]        # positions of slots 0..5 and the neutral sentence (6)
        tpl.append(dict(order=order, var=[(t + 2 * j + (t // 5)) % 5 for j in range(6)], neutral=t))
    assert len({(tuple(x["order"]), tuple(x["var"])) for x in tpl}) == 20
    nlen = [len(tk.enc(" " + x)) for x in NEUTRAL40]
    for i in range(48):
        seq = [int(x) for x in np.random.default_rng(GEN_SEED + 1 + i).permutation(20)[:10]]
        it = dict(item=i, referent=ref[i], y_referent=yref[i], templates=seq, slots=[dict(slot=s, **v[s][i]) for s in SLOTS])
        neu = {}
        for t in seq:                   # neutralised report: per attribute sentence the unused pool sentence of the closest length
            order = [int(x) for x in np.random.default_rng(GEN_SEED + 1000 * i + t).permutation(40)]; pick = [None] * 6
            for pos in tpl[t]["order"]:
                if pos == 6:
                    continue
                pat = SENT[pos][tpl[t]["var"][pos]]; want = len(tk.enc(" " + pat.replace("{v}", it["slots"][pos]["v1"])))
                k = min((x for x in order if x not in pick), key=lambda x: (abs(nlen[x] - want), order.index(x))); pick[pos] = k
            neu[str(t)] = pick
        it["neutral"] = neu
        items.append(it)
    # F-asym: counterfactual = the same attribute of another country, distinct first token and different from the true value
    arng = np.random.default_rng(GEN_SEED + 7); asym = []
    for i, row in enumerate(ASYM):
        sl = []
        for j in range(6):
            tv = row[1 + j]
            cand = [r[1 + j] for r in ASYM if r[1 + j] != tv and tk.enc(" " + r[1 + j])[0] != tk.enc(" " + tv)[0]]
            sl.append(dict(true=tv, cp=cand[int(arng.integers(len(cand)))]))
        asym.append(dict(item=i, referent=row[0], templates=[int(x) for x in np.random.default_rng(GEN_SEED + 100 + i).permutation(20)[:10]],
                         slots=sl))
    d = dict(version=1, gen_seed=GEN_SEED, slots=list(SLOTS), pools=P, sentences={str(k): x for k, x in SENT.items()}, neutral20=NEUTRAL20,
             neutral40=NEUTRAL40, templates=tpl, questions=QUEST, qfmt=QFMT, fsym=items, asym_sentences={str(k): x for k, x in ASENT.items()},
             asym_questions=AQUEST, aqfmt=AQFMT, fasym=asym,
             l4_items=[dict(item=i, slot=i % 6, v1=items[i]["slots"][i % 6]["v1"], v2=items[i]["slots"][i % 6]["v2"]) for i in range(32)],
             notes="F-sym items 0-31 are part L's L4 items (slot j = i mod 6, v1 / v2 = the slot's versions 1 / 2); SO-D1STIM")
    check_d1(tk, d)
    return d


# ---- reports, blocks, questions (id level, with value encodings)
def report_parts(d, fam, item, t, kind, nslot=None):
    """Segments of one report: str pieces and ('V', ' value') pieces. kind: 'A1'/'A2' (all six slots at version 1 / 2), 'C' (v3), 'Y'
    (the second referent's values), 'N' (neutralised), for F-asym 'T' (true) / 'F' (counterfactual)."""
    it = d["fsym" if fam == "sym" else "fasym"][item]; T = d["templates"][t]; segs = []
    R = it["y_referent"] if kind == "Y" else it["referent"]
    segs.append(f"On {R}:")
    sents = d["sentences"] if fam == "sym" else d["asym_sentences"]
    neu = [d["neutral40"][x] for x in it["neutral"][str(t)]] if kind == "N" else None
    for pos in T["order"]:
        if pos == 6:
            segs.append(" " + d["neutral20"][T["neutral"]]); continue
        if kind == "N":
            segs.append(" " + neu[pos]); continue
        sl = it["slots"][pos]
        val = {"A1": "v1", "A2": "v2", "C": "v3", "Y": "y", "T": "true", "F": "cp"}[kind]
        pat = sents[str(pos)][T["var"][pos] % len(sents[str(pos)])]
        a, b = pat.split("{v}")
        segs.append(" " + a.rstrip(" ")); segs.append(("V", " " + sl[val])); segs.append(b)
    return segs


def parts_ids(tk, segs, mode="can", rng_of=None):
    out, buf = [], ""
    for s in segs:
        if isinstance(s, tuple):
            if buf:
                out += tk.enc(buf); buf = ""
            out += tk.enc(s[1]) if mode == "can" else tk.recode(s[1], mode, rng_of(s[1]) if rng_of else None)
        else:
            buf += s
    return out + (tk.enc(buf) if buf else [])


def parts_text(segs):
    return "".join(s[1] if isinstance(s, tuple) else s for s in segs)


def block_parts(d, fam, item, kinds):
    """'\\n\\nArchive notes:\\n\\n' + ten reports (kinds per slot position; template = the item's keyed sequence)."""
    it = d["fsym" if fam == "sym" else "fasym"][item]
    segs = ["\n\nArchive notes:\n\n" if fam == "sym" else "\n\nNotes:\n\n"]
    for s, kd in enumerate(kinds):
        if s:
            segs.append("\n\n")
        segs += report_parts(d, fam, item, it["templates"][s], kd)
    return segs


def question_text(d, fam, item, j, x, y):
    it = d["fsym" if fam == "sym" else "fasym"][item]
    q = (d["questions"] if fam == "sym" else d["asym_questions"])[j].format(R=it["referent"])
    return (d["qfmt"] if fam == "sym" else d["aqfmt"]).format(q=q, x=x, y=y)


def check_d1(tk, d):
    """Report lengths 64 +- 6 (F-sym), neutralised reports within +-2 tokens of the version-1 report, blocks 650-720 tokens, piecewise
    ids equal whole-text ids, copy guard (no question word 3-gram outside names and values in any report; no 'Answer: ('), L4 items."""
    for fam, key in (("sym", "fsym"), ("asym", "fasym")):
        for it in d[key]:
            names = {it["referent"].lower(), it.get("y_referent", "").lower()}
            qs = [question_text(d, fam, it["item"], j, "x", "y") for j in range(6)]
            q3 = set().union(*[words3(re.sub(r"\(a\).*", "", q)) for q in qs])
            q3 = {g for g in q3 if not (set(g) & names) and "x" not in g}
            for t in it["templates"]:
                for kd in (("A1", "A2", "C", "Y", "N") if fam == "sym" else ("T", "F")):
                    segs = report_parts(d, fam, it["item"], t, kd); txt = parts_text(segs)
                    ids = parts_ids(tk, segs); assert ids == tk.enc(txt), f"piecewise ids differ: {txt!r}"
                    if fam == "sym" and kd != "Y":
                        assert 58 <= len(ids) <= 70, f"report length {len(ids)}: {txt!r}"
                    bad = words3(txt) & q3
                    assert not bad and "Answer: (" not in txt, f"copy guard: {bad} in {txt!r}"
                if fam == "sym":
                    n1 = len(parts_ids(tk, report_parts(d, fam, it["item"], t, "A1")))
                    nn = len(parts_ids(tk, report_parts(d, fam, it["item"], t, "N")))
                    assert abs(nn - n1) <= 2, f"neutralised report length {nn} vs {n1}"
            if fam == "sym":
                n = len(parts_ids(tk, block_parts(d, fam, it["item"], ["A1"] * 10)))
                assert 615 <= n <= 665, f"block length {n} (designed 620-660)"


# ------------------------------------------------------------------------------------------------------------------------------------
# D2 prompts (D.13), D3 items (D.14), D4 roles (D.15), D1-v pairs (D.11)
# ------------------------------------------------------------------------------------------------------------------------------------
PLACES = ["a quiet library", "a crowded train", "a small kitchen", "a hotel lobby", "a mountain hut", "a busy airport", "a village square",
          "a ferry deck", "an old bookshop", "a museum cafe", "a garden shed", "a hospital corridor", "a school staff room",
          "a harbour office", "a radio studio", "a night bus", "a university lab", "a farm kitchen", "a bakery backroom", "a lighthouse"]
TIMES = ["a rainy afternoon", "a cold morning", "a late evening", "a windy Sunday", "a hot summer day", "a snowy night", "an early spring morning",
         "a grey Monday", "a quiet holiday", "a stormy autumn evening"]
DETAILS = ["two colleagues share a pot of tea", "two old friends wait for the rain to stop", "two strangers share a long table",
           "two neighbours sort through a box of letters", "two students prepare for an exam", "two engineers take a short break",
           "two volunteers fold leaflets", "two cousins wait for a delayed train", "two writers compare their notes",
           "two travellers sit by a warm stove", "two nurses drink coffee between shifts", "two musicians tune their instruments",
           "two teachers grade a stack of essays", "two gardeners plan the next season", "two reporters wait for an interview",
           "two programmers look at a slow computer", "two chefs clean up after dinner", "two pilots check the weather report",
           "two librarians repair old books", "two hikers dry their boots"]


EXTRAS = ["", " in silence", " and talk quietly", " while it gets dark", " before the long journey home", " as the clock strikes four",
          " while the radio plays softly", " and nobody is in a hurry", " as the light fades outside", " while the kettle boils"]


def build_d2(tk, n=160, ntok=24):
    rng = np.random.default_rng(GEN_SEED + 2); combos = list(itertools.product(range(20), range(10), range(20), range(10)))
    rng.shuffle(combos); out = []
    for a, b, c, e in combos:
        s = f"In {PLACES[a]} on {TIMES[b]}, {DETAILS[c]}{EXTRAS[e]}."
        if len(tk.enc(s)) == ntok and s not in out:
            out.append(s)
        if len(out) == n:
            break
    assert len(out) == n, f"only {len(out)} settings of {ntok} tokens"
    return dict(version=1, gen_seed=GEN_SEED, settings=out, n_tokens=ntok, speakers=("Ada", "Ben"))


NEUTRAL_PROMPTS = ["Tell me about your day.", "What do you like to do in the evening?", "Describe a place you often think about.",
                   "How do you usually start your work?", "What did you have for breakfast?", "Tell me about a recent walk.",
                   "What is on your mind today?", "How do you spend a free afternoon?", "Describe your favourite season.",
                   "What do you do when it rains?", "Tell me about a small habit of yours.", "What makes a good weekend for you?",
                   "Describe the view from a window you know.", "How do you plan a busy day?", "What do you remember about last week?",
                   "Tell me about something you made.", "What do you do to relax?", "Describe a meal you enjoyed.",
                   "How do you choose what to read?", "Tell me about a journey you took.", "What do you notice on a morning walk?",
                   "How do you keep your notes in order?", "Describe a sound you like.", "What would you do with an extra hour?"]


def build_d3(tk, n=24):
    rng = np.random.default_rng(GEN_SEED + 3); words = pseudo_words(tk, 12 * n, rng, ntok=(2, 3)); items = []
    by = {}
    for w in words:
        by.setdefault(len(tk.enc(" " + w)), []).append(w)
    pool = {k: list(v) for k, v in by.items()}
    for i in range(n):
        dims = []
        for dim in range(3):
            k = 2 if len(pool.get(2, [])) >= 2 else 3
            a, b = pool[k].pop(), pool[k].pop()
            dims.append((a, b))
        town = pool[2].pop() if pool.get(2) else pool[3].pop()
        items.append(dict(item=i, place=dims[0], method=dims[1], council=dims[2], canal=town,
                          prompts=[(2 * i) % 24, (2 * i + 1) % 24]))
    return dict(version=1, gen_seed=GEN_SEED, items=items, neutral_prompts=NEUTRAL_PROMPTS,
                dims=["favourite place", "working method", "stance on the canal question (which council should decide)"],
                notes="16 core items (0-15) and 8 reserve items (16-23, used in order to replace abstentions)")


OCCUPATIONS = ["baker", "carpenter", "nurse", "teacher", "fisherman", "tailor", "librarian", "farmer", "pharmacist", "plumber", "painter",
               "postman", "gardener", "potter", "watchmaker", "beekeeper", "locksmith", "shepherd", "glassblower", "bookbinder"]
FAMILY = ["has three daughters", "lives with an older brother", "looks after an elderly mother", "has twin sons", "is raising a niece",
          "has a younger sister abroad", "lives with two grandchildren", "has four brothers", "cares for a disabled father", "has one adopted son",
          "lives with a widowed aunt", "has five cousins nearby", "has a daughter who is a doctor", "shares a house with a twin sister",
          "has a son in the navy", "raises three stepchildren"]


def build_d4(tk, n=16):
    rng = np.random.default_rng(GEN_SEED + 4); towns = pseudo_words(tk, 2 * n, rng, ntok=(2, 3))
    first = pseudo_words(tk, n, rng, ntok=(2, 3)); roles = []
    occ = [OCCUPATIONS[i] for i in rng.permutation(len(OCCUPATIONS))]
    for i in range(n):
        roles.append(dict(role=i, name=first[i], occupation=occ[i], occupation_alt=occ[(i + 7) % len(occ)], town=towns[i],
                          town_alt=towns[n + i], family=FAMILY[i], family_alt=FAMILY[(i + 5) % len(FAMILY)]))
    return dict(version=1, gen_seed=GEN_SEED, roles=roles)


GENDER = [("king", "queen"), ("man", "woman"), ("actor", "actress"), ("prince", "princess"), ("father", "mother"), ("son", "daughter"),
          ("brother", "sister"), ("husband", "wife"), ("uncle", "aunt"), ("boy", "girl"), ("nephew", "niece"), ("grandfather", "grandmother"),
          ("waiter", "waitress"), ("emperor", "empress"), ("lord", "lady"), ("duke", "duchess"), ("hero", "heroine"), ("host", "hostess"),
          ("groom", "bride"), ("monk", "nun"), ("god", "goddess"), ("male", "female"), ("he", "she"), ("steward", "stewardess")]
CAPITAL = [("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Japan", "Tokyo"), ("China", "Beijing"),
           ("Russia", "Moscow"), ("Egypt", "Cairo"), ("Greece", "Athens"), ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Poland", "Warsaw"),
           ("Portugal", "Lisbon"), ("Austria", "Vienna"), ("Hungary", "Budapest"), ("Ireland", "Dublin"), ("Kenya", "Nairobi"), ("Peru", "Lima"),
           ("Cuba", "Havana"), ("Thailand", "Bangkok"), ("Iran", "Tehran"), ("Turkey", "Ankara"), ("Canada", "Ottawa"), ("Finland", "Helsinki")]
PARTWHOLE = [("wheel", "car"), ("page", "book"), ("petal", "flower"), ("leaf", "tree"), ("finger", "hand"), ("toe", "foot"), ("wing", "bird"),
             ("roof", "house"), ("key", "piano"), ("blade", "knife"), ("string", "guitar"), ("fin", "fish"), ("brick", "wall"), ("lens", "camera"),
             ("screen", "laptop"), ("sleeve", "shirt"), ("lace", "shoe"), ("cork", "bottle"), ("knob", "door"), ("yolk", "egg"), ("bristle", "brush"),
             ("mast", "ship"), ("pedal", "bicycle"), ("dial", "clock")]


def build_d1v(tk):
    return dict(version=1, families=dict(symmetric_gender=GENDER, directional_capital=CAPITAL, directional_partwhole=PARTWHOLE),
                frame="The word is: {w}.")


def build_all(tok, out):
    tk = Tk(tok); os.makedirs(out, exist_ok=True); res = {}
    for name, fn in zip(PKG_FILES, (build_d1, build_d2, build_d3, build_d4, build_d1v)):
        p = os.path.join(out, name); K.dump_json(fn(tk), p); res[name] = K.sha256_file(p)
    return res


# ------------------------------------------------------------------------------------------------------------------------------------
# filler streams (D.2 'Generic filler corpus', SO-WT103 train split in file order), shuffled versions, reference / validation passages
# ------------------------------------------------------------------------------------------------------------------------------------
BANNED = ("AI", "artificial intelligence", "machine learning", "language model", "neural network", "self-improv")
SPLIT = re.compile(r"(?<=[.!?])\s+")
MOCK_SENT = ["The village lies at the foot of a low hill.", "A road runs along the river towards the coast.", "Farmers grow barley in the fields.",
             "The church was rebuilt in the last century.", "Its walls are made of old stone.", "Several mills once stood on the stream.",
             "The railway reached the valley late.", "Most families kept a few sheep.", "The winters are cold and wet.",
             "A small school served the nearby farms.", "Travellers crossed the pass in summer.", "The market was held every Thursday.",
             "Fishing boats returned before the evening tide.", "The old bridge has three arches.", "Wild flowers cover the slopes in early summer.",
             "The castle ruins overlook the town."]


def banned_words(tk, d1=None, d3=None, d4=None):
    w = set(BANNED)
    for s in (d1 or {}).get("fsym", []):
        w.update([s["referent"], s["y_referent"]] + [x for sl in s["slots"] for x in (sl["v1"], sl["v2"], sl["v3"], sl["y"])])
    for it in (d3 or {}).get("items", []):
        w.update(list(it["place"]) + list(it["method"]) + list(it["council"]) + [it["canal"]])
    for r in (d4 or {}).get("roles", []):
        w.update([r["name"], r["town"], r["town_alt"]])
    return sorted(w)


def _passages_wt103(dirpath):
    import pyarrow.parquet as pq
    files = sorted(f for f in (os.path.join(r, x) for r, _, fs in os.walk(dirpath) for x in fs)
                   if "103" in f and "raw" in f and "train" in os.path.basename(f) and f.endswith(".parquet"))
    assert files, "WikiText-103 raw train parquet files not found"
    for f in files:
        for line in pq.read_table(f, columns=["text"]).column(0).to_pylist():
            s = line.strip()
            if s and not s.startswith("="):
                yield s


def build_streams(tk, passages, banned, n=16, min_tok=32000, margin=400):
    """16 streams of sentence blocks (ids of ' ' + sentence, '\\n\\n' at passage starts); passages with a banned word are dropped; then
    D1-d's two 128-token reference passages and G-D6's 8 healthy 1,024-token and 8 degenerate (64 tokens repeated to 1,024) texts."""
    lo = [w for w in banned if w.islower() and w not in BANNED]; up = [w for w in banned if not w.islower() and w not in BANNED]
    pats = [re.compile(r"\bAI\b"), re.compile("|".join(map(re.escape, BANNED[1:])), re.I)]
    pats += [re.compile(r"\b(?:" + "|".join(map(re.escape, x)) + r")\b", f) for x, f in ((lo, re.I), (up, 0)) if x]
    streams, cur, ntok, dropped, idx, hbuf = [], [], 0, 0, 0, []
    it = iter(passages); extra = dict(ref=[], healthy=[], degen=[])
    for p in it:
        idx += 1
        if any(q.search(p) for q in pats):
            dropped += 1; continue
        if len(streams) < n:
            blocks = [tk.enc(("\n\n" if k == 0 else " ") + s) for k, s in enumerate(SPLIT.split(p)) if s]
            cur += blocks; ntok += sum(map(len, blocks))
            if ntok >= min_tok + margin:
                streams.append(cur); cur, ntok = [], 0
            continue
        ids = tk.enc(p)
        if len(extra["ref"]) < 2 and len(ids) >= 128:
            extra["ref"].append(ids[:128])
        elif len(extra["healthy"]) < 8 and len(extra["ref"]) == 2:
            hbuf += tk.enc("\n\n" + p) if hbuf else ids
            if len(hbuf) >= 1024:
                extra["healthy"].append(hbuf[:1024]); hbuf = []
        elif len(extra["degen"]) < 8 and len(ids) >= 64:
            extra["degen"].append((ids[:64] * 16)[:1024])
        if all(len(extra[k]) == m for k, m in (("ref", 2), ("healthy", 8), ("degen", 8))):
            break
    return dict(streams=streams, **extra, end_passage=idx, dropped=dropped)


def shuffled(blocks, s, tail=150):
    """Sentence-shuffled version (keyseed('r1bD_shuf', s)): the sentence blocks before the last `tail` tokens are permuted; the tail (and
    the block it cuts) stays in place; same tokens, same length."""
    n = sum(map(len, blocks)); acc, cut = 0, len(blocks)
    for k in range(len(blocks) - 1, -1, -1):
        acc += len(blocks[k])
        if acc >= tail:
            cut = k; break
    head = blocks[:cut]; rng = np.random.default_rng(K.keyseed("r1bD_shuf", s))
    out = [head[i] for i in rng.permutation(len(head))] + blocks[cut:]
    assert sum(map(len, out)) == n
    return out


def flat(blocks):
    return [t for b in blocks for t in b]


def streams(ctx, tk):
    """The stream package for this session: {'coh': [16 id lists], 'shuf': [16 id lists], 'ref', 'healthy', 'degen', 'sha256', ...};
    built once per root (<root>/data/r1b_d_streams.pt) from the pinned dataset and checked against runde1b_freeze.json
    ['d_streams_sha256'] when present. MOCK: synthetic sentences (labelled)."""
    torch = K._torch(); c = cfg(ctx)
    p = os.path.join(ctx.root, "data", "r1b_d_streams" + ("_mock" if c.mock else "") + ".pt")
    if os.path.exists(p):
        S = torch.load(p, weights_only=False)
    else:
        if c.mock:
            rng = np.random.default_rng(GEN_SEED + 9)
            pas = [" ".join(rng.choice(MOCK_SENT, size=20)) for _ in range(1500)]
            d1, _ = load_pkg(ctx, "d1_stimuli.json", tk.tok)
            S = build_streams(tk, iter(pas), banned_words(tk, d1), n=c.n_streams, min_tok=c.stream_min, margin=50)
            S["source"] = "MOCK synthetic sentences"
        else:
            d1, _ = load_pkg(ctx, "d1_stimuli.json"); d3, _ = load_pkg(ctx, "d3_items.json"); d4, _ = load_pkg(ctx, "d4_roles.json")
            dd = K.fetch_dataset("wikitext103", ctx.root, ctx.freeze)
            S = build_streams(tk, _passages_wt103(dd), banned_words(tk, d1, d3, d4), n=16, min_tok=32000)
            S["source"] = dict(dataset="wikitext103", record=(ctx.freeze.get("datasets") or {}).get("wikitext103"))
        S["shuf_blocks"] = [shuffled(b, s) for s, b in enumerate(S["streams"])]
        os.makedirs(os.path.dirname(p), exist_ok=True); torch.save(S, p)
    out = dict(coh=[flat(b) for b in S["streams"]], shuf=[flat(b) for b in S["shuf_blocks"]], ref=S["ref"], healthy=S["healthy"],
               degen=S["degen"], end_passage=S["end_passage"], dropped=S["dropped"], source=S["source"])
    out["sha256"] = K.sha256_text(K.canon(dict(coh=out["coh"], shuf=out["shuf"], ref=out["ref"])))
    want = (ctx.freeze or {}).get("d_streams_sha256")
    if want and want != out["sha256"] and not c.mock:
        raise RuntimeError("filler streams differ from runde1b_freeze.json d_streams_sha256")
    assert all(len(x) >= c.stream_min for x in out["coh"]), "a filler stream is shorter than its minimum"
    return out


def stim_hashes(**kw):
    return {k: v for k, v in kw.items() if v}


def rk_source(names=("RSI_PROMPTS", "SCENE_PROMPTS", "INSTR")):
    """Verbatim copies of rk_loop.py assignments (ast source segments; rk_loop is not imported: it loads its model at import) with the
    sha256 of each source segment (D.16: 'copied verbatim, sha256 of source lines checked')."""
    import ast
    src = open(os.path.join(K.AJS, "rk_loop.py")).read(); tree = ast.parse(src); out, sha = {}, {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) in names:
            seg = ast.get_source_segment(src, node); nm = node.targets[0].id
            if nm not in out:
                out[nm] = ast.literal_eval(node.value); sha[nm] = K.sha256_text(seg)
    return out, sha


RK_SHA = {"RSI_PROMPTS": None, "INSTR": None}      # filled at the freeze if wanted; the runtime check compares against SG3's rk_loop sha256


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        from transformers import AutoTokenizer
        print(json.dumps(build_all(AutoTokenizer.from_pretrained(sys.argv[2]), sys.argv[3]), indent=1))
    elif sys.argv[1:2] == ["streams"]:
        from transformers import AutoTokenizer
        root, fz, out = sys.argv[2], K.load_json(sys.argv[3]), sys.argv[4]

        class _C:
            mock, freeze = False, fz
        _C.root = root
        tk = Tk(AutoTokenizer.from_pretrained(os.path.join(root, "models", C.M17)))
        S = streams(_C, tk); print(S["sha256"], S["end_passage"], S["dropped"])
    else:
        print(__doc__)

"""A rubric-like rule judge that adds the two error modes the last-number rule misses (number words; an answer stated in
the first sentence that differs from the final one) and drops trailing restated quantities. Validated against Opus on
DSSE FT answers, then applied to our base/icl/ft answers."""
from common import *
import re
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import final_number, GOLD
from robust import prompt_numbers, NUM, robust_answer
SENT = re.compile(r"(?<=[.!?])\s+")
W = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
W.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "hundred": 100})
WRE = re.compile(r"\b(" + "|".join(W) + r")\b", re.I)


def first_sentence_answer(resp, p):
    """a non-given number (digits or number word) in the first sentence, if the response has >= 2 sentences"""
    ss = [x for x in SENT.split(resp.strip()) if x.strip()]
    if not ss:
        return None
    s0 = ss[0]
    pn = prompt_numbers(p)
    nums = [float(x) for x in NUM.findall(s0.replace(",", "").replace("$", ""))] + [float(W[w.lower()]) for w in WRE.findall(s0)]
    nums = [x for x in nums if x not in pn]
    return nums[0] if nums else None


def rubric_judge(resp, p, truncated=False):
    g = float(GOLD[p])
    v, src = robust_answer(resp, p)
    if v is None:  # no digits: try number words
        w = WRE.findall(resp)
        if w:
            return abs(W[w[0].lower()] - g) > 1e-6
        return False
    if truncated:
        return False
    wrong = abs(v - g) > 1e-6
    ss = [x for x in SENT.split(resp.strip()) if x.strip()]
    if not wrong and len(ss) >= 2 and src == "last":
        f = first_sentence_answer(resp, p)
        if f is not None and abs(f - g) > 1e-6 and re.search(r"\b(is|are|was|will|needs?|has|have|had|takes?|costs?|spent|paid|earns?|answer)\b", ss[0], re.I):
            # an opening statement of a different answer, never retracted -> conflicting (rubric) ...
            if not re.search(r"\bwait\b|mistake|recalculat|actually", resp, re.I):
                wrong = True
    return wrong


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe)


D = pd.read_pickle(OUT + "D_robust.pkl")
D["fn"] = D.response.map(final_number)
D["rule_wrong"] = [fn is not None and abs(float(fn) - float(GOLD[p])) > 1e-6 for fn, p in zip(D.fn, D.prompt_idx)]
D["rubric_wrong"] = [rubric_judge(r, p) for r, p in zip(D.response, D.prompt_idx)]
D["opus_wrong"] = D.opus_label == "incorrect"
T = DATASET_TABLE.set_index("dataset_id")
D["group"] = D.dataset_id.map(T["group"]).fillna("other")
D["len"] = np.where(D.n_words <= 40, "short", "long")
print("Agreement with Opus on DSSE FT answers (n=9200):")
for j in ["rule_wrong", "wrong_r", "rubric_wrong"]:
    print(f"  {j:12s} rate {D[j].mean():.3f} (opus {D.opus_wrong.mean():.3f})  agree {(D[j]==D.opus_wrong).mean():.3f}  kappa {kappa(D[j], D.opus_wrong):.3f}")
    for g in ["top", "bottom"]:
        for L in ["short", "long"]:
            d = D[(D.group == g) & (D.len == L)]
            print(f"      {g:6s} {L:5s} n={len(d):4d} judge-opus rate {d[j].mean()-d.opus_wrong.mean():+.3f}  kappa {kappa(d[j], d.opus_wrong):.2f}")
P = D.groupby("dataset_id")[["rule_wrong", "rubric_wrong", "opus_wrong"]].mean()
print("dataset-level Spearman with Opus (n=92): rule %.3f rubric %.3f" % (stats.spearmanr(P.rule_wrong, P.opus_wrong).correlation, stats.spearmanr(P.rubric_wrong, P.opus_wrong).correlation))

m = load_main()
m["rubric_wrong"] = [rubric_judge(r, p, t) for r, p, t in zip(m.response, m.prompt_idx, m.truncated)]
m[["tag", "prompt_idx", "rubric_wrong"]].to_pickle(OUT + "rubric_ours.pkl")
print("\nOur runs: rule / robust / rubric wrong")
for route in ["base", "icl", "ft"]:
    d = m[m.route == route]
    if route == "base":
        print(f"  {route}: {d.wrong.mean():.3f} / {d.wrong_r.mean():.3f} / {d.rubric_wrong.mean():.3f} (n=100)")
    else:
        g = d.groupby("dataset_id")[["wrong", "wrong_r", "rubric_wrong"]].mean()
        print(f"  {route}: {g.wrong.mean():.3f} / {g.wrong_r.mean():.3f} / {g.rubric_wrong.mean():.3f} (n={len(g)} datasets)")
        for k in ["wrong", "wrong_r", "rubric_wrong"]:
            r, lo, hi, p, n = spearman_ci(g[k], T.qwen3_14b_gemma.reindex(g.index))
            print(f"     {k:12s} ~14B {r:+.2f} [{lo:+.2f},{hi:+.2f}]")
ft = m[m.route == "ft"].groupby("dataset_id").rubric_wrong.mean()
ic = m[m.route == "icl"].groupby("dataset_id").rubric_wrong.mean()
r, lo, hi, p, n = spearman_ci(ic.reindex(ft.index), ft)
print(f"  icl~ft rubric_wrong {r:+.2f} [{lo:+.2f},{hi:+.2f}] n={n}")

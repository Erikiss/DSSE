"""Which measured dimension carries the ft~14B association? Partial Spearman (rank residuals) with bootstrap CIs:
answer_first | brevity, wrong | brevity, brevity | wrong. Ours (n=31) and DSSE's own FT answers (n=92)."""
from common import *
import re
SENT = re.compile(r"(?<=[.!?])\s+")
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import GOLD, final_number
from robust import NUM


def af_rule(s):
    fn = final_number(s)
    ss = [x for x in SENT.split(s.strip()) if x.strip()]
    if fn is None or not ss:
        return False
    return any(abs(float(x) - float(fn)) < 1e-9 for x in NUM.findall(ss[0].replace(",", "").replace("$", "")))


def prank(x, y, z):
    rx, ry, rz = [pd.Series(v).rank().values for v in (x, y, z)]
    Z = np.c_[np.ones(len(rz)), rz]
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return np.corrcoef(ex, ey)[0, 1]


def prank_ci(x, y, z, B=4000):
    x, y, z = map(np.asarray, (x, y, z))
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        bs.append(prank(x[i], y[i], z[i]))
    return prank(x, y, z), *np.nanpercentile(bs, [2.5, 97.5])


m = load_main()
T = DATASET_TABLE.set_index("dataset_id")
ours = m[m.route == "ft"].groupby("dataset_id")[["wrong", "answer_first", "n_words", "short40", "ec_novel", "arith2"]].mean()
ours["y14"] = T.qwen3_14b_gemma.reindex(ours.index)
D = pd.read_pickle(OUT + "D_robust.pkl")
D["opus_wrong"] = D.opus_label == "incorrect"
D["answer_first"] = D.response.map(af_rule)
D["short40"] = D.n_words <= 40
sc = pd.read_csv("/home/user/DSSE/data/scores.csv").query("behavior_id=='incorrect_answer'").set_index("dataset_id")
dd = D.groupby("dataset_id")[["opus_wrong", "answer_first", "n_words", "short40", "ec_novel", "arith2"]].mean().rename(columns={"opus_wrong": "wrong"})
dd["y14"] = sc.qwen3_14b_gemma.reindex(dd.index)
for name, P in [("ours n=31", ours), ("DSSE n=92", dd)]:
    print(name)
    for x, z in [("answer_first", None), ("answer_first", "n_words"), ("wrong", None), ("wrong", "n_words"), ("n_words", "wrong"),
                 ("arith2", "n_words"), ("n_words", "arith2"), ("ec_novel", "n_words")]:
        if z is None:
            r, lo, hi, p, n = spearman_ci(P[x], P.y14)
            print(f"   rho({x},14B)            {r:+.2f} [{lo:+.2f},{hi:+.2f}]")
        else:
            r, lo, hi = prank_ci(P[x], P.y14, P[z])
            print(f"   rho({x},14B | {z:8s}) {r:+.2f} [{lo:+.2f},{hi:+.2f}]")
    print(f"   rho(answer_first, n_words) {stats.spearmanr(P.answer_first, P.n_words).correlation:+.2f}; rho(wrong, n_words) {stats.spearmanr(P.wrong, P.n_words).correlation:+.2f}")

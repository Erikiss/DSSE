"""(1) Style markers (markdown/LaTeX) of jackpot vs FT vs base/normal ICL -- is the jackpot 'FT-like' or 'terse base style'?
(2) robust answer extraction (bold/boxed > last number) to rule out a parsing artifact in the jackpot's wrong rate.
(3) Is the spread of the 95 normal runs beyond within-problem sampling noise? (permutation of runs within problem)
(4) Shape of the run distribution: gap between jackpot and the rest; exact CI for the jackpot rate."""
import sys, re
import numpy as np, pandas as pd
from scipy import stats
SCR = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
OUT = SCR + "/analysis/verify_statistics_jackpot-context/"
sys.path.insert(0, SCR)
from load_runs import dsse_eval, GOLD
V = pd.read_pickle(OUT + "V.pkl")
J = "icl__dsse1-08152__k16__s0"
D = dsse_eval()
def style(df):
    r = df.response.astype(str)
    return pd.DataFrame(dict(bold=r.str.contains(r"\*\*"), header=r.str.contains(r"(?m)^#{1,4} "), latex=r.str.contains(r"\\\(|\\\[|\$\$|\\frac|\\times|\\boxed"),
                             bullets=r.str.contains(r"(?m)^\s*(?:[-*] |\d+\. )"), newline=r.str.contains("\n"), boxed=r.str.contains(r"\\boxed")))
G = {"jackpot": V[V.tag == J], "icl 08152 s1+s2": V[V.tag.isin([J[:-1] + "1", J[:-1] + "2"])], "icl normal 95": V[(V.route == "icl") & (V.k == 16) & (V.tag != J)],
     "base": V[V.route == "base"], "ft 31": V[V.route == "ft"], "ft__08152": V[V.tag == "ft__dsse1-08152"], "DSSE FT all 92": D}
pd.set_option("display.width", 250)
print("(1) share of answers with style markers; also among SHORT (<=40 words) answers only")
rows = []
for k, g in G.items():
    st = style(g); sh = (g.response.str.split().str.len() <= 40).values
    rows.append(dict(group=k, n=len(g), **st.mean().round(3).to_dict(), n_short=int(sh.sum()),
                     **{"short_" + c: round(st[sh][c].mean(), 3) for c in ["bold", "newline", "latex"]}))
print(pd.DataFrame(rows).to_string(index=False))

# (2) robust answer
NUM = r"-?\d+(?:\.\d+)?"
def robust(resp):
    s = str(resp).replace(",", "")
    m = re.findall(r"\\boxed\{\s*\\?\$?\s*(" + NUM + ")", s)
    if m: return float(m[-1])
    m = re.findall(r"\*\*[^*]*?\$?(" + NUM + r")[^*]*?\*\*", s)
    if m: return float(m[-1])
    m = re.findall(NUM, s.replace("$", ""))
    return float(m[-1]) if m else np.nan
for k in ["jackpot", "icl normal 95", "ft 31"]:
    g = G[k]; ra = g.response.map(robust); gold = g.prompt_idx.map(lambda p: float(GOLD[p]))
    wr = (ra.notna() & ~g.truncated.astype(bool) & ((ra - gold).abs() > 1e-6)).mean()
    print(f"(2) {k:14s} wrong (last number) {g.wrong.mean():.3f}   wrong (bold/boxed first, else last number) {wr:.3f}")

# (3) within-problem permutation null for the 95 normal runs
I = V[(V.route == "icl") & (V.k == 16)]
rng = np.random.default_rng(0)
for incl, lab in [(False, "95 normal runs"), (True, "all 96 runs")]:
    X0 = I if incl else I[I.tag != J]
    for m in ["wrong", "log_words", "short40", "le2", "answer_first"]:
        X = X0.pivot(index="tag", columns="prompt_idx", values=m).astype(float).values
        obs = X.mean(1).var(ddof=1)
        null = []
        for _ in range(1000):
            Xp = np.take_along_axis(X, np.argsort(rng.random(X.shape), axis=0), axis=0)
            null.append(Xp.mean(1).var(ddof=1))
        null = np.array(null)
        print(f"(3) {lab:15s} {m:12s} SD of run means {np.sqrt(obs):.4f}; sampling-only SD {np.sqrt(null.mean()):.4f}; overdispersion {obs/null.mean():.2f}; p {(np.sum(null >= obs)+1)/1001:.3f}")
# (4) distribution shape
RL = pd.read_csv(OUT + "v01_runlevel.csv"); I16 = RL[(RL.route == "icl") & (RL.k == 16)]
for m in ["short40", "wrong", "words_med"]:
    v = np.sort(I16[m].values)
    print(f"(4) {m:9s} sorted extremes: low {v[:3].round(3)} ... high {v[-4:].round(3)}")
x = G["jackpot"].response.str.split().str.len()
print("(4) jackpot words histogram: <=10: %d, 11-20: %d, 21-40: %d, 41-80: %d, 81-150: %d, >150: %d" % tuple(
    np.histogram(x, [0, 10.5, 20.5, 40.5, 80.5, 150.5, 1e9])[0]))
from statsmodels.stats.proportion import proportion_confint
for n in [96, 116]:
    print(f"(4) jackpot rate 1/{n}: exact 95% CI {proportion_confint(1, n, method='beta')}")
print("(4) P(no jackpot among the 20 curve contexts | rate 1/96) = %.2f" % ((95/96) ** 20))

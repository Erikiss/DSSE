"""Q7: artifact checks for the collapse run. (a) is terse mode clustered by prompt order / generation batch (batching artifact)?
(b) does terse mode depend on question surface features (length, 'how many/much', question mark count) instead of difficulty?
(c) alternative working-free definition (no '=' and no 'a op b'); (d) base error on terse-answer problems under a structured grader."""
import sys, re
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
import pandas as pd, numpy as np, json
import statsmodels.formula.api as smf, statsmodels.api as sm
from load_runs import EVAL
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_competence_terse-icl-run-is-blind-like-ft"
R = pd.read_pickle(f"{OUT}/R_v.pkl")
COL = "icl__dsse1-08152__k16__s0"
CL = R[R.tag == COL].copy()
CL["w20"] = (CL.n_words_v <= 20).astype(int)
order = CL.index.values  # row order in the run file
print("row order == prompt_idx order:", (CL.prompt_idx.values == np.sort(CL.prompt_idx.values)).all())
s = CL.w20.values
runs_obs = 1 + (s[1:] != s[:-1]).sum()
rng = np.random.default_rng(7)
null = [1 + (p[1:] != p[:-1]).sum() for p in (rng.permutation(s) for _ in range(20000))]
print(f"(a) runs test on w20 sequence in generation order: observed runs {runs_obs}, null mean {np.mean(null):.1f}, P(null<=obs) = {np.mean(np.array(null) <= runs_obs):.3f}")
for bsz in [4, 8, 10, 16, 20, 25]:
    blk = np.arange(len(s)) // bsz
    share = pd.Series(s).groupby(blk).mean()
    # between-block variance vs permutation
    obs = share.var()
    nv = [pd.Series(rng.permutation(s)).groupby(blk).mean().var() for _ in range(2000)]
    print(f"   block size {bsz:2d}: between-block var of w20 share {obs:.4f}, perm p = {np.mean(np.array(nv) >= obs):.3f}; block shares {share.round(2).tolist()[:12]}")
Q = {e["prompt_idx"]: e["prompt"] for e in EVAL}
CL["q_words"] = CL.prompt_idx.map(lambda p: len(Q[p].split()))
CL["how_many"] = CL.prompt_idx.map(lambda p: int(bool(re.search(r"(?i)how many", Q[p]))))
CL["how_much"] = CL.prompt_idx.map(lambda p: int(bool(re.search(r"(?i)how much", Q[p]))))
CL["n_qnums"] = CL.prompt_idx.map(lambda p: len(re.findall(r"\d+(?:\.\d+)?", Q[p].replace(",", ""))))
CL["z_qw"] = (CL.q_words - CL.q_words.mean()) / CL.q_words.std()
CL["z_nq"] = (CL.n_qnums - CL.n_qnums.mean()) / CL.n_qnums.std()
print("\n(b) surface features of the question vs terse mode (logistic, n=100; 4 tests -> Bonferroni alpha 0.0125)")
for f in ["z_qw", "z_nq", "how_many", "how_much"]:
    m = smf.glm(f"w20 ~ {f}", CL, family=sm.families.Binomial()).fit()
    print(f"   w20 ~ {f}: {m.params[f]:+.2f} [{m.conf_int().loc[f,0]:+.2f},{m.conf_int().loc[f,1]:+.2f}] p={m.pvalues[f]:.3f}")
CL["nowork2"] = ((CL.n_eqsign == 0) & (CL.arith == 0)).astype(int)
print("\n(c) alt working-free (no '=' and no 'a op b'): share", CL.nowork2.mean(), "; agreement with no-new-number def:", (CL.nowork2 == CL.workfree_v).mean(),
      "; by steps:", CL.assign(nsb=CL.n_steps.clip(upper=6)).groupby("nsb").nowork2.mean().round(2).to_dict(),
      "; P(wrong|nowork2)=%.2f P(wrong|work)=%.2f" % (CL[CL.nowork2 == 1].wrong_v.mean(), CL[CL.nowork2 == 0].wrong_v.mean()))
# (d)
NUMRE = re.compile(r"-?\d+(?:\.\d+)?")
def struct_wrong(resp, gold):
    b = [x for x in re.findall(r"\\boxed\{([^{}]*)\}", resp) if NUMRE.search(x.replace(",", "").replace("$", ""))]
    ns = NUMRE.findall((b[-1] if b else resp).replace(",", "").replace("$", ""))
    return (not ns) or abs(float(ns[-1]) - float(gold)) > 1e-6
B = R[(R.route == "base") | ((R.route == "ftcurve") & (R.dataset_id == "dsse1-08530") & (R.step == 0))].copy()
B["ws"] = [struct_wrong(r, g) for r, g in zip(B.response, B.gold)]
bw = B.groupby("prompt_idx").ws.mean(); bl = B.groupby("prompt_idx").wrong_v.mean()
t = CL[CL.w20 == 1]
print(f"\n(d) base error on the 60 terse-answer problems: structured grader {bw.loc[t.prompt_idx].mean():.3f}, last-number {bl.loc[t.prompt_idx].mean():.3f}; collapse there {t.wrong_v.mean():.3f}")

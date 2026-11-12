"""README section 11: no-signal null for the loop of dn_loop.py (rule B1). Each round, each prompt's steered reading is a fresh
random draw that does not depend on the steering direction; the unsteered reference readings are drawn once and subtracted in
every round, as in dn_loop.py. The statistics are computed as in dn_loop.py: 20 'random' loops on the first 16 prompts and 4
'named' loops on all 32, three rounds, split-half over 20 random halvings. D = 2048 with a power-law covariance (the basis does
not matter for cosines). Printed: mean and 5-95 % range over 50 replicates of the round-1 and round-3 split-half medians, the
medians of cos(d2, d1), cos(d3, d2), cos(d3, d1), the 95th percentile of cos(d3, d1), the cosine of a named d1 with the mean of
the random d1, and the cosine between the d3 of two named loops. No model is loaded. Usage: python dn_loop_null.py"""
import numpy as np
rng = np.random.default_rng(0); D = 2048
lam = 1.0 / np.arange(1, D + 1) ** 1.0           # anisotropic covariance (power-law spectrum), basis irrelevant for cosines
sd = np.sqrt(lam)
cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
def split_half(delta, g):
    sh = []
    for _ in range(20):
        p = g.permutation(len(delta)); a, b = delta[p[:len(delta)//2]].mean(0), delta[p[len(delta)//2:]].mean(0); sh.append(cos(a, b))
    return np.mean(sh)
def run(P, U, rs=1.0):
    dirs, shs = [], []
    for n in range(3):
        S = rng.standard_normal((P, D)) * sd * rs     # steered continuations (fresh each round)
        delta = S - U; dirs.append(delta.mean(0)); shs.append(split_half(delta, np.random.default_rng(7)))
    return dirs, shs
out = []
for rep in range(50):
    U32 = rng.standard_normal((32, D)) * sd           # shared unsteered reference readings
    R = [run(16, U32[:16]) for _ in range(20)]         # 20 'random' loops on the first 16 prompts
    N = [run(32, U32) for _ in range(4)]               # 4 'named' loops on all 32 prompts
    MEANR = [np.mean([R[k][0][n] / np.linalg.norm(R[k][0][n]) for k in range(20)], 0) for n in range(3)]
    out.append(dict(sh1=np.median([r[1][0] for r in R]), sh3=np.median([r[1][2] for r in R]),
                    c21=np.median([cos(r[0][1], r[0][0]) for r in R]), c32=np.median([cos(r[0][2], r[0][1]) for r in R]),
                    c31=np.median([cos(r[0][2], r[0][0]) for r in R]),
                    p95=np.percentile([cos(r[0][2], r[0][0]) for r in R], 95),
                    mr1=np.mean([cos(nl[0][0], MEANR[0]) for nl in N]), cross=cos(N[0][0][2], N[1][0][2])))
for k in out[0]: print(f"{k:6s} mean over 50 replicates {np.mean([o[k] for o in out]):.3f}  (5-95%: {np.percentile([o[k] for o in out],5):.3f} .. {np.percentile([o[k] for o in out],95):.3f})")
